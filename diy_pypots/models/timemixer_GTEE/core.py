"""
Description: The core wrapper assembles the submodules of TimeMixer-based GTEE imputation model.
Implemented with reference to the TimeMixer algorithm in PyPOTS (Wenjie Du <wenjay.du@gmail.com>).

Copyright (c) Yuan Feng et al., 2026
License: MIT License
"""

from typing import Callable
import torch
import torch.nn as nn
import numpy as np

from diy_pypots.nn.functional.normalization import (
    nonstationary_norm,
    nonstationary_denorm,
)
from diy_pypots.nn.modules.timemixer.backbone import BackboneTimeMixer
from diy_pypots.nn.modules.timestamp.backbone import TSbackbone
from diy_pypots.nn.modules.timestamp.loss import TS_loss
from diy_pypots.nn.modules.revin.layers import RevIN

from diy_pypots.utils.metrics.error import calc_mae
from my_utils.model_utils import gaussian_likelihood_affine_trans


class _TimeMixer_GTEE(nn.Module):
    def __init__(
        self,
        TS_freqs_params: np.ndarray,
        TS_used_freqs_num: int,
        TS_ch_embedding_dim: int,
        TS_hidden_dim: int,
        TS_reg_weight: float,
        TS_loss_weight: float,
        TS_dropout: float,
        TS_fusion_conv_kernels: int,
        TS_fusion_hidden_dim: int,
        
        n_steps: int,
        n_features: int,
        n_layers: int,
        d_model: int,
        d_ffn: int,
        dropout: float,
        top_k: int,
        channel_independence: bool,
        decomp_method: str,
        moving_avg: int,
        downsampling_layers: int,
        downsampling_window: int,
        apply_nonstationary_norm: bool,
        customized_loss_func: Callable = calc_mae,
    ):
        super().__init__()
        
        TS_freqs_params_tensor = torch.from_numpy(TS_freqs_params).to(torch.float32)
        self.register_buffer('TS_freqs_params', TS_freqs_params_tensor)
        self.channel_embedding = nn.Parameter(torch.randn(n_features, TS_ch_embedding_dim))
        
        self.apply_nonstationary_norm = apply_nonstationary_norm
        self.customized_loss_func = customized_loss_func
        
        self.TS_loss_weight = TS_loss_weight

        self.local_model = BackboneTimeMixer(
            task_name="imputation",
            n_steps=n_steps,
            n_features=n_features,
            n_pred_steps=0,
            n_pred_features=n_features,
            n_layers=n_layers,
            d_model=d_model,
            d_ffn=d_ffn,
            dropout=dropout,
            channel_independence=channel_independence,
            decomp_method=decomp_method,
            top_k=top_k,
            moving_avg=moving_avg,
            downsampling_layers=downsampling_layers,
            downsampling_window=downsampling_window,
            downsampling_method="avg",
            use_future_temporal_feature=False,
        )
        
        self.TS_model = TSbackbone(freqs_num=TS_used_freqs_num,
                                    feature_num=n_features,
                                    ch_embedding_dim=TS_ch_embedding_dim,
                                    hidden_dim=TS_hidden_dim,
                                    TS_dropout=TS_dropout)
        
        self.TS_loss_calculator = TS_loss(reg_weight=TS_reg_weight)
        
        self.TS_fusion_convs = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=4,
                        out_channels=TS_fusion_hidden_dim, 
                        kernel_size=k, 
                        padding=k//2),
            nn.ReLU())
            for k in TS_fusion_conv_kernels
        ])
        self.fusion_weight_layer = nn.Sequential(
            nn.Conv1d(in_channels=TS_fusion_hidden_dim,
                        out_channels=1,
                        kernel_size=1),
            nn.Sigmoid()
        )
        
        self.revin = RevIN(n_features, affine=False)
    
    def forward(self, inputs: dict, training: bool = True) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]
        if training:
            X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
            full_missing_mask = missing_mask + indicating_mask
            X_n = self.revin(X_ori, full_missing_mask, mode="norm")
        else:
            X_n = self.revin(X, missing_mask, mode="norm")
        
        X_TS = inputs["X_TS"]
        
        if self.apply_nonstationary_norm:
            # Normalization from Non-stationary Transformer
            X, means, stdev = nonstationary_norm(X, missing_mask)
        
        # TimesMixer processing
        local_reconstruction = self.local_model.imputation(X, None)
        
        if self.apply_nonstationary_norm:
            # De-Normalization from Non-stationary Transformer
            local_reconstruction = nonstationary_denorm(local_reconstruction, means, stdev)
        
        (TS_result, dist_mus, dist_logvars) = self.TS_model(X_TS,
                                                            self.TS_freqs_params,
                                                            self.channel_embedding)
        
        dist_stds =  torch.exp(torch.mul(-0.5, dist_logvars))
        if training:
            TS_result = gaussian_likelihood_affine_trans(data_out=TS_result,
                                                            dist_mus=dist_mus,
                                                            dist_stds=dist_stds,
                                                            data_X=X_ori,
                                                            mask=full_missing_mask,
                                                            )
        else:
            TS_result = gaussian_likelihood_affine_trans(data_out=TS_result,
                                                            dist_mus=dist_mus,
                                                            dist_stds=dist_stds,
                                                            data_X=X,
                                                            mask=missing_mask,
                                                            )
        
        missing_pattern_feature = missing_mask.detach().clone().to(dist_mus.dtype)
        fusion_in_feature_list = [TS_result, local_reconstruction, dist_logvars, missing_pattern_feature]
        x_shape = TS_result.shape
        fusion_in_feature = torch.stack(fusion_in_feature_list, dim=-1).permute(0,2,3,1)  # batchsize × feature_num × 4 × time_steps
        fusion_in_feature = fusion_in_feature.flatten(start_dim=0,end_dim=1)
        fusion_conv_results = [conv(fusion_in_feature) for conv in self.TS_fusion_convs]
        fusion_feature = sum(fusion_conv_results)
        fusion_weights = self.fusion_weight_layer(fusion_feature).squeeze(-2)  # (batchsize * feature_num) × time_steps
        fusion_weights = fusion_weights.reshape(x_shape[0],x_shape[-1],fusion_weights.shape[-1]).permute(0,2,1)  # batchsize × time_steps × feature_num
        
        X_final = torch.mul((1-fusion_weights),local_reconstruction) + torch.mul(fusion_weights,TS_result)
        
        imputed_data = missing_mask * X + (1 - missing_mask) * X_final
        
        results = {
            "imputed_data": imputed_data,
            "TS_dist_mus": dist_mus,
            "TS_dist_logvars": dist_logvars
        }
        
        if training:
            # `loss` is always the item for backward propagating to update the model
            main_loss = self.customized_loss_func(X_final, X_ori, indicating_mask)
            assert len(torch.unique(full_missing_mask)<=2), "An error exists in either the missing_mask or the indicating_mask."
            ts_loss = self.TS_loss_calculator(dist_mus,
                                                dist_logvars,
                                                X_n,
                                                full_missing_mask,
                                                model=self.TS_model.coding_model.dist_coding_layer)
            
            loss = main_loss + self.TS_loss_weight*ts_loss
            
            results["main_loss"] = main_loss
            results["TS_loss"] = ts_loss
            results["loss"] = loss
        
        return results
