"""
Description: The core wrapper assembles the submodules of GPVAE-based GTEE imputation model.
Implemented with reference to the GPVAE algorithm in PyPOTS (Wenjie Du <wenjay.du@gmail.com>).

Copyright (c) Yuan Feng et al., 2026
License: MIT License
"""

from typing import Callable

import torch
import torch.nn as nn
import numpy as np

from diy_pypots.nn.modules.gpvae.backbone import BackboneGPVAE_GTEE
from diy_pypots.nn.modules.timestamp.backbone import TSbackbone
from diy_pypots.nn.modules.timestamp.loss import TS_loss
from diy_pypots.nn.modules.revin.layers import RevIN

from diy_pypots.utils.metrics.error import calc_mae
from my_utils.model_utils import gaussian_likelihood_affine_trans

class _GPVAE_GTEE(nn.Module):
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
        
        input_dim,
        time_length,
        latent_dim,
        encoder_sizes=(64, 64),
        decoder_sizes=(64, 64),
        beta=1,
        M=1,
        K=1,
        kernel="cauchy",
        sigma=1.0,
        length_scale=7.0,
        kernel_scales=1,
        window_size=24,
        customized_loss_func: Callable = calc_mae,
    ):
        super().__init__()
        n_features = input_dim
        n_steps = time_length
        
        self.customized_loss_func = customized_loss_func
        
        TS_freqs_params_tensor = torch.from_numpy(TS_freqs_params).to(torch.float32)
        self.register_buffer('TS_freqs_params', TS_freqs_params_tensor)
        self.channel_embedding = nn.Parameter(torch.randn(n_features, TS_ch_embedding_dim))
        
        self.TS_loss_weight = TS_loss_weight
        
        self.local_backbone = BackboneGPVAE_GTEE(input_dim,
                                            time_length,
                                            latent_dim,
                                            encoder_sizes,
                                            decoder_sizes,
                                            beta,
                                            M,
                                            K,
                                            kernel,
                                            sigma,
                                            length_scale,
                                            kernel_scales,
                                            window_size,
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
    
    def forward(self, inputs: dict, training: bool = True, n_sampling_times=1) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]
        X_n = self.revin(X, missing_mask, mode="norm")
        X_TS = inputs["X_TS"]
        
        if training:
            elbo_loss, X_local = self.local_backbone(X, missing_mask)
        else:
            X_local = self.local_backbone.impute(X, missing_mask, n_sampling_times)
            
        X_local = X_local.mean(dim=1)  # batch_size, n_steps, n_features
        (TS_result, dist_mus, dist_logvars) = self.TS_model(X_TS,
                                                            self.TS_freqs_params,
                                                            self.channel_embedding)
        
        dist_stds =  torch.exp(torch.mul(-0.5, dist_logvars))
        TS_result = gaussian_likelihood_affine_trans(data_out=TS_result,
                                                        dist_mus=dist_mus,
                                                        dist_stds=dist_stds,
                                                        data_X=X,
                                                        mask=missing_mask,
                                                        )
        
        missing_pattern_feature = missing_mask.detach().clone().to(dist_mus.dtype)
        fusion_in_feature_list = [TS_result, X_local, dist_logvars, missing_pattern_feature]
        x_shape = TS_result.shape
        fusion_in_feature = torch.stack(fusion_in_feature_list, dim=-1).permute(0,2,3,1)  # batchsize × feature_num × 4 × time_steps
        fusion_in_feature = fusion_in_feature.flatten(start_dim=0,end_dim=1)
        fusion_conv_results = [conv(fusion_in_feature) for conv in self.TS_fusion_convs]
        fusion_feature = sum(fusion_conv_results)
        fusion_weights = self.fusion_weight_layer(fusion_feature).squeeze(-2)  # (batchsize * feature_num) × time_steps
        fusion_weights = fusion_weights.reshape(x_shape[0],x_shape[-1],fusion_weights.shape[-1]).permute(0,2,1)  # batchsize × time_steps × feature_num
        
        X_final = torch.mul((1-fusion_weights),X_local) + torch.mul(fusion_weights,TS_result)
        
        imputed_data = missing_mask * X + (1 - missing_mask) * X_final
        imputed_data = imputed_data.unsqueeze(1)  # batch_size × 1 × n_steps × n_features
        results = {
            "imputed_data": imputed_data,
            "TS_dist_mus": dist_mus,
            "TS_dist_logvars": dist_logvars
        }
        
        if training:
            main_loss = self.customized_loss_func(X_final, X, missing_mask)
            assert len(torch.unique(missing_mask)<=2), "An error exists in either the missing_mask or the indicating_mask."
            ts_loss = self.TS_loss_calculator(dist_mus,
                                                dist_logvars,
                                                X_n,
                                                missing_mask,
                                                model=self.TS_model.coding_model.dist_coding_layer)
            loss = main_loss + elbo_loss + self.TS_loss_weight*ts_loss
            results["elbo_loss"] = elbo_loss
            results["TS_loss"] = ts_loss
            results["loss"] = loss
            results["main_loss"] = main_loss
        
        return results
