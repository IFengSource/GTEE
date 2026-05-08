"""
Description: The core wrapper assembles the submodules of Pure_GTEE imputation model (timestamp-dominated imputation model)
Refer to the PyPOTS template.

Copyright (c) Yuan Feng et al., 2026
License: MIT License
"""

from typing import Callable

import torch
import torch.nn as nn
import numpy as np

from diy_pypots.nn.modules.timestamp.backbone import TSbackbone
from diy_pypots.nn.modules.timestamp.loss import TS_loss
from diy_pypots.nn.modules.revin.layers import RevIN

from diy_pypots.utils.metrics.error import calc_mae
from my_utils.model_utils import gaussian_likelihood_affine_trans

class _Pure_GTEE(nn.Module):
    def __init__(
        self,
        TS_freqs_params: np.ndarray,
        TS_used_freqs_num: int,
        TS_ch_embedding_dim: int,
        TS_hidden_dim: int,
        TS_reg_weight: float,
        TS_loss_weight: float,
        TS_dropout: float,
        
        n_steps: int,
        n_features: int,
        customized_loss_func: Callable = calc_mae,
    ):
        super().__init__()
        
        TS_freqs_params_tensor = torch.from_numpy(TS_freqs_params).to(torch.float32)
        self.register_buffer('TS_freqs_params', TS_freqs_params_tensor)
        
        self.channel_embedding = nn.Parameter(torch.randn(n_features, TS_ch_embedding_dim))
        
        self.n_steps = n_steps
        self.n_features = n_features
        self.customized_loss_func = customized_loss_func
        
        self.TS_loss_weight = TS_loss_weight
        self.TS_used_freqs_num= TS_used_freqs_num
        
        self.TS_model = TSbackbone(freqs_num=TS_used_freqs_num,
                                    feature_num=n_features,
                                    ch_embedding_dim=TS_ch_embedding_dim,
                                    hidden_dim=TS_hidden_dim,
                                    TS_dropout=TS_dropout)
        
        self.TS_loss_calculator = TS_loss(reg_weight=TS_reg_weight)
        self.revin = RevIN(n_features, affine=False)
    
    def forward(
        self,
        inputs: dict,
        training: bool = True,
    ) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]
        X_n = self.revin(X, missing_mask, mode="norm")
        X_TS = inputs["X_TS"]
        
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
        X_tilde_final = TS_result
        
        imputed_data = missing_mask * X + (1 - missing_mask) * X_tilde_final
        
        results = {
            "imputed_data": imputed_data,
            "TS_dist_mus": dist_mus,
            "TS_dist_logvars": dist_logvars
        }  # batchsize × time_steps × feature_num
        
        if training:
            main_loss = self.customized_loss_func(X_tilde_final, X, missing_mask)
            TS_loss = self.TS_loss_calculator(dist_mus,
                                                dist_logvars,
                                                X_n,
                                                missing_mask,
                                                model=self.TS_model.coding_model.dist_coding_layer)
            loss = main_loss + self.TS_loss_weight*TS_loss

            results["main_loss"] = main_loss
            results["TS_loss"] = TS_loss
            results["loss"] = loss
        
        return results
