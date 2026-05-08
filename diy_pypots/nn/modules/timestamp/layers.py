"""
Description: Implementation of the Distribution Parameter Encoder/Decoder in the Timestamp Probabilistic Imputation Module (TPIM).

Copyright (c) Yuan Feng et al., 2026
License: MIT License
"""

import torch
import torch.nn as nn

class TS_dist_calculate_layer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        freqs_num: int,
        TS_dropout: float,
        auto_dist_fusion: bool=True,
        
    ):
        super().__init__()
        
        self.auto_dist_fusion = auto_dist_fusion
        self.middle_dim=int(2*freqs_num)
        self.dist_coding_layer = nn.Sequential(nn.Linear(in_features=input_dim, out_features=hidden_dim),
                                                nn.GELU(),
                                                nn.Linear(in_features=hidden_dim, out_features=self.middle_dim))
        if not auto_dist_fusion:
            self.logvar_weights_layer = nn.Softmax(dim=3)
        else:
            self.mu_fusion_layer = nn.Sequential(nn.GELU(),
                                                    nn.Linear(in_features=freqs_num, out_features=1),
                                                    )
            self.logvar_fusion_layer = nn.Sequential(nn.GELU(),
                                                        nn.Linear(in_features=freqs_num, out_features=1),
                                                    )
        self.dropout_layer = nn.Dropout(TS_dropout)
    
    def forward(self, 
                freq_embed_in: torch.Tensor,
                ch_embed_in: torch.Tensor,
                ):
        
        batch_size=freq_embed_in.shape[0]
        time_step_num=freq_embed_in.shape[1]
        feature_num=freq_embed_in.shape[2]
        
        z = torch.cat([freq_embed_in, ch_embed_in], dim=-1)
        z_dist_hidden = self.dist_coding_layer(z)
        z_dist_hidden = self.dropout_layer(z_dist_hidden)
        if self.auto_dist_fusion:
            mu, logvar=self.gather_dists_auto(z_dist_hidden)
        else:
            mu, logvar=self.gather_dists_manual(z_dist_hidden)
        
        return mu, logvar  # (batch_size × time_step_num × feature_num), (batch_size × time_step_num × feature_num)
        
    def gather_dists_manual(self,
                        x_in: torch.Tensor,
                        ):
        z_dists = x_in.reshape(x_in.shape[0], x_in.shape[1],x_in.shape[2],-1, 2)
        sub_mu = z_dists[:,:,:,:,0]
        sub_logvar = z_dists[:,:,:,:,1]
        logvar_weights = self.logvar_weights_layer(sub_logvar)
        sub_logvar = torch.mul(logvar_weights, sub_logvar)
        out_mu = sub_mu.sum(dim=3)
        out_logvar = sub_logvar.sum(dim=3)
        return out_mu, out_logvar  # (batch_size × time_step_num × feature_num), (batch_size × time_step_num × feature_num)
    
    def gather_dists_auto(self,
                            x_in: torch.Tensor,
                            ):
        z_dists = x_in.reshape(x_in.shape[0], x_in.shape[1],x_in.shape[2],-1, 2)
        sub_mu = z_dists[:,:,:,:,0]
        sub_logvar = z_dists[:,:,:,:,1]
        out_mu = self.mu_fusion_layer(sub_mu)
        out_mu = out_mu.squeeze(-1)
        out_logvar = self.logvar_fusion_layer(sub_logvar)
        out_logvar = out_logvar.squeeze(-1)
        
        return out_mu, out_logvar  # (batch_size × time_step_num × feature_num), (batch_size × time_step_num × feature_num)
