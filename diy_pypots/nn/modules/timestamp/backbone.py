"""
Description: Implementation of the Timestamp Probabilistic Imputation Module (TPIM).

Copyright (c) Yuan Feng et al., 2026
License: MIT License
"""

from typing import Tuple

import torch
import torch.nn as nn

from diy_pypots.nn.modules.timestamp.layers import TS_dist_calculate_layer

class TSbackbone(nn.Module):
    def __init__(
        self,
        freqs_num: int,
        feature_num: int,
        ch_embedding_dim: int,
        hidden_dim: int,
        TS_dropout: float,
        TS_resampling_num: int=100,
    ):
        super().__init__()
        
        self.freqs_num = freqs_num
        self.feature_num = feature_num
        self.ch_embedding_dim = ch_embedding_dim
        self.hidden_dim = hidden_dim
        self.TS_resampling_num = TS_resampling_num
        self.input_dim = int(2*freqs_num+ch_embedding_dim)
        
        self.coding_model = TS_dist_calculate_layer(input_dim=int(2*freqs_num+ch_embedding_dim),
                                                            hidden_dim=hidden_dim,
                                                            freqs_num=freqs_num,
                                                            TS_dropout=TS_dropout,
                                                            auto_dist_fusion=True,
                                                            )
        
    def forward(self,
                TS_data: torch.Tensor,
                TS_freqs_params: torch.Tensor,
                channel_embedding: torch.Tensor,
                ) -> Tuple[torch.Tensor, ...]:
        freq_embed_data = self.timestamp_coding(TS_data=TS_data,
                                                TS_freqs_params=TS_freqs_params)
        ch_embed_data = self.channel_coding(TS_data=TS_data,
                                            channel_embedding=channel_embedding)
        
        out_dist_mus, out_dist_logvars=self.coding_model(freq_embed_in=freq_embed_data, ch_embed_in=ch_embed_data)
        
        TS_result = self.reparameterize(mus=out_dist_mus,
                                        logvars=out_dist_logvars,
                                        )
        
        return (TS_result, out_dist_mus, out_dist_logvars)
    
    def timestamp_coding(self,
                        TS_data: torch.Tensor,
                        TS_freqs_params: torch.Tensor,
                        ):
        TS_coding_data = TS_data.unsqueeze(-1)
        realtime_freqs_params = TS_freqs_params.unsqueeze(0).unsqueeze(0)
        sin_coding = torch.sin(2*torch.pi*torch.mul(TS_coding_data, realtime_freqs_params))
        cos_coding = torch.cos(2*torch.pi*torch.mul(TS_coding_data, realtime_freqs_params))
        
        coding_result = torch.stack((sin_coding, cos_coding), dim=-1)
        coding_result = coding_result.flatten(start_dim=-2, end_dim=-1)
        
        return coding_result  # batchsize × time_steps × feature_num ×TS_used_freqs_num * 2
    
    def channel_coding(self,
                        TS_data: torch.Tensor,
                        channel_embedding: torch.Tensor
                        ):
        batchsize = TS_data.shape[0]
        time_steps = TS_data.shape[1]
        ch_embed = channel_embedding.unsqueeze(0).unsqueeze(0).repeat(batchsize, time_steps, 1, 1)
        return ch_embed  # batchsize × time_steps × feature_num × TS_ch_embedding_dim
    
    def reparameterize(self,
                        mus: torch.Tensor,
                        logvars: torch.Tensor,
                        ):
        device = mus.device
        stds = torch.exp(-0.5*logvars)
        epsilons = torch.randn(*mus.shape, self.TS_resampling_num, device=device)
        
        samples = torch.add(mus.unsqueeze(-1), torch.mul(stds.unsqueeze(-1),epsilons))
        
        out = torch.median(samples, dim=-1).values
        
        return out # batchsize × time_steps × feature_num