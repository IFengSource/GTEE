"""
Description: Calculation of TS Loss Function.

Copyright (c) Yuan Feng et al., 2026
License: MIT License
"""

import torch
import torch.nn as nn

class TS_loss(nn.Module):
    def __init__(self,
                    reg_weight: float,
                    reg_type: str='L1',
                    ):
        super().__init__()
        self.reg_weight = reg_weight
        self.reg_type = reg_type
    
    def forward(self,
                dist_mus: torch.Tensor,
                dist_logvars: torch.Tensor,
                X_ori: torch.Tensor,
                full_missing_mask: torch.Tensor,
                model: torch.nn.Module
                ):
        ANLL_loss = self.calc_ANLL_loss(dist_mus=dist_mus,
                                    dist_logvars=dist_logvars,
                                    X_ori=X_ori,
                                    full_missing_mask=full_missing_mask)
        
        reg_loss = self.calc_freqs_encoding_reg_loss(model=model)
        
        out_loss = ANLL_loss + self.reg_weight * reg_loss
        return out_loss
    
    def calc_ANLL_loss(self,
                        dist_mus: torch.Tensor,
                        dist_logvars: torch.Tensor,
                        X_ori: torch.Tensor,
                        full_missing_mask: torch.Tensor,
                        ):
        dist_var = torch.exp(torch.mul(-1, dist_logvars))
        z_nll = torch.div(torch.square(torch.sub(X_ori, dist_mus)), torch.mul(2,dist_var)) + torch.log(torch.sqrt(torch.mul(2, torch.mul(torch.pi, dist_var))))
        out_loss = torch.sum(torch.mul(z_nll, full_missing_mask)) / torch.sum(full_missing_mask)
        return out_loss
    
    def calc_freqs_encoding_reg_loss(self,
                                        model: torch.nn.Module):
        reg_norm = 0
        for param in model.parameters():
            if self.reg_type == 'L1':
                reg_norm += torch.sum(torch.abs(param))
            elif self.reg_type == 'L2':
                reg_norm += torch.sum(torch.square(param))
            else:
                raise ValueError("undefined regularize loss type")
        return reg_norm
