"""
Description: Model utilities, mainly including the implementation of the distribution transformation unit.

Copyright (c) Yuan Feng et al., 2026
License: MIT License
"""

import torch

def gaussian_likelihood_affine_trans(data_out: torch.Tensor,
                                        dist_mus: torch.Tensor,
                                        dist_stds: torch.Tensor,
                                        data_X: torch.Tensor,
                                        mask: torch.Tensor,
                                        eps_down: float=1e-6,
                                        eps_up: float=1.,
                                        ):
    """Function for distribution transformation unit
    
    Args:
        data_out (Tensor): Data before transformation, shape (batch_size × time_steps × feature_num)
        dist_mus (Tensor): Mean estimate, shape (batch_size × time_steps × feature_num)
        dist_stds (Tensor): Standard deviation estimate, shape (batch_size × time_steps × feature_num)
        data_X (Tensor): Observed data, shape (batch_size × time_steps × feature_num)
        mask (Tensor): Missing mask, shape (batch_size × time_steps × feature_num)
        eps_down (float): D_T
        eps_up (float): H_T
    
    Returns:
        ndarray: Transformed data, shape (batch_size × time_steps × feature_num)
    """
    
    batch_size,time_step,n_features = data_X.shape
    squeezed_dist_mus = dist_mus.permute(0,2,1).reshape(-1,time_step).detach()
    squeezed_dist_stds = dist_stds.permute(0,2,1).reshape(-1,time_step).detach()
    squeezed_data_X = data_X.permute(0,2,1).reshape(-1,time_step).detach()
    squeezed_mask = mask.permute(0,2,1).reshape(-1,time_step).detach()
    
    squeezed_alpha = torch.ones(squeezed_data_X.shape[0],1,device=squeezed_data_X.device)
    squeezed_beta = torch.zeros(squeezed_data_X.shape[0],1,device=squeezed_data_X.device)
    
    M = torch.count_nonzero(squeezed_mask, dim=1)
    selected_index = M.nonzero(as_tuple=True)[0]
    
    selected_dist_mus = squeezed_dist_mus[selected_index, :]
    selected_dist_stds = squeezed_dist_stds[selected_index, :]
    selected_data_X = squeezed_data_X[selected_index, :]
    selected_mask = squeezed_mask[selected_index, :]
    
    k_beta = - torch.div(torch.sum(torch.mul(selected_mask, torch.div(selected_dist_mus, torch.pow(selected_dist_stds,2))),dim=-1),
                            torch.sum(torch.mul(selected_mask, torch.div(1, torch.pow(selected_dist_stds,2))),dim=-1))
    
    s_beta = torch.div(torch.sum(torch.mul(selected_mask, torch.div(selected_data_X, torch.pow(selected_dist_stds,2))),dim=-1),
                            torch.sum(torch.mul(selected_mask, torch.div(1, torch.pow(selected_dist_stds,2))),dim=-1))
    
    a_alpha = - torch.count_nonzero(selected_mask, dim=1).float()
    
    b_alpha = - torch.sum(torch.mul(selected_mask,torch.div(torch.mul(torch.add(selected_dist_mus,k_beta.unsqueeze(-1)),torch.sub(selected_data_X,s_beta.unsqueeze(-1))),
                                    torch.pow(selected_dist_stds,2))),dim=-1)
    c_alpha = torch.sum(torch.mul(selected_mask,torch.div(torch.pow(torch.sub(selected_data_X, s_beta.unsqueeze(-1)),2),
                                                            torch.pow(selected_dist_stds,2))), dim=-1)
    selected_alpha = max_real_root_quadratic_general(a=a_alpha,
                                                        b=b_alpha,
                                                        c=c_alpha)
    selected_beta = torch.add(torch.mul(k_beta,selected_alpha),s_beta)
    squeezed_alpha[selected_index,:] = selected_alpha.unsqueeze(-1)
    squeezed_beta[selected_index,:] = selected_beta.unsqueeze(-1)
    
    squeezed_alpha = torch.where(torch.isnan(squeezed_alpha) | torch.isinf(squeezed_alpha) | (torch.abs(squeezed_alpha) <= eps_down) | (torch.abs(squeezed_alpha) >= eps_up), torch.tensor(1.0, device=squeezed_alpha.device), squeezed_alpha)
    squeezed_beta = torch.where(torch.isnan(squeezed_beta) | torch.isinf(squeezed_beta) | (torch.abs(squeezed_alpha) <= eps_down) | (torch.abs(squeezed_alpha) >= eps_up), torch.tensor(0.0, device=squeezed_beta.device), squeezed_beta)
    
    squeezed_data_out = data_out.permute(0,2,1).reshape(-1,time_step)
    squeezed_data_trans = torch.add(torch.mul(squeezed_alpha,squeezed_data_out),squeezed_beta)
    data_trans = squeezed_data_trans.reshape(batch_size,n_features,time_step).permute(0,2,1)
    return data_trans

def max_real_root_quadratic_general(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    """
    Batch solve for the maximum real root of max(real root) of ax^2 + bx + c = 0
    Handle the case a=0 (degenerate to linear equation), return NaN if no real roots exist.
    """
    result = torch.full_like(a, float('nan'))
    
    mask_quad = a != 0
    a_q, b_q, c_q = a[mask_quad], b[mask_quad], c[mask_quad]
    D = b_q ** 2 - 4 * a_q * c_q
    real_mask = D >= 0
    sqrt_D = torch.sqrt(D[real_mask])
    denom = 2 * a_q[real_mask]
    x1 = (-b_q[real_mask] + sqrt_D) / denom
    x2 = (-b_q[real_mask] - sqrt_D) / denom
    max_root_quad = torch.maximum(x1, x2)
    result[mask_quad.nonzero().squeeze(-1)[real_mask]] = max_root_quad
    
    mask_linear = (a == 0) & (b != 0)
    result[mask_linear] = -c[mask_linear] / b[mask_linear]
    
    return result
