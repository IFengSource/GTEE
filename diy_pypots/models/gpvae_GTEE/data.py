"""
Description: Dataset class for the GPVAE-based GTEE imputation model.
Implemented with reference to the GPVAE algorithm in PyPOTS (Wenjie Du <wenjay.du@gmail.com>).

Copyright (c) Yuan Feng et al., 2026
License: MIT License
"""

from typing import Union, Iterable

import torch
from pygrinder import mcar, fill_and_get_mask_torch

from diy_pypots.data.dataset.base import BaseDataset_with_TS

class DatasetForGPVAE_GTEE(BaseDataset_with_TS):
    def __init__(
        self,
        data: Union[dict, str],
        return_X_ori: bool,
        return_y: bool,
        return_TS: bool,
        file_type: str = "hdf5",
    ):
        super().__init__(
            data=data,
            return_X_ori=return_X_ori,
            return_X_pred=False,
            return_y=return_y,
            return_TS=return_TS,
            file_type=file_type,
        )
        
    def _fetch_data_from_array(self, idx: int) -> Iterable:
        X = self.X[idx]
        
        if self.return_X_ori:
            X = self.X[idx]
            missing_mask = self.missing_mask[idx]
            X_ori = self.X_ori[idx]
            indicating_mask = self.indicating_mask[idx]
            sample = [torch.tensor(idx), X, missing_mask, X_ori, indicating_mask]
        else:
            X, missing_mask = fill_and_get_mask_torch(X)
            sample = [torch.tensor(idx), X, missing_mask]
        
        if self.return_y:
            sample.append(self.y[idx].to(torch.long))
        
        if self.return_TS:
            X_TS = self.X_TS[idx]
            sample.append(X_TS)
        
        return sample
    
    def _fetch_data_from_file(self, idx: int) -> Iterable:
        if self.file_handle is None:
            self.file_handle = self._open_file_handle()
        
        if self.return_X_ori:
            X = torch.from_numpy(self.file_handle["X"][idx]).to(torch.float32)
            X_ori = torch.from_numpy(self.file_handle["X_ori"][idx]).to(torch.float32)
            X_ori, X_ori_missing_mask = fill_and_get_mask_torch(X_ori)
            X, missing_mask = fill_and_get_mask_torch(X)
            indicating_mask = (X_ori_missing_mask - missing_mask).to(torch.float32)
            sample = [torch.tensor(idx), X, missing_mask, X_ori, indicating_mask]
        else:
            X = torch.from_numpy(self.file_handle["X"][idx]).to(torch.float32)
            X, missing_mask = fill_and_get_mask_torch(X)
            sample = [torch.tensor(idx), X, missing_mask]
        
        # if the dataset has labels and is for training, then fetch it from the file
        if self.return_y:
            sample.append(torch.tensor(self.file_handle["y"][idx], dtype=torch.long))
            
        if self.return_TS:
            X_TS = torch.from_numpy(self.file_handle["X_TS"][idx]).to(torch.float32)
            sample.append(X_TS)
        
        return sample
