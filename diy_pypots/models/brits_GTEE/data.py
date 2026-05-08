"""
Description: Dataset class for the BRITS-based GTEE imputation model.
Implemented with reference to the BRITS algorithm in PyPOTS (Wenjie Du <wenjay.du@gmail.com>).

Copyright (c) Yuan Feng et al., 2026
License: MIT License
"""

from typing import Union, Iterable

import torch
from pygrinder import fill_and_get_mask_torch

from diy_pypots.data.dataset.base import BaseDataset_with_TS
from diy_pypots.data.utils import _parse_delta_torch

class DatasetForBRITS_GTEE(BaseDataset_with_TS):
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
        
        if not isinstance(self.data, str):
            # calculate all delta here.
            if self.return_X_ori:
                forward_missing_mask = self.missing_mask
                forward_X = self.X
                
            else:
                forward_X, forward_missing_mask = fill_and_get_mask_torch(self.X)

            forward_delta = _parse_delta_torch(forward_missing_mask)
            backward_X = torch.flip(forward_X, dims=[1])
            backward_missing_mask = torch.flip(forward_missing_mask, dims=[1])
            backward_delta = _parse_delta_torch(backward_missing_mask)

            self.processed_data = {
                "forward": {
                    "X": forward_X.to(torch.float32),
                    "missing_mask": forward_missing_mask.to(torch.float32),
                    "delta": forward_delta.to(torch.float32),
                },
                "backward": {
                    "X": backward_X.to(torch.float32),
                    "missing_mask": backward_missing_mask.to(torch.float32),
                    "delta": backward_delta.to(torch.float32),
                },
            }

    def _fetch_data_from_array(self, idx: int) -> Iterable:
        sample = [
            torch.tensor(idx),
            # for forward
            self.processed_data["forward"]["X"][idx],
            self.processed_data["forward"]["missing_mask"][idx],
            self.processed_data["forward"]["delta"][idx],
            # for backward
            self.processed_data["backward"]["X"][idx],
            self.processed_data["backward"]["missing_mask"][idx],
            self.processed_data["backward"]["delta"][idx],
        ]
        

        if self.return_X_ori:
            sample.extend([self.X_ori[idx], self.indicating_mask[idx]])

        if self.return_y:
            sample.append(self.y[idx].to(torch.long))

        if self.return_TS:
            X_TS = self.X_TS[idx]
            sample.append(X_TS)

        return sample
    
    def _fetch_data_from_file(self, idx: int) -> Iterable:
        if self.file_handle is None:
            self.file_handle = self._open_file_handle()

        X = torch.from_numpy(self.file_handle["X"][idx]).to(torch.float32)
        X, missing_mask = fill_and_get_mask_torch(X)

        forward = {
            "X": X,
            "missing_mask": missing_mask,
            "deltas": _parse_delta_torch(missing_mask),
        }

        backward = {
            "X": torch.flip(forward["X"], dims=[0]),
            "missing_mask": torch.flip(forward["missing_mask"], dims=[0]),
        }
        backward["deltas"] = _parse_delta_torch(backward["missing_mask"])

        sample = [
            torch.tensor(idx),
            # for forward
            forward["X"],
            forward["missing_mask"],
            forward["deltas"],
            # for backward
            backward["X"],
            backward["missing_mask"],
            backward["deltas"],
        ]

        if self.return_X_ori:
            X_ori = torch.from_numpy(self.file_handle["X_ori"][idx]).to(torch.float32)
            X_ori, X_ori_missing_mask = fill_and_get_mask_torch(X_ori)
            indicating_mask = X_ori_missing_mask - missing_mask
            sample.extend([X_ori, indicating_mask])

        if self.return_y:
            sample.append(torch.tensor(self.file_handle["y"][idx], dtype=torch.long))
        
        if self.return_TS:
            X_TS = torch.from_numpy(self.file_handle["X_TS"][idx]).to(torch.float32)
            sample.append(X_TS)

        return sample
