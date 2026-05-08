"""
Description: Dataset class for the ModernTCN-based GTEE imputation model.
Implemented with reference to the ModernTCN algorithm in PyPOTS (Wenjie Du <wenjay.du@gmail.com>).

Copyright (c) Yuan Feng et al., 2026
License: MIT License
"""

from typing import Union

from diy_pypots.models.saits_GTEE.data import DatasetForSAITS_GTEE


class DatasetForModernTCN_GTEE(DatasetForSAITS_GTEE):
    def __init__(
        self,
        data: Union[dict, str],
        return_X_ori: bool,
        return_y: bool,
        return_TS: bool,
        file_type: str = "hdf5",
        rate: float = 0.2,
    ):
        super().__init__(data, return_X_ori, return_y, return_TS, file_type, rate)
