"""
Description: A family of functions related to the preprocessing of model data.

Copyright (c) Yuan Feng et al., 2026
License: MIT License
"""
import os
import h5py
import numpy as np
from typing import Union
import torch
import traceback

from diy_pypots.data.saving.h5 import save_dict_into_h5, load_dict_from_h5
from configs.global_config import LAZY_LOAD_DATA
from diy_pypots.utils.logging import logger
from data.preprocess.pipeline import get_timestamp_coding_parameters

def get_datasets_path_with_TS(data_dir, with_time_stamps: bool=True):
    train_set_path = os.path.join(data_dir, "train.h5")
    val_set_path = os.path.join(data_dir, "val.h5")
    test_set_path = os.path.join(data_dir, "test.h5")
    if LAZY_LOAD_DATA:
        prepared_train_set = train_set_path
        prepared_val_set = val_set_path
    else:
        with h5py.File(train_set_path, "r") as hf:
            train_X_arr = hf["X"][:]
            if with_time_stamps:
                train_TS_arr = hf["X_TS"][:]
                
        with h5py.File(val_set_path, "r") as hf:
            val_X_arr = hf["X"][:]
            val_X_ori_arr = hf["X_ori"][:]
            if with_time_stamps:
                val_TS_arr = hf["X_TS"][:]
                
        prepared_train_set = {"X": train_X_arr}
        prepared_val_set = {"X": val_X_arr, "X_ori": val_X_ori_arr}
        if with_time_stamps:
            prepared_train_set["X_TS"]=train_TS_arr
            prepared_val_set["X_TS"]=val_TS_arr
        
    with h5py.File(test_set_path, "r") as hf:
        test_X_arr = hf["X"][:]
        test_X_ori_arr = hf["X_ori"][:]
        if with_time_stamps:
            test_TS_arr = hf["X_TS"][:]
    test_indicating_arr = ~np.isnan(test_X_ori_arr) ^ ~np.isnan(test_X_arr)
    test_X_ori_arr = np.nan_to_num(test_X_ori_arr)
    prepared_test_set = {"X": test_X_arr,
                            "X_ori": test_X_ori_arr,
                            "Indicating_mask": test_indicating_arr}
    if with_time_stamps:
        prepared_test_set["X_TS"]=test_TS_arr
        
    return (prepared_train_set, prepared_val_set, prepared_test_set)

def get_TS_prior_data(data_dir):
    TS_prior_path = os.path.join(data_dir, "TS_prior.h5")
    with h5py.File(TS_prior_path, "r") as hf:
        TS_dataset = hf["TS_dataset"][:]
        TS_start_num = hf["TS_start_num"][:]
        TS_gap = hf["TS_gap"][:]
    prepared_prior_dict = {
        "TS_dataset": TS_dataset,
        "TS_start_num": TS_start_num,
        "TS_gap": TS_gap,
    }
    return prepared_prior_dict

def calculate_main_freqs(config: dict,
                            read_from_extra: bool,
                            auto_saving: bool=True):
    """Function for solving main frequencies

    Args:
        config (dict): Parameter dictionary.
        read_from_extra (bool): Whether to load from external files
        auto_saving (bool, optional): Whether to automatically save/overwrite the results of main frequency calculation
    
    Returns:
        ndarray: Main frequency information as a numpy array
    """
    base_required_keys = ["saving_path","n_features"]
    for key in base_required_keys:
        if key not in config:
            raise ValueError(f"Missing required key: '{key}'")
    full_path = os.path.join(config["saving_path"], "main_freqs.h5")
    
    if read_from_extra:
        if not os.path.isfile(full_path):
            raise ValueError(f"The main freqs file path {full_path} is not right")
        try:
            data_dict = load_dict_from_h5(file_path=full_path)
        except Exception as e:
            print("External data loading failed! Details are as follows:")
            traceback.print_exc()
        main_freqs = data_dict["main_freqs"]
        if main_freqs.shape[0] != config["n_features"]:
            raise ValueError(f"the shape of main freqs is {main_freqs.shape}, which is not equal to n_features.")
        return main_freqs
    else:
        extend_required_keys = ["ts_main_freq_num",
                                "ts_main_freq_solvers",
                                "ts_solvers_combine",
                                "ts_sampling_num",
                                "ts_sampling_length",
                                "ts_scaling_factor",
                                "ts_freqs_merge_value",
                                "ts_refining_mode",
                                "TS_dataset"]
        for key in extend_required_keys:
            if key not in config:
                raise ValueError(f"Missing required key: '{key}'")
        
        ts_sampling_num = config["ts_sampling_num"]
        ts_sampling_length = config["ts_sampling_length"]
        ts_main_freq_num = config["ts_main_freq_num"]
        
        assert config["ts_sampling_length"] <= config["TS_dataset"].shape[0], \
            f"the ts_sampling_length is larger than the total steps."
        
        while ts_sampling_num + ts_sampling_length-1 > config["TS_dataset"].shape[0]:
            ts_sampling_num -= 1
            logger.warning(
                f"⚠️ reduce the ts_sampling_num to {ts_sampling_num}, since the total sample length is out of range."
            )
            
        if (ts_sampling_length//2-1) < ts_main_freq_num:
            ts_main_freq_num = ts_sampling_length//2-1
            logger.warning(
                f"⚠️ ts_main_freqs_num is bigger than half ts_sampling_length {ts_sampling_length//2-1}, it is reset to {ts_main_freq_num}."
            )
        
        main_freqs = get_timestamp_coding_parameters(dataset=config["TS_dataset"],
                                                        main_freq_num=ts_main_freq_num,
                                                        main_freq_solvers=config["ts_main_freq_solvers"],
                                                        solvers_combine=config["ts_solvers_combine"],
                                                        sampling_num=ts_sampling_num,
                                                        sampling_length=ts_sampling_length,
                                                        scaling_factor=config["ts_scaling_factor"],
                                                        freqs_merge_value=config["ts_freqs_merge_value"],
                                                        refining_mode=config["ts_refining_mode"])
        data_dict = {"main_freqs": main_freqs}
        
        if auto_saving:
            save_dict_into_h5(data_dict, full_path)
        
        return main_freqs

def organize_and_save_with_timestamps(data_dict, saving_dir, with_time_stamps: bool=True):
    train = {
        "X": data_dict["train_X"],
        "X_ori": data_dict["train_X_ori"] if "train_X_ori" in data_dict.keys() else "",
        "y": data_dict["train_y"] if "train_y" in data_dict.keys() else "",
    }
    val = {
        "X": data_dict["val_X"],
        "X_ori": data_dict["val_X_ori"],
        "y": data_dict["val_y"] if "val_y" in data_dict.keys() else "",
    }
    test = {
        "X": data_dict["test_X"],
        "X_ori": data_dict["test_X_ori"],
        "y": data_dict["test_y"] if "test_y" in data_dict.keys() else "",
    }
    
    if with_time_stamps:
        train["X_TS"] = data_dict["train_ts"]
        val["X_TS"] = data_dict["val_ts"]
        test["X_TS"] = data_dict["test_ts"]
        
        TS_prior = {
            "TS_dataset": data_dict["TS_dataset"],
            "TS_start_num": data_dict["TS_start_num"],
            "TS_gap": data_dict["TS_gap"],
        }
        
    save_dict_into_h5(train, saving_dir, "train.h5")
    save_dict_into_h5(val, saving_dir, "val.h5")
    save_dict_into_h5(test, saving_dir, "test.h5")
    
    if with_time_stamps:
        save_dict_into_h5(TS_prior, saving_dir, "TS_prior.h5")
    
    return

class TimeStamps_Scaler():
    def __init__(self, copy: bool = True):
        self.copy = copy
        self.gap = None
        self.start_num = None
        self.datatype = None
        
    def fit_transform(self, x_in: Union[np.ndarray, torch.Tensor]):
        
        assert len(x_in.shape)==2, f"the dim of x_in must be two, but got {len(x_in.shape)}"
        self.datatype = np.ndarray if isinstance(x_in, np.ndarray) else torch.Tensor
        
        if isinstance(x_in, np.ndarray):
            z = x_in.copy() if self.copy else x_in
            total_steps, _ = z.shape
            total_gap = np.abs(z[0]-z[-1])
        
        elif isinstance(x_in, torch.Tensor):
            z = x_in.clone() if self.copy else x_in
            total_steps, _ = z.shape
            total_gap = torch.abs(z[0]-z[-1])
        
        else:
            raise TypeError("{} is an unsupported dataType".format(type(x_in)))
        
        gap = total_gap // (total_steps-1)
        start_num = z[0]
        
        z = (z- start_num) // gap
        
        self.gap = gap
        self.start_num = start_num
        
        return z, gap, start_num
    
    def transform(self, x_in: Union[np.ndarray, torch.Tensor]):
        assert isinstance(x_in, self.datatype), f"the dtype of x_in must assert with fit data,which is {self.datatype}, but got {type(x_in)}"
        if isinstance(x_in, np.ndarray):
            z = x_in.copy() if self.copy else x_in
        else:
            z = x_in.clone() if self.copy else x_in
        z = (z- self.start_num) // self.gap
        return z
    
    def inverse_transform(self, x_in: Union[np.ndarray, torch.Tensor]):
        assert isinstance(x_in, self.datatype), f"the dtype of x_in must assert with fit data,which is {self.datatype}, but got {type(x_in)}"
        z = x_in.copy() if self.copy else x_in
        if isinstance(x_in, np.ndarray):
            z = x_in.copy() if self.copy else x_in
        else:
            z = x_in.clone() if self.copy else x_in
        z = z * self.gap + self.start_num
        return z

