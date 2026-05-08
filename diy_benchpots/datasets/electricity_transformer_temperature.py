"""
Preprocessing func for the dataset ETT (Electricity Transformer Temperature).

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

# Modifications made by Yuan Feng et al., 2026:
# Using absolute import paths; Support processing and saving of timestamp data.

import pandas as pd
import tsdb
import numpy as np
from sklearn.preprocessing import StandardScaler

from diy_benchpots.utils.logging import logger, print_final_dataset_info
from diy_benchpots.utils.missingness import create_missingness
from diy_benchpots.utils.sliding import sliding_window
from my_utils.data_utils import TimeStamps_Scaler

def preprocess_ett_with_timestamps(
    subset,
    rate,
    n_steps,
    pattern: str = "point",
    with_time_stamps: bool = False,
    **kwargs,
) -> dict:
    
    all_subset_names = ["ETTm1", "ETTm2", "ETTh1", "ETTh2"]
    assert (
        subset in all_subset_names
    ), f"subset_name should be one of {all_subset_names}, but got {subset}"
    assert 0 <= rate < 1, f"rate must be in [0, 1), but got {rate}"
    assert n_steps > 0, f"sample_n_steps must be larger than 0, but got {n_steps}"
    
    data = tsdb.load("electricity_transformer_temperature")  # load all 4 sub datasets
    df = data[subset]
    feature_names = df.columns.tolist()
    df["datetime"] = pd.to_datetime(df.index)
    
    unique_months = df["datetime"].dt.to_period("M").unique()
    
    if with_time_stamps:
        time_stamps=pd.to_numeric(df["datetime"])
    
    selected_as_train = unique_months[:14]  # use the first 14 months as train set
    logger.info(f"months selected as train set are {selected_as_train}")
    selected_as_val = unique_months[14:19]  # select the following 5 months as val set
    logger.info(f"months selected as val set are {selected_as_val}")
    selected_as_test = unique_months[19:]  # select the left 5 months as test set
    logger.info(f"months selected as test set are {selected_as_test}")
    
    train_set = df[df["datetime"].dt.to_period("M").isin(selected_as_train)]
    val_set = df[df["datetime"].dt.to_period("M").isin(selected_as_val)]
    test_set = df[df["datetime"].dt.to_period("M").isin(selected_as_test)]
    
    if with_time_stamps:
        train_timestamps_set = time_stamps[df["datetime"].dt.to_period("M").isin(selected_as_train)]
        val_timestamps_set = time_stamps[df["datetime"].dt.to_period("M").isin(selected_as_val)]
        test_timestamps_set = time_stamps[df["datetime"].dt.to_period("M").isin(selected_as_test)]
    
    scaler = StandardScaler()
    train_set_X = scaler.fit_transform(train_set.loc[:, feature_names])
    val_set_X = scaler.transform(val_set.loc[:, feature_names])
    test_set_X = scaler.transform(test_set.loc[:, feature_names])
    
    
    if with_time_stamps:
        train_ts_rough = np.expand_dims(train_timestamps_set.values,1)
        val_ts_rough = np.expand_dims(val_timestamps_set.values,1)
        test_ts_rough = np.expand_dims(test_timestamps_set.values,1)
        
        TS_scaler = TimeStamps_Scaler()
        
        train_set_ts, TS_gap, TS_start_num = TS_scaler.fit_transform(train_ts_rough)
        val_set_ts = TS_scaler.transform(val_ts_rough)
        test_set_ts = TS_scaler.transform(test_ts_rough)
        
    
    train_X = sliding_window(train_set_X, n_steps)
    val_X = sliding_window(val_set_X, n_steps)
    test_X = sliding_window(test_set_X, n_steps)
    
    if with_time_stamps:
        train_ts = sliding_window(train_set_ts, n_steps)
        val_ts = sliding_window(val_set_ts, n_steps)
        test_ts = sliding_window(test_set_ts, n_steps)
    
    # assemble the final processed data into a dictionary
    processed_dataset = {
        # general info
        "n_steps": n_steps,
        "n_features": train_X.shape[-1],
        "scaler": scaler,
        # train set
        "train_X": train_X,
        # val set
        "val_X": val_X,
        # test set
        "test_X": test_X,
    }
    
    if with_time_stamps:
        processed_dataset["train_ts"] = train_ts
        processed_dataset["val_ts"] = val_ts
        processed_dataset["test_ts"] = test_ts
        processed_dataset["TS_gap"] = TS_gap
        processed_dataset["TS_start_num"] = TS_start_num
        processed_dataset["TS_scaler"] = TS_scaler
        
        processed_dataset["TS_dataset"] = val_set_X
        
        logger.info("Attention: calculated and transformed time_stamps.")
    else:
        logger.info("Attention: have not calculate and transform time_stamps.")
    
    
    if rate > 0:
        # hold out ground truth in the original data for evaluation
        train_X_ori = train_X
        val_X_ori = val_X
        test_X_ori = test_X

        # mask values in the train set to keep the same with below validation and test sets
        train_X = create_missingness(train_X, rate, pattern, **kwargs)
        # mask values in the validation set as ground truth
        val_X = create_missingness(val_X, rate, pattern, **kwargs)
        # mask values in the test set as ground truth
        test_X = create_missingness(test_X, rate, pattern, **kwargs)

        processed_dataset["train_X"] = train_X
        processed_dataset["train_X_ori"] = train_X_ori

        processed_dataset["val_X"] = val_X
        processed_dataset["val_X_ori"] = val_X_ori

        processed_dataset["test_X"] = test_X
        # test_X_ori is for error calc, not for model input, hence mustn't have NaNs
        processed_dataset["test_X_ori"] = test_X_ori
    else:
        logger.warning("rate is 0, no missing values are artificially added.")

    print_final_dataset_info(train_X, val_X, test_X)
    return processed_dataset
