"""
Preprocessing func for the dataset Beijing Multi-site Air Quality.

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

def preprocess_beijing_air_quality_with_timestamps(
    rate,
    n_steps,
    pattern: str = "point",
    with_time_stamps: bool = False,
    **kwargs,
) -> dict:
    
    assert 0 <= rate < 1, f"rate must be in [0, 1), but got {rate}"
    assert n_steps > 0, f"sample_n_steps must be larger than 0, but got {n_steps}"

    data = tsdb.load("beijing_multisite_air_quality")
    df = data["X"]
    stations = df["station"].unique()

    df_collector = []
    station_name_collector = []

    for station in stations:
        current_df = df[df["station"] == station]
        logger.info(f"Current dataframe shape: {current_df.shape}")

        current_df["date_time"] = pd.to_datetime(
            current_df[["year", "month", "day", "hour"]]
        )
        station_name_collector.append(current_df.loc[0, "station"])
        # remove duplicated date info and wind direction, which is a categorical col
        current_df = current_df.drop(
            ["year", "month", "day", "hour", "wd", "No", "station"], axis=1
        )
        df_collector.append(current_df)

    logger.info(
        f"There are total {len(station_name_collector)} stations, they are {station_name_collector}"
    )
    date_time = df_collector[0]["date_time"]
    df_collector = [i.drop("date_time", axis=1) for i in df_collector]
    df = pd.concat(df_collector, axis=1)
    feature_names = [
        station + "_" + feature
        for station in station_name_collector
        for feature in df_collector[0].columns
    ]
    feature_num = len(feature_names)
    df.columns = feature_names
    logger.info(
        f"Original df missing rate: "
        f"{(df[feature_names].isna().sum().sum() / (df.shape[0] * feature_num)):.3f}"
    )

    df["date_time"] = date_time
    unique_months = df["date_time"].dt.to_period("M").unique()
    
    if with_time_stamps:
        time_stamps=pd.to_numeric(df["date_time"])
    
    selected_as_train = unique_months[:28]  # use the first 28 months as train set
    logger.info(f"months selected as train set are {selected_as_train}")
    selected_as_val = unique_months[28:38]  # select the following 10 months as val set
    logger.info(f"months selected as val set are {selected_as_val}")
    selected_as_test = unique_months[38:]  # select the left 10 months as test set
    logger.info(f"months selected as test set are {selected_as_test}")
    test_set = df[df["date_time"].dt.to_period("M").isin(selected_as_test)]
    val_set = df[df["date_time"].dt.to_period("M").isin(selected_as_val)]
    train_set = df[df["date_time"].dt.to_period("M").isin(selected_as_train)]
    
    if with_time_stamps:
        train_timestamps_set = time_stamps[df["date_time"].dt.to_period("M").isin(selected_as_train)]
        val_timestamps_set = time_stamps[df["date_time"].dt.to_period("M").isin(selected_as_val)]
        test_timestamps_set = time_stamps[df["date_time"].dt.to_period("M").isin(selected_as_test)]

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
