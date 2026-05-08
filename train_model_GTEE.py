"""
-----------------------------------------------------------------------------
This file is adapted from Awesome_Imputation (https://github.com/WenjieDu/Awesome_Imputation)
Related project: “PyPOTS: a Python toolbox for machine learning on Partially-Observed Time Series”
Original license: BSD-3-Clause
Copyright (c) 2023-present, Wenjie Du <wenjay.du@gmail.com>

Modifications made by Yuan Feng et al., 2026:
GTEE model adaptation and and process optimization.
-----------------------------------------------------------------------------
"""
import argparse
import os
import time

import numpy as np
import torch

from diy_pypots.data.saving.pickle import pickle_dump

from diy_pypots.models.pure_GTEE.model import Pure_GTEE
from diy_pypots.models.saits_GTEE.model import SAITS_GTEE
from diy_pypots.models.timesnet_GTEE.model import TimesNet_GTEE
from diy_pypots.models.brits_GTEE.model import BRITS_GTEE
from diy_pypots.models.timemixer_GTEE.model import TimeMixer_GTEE
from diy_pypots.models.moderntcn_GTEE.model import ModernTCN_GTEE
from diy_pypots.models.gpvae_GTEE.model import GPVAE_GTEE

from diy_pypots.optim.adam import Adam
from diy_pypots.utils.logging import logger
from diy_pypots.utils.metrics.error import calc_mae, calc_mse, calc_mre
from diy_pypots.utils.random import set_random_seed

from configs.global_config import (
    TORCH_N_THREADS,
    RANDOM_SEEDS,
)
from configs.hpo_results import HPO_RESULTS
from my_utils.data_utils import get_datasets_path_with_TS, get_TS_prior_data, calculate_main_freqs

SUPPORT_MODELS = {
    "Pure_GTEE": Pure_GTEE,
    "SAITS_GTEE": SAITS_GTEE,
    "TimesNet_GTEE": TimesNet_GTEE,
    "BRITS_GTEE": BRITS_GTEE,
    "TimeMixer_GTEE": TimeMixer_GTEE,
    "ModernTCN_GTEE": ModernTCN_GTEE,
    "GPVAE_GTEE": GPVAE_GTEE,
}
SUPPORT_DATASETS = ["ETT_h1",
                    "BeijingAir",
                    "ETT_m1",
                    "wind_turbine",
                    ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(SUPPORT_MODELS.keys()),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=SUPPORT_DATASETS,
    )
    parser.add_argument(
        "--dataset_fold_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--saving_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="device to run the model, e.g. cuda:0",
        required=True,
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--load_main_freqs_extra",
        type=bool,
        default=False,
        help="If set to true, the main frequencies will be loaded from external files."
    )
    parser.add_argument(
        "--impute_all_sets",
        help="whether to impute all sets or only the test set",
        action="store_true",
    )
    args = parser.parse_args()
    torch.set_num_threads(TORCH_N_THREADS)
    
    if "GTEE" in args.model:
        (
            train_set,
            val_set,
            test_set,
        ) = get_datasets_path_with_TS(args.dataset_fold_path,with_time_stamps=True)
    else:
        (
            train_set,
            val_set,
            test_set,
        ) = get_datasets_path_with_TS(args.dataset_fold_path,with_time_stamps=False)
    mae_collector = []
    mse_collector = []
    mre_collector = []
    time_collector = []
    
    result_saving_path = os.path.join(args.saving_path, f"{args.model}_{args.dataset}")
    for n_round in range(args.n_rounds):
        set_random_seed(RANDOM_SEEDS[n_round])
        round_saving_path = os.path.join(result_saving_path, f"round_{n_round}")
        hyperparameters = HPO_RESULTS[args.dataset][args.model].copy()
        lr = hyperparameters.pop("lr")
        hyperparameters["device"] = args.device
        hyperparameters["saving_path"] = round_saving_path
        hyperparameters["model_saving_strategy"] = "best"
        hyperparameters["optimizer"] = Adam(lr=lr)
        
        if "GTEE" in args.model:
            prior_dict = get_TS_prior_data(args.dataset_fold_path)
            hyperparameters["TS_dataset"] = prior_dict["TS_dataset"]
            hyperparameters["TS_start_num"] = prior_dict["TS_start_num"]
            hyperparameters["TS_gap"] = prior_dict["TS_gap"]
            hyperparameters["main_freqs"] = calculate_main_freqs(config=hyperparameters,
                                                                    read_from_extra=args.load_main_freqs_extra,
                                                                    auto_saving=True)
            hyperparameters["freqs_missing"]=False
            
        model = SUPPORT_MODELS[args.model](**hyperparameters)
        model.fit(train_set=train_set, val_set=val_set)
        
        start_time = time.time()
        if args.model in ["GPVAE", "GPVAE_GTEE"]:
            results = model.predict(test_set, n_sampling_times=10)
            test_set_imputation = results["imputation"].mean(axis=1)
        else:
            results = model.predict(test_set)
            
            test_set_imputation = results["imputation"]
        test_X_ori = test_set["X_ori"]
        test_indicating_mask = test_set["Indicating_mask"]
        
        time_collector.append(time.time() - start_time)
        mae = calc_mae(test_set_imputation, test_X_ori, test_indicating_mask)
        mse = calc_mse(test_set_imputation, test_X_ori, test_indicating_mask)
        mre = calc_mre(test_set_imputation, test_X_ori, test_indicating_mask)
        mae_collector.append(mae)
        mse_collector.append(mse)
        mre_collector.append(mre)
        
        train_set_imputation, val_set_imputation = None, None
        
        if args.impute_all_sets:
            if args.model in ["GPVAE", "GPVAE_GTEE"]:
                train_set_imputation = model.predict(train_set, n_sampling_times=10)[
                    "imputation"
                ].mean(axis=1)
                val_set_imputation = model.predict(val_set, n_sampling_times=10)[
                    "imputation"
                ].mean(axis=1)
            else:
                train_set_imputation = model.predict(train_set)["imputation"]
                val_set_imputation = model.predict(val_set)["imputation"]
        
        pickle_dump(
            {
                "train_set_imputation": train_set_imputation,
                "val_set_imputation": val_set_imputation,
                "test_set_imputation": test_set_imputation,
            },
            os.path.join(round_saving_path, "imputation.pkl"),
        )
        logger.info(
            f"Round{n_round} - {args.model} on {args.dataset}: MAE={mae:.4f}, MSE={mse:.4f}, MRE={mre:.4f}"
        )
        
    mean_mae, mean_mse, mean_mre = (
        np.mean(mae_collector),
        np.mean(mse_collector),
        np.mean(mre_collector),
    )
    std_mae, std_mse, std_mre = (
        np.std(mae_collector),
        np.std(mse_collector),
        np.std(mre_collector),
    )
    num_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    logger.info(
        f"Done! Final results:\n"
        f"Averaged {args.model} ({num_params:,} params) on {args.dataset}: "
        f"MAE={mean_mae:.4f} ± {std_mae}, "
        f"MSE={mean_mse:.4f} ± {std_mse}, "
        f"MRE={mean_mre:.4f} ± {std_mre}, "
        f"average inference time={np.mean(time_collector):.2f}"
    )
