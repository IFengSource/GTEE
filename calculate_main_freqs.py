"""
Description: Generate the main frequency file based on Stage 1 parameters of the Timestamp-Dominated Imputation Model (TSDIM) for subsequent training.

Copyright (c) Yuan Feng et al., 2026
License: MIT License
"""

import os
import argparse

from configs.hpo_results import HPO_RESULTS

from diy_pypots.models.pure_GTEE.model import Pure_GTEE

from diy_pypots.utils.logging import logger
from diy_pypots.utils.random import set_random_seed
from my_utils.data_utils import get_TS_prior_data, calculate_main_freqs

SUPPORT_MODELS = {
    "Pure_GTEE": Pure_GTEE,
}

DATASET_NAME_MAPPING = {
    "ETT_m1": 'ETTm1',
    "ETT_h1": 'ETTh1',
    "BeijingAir": 'beijing_air',
    "wind_turbine": "wind_turbine",
}

def get_parser():
    parser = argparse.ArgumentParser(
        description=
        "Calculate main freqs."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="check the main model type is supported",
        required=True,
        choices=list(SUPPORT_MODELS.keys()),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="check the dataset is supported",
        required=True,
        choices=list(DATASET_NAME_MAPPING.keys()),
    )
    parser.add_argument(
        "--dataset_fold_path",
        type=str,
        help="the dataset fold path, where should include 3 H5 files train.h5, val.h5 and test.h5",
        required=True,
    )
    
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    dataset_nickname = DATASET_NAME_MAPPING[args.dataset]
    saving_path = os.path.join("prior_knowledge",f"{args.model}_{dataset_nickname}")
    
    random_seed = os.getenv("random_seed", False)
    if random_seed:
        set_random_seed(int(random_seed))
    else:
        set_random_seed()
    
    hyperparameters = HPO_RESULTS[args.dataset][args.model].copy()
    hyperparameters["saving_path"] = saving_path
    
    assert "GTEE" in args.model, "the model without GTEE is unsupported"
    prior_dict = get_TS_prior_data(args.dataset_fold_path)
    hyperparameters["TS_dataset"] = prior_dict["TS_dataset"]
    hyperparameters["TS_start_num"] = prior_dict["TS_start_num"]
    hyperparameters["TS_gap"] = prior_dict["TS_gap"]
    
    main_freqs = calculate_main_freqs(config=hyperparameters,
                                            read_from_extra=False,
                                            auto_saving=True)
