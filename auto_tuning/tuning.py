"""
CLI tools to help initialize environments for running and developing PyPOTS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

# Modifications made by Yuan Feng et al., 2026:
# Using absolute import paths; Adjustments aimed at tuning GTEE hyperparameters via NNI.

import inspect
import os
from argparse import ArgumentParser, Namespace

import torch

from auto_tuning.base import BaseCommand
from auto_tuning.utils import load_package_from_path
from diy_pypots.data.saving.h5 import load_dict_from_h5

from diy_pypots.models.pure_GTEE.model import Pure_GTEE
from diy_pypots.models.saits_GTEE.model import SAITS_GTEE
from diy_pypots.models.timesnet_GTEE.model import TimesNet_GTEE
from diy_pypots.models.brits_GTEE.model import BRITS_GTEE
from diy_pypots.models.timemixer_GTEE.model import TimeMixer_GTEE
from diy_pypots.models.moderntcn_GTEE.model import ModernTCN_GTEE
from diy_pypots.models.gpvae_GTEE.model import GPVAE_GTEE

from diy_pypots.optim.adam import Adam
from diy_pypots.utils.logging import logger
from diy_pypots.utils.random import set_random_seed
from my_utils.data_utils import get_datasets_path_with_TS, get_TS_prior_data, calculate_main_freqs

try:
    import nni
except ImportError:
    logger.error(
        "❌ Hyperparameter tuning mode needs NNI (https://github.com/microsoft/nni) installed, "
        "but is missing in the current environment."
    )

NN_MODELS = {
    # imputation models, sorted by the first letter of the model name
    "Pure_GTEE": Pure_GTEE,
    "SAITS_GTEE": SAITS_GTEE,
    "TimesNet_GTEE": TimesNet_GTEE,
    "BRITS_GTEE": BRITS_GTEE,
    "TimeMixer_GTEE": TimeMixer_GTEE,
    "ModernTCN_GTEE": ModernTCN_GTEE,
    "GPVAE_GTEE": GPVAE_GTEE,
}

os.environ['enable_tuning']="1"

def env_command_factory(args: Namespace):
    return TuningCommand(
        args.model,
        args.model_package_path,
        args.dataset_path,
        args.lazy_load,
        args.torch_n_threads,
        args.load_main_freqs_extra,
    )


class TuningCommand(BaseCommand):
    """CLI tools helping users and developer setup python environments for running and developing PyPOTS.

    Notes
    -----
    Using this tool supposes that you've already installed `pypots` with at least the scope of `basic` dependencies.
    Please refer to file setup.cfg in PyPOTS project's root dir for definitions of different dependency scopes.

    Examples
    --------
    $ pypots-cli tuning --model pypots.imputation.SAITS --train_set path_to_the_train_set --val_set path_to_the_val_set

    """

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "tuning",
            help="CLI tools helping run hyper-parameter tuning for specified models",
            allow_abbrev=True,
        )
        sub_parser.add_argument(
            "--model",
            dest="model",
            type=str,
            required=True,
            help="Install specified dependencies in the current python environment",
        )
        sub_parser.add_argument(
            "--model_package_path",
            dest="model_package_path",
            type=str,
            required=False,
            help="If the model is not in the pypots package, specify the path to the model package here.",
        )
        sub_parser.add_argument(
            "--dataset_path",
            dest="dataset_path",
            type=str,
            required=True,
            help="",
        )
        sub_parser.add_argument(
            "--lazy_load",
            dest="lazy_load",
            action="store_true",
            help="Whether to use lazy loading for the dataset. If `True`, the dataset will be lazy loaded for model "
            "training, i.e. only the current batch will be fetched from the file. Lazy loading needs less memory but "
            "more time and CPU rate to read data each time.",
        )
        sub_parser.add_argument(
            "--torch_n_threads",
            dest="torch_n_threads",
            type=int,
            default=1,
            help="The input value for torch.set_num_threads()",
        )
        sub_parser.add_argument(
            "--load_main_freqs_extra",
            help="if it is set to true, the main freqs will be loaded from outer files",
            action="store_true",
        )
        
        sub_parser.set_defaults(func=env_command_factory)

    def __init__(
        self,
        model: str,
        model_package_path: str,
        dataset_path: str,
        lazy_load: bool = False,
        torch_n_threads: int = 1,
        load_main_freqs_extra: bool = False,
    ):
        self._model = model
        self._model_package_path = model_package_path
        self._dataset_path = dataset_path
        self._lazy_load = lazy_load
        self._torch_n_threads = torch_n_threads
        self._load_main_freqs_extra = load_main_freqs_extra

    def checkup(self):
        """Run some checks on the arguments to avoid error usages"""
        pass

    def run(self):
        """Execute the given command."""

        # set with PyPOTS default random seed
        random_seed = os.getenv("random_seed", False)
        if random_seed:
            set_random_seed(int(random_seed))
        else:
            set_random_seed()

        # set the number of threads for torch, avoid using too many CPU cores
        torch.set_num_threads(self._torch_n_threads)
        logger.info(f"Have set the num_threads.")
        if os.getenv("enable_tuning", False):
            # fetch a new set of hyperparameters from NNI tuner
            logger.info(f"Have set the log_info.")
            tuner_params = nni.get_next_parameter()
            logger.info(f"The tunner assigns a new group of params: {tuner_params}")
            # get the specified model class
            if self._model not in NN_MODELS:
                logger.info(
                    f"The specified model {self._model} is not in PyPOTS. Available models are {NN_MODELS.keys()}. "
                    f"Trying to fetch it from the given model package {self._model_package_path}"
                )
                assert self._model_package_path is not None, (
                    f"The given model {self._model} is not in PyPOTS. "
                    f"Please give the full import path of the model in PyPOTS like pypots.imputation.SAITS\n"
                    f"If you're trying to tune an outside model, "
                    f"please specify the path to the model package with argument `--model_package_path`."
                )
                model_package = load_package_from_path(self._model_package_path)
                assert self._model in model_package.__all__, (
                    f"{self._model} is not in the given model package {self._model_package_path}"
                    f"Please ensure that the model class is in the __all__ list of the model package."
                )
                model_class = getattr(model_package, self._model)
            else:
                if self._model_package_path is not None:
                    logger.warning(
                        f"‼️ Find the specified model {self._model} in PyPOTS, "
                        f"but also find the argument --model_package_path is not None."
                        f"Note that --model_package_path is ignored."
                    )

                model_class = NN_MODELS[self._model]
            # pop out the learning rate
            lr = tuner_params.pop("lr")

            # check if hyperparameters match
            model_all_arguments = inspect.signature(model_class).parameters.keys()
            tuner_params_set = set(tuner_params.keys())
            model_arguments_set = set(model_all_arguments)
            if_hyperparameter_match = tuner_params_set.issubset(model_arguments_set)
            if not if_hyperparameter_match:  # raise runtime error if mismatch
                hyperparameter_intersection = tuner_params_set.intersection(
                    model_arguments_set
                )
                mismatched = tuner_params_set.difference(
                    set(hyperparameter_intersection)
                )
                raise RuntimeError(
                    f"Hyperparameters do not match. Mismatched hyperparameters "
                    f"(in the tuning configuration but not in {model_class.__name__}'s arguments): {list(mismatched)}"
                )

            # initializing optimizer and model
            # if tuning a GAN model, we need two optimizers
            if "G_optimizer" in model_all_arguments:
                # optimizer for the generator
                tuner_params["G_optimizer"] = Adam(lr=lr)
                # optimizer for the discriminator
                tuner_params["D_optimizer"] = Adam(lr=lr)
            else:
                tuner_params["optimizer"] = Adam(lr=lr)
            
            if "GTEE" in self._model:
                prior_dict = get_TS_prior_data(self._dataset_path)
                tuner_params["TS_dataset"] = prior_dict["TS_dataset"]
                tuner_params["TS_start_num"] = prior_dict["TS_start_num"]
                tuner_params["TS_gap"] = prior_dict["TS_gap"]
                parts = self._dataset_path.strip("/").split("/")
                dataset_nickname = parts[-2]
                tuner_params["saving_path"] = os.path.join("prior_knowledge",f"{self._model}_{dataset_nickname}")
                tuner_params["main_freqs"] = calculate_main_freqs(config=tuner_params,
                                                                    read_from_extra=self._load_main_freqs_extra,
                                                                    auto_saving=False)
                tuner_params["freqs_missing"]=False
                tuner_params.pop(("saving_path"))
            
            # init an instance with the given hyperparameters for the model class
            model = model_class(**tuner_params)
            if "GTEE" in self._model:
                (
                    train_set,
                    val_set,
                    test_set,
                ) = get_datasets_path_with_TS(self._dataset_path,with_time_stamps=True)
            else:
                (
                    train_set,
                    val_set,
                    test_set,
                ) = get_datasets_path_with_TS(self._dataset_path,with_time_stamps=False)
            
            # load the dataset
            # if self._lazy_load:
            #     train_set, val_set = self._train_set, self._val_set
            # else:
            #     logger.info(
            #         "Option lazy_load is set as False, hence loading all data from file..."
            #     )
            #     train_set = load_dict_from_h5(self._train_set)
            #     val_set = load_dict_from_h5(self._val_set)
            
            # train the model and report to NNI
            model.fit(train_set=train_set, val_set=val_set)
        else:
            raise RuntimeError("Argument `enable_tuning` is not set. Aborting...")
