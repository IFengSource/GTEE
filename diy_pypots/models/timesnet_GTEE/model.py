"""
Description: The implementation of TimesNet-based GTEE imputation model for the partially-observed time-series.
Implemented with reference to the TimesNet algorithm in PyPOTS (Wenjie Du <wenjay.du@gmail.com>).

Copyright (c) Yuan Feng et al., 2026
License: MIT License
"""

from typing import Union, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from diy_pypots.models.timesnet_GTEE.core import _TimesNet_GTEE
from diy_pypots.models.timesnet_GTEE.data import DatasetForTimesNet_GTEE
from diy_pypots.models.base import BaseNNImputer
from diy_pypots.data.checking import key_in_data_set
from diy_pypots.data.dataset.base import BaseDataset_with_TS
from diy_pypots.optim.adam import Adam
from diy_pypots.optim.base import Optimizer
from diy_pypots.utils.logging import logger
from data.preprocess.pipeline import get_timestamp_coding_parameters

class TimesNet_GTEE(BaseNNImputer):

    def __init__(
        self,
        TS_dataset: np.ndarray,
        ts_main_freq_num: int,
        ts_main_freq_solvers: list,
        ts_solvers_combine: bool,
        ts_sampling_num: int,
        ts_sampling_length:int,
        ts_scaling_factor:float,
        ts_freqs_merge_value: float,
        ts_refining_mode: str,
        
        TS_ch_embedding_dim: int,
        TS_hidden_dim: int,
        TS_reg_weight: float,
        TS_fusion_conv_kernels: list,
        TS_fusion_hidden_dim: int,
        TS_loss_weight: float,
        TS_dropout: float,
        TS_start_num: int,
        TS_gap: int,
        
        n_steps: int,
        n_features: int,
        n_layers: int,
        top_k: int,
        d_model: int,
        d_ffn: int,
        n_kernels: int,
        dropout: float = 0,
        apply_nonstationary_norm: bool = False,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = None,
        optimizer: Optional[Optimizer] = Adam(),
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
        
        main_freqs: np.ndarray=None,
        freqs_missing: bool=True,
    ):
        super().__init__(
            batch_size,
            epochs,
            patience,
            num_workers,
            device,
            saving_path,
            model_saving_strategy,
            verbose,
        )
        
        self.n_steps = n_steps
        self.n_features = n_features
        
        self.TS_start_num = TS_start_num
        self.TS_gap = TS_gap
        
        if freqs_missing or (main_freqs is None):
            
            self.ts_sampling_num = ts_sampling_num
            self.ts_sampling_length = ts_sampling_length
            self.ts_solvers_combine = ts_solvers_combine
            self.ts_solver_num = len(ts_main_freq_solvers)
            self.ts_main_freq_num = ts_main_freq_num
            
            assert ts_sampling_length <= TS_dataset.shape[0], f"the ts_sampling_length {ts_sampling_length} is larger than the total steps {TS_dataset.shape[0]}"
            
            while self.ts_sampling_num + self.ts_sampling_length-1 > TS_dataset.shape[0]:
                self.ts_sampling_num -= 1
                logger.warning(
                    f"⚠️ reduce the ts_sampling_num {ts_sampling_num} to {self.ts_sampling_num}, since the total sample length is out of range."
                )
                
            if (ts_sampling_length//2-1) < ts_main_freq_num:
                self.ts_main_freq_num = ts_sampling_length//2-1
                logger.warning(
                    f"⚠️ ts_main_freqs_num {ts_main_freq_num} is bigger than half ts_sampling_length {ts_sampling_length//2-1}, it is reset to {self.ts_main_freq_num}."
                )
            
            TS_freqs_params = get_timestamp_coding_parameters(dataset=TS_dataset,
                                                                main_freq_num=self.ts_main_freq_num,
                                                                main_freq_solvers=ts_main_freq_solvers,
                                                                solvers_combine=self.ts_solvers_combine,
                                                                sampling_num=self.ts_sampling_num,
                                                                sampling_length=self.ts_sampling_length,
                                                                scaling_factor=ts_scaling_factor,
                                                                freqs_merge_value=ts_freqs_merge_value,
                                                                refining_mode=ts_refining_mode)
        else:
            TS_freqs_params=main_freqs
        
        self.TS_freqs_params = TS_freqs_params
        self.TS_used_freqs_num = TS_freqs_params.shape[-1]
        self.TS_ch_embedding_dim = TS_ch_embedding_dim
        
        self.n_layers = n_layers
        self.top_k = top_k
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.n_kernels = n_kernels
        self.dropout = dropout
        self.apply_nonstationary_norm = apply_nonstationary_norm
        
        self.TS_hidden_dim=TS_hidden_dim
        self.TS_reg_weight=TS_reg_weight
        self.TS_fusion_conv_kernels=TS_fusion_conv_kernels
        self.TS_fusion_hidden_dim=TS_fusion_hidden_dim
        self.TS_loss_weight=TS_loss_weight
        self.TS_dropout=TS_dropout
        
        self.model = _TimesNet_GTEE(
            self.TS_freqs_params,
            self.TS_used_freqs_num,
            self.TS_ch_embedding_dim,
            self.TS_hidden_dim,
            
            self.TS_reg_weight,
            self.TS_loss_weight,
            self.TS_dropout,
            self.TS_fusion_conv_kernels,
            self.TS_fusion_hidden_dim,
            
            self.n_layers,
            self.n_steps,
            self.n_features,
            self.top_k,
            self.d_model,
            self.d_ffn,
            self.n_kernels,
            self.dropout,
            self.apply_nonstationary_norm,
        )
        self._send_model_to_given_device()
        self._print_model_size()
        
        # set up the optimizer
        self.optimizer = optimizer
        self.optimizer.init_optimizer(self.model.parameters())
    
    def _assemble_input_for_training(self, data: list) -> dict:
        (
            indices,
            X,
            missing_mask,
            X_ori,
            indicating_mask,
            X_TS,
        ) = self._send_data_to_given_device(data)

        inputs = {
            "X": X,
            "missing_mask": missing_mask,
            "X_ori": X_ori,
            "indicating_mask": indicating_mask,
            "X_TS": X_TS,
        }

        return inputs

    def _assemble_input_for_validating(self, data: list) -> dict:
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: list) -> dict:
        indices, X, missing_mask, X_TS = self._send_data_to_given_device(data)

        inputs = {
            "X": X,
            "missing_mask": missing_mask,
            "X_TS": X_TS,
        }

        return inputs

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        # Step 1: wrap the input data with classes Dataset and DataLoader
        training_set = DatasetForTimesNet_GTEE(
            train_set, 
            return_X_ori=False, 
            return_y=False, 
            return_TS=True,
            file_type=file_type
        )
        training_loader = DataLoader(
            training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = None
        if val_set is not None:
            if not key_in_data_set("X_ori", val_set):
                raise ValueError("val_set must contain 'X_ori' for model validation.")
            val_set = DatasetForTimesNet_GTEE(
                val_set, 
                return_X_ori=True, 
                return_y=False, 
                return_TS=True,
                file_type=file_type
            )
            val_loader = DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        # Step 2: train the model and freeze it
        self._train_model(training_loader, val_loader)
        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()  # set the model as eval status to freeze it.

        # Step 3: save the model if necessary
        self._auto_save_model_if_necessary(confirm_saving=False)

    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        return_latent_vars: bool = False,
    ) -> dict:
        # Supports output of mean and log-variance estimates
        # Step 1: wrap the input data with classes Dataset and DataLoader
        self.model.eval()  # set the model as eval status to freeze it.
        test_set = BaseDataset_with_TS(
            test_set,
            return_X_ori=False,
            return_X_pred=False,
            return_y=False,
            return_TS=True,
            file_type=file_type,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        imputation_collector = []
        TS_dist_mus_collector = []
        TS_dist_logvars_collector = []

        # Step 2: process the data with the model
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self._assemble_input_for_testing(data)
                results = self.model.forward(inputs, training=False)
                imputation_collector.append(results["imputed_data"])
                
                if return_latent_vars:
                    TS_dist_mus_collector.append(
                        results["TS_dist_mus"].cpu().numpy()
                    )
                    TS_dist_logvars_collector.append(
                        results["TS_dist_logvars"].cpu().numpy()
                    )
                    
        # Step 3: output collection and return
        imputation = torch.cat(imputation_collector).cpu().detach().numpy()
        result_dict = {
            "imputation": imputation,
        }
        if return_latent_vars:
            latent_var_collector = {
                "TS_dist_mus": np.concatenate(
                    TS_dist_mus_collector
                ),
                "TS_dist_logvars": np.concatenate(
                    TS_dist_logvars_collector
                ),
            }
            result_dict["latent_vars"] = latent_var_collector
        return result_dict

    def impute(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> np.ndarray:

        result_dict = self.predict(test_set, file_type=file_type)
        return result_dict["imputation"]
