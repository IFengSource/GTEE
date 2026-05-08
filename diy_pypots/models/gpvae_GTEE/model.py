"""
Description: The implementation of GPVAE-based GTEE imputation model for the partially-observed time-series.
Implemented with reference to the GPVAE algorithm in PyPOTS (Wenjie Du <wenjay.du@gmail.com>).

Copyright (c) Yuan Feng et al., 2026
License: MIT License
"""


import os
from typing import Union, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import nni
except ImportError:
    pass


from diy_pypots.models.gpvae_GTEE.core import _GPVAE_GTEE
from diy_pypots.models.gpvae_GTEE.data import DatasetForGPVAE_GTEE
from diy_pypots.models.base import BaseNNImputer
from diy_pypots.data.checking import key_in_data_set
from diy_pypots.optim.adam import Adam
from diy_pypots.optim.base import Optimizer
from diy_pypots.utils.logging import logger
from diy_pypots.utils.metrics.error import calc_mse

from data.preprocess.pipeline import get_timestamp_coding_parameters

class GPVAE_GTEE(BaseNNImputer):
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
        latent_size: int,
        encoder_sizes: tuple = (64, 64),
        decoder_sizes: tuple = (64, 64),
        kernel: str = "cauchy",
        beta: float = 0.2,
        M: int = 1,
        K: int = 1,
        sigma: float = 1.0,
        length_scale: float = 7.0,
        kernel_scales: int = 1,
        window_size: int = 3,
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = None,
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
        
        available_kernel_type = ["cauchy", "diffusion", "rbf", "matern"]
        assert (
            kernel in available_kernel_type
        ), f"kernel should be one of {available_kernel_type}, but got {kernel}"
        
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
        self.TS_hidden_dim=TS_hidden_dim
        self.TS_reg_weight=TS_reg_weight
        self.TS_fusion_conv_kernels=TS_fusion_conv_kernels
        self.TS_fusion_hidden_dim=TS_fusion_hidden_dim
        self.TS_loss_weight=TS_loss_weight
        self.TS_dropout=TS_dropout
        
        self.latent_size = latent_size
        self.kernel = kernel
        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = decoder_sizes
        self.beta = beta
        self.M = M
        self.K = K
        self.sigma = sigma
        self.length_scale = length_scale
        self.kernel_scales = kernel_scales
        
        self.model = _GPVAE_GTEE(
            self.TS_freqs_params,
            self.TS_used_freqs_num,
            self.TS_ch_embedding_dim,
            self.TS_hidden_dim,
            self.TS_reg_weight,
            self.TS_loss_weight,
            self.TS_dropout,
            self.TS_fusion_conv_kernels,
            self.TS_fusion_hidden_dim,
            
            input_dim=self.n_features,
            time_length=self.n_steps,
            latent_dim=self.latent_size,
            kernel=self.kernel,
            encoder_sizes=self.encoder_sizes,
            decoder_sizes=self.decoder_sizes,
            beta=self.beta,
            M=self.M,
            K=self.K,
            sigma=self.sigma,
            length_scale=self.length_scale,
            kernel_scales=self.kernel_scales,
            window_size=window_size,
        )
        self._send_model_to_given_device()
        self._print_model_size()
        
        # set up the optimizer
        self.optimizer = optimizer
        self.optimizer.init_optimizer(self.model.parameters())
        
    def _assemble_input_for_training(self, data: list) -> dict:
        # fetch data
        (
            indices,
            X,
            missing_mask,
            X_TS,
        ) = self._send_data_to_given_device(data)

        # assemble input data
        inputs = {
            "indices": indices,
            "X": X,
            "missing_mask": missing_mask,
            "X_TS": X_TS,
        }

        return inputs
    
    def _assemble_input_for_validating(self, data: list) -> dict:
        # fetch data
        (
            indices,
            X,
            missing_mask,
            X_ori,
            indicating_mask,
            X_TS,
        ) = self._send_data_to_given_device(data)

        # assemble input data
        inputs = {
            "indices": indices,
            "X": X,
            "missing_mask": missing_mask,
            "X_ori": X_ori,
            "indicating_mask": indicating_mask,
            "X_TS": X_TS,
        }

        return inputs
    
    def _assemble_input_for_testing(self, data: list) -> dict:
        return self._assemble_input_for_training(data)
    
    def _train_model(
        self,
        training_loader: DataLoader,
        val_loader: DataLoader = None,
    ) -> None:
        # each training starts from the very beginning, so reset the loss and model dict here
        self.best_loss = float("inf")
        self.best_model_dict = None

        try:
            training_step = 0
            for epoch in range(1, self.epochs + 1):
                self.model.train()
                epoch_train_loss_collector = []
                for idx, data in enumerate(training_loader):
                    training_step += 1
                    inputs = self._assemble_input_for_training(data)
                    self.optimizer.zero_grad()
                    results = self.model.forward(inputs)
                    # use sum() before backward() in case of multi-gpu training
                    results["loss"].sum().backward()
                    self.optimizer.step()
                    epoch_train_loss_collector.append(results["loss"].sum().item())

                    # save training loss logs into the tensorboard file for every step if in need
                    if self.summary_writer is not None:
                        self._save_log_into_tb_file(training_step, "training", results)

                # mean training loss of the current epoch
                mean_train_loss = np.mean(epoch_train_loss_collector)

                if val_loader is not None:
                    self.model.eval()
                    imputation_loss_collector = []
                    with torch.no_grad():
                        for idx, data in enumerate(val_loader):
                            inputs = self._assemble_input_for_validating(data)
                            results = self.model.forward(
                                inputs, training=False, n_sampling_times=1
                            )
                            imputed_data = results["imputed_data"].mean(axis=1)
                            imputation_mse = (
                                calc_mse(
                                    imputed_data,
                                    inputs["X_ori"],
                                    inputs["indicating_mask"],
                                )
                                .sum()
                                .detach()
                                .item()
                            )
                            imputation_loss_collector.append(imputation_mse)

                    mean_val_loss = np.mean(imputation_loss_collector)

                    # save validation loss logs into the tensorboard file for every epoch if in need
                    if self.summary_writer is not None:
                        val_loss_dict = {
                            "imputation_loss": mean_val_loss,
                        }
                        self._save_log_into_tb_file(epoch, "validating", val_loss_dict)

                    logger.info(
                        f"Epoch {epoch:03d} - "
                        f"training loss: {mean_train_loss:.4f}, "
                        f"validation loss: {mean_val_loss:.4f}"
                    )
                    mean_loss = mean_val_loss
                else:
                    logger.info(
                        f"Epoch {epoch:03d} - training loss: {mean_train_loss:.4f}"
                    )
                    mean_loss = mean_train_loss

                if np.isnan(mean_loss):
                    logger.warning(
                        f"‼️ Attention: got NaN loss in Epoch {epoch}. This may lead to unexpected errors."
                    )

                if mean_loss < self.best_loss:
                    self.best_epoch = epoch
                    self.best_loss = mean_loss
                    self.best_model_dict = self.model.state_dict()
                    self.patience = self.original_patience
                else:
                    self.patience -= 1

                # save the model if necessary
                self._auto_save_model_if_necessary(
                    confirm_saving=mean_loss < self.best_loss,
                    saving_name=f"{self.__class__.__name__}_epoch{epoch}_loss{mean_loss}",
                )

                if os.getenv("enable_tuning", False):
                    nni.report_intermediate_result(mean_loss)
                    if epoch == self.epochs - 1 or self.patience == 0:
                        nni.report_final_result(self.best_loss)

                if self.patience == 0:
                    logger.info(
                        "Exceeded the training patience. Terminating the training procedure..."
                    )
                    break

        except KeyboardInterrupt:  # if keyboard interrupt, only warning
            logger.warning("‼️ Training got interrupted by the user. Exist now ...")
        except Exception as e:  # other kind of exception follows below processing
            logger.error(f"❌ Exception: {e}")
            if self.best_model_dict is None:  # if no best model, raise error
                raise RuntimeError(
                    "Training got interrupted. Model was not trained. Please investigate the error printed above."
                )
            else:
                RuntimeWarning(
                    "Training got interrupted. Please investigate the error printed above.\n"
                    "Model got trained and will load the best checkpoint so far for testing.\n"
                    "If you don't want it, please try fit() again."
                )

        if np.isnan(self.best_loss):
            raise ValueError("Something is wrong. best_loss is Nan after training.")

        logger.info(
            f"Finished training. The best model is from epoch#{self.best_epoch}."
        )
        
    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        # Step 1: wrap the input data with classes Dataset and DataLoader
        training_set = DatasetForGPVAE_GTEE(
            train_set,
            return_X_ori=False,
            return_y=False,
            return_TS=True,
            file_type=file_type,
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
            val_set = DatasetForGPVAE_GTEE(
                val_set,
                return_X_ori=True,
                return_y=False,
                return_TS=True,
                file_type=file_type,
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
        n_sampling_times: int = 1,
    ) -> dict:
        
        assert n_sampling_times > 0, "n_sampling_times should be greater than 0."

        self.model.eval()  # set the model as eval status to freeze it.
        test_set = DatasetForGPVAE_GTEE(
            test_set,
            return_X_ori=False,
            return_y=False,
            return_TS=True,
            file_type=file_type
        )
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        imputation_collector = []

        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self._assemble_input_for_testing(data)
                results = self.model.forward(
                    inputs, training=False, n_sampling_times=n_sampling_times
                )
                imputed_data = results["imputed_data"]
                imputation_collector.append(imputed_data)

        imputation = torch.cat(imputation_collector).cpu().detach().numpy()
        result_dict = {
            "imputation": imputation,
        }
        return result_dict

    def impute(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> np.ndarray:

        results_dict = self.predict(test_set, file_type=file_type)
        return results_dict["imputation"]
