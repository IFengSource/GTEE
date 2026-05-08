# GTEE: A Global Timestamp Encoding Enhanced Method for Robust Time Series Imputation in Complex Missing Scenarios

This repository provides the official implementation of the paper:

**GTEE: A Global Timestamp Encoding Enhanced Method for Robust Time Series Imputation in Complex Missing Scenarios**

## Requirements

We recommend creating the environment with Conda:

```bash
conda env create -f requirements.yaml
conda activate TSI
```

If you use NNI for automatic model hyperparameter tuning and encounter the error:
`ModuleNotFoundError: No module named 'pkg_resources'`

Please downgrade the `setuptools` version with the following command:
```bash
pip install setuptools==68.2.2
```

## Dataset Preparation

The missing-value datasets are constructed from raw time-series data through sample splitting, normalization, and artificial masking.

This project includes scripts for dataset generation with synchronized timestamp extraction and normalization.

For example, to generate the missing dataset for **ETTh1**, follow the steps below.

### Generated dataset structure

The following directory structure covers three missingness patterns: **Point 05**, **Block 05**, and **Subseq 05**.

```text
data/
    generated_datasets/
        ETTh1/
            ett_rate05_step48_point/
                train.h5
                test.h5
                val.h5
                TS_prior.h5
            ett_rate03_step48_block_blocklen6/
                train.h5
                test.h5
                val.h5
                TS_prior.h5
            ett_rate05_step48_subseq_seqlen36/
                train.h5
                test.h5
                val.h5
                TS_prior.h5
```

### Generation Command

```bash
cd <your_project_path>
PYTHONPATH=. python -u data/data_factory/dataset_generating_ETTh1.py
```
> **Note:** Replace `<your_project_path>` with the root directory of your local project.


## Quick Test

To perform imputation on **ETTh1** with **SAITS_GTEE**, first change to the project root directory, then run:

```bash
sh scripts/SAITS_GTEE/run_single.sh
```

The training process and evaluation results for the three missing patterns, **Point 05**, **Block 05**, and **Subseq 05**, can be found in the three log files under `logs/imputation`.

## Hyper-parameter Optimization (Stage 1)

In Stage 1, we use the **NNI** hyper-parameter optimization framework to tune the **Timestamp-Dominated Imputation Model** (project alias: **Pure_GTEE**).

Using **ETTh1** as an example, the procedure is as follows:

1. Open `scripts/Pure_GTEE/run_nni.sh`, update line 3 from `export PYTHONPATH=/root/autodl-tmp/GTEE` to your actual project path, and save the file. 

2. Open `auto_tuning/models_tuning_space_setting/Pure_GTEE/ETT_h1/Pure_GTEE_searching_config.yml`, update line 8 from `trialCodeDirectory: /root/autodl-tmp/GTEE` to your actual project path, and save the file. 

3. Run the following command in the terminal:

    ```bash
    sh scripts/Pure_GTEE/run_nni.sh
    ```

   This will launch NNI tuning according to the configuration file `Pure_GTEE_searching_config.yml`. For more advanced or customized tuning settings, please refer to the official NNI documentation: [https://nni.readthedocs.io/en/stable/index.html](https://nni.readthedocs.io/en/stable/index.html).

 4. Open the NNI web UI in your browser (default: `http://127.0.0.1:8080`). After tuning is complete, copy the best-performing hyper-parameter set, selected by the lowest validation loss, into `configs/hpo_results/ett_h1.py` under `ETT_h1["Pure_GTEE"]`, for example:
    ```python
    ETT_h1 = {
        "Pure_GTEE": {
            "ts_main_freq_num": 5,
            "ts_main_freq_solvers": ["ACF", "Burg_roots"],
            "ts_solvers_combine": False,
            "ts_sampling_num": 1,
            "ts_sampling_length": 256,
            "ts_scaling_factor": 0.6067428453718272,
            "ts_freqs_merge_value": 0.022596998462003477,
            "ts_refining_mode": "median",
            # ... other hyper-parameters
        },
        ...
    }
    ```
5. Stop the NNI process with:

    ```bash
    nnictl stop --all
    ```
6. Based on the Stage 1 hyper-parameters, compute the final main frequency mining result $\bm{F}_{main}$ using:

    ```bash
    python -u calculate_main_freqs.py --model Pure_GTEE --dataset ETT_h1 --dataset_fold_path data/generated_datasets/ETTh1/ett_rate05_step48_point
    ```

   The resulting $\bm{F}_{main}$ file will be saved to `prior_knowledge/Pure_GTEE_ETTh1/main_freqs.h5`.


> **Note:** Under the experimental setting used in the paper, Stage 1 tuning is performed only on the dataset generated under the **Point 05** missing pattern, and the resulting hyper-parameters are shared across all missing patterns.

## Hyper-parameter Optimization (Stage 2)

Stage 2 performs joint hyper-parameter tuning for the **timestamp-local joint imputation framework**.

Using **SAITS_GTEE** on **ETTh1** as an example, the procedure is as follows:

1. Since Stage 2 tuning does not modify the Stage 1 hyper-parameters, you can directly reuse the Stage 1 $\bm{F}_{main}$ to accelerate optimization. For example, on Linux:

    ```bash
    mkdir -p ./prior_knowledge/SAITS_GTEE_ETTh1 && cp ./prior_knowledge/Pure_GTEE_ETTh1/main_freqs.h5 ./prior_knowledge/SAITS_GTEE_ETTh1/
    ```
2. Open `scripts/SAITS_GTEE/run_nni.sh`, update line 3 from `export PYTHONPATH=/root/autodl-tmp/GTEE` to your actual project path, and save the file.

3. Open `auto_tuning/models_tuning_space_setting/SAITS_GTEE/ETT_h1/SAITS_GTEE_searching_config.yml`, update line 8 from `trialCodeDirectory: /root/autodl-tmp/GTEE` to your actual project path, and save the file. 

4. Run the following command in the terminal:

    ```bash
    sh scripts/SAITS_GTEE/run_nni.sh
    ```

   This will launch NNI tuning according to the configuration file `SAITS_GTEE_searching_config.yml`.
5. After tuning is complete, copy the best-performing hyper-parameter set, selected by the lowest validation loss, into `configs/hpo_results/ett_h1.py` under `ETT_h1["SAITS_GTEE"]`. Note that all hyper-parameters prefixed with `ts_` must remain identical to the Stage 1 settings rather than using the Stage 2 tuning results. For example:

    ```python
    ETT_h1 = {
        "Pure_GTEE": {
            "ts_main_freq_num": 5,
            "ts_main_freq_solvers": ["ACF", "Burg_roots"],
            "ts_solvers_combine": False,
            "ts_sampling_num": 1,
            "ts_sampling_length": 256,
            "ts_scaling_factor": 0.6067428453718272,
            "ts_freqs_merge_value": 0.022596998462003477,
            "ts_refining_mode": "median",
            # ... other hyper-parameters
        },
        "SAITS_GTEE": {
            # Stage 1 settings
            "ts_main_freq_num": 5,
            "ts_main_freq_solvers": ["ACF", "Burg_roots"],
            "ts_solvers_combine": False,
            "ts_sampling_num": 1,
            "ts_sampling_length": 256,
            "ts_scaling_factor": 0.6067428453718272,
            "ts_freqs_merge_value": 0.022596998462003477,
            "ts_refining_mode": "median",

            # Stage 2 tuned hyper-parameters
            "TS_ch_embedding_dim": 32,
            "TS_hidden_dim": 8,
            "TS_reg_weight": 0.014487810264286003,
            "TS_fusion_conv_kernels": [5, 7, 9],
            "TS_fusion_hidden_dim": 8,
            "TS_loss_weight": 0.004579005749274037,
            "TS_dropout": 0.2,
            "n_steps": 48,
            "n_features": 7,
            "epochs": 100,
            "patience": 10,
            "n_layers": 2,
            "d_model": 64,
            "d_ffn": 256,
            "n_heads": 1,
            "d_k": 128,
            "d_v": 256,
            "dropout": 0,
            "attn_dropout": 0.2,
            "lr": 0.0011752356815298876
        },
        ...
    }
    ```
6. Stop the NNI process with:

    ```bash
    nnictl stop --all
    ```
7. Refer to the Quick Test section for subsequent analysis and evaluation.

## Acknowledgements

The overall project framework is built upon **PyPOTS**: [https://github.com/WenjieDu/PyPOTS](https://github.com/WenjieDu/PyPOTS)

The hyper-parameter search spaces for the baseline models are primarily based on **Awesome_Imputation**: [https://github.com/WenjieDu/Awesome_Imputation](https://github.com/WenjieDu/Awesome_Imputation)

We sincerely thank the authors, especially Wenjie Du and collaborators, for their valuable contributions.

## Citation

If this work is helpful for your research, please cite our paper:

```bibtex
@article{feng2026gtee,
  title={GTEE: A global timestamp encoding enhanced method for robust time series imputation in complex missing scenarios},
  author={Feng, Yuan and Liu, Zhenyu and Liu, Hui and Jiang, Xishan and Liu, Xin and Sun, Jiacheng and Tan, Jianrong},
  journal={Information Fusion},
  pages={104220},
  year={2026},
  publisher={Elsevier}
}
```
