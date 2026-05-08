"""
Description: Missing data generation file for the BeijingAir dataset, adapted from Awesome_Imputation (https://github.com/WenjieDu/Awesome_Imputation).

Copyright (c) Yuan Feng et al., 2026
License: MIT License
"""

from diy_benchpots.datasets.beijing_multisite_air_quality import preprocess_beijing_air_quality_with_timestamps
from diy_pypots.utils.random import set_random_seed
from my_utils.data_utils import organize_and_save_with_timestamps
import os


if __name__ == "__main__":
    set_random_seed(2024)
    step = 24
    with_time_stamps=True
    root_dir = "data/generated_datasets/beijing_air"
    
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    # block 05
    rate = 0.0055
    block_len = 6
    block_width = 6
    beijing_air_quality_block_05 = preprocess_beijing_air_quality_with_timestamps(
        rate=rate,
        n_steps=step,
        pattern="block",
        with_time_stamps=with_time_stamps,
        **{"block_len":block_len,
        "block_width":block_width}
    )
    save_floder_name = f"beijing_air_rate{int(rate * 10):02d}_step{step}_block_blocklen{block_len}"
    save_path = os.path.join(root_dir, save_floder_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    organize_and_save_with_timestamps(
        data_dict=beijing_air_quality_block_05,
        saving_dir=save_path,
        with_time_stamps=with_time_stamps,
    )
    
    # subseq 05
    rate = 0.5
    seq_len = 18
    beijing_air_quality_seq_05 = preprocess_beijing_air_quality_with_timestamps(
        rate=rate,
        n_steps=step,
        pattern="subseq",
        with_time_stamps=with_time_stamps,
        **{"seq_len": seq_len},
    )
    save_floder_name = f"beijing_air_rate{int(rate * 10):02d}_step{step}_subseq_seqlen{seq_len}"
    save_path = os.path.join(root_dir, save_floder_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    organize_and_save_with_timestamps(data_dict=beijing_air_quality_seq_05,
                                        saving_dir=save_path,
                                        with_time_stamps=with_time_stamps,
    )
    
    # point 05
    rate=0.5
    beijing_air_quality_point_05 = preprocess_beijing_air_quality_with_timestamps(
        rate=rate,
        n_steps=step,
        pattern="point",
        with_time_stamps=with_time_stamps,
    )
    save_floder_name = f"beijing_air_rate{int(rate * 10):02d}_step{step}_point"
    save_path = os.path.join(root_dir, save_floder_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    organize_and_save_with_timestamps(data_dict=beijing_air_quality_point_05,
                                        saving_dir=save_path,
                                        with_time_stamps=with_time_stamps,
                                        )
