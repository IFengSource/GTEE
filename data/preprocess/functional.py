"""
Description: Core functional units in the Prior Mining Module.

Copyright (c) Yuan Feng et al., 2026
License: MIT License
"""


import numpy as np
import pandas as pd

from data.preprocess.utils import ACF_main_freqs_solver, Burg_freqz_main_freqs_solver, Burg_roots_main_freqs_solver, FFT_main_freqs_solver, merge_freqs_1d, refine_freqs_1d

def random_sampling_from_dataset(dataset: np.ndarray,
                                    sampling_num: int,
                                    sampling_length: int):
    """Randomly sample several unique samples from the dataset
    
    Args:
        dataset (np.ndarray): Dataset sequence with shape: total_steps × num_features
        sampling_num (int): Number of samples to be sampled
        sampling_length (int): Length of each sampled sample
    
    Returns:
        np.ndarray: Sampled results with shape: sampling_num × num_features × sampling_length
    """
    assert dataset.ndim == 2, f"dataset must be a 2D np array, but got {dataset.ndim}D"
    total_steps, num_features = dataset.shape
    max_start_index = total_steps - sampling_length
    assert sampling_num <= max_start_index + 1, \
        f"Cannot sample {sampling_num} sequences of length {sampling_length} from {total_steps} steps."
    start_indices = np.random.choice(max_start_index + 1, sampling_num, replace=False)

    samples_list = []
    for start_index in start_indices:
        end_index = start_index + sampling_length
        sample = dataset[start_index:end_index, :]
        samples_list.append(sample)
    final_samples = np.stack(samples_list, axis=0)
    return final_samples.transpose(0, 2, 1)

def get_main_freqs_with_multiple_analysis(samples: np.ndarray,
                                            main_freq_num: int,
                                            main_freq_solvers: list,
                                            scaling_factor: float):
    """
    Function for calculating main frequencies using multiple main frequency analysis algorithms

    Args:
        samples (np.ndarray): Sample set with shape: (sampling_num, num_features, sampling_length)
        main_freq_num (int): Number of main frequencies to retain finally
        main_freq_solvers (list): Main frequency solving modes. Candidates: ['ACF', 'Burg_freqz', 'Burg_roots', 'FFT']
        scaling_factor (float): Scaling factor to retain sufficient margin during candidate main frequency screening

    Returns:
        np.ndarray: Candidate frequencies with shape: (sampling_num, num_features, select_freqs_num * len(main_freq_solvers))
    """
    s,c,n = samples.shape
    
    assert set(main_freq_solvers).issubset(['ACF', 'Burg_freqz', 'Burg_roots', 'FFT']), "there are unsupported freq solvers"
    assert 0<scaling_factor and scaling_factor <= 1 and int(main_freq_num/scaling_factor)<=n//2, f"the scaling factor {scaling_factor}  is not suitable."
    
    select_freqs_num = int(main_freq_num/scaling_factor)
    
    main_freq_collector = []
    t_f = select_freqs_num * len(main_freq_solvers)
    for sn in range(s):
        temp_full_ch_freqs = np.zeros((c,t_f))
        for ch in range(c):
            head = 0
            for solver in main_freq_solvers:
                temp_data = pd.Series(samples[sn,ch,:])
                interpolated_temp_data = temp_data.interpolate(method='linear', limit_direction='both')
                td = np.nan_to_num(interpolated_temp_data.values, nan=0.0)
                
                if solver == 'ACF':
                    temp_dominant_freqs,_,_ = ACF_main_freqs_solver(data=td,
                                                                    selected_freqs_num=select_freqs_num,
                                                                    fs=1.)
                elif solver == 'Burg_freqz':
                    p_c = max(n//2,1)
                    temp_dominant_freqs,_,_ = Burg_freqz_main_freqs_solver(data=td,
                                                                        p=p_c,
                                                                        selected_freqs_num=select_freqs_num,
                                                                        fs=1.)
                elif solver == 'Burg_roots':
                    p_c = max(n//2,1)
                    temp_dominant_freqs,_,_ = Burg_roots_main_freqs_solver(data=td,
                                                                            p=p_c,
                                                                            selected_freqs_num=select_freqs_num,
                                                                            fs=1.)
                else:
                    temp_dominant_freqs,_,_ = FFT_main_freqs_solver(data=td,
                                                                    selected_freqs_num=select_freqs_num,
                                                                    fs=1.)
                temp_full_ch_freqs[ch,head:head+select_freqs_num] = temp_dominant_freqs[:]
                head += select_freqs_num
            assert head==t_f, "Unexpected error occurred."
        main_freq_collector.append(temp_full_ch_freqs)
    estimated_main_freqs = np.stack(main_freq_collector,axis=0)
    return estimated_main_freqs

def refining_main_freqs(rough_freqs: np.ndarray,
                        main_freq_solvers: list,
                        refined_freqs_num: int,
                        refining_mode: str='median',
                        merge_value: float=1e-1,
                        solvers_combine=True):
    """
    Function for refining main frequencies
    
    Args:
        rough_freqs (np.ndarray): Raw main frequencies with shape: (sampling_num, num_features, candidate_freqs_num)
        main_freq_solvers (list): List of main frequency solvers
        refined_freqs_num (int): Number of main frequencies to retain after refinement
        refining_mode (str, optional): Aggregation mode for frequency refinement. Defaults to 'median'.
        merge_value (float, optional): Reference threshold in frequency aggregation. Defaults to 1e-1.
        solvers_combine (bool, optional): Whether to combine results from all solvers before analysis
    
    Returns:
        np.ndarray: Refined main frequencies with shape: (num_features, refined_freqs_num) / (num_features, refined_freqs_num * solvers_num)
    """
    s,c,rf = rough_freqs.shape
    if not solvers_combine:
        solvers_num = len(main_freq_solvers)
        freq_group = np.split(rough_freqs, solvers_num, axis=-1)
        total_freqs_list = []
        for freqs in freq_group:
            single_solver_freqs = np.transpose(freqs,(1,0,2))
            single_solver_freqs = single_solver_freqs.reshape(c,int(s*rf/solvers_num))
            s_s_freqs_collector = np.zeros((c,refined_freqs_num))
            for ch in range(c):
                singal_ch_rough_freqs = single_solver_freqs[ch,:]
                merged_freqs, freq_counts = merge_freqs_1d(singal_ch_rough_freqs,
                                                            es=1e-1,
                                                            mode=refining_mode)
                sc_refined_freqs = refine_freqs_1d(refined_num=refined_freqs_num,
                                                merged_freqs=merged_freqs,
                                                freq_counts=freq_counts)
                s_s_freqs_collector[ch,:]=sc_refined_freqs[:]
            total_freqs_list.append(s_s_freqs_collector)
        output_freqs=np.concatenate(total_freqs_list, axis=-1)
    else:
        trans_freqs = np.transpose(rough_freqs,(1,0,2))
        trans_freqs=trans_freqs.reshape(c, s*rf)
        output_freqs = np.zeros((c,refined_freqs_num))
        for ch in range(c):
            singal_ch_rough_freqs = trans_freqs[ch,:]
            merged_freqs, freq_counts = merge_freqs_1d(singal_ch_rough_freqs,
                                                            es=merge_value,
                                                            mode=refining_mode)
            sc_refined_freqs = refine_freqs_1d(refined_num=refined_freqs_num,
                                                merged_freqs=merged_freqs,
                                                freq_counts=freq_counts)
            output_freqs[ch,:]=sc_refined_freqs[:]
    return output_freqs