"""
Description: Implementation of the Prior Mining Module.

Copyright (c) Yuan Feng et al., 2026
License: MIT License
"""

import numpy as np
from data.preprocess.functional import random_sampling_from_dataset, get_main_freqs_with_multiple_analysis, refining_main_freqs

def get_timestamp_coding_parameters(dataset: np.ndarray,
                                    main_freq_num: int,
                                    main_freq_solvers: list=['ACF', 'Burg_freqz', 'Burg_roots', 'FFT'],
                                    solvers_combine: bool=False,
                                    sampling_num: int=1,
                                    sampling_length: int=1024,
                                    scaling_factor: float=0.5,
                                    freqs_merge_value: float=1e-1,
                                    refining_mode: str='median'):
    samples = random_sampling_from_dataset(dataset=dataset,
                                            sampling_num=sampling_num,
                                            sampling_length=sampling_length)
    estimated_main_freqs = get_main_freqs_with_multiple_analysis(samples=samples,
                                                                    main_freq_num=main_freq_num,
                                                                    main_freq_solvers=main_freq_solvers,
                                                                    scaling_factor=scaling_factor)
    final_main_freqs = refining_main_freqs(rough_freqs=estimated_main_freqs,
                                            main_freq_solvers=main_freq_solvers,
                                            refined_freqs_num=main_freq_num,
                                            refining_mode=refining_mode,
                                            merge_value=freqs_merge_value,
                                            solvers_combine=solvers_combine)
    
    return final_main_freqs
