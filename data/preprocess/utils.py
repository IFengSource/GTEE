"""
Description: Basic components used in the Prior Mining Module.

Copyright (c) Yuan Feng et al., 2026
License: MIT License
"""

import numpy as np
import scipy.signal
from scipy.signal import find_peaks

from diy_benchpots.utils.logging import logger

def xcorr(data):
    length=len(data)
    R=[]
    for m in range(0,length):
        sum = 0.0
        for n in range(0,length-m):
            sum=sum+data[n]*data[n+m]
        R.append(sum/length)
    return R

def FFT_main_freqs_solver(data: np.ndarray,
                            selected_freqs_num: int,
                            fs: float=1.0):
    """
    Function for main frequency analysis using FFT
    
    Args:
        data (np.ndarray): Input signal sequence
        selected_freqs_num (int): Number of main frequencies to extract
        fs (float): Sampling frequency (Hz), default is 1 Hz
    
    Returns:
        np.ndarray: Array of main frequencies (Hz)
        np.ndarray: Frequency axis (Hz)
        np.ndarray: Normalized power spectral density
    """
    N = len(data)
    
    Xk = np.fft.fft(data)
    
    freqs = np.fft.fftfreq(N, d=1/fs)
    
    Sx = np.abs(Xk[:N//2])**2 / N
    freqs = freqs[:N//2]
    
    peak_indices = np.argsort(-Sx)[:selected_freqs_num]  
    dominant_freqs = freqs[peak_indices]
    
    return dominant_freqs, freqs, Sx

def ACF_main_freqs_solver(data: np.ndarray,
                            selected_freqs_num: int,
                            fs: float=1.0):
    """
    Perform main frequency analysis using the Autocorrelation Function (ACF)
    
    Args:
        data (np.ndarray): Input signal
        selected_freqs_num (int): Number of main frequencies to extract
        fs (float): Sampling frequency (Hz), default is 1 Hz
    
    Returns:
        np.ndarray: Array of main frequencies (Hz)
        np.ndarray: Frequency axis (Hz)
        np.ndarray: Normalized autocorrelation function
    """
    
    autocorr = np.correlate(data, data, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    
    autocorr /= np.max(np.abs(autocorr))
    
    peaks, _ = find_peaks(autocorr)
    
    valid_peaks = peaks[autocorr[peaks] > 0]
    peak_values = autocorr[valid_peaks]
    if len(valid_peaks) < selected_freqs_num:
        logger.warning(
                "⚠️  Main frequency: insufficient correlation peaks from autocorrelation; signal may be random noise or non-periodic."
            )
        dominant_cycles = np.argsort(autocorr)[-selected_freqs_num:][::-1]
        
    else:
        selected_indices = np.argsort(peak_values)[-selected_freqs_num:][::-1]
        dominant_cycles = valid_peaks[selected_indices]
    
    if np.any(dominant_cycles == 0):
        logger.warning("⚠️  Main frequency calculation detected zero period; results may be unreliable.")
    dominant_freqs = np.zeros_like(dominant_cycles, dtype=np.float64)
    non_zero_mask = dominant_cycles > 0
    dominant_freqs[non_zero_mask] = fs / dominant_cycles[non_zero_mask]
    dominant_freqs[~non_zero_mask] = fs / len(autocorr)
        
    return dominant_freqs, fs/np.arange(1, len(autocorr)), autocorr[1:]

def Burg_freqz_main_freqs_solver(data: np.ndarray,
                                    p: int,
                                    selected_freqs_num: int,
                                    fs: float=1.0):
    """
    Function for main frequency analysis using the Burg + freqz method
    
    Args:
        data (np.ndarray): Input signal
        p (int): Order of the autoregressive model
        selected_freqs_num (int): Number of main frequencies to extract
        fs (float): Sampling frequency (Hz), default is 1 Hz
    
    Returns:
        np.ndarray: Array of main frequencies (Hz)
        np.ndarray: Frequency axis (Hz)
        np.ndarray: Power spectrum
    """
    N = len(data)
    R=xcorr(data)
    
    rou=np.zeros(p+1)
    rou[0]=R[0]
    k=np.zeros(p+1)
    a=np.zeros([p+1,p+1])
    ef=np.zeros([p+1,N])
    eb=np.zeros([p+1,N])
    
    ef[0] = data.copy()
    eb[0] = data.copy()
    
    for m in range(1,p+1):
        up = np.sum(ef[m-1, m:N] * eb[m-1, m-1:N-1])
        down = np.sum(ef[m-1, m:N]**2 + eb[m-1, m-1:N-1]**2)
        
        k[m]=-2*up/down if down != 0 else 0
        
        if abs(k[m])>1:
            break
        
        a[m][m]=k[m]
        ef[m, m:N] = ef[m-1, m:N] + k[m] * eb[m-1, m-1:N-1]
        eb[m, m:N] = eb[m-1, m-1:N-1] + k[m] * ef[m-1, m:N]
        
        if m>1:
            a[m, 1:m] = a[m-1, 1:m] + k[m] * np.flip(a[m-1, 1:m])
        rou[m]=rou[m-1]*(1-(k[m]**2))
    
    ap = np.hstack(([1], a[p, 1:p+1]))
    G = np.sqrt(rou[p])
    
    w,h=scipy.signal.freqz(G,ap,worN=N,fs=fs)
    Sx = np.abs(h)**2
    
    peaks, _ = scipy.signal.find_peaks(Sx)
    if len(peaks) < selected_freqs_num:
        peak_indices = np.argsort(-Sx)[:selected_freqs_num]
        dominant_freqs =w[peak_indices]
        logger.warning(
                "⚠️  Main frequency: insufficient power peaks from Burg_freq; signal may be random noise or non-periodic."
            )
    else:
        peaks, _ = scipy.signal.find_peaks(Sx)
        peak_indices = np.argsort(-Sx[peaks])[:selected_freqs_num]
        dominant_freqs = w[peaks][peak_indices]
    return dominant_freqs, w, Sx

def Burg_roots_main_freqs_solver(data: np.ndarray,
                                    p: int,
                                    selected_freqs_num: int,
                                    fs: float=1.0):
    """
    Main frequency analysis algorithm using the Burg + root method
    
    Args:
        data (np.ndarray): Input signal
        p (int): Order of the autoregressive model
        selected_freqs_num (int): Number of main frequencies to extract
        fs (float): Sampling frequency (Hz), default is 1 Hz
    
    Returns:
        np.ndarray: Array of main frequencies (Hz)
        np.ndarray: Frequencies corresponding to roots (Hz), length may be variable
        np.ndarray: Magnitudes of the roots
    """
    N = len(data)
    R=xcorr(data)
    
    rou=np.zeros(p+1)
    rou[0]=R[0]
    k=np.zeros(p+1)
    a=np.zeros([p+1,p+1])
    ef=np.zeros([p+1,N])
    eb=np.zeros([p+1,N])
    
    ef[0] = data.copy()
    eb[0] = data.copy()
    
    for m in range(1,p+1):
        up = np.sum(ef[m-1, m:N] * eb[m-1, m-1:N-1])
        down = np.sum(ef[m-1, m:N]**2 + eb[m-1, m-1:N-1]**2)
        
        k[m]=-2 * up / (down + np.finfo(float).eps)
        
        if abs(k[m])>1:
            break
        
        a[m][m]=k[m]
        ef[m, m:N] = ef[m-1, m:N] + k[m] * eb[m-1, m-1:N-1]
        eb[m, m:N] = eb[m-1, m-1:N-1] + k[m] * ef[m-1, m:N]
        
        if m>1:
            a[m, 1:m] = a[m-1, 1:m] + k[m] * np.flip(a[m-1, 1:m])
        rou[m]=rou[m-1]*(1-(k[m]**2))
    
    ap = np.hstack(([1], a[p, 1:p+1]))
    
    poles = np.roots(ap)
    poles_inside = poles[np.abs(poles) < 1]
    angles = np.angle(poles_inside)
    freqs = np.abs(angles) / (2 * np.pi) * fs
    
    if len(freqs) < selected_freqs_num:
        logger.warning(
            f"⚠️ Insufficient poles computed by Burg: {len(freqs)} < {selected_freqs_num}, supplementing with hybrid strategy"
        )
        
        temp_dominant_freqs_list=[freqs] if len(freqs)>0 else []
        miss_freqs_num = selected_freqs_num-len(freqs)
        w,h=scipy.signal.freqz(np.sqrt(rou[p]),ap,worN=N,fs=fs)
        Sx = np.abs(h)**2
        peak_indices = np.argsort(-Sx)[:miss_freqs_num]
        temp_dominant_freqs_list.append(w[peak_indices])
        dominant_freqs = np.concatenate(temp_dominant_freqs_list)
    else:
        sorted_indices = np.argsort(np.abs(np.abs(poles_inside) - 1))[:selected_freqs_num]
        dominant_freqs = freqs[sorted_indices]
    
    return dominant_freqs, freqs, np.abs(poles_inside)

def merge_freqs_1d(freqs: np.ndarray,
                es: float=1e-1,
                mode: str='median'):
    """
    Calculate distinct frequencies and their occurrence counts from a 1D frequency array
    
    Args:
        freqs (np.array): Raw 1D frequency array
        es (float, optional): Equivalent period difference for frequency merging; frequencies that are too close will be merged. Defaults to 1e-1.
        mode (str, optional): Frequency merging mode, supports 'median' and 'mean'. Defaults to 'median'.
    
    Returns:
        np.ndarray: Merged frequencies
        np.ndarray: Occurrence counts corresponding to each frequency
    """
    sorted_freqs = sorted(freqs)
    merged_freqs = []
    counts = []
    temp_group = [freqs[0]]
    
    for freq in sorted_freqs[1:]:
        if abs(1/(freq+np.finfo(float).eps) - 1/(temp_group[-1]+np.finfo(float).eps)) < es:
            temp_group.append(freq)
        else:
            if mode == 'median':
                combined_value = np.median(temp_group)
            elif mode == 'mean':
                combined_value = np.mean(temp_group)
            else:
                logger.info("the freq refine mode is not supported yet.")
                raise NotImplementedError
            
            merged_freqs.append(combined_value)
            counts.append(len(temp_group))
            temp_group = [freq]
    
    if temp_group:
        if mode == 'median':
            combined_value = np.median(temp_group)
        elif mode == 'mean':
            combined_value = np.mean(temp_group)
        merged_freqs.append(combined_value)
        counts.append(len(temp_group))
    
    return np.array(merged_freqs), np.array(counts)

def refine_freqs_1d(refined_num: np.ndarray,
                        merged_freqs: np.ndarray,
                        freq_counts: np.ndarray,
                        ):
    """
    Frequency Refinement: Select and expand the most likely main frequencies based on weights.
    
    Args:
        refined_num (int): Total number of frequencies to be selected
        merged_freqs (np.array): Distinct frequencies (float format)
        freq_counts (np.array): Weights for each frequency (integer format)
    
    Returns:
        np.array: Refined frequencies with a length of refined_num
    """
    sorted_indices = np.argsort(-freq_counts)
    sorted_freqs = merged_freqs[sorted_indices]
    if refined_num <= len(sorted_freqs):
        refined_freqs = sorted_freqs[:refined_num]
    else:
        logger.info(f"The number of merged_freqs {len(merged_freqs)} is less then the refined freq num{refined_num}, using repeat.")
        repeat_times = refined_num // len(sorted_freqs)
        remainder = refined_num % len(sorted_freqs)
        refined_freqs = np.tile(sorted_freqs, repeat_times)
        refined_freqs = np.concatenate([refined_freqs, sorted_freqs[:remainder]])
    return refined_freqs