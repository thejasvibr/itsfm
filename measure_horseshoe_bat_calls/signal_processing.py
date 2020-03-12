# -*- coding: utf-8 -*-
"""Module with signal processing functions in it 
used by both measure and segment modules.
Created on Wed Mar 11 16:28:46 2020

@author: tbeleyur
"""
import numpy as np 
import scipy.signal as signal 

def dB(X):
    '''Calculates the 20log of X'''
    return 20*np.log10(X)

def rms(X):
    '''Root mean square of a signal '''
    return np.sqrt(np.mean(X**2.0))

def calc_energy(X):
    '''Sum of all squared samples '''
    return np.sum(X**2.0)

def get_power_spectrum(audio, fs=250000.0):
    '''Calculates an RFFT of the audio.
    Parameters
    ------------
    audio : np.array
    fs : int
        Frequency of sampling in Hz

    Returns
    -------
    dB_power_spectrum : np.array
        dB(power_spectrum)
    freqs : np.array
        Centre frequencies of the RFFT. 
    '''
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(audio.size, 1.0/fs)
    dB_power_spectrum = dB(abs(spectrum))
    return dB_power_spectrum, freqs

def calc_sound_borders(audio, percentile=99):
    '''Gives the start and stop of a sound based on the percentile 
    cumulative energy values. 

    Parameters
    ----------
    audio : np.array
    percentile : float, optional
        Value between 100 and 0. The sound border is calcualted
        as the border which encapsulates the percentile of energy
        Defaults to 99.

    Returns
    --------
    start, end : int
    '''
    audio_sq = audio**2.0
    cum_energy = np.cumsum(audio_sq)
    outside_percentile = (100-percentile)*0.5
    lower, higher = outside_percentile, 100-outside_percentile
    start, end = np.percentile(cum_energy,[lower, higher])
    start_ind = np.argmin(abs(cum_energy-start))
    end_ind = np.argmin(abs(cum_energy-end))
    return start_ind, end_ind

def get_robust_peak_frequency(audio, **kwargs):
    '''Makes a spectrogram from the audio 
    and calcualtes the peak frequency by averaging
    each slice of the spectrogram's FFT's. 

    This 'smooths' out the structure of the power 
    spectrum and allows a single and clear peak 
    detection. 

    Thanks to Holger Goerlitz for the suggestion. 
    
    Parameters
    ----------
    audio : np.array
    fs : float
        Frequency of sampling in Hz
    seg_length : int, optional
        The size of the FFt window used to calculate the moving FFT slices. 
        DEfaults to 256
    noverlap : int, optional 
        The number of samples overlapping between one FFT slice and the next. 
        Defaults to seg_length -1

    Returns
    --------
    peak_frequency : float
        Frequency with highest power in the audio in Hz. 
    '''
    seg_length = kwargs.get('seg_length',256)
    frequency,t,sxx = signal.spectrogram(audio, fs=int(kwargs['fs']), nperseg=seg_length, noverlap=seg_length-1)
    averaged_spectrogram = np.apply_along_axis(np.sum, 1, sxx)
    peak = np.argmax(averaged_spectrogram)
    peak_frequency = frequency[peak]
    return peak_frequency
    
def get_peak_frequency(audio, fs=250000):
    '''
    '''
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(audio.size, 1.0/fs)
    peak_freq = freqs[np.argmax(spectrum)]
    return peak_freq


def moving_rms(X, **kwargs):
    '''Calculates moving rms of a signal with given window size. 
    Outputs np.array of *same* size as X. The rms of the 
    last few samples <= window_size away from the end are assigned
    to last full-window rms calculated

    Parameters
    ----------
    X :  np.array
        Signal of interest. 

    window_size : int, optional
                 Defaults to 125 samples. 

    Returns
    -------
    all_rms : np.array
        Moving rms of the signal. 
    '''
    window_size = kwargs.get('window_size', 125)
    starts = np.arange(0, X.size)
    stops = starts+window_size
    valid = stops<X.size
    valid_starts = np.int32(starts[valid])
    valid_stops = np.int32(stops[valid])
    all_rms = np.ones(X.size).reshape(-1,1)*999

    for i, (start, stop) in enumerate(zip(valid_starts, valid_stops)):
        rms_value = rms(X[start:stop])
        all_rms[i] = rms_value
    
    # replace all un-assigned samples with the last rms value
    all_rms[all_rms==999] = np.nan

    return all_rms


def moving_rms_edge_robust(X, **kwargs):
    '''Calculates moving rms of a signal with given window size. 
    Outputs np.array of *same* size as X. This version is robust 
    and doesn't suffer from edge effects as it calculates the 
    moving rms in both forward and backward directions
    and calculates a consensus moving rms profile.
    
    The consensus rms profile is basically achieved by 
    taking the left half of the forward rms profile 
    and concatenating it with the right hald of the
    backward passed rms profile. 
    
    Parameters
    ----------
    X :  np.array
        Signal of interest. 

    window_size : int, optional
                 Defaults to 125 samples. 

    Returns
    -------
    all_rms : np.array
        Moving rms of the signal.

    Notes
    -----
    moving_rms_edge_robust may not be too accurate when the rms
    is expected to vary over short time scales in the centre of 
    the signal!! 
    '''

    forward_run = moving_rms(X, **kwargs)
    backward_run = np.flip(moving_rms(np.flip(X), **kwargs))
    consensus = form_consensus_moving_rms(forward_run, backward_run)
    return consensus


def form_consensus_moving_rms(forward, backward):
    '''
    '''
    half_samples = int(forward.size/2.0)
    
    consensus_rms = np.concatenate((forward[:half_samples], 
                                    backward[half_samples:]))

    return consensus_rms
