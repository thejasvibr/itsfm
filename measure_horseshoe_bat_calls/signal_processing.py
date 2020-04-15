# -*- coding: utf-8 -*-
"""Module with signal processing functions in it 
used by both measure and segment modules.


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
    
def get_peak_frequency(audio, fs):
    '''Gives peak frequency and frequency resolution
    with which the measurement is made

    Parameters
    ----------
    audio : np.array
    fs : float>0
        sampling rate in Hz

    Returns
    -------
    peak_freq, freq_resolution : float
        The peak frequency and frequency resolution of this
        peak frequency in Hz.
    '''
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(audio.size, 1.0/fs)
    freq_resolution = get_frequency_resolution(freqs, fs)
    peak_freq = freqs[np.argmax(spectrum)]
    return peak_freq, freq_resolution

def get_frequency_resolution(audio, fs):
    '''
    Parameters
    ----------
    audio : np.array
    fs : float>0
        sampling rate in Hz

    Returns
    -------
    resolution : float
        The frequency resolution in Hz. 
    '''
    resolution = float(fs/audio.size)
    return resolution


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


def median_filter(input_signal, fs, **kwargs):
    '''Median filters a signal according to a user-settable
    window size. 

    Parameters
    ----------
    input_signal : np.array
    fs : float
        Sampling rate in Hz.
    medianfilter_size : float, optional
        The window size in seconds. Defaults to 0.001 seconds. 

    Returns
    -------
    med_filtered : np.array
        Median filtered version of the input_signal. 
    '''
    window_duration = kwargs.get('medianfilter_size',
                              0.001)
    kernel_size = calc_proper_kernel_size(window_duration, fs)
    med_filtered = signal.medfilt(input_signal, kernel_size)
    return med_filtered


def calc_proper_kernel_size(durn, fs):
    '''scipy.signal.medfilt requires an odd number of samples as
    kernel_size. This function calculates the number of samples
    for a given duration which is odd and is close to the 
    required duration. 
    
    Parameters
    ----------
    durn : float
        Duration in seconds. 
    fs : float
        Sampling rate in Hz
    
    Returns
    -------
    samples : int
        Number of odd samples that is equal to or little 
        less (by one sample) than the input duration. 
    '''
    samples = int(durn*fs)
    if np.remainder(samples,2)==0:
        samples -= 1
    return samples

def resize_by_adding_one_sample(input_signal, original_signal, **kwargs):
    '''Resizes the input_signal to the same size as the original signal by repeating one
    sample value. The sample value can either the last or the first sample of the input_signal. 
    '''
    check_signal_sizes(input_signal, original_signal)
    
    repeat_start = kwargs.get('repeat_start', True)
    
    if repeat_start:
        return np.concatenate((np.array([input_signal[0]]), input_signal))
    else:
        return np.concatenate((input_signal, np.array([input_signal[-1]])))

def check_signal_sizes(input_signal, original_signal):
    if int(input_signal.size) >= int(original_signal.size):
        msg1 = 'The input signal"s size %d'%int(input_signal.size)
        msg2 = ' is greater or equal to the original signal"s size: %d'%(int(original_signal.size))
        raise ValueError(msg1+msg2)
    
    if int(original_signal.size) - int(input_signal.size) >= 2:
        raise ValueError('The original signal is >= 2 samples longer than the input signal.')



