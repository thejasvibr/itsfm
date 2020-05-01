# -*- coding: utf-8 -*-
"""
Even though the spectrogram is one of the most dominant time-frequency 
representation, there are whole class of alternate representations. This
module has the code which tracks the dominant frequency in a sound using 
non-spectrogram methods. 

The Pseudo Wigner Ville Distribution
....................................
The Pseudo Wigner Ville Distribution is an accurate but not so well known 
method to represent a signal on the time-frequency axis[1]. This time-frequency
representation is implemented in the `get_pwvd_frequency_profile`. 


References
    

[1] Cohen, L. (1995). Time-frequency analysis (Vol. 778). Prentice hall.

"""
import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal 
import skimage.filters as filters
from tftb.processing import PseudoWignerVilleDistribution
import itsfm.signal_cleaning 
from itsfm.signal_cleaning import suppress_background_noise, remove_bursts, smooth_over_potholes
from itsfm.signal_cleaning import exterpolate_over_anomalies
from itsfm.signal_cleaning import clip_tfr, smooth_over_potholes
from itsfm.signal_processing import moving_rms_edge_robust, dB

def get_pwvd_frequency_profile(input_signal, fs, **kwargs):
    '''Generates a clean frequency profile through the PWVD. 
    The order of frequency profile processing is as follows:

    #. Split input signal into regions that are 
       greater or equal to the `signal_level`. This
       speeds up the whole process of pwvd tracking
       multiple sounds, and ignores the  fainter samples. 

    #. Generate PWVD for each above-noise region.
    
    #. Set regions below background noise to 0Hz
    
    #. Remove sudden spikes and set these regions to values
       decided by interpolation between adjacent non-spike regions. 
      
    Parameters
    ----------
    input_signal : np.array
    fs  : float

    
    Notes
    -----
    The fact that each signal part is split into independent 
    above-background segments and then frequency tracked can 
    have implications for frequency resolution. Short sounds
    may end up with frequency profiles that have a lower
    resolution than longer sounds. Each sound is handled separately
    primarily for memory and speed considerations.


    Example
    -------
    
    Create two chirps in the middle of a somewhat silent recording
    
    >>> import matplotlib.pyplot as plt
    >>> from itsfm.simulate_calls import make_fm_chirp
    >>> from itsfm.view_horseshoebat_call import plot_movingdbrms
    >>> from itsfm.view_horseshoebat_call import visualise_call, make_x_time
    >>> fs = 44100
    >>> start_f, end_f = 1000, 10000
    >>> chirp = make_fm_chirp(start_f, end_f, 0.01, fs)  
    >>> rec = np.random.normal(0,10**(-50/20), 22100)
    >>> chirp1_start, chirp1_end = 10000, 10000 + chirp.size
    >>> chirp2_start, chirp2_end = np.array([chirp1_start, chirp1_end])+int(fs*0.05)
    >>> rec[chirp_start:chirp_end] += chirp
    >>> rec[chirp2_start:chirp2_end] += chirp
    >>> rec /= np.max(abs(rec))
    >>> actual_fp = np.zeros(rec.size)
    >>> actual_fp[chirp1_start:chirp1_end] = np.linspace(start_f, end_f, chirp.size)
    >>> actual_fp[chirp2_start:chirp2_end] = np.linspace(start_f, end_f, chirp.size)
    
    Check out the dB rms profile of the recording to figure out where the
    noise floor is 
    
    >>> plot_movingdbrms(rec, fs)
    
    >>> clean_fp, info = get_pwvd_frequency_profile(rec, fs,
                                                         signal_level=-9,
                                                         extrap_window=10**-3,
                                                         max_acc = 0.6)
    >>> plt.plot(clean_fp, label='obtained')
    >>> plt.plot(actual_fp, label='actual')
    >>> plt.legend()

    Now, let's overlay the obtained frequency profile onto a spectrogram to 
    check once more how well the dominant frequency has been tracked. 

    >>> w,s = visualise_call(rec, fs, fft_size=128)
    >>> s.plot(make_x_time(clean_fp, fs), clean_fp)

    See Also
    --------
    itsfm.signal_cleaning.smooth_over_potholes
    find_above_noise_regions
    '''
    info = {}
    above_noise_regions, moving_dbrms = find_geq_signallevel(input_signal, fs, **kwargs)

    full_fp = np.zeros(input_signal.size)
    full_raw_fp = np.zeros(input_signal.size)
    acc_profile = np.zeros(input_signal.size)
    spikey_regions = np.zeros(input_signal.size)
    #print('generating PWVD frequency profile....')
    for region in above_noise_regions:    
        raw_fp, frequency_index = generate_pwvd_frequency_profile(input_signal[region],
                                                                  fs, **kwargs)
        weird_parts, accelaration_profile = frequency_spike_detection(raw_fp, fs, **kwargs)
        cleaned_fp = exterpolate_over_anomalies(raw_fp, fs, weird_parts, **kwargs)
        full_raw_fp[region] = raw_fp
        cleaned_fp = exterpolate_over_anomalies(raw_fp, fs, weird_parts,
                                                **kwargs)
        acc_profile[region] = accelaration_profile

        full_fp[region] = cleaned_fp
        spikey_regions[region[0]][weird_parts] = 1

    info['moving_dbrms'] = moving_dbrms
    info['geq_signal_level'] = above_noise_regions
    info['raw_fp'] = full_raw_fp
    info['acc_profile'] = acc_profile
    info['spikey_regions'] = spikey_regions

    return full_fp, info

def find_geq_signallevel(X, fs, **kwargs):
    '''
    Find regions greater or equal to signal level
    '''
    signal_level = kwargs.get('signal_level', -20)
    rec_level = dB(moving_rms_edge_robust(X, **kwargs))
    
    ids_above_noise, num_regions = ndimage.label(rec_level>signal_level)
    if num_regions <1:
        raise ValueError('No regions above signal level found!')

    return ndimage.find_objects(ids_above_noise), rec_level
    
    


def clean_up_spikes(whole_freqeuncy_profile, fs, **kwargs):
    '''Applies smooth_over_potholes on each non-zero frequency segment
    in the profile. 

    Parameters
    ----------
    
    Returns 
    -------
    
    See Also
    --------
    smooth_over_potholes
    
    Example
    -------
    Let's create a case with an FM and CF tone
  >>> from itsfm.simulate_calls import make_tone, make_fm_chirp, silence
    >>> fs = 22100
    >>> tone = make_tone(5000, 0.01, fs)
    >>> sweep = make_fm_chirp(1000, 6000, 0.005, fs)
    >>> gap = silence(0.005, fs)
    >>> full_call = np.concatenate((tone, gap, sweep))
    
    The raw frequency profile, with very noisy frequency estimates needs
    to be further cleaned 
   
    >>> raw_fp, frequency_index = generate_pwvd_frequency_profile(full_call,
                                                                    fs)
    >>> noise_supp_fp = noise_supp_fp = suppress_background_noise(raw_fp,
                                              full_call, 
                                              window_size=25,
                                              background_noise=-30)
    
    Even after the noisy parts have been suppressed, there're still some 
    spikes caused by the 
    
    >>> 
    
    '''
    
    nonzero_freqs, num_regions = ndimage.label(whole_freqeuncy_profile>0)
    segment_locations = ndimage.find_objects(nonzero_freqs)
    
    if len(segments) <1 : 
        raise ValueError('No non-zero frequency sounds found..!')
    
    de_spiked = np.zeros(whole_freqeuncy_profile.size)
    
    for segment in segment_locations:
        smoothed, _ = smooth_over_potholes(whole_freqeuncy_profile[segment],
                                           fs, **kwargs)
        de_spiked[segment] = smoothed
    return de_spiked

def generate_pwvd_frequency_profile(input_signal, fs, **kwargs):
    '''Generates the raw instantaneous frequency estimate at each sample. 
    using the Pseudo Wigner Ville Distribution

    Parameters
    ----------
    input_signal : np.array
    fs : float
    pwvd_filter : Boolean, optional
        Whether to perform median filtering with a 2D kernel. 
        Defaults to False
    pwvd_filter_size : int, optional
        The size of the square 2D kernel used to median filter the 
        initial PWVD time-frequency representation. 
    pwvd_window : float>0, optional 
        The duration of the window used in the PWVD. See pwvd_transform
        for the default value.
    pwvd_zero_pad : int, optional
        Number of samples to zero-pad on left and right of the 
        input_signal. The zero-padding prevents 0's and spikes
        at the start and end of the signal. Defaults to
        the equivalent samples for 1ms. 
    tfr_cliprange: float >0, optional
        The clip range in dB.
        Clips all values in the abs(pwvd) time-frequency
        representation to between max and max*10*(-tfr_cliprange/20.0).
        Defaults to None, which does not alter the pwvd transform in anyway. 

    Returns
    -------
    raw_frequency_profile, frequency_indx : np.array
        Both outputs are the same size as input_signal. 
        raw_frequency_profile is the inst. frequency in Hz. 
        frequency_indx is the row index of the PWVD array. 
        
    
    
    See Also 
    --------
    pwvd_transform
    track_peak_frequency_over_time
    itsfm.signal_cleaning.clip_tfr

    '''
    pwvd_filter = kwargs.get('pwvd_filter', False)
    pwvd_filter_size = kwargs.get('pwvd_filter_size', 10)
    filter_dims = (pwvd_filter_size, pwvd_filter_size)


    time_freq_rep = np.abs(pwvd_transform(input_signal, fs, 
                                             **kwargs))
    clipped_tfr = clip_tfr(time_freq_rep, **kwargs)
    
    if pwvd_filter:
        print('....A 2D median filter kernel is being applied to the PWVD...')
        median_filtered_tf = filters.median_filter(clipped_tfr, size=filter_dims)
        print('..done with PWVD filtering..')
        raw_frequency_profile, frequency_indx = track_peak_frequency_over_time(input_signal, fs,
                                                                           median_filtered_tf,
                                                                          **kwargs)
    else:
        raw_frequency_profile, frequency_indx = track_peak_frequency_over_time(input_signal, fs,
                                                                           clipped_tfr,
                                                                          **kwargs)       
    return raw_frequency_profile, frequency_indx
    
    
    


def pwvd_transform(input_signal, fs, **kwargs):
    '''Converts the input signal into an analytical signal and then generates
    the PWVD of the analytical signal. 

    Uses the PseudoWignerVilleDistribution class from the tftb package [1]. 

    Parameters
    ----------
    input_signal : np.array
    fs : float
    
    window : np.array, optional
        The window to be used for the pseudo wigner-ville distribution.
        If not given, then a hanning signal is used of the default length.
        The window given here supercedes the 'window_length' argument below.

    window_length : float>0, optional 
        The duration of the window used in the PWVD. Defaults to 0.001s

    Returns
    -------
    time_frequency_output : np.array
        Two dimensional array with dimensions of NsamplesxNsamples, where
        Nsamples is the number of samples in input_signal. 

    References
    ----------
    [1] Jaidev Deshpande, tftb 0.1.1 ,Python module for time-frequency analysis, 
        https://pypi.org/project/tftb/
    '''
    window_length = kwargs.get('window_length', 0.001)
    window = kwargs.get('window', signal.hanning(int(fs*window_length)))
    analytical = signal.hilbert(input_signal)
    p = PseudoWignerVilleDistribution(analytical, fwindow=window)
    pwvd_output = p.run();
    time_frequency_output = pwvd_output[0]
    return time_frequency_output


def track_peak_frequency_over_time(input_signal, fs, time_freq_rep, **kwargs):
    '''Tracks the lowest possible peak frequency. This ensures that the 
    lowest harmonic is being tracked in a multiharmonic signal with similar
    levels across the harmonics. 

    EAch 'column' of the 2D PWVD is inspected for the lowest peak that crosses
    a percentile threshold, and this is then taken as the peak frequency. 

    Parameters
    ----------
    input_signal : np.array
    fs : float>0
    time_freq_rep : np.array
        2D array with the PWVD representation. 
    percentile : 0<float<100, optional 

    Returns
    -------
    peak_freqs, peak_inds : np.array
        Arrays with same size as the input_signal. peak_freqs is the 
        frequencies in Hz, peak_inds is the row index. 
    
    See Also
    --------
    find_lowest_intense_harmonic_across_TFR
    get_most_intense_harmonic
    '''
    peak_inds = find_lowest_intense_harmonic_across_TFR(abs(time_freq_rep), **kwargs)
    freqs = np.linspace(0, fs*0.5, input_signal.size)
    peak_freqs = freqs[peak_inds]
    return peak_freqs, peak_inds


def find_lowest_intense_harmonic_across_TFR(tf_representation, **kwargs):
    '''
    '''
    return np.apply_along_axis(get_most_intense_harmonic,0,tf_representation, **kwargs)


def get_most_intense_harmonic(time_slice, **kwargs):
    '''Searches a single column in a 2D array for the first region which
    crosses the given percentile threshold. 
    '''
    one_region_above_threshold = get_first_region_above_threshold(time_slice, **kwargs)
    loudest_harmonic = get_midpoint_of_a_region(one_region_above_threshold)
    return loudest_harmonic


def get_midpoint_of_a_region(region_object):
    '''
    '''
    if region_object is None:
        return 0
    
    mid_point = int(np.mean([region_object[0].stop,region_object[0].start]))
    return mid_point

def accelaration(X, fs):
    '''Calculates the absolute accelrateion of a frequency profile in kHz/ms^2
    '''
    speed_X = speed(X,fs)
    return np.abs(np.gradient(speed_X))

def speed(X,fs):
    '''Calculates the abs speed of the frequency profile in kHz/ms
    '''
    speed = 10**-6*np.abs(np.gradient(X))/(1.0/fs)
    return speed
    

def get_first_region_above_threshold(input_signal,**kwargs):
    '''Takes in a 1D signal expecting a few peaks in it above the percentil threshold. 
    If all samples are of the same value, the region is restricted to the first two samples. 

    Parameters
    ----------
    input_signal :np.array
    percentile : 0<float<100, optional 
        The percentile threshold used to set the threshold. 
        Defaults to 99.5
    
    Returns
    -------
    region_location : tuple or None
        If there is at least one region above the threshold a tuple with
        the output from scipy.ndimage.find_objects. Otherwise None. 


    '''
    percentile = kwargs.get('percentile', 99.5)
    above_threshold  = input_signal > np.percentile(input_signal, percentile)
    regions, num_regions = ndimage.label(above_threshold)
    
    if num_regions>=1:
        region_location = ndimage.find_objects(regions)[0]
        return region_location
    else:
        return None


def frequency_spike_detection(X, fs, **kwargs):
    '''Detects spikes in the frequency profile by 
    monitoring the accelration profile through the sound. 
    
    Parameters
    ----------
    X : np.array
        A frequency profile with sample-level estimates of frequency in Hz
    fs : float>0
    max_acc : float>0, optional
        Maximum acceleration in the frequency profile. 
        Defaults to 0.5kHz/ms^2
    
    Returns
    --------
    anomalous : np.array
        Boolean 
    '''
    max_acc = kwargs.get('max_acc', 1.0) # kHz/ms^2
    freq_accelaration = accelaration(X,fs)
    anomalous = freq_accelaration>max_acc
    return anomalous, freq_accelaration

