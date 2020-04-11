# -*- coding: utf-8 -*-
"""The Pseudo Wigner Ville Distribution is an accurate but not so well known 
method to represent a signal on the time-frequency axis[1]. This module tracks the
instantaneous frequency over the signal's duration, calcualtes the rate of
frequency modulation and then segments it into CF and FM.



References
----------
[1] Cohen, L. (1995). Time-frequency analysis (Vol. 778). Prentice hall.

"""
import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal 
import skimage.filters as filters
from tftb.processing import PseudoWignerVilleDistribution
from measure_horseshoe_bat_calls.signal_processing import suppress_background_noise
from measure_horseshoe_bat_calls.signal_processing import remove_bursts

def get_frequency_profile_through_pwvd(input_signal, fs, **kwargs):
    '''Generate the sample-resolution frequency profile of the input signal and 
    also perform some cleaning. 

    Parameters
    ----------
    input_signal : np.array
    fs : float
        Sampling rate in Hz. 
    
    Returns
    -------
    raw_freq_profile, noise_suppressed_freq_profile, cleaned_frequency_profile : np.array
    The instantaneous frequency over the input signal. All output arrays are the same size as the input_signal:

        * raw_freq_profile : the raw frequency estimate across samples

        * noise_suppressed_freq_profile : regions of the input signal below the threshold dB rms are set to zero frequency.

        * cleaned_frequency_profile : the noise_suppressed frequency profile is further
            checked for any abrupt frequency transitions and corrected/filtered.

    Notes
    -----
    This is a higher-level function which can be modulated to achieve better
    frequency tracking. Check the optional arguments in See Also. 
    
    See Also
    --------
    generate_pwvd_frequency_profile
    suppress_background_noise
    remove_bursts
    '''    

    raw_freq_profile, frequency_index = generate_pwvd_frequency_profile(input_signal, fs, **kwargs)
    # get rid of the silent parts of the audio
    noise_suppressed_freq_profile = suppress_background_noise(raw_freq_profile,input_signal, **kwargs)
    # get rid of abrupt large frequency jumps
    #nonspikey_frequency_profile = suppress_frequency_spikes(noise_suppressed_freq_profile, input_signal, fs, **kwargs)
    # remove any small frequency spikes that still remain based on duration
    cleaned_frequency_profile = remove_bursts(noise_suppressed_freq_profile, fs, **kwargs)
    return raw_freq_profile, noise_suppressed_freq_profile, cleaned_frequency_profile



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
    Returns
    -------
    raw_frequency_profile, frequency_indx : np.array
        Both outputs are the same size as input_signal. 
        raw_frequency_profile is the inst. frequency in Hz. 
        frequency_indx is the row index of the PWVD array. 

    '''
    pwvd_filter = kwargs.get('pwvd_filter', False)
    pwvd_filter_size = kwargs.get('pwvd_filter_size', 10)
    filter_dims = (pwvd_filter_size, pwvd_filter_size)

    time_freq_course = np.abs(pwvd_transform(input_signal, fs, **kwargs))
    if pwvd_filter:
        print('....median filtering the PWVD...')
        median_filtered_tf = filters.median_filter(time_freq_course, size=filter_dims)
        print('..done with PWVD filtering..')
        raw_frequency_profile, frequency_indx = track_peak_frequency_over_time(input_signal, fs,
                                                                           median_filtered_tf,
                                                                          **kwargs)
    else:
        raw_frequency_profile, frequency_indx = track_peak_frequency_over_time(input_signal, fs,
                                                                           time_freq_course,
                                                                          **kwargs)       
    return raw_frequency_profile, frequency_indx



def pwvd_transform(input_signal, fs, **kwargs):
    '''Converts the input signal into an analytical signal and then generates
    the PWVD of the analytical signal. 

    Uses the PseudoWignerVilleDistribution class from the tftb package[1]. 
    
    
    

    Parameters
    ----------
    input_signal : np.array
    fs : float
    pwvd_window : float>0, optional 
        The duration of the window used in the PWVD. Defaults to 0.001s

    Returns
    -------
    time_frequency_output : np.array
        Two dimensional array with dimensions of NsamplesxNsamples, where
        Nsamples is the number of samples in input_signal. 

    References
    ----------
    [1] tftb 0.1.1 ,Python module for time-frequency analysis, Jaidev Deshpande, 
        https://pypi.org/project/tftb/
    '''
    pwvd_window = kwargs.get('pwvd_window', 0.001)
    fw = signal.hamming(int(fs*pwvd_window))
    analytical = signal.hilbert(input_signal)
    p = PseudoWignerVilleDistribution(analytical, fwindow=fw)
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


def get_first_region_above_threshold(input_signal,**kwargs):
    '''Takes in a 1D signal expecting a few peaks in it above the percentil threshold. 
    If all samples are of the same value, the region is restricted to the first two samples. 

    Parameters
    ----------
    input_signal :np.array
    percentile : 0<float<100, optional 
        The percentile threshold used to set the threshold. 
        Defaults to 98.5
    
    Returns
    -------
    region_location : tuple or None
        If there is at least one region above the threshold a tuple with
        the output from scipy.ndimage.find_objects. Otherwise None. 

    See Also
    --------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.find_objects.html
    '''
    percentile = kwargs.get('percentile', 98.5)
    above_threshold  = input_signal > np.percentile(input_signal, percentile)
    regions, num_regions = ndimage.label(above_threshold)
    
    if num_regions>=1:
        region_location = ndimage.find_objects(regions)[0]
        return region_location
    else:
        return None
