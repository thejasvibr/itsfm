# -*- coding: utf-8 -*-
"""Module that segments the horseshoebat call into FM and CF parts
The primary logic of this 

"""
import warnings
import numpy as np 
import pywt
import scipy.signal as signal 
from measure_horseshoe_bat_calls.signal_processing import *
from measure_horseshoe_bat_calls.sanity_checks import make_sure_its_positive


def segment_call_into_cf_fm(call, fs, **kwargs):
    '''
    Parameters
    -----------
    call : np.array
        Audio with horseshoe bat call
    fs : float>0
        Frequency of sampling in Hz. 
    
    Returns
    --------
    cf_samples, fm_samples : np.array
        Boolean numpy array showing which of the samples belong 
        to the cf and the fm respectively. 
    info : list
        List with two np.array. The first array has the 
        max normalised dB rms profile of the highpassed filtered
        call (cf dominant). The second array has the max normalised
        dB rms profile of the  lowpassed filtered call (fm dominant).
    
    Notes
    ------
    For more information on how to handle/improve the segmentation see
    documentation for pre_process_for_segmentation
    '''
    cf_dbrms, fm_dbrms = pre_process_for_segmentation(call, fs, **kwargs)
    cf_samples, fm_samples, info = segment_cf_and_fm(cf_dbrms, fm_dbrms, 
                                                     fs,**kwargs)
    return cf_samples, fm_samples, info

def segment_cf_and_fm(cf_dbrms, fm_dbrms, fs, **kwargs):
    '''Calculates the relative increase in signal levels 
    between the CF and FM dominant versions of the audio. 
    
    Regions which have not been amplified will show <= 0 dB change, 
    and this is used to identify the CF and FM portions reliably.
    '''
    fm_re_cf = fm_dbrms - cf_dbrms
    cf_re_fm = cf_dbrms - fm_dbrms
    
    fm_samples = fm_re_cf > 0 
    cf_samples = cf_re_fm > 0

    main_cf = identify_valid_regions(cf_samples, 1)
    main_fm = get_fm_regions(fm_samples, fs, **kwargs)

    return main_cf, main_fm, [cf_re_fm, fm_re_cf]

def get_fm_regions(fm_samples, fs, **kwargs):
    '''
    Parameters
    ----------
    fm_samples : np.array
        Boolean numpy array with candidate FM samples. 
    fs : float>0
    min_fm_duration : float, optional
        minimum fm duration expected in seconds. Any fm segment lower than this
        duration  is considered to be a bad read and discarded.
        Defaults to 0.5 milliseconds.
    Returns
    -------
    valid_fm : np.array
        Boolean numpy array with the corrected fm samples.
    
    '''
    min_fm_duration = kwargs.get('min_fm_duration', 0.5*10**-3)
    make_sure_its_positive(min_fm_duration, variable='min_fm_duration')
    min_fm_samples = int(fs*min_fm_duration)

    valid_fm = np.zeros(fm_samples.size, dtype='bool')
    try:
        main_fm = identify_valid_regions(fm_samples, 2)
        regions, region_id_and_samples = identify_maximum_contiguous_regions(main_fm, 2)
        regions, region_lengths = np.unique(region_id_and_samples[:,0],
                                            return_counts=True)
        regions_above_min_length = regions[region_lengths >= min_fm_samples]

        if len(regions_above_min_length) >0:
            valid_rows = []
            for each in regions_above_min_length:
                valid_rows.append(np.argwhere(region_id_and_samples[:,0]==each))
            valid_rows = np.concatenate(valid_rows).flatten()
            valid_samples = region_id_and_samples[valid_rows,1].flatten()
            
            valid_fm[valid_samples] = True

    except:
        candidate_fm = identify_valid_regions(fm_samples, 1)
        if np.sum(candidate_fm) >= min_fm_samples:
            valid_fm = candidate_fm.copy()
             
    return valid_fm
        

def segment_call_from_background(audio, fs,**kwargs):
    '''Performs a wavelet transform to track the signal within the relevant portion of the bandwidth. 
    
    This methods broadly works by summing up all the signal content 
    above the ```lowest_relevant_frequency``` using a continuous wavelet transform. 
    
    If the call-background segmentation doesn't work well it's probably due 
    to one of these things:
    
    #. Incorrect ``background_threshold`` : Play around with different ``background_threshold values``.
    
    #. Incorrect ``lowest_relevant_frequency`` : If the lowest relevant frequency is set outside of the signal's actual frequency range, then the segmentation will fail.
       Try lower this parameter till you're sure all of the signal's spectral range is above it.     
    
    #. Low signal spectral range : This method uses a continuous wavelet transform to localise the relevant signal. Wavelet transforms have high temporal resolution 
       in for high frequencies, but lower temporal resolutions for lower frequencies.
       If your signal is dominantly low-frequency, try resampling it to a lower 
       sampling rate and see if this works?

    If the above tricks don't work, then try bandpassing your signal - may be it's
    an issue with the in-band signal to noise ratio.
    
    Parameters
    ----------
    audio : np.array
    fs : float>0
        Frequency of sampling in Hertz.
    lowest_relevant_freq : float>0, optional
        The lowest frequency band in Hz whose coefficients will be tracked.
        The coefficients of all frequencies in the signal >= the lowest relevant
        frequency are tracked. This is the lowest possible frequency the signal can take. It is best to give a few kHz of berth.
        Defaults to 35kHz.
    background_threshold : float<0, optional
		The relative threshold which is used to define the background. The segmentation is 
		performed by selecting the region that is above background_threshold dB relative
		to 	the max dB rms value in the audio. 
		Defaults to -20 dB
    wavelet_type : str, optional
        The type of wavelet which will be used for the continuous wavelet transform. 
        Run  `pywt.wavelist(kind='continuous')` for all possible types in case the default
        doesn't seem to work.
        Defaults to mexican hat, 'mexh'
    scales : array-like, optional
        The scales to be used for the continuous wavelet transform. 
        Defaults to np.arange(1,10).

    Returns
    -------
    potential_region : np.array
        A boolean numpy array where True corresponds to the regions which
        are call samples, and False are the background samples. The single 
        longest continuous region is output.
    dbrms_profile : np.array
        The dB rms profile of the summed up wavelet transform for all 
        centre frequencies >= lowest_relevant_frequency.s

    Raises
    ------
    ValueError
        When lowest_relevant_frequency is too high or not included in 
        the centre frequencies of the default/input scales for 
        wavelet transforms. 
    IncorrectThreshold
        When the dynamic range of the relevant part of the signal is smaller
        or equal to the background_threshold.
   
    '''
    lowest_relevant_freq = kwargs.get('lowest_relevant_freq', 35000.0)
    make_sure_its_positive(lowest_relevant_freq, variable='lowest_relevant_freq')
    
    wavelet_type = kwargs.get('wavelet_type', 'mexh')
    background_threshold = kwargs.get('background_threshold', -20)
    scales = kwargs.get('scales',np.arange(1,10))

    coefs, freqs = pywt.cwt(audio,
                            scales,
                            wavelet_type,
                            sampling_period=1.0/(fs))
    relevant_freqs = freqs[freqs>=lowest_relevant_freq]
    if np.sum(relevant_freqs) == 0:
        raise ValueError('The lowest relevant frequency is too high. Please re-check the value')
    
    within_centre_freqs = np.logical_and(np.min(freqs)<=lowest_relevant_freq,
                                         np.max(freqs)>=lowest_relevant_freq)
    if not within_centre_freqs:
        raise ValueError('The lowest relevant frequency %.2f is not included in the centre frequencies of the wavelet scales.\
                          Increase the scale range.'%np.round(lowest_relevant_freq,2))

    lowest_frequency_row  =  int(np.argwhere(np.min(relevant_freqs)==freqs))
    summed_profile = np.sum(abs(coefs[:lowest_frequency_row+1,:]), 0)
    
    dbrms_profile = dB(moving_rms_edge_robust(summed_profile, **kwargs))
    dbrms_profile -= np.max(dbrms_profile)

    if np.min(dbrms_profile) >= background_threshold:
        raise IncorrectThreshold('The dynamic range of the signal is lower than the background threshold.\
        Please decrease the background threshold')

    potential_region = identify_valid_regions(dbrms_profile>=background_threshold, 1)

    return potential_region, dbrms_profile

def identify_valid_regions(condition_satisfied, num_expected_regions=1):
    '''
    
    Parameters
    ----------
    condition_satisfied : np.array
        Boolean numpy array with samples either being True or False. 
        The array may have multiple regions which satisfy a conditions (True)
        separated by smaller regions which don't (False).
    num_expected_regions : int > 0 
        The number of expected regions which satisfy a condition. 
        If >1, then the first two longest continuous regions will be returned,
        and the smaller regions will be suppressed/eliminated.
        Defaults to 1. 

    Returns
    -------
    valid_regions : np.array
        Boolean array which identifies the regions with the longest
        contiguous lengths.
    ADDDDD HERE !! 
    '''
    regions_of_interest, all_region_data = identify_maximum_contiguous_regions(condition_satisfied, num_expected_regions)
    valid_samples = []
    
    all_region_ids = all_region_data[:,0]
    for each in regions_of_interest:
        valid_samples.append(all_region_data[all_region_ids==each,1])
    valid_samples = np.concatenate(valid_samples)
    
    valid_regions = np.asarray(np.zeros(condition_satisfied.size), dtype='bool')
    valid_regions[valid_samples] = True
    
    return valid_regions

def identify_maximum_contiguous_regions(condition_satisfied, number_regions_of_interest=1):
    '''Given a Boolean array - this function identifies regions of contiguous samples that
    are true and labels each with its own region_number. 
    
    Parameters
    ----------
    condition_satisfied : np.array
        Numpy array with Boolean (True/False) entries for each sample. 
    number_regions_of_interest : integer > 1
        Number of contiguous regions which are to be detected. The region ids 
        are output in descending order (longest-->shortest).
        Defaults to 1. 
    
    Returns
    -------
    region_numbers : list
        List with numeric IDs given to each contiguous region which is True.
    region_id_and_samples : np.array
        Two columns numpy array. Column 0 has the region_number, and Column 1 has 
        the individual samples that belong to each region_number. 

    Raises
    -------
    ValueError : This happens if the condition_satisfied array has no entries that are True. 
    
    '''
    region_number = 0
    region_and_samples = []
    # identify the Trues, and assign the sample index to a region number
    for i,each in enumerate(condition_satisfied):
        if each:
            region_and_samples.append([region_number, i])
        else:
            region_number += 1
    # count number of samples in each region and output the top 1/2/... regions 
    try:
        region_id_and_samples = np.concatenate(region_and_samples).reshape(-1,2)
        regions, region_length = np.unique(region_id_and_samples[:,0], return_counts=True)

        region_numbers = []
        for i in range(number_regions_of_interest):
            if i ==0:
                index = np.argmax(region_length)
                region_numbers.append(regions[index])
                remaining_region_lengths = np.delete(region_length, index)
                remaining_regions = np.delete(regions, index)
            elif i>0:
                index = np.argmax(remaining_region_lengths)
                region_numbers.append(remaining_regions[index])
                remaining_region_lengths = np.delete(remaining_region_lengths, index)
                remaining_regions = np.delete(remaining_regions, index)

        return region_numbers, region_id_and_samples
    except:
        raise ValueError('No regions satisfying the condition found: all entries are False') 


def pre_process_for_segmentation(call, fs, **kwargs):
    '''Performs a series of steps on a raw cf call before passing it for temporal segmentation 
    into cf and fm. 
    Step 1: find peak frequency
    Step 2: lowpass (fm_audio) and highpass (cf_audio) below
            a fixed percentage of the peak frequency
    Step 3: calculate the moving dB  of the fm and cf audio
    
    Parameters
    -----------
    call : np.array
    fs : int.
        Frequency of sampling in Hertz
    peak_percentage : 0<float<1, optional
        This is the fraction of the peak at which low and high-pass filtering happens.
        Defaults to 0.98.
    lowpass : optional
        Custom lowpass filtering coefficients. See low_and_highpass_around_threshold
    highpass : 
        Custom highpass filtering coefficients. See low_and_highpass_around_threshold
    window_size : integer, optional
        The window size in samples over which the moving rms of the low+high passed signals will be calculated.
        For default value see documentation of moving_rms
    
    Returns
    -------
    cf_dbrms, fm_dbrms : np.arrays
        The dB rms profile of the high + low passed versions of the input audio.
    '''
    peak_percentage = kwargs.get('peak_percentage', 0.99)
    if peak_percentage >= 1.0:
        raise ValueError('Peak percentage is %f. It cannot be >=1 '%np.round(peak_percentage,2))
    make_sure_its_positive(peak_percentage, variable='peak percentage')
    
    peak_frequency, _ = get_peak_frequency(call, fs)
    
    fraction_of_nyquist = peak_frequency/(fs*0.5)
    if  fraction_of_nyquist >= 0.75*(fs*0.5):
        print(warnings.warn('The peak frequency in the call is %f ... this might lead to erroneous output!'%fraction_of_nyquist))

    threshold_frequency = peak_frequency*peak_percentage
    fm_dominant_audio, cf_dominant_audio = low_and_highpass_around_threshold(call, fs, threshold_frequency, **kwargs)

    fm_rms = moving_rms_edge_robust(fm_dominant_audio, **kwargs)
    cf_rms = moving_rms_edge_robust(cf_dominant_audio, **kwargs)

    fm_dbrms, cf_dbrms = dB(fm_rms), dB(cf_rms)
    return cf_dbrms, fm_dbrms

def low_and_highpass_around_threshold(audio, fs, threshold_frequency, **kwargs):
    '''Make two version of an audio clip: the low pass and high pass versions.
    
    Parameters
    ----------
    audio : np.array
    fs : float>0
        Frequency of sampling in Hz
    threshold_frequency : float>0
        The frequency at which the lowpass and highpass operations are 
        be done. 
    lowpass,highpass : ndarrays, optional
        The b & a polynomials of an IIR filter which define the
        lowpass and highpass filters.
        Defaults to a second order elliptical filter with rp of 3dB
        and rs of 10 dB. See signal.ellip for more details of rp and
        rs.
    pad_duration : float>0, optional
        Zero-padding duration in seconds before low+high pass filtering. 
        Defaults to 0.1 seconds.

    Returns
    -------
    lp_audio, hp_audio : np.arrays
        The low and high pass filtered versions of the input audio. 
        
    '''
    lowpass = kwargs.get('lowpass', signal.ellip(2,3,10, threshold_frequency/(0.5*fs), 'lowpass'))
    highpass = kwargs.get('highpass', signal.ellip(2,3,10, threshold_frequency/(0.5*fs), 'highpass'))
    pad_duration = kwargs.get('pad_duration', 0.1)
    make_sure_its_positive(pad_duration, variable='pad_duration')
    pad_length = int(pad_duration*fs)
    
    
    audio_padded = np.pad(audio, [pad_length]*2, mode='constant', constant_values=(0,0))

    lp_audio_raw = signal.filtfilt(lowpass[0], lowpass[1], audio_padded)
    lp_audio = lp_audio_raw[pad_length:-pad_length]
    
    hp_audio_raw = signal.filtfilt(highpass[0], highpass[1], audio_padded)
    hp_audio = hp_audio_raw[pad_length:-pad_length]
    
    return lp_audio, hp_audio

def get_thresholds_re_max(cf_dbrms, fm_dbrms):
    '''
    '''
    fm_threshold = np.arange(-10,0)
    cf_threshold = np.arange(-10,0)
    fm_db_re_max = fm_dbrms - np.max(fm_dbrms)
    cf_db_re_max = cf_dbrms - np.max(cf_dbrms)

    fm_cf_duration = []
    fm_and_cf_thresholds = []
    num_shared_fm_cf_samples = []
    for each_fm in fm_threshold:
        for each_cf in cf_threshold:
            fm_and_cf_thresholds.append((each_fm, each_cf))
            fm_samples = fm_db_re_max >= each_fm
            cf_samples = cf_db_re_max >= each_cf

            common_fmcf_samples = np.sum(np.logical_and(cf_samples, fm_samples))
            num_shared_fm_cf_samples.append(common_fmcf_samples)

            cf_and_fm_samples = np.sum(cf_samples)*np.sum(fm_samples)
            fm_cf_duration.append(cf_and_fm_samples)

    # choose the parameter region that will allow the best compromise between number of common fm_cf samples 
    # and the longest fm_cf durations. 
    optimisation_metric = np.array(num_shared_fm_cf_samples)/np.array(cf_and_fm_samples)
    best_index = np.argmin(optimisation_metric)
    best_threshold = fm_and_cf_thresholds[best_index]
    
    return num_shared_fm_cf_samples, optimisation_metric, best_threshold

def instantaneous_frequency_profile(audio, fs, **kwargs):
    hil = signal.hilbert(audio)
    instantaneous_phase = np.unwrap(np.angle(hil))
    instantaneous_frequency = (np.diff(instantaneous_phase)/(2.0*np.pi)) * fs
    instant_frequency_resized = resize_by_adding_one_sample(instantaneous_frequency, audio, **kwargs)
    return instant_frequency_resized



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

def identify_cf_ish_regions(frequency_profile, fs, **kwargs):
    '''Identifies CF regions by comparing the rate of frequency modulation 
    across the signal. If the frequency modulation within a region of 
    the signal is less than the limit then it is considered a CF region. 

    Parameters
    ----------
    frequency_profile : np.array
        The instantaneous frequency of the signal over time in Hz. 
    fm_limit : float, optional 
        The maximum rate of frequency modulation in Hz/ms. 
        Defaults to 200 Hz/ms
    medianfilter_size : float, optional

    Returns
    -------
    cfish_regions : np.array
        Boolean array where True indicates a low FM rate region. 
        The output may still need to be cleaned before final use. 
    clean_fmrate_resized

    See Also
    --------
    median_filter
    '''
    max_modulation = kwargs.get('fm_limit', 200) # Hz/msec
    fm_rate = np.diff(frequency_profile)
    
    #convert from Hz/(1/fs) to Hz/msec
    fm_rate_hz_sec = fm_rate/(1.0/fs)
    fm_rate_hz_msec = fm_rate_hz_sec*10**-3
    
    clean_fmrate = median_filter(fm_rate_hz_msec, fs, **kwargs)
    clean_fmrate_resized = resize_by_adding_one_sample(clean_fmrate, frequency_profile, **kwargs)

    cfish_regions = np.abs(clean_fmrate_resized)<= max_modulation
    return cfish_regions, clean_fmrate_resized

def segment_cf_regions(audio, fs, **kwargs):
    '''
    '''
    freq_profile_raw = instantaneous_frequency_profile(audio,fs, **kwargs)
    freq_profile_clean = median_filter(freq_profile_raw, fs, **kwargs)
    cf_region, fmrate_hz_per_msec = identify_cf_ish_regions(freq_profile_clean, fs, **kwargs)
    return cf_region, fmrate_hz_per_msec
    


class IncorrectThreshold(ValueError):
    pass