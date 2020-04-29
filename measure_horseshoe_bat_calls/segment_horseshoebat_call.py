# -*- coding: utf-8 -*-
"""Module that segments the horseshoebat call into FM and CF parts
The primary logic of this 

"""
import warnings
import numpy as np 
import scipy.interpolate as interpolate 
from scipy import ndimage
import scipy.ndimage.filters as flts
import scipy.signal as signal 
from measure_horseshoe_bat_calls.signal_processing import *
from measure_horseshoe_bat_calls.sanity_checks import make_sure_its_positive
from measure_horseshoe_bat_calls.frequency_tracking import get_pwvd_frequency_profile
import measure_horseshoe_bat_calls.refine_cfm_regions as refine_cfm
from measure_horseshoe_bat_calls.signal_cleaning import suppress_background_noise
from measure_horseshoe_bat_calls.signal_cleaning import conditionally_set_to

def segment_call_into_cf_fm(call, fs, **kwargs):
    '''Function which identifies regions into CF and FM based on the following   process. 

    1. Candidate regions of CF and FM are first produced based on the segmentation
    method chosen'.

    2. These candidate regions are then refined based on the 
    user's requirements (minimum length of region, maximum number of CF/FM
    regions in the sound)

    3. The finalised CF and FM regions are output as Boolean arrays.

    Parameters
    -----------
    call : np.array
        Audio with horseshoe bat call
    fs : float>0
        Frequency of sampling in Hz. 
    segment_method : str, optional 
        One of ['peak_percentage', 'pwvd', 'inst_freq'].
        Checkout 'See Also' for more information. 
        Defaults to 'peak_percentage'
    refinement_method : function, str, optional
        The method used to refine the initial CF and FM
        candidate regions according to the different constraints
        and rules set by the user. 
        
        Defaults to 'do_nothing'
        
        
    Returns
    --------
    cf_samples, fm_samples : np.array
        Boolean numpy array showing which of the samples belong 
        to the cf and the fm respectively. 

    info : dictionary
        Post-processing information depending on 
        the methods used. 

    Example
    -------
    Create a chirp in the middle of a somewhat silent recording
    
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np 
    >>> from measure_horseshoe_bat_calls.simulate_calls import make_fm_chirp, make_tone
    >>> from measure_horseshoe_bat_calls.view_horseshoebat_call import plot_movingdbrms
    >>> from measure_horseshoe_bat_calls.view_horseshoebat_call import visualise_call, make_x_time
    >>> from measure_horseshoe_bat_calls.view_horseshoebat_call import plot_cffm_segmentation
    >>> fs = 44100
    >>> start_f, end_f = 1000, 10000
    >>> chirp = make_fm_chirp(start_f, end_f, 0.01, fs)  
    >>> tone_freq = 11000
    >>> tone = make_tone(tone_freq, 0.01, fs)
    >>> tone_start = 30000; tone_end = tone_start+tone.size
    >>> rec = np.random.normal(0,10**(-50/20), 44100)
    >>> chirp_start, chirp_end = 10000, 10000 + chirp.size
    >>> rec[chirp_start:chirp_end] += chirp
    >>> rec[tone_start:tone_end] += tone
    >>> rec /= np.max(abs(rec))
    >>> actual_fp = np.zeros(rec.size)
    >>> actual_fp[chirp_start:chirp_end] = np.linspace(start_f, end_f, chirp.size)
    >>> actual_fp[tone_start:tone_end] = np.tile(tone_freq, tone.size)
    
    Track the frequency of the recording and segment it according to frequency
    modulation

    >>> cf, fm, info = segment_call_into_cf_fm(rec, fs, signal_level=-10,
                                                   segment_method='pwvd',)
    
    View the output and plot the segmentation results over it:
    >>> plot_cffm_segmentation(cf, fm, rec, fs)

    See Also
    ----------
    segment_by_peak_percentage
    segment_by_pwvd
    segment_by_inst_frequency
    measure_horseshoe_bat_calls.refine_cfm_regions 
    refine_cf_fm_candidates

    Notes
    -----
    The post-processing information in the object `info` depends on the method 
    used. 
    
    peak_percentage : the two keys 'fm_re_cf' and 'cf_re_fm' which are the 
        relative dBrms profiles of FM with relation to the CF portion and vice versa
    
    pwvd : 
    
    
    '''
    segment_method = kwargs.get('segment_method', 'peak_percentage')
    refinement_method = kwargs.get('refinement_method', 'do_nothing')
    # identify candidate CF and FM regions 
    cf_candidates, fm_candidates, info = perform_segmentation[segment_method](call, fs,
                                                                 **kwargs)

    cf, fm = refine_cf_fm_candidates(refinement_method,
                                     [cf_candidates, fm_candidates],
                                     fs, info, **kwargs)

    return cf, fm, info

def refine_cf_fm_candidates(refinement_method, cf_fm_candidates,
                            fs, info, 
                            **kwargs):
    '''Parses the refinement method, checks if its string or function
    and calls the relevant objects. 
    
    Parameters
    ----------
    refinement_method : str/function 
        A string from the list of inbuilt functions in the module
        `refine_cfm_regions` or a user-defined function. 
        Defaults to `do_nothing`, an inbuilt function which
        doesn't returns the candidate Cf-fm regions without 
        alteration. 
    cf_fm_candidates : list with 2 np.arrays
        Both np.arrays need to be Boolean and of the same size as the original
        audio. 
    fs : float>0
    info : dictionary


    Returns
    -------
    cf, fm : np.array
        Boolean arrays wher True indicates the sample is of the corresponding
        region. 

    '''

    if isinstance(refinement_method, str):
        refinement_function = getattr(refine_cfm, refinement_method)
        cf_samples, fm_samples = refinement_function(cf_fm_candidates,
                                                     fs,
                                                     info, 
                                                     **kwargs)
    elif callable(refinement_method):
        # could cause issues with inbuilt functions apparently?
        cf_samples, fm_samples = refinement_method(cf_fm_candidates,
                                                         fs,
                                                         info, 
                                                         **kwargs)
    else:
        raise ValueError('Unable to parse refinement method -  please check input:')

    return cf_samples, fm_samples

def segment_by_peak_percentage(call, fs, **kwargs):
    '''This is ideal for calls with one clear CF section with the CF 
    portion being the highest frequency in the call: bat/bird CF-FM
    calls which have on CF and one/two sweep section.

    Calculates the peak frequency of the whole call and performs 
    low+high pass filtering at a frequency slightly lower than the peak frequency. 


    Parameters
    ----------
    call : np.array
    fs : float>0
    
    Returns
    -------
    cf_samples, fm_samples : np.array
        Boolean array with True indicating that sample has been categorised
        as being CF and/or FM. 
    info : dictionary
        With keys 'fm_re_cf' and 'cf_re_fm' indicating the relative 
        dBrms profiles of the candidate FM regions relative to Cf 
        and vice versa.
    
    Notes
    -----
    This method unsuited for audio with non-uniform call envelopes. 
    When there is high variation over the call envelope, the peak frequency 
    is likely to be miscalculated, and thus lead to wrong segmentation.

    This method is somewhat inspired by the protocol in Schoeppler et al. 2018. 
    However, it differs in the important aspect of being done entirely in the 
    time domain. Schoeppler et al. 2018 use a spectrogram based method 
    to segment the CF and FM segments of H. armiger calls. 

    References
    ----------
    [1] Schoeppler, D., Schnitzler, H. U., & Denzinger, A. (2018). 
        Precise Doppler shift compensation in the hipposiderid bat, 
        Hipposideros armiger. Scientific Reports, 8(1), 1-11.     

    See Also
    --------
    pre_process_for_segmentation
    '''
    cf_dbrms, fm_dbrms = pre_process_for_segmentation(call, fs, **kwargs)
    fm_re_cf = fm_dbrms - cf_dbrms
    cf_re_fm = cf_dbrms - fm_dbrms
    
    fm_samples = fm_re_cf > 0 
    cf_samples = cf_re_fm > 0

    fm_samples = suppress_background_noise(fm_samples, call, **kwargs)
    cf_samples = suppress_background_noise(cf_samples, call, **kwargs)

    info = {'fm_re_cf': fm_re_cf,
            'cf_re_fm':cf_re_fm,
            'cf_dbrms':cf_dbrms,
            'fm_dbrms':fm_dbrms}

    return cf_samples, fm_samples, info 


def segment_by_pwvd(call, fs, **kwargs):
    '''This method is technically more accurate in segmenting CF and FM portions
    of a sound. The Pseudo-Wigner-Ville Distribution of the input signal 
    is generated. 

    Parameters
    ----------
    call : np.array
    fs : float>0
    fmrate_threshold : float >=0
        The threshold rate of frequency modulation in kHz/ms. Beyond this value a segment
        of audio is considered a frequency modulated region. 
        Defaults to 0.2 kHz/ms

    
    Returns
    -------
    cf_samples, fm_samples : np.array
        Boolean array of same size as call indicating candidate CF and FM regions. 
    
    info : dictionary
        See get_pwvd_frequency_profile for the keys it outputs in the `info` 
        dictioanry. In addition, another key 'fmrate' is also calculated
        which has an np. array with the rate of frequency modulation across
        the signal in kHz/ms.

    Notes
    -----
    This method may takes some time to run. It is computationally intensive. 
    This method may not work very well in the presence of multiple harmonics
    or noise. Some basic tweaking of the optional parameters may be required. 

    
    See Also
    --------
    get_pwvd_frequency_profile
    
    Example
    -------
    Let's create a two component call with a CF and an FM part in it 
    >>> from measure_horseshoe_bat_calls.simulate_calls import make_tone, make_fm_chirp, silence
    >>> from measure_horseshoe_bat_calls.view_horseshoebat_call import plot_cffm_segmentation    
    >>> from measure_horseshoe_bat_calls.view_horseshoebat_call import make_x_time
    >>> fs = 22100
    >>> tone = make_tone(5000, 0.01, fs)
    >>> sweep = make_fm_chirp(1000, 6000, 0.005, fs)
    >>> gap = silence(0.005, fs)
    >>> full_call = np.concatenate((tone, gap, sweep))
    >>> # reduce rms calculation window size because of low sampling rate!
    >>> cf, fm, info = segment_by_pwvd(full_call, 
                                           fs,
                                            window_size=10,
                                            signal_level=-12,
                                            sample_every=1*10**-3,
                                            extrap_length=0.1*10**-3)
    >>> w,s = plot_cffm_segmentation(cf, fm, full_call, fs)
    >>> s.plot(make_x_time(cf,fs), info['fitted_fp'])
    '''
    fmrate_threshold = kwargs.get('fmrate_threshold', 0.2) # kHz/ms

    clean_frequency_profile, info = get_pwvd_frequency_profile(call, fs, **kwargs)

    fmrate, fitted_freq_profile = whole_audio_fmrate(clean_frequency_profile, 
                                                         fs, 
                                                         **kwargs)

    info['fmrate'] = fmrate
    info['cleaned_fp'] = clean_frequency_profile
    info['fitted_fp'] = fitted_freq_profile

    fm_candidates = fmrate > fmrate_threshold
    fm_samples = conditionally_set_to(fm_candidates, 
                                      fitted_freq_profile==0,
                                      False)
    
    cf_candidates = fmrate <= fmrate_threshold
    cf_samples = conditionally_set_to(cf_candidates,
                                      fitted_freq_profile==0, False)

    fm_samples = suppress_background_noise(fm_samples, call, **kwargs)
    cf_samples = suppress_background_noise(cf_samples, call, **kwargs)

    return cf_samples, fm_samples, info
   



def whole_audio_fmrate(whole_freq_profile, fs, **kwargs):
    '''
    When a recording has multiple components to it, there are silences
    in between. These silences/background noise portions are assigned
    a value of 0 Hz. 
    
    When a 'whole audio' fm rate is naively calculated by taking the diff
    of the whole frequency profile, there will be sudden jumps in the fm-rate
    due to the silent parts with 0Hz and the sound segments with non-zero 
    segments. Despite these spikes being very short, they then propagate their
    influence due to the median filtering that is later down downstream. This
    essentially causes an increase of false positive FM segments because of the
    apparent high fmrate. 
    
    To overcome the issues caused by the sudden zero to non-zero transitions 
    in frequency values, this function handles each non-zero sound segment
    separately, and calculates the fmrate over each sound segment independently.

    Parameters
    -----------
    whole_freq_profile : np.array
        Array with sample-level frequency values of the same size as the 
        audio. 
    fs : float>0
    
    Returns
    -------
    fmrate : np.array
        The rate of frequency modulation in kHz/ms. Same size as `whole_freq_profile`
        Regions in `whole_freq_profile` with 0 frequency are set to 0kHz/ms.
    fitted_frequency_profile : np.aray
        The downsampled, smoothed version of `whole_freq_profile`, of the same size. 
    
    Attention
    ---------
    The `fmrate` *must* be processed further downstream! 
    In the whole-audio `fmrate` array, all samples that were 0 frequency 
    in the original `whole_freq_profile` are set to 0 kHz/ms!!!
    
    
    
    See Also
    --------
    calculate_fm_rate
    
    
    Example
    -------
    Let's make a synthetic multi-component sound with 2 FMs and 1 CF component.
    
    >>> fs = 22100
    >>> onems = int(0.001*fs)
    >>> sweep1 = np.linspace(1000,2000,onems) # fmrate of 1kHz/ms
    >>> tone = np.tile(3000, 2*onems) # CF part
    >>> sweep2 = np.linspace(4000,10000,3*onems) # 2kHz/ms
    >>> gap = np.zeros(10)
    >>> freq_profile = np.concatenate((sweep1, gap, tone, gap, sweep2))
    >>> fmrate, fit_freq_profile = whole_audio_fmrate(freq_profile, fs)
    
    '''
    
    sound_segments, num_segments = ndimage.label(whole_freq_profile.flatten()>0)
    location_segments = ndimage.find_objects(sound_segments)

    whole_fmrate = np.zeros(whole_freq_profile.size)
    fitted_frequency_profile = whole_freq_profile.copy()

    if num_segments <1 :
        raise ValueError('No non-zero frequency segments detected!')
    
    for index, location in enumerate(location_segments):
        segment_frequency_profile = whole_freq_profile[location]
        fmrate, fitted_freq_profile = calculate_fm_rate(segment_frequency_profile, 
                                                                    fs, **kwargs)
        whole_fmrate[location] = fmrate
        fitted_frequency_profile[location] = fitted_freq_profile
    
    return whole_fmrate, fitted_frequency_profile
    




def segment_by_inst_frequency(call, fs, **kwargs):
    
    raise NotImplementedError('Plain Instant Tracking has not yet been implemented!')
    
    return None, None, None 



def calculate_fm_rate(frequency_profile, fs, **kwargs):
    '''A frequency profile is generally oversampled. This means that 
    there will be many repeated values and sometimes minor drops in 
    frequency over time. This leads to a higher FM rate than is actually
    there when a sample-wise diff is performed. 
    
    This method downsamples the frequency profile, fits a polynomial 
    to it and then gets the smoothened frequency profile with unique values. 
    
    The sample-level FM rate can now be calculated reliably. 

    Parameters
    ----------
    frequency_profile : np.array
        Array of same size as the original audio. Each sample has 
        the estimated instantaneous frequency in Hz. 
    fs : float>0
        Sampling rate in Hz
    
    Returns
    -------
    fm_rate : np.array
        Same size as frequency_profile. The rate of frequency modulation in 
        kHz/ms
    
    See Also
    --------
    fit_polynomial_on_downsampled_version
    '''
    
    medianfilter_length = kwargs.get('medianfilter_length', 0.1*10**-3)
    try:
        medianfilter_samples = calc_proper_kernel_size(medianfilter_length, fs)
    except:
        raise ValueError('The current medianfilter_length of %fs is too short, increase it a bit more'%medianfilter_length)
    
    fitted = fit_polynomial_on_downsampled_version(frequency_profile, fs, **kwargs)

    fm_rate_hz_per_sec = np.abs(np.gradient(fitted))
    median_filtered = flts.percentile_filter(fm_rate_hz_per_sec, 50, 
                                             medianfilter_samples)
    fm_rate =10**-6*(median_filtered/(1/fs))
    return fm_rate, fitted

def fit_polynomial_on_downsampled_version(frequency_profile, fs, **kwargs):
    '''
    '''
    sample_every = kwargs.get('sample_every', 0.5*10**-3) #seconds
    interpolation_kind = kwargs.get('interpolation_kind', 1) # polynomial order
    ds_factor = int(fs*sample_every)
    
    full_x = np.arange(frequency_profile.size)
    partX = np.concatenate((full_x[:2], full_x[3::ds_factor], 
                                      full_x[-2:])).flatten()
    partX = np.unique(partX).flatten()
    partY = frequency_profile[partX]
    
    fit = interpolate.interp1d(partX, partY, kind=interpolation_kind)
    fitted = fit(np.arange(frequency_profile.size))                           
    return fitted 
    

def refine_candidate_regions():
    '''Takes in candidate CF and FM regions and tries to satisfy the 
    constraints set by the user. 
    '''
    pass


perform_segmentation = {'peak_percentage':segment_by_peak_percentage, 
                        'pwvd':segment_by_pwvd,
                        'inst_freq':segment_by_inst_frequency}



def check_segment_cf_and_fm(cf_samples, fm_samples, fs, **kwargs):
    '''
    '''

    main_cf = get_cf_region(cf_samples, 1)
    main_fm = get_fm_regions(fm_samples, fs, **kwargs)

    return main_cf, main_fm


def get_cf_region(cf_samples, fs, **kwargs):
    '''TODO : generalise to multiple CF regions 
    
    Parameters
    ----------
    cf_samples : np.array
        Boolean with True indicating a Cf region. 
    fs : float

    Returns
    -------
    cf_region : np.array
        The longest continuous stretch
    
    '''
    min_cf_duration = kwargs.get('min_cf_duration', 0.001)
    make_sure_its_positive(min_cf_duration, variable='min_cf_duration')
    min_cf_samples = int(fs*min_cf_duration)
    cf_region = identify_valid_regions(cf_samples, 1)
    if sum(cf_region) < min_cf_samples:
        msg1 = 'CF segment of minimum length (%3f)s'%(min_cf_duration)
        msg2 = ' could not be found'
        raise CFIdentificationError(msg1+msg2)
        
    return cf_region


def get_fm_regions(fm_samples, fs, **kwargs):
    '''TODO : generalise to multiple FM regions
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
        If >2, then the first two longest continuous regions will be returned,
        and the smaller regions will be suppressed/eliminated.
        Defaults to 1. 

    Returns
    -------
    valid_regions : np.array
        Boolean array which identifies the regions with the longest
        contiguous lengths.
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
    if samples < 3:
        msg_part1 = 'The given kernel length of %3f seconds and sampling rate of'%durn
        msg_part2 = ' %f leads to a kernel of < 3 samples length. Increase kernel length!'%fs
        raise ValueError(msg_part1+msg_part2)
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
        The maximum rate of frequency modulation in Hz/s. 
        Defaults to 1000 Hz/s
    medianfilter_size : float, optional

    Returns
    -------
    cfish_regions : np.array
        Boolean array where True indicates a low FM rate region. 
        The output may still need to be cleaned before final use. 
    clean_fmrate_resized
    
    Notes
    -----
    If you're used to reading FM modulation rates in kHz/ms then just 
    follow this relation to get the required modulation rate in Hz/s:
    
    X kHz/ms = (X Hz/s)* 10^-6 
    
    OR 
    
    X Hz/s = (X kHz/ms) * 10^6

    See Also
    --------
    median_filter
    '''
    max_modulation = kwargs.get('fm_limit', 10000) # Hz/sec
    fm_rate = np.diff(frequency_profile)
    
    #convert from Hz/sec to Hz/msec
    fm_rate_hz_sec = fm_rate/(1.0/fs)
    
    clean_fmrate = median_filter(fm_rate_hz_sec, fs, **kwargs)
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
    








class CFIdentificationError(ValueError):
    pass

class IncorrectThreshold(ValueError):
    pass