# -*- coding: utf-8 -*-
"""
This module handles the identification and cleaning of  noise in signals. A 'noisy' signal 
is one that has spikes in it or sudden variations in a continuous looking 
function. Most of these functions are built to detect and handle sudden
spikes in the frequency profile estimates of a sound. 


"""
import numpy as np 
from scipy import ndimage, stats
from measure_horseshoe_bat_calls.signal_processing import moving_rms_edge_robust
from measure_horseshoe_bat_calls.signal_processing import median_filter, resize_by_adding_one_sample
from measure_horseshoe_bat_calls.signal_processing import dB
from measure_horseshoe_bat_calls.sanity_checks import make_sure_its_positive


def exterpolate_over_anomalies(X, fs, anomalous, **kwargs):
    ''' 
    Ex(tra)+(in)ter-polates --> Exterpolates over  anomalous regions. Anomalous
    regions are either 'edge' or 'island' types. The 'edge' anomalies are those which are 
    at the extreme ends of the signal. The 'island' anomalies are regions with 
    non-anomalous regions on the left and right. 
   
    An 'edge' anomalous region is handled by running a linear regression on the 
    neighbouring non-anomalous region, and using the slope to extrapolate over
    the edge anomaly. 
    
    An 'island' anomaly is handled by interpolating between the end values of the 
    neighbouring non-anomalous regions. 
   
    Parameters
    ----------
    X : np.array
    fs : float>0
        Sampling rate in Hz
    anomalous : np.array
        Boolean array of same size as X
        True indicates an anomalous sample. 
    extrap_window : float>0, optional
        The duration of the extrapolation window in seconds.
        Defaults to 0.5ms 
        
    Returns
    -------
    smooth_X : np.array
        Same size as X, with the anomalous regions 
    
    Notes
    -----
    Only extrapolation by linear regression is supported currently. The `extrap_window`
    parameter is important especially if there is a high rate of frequency modulation
    towards the edges of the sound. When there is a high freq. mod. at the edges it
    is better to set the `extrap_window` small. However, setting it too small also
    means that the extrapolation may not be as nice anymore. 

    Example
    -------
    
    `not up to date!!!`
    
    See Also
    --------
    find_closest_normal_region

    '''
    smooth_X = X.copy()
    extrap_window = kwargs.get('extrap_window', 0.5*10**-3)
    ref_region_length = int(extrap_window*fs)
    
    anomalous_broader = ndimage.filters.percentile_filter(anomalous, 100, 
                                                              ref_region_length)
    
    anomalous_labelled, num_regions = ndimage.label(anomalous_broader)
    if num_regions == 0:
        return smooth_X

    anomalous_regions = ndimage.find_objects(anomalous_labelled)
    
    for each_region in anomalous_regions:
        region_type = anomaly_type(each_region, X)
        if region_type == 'edge':
            smooth_X[each_region] = anomaly_extrapolation(each_region, X, 
                                                            ref_region_length)
        elif region_type == 'island':
            smooth_X[each_region] = fix_island_anomaly(X, fs, each_region,
                                                       ref_region_length, 
                                                       **kwargs)
    return smooth_X

def fix_island_anomaly(X, fs, anomaly, ref_region_length, **kwargs):
    '''
    First tries to interpolate between the edges of the anomaly at hand. 
    If the interpolation leads to a very drastic slope, a 'sensible' extrapolation 
    is attempted using parts of the non-anomalous signal. 
    
    Parameters
    ----------
    X : np.array
    fs : float>0
    anomaly : tuple slice
        scipy.ndimage.find_objects output
        (slice(start,stop,None),)
    ref_region_length : int>0
        The number of samples to be used as a reference region in 
        case of extrapolation
    max_fmrate : float>0, optional
        The maximum fm rate to be tolerated while interpolating in kHz/ms
        Defaults to 100 kHz/ms.
    
    Returns
    -------
    interpolated : np.array
        Array of same size as anomaly. 
    '''
    max_fmrate = kwargs.get('max_fmrate', 100)
    trial_fix = anomaly_interpolation(anomaly, X)
    fmrate_trialfix = calc_coarse_fmrate(trial_fix, fs)
    if fmrate_trialfix <= max_fmrate:
        return trial_fix
    else:
        return extrapolate_sensibly(X, fs, anomaly, ref_region_length, **kwargs)
        
def extrapolate_sensibly(X, fs, anomaly, ref_region_length, **kwargs):
    '''
    Function called when `fix_island_anomaly` detects direct interpolation
    will lead to unrealistic slopes. This function is called when there's
    a big difference in values across an anomalous region and an
    extrapolation must be performed which will not alter the signal drastically. 
    
    The method tries out the following:
        #. Look left and right of the anomaly to see which region 
           has higher frequency content.
        #. Extrapolate in the high-to-low frequency direction. 
    
    This basically means that if the local inspection window around anomaly has
    a sweep between 20-10kHZ on the left and a 0Hz region on the right - the 
    anomaly will be extrapolated with the slope from the sweep region because it
    has higher frequency content. 
    
    Example
    -------
    >>> freq_profile = [np.zeros(10), np.arange(15,30,5)*1000]
    >>> fs = 1.0
    >>> x = np.concatenate(freq_profile)[::-1]
    >>> anom = (slice(2, 5, None),)
    >>> 
    >>> plt.plot(x, label='noisy frequency profile')
    >>> anom_x = np.zeros(x.size, dtype='bool')
    >>> anom_x[anom[0]] = True
    >>> plt.plot(anom_x*8000, label='identified anomaly')
    >>> extrap_out = extrapolate_sensibly(x, fs, anom, 4)
    >>> sensibly_extrap = x.copy()
    >>> sensibly_extrap[anom_x] = extrap_out
    >>> plt.plot(sensibly_extrap, label='extrapolated')
    >>> plt.legend()
    '''

    left_and_right_of_X = get_neighbouring_regions(X, anomaly, ref_region_length)
    left_median, right_median = map(np.median, left_and_right_of_X)
    
    start, stop = anomaly[0].start, anomaly[0].stop
    anom_size = stop-start  
    if left_median > right_median:
        relevant_region = anomaly
        extrap_chunk = anomaly_extrapolation(relevant_region, X[:stop],
                                                             ref_region_length,
                                                             **kwargs)
        
    else:
        relevant_region = (slice(0,anom_size,None),)
        extrap_chunk = anomaly_extrapolation(relevant_region, 
                                             X[start:],
                                             ref_region_length, **kwargs)
    
    return extrap_chunk
    

def get_neighbouring_regions(X, target, region_size):
    '''
    Takes out samples of `region_size` on either size of the target. 
    
    Parameters
    ----------
    X: np.array
    target : slice
        ndimage.find_objects type slice
    region_size : int >0
    
    Returns
    -------
    left_and_right : list
    '''
    start = target[0].start
    stop = target[0].stop

    before_start = start-region_size
    if before_start < 0 :
        left_of_X = X[:start]
    else:
        left_of_X = X[before_start:start]
  
    try:
        right_of_X = X[stop:stop+region_size]
    except:
        right_of_X = X[stop:]
    return [left_of_X, right_of_X]

def calc_coarse_fmrate(X,fs,**kwargs):
    '''
    Calculates slope by subtracting the difference between 1st and 
    last sample and dividing it by the length of the array. 
    The output is then converted to units of kHz/ms. 
    
    Parameters
    ----------
    X : np.array
        Frequency profile with values in Hz. 
    fs : float>0
    
    '''
    diff = X[-1] - X[0]
    length = X.size/fs
    return  np.abs((diff/length)*10**-6)
    
    
    


def anomaly_extrapolation(region, X, num_samples, **kwargs):
    '''
    Takes X values next to the region and fits a linear regression 
    into the region. This is only suitable for cases where the 
    anomalous region is at an 'edge' - either one of its samples
    is 0 or the last sample of X. 
    
    Parameters
    ----------
    region : object tuple
        A slice type object which is the output from scipy.ndimage.find_objects
        This is a slice inside a list/tuple.
    X : np.array
        The original array over which the extrapolation is to be performed
    num_samples : int>0
        The number of samples next to the region to be used to fit the data
        for extrapolation into the region. 

    Returns
    -------
    extrapolated : np.array
        The values corresponding to the extrapolated region. 
    
    Notes
    ------
    1. This function covers 90% of cases...if there is an anomaly right next
    to an edge anomaly with <num_samples distance -- of course things will
    go whack.
    
    Warning
    -------
    A mod on this function also allows extrapolation to occur if there
    are < num_samples next to the anomaly - this might make the function
    a bit lax in terms of the extrapolations it produces.
    
    '''
    start, stop = region[0].start, region[0].stop
    x = np.arange(start,stop)
    if start == 0:
        try:
            ref_x = range(stop, stop+num_samples)
            ref_values = X[ref_x]
        except:
            ref_x = range(stop, X.size)
            ref_values = X[stop:]
        
    elif stop==X.size:
        try:
            if start-num_samples <0:
                raise ValueError()
            else:
                ref_x = range(start-num_samples, start)
                ref_values = X[ref_x]
        except:
            ref_x = range(start)
            ref_values = X[ref_x]
    else:
        print(start, stop, x, 'anomaly x', X.size)
        raise NotImplementedError('the handling of none-edge case is not yet doen')
    m, c,rv, pv, stderr = stats.linregress(ref_x, ref_values)
    extrapolated = m*x + c 
    return extrapolated

def anomaly_interpolation(region, X, **kwargs):
    '''
    Interpolates X values using values of X adjacent to the 
    region. 
    
    Parameters
    ----------
    region : object tuple
        Output from scipy.ndimage.find_objects
    X : np.array
    
    Returns
    -------
    full_span : np.array
        The values of interpolated X, of same size as the 
        region length. 
    '''
    start, stop = region[0].start, region[0].stop
    left_point = start-1
    full_span = np.linspace(X[left_point],X[stop],stop-left_point+1)
    return full_span[1:-1]

def anomaly_type(region, X):
    start, stop = region[0].start, region[0].stop
    
    at_left_edge = start==0
    at_right_edge = stop==X.size
    
    if np.logical_and(at_left_edge, at_right_edge):
        raise ValueError('The anomaly spans the whole array - please check again')
    at_either_edge = np.logical_or(at_left_edge, at_right_edge)        
    
    if at_either_edge:
        return 'edge'
    else:
        return 'island'



def smooth_over_potholes(X, fs, **kwargs):
    '''
    A signal can show drastic changes in its value because of measurement errors.
    These drastic variations in signal are called `potholes <https://en.wikipedia.org/wiki/Pothole>`_
    (uneven parts of a road). This method tries to 'level' out the pothole by re-setting the samples of the 
    pothole. A linear interpolation is done from the start of a pothole till its end using the closest 
    non-pothole samples. 
    
    A pothole is identified by a region of the signal with drastic changes in slope. A moving window
    calculates N slopes between the focal sample and the Nth sample after it to estimate if 
    the Nth sample could be part of a pothole or not. 

    Parameters
    ----------
    X : np.array
    fs : float>0
    max_stepsize : float>0, optional 
        The maximum absolute difference between adjacent samples. 
        Defaults to 50. 
    pothole_inspection_window : float>0, optional
        The length of the moving window that's used to discover potholes.
        See identify_pothole_samples for default value.

    Returns
    -------
    pothole_covered
    pothole_regions

    See Also
    --------
    identify_pothole_samples
    pothole_inspection_window

    '''
    kwargs['max_stepsize'] = kwargs.get('max_stepsize', 50)
    potholes = identify_pothole_samples(X, fs, **kwargs)

    abnormal_fm, num_regions = ndimage.label(potholes)
    if num_regions < 1:
        return X, []
    else:
        pothole_covered = X.copy()
        pothole_regions = ndimage.find_objects(abnormal_fm)
        spikeish_indices = get_all_spikeish_indices(pothole_regions)
        for each_region in pothole_regions:
            region = each_region[0]
            start, stop = region.start, region.stop
            region_length = stop-start
            # find the sample values next to the start and stop

            next_to_start = find_non_forbidden_index(start, spikeish_indices, -1, X)
            next_to_stop =  find_non_forbidden_index(stop, spikeish_indices, +1, X)
            interpolated_values = np.linspace(pothole_covered[next_to_start],
                                              pothole_covered[next_to_stop],
                                              region_length)
            pothole_covered[start:stop] = interpolated_values

    return pothole_covered, pothole_regions



def identify_pothole_samples(X, fs,  **kwargs):
    '''Moves a sliding window and checks the values of samples in the sliding window. 
    If the jump of values between samples is not linearly propotional to the 
    expected max_stepsize, then it is labelled a pothole sample. 
    
    A pothole sample is one which represents a sudden jump in the values - indicating
    a noisy tracking of the frequency. The jump in values in a non-noisy signal is expected
    to be proportional to the distance between the samples. 
    
    For instance, if : 
    
    >>> a = np.array([10, 2, 6, 10, 12])
    
    If the max step size is 2, then because abs(10-2)>2, it causes a pothole to appear on 2.
    There is no pothole label on the 2nd index because abs(10-6) is not >4. Because 10 and 6
    are two samples apart, the maximum allowed jump in value is max_stepsize*2, which is 4. 
    
    For optimal pothole detection the 'look-ahead' span of the pothole_inspection_window
    should at least the size of the longest expected potholes. Smaller window sizes
    will lead to false negatives.
    
    Parameters
    ----------
    X : np.array
    fs : float>0
    max_stepsize : float>0
        The max absolute difference between the values of one sample to the next. 

    pothole_inspection_window : float>0, optional
        Defaults to 0.25ms

    Returns
    -------
    pothole_candidates : np.array
        Boolean array with same size as X. Sample that are True represent pothole candidates.


    See Also
    --------
    detect_local_potholes
    '''
    # forward pass
    left2right_pothole_candidates, _ = onepass_identify_potholes(X, fs,
                                                                 **kwargs)
    # backward pass
    right2left_pothole_candidates, _ = onepass_identify_potholes(X[::-1], fs,
                                                              **kwargs)
    pothole_candidates = np.logical_and(left2right_pothole_candidates>0,
                                        right2left_pothole_candidates[::-1]>0)
    
    return pothole_candidates
    
    
def onepass_identify_potholes(X, fs, max_stepsize, **kwargs):
    '''
    '''
    window_duration = kwargs.get('pothole_inspection_window', 0.25*10**-3)
    window_size = int(fs*window_duration)
    potholes = []
    consensus = np.zeros(X.size)

    for i,each in enumerate(X):
        candidates = detect_local_potholes(X[i:i+window_size], max_stepsize)
        potholes.append(candidates)
        consensus[i+candidates] += 1
    return consensus, potholes
    
def detect_local_potholes(X, max_step_size):
    '''accepts a 1D array and checks the absolute difference between 
    the first sample and all other samples. 
    
    The samples with difference greater than the linearly expected increase
    from max_step_sizes are labelled candidate potholes.
    
    Parameters
    ----------
    X : np.array
    max_step_size : float>=0
    
    Returns
    -------
    candidate_potholes : np.array
        Boolean array of same size as X
    '''
    pothole_depth = np.abs(X-X[0])
    max_allowed = np.arange(X.size)*max_step_size
    candidate_potholes = np.argwhere(pothole_depth > max_allowed).flatten()
    return candidate_potholes


def get_all_spikeish_indices(regions):
    '''
    '''
    indices = np.concatenate(list(map(extract_indices, regions)))
    return indices

def find_non_forbidden_index(candidate, forbidden_indices, search_direction, X):
    '''
    '''

    index_in_spike = candidate in forbidden_indices
    if index_in_spike:
        next_candidate = search_operation[search_direction](candidate)
        return find_non_forbidden_index(next_candidate, forbidden_indices, search_direction, X)
    else:
        candidate_index_within_array = np.logical_and(candidate>=0, candidate<=X.size-1)
        if candidate_index_within_array:
            return candidate 
        else:
            search_direction *= -1 
            next_candidate = search_operation[search_direction](candidate)
            return find_non_forbidden_index(next_candidate, forbidden_indices, search_direction, X)

def extract_indices(X):
    indices = np.arange(X[0].start, X[0].stop)
    return indices 


search_operation = {-1: lambda X: X-1,
                    1: lambda X : X+1}



def remove_bursts(X, fs, **kwargs):
    '''Bursts are brief but large jumps in the signal above zero. Even though they satisfy
    most of the other conditions of beginning above the noise floor and of 
    being above 0 value, they still are too short to be relevant signals. 

    Parameters
    ----------
    X : np.array 
        The noisy signal to be handled
    fs : float>0
        Sampling rate in Hz. 
    min_element_length : float>0, optional
        The minimum length a section of the signal must be to be 
        kept in seconds. Defaults to 5 inter-sample-intervals. 
    
    Returns
    -------
    X_nonspikey : np.array
        Same size as X, and without very short segments. 
    
    See Also
    --------
    segments_above_min_duration
    
    Notes
    -----
    An inter-sample-interval is defined as 1/fs
    

    '''
    inter_sample_durn = 1.0/fs
    min_element_length = kwargs.get('min_element_length', 5*inter_sample_durn) #to 5 samples 
    min_element_samples = int(fs*min_element_length)
    
    if  min_element_length <= inter_sample_durn:
        raise ValueError('Please set the min element length.\
        The current value of:%f is less than 1/sampling rate'%(min_element_length))
    min_element_samples = int(fs*min_element_length)
    
    non_spikey_regions = segments_above_min_duration(X>0, min_element_samples)
   
    X_nonspikey = np.zeros(X.size)
    X_nonspikey[non_spikey_regions] = X[non_spikey_regions]
    return X_nonspikey


def segments_above_min_duration(satisfies_condition, min_samples):
    '''Accepts a boolean array and looks for continuous chunks 
    that are above a minimum length. 

    Parameters
    ----------
    satisfies_condition : np.array
        Boolean array where samples with True satisfy a condition. 
    min_samples : int >0 
        The minimum number of samples a continuous region of True 
        must be to be kept. 
    
    Returns
    -------
    above_min_duration : np.array
        Same size as satisfies_condition, with only the continuous
        chunks that are above min_samples. 
       
    '''
    all_regions, number_regions = ndimage.label(satisfies_condition)
    region_stretches = ndimage.find_objects(all_regions)
    
    above_min_duration = np.tile(False, satisfies_condition.size)
    
    for each_stretch in region_stretches:
        if satisfies_condition[each_stretch].size > min_samples:
            above_min_duration[each_stretch] = True
    return above_min_duration


def suppress_background_noise(main_signal, input_audio, **kwargs):
    '''
    '''
    signal_level = kwargs.get('signal_level', -20) # dBrms
    signal_dBrms = dB(moving_rms_edge_robust(input_audio, **kwargs))
    bg_noise_suppressed = suppress_to_zero(main_signal, signal_dBrms,
                                           signal_level, 'below')
    return bg_noise_suppressed


def suppress_frequency_spikes(noisy_profile, input_audio, fs, **kwargs):
    '''
    '''
    max_spike_rate = kwargs.get('max_spike_rate', 3000) # Hz jump/sample 
    
    # median filter to get rid of smaller fluctuations in the noisy profile *not*
    # caused by abrupt transitions in the edges. 
    med_filtered = median_filter(noisy_profile, fs, **kwargs)
    
    raw_fmrate = abs(np.diff(med_filtered))
    delta_profile = resize_by_adding_one_sample(raw_fmrate, input_audio)
    spike_suppressed = suppress_to_zero(noisy_profile, delta_profile, max_spike_rate, 'above')
    return spike_suppressed



def suppress_to_zero(target_signal, basis_signal, threshold, mode='below'):
    '''
    Sets the values of the target signal to zero if the 
    samples in the basis_signal are $\geq$ or $\leq$ the threshold

    Parameters
    ----------
    target_signal, basis_signal : np.array
    threshold : float
    mode : ['below', 'above'], str

    Returns
    -------
    cleaned_signal : np.array
        A copy of the target signal with the values that are below/above the threshold 
        set to zero

    Example
    --------
    # create a basis signal with a 'weak' left half and a 'loud' right hald
    # we want to suppress the we
    >>> basis = np.concatenate((np.arange(10), np.arange(100,200)))
    >>> target_signal = np.random.normal(0,1,basis.size)
    >>> cleaned_target = suppress_to_zero(basis, target_signal, 100, mode='above')
    '''
    if mode == 'below':
        to_suppress = basis_signal < threshold
    elif mode == 'above':
        to_suppress = basis_signal > threshold
    else: 
        raise ValueError('Mode should be either "below" or "above" and not: %s'%(mode))
    cleaned_signal = np.copy(target_signal)
    cleaned_signal[to_suppress.flatten()] = 0 
    return cleaned_signal


def clip_tfr(tfr, **kwargs):
    '''
    Parameters
    ----------
    tfr : np.array
        2D array with the time-frequency representation of choice
        (pwvd, fft etc). The tfr must have real-valued non-negative
        values as the clip range is defined in dB. 
    tfr_cliprange: float >0, optional
        The maximum dynamic range in dB which will be used to 
        track the instantaneous frequency. Defaults to 
        None. See `Notes` for more details
    
    Returns
    -------
    clipped_tfr : np.array
        A 2d array of same shape as `tfr`, with values 
        clipped between [max, max x 10^(tfr_range/20)]
    Notes
    -----
    The `tfr_cliprange` is used to remove the presence of 
    background noise, faint harmonics or revernberations/echoes
    in the audio. This of course all assumes that the main 
    signal itself is sufficiently intense in the first place. 
    
    After the PWVD time-frequency represenation is made, 
    values below X dB of the maximum value are 'clipped' to 
    the same minimum value. eg. if the pwvd had values of 
    [0.1, 0.9, 0.3, 1, 0.001, 0.0006] and the tfr_cliprange is
    set to  6dB, then the output of the clipping will be 
    [0.5, 0.9, 0.3, 1, 0.5, 0.5].  This step essentially eliminates
    any variation in the array, thus allowing a clear 
    tracking of the highest component in it. 
    '''
    tfr_cliprange = kwargs.get('tfr_cliprange')
    if tfr_cliprange is None:
        return tfr
    else:
        make_sure_its_positive(tfr_cliprange, variable='tfr_cliprange')
        max_value = np.max(tfr)
        clip_value = max_value*10**(-tfr_cliprange/20.0)
        
        clipped_tfr = tfr.copy()
        clipped_tfr[clipped_tfr<clip_value] = clip_value
        
        return clipped_tfr
        

def conditionally_set_to(X, conditional, bool_state):
    '''Inverts the samples in X where the conditional is True. 
    Parameters
    ----------
    X : np.array
        Boolean
    conditional : np.array
        Boolean
    bool_state : [True, False]
    
    Returns
    -------
    cond_set_X : np.array
        conditionally set X
    
    Notes
    -----
    this function is useful if you want to 'suppress' a few samples
    conditionally based ont he values of the same samples 
    on another array. 
    
    Example
    -------
    >>> x = np.array([True, True, False, False, True])
    >>> y = np.array([0,0,10,10,10])
    Imagine x is some kind of detection array, while y is the 
    signal-to-noise ratio at each of the sample points. Of course, 
    you'd like to discard all the predictions from low SNR measurements.
    Let's say you want to keep only those entries in X where y is >1.
    >>> x_cond = conditionally_set_to(x, y<10, False)
    >>> x_cond
    
    np.array([False, False, False, False, True ])
    '''
    if sum(conditional)==0:
        return X
    else:
        cond_set_X = X.copy()
        cond_set_X[conditional] = bool_state
        return cond_set_X
