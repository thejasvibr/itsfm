# -*- coding: utf-8 -*-
"""
Signal Cleaning Module
~~~~~~~~~~~~~~~~~~~~~~
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
    extrap_window : float>0
        The duration of the extrapolation window in seconds.
        Defaults to 0.5*10^-3s

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
            smooth_X[each_region] = anomaly_interpolation(each_region, X)
    return smooth_X


def anomaly_extrapolation(region, X, num_samples):
    '''
    Takes X values next to the region and fits a linear regression 
    into the region
    
    Notes
    ------
    This function covers 90% of cases...if there is an anomaly right next
    to an edge anomaly with <num_samples distance -- of course things will
    go whack.
    '''
    start, stop = region[0].start, region[0].stop
    x = np.arange(start,stop)

    try:
        ref_x = range(stop, stop+num_samples)
        ref_range = X[ref_x]
    except:
        ref_x = range(start-num_samples, start)
        ref_range = X[ref_x]

    m, c,rv, pv, stderr = stats.linregress(ref_x, ref_range)
    extrapolated = m*x + c 
    return extrapolated

def anomaly_interpolation(region, X):
    '''
    Interpolates X values bet
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
    These drastic variations in signal can cause the creation of [potholes](https://en.wikipedia.org/wiki/Pothole)
    (holes in a road). This method tries to 'level' out the pothole by re-setting the samples of the 
    pothole. A linear interpolation is done from the start of a pothole till its end using the closest 
    non-pothole samples. 
    
    A pothole is identified by a region of the signal with drastic changes in slope. A moving window
    calculates the slopes between the focal sample and the Nth sample after it to estimate if 
    the values move gradually or not. 

    Parameters
    ----------
    X : np.array
    fs : float>0
    max_stepsize : float>0, optional 
        The maximum absolute difference between adjacent samples. 
        Defaults to 50. 
    pothole_inspection_window : float>0, optional
        The length of the moving window that's used to discover potholes.

    Returns
    -------
    pothole_covered
    pothol
    
    
    -=
    See Also
    --------
    identify_pothole_samples
    
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
    background_noise = kwargs.get('background_noise', -40) # dBrms
    signal_dBrms = dB(moving_rms_edge_robust(input_audio, **kwargs))
    bg_noise_suppressed = suppress_to_zero(main_signal, signal_dBrms,
                                           background_noise, 'below')
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
