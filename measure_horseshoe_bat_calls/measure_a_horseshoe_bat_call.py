#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Module that segments and measures parts of a CF-FM call. 

"""
from __future__ import absolute_import

from measure_horseshoe_bat_calls.segment_horseshoebat_call import identify_maximum_contiguous_regions
from measure_horseshoe_bat_calls.signal_processing import *
from measure_horseshoe_bat_calls.sanity_checks import make_sure_its_negative
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize']
import numpy as np
import pandas as pd
import scipy.signal as signal


def measure_hbc_call(call, fs, cf_segment, fm_segment, **kwargs):
    '''Segments and measures a horseshoe bat call 
    
    Parameters
    ----------
    audio : np.array
    
    fs : float>0.
         Frequency of sampling in Hz.
    cf_segment : np.array
        Boolean array with True indicating samples that define the CF
    fm_segment : np.array
        Boolean array with True indicating samples that define the FM

    Returns
    --------
    sound_segments : dictionary
        Dictionary with keys 'cf' and 'fm'. 
        There is only one entry for 'cf'
        and upto two entries for 'fm'.
    measurements : pd.DataFrame
        A single row of all the measurements. 
    '''
    sound_segments = {}
    measurements = {}
    # whole call measurements
    measurements['call_duration'] = call.size/float(fs)
    measurements['call_energy'] = calc_energy(call)
    measurements['call_rms'] = rms(call)
    peak_f, peak_f_resolution = get_peak_frequency(call, fs)
    measurements['peak_frequency'] = peak_f
    measurements['peak_frequency_resolution'] = peak_f_resolution

    # CF measurements
    cf = call[cf_segment]
    sound_segments['cf'] = cf
    measurements['cf_start'] = np.min(np.argwhere(cf_segment))/float(fs)
    measurements['cf_end'] = np.max(np.argwhere(cf_segment))/float(fs)

    measurements['cf_duration'] = cf.size/float(fs)
    measurements['cf_energy'] = calc_energy(cf)
    measurements['cf_rms'] = rms(cf)
    measurements['cf_peak_frequency'], measurements['cf_peakfreq_resolution']  = get_peak_frequency(cf, fs)

    # FM measurements
    cf_startstop = (measurements['cf_start'], measurements['cf_end'])
    fm_types, fm_sweeps, fm_startstop = get_fm_snippets(call,
                                                        fm_segment,
                                                        fs, cf_startstop)
    sound_segments['fm'] = fm_sweeps

    for each_type, sweep, fm_boundaries in zip(fm_types, fm_sweeps, fm_startstop):
        measurements[each_type+'duration'] = sweep.size/float(fs)
        measurements[each_type+'energy'] = calc_energy(sweep)
        measurements[each_type+'rms'] = rms(sweep)
        measurements[each_type+'terminal_frequency'], threshold = get_terminal_frequency(sweep, 
                                                                    fs=fs,
                                                                    **kwargs)
        measurements[each_type+'terminalfreq_resolution'] = get_frequency_resolution(sweep, fs)

        start_time, stop_time = fm_boundaries
        measurements[each_type+'start'] = start_time
        measurements[each_type+'end'] = stop_time
        
    measurements['terminal_frequency_threshold'] = threshold

    return sound_segments, measurements

def get_fm_snippets(whole_call, fm_segments, fs, cf_startstop):
    '''Creates separate audio clips for each FM sweep and labels them as
    either 'upfm' or 'downfm' 

    Parameters
    ----------
    whole_call : np.array
        Whole call audio
    fm_segments : np.array
        Boolean numpy array with True indicating samples in the whole_call 
        that are FM sweeps. 
    fs : float>0
        Sampling rate in Hz
    cf_startstop : tuple
        The start and stop times of the CF segment in seconds. 

    Returns
    --------
    fm_types : list
        List with strings labelling the type of FM sweep. 
        All sweeps ending in the first half of the call are labelled 'upfm'
        and those ending in the second half are labelled 'downfm'.
        There can be 1 OR 2 entries in fm_types depending on the call segmentation
        results. 
    fm_audio : list
        List with audio snippets
    fm_boundaries : list
        list with start and stop time of fm sweep in seconds.
    '''
    try:
        fms, fm_and_samples = identify_maximum_contiguous_regions(fm_segments, 2)
    except:
        fms, fm_and_samples = identify_maximum_contiguous_regions(fm_segments, 1)
    finally:
        raise FMIdentificationError('Could not detect any FM parts...')

    fm_id = fm_and_samples[:,0]
    fm_samples = fm_and_samples[:,1]

    fm_types = []
    fm_audio = []
    fm_boundaries = []
    for each_fm in fms:        
        this_fm_samples = fm_samples[fm_id==each_fm]
        fm_startand_stop = np.array([np.min(this_fm_samples),
                                     np.max(this_fm_samples)])/float(fs)
        this_fm_audio = whole_call[this_fm_samples]
        this_fm_type = which_fm_type(fm_startand_stop, cf_startstop)
        fm_audio.append(this_fm_audio)
        fm_types.append(this_fm_type)
        fm_boundaries.append(fm_startand_stop)
    
    return fm_types, fm_audio, fm_boundaries
        


## from the make_CF_training_data module
def make_one_CFcall(call_durn, fm_durn, cf_freq, fs, call_shape, **kwargs):
    '''A test function used to check how well the segmenting+measurement
    functions in the module work. 
    
    Parameters
    ----------
    call_durn : float
    fm_durn : float
    cf_freq : float
    fs : float
    call_shape : str
        One of either 'staplepin' OR 'rightangle'
    fm_bandwidth : float, optional
        FM bandwidth in Hz.


    Returns
    --------
    cfcall : np.array
        The synthesised call. 

    Raises
    -------
    ValueError
        If a call_shape that is not  'staplepin' OR 'rightangle' is given

    Notes
    ------
    This is not really the besssst kind of CF call to test the functions on, 
    but it works okay. The CF call is made by using the poly spline function 
    and this leads to weird jumps in frequency especially around the CF-FM
    junctions. Longish calls with decently long FM parts look fine, but calls
    with very short FM parts lead to rippling of the frequency. 
    '''
    # choose an Fm start/end fr equency :
    FM_bandwidth = np.arange(2,20)
    fm_bw = kwargs.get('fm_bandwidth', np.random.choice(FM_bandwidth, 1)*10.0**3)
    start_f = cf_freq - fm_bw
    # 
    polynomial_num = 25
    t = np.linspace(0, call_durn, int(call_durn*fs))
    # define the transition points in the staplepin
    freqs = np.tile(cf_freq, t.size)
    numfm_samples = int(fs*fm_durn)
    if call_shape == 'staplepin':       
        freqs[:numfm_samples] = np.linspace(start_f,cf_freq,numfm_samples,
                                                     endpoint=True)
        freqs[-numfm_samples:] = np.linspace(cf_freq,start_f,numfm_samples,
                                                     endpoint=True)
        p = np.polyfit(t, freqs, polynomial_num)

    elif call_shape == 'rightangle':
        # alternate between rising and falling right angle shapes
        rightangle_type = np.random.choice(['rising','falling'],1)
        if rightangle_type == 'rising':
            freqs[:numfm_samples] = np.linspace(cf_freq,start_f,numfm_samples,
                                                         endpoint=True)
        elif rightangle_type == 'falling':
            freqs[-numfm_samples:] = np.linspace(cf_freq,start_f,numfm_samples,
                                                         endpoint=True)
        p = np.polyfit(t, freqs, polynomial_num)

    else: 
        raise ValueError('Wrong input given')
      
    cfcall = signal.sweep_poly(t, p)

    #windowing = np.random.choice(['hann', 'nuttall', 'bartlett','boxcar'], 1)[0]
    windowing= 'boxcar'
    cfcall *= signal.get_window(windowing, cfcall.size)
    cfcall *= signal.tukey(cfcall.size, 0.01)
    return cfcall




def which_fm_type(fm_startand_stop, cf_startstop):
    '''figures out whether its an up or down fm using the start and stop times. 
    The distance between the midpoint of the fm segment and the cf start
    and stop points are calculated. If the cf start is closer to the fm midpoint
    the fm is considered a 'upfm_' and a 'downfm_' otherwise. 

    Parameters
    ----------
    fm_startand_stop, cf_startstop : np.array
     The start and stop time of the FM and CF segments in seconds. 

    Returns
    -------
    fm_type : str. 
        Either 'upfm_' or 'downfm_'
    '''
    fm_middle = np.mean(fm_startand_stop)
    fm_distance = np.abs(np.array(cf_startstop).flatten() - fm_startand_stop)
    fm_types = ['upfm_', 'downfm_']
    this_fm_type = fm_types[np.argmin(fm_distance)]
    return this_fm_type


def get_terminal_frequency(audio, **kwargs):
    '''Gives the -XdB frequency from the peak. 

    The power spectrum is calculated and smoothened over 3 frequency bands to remove
    complex comb-like structures. 
    
    Then the lowest frequency below XdB from the peak is returned. 

    Parameters
    ----------
    audio : np.array
    fs : float>0
        Sampling rate in Hz
    terminal_frequency_threshold : float, optional
        The terminal frequency is calculated based on finding the level of the peak frequency
        and choosing the lowest frequency which is -10 dB (20log10) below the peak level. 
        Defaults to -10 dB

    Returns 
    ---------
    terminal_frequency       
    threshold 

    Notes
    -----
    Careful about setting threshold too low - it might lead to output of terminal
    frequencies that are actually in the noise, and not part of the signal itself. 
    '''
    threshold = kwargs.get('terminal_frequency_threshold', -10)
    make_sure_its_negative(threshold, variable='terminal frequency threshold')
    
    power_spectrum, freqs,  = get_power_spectrum(audio, kwargs['fs'])
    # smooth the power spectrum over 3 frequency bands to remove 'comb'-iness in the spectrum
    smooth_spectrum = np.convolve(10**(power_spectrum/20.0), np.ones(3)/3,'same')
    smooth_power_spectrum = dB(abs(smooth_spectrum))

    peak = np.max(smooth_power_spectrum)
    geq_threshold = smooth_power_spectrum >= peak + threshold
    all_frequencies_above_threshold = freqs[geq_threshold]

    terminal_frequency = np.min(all_frequencies_above_threshold)
    return terminal_frequency, threshold
  
class FMIdentificationError(ValueError):
    pass

class BackgroundSegmentationError(ValueError):
    pass
    
    
