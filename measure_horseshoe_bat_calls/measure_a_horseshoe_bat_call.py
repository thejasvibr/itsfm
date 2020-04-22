#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Module that measures each continuous CF and FM segment with either 
inbuilt or user-defined functions. 


"""
from measure_horseshoe_bat_calls.signal_processing import *
from measure_horseshoe_bat_calls.sanity_checks import make_sure_its_negative
import measure_horseshoe_bat_calls.measurement_functions as measurefuncs
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import ndimage

def measure_hbc_call(call, fs, cf, fm, **kwargs):
    '''Performs common or unique measurements on the 
    
    Parameters
    ----------
    audio : np.array
    fs : float>0.
         Frequency of sampling in Hz.
    cf : np.array
        Boolean array with True indicating samples that define the CF
    fm : np.array
        Boolean array with True indicating samples that define the FM
    measurements : list, optional
        List with measurement functions 

    Returns
    --------
    sound_segments : dictionary
        Dictionary with keys 'cf' and 'fm'. 
        There is only one entry for 'cf'
        and upto two entries for 'fm'.
    measurement_values : pd.DataFrame
        A wide format dataframe with one row corresponding to all 
        the measured values for a CF or FM segment
    
    
    See Also
    --------
    measure_horseshoe_bat_calls.measurement_functions

    Example 
    -------
    Create a call with fs and make fake CF and FM segments
    
   
    >>> fs = 1.0    
    >>> call = np.random.normal(0,1,100)    
    >>> cf = np.concatenate((np.tile(0, 50), np.tile(1,50))).astype('bool')    
    >>> fm = np.invert(cf)

    Get the default measurements by not specifying any measurements explicitly.
    
    >>> sound_segments, measures = measure_hbc_call(call, fs,
                                                        cf, fm )    
    >>> print(measures)

    And here's an example with some custom functions.The default measurements
    will appear in addition to the custom measurements.
    
    >>> from measure_horseshoe_bat_calls.measurement_functions import measure_peak_amplitude, measure_peak_frequency
    >>> custom_measures = [peak_frequency, measure_peak_amplitude]    
    >>> sound_segments, measures = measure_hbc_call(call, fs,
                                                        cf, fm, 
                                                        measurements=custom_measures)
    '''
    all_cf_fm_segments = parse_cffm_segments(cf, fm)
    if len(all_cf_fm_segments)==0:
        raise ValueError('No CF or FM segments were found -- please re-check')
    
    if kwargs.get('measurements') is not None:
        all_measurements = common_measurements() + kwargs['measurements']
    else:
        all_measurements = common_measurements()
    
    sound_segments = {}
    measurement_values = []
    for segment in all_cf_fm_segments:
        segment_id, segment_indices = segment
        segment_measurements = perform_segment_measurements(call, fs,
                                                    segment,
                                                    all_measurements, **kwargs)
        sound_segments[segment_id] = call[segment_indices]
        measurement_values.append(segment_measurements)
    
    measurement_values = pd.concat(measurement_values).reset_index(drop=True)

    return sound_segments, measurement_values

def parse_cffm_segments(cf, fm):
    '''Recognises continuous stretches of Cf and FM segments, 
    organises them into separate 'objects' and orders them in time.

    Parameters
    ----------
    cf, fm : np.array
        Boolean arrays indicating which samples are CF/FM.
    
    Returns 
    -------
    cffm_regions_numbered : np.array with tuples.
        Each tuple corresponds to one CF or FM region in the audio. 
        The tuple has two entries 1) the region identifier, eg. 'fm1'
        and 2) the indices that correspond to the region eg. slice(1,50)

    Example
    -------
    # an example sound with two cfs and an fm in the middle
    
    >>> cf = np.array([0,1,1,0,0,0,1,1,0]).astype('bool')
    >>> fm = np.array([0,0,0,1,1,1,0,0,0]).astype('bool')
    >>> ordered_regions = parse_cffm_segments(cf, fm)
    >>> print(ordered_regions)
    '''
    cf_regions, fm_regions = find_regions(cf), find_regions(fm)
    cf_fm_regions_ordered = combine_and_order_regions(cf_regions, fm_regions)
    cffm_regions_numbered = assign_cffm_regionids(cf_fm_regions_ordered, cf_regions,
                                             fm_regions)
    return cffm_regions_numbered
 

def perform_segment_measurements(full_sound, fs, 
                                 segment, functions_to_apply, **kwargs):
    '''Performs one or more measurements on a specific segment of a full audio 
    clip. 
    
    Parameters
    ----------
    full_sound : np.array
    fs : float>0
    segment : tuple
        First object is a string with the segment's id, eg. 'fm1' or 'cf2'
        Second object is a slice with the indices of the segment, eg. slice(0,100)
    functions_to_apply : list of functions
        Each function must be a 'measurement function'. A measurement function
        is one that accepts a strict set of inputs. check See Also for more
        details. 
    
    Returns
    -------
    results : pd.DataFrame
        A single row with all the measurements results. 
        The first column is always the 'regionid', the rest of the columns 
        are measurement function dependent.

    Example
    -------
    Here we'll create a short segment and take the rms and the peak value of
    the segment. The `relevant_region` is not an FM region, it is only labelled
    so here to show how it works with the rest of the package!
    
    >>> np.random.seed(909)
    >>> audio = np.random.normal(0,1,100)    
    >>> relevant_region = ('fm1',slice(10,30))    
    
    The sampling rate  doesn't matter for the custom functions defined below, 
    but, it may be important for some other functions. 
    
    >>> fs = 1 # Hz    
    >>> from measure_horseshoe_bat_calls.measurement_functions import measure_rms, measure_peak    
    >>> results = perform_segment_measurements(audio, fs, relevant_region, 
                                                   [measure_rms, measure_peak])    
    '''
    segment_id, segment_location = segment

    measurement_values = {}
    for each_function in functions_to_apply:
        value = each_function(full_sound, fs, segment_location, **kwargs)
        measurement_values.update(value)
    results = pd.DataFrame(data=measurement_values, index=[0])
    results['region_id'] = segment_id
    return results

def find_regions(X):
    '''
    '''
    region_ids, num_regions = ndimage.label(X.flatten())
    region_locations = np.array(ndimage.find_objects(region_ids)).flatten()
    return region_locations

def combine_and_order_regions(cf_slices, fm_slices):
    '''
    '''
    cffm_regions = np.sort(np.concatenate((cf_slices, fm_slices)))
    return cffm_regions

def assign_cffm_regionids(cffm, cf_regions, fm_regions):
    '''
    '''
    cf_counter = 1 
    fm_counter = 1 
    
    assigned_ids = []
    for i, region in enumerate(cffm):
        if region in cf_regions:
            regiontype = 'cf'
            region_number = str(cf_counter)
            cf_counter += 1 
        elif region in fm_regions:
            regiontype = 'fm'
            region_number = str(fm_counter)
            fm_counter += 1 
        else:
            raise ValueError('Could not find the current regions, please check the region', region)
        region_id = regiontype + region_number
        assigned_ids.append([region_id, region])
    return assigned_ids


def common_measurements():
    '''Loads the default common measurement set
    for any region. 
    '''
    common_funcs =  [ getattr(measurefuncs, each) for each in ['start', 'stop',
                                                             'duration'] ]
    return common_funcs


