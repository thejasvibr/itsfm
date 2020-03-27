# -*- coding: utf-8 -*-
"""Runs the batch processing option.
The main outputs are the call measurements and the visualisations. 

If you'd like to access the raw audio - then it's better writing a custom
script yourself. 

Created on Fri Mar 27 15:46:00 2020

@author: tbeleyur
"""
import pandas as pd
import soundfile as sf
import measure_horseshoe_bat_calls.segment_horseshoebat_call 
from measure_horseshoe_bat_calls.segment_horseshoebat_call import segment_call_into_cf_fm

#keyword arguments for call-background segmentation
call_background_keywords = ['lowest_relevant_freq', 
                            'wavelet_type',
                            'background_threshold',
                            'scales']

# keyword arguments for cf-fm segmentation
cf_fm_keywords = ['peak_percentage', 
                  'window_size',
                  'lowpass',
                  'highpass'
                  'pad_duration'
                  'min_fm_duration']

# keywords for the visualisation module
view_keywords = [
        
                ]

def run_from_batchfile(batchfile_path):
    '''
    Parameters
    ----------
    batchfile_path : str/path
        Path to a batchfile 

    '''
    batch_data = load_batchfile(batchfile_path)
    
    for row_number, one_batchfile_row in batch_data.iterrows():
        input_arguments = parse_batchfile_row(one_batchfile_row)
        raw_audio, fs = load_raw_audio(input_arguments)
        segment_from_background = to_separate_from_background(input_arguments)
        print('TO SEGMENT?', segment_from_background)
        print('input type', type(input_arguments))
        if segment_from_background:
            main_call_window, _ = segment_call_into_cf_fm(raw_audio, 
                                                              fs,
                                                              input_arguments)
            print(main_call_window.size)

def load_batchfile(batchfile):
    try:
        return pd.read_csv(batchfile)
    except:
        error_msg = 'could not read batchfile:'+ batchfile+'. Please check file path again'
        raise ValueError(error_msg)
 
def load_raw_audio(kwargs):
    '''Checks to see 
    Parameters
    -----------
    audio_path : str/path
        Path to audio file 
    channel : int, optional
        Channel number to be loaded - starting from 1!
        Defaults to 1.
    Returns
    --------
    raw_audio : np.array
        The audio corresponding to the start and stop times
        and the required channel. 
    '''
    audio_path = kwargs.get('audio_path', None)
    try: 
        fs = sf.info(audio_path).samplerate
    except:
        errormsg = 'Could not access: '+audio_path
        raise ValueError(errormsg)
    channel_to_load = int(kwargs.get('channel', 1)) -1 

    start_time, stop_time = kwargs.get('start', None), kwargs.get('stop',  None)
    start_sample = convert_time_to_samples(start_time, fs)
    stop_sample = convert_time_to_samples(stop_time, fs)

    audio, fs = sf.read(audio_path, start=start_sample, stop=stop_sample)
    num_channels = get_number_channels(audio)
    
    if num_channels>1:
        return audio[:, channel_to_load], fs
    else:
        return audio, fs

def to_separate_from_background(arguments):
    '''
    '''
    try:
        user_input = arguments.get('segment_call_background', True)
        boolean_user_input = get_boolean_from_string[user_input]
        return boolean_user_input
    except:
        error = 'user input '+user_input+' for segment_call_background is not True or False or DEFAULT - please check'
        raise ValueError(error)

get_boolean_from_string = {'True':True, 
                          'False':False,
                          True:True,
                          False:False}
    

def get_number_channels(audio):
    try:
        rows,cols = audio.shape
        return cols
    except:
        return 1
    

def convert_time_to_samples(time, fs):
    
    if not(time is None):
        samples = int(time*fs)
    else:
        samples = None
    return samples 
        


def parse_batchfile_row(one_row):
    '''checks for all user-given arguments 
    and removes any columns with DEFAULT in them. 

    Parameters
    ---------
    one_row : pd.DataFrame
        A single row with multiple column names, corresponding to 
        compulsory required arguments and the optional 
        ones
    
    Returns
    -------
    arguments : dictionary
        Simple dictioanry with one entry for each key.
    '''
    arguments = one_row.to_dict()
    
    # remove all keys with 'NONE' in them
    columns_to_remove = []
    for column, value in arguments.items():
        if value=='DEFAULT':
            columns_to_remove.append(column)
        else:
            # convert to relevant type:
            try:
                arguments[column] = float(value)
            except:
                pass
    
    if len(columns_to_remove)>0:
        for each in columns_to_remove:
            try:
                del arguments[each]
            except KeyError:
                pass
    
    return arguments
    