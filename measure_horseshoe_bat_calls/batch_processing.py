# -*- coding: utf-8 -*-
"""Runs the batch processing option.
The main outputs are the call measurements and the visualisations. 

If you'd like to access the raw audio - then it's better writing a custom
script yourself. 

Created on Fri Mar 27 15:46:00 2020

@author: tbeleyur
"""
import os
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from measure_horseshoe_bat_calls.segment_horseshoebat_call import segment_call_into_cf_fm
from measure_horseshoe_bat_calls.segment_horseshoebat_call import segment_call_from_background
from measure_horseshoe_bat_calls.view_horseshoebat_call import check_call_background_segmentation
from measure_horseshoe_bat_calls.view_horseshoebat_call import make_overview_figure
from measure_horseshoe_bat_calls.user_interface import segment_and_measure_call
from measure_horseshoe_bat_calls.user_interface import save_overview_graphs
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
    batchfile_name = get_only_filename(batchfile_path)

    analysis_name = '_'.join(['measurements',batchfile_name])
    measurements_output_file = analysis_name + '.csv'

    all_measurements = []

    for row_number, one_batchfile_row in tqdm(batch_data.iterrows(),
                                              total=batch_data.shape[0]):
        subplots_to_graph = []
        input_arguments = parse_batchfile_row(one_batchfile_row)
        main_audio, fs = load_raw_audio(input_arguments)
        
        audio_file_name = get_only_filename(input_arguments['audio_path'])
        print('Processing '+audio_file_name+' ...')
        segment_from_background = to_separate_from_background(input_arguments)
        
        if segment_from_background:
            main_call_window, _ = segment_call_from_background(main_audio, 
                                                              fs,
                                                              **input_arguments)
            callbg_wavef, _ = check_call_background_segmentation(main_audio,
                                                               fs,
                                                               main_call_window,
                                                               **input_arguments)
            subplots_to_graph.append(callbg_wavef)
        
        (cf, fm, info), call_parts, measurements = segment_and_measure_call(main_audio,
                                                    fs, **input_arguments)
        
        overview_figure = make_overview_figure(main_audio, fs,
                             measurements,
                             **input_arguments)
        subplots_to_graph.append(overview_figure)
        
        save_overview_graphs(subplots_to_graph, batchfile_name, audio_file_name,
                             row_number, **input_arguments)
        measurements['audio_file'] = audio_file_name
        all_measurements = save_measurements_to_file(measurements_output_file, 
                                  audio_file_name,all_measurements,
                                  measurements, row_number)

def save_measurements_to_file(output_filepath,
                              audio_file_name, 
                              previous_rows, measurements, row_number):
    '''
    Thanks to tmss @ https://stackoverflow.com/a/46775108
    
    Parameters
    ----------
    output_filepath :str/path
    previous_rows : pd.DataFrame
        All the previous measurements. 
        Can also just have a single row. 
    row_data : dictionary 
    row_number : int
    
    Returns
    -------
    None
    
    Notes
    -----
    Main side effect is to write an updated version of the 
    output file. 
    '''
    current_row = pd.DataFrame(measurements, index=[row_number])
    if row_number == 0:
        previous_rows = current_row.copy()
        previous_rows.sort_index(axis=1, inplace=True)
        check_preexisting_file(output_filepath)
        previous_rows.to_csv(output_filepath, 
               mode='a', index=True, sep=',', encoding='utf-8')
    else:
        previous_rows = pd.concat((previous_rows, current_row))
        print(previous_rows)
        previous_rows.iloc[row_number: row_number+1,:].to_csv(output_filepath,
               mode='a', index=True, sep=',', encoding='utf-8', header=False)
    return previous_rows


def check_preexisting_file(file_name):
    '''
    Raises
    ------
    ValueError : if the target file name already exists in the current directory
    '''
    exists = os.path.exists(file_name)

    if exists:
        mesg = 'The file: '+file_name+' already exists- please move it elsewhere or rename it!'
        raise ValueError(mesg)

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

def get_only_filename(file_path):
    folder, file_w_extension = os.path.split(file_path)
    filename, extension = os.path.splitext(file_w_extension)
    return filename
    

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
        

to_string = lambda X: str(X)
to_float = lambda X: float(X)
to_integer = lambda X: int(X)

convert_column_to_proper_type = {
        'audio_path': to_string,
        'start': to_float,
        'stop' : to_float,
        'channel' : to_integer,
        'peak_percentage' : to_float,
        'window_size' : to_integer,
        'min_fm_duration': to_float,
        'lowest_relevant_freq' : to_float,
        'background_threshold' : to_float,
        'terminal_frequency_threshold' : to_float,
        'fft_size' : to_integer
        }

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
                arguments[column] = convert_column_to_proper_type[column](value)
            except:
                pass
    
    if len(columns_to_remove)>0:
        for each in columns_to_remove:
            try:
                del arguments[each]
            except KeyError:
                pass
    
    return arguments
    