# -*- coding: utf-8 -*-
"""Runs the batch processing option. The main outputs are the call measurements
and the visualisations. (See __main__.py)

.. code-block:: bash

    $ python -m itsfm -batchfile template_batchfile.csv

Also allows the user to run only one specific row of the whole batch file 

.. code-block:: bash

    $ python -m itsfm -batchfile template_batchfile.csv -one_row 10

The line above loads the 11th row (0-based indexing!!) of the template_batchfile


"""
from copy import copy
import os
import pdb
import matplotlib.pyplot as plt
import pandas as pd
try:
	import soundfile as sf
except:
	print('Cannot import SoundFile!!') # a hack for rtd build to pass.
from tqdm import tqdm
import itsfm
from itsfm.user_interface import segment_and_measure_call
from itsfm.user_interface import save_overview_graphs
from itsfm.view import itsFMInspector
from itsfm.sanity_checks import check_preexisting_file

def run_from_batchfile(batchfile_path, one_row=None):
    '''
    Parameters
    ----------
    batchfile_path : str/path
        Path to a batchfile 
    one_row : int
        A specific row to be loaded from the whole batchfile
        The first row starts with 1!!!
    
    
    '''
    batch_data = load_batchfile(batchfile_path)
    if one_row is not None:
        try:
            batch_data = make_to_oned_dataframe(batch_data.loc[one_row])
        except:
            print(f"Unable to subset batch file with row number: {one_row}")

    batchfile_name = get_only_filename(batchfile_path)

    analysis_name = '_'.join(['measurements',batchfile_name])
    measurements_output_file = analysis_name + '.csv'

    all_measurements = []

    for row_number, one_batchfile_row in tqdm(batch_data.iterrows(),
                                              total=batch_data.shape[0]):

        input_arguments = parse_batchfile_row(one_batchfile_row)
        main_audio, fs = load_raw_audio(input_arguments)
        
        audio_file_name = get_only_filename(input_arguments['audio_path'])
        print('Processing '+audio_file_name+' ...')
        segment_and_measure = segment_and_measure_call(main_audio,
                                                           fs, 
                                                    **input_arguments)
        out_inspect = itsFMInspector(segment_and_measure, main_audio, fs, 
                                     **input_arguments)
        (cf, fm, info), call_parts, measurements = segment_and_measure
        
        # start making diagnostic plots
        one = out_inspect.visualise_geq_signallevel()
        two, _ = out_inspect.visualise_cffm_segmentation()
        three = out_inspect.visualise_frequency_profiles()
        four, _ = out_inspect.visualise_fmrate()
        five, _ = out_inspect.visualise_accelaration()
        
        
        subplots_to_graph = [one, two, three, four, five]
        
        save_overview_graphs(subplots_to_graph, batchfile_name, audio_file_name,
                             row_number, **input_arguments)
        measurements['audio_file'] = audio_file_name
        all_measurements = save_measurements_to_file(measurements_output_file, 
                                  audio_file_name,all_measurements,
                                  measurements)
        plt.close('all')

def save_measurements_to_file(output_filepath,
                              audio_file_name, 
                              previous_rows, measurements):
    '''
    Continously saves a row to a csv file and updates it. 

    Thanks to tmss @ https://stackoverflow.com/a/46775108
    
    Parameters
    ----------
    output_filepath :str/path
    audio_file_name : str. 
    previous_rows : pd.DataFrame
        All the previous measurements. 
        Can also just have a single row. 
    measurements : pd.DataFrame
        Current measurements to be incorporated

    Returns
    -------
    None, previous rows
    
    Notes
    -----
    Main side effect is to write an updated version of the 
    output file. 
    '''
    #raise NotImplementedError('Long format measurement saving not implemented!!')
    current_measures = measurements.copy()
    if len(previous_rows)==0:
        previous_rows = current_measures.copy()
        previous_rows.sort_index(axis=1, inplace=True)
        check_preexisting_file(output_filepath)
        previous_rows.to_csv(output_filepath, 
               mode='a', index=True, sep=',', encoding='utf-8')
    else:
        num_new_rows = current_measures.shape[0]
        current_last_row = previous_rows.shape[0]        
        previous_rows = pd.concat((previous_rows, current_measures))
        
        new_row, new_row_end = current_last_row, current_last_row+num_new_rows
        previous_rows.iloc[new_row: new_row_end,:].to_csv(output_filepath,
                                        mode='a', index=True,
                                        sep=',', encoding='utf-8',
                                        header=False)
    return previous_rows

def load_batchfile(batchfile):
    try:
        return pd.read_csv(batchfile)
    except:
        error_msg = 'Could not read batchfile:'+ batchfile+'. Please check file path again'
        raise ValueError(error_msg)

def load_raw_audio(kwargs):
    '''Takes a dictioanry input. 
    All the parameter names need to be keys in the
    input dictionary. 

    Parameters
    -----------
    audio_path : str/path
        Path to audio file 
    channel : int, optional
        Channel number to be loaded - starting from 1!
        Defaults to 1.
    start,stop : float, optional

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
to_bool = lambda X: {'True':True, 'False':False}[X]

def to_list_w_funcs(X, source_module=itsfm.measurement_functions,
                    **kwargs):
    """
    

    Parameters
    ----------
    X : str
        String defining a list with commas as separators
        eg. "[func_name1, func_name2] "
    source_module : str, optional 
        Defaults to itsfm.measurement_functions
    signs_to_remove : list w str
        Any special signs to remove from each str
        in the list of comma separated strings. 
        Defaults to None. 
    Returns
    -------
    list_w_funcs
        list with functions belonging to the source module

    Example
    -------
    >>> x = "[measure_rms, measure_peak_amplitude]"
    >>> list_w_funcs = to_list_w_funcs(x)

    """
    individual_strings = X.split(',')
    # remove unnecessary punctuations
    
    list_w_funcs = []
    for each in individual_strings:
        cleaned = remove_punctuations(each, **kwargs)
        try:
            list_w_funcs.append(getattr(source_module, cleaned))
        except:
            raise ValueError(f"Unable to find function {cleaned} in module {source_module}")
    return list_w_funcs


def remove_punctuations(full_str, **kwargs):
    """
    Removes spaces, ], and [ in a string. 
    Additional signs can be removed too

    Parameters
    ----------
    full_str : str
        A long string with multiple punctuation marks 
        to be removed (space, comma, ])
    signs_to_remove : list w str', optional
        Additional specific punctuation/s to be removed
        Defaults to None
    Returns
    -------
    clean_str : str
    """
    clean_str = copy(full_str)
    # remove spaces
    clean_str = clean_str.replace(" ", "")
    # remove ]
    clean_str = clean_str.replace("]", "")
    # remove [
    clean_str = clean_str.replace("[", "")
    
    if kwargs.get('signs_to_remove') is not None:
        for each in kwargs['signs_to_remove']:
            clean_str = clean_str.replace(each, "")
    
    return clean_str
        
    
    

# dictionary which converts the entries in a column to 
# their appropriate types
convert_column_to_proper_type = {
        'audio_path': to_string,
        'start': to_float,
        'stop' : to_float,
        'channel' : to_integer,
        'peak_percentage' : to_float,
        'window_size' : to_integer,
        'signal_level' : to_float,
        'terminal_frequency_threshold' : to_float,
        'fft_size' : to_integer,
        'segment_method' : to_string,
        'tfr_cliprange' : to_float,
        'pwvd_window' : to_integer,
        'pwvd_filter' : to_bool,
        'measurements' : to_list_w_funcs,
        'sample_every' : to_float
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


def make_to_oned_dataframe(oned_series):
    """
    

    Parameters
    ----------
    oned_series : pd.Series
        One dimensional pd.Series with columns and values

    Returns
    -------
    oned_df

    """
    columns = oned_series.index.to_list()
    values = oned_series.values
    
    entries = data={key:value for key, value in zip(columns, values)}
    oned_df = pd.DataFrame(data=entries, index=[0])
    return oned_df
    
    