# -*- coding: utf-8 -*-
"""
This is a set of *measurement functions* which are used to measure various
things about a part of an audio.  A *measurement function* is a specific kind of
 function which accepts three arguments and outputs a dictionary. 

What is a *measurement function*:
#################################
A *measurement function* is a specific kind of function which accepts three arguments and outputs a dictionary.
User-defined functions can be used to perform custom measurements on the segment of interest. 

Measurement function parameters
--------------------------------

    #. the full audio, a np.array
    #. the sampling rate, a float>0
    #. the `segment`, a slice object which defines the span
       of the segment. For instance ('fm1', slice(0,100))

What needs to be returned:
--------------------------

    A measurement function must return a dictionary with >1 keys that are strings
    and items that can be easily incorporated into a Pandas DataFrame and viewed on
    a csv file with ease. Ideal item types include strings, floats, or tuples. 
    
    See the source code of the built-in measurement functions below for an example of 
    how to satisfy the measurement function pattern. 


Attention
---------
Remember to name the output of the measurement function properly. 
If the output key of one measurement function is the same as the 
other, it will get overwritten in the final dictionary!

"""
from itsfm.signal_processing import *


def measure_rms(audio, fs, segment ,**kwargs):
    '''
    
    See Also
    --------
    itsfm.signal_processing.rms
    '''
    return {'rms': rms(audio[segment])}

def measure_peak_amplitude(audio, fs, segment ,**kwargs):
    '''
    '''
    return {'peak_amplitude': np.max(np.abs(audio[segment]))}

def start(audio, fs, segment ,**kwargs):
    '''
    '''
    start_time = segment.start/fs
    return {'start': start_time}

def stop(audio, fs, segment ,**kwargs):
    '''
    '''
    end_time = (segment.stop)/fs
    return {'stop':end_time}

def duration(audio, fs, segment ,**kwargs):
    '''
    '''
    durn = (segment.stop-segment.start)/float(fs)
    return {'duration':durn}

def measure_peak_frequency(audio, fs, segment ,**kwargs):
    '''
    
    See Also
    --------
    itsfm.signal_processing.get_peak_frequency
    '''
    peak_freq, freq_res = get_peak_frequency(audio[segment], fs)
    return {'peak_frequency':peak_freq, 'peak_freq_resolution':freq_res}

def measure_terminal_frequency(audio, fs, segment,**kwargs):
    '''
    
    See Also
    --------
    itsfm.get_terminal_frequency
    '''
    terminal_freq, threshold = get_terminal_frequency(audio, fs, **kwargs)
    return {'terminal_frequency':terminal_freq,
            'terminal_frequency_threshold':threshold}
    