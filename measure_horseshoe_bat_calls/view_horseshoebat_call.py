# -*- coding: utf-8 -*-
"""Bunch of functions which help in visualising data 
and results
Created on Tue Mar 10 19:55:41 2020

@author: tbeleyur
"""
import matplotlib.pyplot as plt
import numpy as np 

from measure_horseshoe_bat_calls.measure_a_horseshoe_bat_call import get_fm_snippets

make_x_time = lambda X, fs: np.linspace(0, X.size/float(fs), X.size)

def check_call_background_segmentation(whole_segment, fs, fine_selection, 
                                                   **kwargs):
    '''
    '''
    waveform, spec = visualise_call(whole_segment, fs)
    waveform.plot(make_x_time(fine_selection, fs),fine_selection*np.max(whole_segment),'k')
    waveform.plot(make_x_time(fine_selection, fs),fine_selection*np.min(whole_segment),'k')
    spec.plot(make_x_time(fine_selection, fs),fine_selection*120000,'w')
    return waveform, spec

def check_call_parts_segmentation(only_call, fs, cf, fm,
                                      **kwargs):
    '''
    '''

    wavef, specg = visualise_call(only_call, fs)
    
    cf_time = np.argwhere(cf).flatten()/float(fs)
    wavef.vlines(np.array([np.min(cf_time), np.max(cf_time)]),
                         np.min(only_call), np.max(only_call), zorder=3)
    
    fm_types, fm_sweeps, fm_startstop = get_fm_snippets(only_call, fm, fs)
    
    for each in fm_startstop:
        wavef.vlines([each[0], each[1]],
                     np.max(only_call)*-0.5, np.max(only_call)*0.5, 
                     'r',zorder=3)

    specg.plot(make_x_time(cf, fs),cf*120000,'k',label='CF')
    specg.plot(make_x_time(fm, fs),fm*70000,'r',label='FM')
    plt.legend()
    
    return wavef, specg

def show_all_call_parts(only_call, call_parts, fs, **kwargs):
    plt.figure(figsize=(6,8))
    plt.subplot(421)
    make_specgram(only_call, fs, **kwargs);
    plt.subplot(423)
    make_specgram(call_parts['cf'], fs, **kwargs);
    
    plt.subplot(422);make_waveform(only_call, fs)
    plt.subplot(424);make_waveform(call_parts['cf'], fs)
    
    for i,each in enumerate(call_parts['fm']):
        try:
            plt.subplot(420+i*2+6);make_waveform(each, fs)
            plt.subplot(420+i*2+5);make_specgram(each, fs, **kwargs);
        except:
            pass


def visualise_call(audio, fs, **kwargs):
    '''
    Parameters
    ----------
    audio 
    fs 
    fft_size : integer>0, optional
    
    Returns
    -------
    a0, a1 : subplots
    '''
    

    plt.figure()
    a0 = plt.subplot(211)
    make_waveform(audio, fs)
    
    a1 = plt.subplot(212, sharex=a0)
    make_specgram(audio, fs, **kwargs)
    
    return a0, a1

def make_specgram(audio, fs, **kwargs):

    fft_size = get_fftsize(fs, **kwargs)
    n_overlap = fft_size-1
    cmap = kwargs.get('cmap', 'viridis')
    vmin = kwargs.get('vmin', -100)

    specgram = plt.specgram(audio, Fs=fs, 
                               NFFT=fft_size,
                               noverlap=n_overlap,
                               vmin=vmin, 
                               cmap=cmap)
    return specgram

def make_waveform(audio, fs):
     plt.plot(make_x_time(audio,fs), audio)
    

def get_fftsize(fs, **kwargs):
    '''
    '''
    fft_size_given = not(kwargs.get('fft_size') is None)
    freq_resolution_given = not(kwargs.get('freq_resolution') is None)
    both_not_given = [False, False] == [fft_size_given, freq_resolution_given]

    if freq_resolution_given:
        window_size = calculate_window_size(kwargs.get('freq_resolution'), fs)
        return window_size
    elif fft_size_given:
        return kwargs['fft_size']
    elif both_not_given:
        default_freq_resoln = 1000.0 # Hz
        window_size = calculate_window_size(default_freq_resoln, fs)
        return window_size

def calculate_window_size(freq_resoln, fs):
    return int(fs/freq_resoln)        
        



def make_overview_figure(call, fs,
                         measurements,
                         **kwargs):
    '''
    '''
    plt.figure()
    a0 = plt.subplot(111)
    specgram = make_specgram(call, fs, **kwargs);

    plot_fm_measurements(call, fs, measurements, a0, **kwargs)
    plot_cf_measurements(call, fs, measurements, a0, **kwargs)
    
    return a0

def plot_fm_measurements(call, fs, measures, subplot, **kwargs):
    # check if there's upfm 
    for fm in ['upfm_','downfm_']:
        try:
            fm_start, fm_stop =  measures[fm+'start'],  measures[fm+'end']
            subplot.vlines((fm_start, fm_stop),
                                    0,fs*0.5, 'r')
            subplot.hlines(measures[fm+'terminal_frequency'],
                           fm_start, fm_stop, 'b')
            
        except:
            pass
    return subplot

def plot_cf_measurements(call, fs, measures, subplot, **kwargs):
    peak_frequency = measures['peak_frequency']
    subplot.hlines(peak_frequency,measures['cf_start'],
                               measures['cf_end'], 'b')
    return subplot 
    