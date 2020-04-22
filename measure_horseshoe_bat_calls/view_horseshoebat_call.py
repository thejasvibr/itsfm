# -*- coding: utf-8 -*-
"""Bunch of functions which help in visualising data 
and results

"""
import matplotlib.pyplot as plt
import numpy as np 

from measure_horseshoe_bat_calls.signal_processing import get_peak_frequency
from measure_horseshoe_bat_calls.signal_processing import moving_rms_edge_robust, dB
from measure_horseshoe_bat_calls.sanity_checks import make_sure_its_positive
from measure_horseshoe_bat_calls.frequency_tracking import accelaration
make_x_time = lambda X, fs: np.linspace(0, X.size/float(fs), X.size)

def check_call_background_segmentation(whole_call, fs, main_call_mask, 
                                                   **kwargs):
    '''Visualises the main call selection

    Parameters
    ----------
    whole_call : np.array
        Call audio
    fs : float>0
        Sampling rate in Hz
    main_call_mask : np.array
        Boolean array where True indicates the sample
        is part of the main call, and False that it is not. 
    
    Returns
    -------
    waveform, spec : pyplot.subplots
    
    Notes
    -----
    The appearance of the two subplots can be further changes by varying the 
    keyword arguments. For available keyword arguments see the visualise_call
    function. 
    '''
    peak_freq, _ = get_peak_frequency(whole_call, fs)
    horizontal_line = peak_freq*1.1

    waveform, spec = visualise_call(whole_call, fs, **kwargs)
    waveform.plot(make_x_time(main_call_mask, fs),
                  main_call_mask*np.max(whole_call),'k')
    waveform.plot(make_x_time(main_call_mask, fs),
                  main_call_mask*np.min(whole_call),'k')
    spec.plot(make_x_time(main_call_mask, fs),main_call_mask*horizontal_line,'k')
    return waveform, spec


def show_all_call_parts(only_call, call_parts, fs, **kwargs):
    '''
    Parameters
    ----------
    only_call : np.array
    call_parts : dictionary
        Dictionary with keys 'cf' and 'fm'
        The entry for 'cf' should only have one audio segment.
        The entry for 'fm' can have multiple audio segments. 
    fs : float>0
        Sampling rate in Hz. 
    
   
    Returns
    -------
    None
    
    Notes
    -----
    For further keyword arguments to customise the spectrograms 
    see documentation for make_specgram
    This function does not return any output, it only produces a 
    figure with subplots.   
    '''
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

def plot_cffm_segmentation(cf,fm,X,fs, **kwargs):
    w,s = visualise_call(X,fs, **kwargs)
    w.plot(make_x_time(cf, fs), cf*np.max(np.abs(X)),'k')
    w.plot(make_x_time(fm, fs), fm*np.max(np.abs(X)), 'r')
    s.plot(make_x_time(cf, fs), cf*fs*0.5, 'k',label='CF')
    s.plot(make_x_time(fm, fs), fm*fs*0.5, 'r',label='FM')
    plt.legend()
    return w,s

def plot_accelaration_profile(X,fs):
    '''
    Plots the frequency acclearation profile of a frequency
    profile
    
    Parameters
    ----------
    X : np.array
        The frequency profile with sample-level 
        estimates of frequency in Hz. 
    fs : float>0

    Returns
    -------
    A plt.plot which can be used as an independent figure ot
    a subplot. 
    '''
    acc_profile = accelaration(X,fs)
    t = np.linspace(0,X.size/fs, X.size)
    plt.figure()
    A = plt.subplot(111)
    plt.plot(t, acc_profile)
    plt.ylabel('Frequency accelaration, $\\frac{kHz}{ms^{2}}$')
    plt.xlabel('Time, s')
    return A


def plot_movingdbrms(X,fs,**kwargs):
    '''
    '''
    m_dbrms = dB(moving_rms_edge_robust(X, **kwargs))
    plt.plot(make_x_time(m_dbrms, fs), m_dbrms)
    

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
    '''
    '''

    fft_size = get_fftsize(fs, **kwargs)
    n_overlap = fft_size-1
    cmap = kwargs.get('cmap', 'viridis')
    vmin = kwargs.get('vmin', -100)

    specgram = plt.specgram(audio, Fs=fs, 
                               NFFT=fft_size,
                               noverlap=n_overlap,
                               vmin=vmin, 
                               cmap=cmap)
    plt.ylabel('Frequency, Hz')
    plt.xlabel('Time, s')
    return specgram

def make_waveform(audio, fs):
     plt.plot(make_x_time(audio,fs), audio)
    
def time_plot(X, fs):
    plt.plot(make_x_time(X,fs), X)

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
    