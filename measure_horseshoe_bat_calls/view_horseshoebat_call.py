# -*- coding: utf-8 -*-
"""Bunch of functions which help in visualising data 
and results
Created on Tue Mar 10 19:55:41 2020

@author: tbeleyur
"""
import matplotlib.pyplot as plt
import numpy as np 

make_x_time = lambda X, fs: np.linspace(0, X.size/float(fs), X.size)

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
    fft_size = kwargs.get('fft_size', 128)
    n_overlap = fft_size-1

    plt.figure()
    a0 = plt.subplot(211)
    plt.plot(make_x_time(audio,fs), audio)
    
    a1 = plt.subplot(212, sharex=a0)
    plt.specgram(audio, Fs=fs, 
                 NFFT=fft_size,
                 noverlap=n_overlap,
                 vmin=-100)
    return a0, a1



def make_overview_figure(audio, sounds, msmts, **kwargs):
    '''
    '''
    fs = kwargs['fs']
    fftsize = kwargs.get('fft_size', 128)
    dyn_range = kwargs.get('dyn_range', 80)
    

    dyn_vmin = 20*np.log10(np.max(abs(sounds[0]))) - dyn_range
    
    plt.figure(figsize=(8,7))
    plt.subplot(231)
    plt.title('Oscillogram')
    plt.plot(sounds[0], label='whole call')
    plt.plot(sounds[1], label='non CF')
    plt.legend()
    plt.subplot(232)
    plt.title('Manual selection')
    plt.specgram(audio, Fs=fs, NFFT=fftsize, noverlap=fftsize-1, vmin=dyn_vmin);
    plt.yticks([])
    plt.subplot(233)
    plt.title('Fine selection of call')
    plt.specgram(sounds[0], Fs=fs, NFFT=fftsize, noverlap=fftsize-1, vmin=dyn_vmin);
    plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
    # show CF peak frequency
    plt.hlines(msmts['peak_frequency'],0,sounds[0].size/float(fs), 'k', 
               linewidth=2.0)
    # show FM sections along with the terminal frequencies
    for fm_start, fm_end in [ ('downfm_start_time', 'downfm_end_time'), ('upfm_start_time','upfm_end_time')]:
        try:
            plt.vlines([msmts[fm_start],msmts[fm_end]], 50000, fs*0.5,'r' )
            terminal_frequency = fm_start.split('_')[0]+'_terminal_frequency'
            plt.hlines(msmts[terminal_frequency],msmts[fm_start],msmts[fm_end],'r')
           
        except:
            pass

    plt.subplot(234)
    plt.title('Non-CF call')
    plt.specgram(sounds[1], Fs=fs, NFFT=fftsize, noverlap=fftsize-1, vmin=dyn_vmin);
    plt.subplot(235)
    plt.title('First FM')
    # plot FM with 0.5*fftsize
    half_winsize = int(fftsize*0.5)
    plt.specgram(sounds[2], Fs=fs, NFFT=half_winsize, noverlap=half_winsize-1, vmin=dyn_vmin);
    plt.yticks([])

    try:
        plt.subplot(236)
        plt.title('Second FM')
        plt.specgram(sounds[3], Fs=fs, NFFT=half_winsize, noverlap=half_winsize-1, vmin=dyn_vmin);
        plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')

    except:
        pass
    