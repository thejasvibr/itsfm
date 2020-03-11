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