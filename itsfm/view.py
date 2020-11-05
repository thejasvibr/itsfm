# -*- coding: utf-8 -*-
"""Bunch of functions which help in visualising data 
and results

There is a common pattern in the naming of viewing functions. 

    #. functions starting with 'visualise' include an overlay of 
       a particular output attribute on top of or with the 
       the original signal. For example `visualise_sound`
    #. functions starting with 'plot' are bare bones 
       plots with just the attribute on the y and time on the x. 
"""
import matplotlib.pyplot as plt
import numpy as np 

from itsfm.signal_processing import get_peak_frequency
from itsfm.signal_processing import moving_rms_edge_robust, dB
from itsfm.frequency_tracking import accelaration, speed
make_x_time = lambda X, fs: np.linspace(0, X.size/float(fs), X.size)

class itsFMInspector:
    '''
    Handles the output from measure_and_segment calls, and allows plotting
    of the outputs. 
    
    Parameters
    ----------
    segmeasure_out : tuple
        Tuple object containing three other objects which are the output from segment_and_measure_call
        1. segmentation_output : tuple
            Tuple with the `cf` boolean array, `fm` boolean array and `info` dictioanry
        2. audio_parts : dictionary 
            Dictionary with call part labels and values as selected audio parts as np.arrays
        3. measurements : pd.DataFrame
            A wide-formate dataframe with one row referring to meaurements done on one call part
            eg. if a call has 3 parts (fm1, cf1, fm2), then there will be three columns and 
            N columns, if N measurements have been done. 

    whole_audio :  np.array
        The audio that was analysed. 
    
    fs : float>0
        Sampling rate in Hz. 

    Notes
    ----
    * Not all `visualise` methods may be supported. It depends on the segmentation method at hand. 
    * All `visualise` methods return one/multiple subplots that could be used and embellished further
      for your own custom laying over.
    
    '''
    def __init__(self, segmeasure_out, whole_audio, fs, **kwargs):
        self.seg_details, self.audio_parts, self.measurements = segmeasure_out
        self.whole_audio = whole_audio
        self.fs = fs
        self.kwargs = kwargs
        self.cf, self.fm, self.info = self.seg_details 
        
        
    def visualise_audio(self):
        w, s = visualise_sound(self.whole_audio, self.fs, **self.kwargs)
        return w,s 
    
    def visualise_fmrate(self):
        '''
        Plots the spectrogram + FM rate profile in a 2 row plot
        '''
        try:
            self.fmrate = self.info['fmrate']
            plt.figure()
            a = plt.subplot(311)
            make_waveform(self.fmrate, self.fs)
            plt.ylabel('FM rate, kHz/ms')
            b = plt.subplot(312,sharex=a)
            make_specgram(self.whole_audio, self.fs, **self.kwargs)
            b.set_ylabel('Frequency, Hz', labelpad=-1.5)
            c = plt.subplot(313,sharex=a)
            make_waveform(self.whole_audio, self.fs)
            return a, b, c
        except:
            raise AttributeError('Cannot make fmrate plot. Check if variable found in the output!')

    def visualise_accelaration(self):
        '''
        Plots the spectrogram + accelaration of the 
        frequency profile
        in a 2 row plot
        '''
        try:
            self.acc_profile = self.info['acc_profile']

            plt.figure()
            a = plt.subplot(311)
            make_waveform(self.acc_profile, self.fs)
            plt.ylabel('Accelaration, $kHz/ms^{2}$')
            b = plt.subplot(312, sharex=a)
            make_specgram(self.whole_audio, self.fs, **self.kwargs)
            c = plt.subplot(313, sharex=a)
            make_waveform(self.whole_audio, self.fs)
            return a, b, c
        except:
            raise AttributeError('Cannot make accelaration profile plot')

    def visualise_cffm_segmentation(self):
        '''
        '''
        w,s = visualise_cffm_segmentation(self.cf, self.fm, 
                                    self.whole_audio, self.fs,
                                   **self.kwargs)
        return w,s
    
    def visualise_frequency_profiles(self, fp_type='all'):
        '''
        Visualises either one or all of the frequency profiles that are present in the 
        info dictionary. 
        The function relies on picking up all keys in the info dictionary that end with '<>_fp'
        pattern. 
        
        Parameters
        ----------
        fp_type : str/list with str's
            Needs to correspond to a key found in the info dictionary 
        '''
        if fp_type=='all':
            all_fps = self._get_fp_keys(self.info)
        elif isinstance(fp_type, str):
            all_fps = [fp_type]
    
        plt.figure()
        a = plt.subplot(211)
        make_specgram(self.whole_audio, self.fs, **self.kwargs);
        time_axis = make_x_time(self.whole_audio, self.fs)
        for each_fp in all_fps:
            plt.plot(time_axis, self.info[each_fp], label=each_fp)
        plt.legend()
        b = plt.subplot(212, sharex=a)
        make_waveform(self.whole_audio, self.fs,)
        return a,b
    
    def visualise_geq_signallevel(self):
        '''
        Some tracking/segmentation methods rely on using only
        regions that are above a threshold, the `signal_level`
        . A moving dB rms window is pass
        
        ed, and only regions above it are 
        
        '''
        time_axis = make_x_time(self.whole_audio, self.fs)
        above_siglevel = np.zeros(self.whole_audio.size)
        plt.figure()
        a = plt.subplot(211)
        s = make_specgram(self.whole_audio, self.fs, **self.kwargs);
        ymin, ymax = a.get_ylim()
        for each in self.info['geq_signal_level']:
            above_siglevel[each] = 1 
        plt.plot(time_axis, above_siglevel*ymax*0.5, 
                         label='$\geq$ signal level',
                         color='C1')
        plt.legend()
        b = plt.subplot(212, sharex=a)
        make_waveform(self.whole_audio, self.fs)
        wave_max = np.max(np.abs(self.whole_audio))
        plt.plot(time_axis, above_siglevel*wave_max, label='$\geq$ signal level')
        return a, b
        
        
        
        
    
    def _get_fp_keys(self, info_dictionary):
        fp_keys = list(filter(lambda x : '_fp' in x ,info_dictionary.keys()))
        if len(fp_keys)==0:
            raise ValueError("There's no frequency profile (fp) in the output info. Check the output object or method")
        return fp_keys

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
    keyword arguments. For available keyword arguments see the visualise_sound
    function. 
    '''
    peak_freq, _ = get_peak_frequency(whole_call, fs)
    horizontal_line = peak_freq*1.1

    waveform, spec = visualise_sound(whole_call, fs, **kwargs)
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

def visualise_cffm_segmentation(cf,fm,X,fs, **kwargs):
    w,s = visualise_sound(X,fs, **kwargs)
    w.plot(make_x_time(cf, fs), cf*np.max(np.abs(X)),'k')
    w.plot(make_x_time(fm, fs), fm*np.max(np.abs(X)), 'r')
    s.plot(make_x_time(cf, fs), cf*fs*0.5, 'k',label='CF')
    s.plot(make_x_time(fm, fs), fm*fs*0.5, 'r',label='FM')
    plt.legend()
    return w,s

def visualise_fmrate_profile(X, freq_profile, fs):
    '''
    '''
    

def plot_fmrate_profile(X,fs):
    speed_profile = speed(X,fs)
    t = np.linspace(0,X.size/fs, X.size)
    plt.plot(t, speed_profile)
    plt.ylabel('Frequency modulation rate, $\\frac{kHz}{ms}$')
    plt.xlabel('Time, s')
    


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
    

def visualise_sound(audio, fs, **kwargs):
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
    
    plt.tight_layout()
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
                               cmap=cmap);
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
    