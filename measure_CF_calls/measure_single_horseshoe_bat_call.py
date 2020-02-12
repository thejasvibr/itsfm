#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Module that segments and measures
parts of a horseshoebat CF call

Created on Sun Feb  9 18:11:16 2020

@author: tbeleyur
"""
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize']
import numpy as np
import pandas as pd
from peakutils import peak
import scipy.signal as signal
import soundfile as sf

def dB(X):
    return 20*np.log10(X)

def rms(X):
    return np.sqrt(np.mean(X**2.0))

def calc_energy(X):
    return np.sum(X**2.0)

def get_power_spectrum(audio, fs=250000.0):
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(audio.size, 1.0/fs)
    return dB(abs(spectrum)), freqs

def calc_sound_borders(audio, percentile=99):
    '''Gives the start and stop of a sound based on the percentile 
    cumulative energy values. 
    
    Returns
    --------
    start, end : int.
    '''
    audio_sq = audio**2.0
    cum_energy = np.cumsum(audio_sq)
    outside_percentile = (100-percentile)*0.5
    lower, higher = outside_percentile, 100-outside_percentile
    start, end = np.percentile(cum_energy,[lower, higher])
    start_ind = np.argmin(abs(cum_energy-start))
    end_ind = np.argmin(abs(cum_energy-end))
    return start_ind, end_ind

def get_robust_peak_frequency(audio, **kwargs):
    '''Makes a spectrogram from the audio 
    and calcualtes the peak frequency by averaging
    each slice of the spectrogram's FFT's. 
    
    This 'smooths' out the structure of the power 
    spectrum and allows a single and clear peak 
    detection. 
    
    Thanks to Holger Goerlitz for the suggestion. 
    
    Parameters
    ----------
    
    Keyword Arguments
    ------------------
    fs
    
    NFFT
    
    noverlap
    
    
    Returns
    --------
    
    
    
    '''
    seg_length = kwargs.get('seg_length',256)
    frequency,t,sxx = signal.spectrogram(audio, fs=int(kwargs['fs']), nperseg=seg_length, noverlap=seg_length-1)
    averaged_spectrogram = np.apply_along_axis(np.sum, 1, sxx)
    peak = np.argmax(averaged_spectrogram)
    peak_frequency = frequency[peak]
    return peak_frequency
    
def remove_frequency(audio, target_frequency, **kwargs):
    '''
    Parameters
    -----------
    audio : np.array

    target_frequency : float>0
          Frequency component to be filtered out from audio in Hz. 

    fs : float>0
          Sampling frequency in Hz. 

    Keyword Arguments
    ----------------
    q_factor : float >0
              q_factor of the notch filter. See signal.iirnotch for more details. 
              Defaults to 1. 
    
    Returns
    ----------
    filtered_audio : np.array

    
    Notes
    ------
    It may seem like setting a higher q_factor is better to remove the Cf. This is not true in 
    my experience. A higher q-factor leads to a very narrow filtering, which can leave a lot of 
    energy remaining in the CF part - even though the target_frequency per se is filtered out. 

    To effectively remove the CF part of a call lower may be better!
    '''
    nyquist = kwargs['fs']*0.5
    q_factor = kwargs.get('q_factor',1) 
    b,a = signal.iirnotch(target_frequency/nyquist,q_factor)
    
    filtered_audio = signal.lfilter(b,a,audio)
    return filtered_audio

    
def get_peak_frequency(audio, fs=250000):
    '''
    '''
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(audio.size, 1.0/fs)
    peak_freq = freqs[np.argmax(spectrum)]
    return peak_freq


## from the make_CF_training_data module
def make_one_CFcall(call_durn, fm_durn, cf_freq, fs, call_shape):
    '''
      
    TODO : make harmonics
    '''
    # choose an Fm start/end fr equency :
    FM_bandwidth= xrange(5,25)
    fm_bw = np.random.choice(FM_bandwidth, 1)*10.0**3
    start_f = cf_freq - fm_bw
    # 
    polynomial_num = 25
    t = np.linspace(0, call_durn, int(call_durn*fs))
    # define the transition points in the staplepin
    freqs = np.tile(cf_freq, t.size)
    numfm_samples = int(fs*fm_durn)
    if call_shape == 'staplepin':       
        freqs[:numfm_samples] = np.linspace(start_f,cf_freq,numfm_samples, endpoint=True)
        freqs[-numfm_samples:] = np.linspace(cf_freq,start_f,numfm_samples, endpoint=True)
        p = np.polyfit(t, freqs, polynomial_num)

    elif call_shape == 'rightangle':
        # alternate between rising and falling right angle shapes
        rightangle_type = np.random.choice(['rising','falling'],1)
        if rightangle_type == 'rising':
            freqs[:numfm_samples] = np.linspace(cf_freq,start_f,numfm_samples, endpoint=True)
        elif rightangle_type == 'falling':
            freqs[-numfm_samples:] = np.linspace(cf_freq,start_f,numfm_samples, endpoint=True)
        p = np.polyfit(t, freqs, polynomial_num)

    else: 
        raise ValueError('Wrong input given')
      
    cfcall = signal.sweep_poly(t, p)

    #windowing = np.random.choice(['hann', 'nuttall', 'bartlett','boxcar'], 1)[0]
    windowing= 'boxcar'
    cfcall *= signal.get_window(windowing, cfcall.size)
    return(cfcall)


def moving_rms(X, **kwargs):
    '''Calculates moving rms of a signal with given window size. 
    Outputs np.array of *same* size as X. The rms of the 
    last few samples <= window_size away from the end are assigned
    to last full-window rms calculated

    Parameters
    ----------
    X :  np.array of Nsample values

    Keyword Arguments
    -----------------    
    window_size : int>0
                 Defaults to 125 samples. 


    Returns
    -------
    all_rms : np.array of Nsamples values.
    
    '''
    window_size = kwargs.get('window_size', 125)
    starts = np.arange(0, X.size)
    stops = starts+window_size
    valid = stops<X.size
    valid_starts = np.int32(starts[valid])
    valid_stops = np.int32(stops[valid])
    all_rms = np.ones(X.size).reshape(-1,1)

    for i, (start, stop) in enumerate(zip(valid_starts, valid_stops)):
        rms_value = rms(X[start:stop])
        all_rms[i] = rms_value
    
    # replace all un-assigned samples with the last rms value
    all_rms[all_rms==1] = rms_value

    return all_rms

def segment_sound_from_background(audio, **kwargs):
    '''Selects the sound from an audio clip when the majority of the 
    clip is background, and the sound of interest only ocuppies a small
    region of it. 

    Here a 95% energy window (as used in calc_sound_borders) will fail as it will lead 
    to extremely generous start-stop windows - and include a lot of the background. 
    
    The moving dB rms of the audio clip is calcualted and portions of the audio clip
    with above threshold dB rms are considered the signal. 

    Keyword Arguments
    -------------------
    window_size : int>0
                Window size for moving rms calcualtions. 
                Defaults to 125 samples. 

    background_percentile: 100 > float > 0
                How much of the audio is generally expected to 

    
    Returns
    ----------
    
  
    
    '''
    signal_moving_rms = moving_rms(audio, **kwargs)

    moving_dBrms = np.around(dB(signal_moving_rms))
    moving_dBrms -= np.max(moving_dBrms)

    background_percentile = kwargs.get('background_percentile', 50)
    threshold = np.percentile(moving_dBrms.flatten(), background_percentile) + 3 
    
    try:
        above_threshold = np.where(moving_dBrms>threshold)[0]
        start, stop = np.min(above_threshold), np.max(above_threshold)
        return start, stop
    except:
        raise FMIdentificationError('There is not enough background in the audio clip. Please check if \
        the whole audio has signal!,or reduce the background_percentile  from current value.')


def measure_hbc_call(audio, **kwargs):
    '''
    
    Parameters
    ----------
    audio : np.array


    Keyword Arguments
    ---------
    fs : float>0.
         sampling rate in Hz.



    Returns
    --------
    
    measurements : pd.DataFrame with the following columns and entries
        call_duration : float>0
                        The duration of the entire call

        call_energy : float>0
                    Eenrgy in the whole call.


        FM_energy : float>0
                    The energy of the FM portion/s. This is the energy in the call 
                    post CF removal. 

        CF_energy : float>0
                    Energy in the CF component. This is calculated by subtracting the 
                    whole calls energy from the energy remaining post CF removal.

        CF_duration : float>0
                      Duration of whole call in seconds. 

        FM_duration : array, list-like
                      Duration of the FM portions. if there are up and down FM sweeps, then 
                      two values are returned. 


        peak_CF : float>0
                  The peak CF in Hz

        FM_terminal_-10dB : float>0
                        The terminal frequency of the FM. The Cf component of a call 
                        is notch filtered out, and the FM part/s are left. The
                        FM parts are then isolated and their power spectrum is calculated. 
                        The terminal frequency is the lowest frequency that is -10dB
                        from the peak of the power spectrum.

        FM_times : nFM x 2 np.array
                    Start and end times for the detected FM sweep/s. 
                    One row is one FM sweep. 

        CF_times : 1x2 np.array
                   1 entry with start and end time. 

    sound_segments : 
        call : np.array. 
               A slightly narrower selection of the audio with 99% of the audio's energy 
               in it. 

        nocf_call : np.array. 
                The call without the CF segment in it. 

        fm_segments: np.array
                One or two extracted FM segments. 


    '''
    # narrow down the audio clip even more to extract exactly the sound
    call_window = calc_sound_borders(audio, 95) 
    call = audio[call_window[0]:call_window[1]]
    
    # call duration of the entire call
    call_duration = call.size/float(kwargs['fs'])
    #energy of the entire call
    call_energy = calc_energy(call)

    #identify peak frequency 
    peak_frequency = get_robust_peak_frequency(call, **kwargs)

    #get rid of peak frequency
    call_no_CF = remove_frequency(call, peak_frequency, **kwargs)

    # calculate energy of CF
    non_cf_energy = calc_energy(call_no_CF)
    CF_energy = call_energy - non_cf_energy

    # extract FM sections and measure things about them
    fm_segments, fm_times = identify_FM(call_no_CF, **kwargs)
    fm_terminal_freqs = [get_terminal_frequency(eachfm, **kwargs) for eachfm in fm_segments]
    fm_energy = [ calc_energy(eachfm) for eachfm in fm_segments]
        
    measurements = assemble_all_measurements(call_duration, call_energy, CF_energy, fm_energy,
                                             peak_frequency, fm_times, fm_terminal_freqs, **kwargs)
    
    sound_segments = assemble_all_sound_segments(call, call_no_CF, fm_segments)

    return sound_segments, measurements

def assemble_all_measurements(call_duration, call_energy, CF_energy, fm_energy,
                                             peak_frequency, fm_times, fm_terminal_freqs, **kwargs):
    '''
    '''
    msmts = {'call_duration':[call_duration],
            'call_energy':[call_energy],
            'cf_energy':[CF_energy],
            'peak_frequency':[peak_frequency],
            }

    for fm_time, fm_tf, fm_engy in zip(fm_times, fm_terminal_freqs, fm_energy):
        start_of_fm = fm_time[0]/float(kwargs['fs']) 
        end_of_fm = fm_time[1]/float(kwargs['fs']) 
        
        if  end_of_fm <= call_duration*0.5:
            fm_type = 'upfm_'
        elif end_of_fm > call_duration*0.5:
            fm_type = 'downfm_'
        
        msmts[fm_type+'energy'] = [fm_engy]
        msmts[fm_type+'terminal_frequency'] = [fm_tf]
        msmts[fm_type+'start_time'] = [start_of_fm]
        msmts[fm_type+'end_time'] = [end_of_fm]
        
    measurements = pd.DataFrame(data=msmts)
    return measurements
        

def assemble_all_sound_segments(call, nocf, fm_segments):
    '''
    '''
    all_sounds = [call, nocf]
    for eachfm in fm_segments:
        all_sounds.append(eachfm)
    return all_sounds
    

def identify_FM(no_CF_call, **kwargs):
    '''
    
    Keyword Arguments
    -------------------
    fs 
    fm_energy_percentile
    smoothing_durn
    min_cf_duration
    relative_threshold
    fm_max_durn
    '''
    fs = kwargs['fs']
    fm_energy_percentile = kwargs.get('fm_energy_percentile', 95)
    smoothing_length = int(kwargs.get('smoothing_durn', 0.002)*fs)
    smoothed_nocf = np.convolve(abs(no_CF_call), np.ones(smoothing_length),'same')
    
    min_cf_duration = kwargs.get('min_cf_duration', 0.01)
    relative_threshold = kwargs.get('relative_threshold', 0.5)
    peaks = peak.indexes(smoothed_nocf,thres=relative_threshold, min_dist=int(min_cf_duration*fs))
    
    if len(peaks)>2:
        print(peaks)
        raise FMIdentificationError('Too many FM peaks detected')

    fm_max_durn = kwargs.get('fm_max_durn', 0.005)
    half_fm_samples = int(fm_max_durn*0.5*fs)
    
    fm_segments = []
    fm_times =[]
    for each_peak in peaks:
        broad_fm_section, audio_startstop = take_segment_around_peak(no_CF_call, each_peak, half_fm_samples)
        fm_start,fm_stop = segment_sound_from_background(broad_fm_section,**kwargs)
        fm = broad_fm_section[fm_start:fm_stop]
        fm_segments.append(fm)
        start_stop = [fm_start + audio_startstop[0], fm_start + audio_startstop[0] + fm.size]
        fm_times.append(start_stop)
    return fm_segments, fm_times


def take_segment_around_peak(audio, peak, samples_LR_of_peak):
    '''
    '''
    start = peak-samples_LR_of_peak
    stop = peak+samples_LR_of_peak
    
    if start <0:
        start = 0
    
    if stop > audio.size:
        stop = audio.size-1
    
    segment_around_peak = audio[start:stop]
    return segment_around_peak, [start,stop]

def get_terminal_frequency(audio, **kwargs):
    '''Gives the -XdB frequency from the peak. 

    The power spectrum is calculated and smoothened over 3 frequency bands to remove
    complex comb-like structures. 
    
    Then the lowest frequency below XdB from the peak is returned. 

    Parameters
    ----------
    audio : np.array

    Keyword Arguments
    --------------------
    
    fs : float>0
        Sampling rate in Hz

    threshold : float < 0
        The terminal frequency is calculated based on finding the level of the peak frequency
        and choosing the lowest frequency which is -10 dB (20log10) below the peak level. 
        Defaults to -10 dB

    Returns 
    ---------
    terminal_frequency       
    

    '''
    
    threshold = kwargs.get('threshold', -10)
    
    power_spectrum, freqs,  = get_power_spectrum(audio, kwargs['fs'])
    # smooth the power spectrum over 3 frequency bands to remove 'comb'-iness in the spectrum
    smooth_spectrum = np.convolve(10**(power_spectrum/20.0), np.ones(3)/3,'same')
    smooth_power_spectrum = dB(abs(smooth_spectrum))

    peak = np.max(smooth_power_spectrum)
    geq_threshold = smooth_power_spectrum >= peak + threshold
    all_frequencies_above_threshold = freqs[geq_threshold]

    terminal_frequency = np.min(all_frequencies_above_threshold)
    return terminal_frequency
  
def make_overview_figure(audio, sounds, msmts, **kwargs):
    '''
    '''
    fs = kwargs['fs']
    fftsize = kwargs.get('fftsize', 128)
    dyn_range = kwargs.get('dyn_range', 80)
    

    dyn_vmin = 20*np.log10(np.max(abs(sounds[0]))) - dyn_range
    
    plt.figure(figsize=(8,7))
    a0 = plt.subplot(231)
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
    
class FMIdentificationError(ValueError):
    pass
    
    
