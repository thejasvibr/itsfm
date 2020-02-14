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
    '''Calculates the 20log of X'''
    return 20*np.log10(X)

def rms(X):
    return np.sqrt(np.mean(X**2.0))

def calc_energy(X):
    return np.sum(X**2.0)

def get_power_spectrum(audio, fs=250000.0):
    '''Calculates an RFFT of the audio.
    Parameters
    ------------
    audio : np.array
    fs : int
        Frequency of sampling in Hz

    Returns
    -------
    dB_power_spectrum : np.array
        dB(power_spectrum)
    freqs : np.array
        Centre frequencies of the RFFT. 
    '''
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(audio.size, 1.0/fs)
    dB_power_spectrum = dB(abs(spectrum))
    return dB_power_spectrum, freqs

def calc_sound_borders(audio, percentile=99):
    '''Gives the start and stop of a sound based on the percentile 
    cumulative energy values. 

    Parameters
    ----------
    audio : np.array
    percentile : float, optional
        Value between 100 and 0. The sound border is calcualted
        as the border which encapsulates the percentile of energy
        Defaults to 99.

    Returns
    --------
    start, end : int
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
    audio : np.array
    fs : float
        Frequency of sampling in Hz
    seg_length : int, optional
        The size of the FFt window used to calculate the moving FFT slices. 
        DEfaults to 256
    noverlap : int, optional 
        The number of samples overlapping between one FFT slice and the next. 
        Defaults to seg_length -1

    Returns
    --------
    peak_frequency : float
        Frequency with highest power in the audio in Hz. 
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
    target_frequency : float
        Frequency component to be filtered out from audio in Hz. 
    fs : float
        Sampling frequency in Hz. 
    q_factor : float, optional
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

    The CF part of a horseshoe bat call is not like a synthetised pure tone - it might have 
    a broader bandwidth. To effectively remove the CF part of a call lower may be better!
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
    '''A test function used to check how well the segmenting+measurement
    functions in the module work. 
    
    Parameters
    ----------
    call_durn : float
    fm_durn : float
    cf_freq : float
    fs : float
    call_shape : str
        One of either 'staplepin' OR 'rightangle'

    Returns
    --------
    cfcall : np.array
        The synthesised call. 

    Raises
    -------
    ValueError
        If a call_shape that is not  'staplepin' OR 'rightangle' is given

    Notes
    ------
    This is not really the besssst kind of CF call to test the functions on, 
    but it works okay. The CF call is made by using the poly spline function 
    and this leads to weird jumps in frequency especially around the CF-FM
    junctions. Longish calls with decently long FM parts look fine, but calls
    with very short FM parts lead to rippling of the frequency. 
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
    X :  np.array
        Signal of interest. 

    window_size : int, optional
                 Defaults to 125 samples. 

    Returns
    -------
    all_rms : np.array
        Moving rms of the signal. 
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
    3dB above threshold dB rms are considered the signal. 

    Parameters
    ----------
    audio : np.array
    window_size : int, optional 
                Window size for moving rms calcualtions. 
                Defaults to 125 samples. 
    background_percentile: float, optional
                A value between 0 and 100 indicating what percentile of the 
                audio is generally expected to be background. 
                Defaults to 50. 

    Returns
    -------
    start, stop : int
        Start and stop indices of the sound in the audio clip. 

    Raises
    ------
    BackgroundSegmentationError
        If there are no samples +3dB above the background_percentile 
        dB rms in the signal. This means the input signal is very uniform
        in dB rms, and there is either no signal, or very 
        little background in the audio. This issue can be solved by 
        choosing a more generous selection of the input audio clip. 

    Notes
    ------
    At this point, a +3dB threshold is implemented to keep things 
    simple. Future releases should look into making this a variable
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
        raise BackgroundSegmentationError('There is not enough background in the audio clip.\
                                          Please check if the whole audio has signal!,\
                                          or reduce the background_percentile  from current value.')


def measure_hbc_call(audio, **kwargs):
    '''
    
    Parameters
    ----------
    audio : np.array
    fs : float>0.
         Frequency of sampling in Hz.

    Returns
    --------   
    measurements : pd.DataFrame 
        With the following columns and entries
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
    call_rms = rms(call)
    
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
    fm_segments, fm_times = get_FM_parts(call_no_CF, **kwargs)
    try:
        fm_terminal_freqs = [get_terminal_frequency(eachfm, **kwargs) for eachfm in fm_segments]
        fm_energy = [ calc_energy(eachfm) for eachfm in fm_segments]
        fm_rms = [ rms(eachfm) for eachfm in fm_segments]
    except:
        fm_terminal_freqs = [np.nan, np.nan]
        fm_energy = [np.nan, np.nan]
        fm_rms = [np.nan, np.nan]

    measurements = assemble_all_measurements(call_duration, call_energy, call_rms, 
                                             CF_energy,peak_frequency,
                                             fm_energy, fm_times, fm_terminal_freqs,
                                             fm_rms, **kwargs)
   
    sound_segments = assemble_all_sound_segments(call, call_no_CF, fm_segments)

    return sound_segments, measurements

def assemble_all_measurements(call_duration, call_energy, call_rms, CF_energy, 
                              peak_frequency, fm_energy, fm_times, fm_terminal_freqs,
                              fm_rms, **kwargs):
    '''
    '''
    msmts = {'call_duration':[call_duration],
            'call_energy':[call_energy],
            'call_rms':[call_rms],
            'cf_energy':[CF_energy],
            'peak_frequency':[peak_frequency],
            }

    for eachfm_time, eachfm_tf, eachfm_engy, eachfm_rms in zip(fm_times, fm_terminal_freqs, fm_energy, 
                                                       fm_rms):
        start_of_fm = eachfm_time[0]/float(kwargs['fs']) 
        end_of_fm = eachfm_time[1]/float(kwargs['fs']) 
        # whetehr upfm or downfm
        fm_type = which_fm_type(end_of_fm, call_duration)

        msmts[fm_type+'energy'] = [eachfm_engy]
        msmts[fm_type+'terminal_frequency'] = [eachfm_tf]
        msmts[fm_type+'start_time'] = [start_of_fm]
        msmts[fm_type+'end_time'] = [end_of_fm]
        msmts[fm_type+'rms'] = [eachfm_rms]

    msmts['cf_duration'] = infer_cf_duration(call_duration, fm_times, **kwargs)
    msmts['cf_rms'] = infer_cf_rms(CF_energy, msmts['cf_duration'], **kwargs)
    measurements = pd.DataFrame(data=msmts)
    return measurements

def infer_cf_duration(call_duration, fm_times,**kwargs):
    '''This function works by assuming anythin that is not fm 
    in the call is cf. 

    .. math::
    
        Duration_{CF call} = Duration_{whole call} - (Duration_{upFM}+ Duration_{downFM})
    
    This approach is necessary because it is easier to eliminate the CF part and find the
    FM portions, than to eliminate the FM portions and find the CF parts. 

    Parameters
    -----------
    call_duration : float
        Duration of whole call
    fm_times : list
        List with a pair of start and stop indices for each detected FM portion
        fm_times can also be an empty list. 
    fs : float
        Frequency of sampling in Hz

    Returns
    --------
    cf_duration : float
        Duration of CF part. See description above for how it is calcualted.     
    '''
    fm_durations = [ (stop-start)/float(kwargs['fs']) for (start, stop) in fm_times]
    cf_duration = call_duration - np.sum(fm_durations)
    return cf_duration

def infer_cf_rms(cf_energy, cf_duration, **kwargs):
    '''Calculates the CF portions rms using information on the amount of energy
    and the duration of the CF segment. 
    
    .. math::

        CF_{rms} = \sqrt{\\frac{CF \;energy}{sampling \;rate \\times CF_{duration}}}
    
    Parameters
    -----------
    cf_energy:
        CF energy, which is the sum of all squared values of the CF
    cf_duration : float
        CF duration in seconds
    fs : float
        Frequency of sampling in Hz
    
    '''
    cf_rms = np.sqrt(cf_energy/(cf_duration*kwargs['fs']))
    if cf_rms > 1:
        raise ValueError('The CF rms is > 1 - pelase check the energy or the duration given')
    return cf_rms
    

def which_fm_type(end_of_fm, call_duration)        :
    '''figures out whether its an up or down fm '''
    if  end_of_fm <= call_duration*0.5:
        fm_type = 'upfm_'
    elif end_of_fm > call_duration*0.5:
        fm_type = 'downfm_'
    return fm_type


def assemble_all_sound_segments(call, nocf, fm_segments):
    '''
    '''
    all_sounds = [call, nocf]
    for eachfm in fm_segments:
        all_sounds.append(eachfm)
    return all_sounds


def get_FM_parts(no_cf_call,**kwargs):
    '''Function which identifies and segments out FM parts. If there are no FM parts detected, 
    then an empty list is given out. 
    
    Parameters
    ---------
    no_cf_call : np.array
        Audio segment with no Cf-frequency filtered out of it.     
    fs : float>0
        Frequency of sampling in Hertz. 
    min_fm_duration: float>0, optional
        The lowest duration the FM part of a call can have. 
        Defaults to 0.001 seconds.
    min_cf_duration : float>0, optional
        The minimum duration of the CF poertion in a call. 
        This duration is used to set the minimum distance
        between two detected FM peaks in the no_cf_call. 
        Defaults to 0.010 seconds.                 
    percentile_threshold: 100>float>0, optional
        This sets the baseline rms threshold of the no_cf_call. 
        By default the 75th percentile value of the no_cf_call's 
        moving dB rms is used. 
        Defaults to 75.        
    dB_above_threshold : float, optional 
        Adds dB_above_threshold dB's to the percentile threshold dB rms
        Defaults to +3dB
    fm_max_durn : float>0, optional 
        Maximum possible duration of an FM part. 
        Defaults to 0.005 seconds. 

    Returns
    --------
    fm_segments : list
        The segmented fm part/s as np.arrays. fm_parts can be empty 
        if no fm parts were detected, have one or two fm segments in ti. 
    fm_times : list
        The start and stop indices of the segmented fm parts. The indices
        are given assuming the audio is the same size as no_cf_call.

    Raises
    -------
    FMIdentificationError
        When the FM segmentation isn't as clean as expected.Raised when there 
        are >2 peaks detected in the no_cf_call. This is known to occur when
        the no_cf_call is poorly filtered, thus leaving a lot of energy
        in the CF part of the call. 
    '''
    fs = kwargs['fs']
    min_fm_duration = int(kwargs.get('min_fm_duration', 0.001)*fs)
    percentile_threshold = kwargs.get('percentile_threshold', 75)
    dB_above_threshold = kwargs.get('dB_above_threshold', 3)
    min_cf_duration = kwargs.get('min_cf_duration', 0.01)
    fm_max_durn = kwargs.get('fm_max_durn', 0.005)
    half_fm_samples = int(fm_max_durn*0.5*fs)
    
    
    # calcualte running rms and a pessimistic threshold to segment FM portions
    rms_over_call = moving_rms(no_cf_call, windowsize=min_fm_duration)
    dB_rms_over_call = dB(rms_over_call)
    threshold = np.percentile(dB_rms_over_call, percentile_threshold)
    pessimistic_threshold = threshold + dB_above_threshold
    
    # get samples aboe the pessimistic threshold
    samples_above_threshold = dB_rms_over_call >= pessimistic_threshold
    signal_above_threshold = np.zeros(dB_rms_over_call.shape)
    signal_above_threshold[samples_above_threshold] = rms_over_call[samples_above_threshold]
    
    # do peak detection on the parts that are above 
    fm_peaks = peak.indexes(signal_above_threshold.flatten(), thres=0.1, min_dist=int(min_cf_duration*fs))
    # if >2 peaks raise an error
    check_number_of_fmpeaks(fm_peaks)

    # segment and check duration of fm. If it's too short discard it. 
    fm_segments = []
    fm_times =[]
    for each_peak in fm_peaks:
        broad_fm_section, audio_startstop = take_segment_around_peak(no_cf_call, each_peak, half_fm_samples)
        fm_start,fm_stop = segment_sound_from_background(broad_fm_section,**kwargs)
        fm_duration = sound_duration(fm_start,fm_stop, **kwargs)
        if fm_duration >= min_fm_duration:
            well_segmented_fm = broad_fm_section[fm_start:fm_stop+1]
            fm_segments.append(well_segmented_fm)
            start_stop = [fm_start + audio_startstop[0], fm_start + audio_startstop[0] + well_segmented_fm.size]
            fm_times.append(start_stop)
    
    
    return fm_segments, fm_times

def check_number_of_fmpeaks(fmpeaks):
    if len(fmpeaks)>2:
        print(fmpeaks)
        raise FMIdentificationError('Too many FM peaks detected, aborting call processing..')

def sound_duration(start_sample, end_sample, **kwargs):
    '''
    Keyword Arguments
    -----------------
    fs 
    '''
    num_samples = end_sample-start_sample+1
    duration = num_samples*kwargs['fs']
    return duration



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
    fs : float>0
        Sampling rate in Hz
    threshold : float, optional
        The terminal frequency is calculated based on finding the level of the peak frequency
        and choosing the lowest frequency which is -10 dB (20log10) below the peak level. 
        Defaults to -10 dB

    Returns 
    ---------
    terminal_frequency       

    Notes
    -----
    Careful about setting threshold too low - it might lead to output of terminal
    frequencies that are actually in the noise, and not part of the signal itself. 
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
    
class FMIdentificationError(ValueError):
    pass

class BackgroundSegmentationError(ValueError):
    pass
    
    
