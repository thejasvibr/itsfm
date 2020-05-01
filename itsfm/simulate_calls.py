# -*- coding: utf-8 -*-
""" The 'simulate_calls' module has functions which simulate CF-FM calls 
with parameters of choice. Let's say we want to make a CF-FM call with the
following parameters. 

* CF portion at 100kHz and of 10ms length. 
* up FM starting at 80kHz of 2ms
* down FM end at 60kHz of 3ms

The code snippet below recreates the call with the above parameters

.. code-block:: python

    from itsfm.view_horseshoebat_call import visualise_call
    from itsfm.simulate_calls import make_cffm_call

    call_parameters = {'cf':(100000, 0.01),
                        'upfm':(80000, 0.002),
                        'downfm':(60000, 0.003),
                        }
    
    fs = 500*10**3 # 500kHz sampling rate 
    synthetic_call, freq_profile = make_cffm_call(call_parameters, fs) 
    
    # plot 
    wavef, spec = visualise_call(synthetic_call, fs, fft_size=512)
    spec.set_ylim(0,125000)

Note
----
The 'make_cffm_call' makes simulated calls  which match actual bat calls
in all relevant aspects pretty well (temporal+spectral).
However, there are still some issues to be sorted - for example, the 
level of the CF portion of the signal is always a little bit lower. If you 
have any suggestions for that it'd be great to hear. See :func:`make_cffm_call`,
and :func:`make_call_frequency_profile`  and :func:`make_FM_with_joint` for more details.
"""
import numpy as np
import scipy.signal as signal 

def make_cffm_call(call_properties, fs, **kwargs):
    '''
    Parameters
    ----------
    call_properties : dictionary
        With keys corresponding to the upto 3 components
        cf, upfm, downfm
        See make_call_frequency_profile for further info. 
    fs : float>0
        sampling rate in Hz
    poly_order : int, optional
        see make_cffm_joint_profile
    joint_duration : float, optional
        see make_cffm_joint_profile

    Returns
    -------
    call, frequency_profile : np.array
        The audio and the final frequency profile. 

    See Also
    --------
    make_call_frequency_profile

    References
    ----------
    ..1 Thanks to Paul Panzer's SO example code for forming the main basis for this function.
         https://stackoverflow.com/questions/47664360/list-of-frequencies-in-time-to-signal-in-python
    '''
    call_frequency_profile =  make_call_frequency_profile(call_properties, 
                                                               fs,
                                                               **kwargs)
    
    dt = 1.0/fs
    call = np.sin(2*np.pi*dt*np.cumsum(call_frequency_profile)) 
    return call, call_frequency_profile



def make_fm_chirp(start_f, end_f, durn, fs, chirp_type='linear'):
    t = np.linspace(0,durn, int(fs*durn))
    chirp = signal.chirp(t, start_f, t[-1], end_f, method=chirp_type)
    chirp *= signal.tukey(chirp.size, 0.05)
    return chirp

def make_tone(tone_freq, durn, fs):
    t = np.linspace(0,durn, int(fs*durn))
    tone = np.sin(2*np.pi*tone_freq*t)
    tone *= signal.tukey(tone.size, 0.05)
    return tone

def silence(durn, fs):
    return np.zeros(int(fs*durn))

def add_noise(sound, dBrms):
    sound += np.random.normal(0,10**(dBrms/20.0),sound.size)
    return sound



def make_call_zoo(**kwargs):
    '''
    Makes a range of test sounds with known properties across a range of 
    the sampling rate.

    The sound durations 
    
    Parameters
    ----------
    fs : float>0, optinoal 
    freq_range : np.array, optional
    gap : float>0, optional 
    sweep_types : list with strings, optional 
    make_birdish : boolean, optional 
    
    Returns 
    -------
    freq_profile, audio : np.array

    '''
    fs = kwargs.get('fs', 44100)
    sound_durns = kwargs.get('sound_durns', np.array([0.003, 0.01, 0.1]))
    freq_range = kwargs.get('freq_range',  np.array([0.1, 0.25, 0.45]))
    gap = kwargs.get('gap', 0.01)*0.5

    gap_sound = silence(gap, fs)

    audio = []
    audio_fp = []

    for tone_f in freq_range*fs:
        for durn in sound_durns:
            tone_fp = np.tile(tone_f, int(fs*durn))
            actual_tone = make_tone(tone_f, durn, fs)
            
            audio_fp.append(sandwich_between(gap_sound, tone_fp))
            audio.append( sandwich_between(gap_sound, actual_tone))

    sweep_types = kwargs.get('sweep_types', ['linear','log','hyperbolic'])
    
    chirp_durn = np.min(sound_durns)
    t = np.linspace(0, chirp_durn, int(fs*chirp_durn))
    start_f, end_f = np.min(freq_range)*fs, np.max(freq_range)*fs
    for shape in sweep_types:
        chirp = make_fm_chirp(start_f, end_f, chirp_durn,fs,
                                                  shape)
        chirp_fp = make_sweep_fp([start_f, end_f], t, shape)
        
        audio.append(sandwich_between(gap_sound, chirp))
        audio_fp.append(sandwich_between(gap_sound, chirp_fp))
    
    
    
    
    if kwargs.get('make_birdish', True):
    
        
        cf = np.tile(np.mean(freq_range)*fs, 2*t.size)    
        upfm1 = np.linspace(np.min(freq_range)*fs, cf[0], cf.size)
        upfm2 = np.linspace(cf[-1], np.max(freq_range)*fs, cf.size)
        
        birdish_fp = np.concatenate((upfm1, cf, upfm2))
        birdish_cs_fp = np.cumsum(birdish_fp)
        t_bird = np.linspace(0, birdish_fp.size/float(fs), birdish_cs_fp.size)
        birdish_sound = np.sin(2*np.pi*birdish_cs_fp*t_bird)
        birdish_sound[:10] *= signal.hann(20)[:10]
        birdish_sound[-10:] *= signal.hann(20)[-10:]
        
        audio.append(sandwich_between(gap_sound, birdish_sound))
        audio_fp.append(sandwich_between(gap_sound, birdish_fp))
    
    return np.concatenate((audio_fp)).flatten(), np.concatenate((audio)).flatten()
        
        
def sandwich_between(bread, cheese):
    
    return np.concatenate((bread, cheese, bread))
    

def make_sweep_fp(freqs, t, sweep_type):
    '''
    making the sweep frequency profile of the scipy.signal.chirp types
    '''
    f0,f1 = freqs
    t1 = t[-1]

    if sweep_type=='hyperbolic':
        f_t = f0*f1*t1 / ((f0 - f1)*t + f1*t1)
    elif sweep_type=='log':
        f_t = f0 * (f1/f0)**(t/t1)
    elif sweep_type=='linear':
        f_t = f0 + (f1 - f0) * t / t1
    else:
        raise NotImplementedError('The sweep type "%s" has not been implemented in the simulated calls..please check again'%sweep_type)
    return f_t

def make_call_frequency_profile(call_properties, fs, **kwargs):
    ''' 
    Makes the call frequency profile for a CF-FM call.
    
    Parameters
    ----------
    call_properties : dictionary
        With keys : 'cf', 'upfm', 'downfm'
        Each key has a tuple entry with a frequency and a duration value

    fs : float
        Sampling rate in Hz
    
    Returns
    --------
    final_frequency_profile : np.array
        The call frequency profile.
    
    See Also
    --------
    make_FM_with_joint
    
    Example
    --------
    This corresponds to a call with an upfm starting at 50kHz of 5ms
    and a CF at 100kHz of 10ms, followed by a downfm ending at 20kHz of 3ms. 

    >>> cp = {'cf':(100000, 0.01),
              'upfm':{50000, 0.005},
              'downfm':{20000, 0.003}}        
    >>> fs = 500000
    >>> call_freq_profile = make_call_frequency_profile(cp, fs)
    '''
    cf_freq, cf_durn = call_properties['cf']

    double_fm_call = np.all([call_properties.get('upfm')!=None, 
                             call_properties.get('downfm')!=None,])
    if double_fm_call:
        upfm_freq_profile, joint_durn = make_FM_with_joint(call_properties['upfm'],
                                               cf_freq,fs,
                                                           **kwargs)
        upfm_freq_profile = upfm_freq_profile[::-1]
        downfm_freq_profile, joint_durn = make_FM_with_joint(call_properties['downfm'],
                                               cf_freq, fs,
                                               **kwargs)

        effective_cf_durn = cf_durn - 2*joint_durn
        cf_time_effective = np.linspace(0,effective_cf_durn,1000)
        rough_cf_freq_profile = np.concatenate( (np.array([upfm_freq_profile[-1]]),
                                                 np.tile(cf_freq, 998),
                                                 np.array([downfm_freq_profile[0]]))
                                                )
        
        cf_time_highres = np.linspace(0,effective_cf_durn, int(fs*effective_cf_durn))
        cf_freq_profile  = np.interp(cf_time_highres, cf_time_effective, rough_cf_freq_profile)
        
        final_frequency_profile = np.concatenate((upfm_freq_profile, 
                                                  cf_freq_profile,
                                                   downfm_freq_profile))
        return final_frequency_profile
    elif not double_fm_call:
        raise ValueError('single FM calls not yet developed...please either contribute the code :P, \
                    or wait for the next version')


def make_FM_with_joint(fm_properties, cf_start, fs, **kwargs):
    '''Outputs an FM segment with the CF part of the joint attached. 
    Think of it like a bent gamma ( :math:`\Gamma`) with the part coming down 
    at an angle instead. 

    Parameters
    ----------
    fm_properties : tuple
        Tuple with format (end_frequency_Hz, fm_duration_seconds)
    cf_start : float
        CF frequency

    See Also
    --------
    make_cffm_joint_profile
    
    Returns
    -------
    fm_with_joint : np.array
        Frequency profile of the FM segment with a bit of the CF part
        of the joint sticking out. 
    '''
    fm_terminal, fm_duration = fm_properties
    fm_bw = cf_start - fm_terminal
    fm_slope = fm_bw/fm_duration


    joint_freq_profile, min_dur = make_cffm_joint_profile(cf_start, fm_slope, fs, 
                                                      **kwargs)
    # fm post/pre joint
    fm_time = np.linspace(0, fm_duration-min_dur, int(fs*(fm_duration-min_dur)))
    start_end_frequency = [joint_freq_profile[-1], fm_terminal]
    fm_post_joint = np.interp(fm_time, [0, fm_duration-min_dur],
                                   start_end_frequency)
    
    fm_with_join = np.concatenate((joint_freq_profile,fm_post_joint))
    return fm_with_join, min_dur




def make_cffm_joint_profile(cf, fm_slope, fs, joint_type='down', **kwargs):
    '''Makes a 'joint' in the frequency profile at transition betweent eh CF and FM parts

    Parameters
    ----------
    cf : float>0
    fm_slope : float>0
    fs : float>0
    poly_order : int, optional
        Polynomial order to be used by np.polyfit
        Defaults to 10
    joint_duration : float, optional 
        The length of the CF and FM joints.
        Default to 0.5 ms
    Returns 
    -------
    freq_profile : np.array
        Frequency at each sample point over the 2*joint_duration 
        length array.
    '''
    poly_order  = kwargs.get('poly_order', 10)
    joint_duration = kwargs.get('joint_duration', 0.0005)

    fm_join_end = cf - fm_slope*joint_duration
    lower_fs = fs*0.75
    
    cf_part = np.tile(cf, int(lower_fs*joint_duration))
    fm_part = np.linspace(cf, fm_join_end, int(lower_fs*joint_duration))
    freqs = np.concatenate((cf_part, fm_part))
    time_lowfs = np.linspace(0, 2*joint_duration, freqs.size)

    fit_joint = np.poly1d(np.polyfit(time_lowfs, freqs, poly_order))
    
    time_highres = np.linspace(0, 2*joint_duration, int(fs*2*joint_duration))
    freq_profile = fit_joint(time_highres)
    if joint_type=='up':
        freq_profile = np.flip(freq_profile)
    return freq_profile, joint_duration


## from the make_CF_training_data module
def make_one_CFcall(call_durn, fm_durn, cf_freq, fs, call_shape, **kwargs):
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
    fm_bandwidth : float, optional
        FM bandwidth in Hz.


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
    FM_bandwidth = np.arange(2,20)
    fm_bw = kwargs.get('fm_bandwidth', np.random.choice(FM_bandwidth, 1)*10.0**3)
    start_f = cf_freq - fm_bw
    # 
    polynomial_num = 25
    t = np.linspace(0, call_durn, int(call_durn*fs))
    # define the transition points in the staplepin
    freqs = np.tile(cf_freq, t.size)
    numfm_samples = int(fs*fm_durn)
    if call_shape == 'staplepin':       
        freqs[:numfm_samples] = np.linspace(start_f,cf_freq,numfm_samples,
                                                     endpoint=True)
        freqs[-numfm_samples:] = np.linspace(cf_freq,start_f,numfm_samples,
                                                     endpoint=True)
        p = np.polyfit(t, freqs, polynomial_num)

    elif call_shape == 'rightangle':
        # alternate between rising and falling right angle shapes
        rightangle_type = np.random.choice(['rising','falling'],1)
        if rightangle_type == 'rising':
            freqs[:numfm_samples] = np.linspace(cf_freq,start_f,numfm_samples,
                                                         endpoint=True)
        elif rightangle_type == 'falling':
            freqs[-numfm_samples:] = np.linspace(cf_freq,start_f,numfm_samples,
                                                         endpoint=True)
        p = np.polyfit(t, freqs, polynomial_num)

    else: 
        raise ValueError('Wrong input given')
      
    cfcall = signal.sweep_poly(t, p)

    #windowing = np.random.choice(['hann', 'nuttall', 'bartlett','boxcar'], 1)[0]
    windowing= 'boxcar'
    cfcall *= signal.get_window(windowing, cfcall.size)
    cfcall *= signal.tukey(cfcall.size, 0.01)
    return cfcall

