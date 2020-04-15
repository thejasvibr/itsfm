# -*- coding: utf-8 -*-
""" The 'simulate_calls' module has functions which simulate CF-FM calls 
with parameters of choice. Let's say we want to make a CF-FM call with the
following parameters. 

* CF portion at 100kHz and of 10ms length. 
* up FM starting at 80kHz of 2ms
* down FM end at 60kHz of 3ms

The code snippet below recreates the call with the above parameters

.. code-block:: python

    from measure_horseshoe_bat_calls.view_horseshoebat_call import visualise_call
    from measure_horseshoe_bat_calls.simulate_calls import make_cffm_call

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

def make_call_frequency_profile(call_properties, fs, **kwargs):
    '''
    
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