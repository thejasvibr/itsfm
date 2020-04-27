"""
Segmenting real-world sounds correctly with synthetic sounds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
It's easy to figure out if a sound is being correcly segmented if the 
signal at hand is well defined, and repeatable, like in many technological/
engineering applications. However, in bioacoustics, or 
a more open-ended field recording situation, it can be very hard 
to know the kind of signal that'll be recorded, or what its 
parameters are. 

Just because an output is produced by the package, it doesn't 
always lead to a meaningful result. Given a set of parameters, 
any function will produce an output as long as its sensible. This 
means, with one set of parameters/methods the CF segment might 
be 10ms long, while with another more lax parameter set it might
be 20ms long! Remember, as always, `GIGO <https://en.wikipedia.org/wiki/Garbage_in,_garbage_out>`_ (Garbage In, Garbage Out):P.

How to segment a sound into CF and FM segments in an accurate
way?

Synthetic calls to the rescue
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Synthetic calls are sounds that we know to have specific properties 
and can be used to test if a parameter set/ segmentation method
is capable of correctly segmenting our real-world sounds and 
uncovering the true underlying properties.

The `simulate_calls` module has a bunch of helper functions 
which allow the creation of FM sweeps, constant frequency 
tones and silences. In combination, these can be used to 
get a feeling for which segmentation methods and parameter sets
work well for your real-world sound (bat, bird, cat, <insert sound source of choice>)


Generating a 'classical' CF-FM bat call
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
"""

import matplotlib.pyplot as plt
import numpy as np 
import scipy.signal as signal 
from measure_horseshoe_bat_calls.simulate_calls import make_cffm_call,make_tone, make_fm_chirp, silence 
from measure_horseshoe_bat_calls.view_horseshoebat_call import visualise_call
from measure_horseshoe_bat_calls.segment_horseshoebat_call import segment_call_into_cf_fm 
from measure_horseshoe_bat_calls.signal_processing import dB, rms

fs = 96000
call_props = {'cf':(40000, 0.01), 
			 'upfm':(38000,0.002),
			 'downfm':(30000,0.003)} 

cffm_call, freq_profile = make_cffm_call(call_props, fs)
cffm_call *= signal.tukey(cffm_call.size, 0.1)
 

w,s = visualise_call(cffm_call, fs, fft_size=128)

# %% 
# Remember, the terminal frequencies and durations of the CF-FM calls can be adjusted to the
# calls of your species of interest!!

# %%
# A multi-component bird call
# >>>>>>>>>>>>>>>>>>>>>>>>>>>
# 
# Let's make a sound with two FMs and CFs, and gaps in between


fs = 44100

fm1 = make_fm_chirp(1000, 5000, 0.01, fs)
cf1 = make_tone(5000, 0.005, fs)
fm2 = make_fm_chirp(5500, 9000, 0.01, fs)
cf2 = make_tone(8000, 0.005, fs)
gap = silence(0.005, fs)

synth_birdcall = np.concatenate((gap,
                                 fm1, gap, 
                                 cf1, gap,
                                 fm2, gap,
                                 cf2, 
                                 gap))

w, s = visualise_call(synth_birdcall, fs, fft_size=64)

# %% 
# Let there be Noise
# >>>>>>>>>>>>>>>>>>
#
# Any kind of field recording *will* have some form of noise. Each of the 
# the segmentation methods is differently susceptible to noise, and it's
# a good idea to test how well they can tolerate it. For starters, let's 
# just add white noise and simulate different signal-to-noise ratios (SNR).

noisy_bird_call = synth_birdcall.copy()
noisy_bird_call += np.random.normal(0,10**(-10/20), noisy_bird_call.size)
noisy_bird_call /= np.max(np.abs(noisy_bird_call)) # keep sample values between +/- 1

# %%
# Estimate an approximate SNR by looking at the rms of the gaps to that of 
# a song component

level_background = dB(rms(noisy_bird_call[gap.size]))

level_song = dB(rms(noisy_bird_call[gap.size:2*gap.size]))

snr_approx = level_song-level_background

print('The SNR is approximately: %f'%np.around(snr_approx))

w, s = visualise_call(noisy_bird_call, fs, fft_size=64)

# %% 
# We could try to run the segmentation + measurement on a noisy sound straight away, 
# but this might lead to poor measurements. Now, let's bandpass the audio 
# to remove the ambient noise outside of the song's range. 





