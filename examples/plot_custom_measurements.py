"""
Inbuilt and custom measurements on CF and FM segments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
By default, a baic set of information/measurements is given for each
recognised CF/FM segment in the input audio, its start, stop and duration. 

"""

# %%
# Let's begin by making a synthetic CF-FM call which looks a lot like a horseshoe/leaf nosed bat's call 

import matplotlib.pyplot as plt
import scipy.signal as signal 
from measure_horseshoe_bat_calls.simulate_calls import make_cffm_call
from measure_horseshoe_bat_calls.view_horseshoebat_call import visualise_call
from measure_horseshoe_bat_calls.user_interface import segment_and_measure_call

# %% 
# Lets now create a sound that's got only one CF and one FM component
# in it. Horseshoe/leaf nosed bats emit these kinds of calls too. 

fs = 44100 
call_props = {'cf':(8000, 0.01),
'upfm':(8000,0.002), # not that the 'upfm' frequency starts at the CF frequency!
'downfm':(100,0.003)} 

cffm_call, freq_profile = make_cffm_call(call_props, fs)
cffm_call *= signal.tukey(cffm_call.size, 0.1)

w,s = visualise_call(cffm_call, fs, fft_size=64)

# %%
# Now, segment and measure using the 'peak pecentage' method

output = segment_and_measure_call(cffm_call, fs, 
                                  segment_method = 'peak_percentage',
                                  peak_percentage=0.95,
                                  window_size=44)
segment_info, call_parts, results, _ = output

# %%
#  If everything went well, the output should give us one CF and one FM component. 
# The parameters may need to be tweaked based on the sampling rate and the 
# amount of frequency modulation in the calls. This is true especially for 
# sounds with a 'curvature' in the frequency profile, because sometimes the 
# frequency change may be gradual and then become sudden, eg in the transition between
# CF and FM in this example call. 

# %%
# Another important aspect to notice is that the window size has been set to 44 samples. 
# This corresponds to a short window of ~0.1ms. This short window size is used to compare the relative CF and FM emphasised dB rms
# profiles (see 'The peak percentage method').

print(results)

# %%
# What if we want more than just the duration of each component?
# There are inbuilt functions such which allow the measurement of the 
# rms, peak-amplitude, peak frequency and terminal frequency of each segment. 
# Let's get the peak frequency and peak amplitude for all segments

from measure_horseshoe_bat_calls.measurement_functions import measure_peak_frequency, measure_peak_amplitude

added_measures = [measure_peak_amplitude, measure_peak_frequency]

output = segment_and_measure_call(cffm_call, fs, 
                                  peak_percentage=0.95,
                                  window_size=44,
                                  measurements=added_measures)

segment_info, call_parts, results, _ = output

print(results)

# %% 
# Now, what if this is not what we're looking for and we needed to get, say, the *dB peak* amplitude?
# This calls for a custom measurement function. Each measurement function follows
# a particular pattern of three inputs and one output. See the `measurement_function`
# documentation or call it through the help

from measure_horseshoe_bat_calls import measurement_functions as measure_funcs
help(measure_funcs)

# %% 
# Let's also take a look at the source code for one of the measurement functions
# we just used above `measure_peak_amplitude`:

import inspect
print(inspect.getsource(measure_peak_amplitude))

# %% 
# The output needs to be a dictionary with the measurement names and values in 
# the keys and items respectively. 

# %%
# So, now let's get the dB peak value of our audio segments
import numpy as np 
from measure_horseshoe_bat_calls.signal_processing import dB

def measure_dBpeak(audio, fs, segment, **kwargs):
    relevant_audio = audio[segment]
    dB_peak_value = dB(np.max(np.abs(relevant_audio)))
    return {'dB_peak': dB_peak_value}


output = segment_and_measure_call(cffm_call, fs, 
                                  peak_percentage=0.95,
                                  window_size=44,
                                  measurements=[measure_dBpeak])

segment_info, call_parts, results, _ = output

print(results)

# %% 
# So, looking at the dB peak value tells us that both CF and FM components are
# pretty strong, and of comparable levels. Both are close to 0 dB (re 1), which means
# they're pretty close to the maximum signal value. 

# %% 
# Just like the `measure_dBpeak`, we can chain a series of inbuilt or custom measurement
# functions in a list - and the outputs will all appear as a wide-formate Pandas DataFrame. 
