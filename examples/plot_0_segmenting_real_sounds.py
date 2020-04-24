"""
Analysing real recordings : a horseshoe bat call
================================================
The <INSERTNEWNAME> package has many example recordings of bat calls thanks to
the generous contributions of bioacousticians around the world:
"""
import numpy as np
import measure_horseshoe_bat_calls as mhbc
from measure_horseshoe_bat_calls.run_example_analysis import contributors
print(contributors)

# %% Let's load the example data from the `data` module of the package

from measure_horseshoe_bat_calls.data import example_calls

# %% 
# Separating the constant frequency (CF) and frequency-modulated parts of a call
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Here, let's take an example *R. mehelyi/euryale(?)* call recording. These
# bats emit what are called 'CF-FM'  calls. This is what it looks like. 

first_call = example_calls[10]
audio, fs = first_call[0], first_call[1]
w,s = mhbc.visualise_call(audio,fs, fft_size=128)

# set the ylim of the spectrogram narrow to check out the call in more detail
s.set_ylim(60000, 125000)

# %%
# Now, let's segment and get some basic measurements from this call. Ignore the 
# actual parameter settings for now. We'll ease into it later !

outputs = mhbc.segment_and_measure_call(audio, fs, 
                                        segment_method='pwvd',
                                        signal_level=-45,
                                        fmrate_threshold=1.0,
                                        extrap_window=25*10**-6)

seg_out, call_parts, measurements, backg = outputs

# %% 
# Let's take a look at how long the different parts of the call are. 

measurements

# %% 
# Verifying the CF-FM segmentations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Here, let's see where the calls are in time and how they match the spectrogram output
cf, fm, info = seg_out

mhbc.plot_cffm_segmentation(cf, fm, audio, fs, fft_size=128)


# %% 
# Even without understanding what's happening here, you can see the 
# 'sloped' regions are within the red boxes, and the 'relatively even 
# region is in the black box. These are the FM and CF parts of this call.

# %%
# The underlying frequency profile of a sound
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The CF and FM parts of a call in the 'pwvd' method is based on actually
# tracking the instantaneous frequency of the call with high temporal 
# resolution. With this profile, the rate of frequency change, or modulation
# can be calculated for each region. Using a threshold rate of the
# frequency modulation, call regions above and below it can be easily identified!

w,s = mhbc.visualise_call(audio,fs,fft_size=128)
s.plot()
mhbc.time_plot(info['fitted_fp'], fs) # the 'fitted fp' is used to calculate the fm rate in a sound

# %% 
# You can see from the plot above that the frequency profile of the sound
# shows a relatively constant frequency region of the call in middle and 
# with frequency modulated regions in the middle.

# %% 
# Performing measurements on the CF and FM parts of a call
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We were just able to get some measurements on the Cf and FM 
# parts of the call. What if we want *more* information, eg. the 
# rms, and peak frequency of each CF and FM call part? This is 
# where <insertname> has a bunch of inbuilt and customisable 
# measurement functions. 

inbuilt_measures = [mhbc.measure_peak_frequency,
                       mhbc.measure_rms]

outputs = mhbc.segment_and_measure_call(audio, fs, 
                                        segment_method='pwvd',
                                        signal_level=-45,
                                        fmrate_threshold=1.0,
                                        extrap_window=25*10**-6,
                                        measurements=inbuilt_measures)

seg_out, call_parts, detailed_measurements, backg = outputs

detailed_measurements

# %%
# The results are output as a pandas DataFrame, which means they can be easily
# saved as a csv file if you were to run it in your system. Each row corresponds
# to one identified CF or FM region in an audio recording. 

# %% 
# Defining custom measurements
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# If the inbuilt measurement functions are not enough - then you may 
# want to write your own. See the documentation for what a measurement
# function must look like by typing `help(mhbc.measurement_function)`. 

def peak_to_peak(whole_audio, fs, segment, **kwargs):
    '''
    Calculates the range between the minimum and the maximum of the audio 
    samples. 
    '''
    relevant_audio = whole_audio[segment]
    peak2peak = np.max(relevant_audio) - np.min(relevant_audio)
    return {'peak2peak':peak2peak}

custom_measure = [peak_to_peak]

outputs = mhbc.segment_and_measure_call(audio, fs, 
                                        segment_method='pwvd',
                                        signal_level=-45,
                                        fmrate_threshold=1.0,
                                        extrap_window=25*10**-6,
                                        measurements=custom_measure)

seg_out, call_parts, custom_measurements, backg = outputs

custom_measurements

# % 
# Of course, needless to say, you can also mix and match inbuilt with 
# custom defined measurement functions. 

mixed_measures = [peak_to_peak, mhbc.measure_rms]

outputs = mhbc.segment_and_measure_call(audio, fs, 
                                        segment_method='pwvd',
                                        signal_level=-45,
                                        fmrate_threshold=1.0,
                                        extrap_window=25*10**-6,
                                        measurements=mixed_measures)


seg_out, call_parts, mixed_measurements, backg = outputs

mixed_measurements


# %% 
# Choosing the right parameters for the recordings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TO BE COMPLETED!! 
#
# How do we know what settings to choose. The main important settins (for most recordings)
# are the `signal_level`, `fmrate_threshold` (while using the 'pwvd' segment method), and
# in some cases the `extrap_window`. 

# %%
# The 


# %% 
# In addition to the CF-FM segmentation, depending on the `method` used to segment the call
# we can also get the estimated frequency profile of a sound. The `pwvd` method provides
# a frequency profile, so let's plot the frequecny profile over the call. 


