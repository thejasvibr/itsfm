# -*- coding: utf-8 -*-
"""
Bird song example
^^^^^^^^^^^^^^^^^
Here we'll use the recordings of a common bird, the great tit (*Parus major*). 
The recording is an excerpt of a bigger recording made by Jarek Matusiak 
(Xeno Canto, XC235125) - give it a listen `here <https://www.xeno-canto.org/235125>`_.

Note
>>>>
As of version 0.0.X, this recording is also a very good example of how
multi-harmonic sounds can't be tracked very well! 
"""
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000
import numpy as np 
import scipy.signal as signal 
import itsfm 
from itsfm.data import example_calls, all_wav_files,folder_with_audio_files

great_tit_rec = list(map( lambda X: 'Parus_major_Poland' in X, all_wav_files))
index = great_tit_rec.index(True)
full_audio, fs = example_calls[index] # load the relevant example audio

#
w,s = itsfm.visualise_sound(full_audio, fs, fft_size=512)
s.set_ylim(0,10000)

# %% 
# The complete audio recording takes a long time to run, and so let's focus on
# the sections between 0.8-1.5s. It contains one example of the three types of 
# the great tit's calls. 
t_start, t_stop = 0.8, 1.5
selection = slice(int(fs*t_start), int(fs*t_stop))
audio = full_audio[selection]

w,s = itsfm.visualise_sound(audio, fs, fft_size=256)
s.set_ylim(0,10000)


# %% 
# The bird song has a three types of calls, a smooth frequency modulated sweep
# a constant frequency tone, and the last element has a rather rapid frequency
# sweep which then transitions into a constant frequency segment.

# %%
# Setting the correct signal level
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# The frequency profile of a sound is calculated only for those chunks of the
# audio that are above a threshold dBrms, called the `signal_level`.
# Make a moving dBrms plot to see which a sensible signal threshold to set

plt.figure()
a = plt.subplot(211)
itsfm.plot_movingdbrms(audio, fs, window_size=int(0.005*fs))
plt.subplot(212, sharex=a)
out = plt.specgram(audio, Fs=fs, NFFT=256, noverlap=255)
a.grid()

# %% 
# With this plot, we can see that a level of -34 dB rms with a 5ms window
# will choose the song elements well. Let's try it out. 

non_default_params = {
                    'segment_method':'pwvd',
                    'signal_level':-34,
                    'window_size':int(fs*0.005),
                    'pwvd_window':0.010,
                    'medianfilter_size':0.005,
                    'sample_every':20*10**-3
                    }

output = itsfm.segment_and_measure_call(audio, fs,**non_default_params )

bird_inspect = itsfm.itsFMInspector(output,audio,fs, fft_size=512)

# %% 
# First, let's check if we're actually picking up the bird signals reliable
# with the `signal_level` we chose. 

bird_inspect.visualise_geq_signallevel()

# %% 
# And let's look at the measurements
bird_inspect.measurements

# %%
# We see there are 9 valid sound segments picked up, and their start and stop
# times are displayed. How have they been classified? 

bird_inspect.visualise_cffm_segmentation()

# %% 
# Whoops, it seems like they've all been classified as CF parts. Even though
# the audio actually has FM parts in it, or so we think. Well, whether something
# is frequency modulated or not is set by the `fmrate_threshold`. We need to 
# correct the situation by setting it to a more sensible value. 

# %%
# Setting a non-default FM rate
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# The segmentation of sounds into FM and CF regions happens
# by looking at the FM rate over the sound. Whenever a region 
# crosses the FM rate threshold, it is considered an FM region. 
# Let's check out the FM rate over the sound with the current parameters,
# and then choose a more sensible, non-default `fmrate_threshold` parameter.

bird_inspect.visualise_fmrate()


# %% 
# As you can see the constant frequency and modulated parts are being tracked pretty well, 
# but they're not being classified properly. The CF or FM 
# classification is based on the estimated reate of frequency modulation over the sound,
# ,the `fmrate_threshold`. The default if 1kHz/ms, which is a *lot* if you 
# think about it. At this rate, the bird would have gone from 20kHz to 20 Hz in about
# 20 milliseconds, and you would have *barely* heard it. This default FM rate is set
# to pick up FM regions in bats, and so it needs to be adjusted for other animals.

# %%
# The fm segments in the great tits song correspond to an FM rate of >= 0.005 kHz/ms.
# Remember that all frequency modulation rates are in kHz/ms. Let's set this as the
# threshold and proceed to segment. 
non_default_params['fmrate_threshold'] = 0.02 # 

output_newrate = itsfm.segment_and_measure_call(audio, fs,
                                        **non_default_params)

newrate_inspect = itsfm.itsFMInspector(output_newrate, audio, fs, fft_size=512)

# %% 
# And let's look at the measurements

newrate_inspect.measurements

# %% 
# Let's check the the segmentation output again now 
newrate_inspect.visualise_cffm_segmentation()

# %% 
# So, it's improved, and there seem to be mainly FM regions in at the edges of 
# the sounds. Is this real, or an artifact of the frequency profile fitting. Let's 
# inspect the actual frequency profiles underlying the `fmrate` calcultions

newrate_inspect.visualise_frequency_profiles()

# %% 
# The issue with the third element is that there's a multiple harmonics
# and this may cause the local frequency estiamte to vary up and down
# . We can try to overcome the effect of non-peak frequencies using the 
# :code:`percentile` parameter. The :code:`percentile` essentially

# %% 
# *to be completed....*

