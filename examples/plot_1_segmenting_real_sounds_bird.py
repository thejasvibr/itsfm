# -*- coding: utf-8 -*-
"""
Analysing real sounds : bird calls
==================================
Here we'll use the recordings of a common bird, the great tit (*Parus major*). 
The recording is an excerpt of a bigger recording made by Jarek Matusiak 
(Xeno Canto, XC235125) - give it a listen `here <https://www.xeno-canto.org/235125>`_.

Note
^^^^
As of version 0.1.0, this recording is also a very good example of how
multi-harmonic sounds can't be tracked very well! 
"""
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000
import numpy as np 
import scipy.signal as signal 
import measure_horseshoe_bat_calls as mhbc
from measure_horseshoe_bat_calls.data import example_calls, all_wav_files,folder_with_audio_files
#
index = int(np.argwhere(['Parus_major_Poland' in each for each in all_wav_files])[0])
audio, fs = example_calls[index] # load the relevant example audio


#
w,s = mhbc.visualise_call(audio, fs, fft_size=512)
s.set_ylim(0,10000)

# %% 
# The bird song has a relatively constant frequency and frequency modulated elements
# in it as you can see in the graph above.

# %%
# Setting the correct signal level
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The frequency profile of a sound is calculated only for those chunks of the
# audio that are above a threshold dBrms, called the `signal_level`.
# Make a moving dBrms plot to see which a sensible signal threshold to set

plt.figure()
a = plt.subplot(211)
mhbc.plot_movingdbrms(audio, fs, window_size=int(0.005*fs))
plt.subplot(212, sharex=a)
out = plt.specgram(audio, Fs=fs, NFFT=256, noverlap=255)
a.grid()

# %% 
# With this plot, we can see that a level of -34 dB rms with a 5ms window
# will choose the song elements well. Let's try it out. 

output = mhbc.segment_and_measure_call(audio, fs, segment_method='pwvd',
                                       signal_level=-34,
                                       window_size=int(fs*0.005),
                                       pwvd_window=0.010,
                                       sample_every=0.015,
                                       medianfilter_size=0.010,
                                       extrap_window=2*10**-3,
                                       )
#
seg_out, call_parts, measurements, _ = output
cf,fm,info = seg_out

# %% 
# And let's look at the measurements
measurements

# %%
# We see there are 9 valid sound segments picked up, and their start and stop
# times are displayed. How have they been classified? 

w,s=mhbc.plot_cffm_segmentation(cf, fm, audio, fs, fft_size=512)
s.plot()
mhbc.time_plot(info['fitted_fp'], fs)

# %% 
# Whoops, it seems like they've all been classified as CF parts. Even though
# the audio actually has FM parts in it, or so we think. Well, whether something
# is frequency modulated or not is set by the `fmrate_threshold`. We need to 
# correct the situation by setting it to a more sensible value. 

# %%
# Setting a non-default FM rate
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The segmentation of sounds into FM and CF regions happens
# by looking at the FM rate over the sound. Whenever a region 
# crosses the FM rate threshold, it is considered an FM region. 
# Let's check out the FM rate over the sound with the current parameters,
# and then choose a more sensible, non-default `fmrate_threshold` parameter.


plt.figure()
a = plt.subplot(211)
plt.specgram(audio, Fs=fs, NFFT=256, noverlap=255)
plt.subplot(212, sharex=a)
mhbc.plot_fmrate_profile(info['fitted_fp'], fs)
plt.ylim(0,0.01) # reduce the ylim to show the actual data properly


# %% 
# As you can see the constant frequency and modulated parts are being tracked well, 
# but they're not being classified properly - this is because we have to set
# the right `fmrate_threshold`. The default if 1kHz/ms, which is a *lot* if you 
# think about it. At this rate, the bird would have gone from 20kHz to 20 Hz in about
# 20 milliseconds, and you would have *barely* heard it. This default FM rate is set
# to pick up FM regions in bats, and so it needs to be adjusted for other animals.

# %%
# The fm segments in the great tits song correspond to an FM rate of >= 0.005 kHz/ms.
# Remember that all frequency modulation rates are in kHz/ms. Let's set this as the
# threshold and proceed to segment. 

output = mhbc.segment_and_measure_call(audio, fs, segment_method='pwvd',
                                       signal_level=-34,
                                       window_size=int(fs*0.005),
                                       pwvd_window=0.010,
                                       sample_every=0.015, 
                                       medianfilter_size=0.010,
                                       extrap_window=5*10**-3,
                                       fmrate_threshold=5*10**-3
                                       )


seg_out, call_parts, measurements, _ = output
cf,fm,info = seg_out


# %% 
# And let's look at the measurements

measurements

# %% 
# Let's check the the segmentation output again now 

w,s=mhbc.plot_cffm_segmentation(cf, fm, audio, fs, fft_size=512)
s.plot()
mhbc.time_plot(info['fitted_fp'], fs)

# %% 
# Right, the CF-FM segmentation has improved, but it's still not that great..it's a work
# under progress. But I hope you are convinced the methods in the package can 
# actually be used for bird song too?