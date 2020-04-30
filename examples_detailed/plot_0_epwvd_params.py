"""
How to: troubleshoot PWVD segmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The <insertname> package was mainly designed keeping horseshoe bat calls 
in mind. These calls are high-frequency (>50kHz) and short (20-50ms) sounds
which are quite unique in their structure. Many of the default parameter
values reflect the original dataset. In fact, many of the default parameters
don't even work for some of the example datasets themselves!
It should be no surprise that unpredictable things happen when segmentation
 and tracking is run with default values. 

This example will guide you through understanding the various parameters
that can be tweaked and what effect they actually have. It is not 
an exhaustive treatment of the implementation, but a 'lite' intro. For more
details of course, the original documentation should hopefully be helpful. 
"""
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np 
import measure_horseshoe_bat_calls as mhbc
from measure_horseshoe_bat_calls.data import example_calls

# a chosen set of tricky calls to illustrate various points
tricky_indices = [4,5,6,8,11,12,18,25]
audio_examples = { index: example_calls[index] for index in tricky_indices}



# %%
# Step 1: the right `signal_level`
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# In the given audio segment, the first step is to identify what is 
# background and what is signal. The signal of interest is identified as
# being above a particular dB rms, as calculated y a moving dB rms window
# of a user-defined `window_size`. 

audio, fs = audio_examples[4]
mhbc.visualise_call(audio, fs)

# %% 
# If we want high temporal resolution to segment out the call, we need a short
# `window_size`. Let's try out  0.5 and 2ms for now. 

halfms_windowsize = int(fs*0.5*10**-3)
twoms_windowsize = halfms_windowsize*4
plt.figure()
ax = plt.subplot(211)
mhbc.plot_movingdbrms(audio, fs, window_size=halfms_windowsize)
mhbc.plot_movingdbrms(audio, fs, window_size=twoms_windowsize)

first_color = '#1f77b4'
second_color = '#ff7f0e'
custom_lines = [Line2D([0],[0], color=first_color),
                Line2D([1],[1],color=second_color),]
ax.legend(custom_lines, ['0.5ms', '2ms'])
plt.ylabel('Moving dB rms')
plt.subplot(212, sharex=ax)
_ = mhbc.make_specgram(audio, fs);

# %% 
# The fact that the 0.5ms moving rms profile is so 'rough' is already a bad
# sign. The signal of interest is any region/s which are above or equal to 
# the `signal_level`. When the moving rms fluctuates so wildly, the relevant
# signal region may be hard to capture because it keeps going above and 
# below the threshold - leading to many tiny 'íslands'. Let's choose the 2ms `window_size` 
# because it doesn't fluctuate crazily and is also a relatively short time scale
# in comparison the the signal duration. -40 dB rms seems to be a sensible value 
# when we compare the approximate start and end times of the signal with the dB rms profile. 

keywords = {'segment_method':'pwvd',
            'signal_level':-40,
            'window_size':twoms_windowsize}

outputs = mhbc.segment_and_measure_call(audio, fs,**keywords)
seg_out, call_parts, measurements, backg = outputs

cf, fm, info = seg_out

# %% 
# Let's check the output as it is right now
mhbc.plot_cffm_segmentation(cf, fm, audio, fs)

# %% 
# Inspect initial outputs
# The CF-FM segmentation is *clearly* not correct. There's way too many CF and 
# FM segments. Where is this coming from? Let's inspect the `info` dictionary
# and its components. 
info.keys()

# %% 
# The `info` dictionary : a peek into how it all works
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# With the 'info' dictionary, we get access to the various underlying data that `segment_and_measure_call` used
# to come to the end point. You can check the actual region that was defined 
# as being greater or equal to the input `signal_level` by inspecting 
# the `geq_signal_level` which is a list of slices above the threshold. 
#
print(info['geq_signal_level'])

# assign the slice corresponding to the region above the threshold 
above_threshold = np.zeros(audio.size,dtype='bool')
above_threshold[info['geq_signal_level'][0]] = True
# make a plot of the region above the threshold in the form of a binary array.
plt.figure()
mhbc.make_specgram(audio, fs)
mhbc.make_waveform(above_threshold*125000, fs)

# %% 
# The main signal has been identified correctly, even though part of the 
# beginning of the call is not in the window. Whether you want to change the 
# `signal_level` again or not is a matter of the exact use case!

# %% 
# How the CF-FM segmentation works
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#
# .. image:: ../_static/fmrate_workflow.png
#    :width: 70 %
#    :align: left
# CF-FM segmentation occurs through a multi step process. 
# First the instantaneous frequency of the signal is estimated at a sample-level
# resolution, the raw frequency profile - `raw_fp`. Then the `raw_fp` is refined 
# as it can be quite noisy because of well, noise, or abrupt changes in signal
# level across the sound. 
#
# Minor jumps will be corrected to give rise to the 
# cleaned frequency profile - `cleaned_fp`.  The `cleaned_fp` however, is 
# a very high-resolution look into the sound's frequency profile. Even though
# the temporal resolution is high, the spectral resolution is limited by the 
# size of the `pwvd_window` (refer to the original docs here). This limited
# spectral resolution means each sample will not have a unique value. For instance
# if the frequency of sound is increasing linearly with time, the `cleaned_fp`
# may actually look like steps going up. These 'steps' cause issues while calculating
# the rate of frequency modulation - `fmrate`, and so , the `cleaned_fp` is 
# actually downsampled and then upsampled by interpolation. This gives rise to 
# the fitted frequency profile - `fitted_fp`.
#
# The `fitted_fp` captures the local trends and doesn't have the step like nature of
# `cleaned_fp`. If we were to actually measure frequency modulation from `cleaned_fp`
# there'd be lots of 0 modulation regions and many very brief bursts of FM regions
# wherever a 'step' rose or dropped. Thanks to the sample-wise unique values in `fitted_fp` we can now 
# calculate the local variation in frequency modulation across the sound.
#
# Let's now check how well the frequency profiles have been tracked. Typically 
# a weird segmentation is the result of poor underlying parameter choices for
# the signal at hand. 
raw_fp, cleaned_fp, fitted_fp = [info[key] for key in ['raw_fp', 'cleaned_fp', 'fitted_fp']]

plt.figure()
mhbc.make_specgram(audio, fs)
plt.plot(mhbc.make_x_time(raw_fp, fs), raw_fp, label='raw fp')
plt.plot(mhbc.make_x_time(cleaned_fp, fs), cleaned_fp, label='cleaned fp')
plt.plot(mhbc.make_x_time(fitted_fp, fs), fitted_fp, label='fitted fp')
plt.legend()

# %% 
# The raw and cleaned frequency profiles are very similar, though the 'cleanliness'
# in the `cleaned_fp` is visible especially because the frequency profile doesn'
# wildly jump around towards the end of the call. The `fitted_fp` also 
# closely matches the `cleaned_fp` though it seems to rise later and drop faster.
# This is because of the downsampling that happens to estimate the `fmrate`. 
# The rise time is a direct indicator of the downsampling factor, which samples
# the `cleaned_fp` at periodic intervals, and is thus called `sample_every`. The
# `sample_every` parameter defaults to 0.5ms. If the frequency profiles broadly match
# the actual call as seen coarsely on a spectrogram - where is the issue here?

# %% 
# Step 2: Check the `fmrate` profile
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# CF and FM parts of a call are segmented based on the rate of frequency modulation
# they show. The `fmrate` is a np.array with the estimated frequency modulation
# rate in **kHz/ms**. Yes, pay attention to the units, *it's not kHz/s, but kHz/ms*!
# Let's take a look at the FM rate profile for this sound. 
fmrate = info['fmrate']

plt.figure()
plt.subplot(311)
plt.title('FM rate')
out = mhbc.make_specgram(audio, fs);
plt.xticks([])
plt.xlabel('')
plt.subplot(312)
mhbc.make_waveform(fmrate, fs)
plt.xticks([])
plt.xlabel('')
plt.ylabel('FM rate, kHz/ms')
ax2=plt.subplot(313)
plt.title('CF-FM segmentation')
mhbc.make_waveform(cf, fs)
mhbc.make_waveform(fm, fs)
plt.ylabel('CF/FM')
ax2.legend(custom_lines, ['CF', 'FM'])

# %%
# Something's odd -- even though the FM rate seems to be close to zero
# in the middle, parts of it are still being classified as FM!! What's happening. 
# Let's take a closer look at the FM rate profile, but zoom in so the y-axis is
# more limited. Let's also overlay the CF/FM outputs over this plot. 

plt.figure()
plt.subplot(211)
mhbc.make_waveform(fmrate, fs)
plt.ylim(0,1.5)
plt.xticks([])
plt.ylabel('FM rate, kHz/ms')
ax3 = plt.subplot(212)
mhbc.make_waveform(cf, fs,)
mhbc.make_waveform(fm, fs,)
plt.ylabel('CF/FM')
ax3.legend(custom_lines, ['CF', 'FM'])

# %% 
# From this you can clearly see that the FM parts correspond to tiny peaks in 
# the `fmrate` which reach around 0.25 kHz/ms. It may of course be no surprise
# once you know the default `fmrate_threshold` is 0.2 kHz. This rate doesn'
# make sense for bat call FM portions as they have much high frequency modulation
# rates. In general, 0.2 kHz/ms is *very* low, in comparison many bat calls can
# go from 90-30 kHz (60kHz bandwidth) in about 3ms, this corresponds to 20kHz/ms!
# The easy way to estimate the relevant `fmrate_threshold` is to eyeball 
# the start and end frequencies of a call part and calculate the fm rate!

# %% 
# Step 3: Set a relevant `fmrate_threshold`
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# For this example call any FM rate above 0.5kHz/ms will allow a sensible segmentation of the CF and FM
# parts. Lets set it more conservatively at 1kHz/ms, this will reduce false 
# positives. In general, for this particular call type, the FM sweep has an approximate
# rate of 5-6kHz/ms, and so we should definitely be able to pick up the FM region 
# with a threshold of 1kHz/ms. 

# add an additional keyword argument
keywords['fmrate_threshold'] = 1.0 # kHz/ms

outputs = mhbc.segment_and_measure_call(audio, fs,**keywords)
seg_out, call_parts, measurements, backg = outputs
cf, fm, info = seg_out
# %% 
# Let's take a look at the new segmentation results
mhbc.plot_cffm_segmentation(cf, fm, audio, fs)
raw_fp, cleaned_fp, fitted_fp = [info[key] for key in ['raw_fp', 'cleaned_fp', 'fitted_fp']]

# %% 
# Remember, there was no change to the actual frequency tracking parameters here, 
# and so there's no change expected in any of the frequency profiles. However, 
# There is still one thing that's not quite right. The spectrogram doesn't really show an
# FM segment at the beginning of the call - why is it being detected as one?
# This could be two things: 1) there's poor tracking of the frequency in the 
# signal edges, or 2) there is *actually* an frequency modulation, but it's not
# quite visible. Let's zoom in and take a look. 

plt.figure()
ax4 = plt.subplot(311)
mhbc.make_specgram(audio, fs)
plt.plot(mhbc.make_x_time(raw_fp, fs), raw_fp, label='raw fp')
plt.plot(mhbc.make_x_time(cleaned_fp, fs), cleaned_fp, label='cleaned fp')
plt.plot(mhbc.make_x_time(fitted_fp, fs), fitted_fp, label='fitted fp')
plt.legend()
plt.xlim(0.0015,0.0035);plt.ylim(0,110000)
plt.subplot(312, sharex=ax4)
mhbc.make_waveform(fmrate, fs)
plt.ylabel('FM rate, kHz/ms')
ax5 = plt.subplot(313, sharex=ax4)
mhbc.make_waveform(cf, fs,)
mhbc.make_waveform(fm, fs,)
plt.ylabel('CF/FM')
ax5.legend(custom_lines, ['CF', 'FM'])



# add an additional keyword argument
keywords['sample_every'] = 100*10**-6

outputs = mhbc.segment_and_measure_call(audio, fs,**keywords)
seg_out, call_parts, measurements, backg = outputs
cf, fm, info = seg_out
# %% 
# Let's take a look at the new segmentation results
mhbc.plot_cffm_segmentation(cf, fm, audio, fs)
raw_fp, cleaned_fp, fitted_fp = [info[key] for key in ['raw_fp', 'cleaned_fp', 'fitted_fp']]
