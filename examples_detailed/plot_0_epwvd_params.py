"""
Effect of PWVD parameter choices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The <insertname> package was mainly designed keeping horseshoe bat calls 
in mind. These calls are high-frequency (>50kHz) and short (20-50ms) sounds
which are quite unique in their structure. Many of the default parameter
values reflect the original dataset. It should be no surprise that functions 
fail unpredictably when run with their default methods. 

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
# The fact that the 0.5ms moving rms profile is so rough is already a bad
# sign. The signal of interest is any region/s which are above or equal to 
# the `signal_level`. When the moving rms fluctuates so wildly, the relevant
# signal region may be hard to capture because it keeps going above and 
# below the threshold. Let's choose 2ms because it doesn't fluctuate crazily and 
# is also a relatively short time scale in comparison the the signal duration. 
# -40 dB rms seems to be a sensible value when we compare the approximate start
# and end times of the signal with the dB rms profile. 

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
# The `info` dictionary : the pwvd case
# With the 'info' dictioanry, we get access to the various underlying data that `segment_and_measure_call` used
# to come to the end point. You can check the actual region that was defined 
# as being greater or equal to the input `signal_level` by inspecting 
# the `geq_signal_level ` which is a list of slices above the threshold. 
#
print(info['geq_signal_level'])

above_threshold = np.zeros(audio.size,dtype='bool')
above_threshold[info['geq_signal_level'][0]] = True
plt.figure()
mhbc.make_specgram(audio, fs)
mhbc.make_waveform(above_threshold*125000, fs)

# %% 
# The main signal has been identified correctly, even though part of the 
# beginning of the call is not in the window. Whether you want to change the 
# `signal_level` again or not is a matter of the exact use case!

# %% 
# Step 2: Troubleshooting CF-FM segmentation and frequency tracking
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# CF-FM segmentation occurs through a two step process. 
# First the instantaneous frequency of the signal is estimated at a sample-level
# resolution, the raw frequency profile - `raw_fp`. Then the `raw_fp` is refined 
# as it can be quite noisy because of well, noise, or abrupt changes in signal
# level across the sound. Minor jumps will be corrected to give rise to the 
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
# Step 3: Check the `fmrate` profile
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# CF and FM parts of a call are segmented based on teh rate of frequency modulation
# they show. The `fmrate` is a np.array with the estimated frequency modulation
# rate in kHz/ms. Yes, pay attention to the units, it's not kHz/s, but kHz/ms!
# Let's take a look at the FM rate profile for this sound. 
fmrate = info['fmrate']
plt.figure()
plt.subplot(211)
mhbc.make_waveform(fmrate, fs)
plt.ylabel('FM rate, kHz/ms')
plt.subplot(212)
out = mhbc.make_specgram(audio, fs);

# %% 
# Step 4: Set a relevant `fmrate_threshold`
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 
