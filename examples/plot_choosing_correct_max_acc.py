"""
Setting the correct `max_acc` value
===================================

Some of the methods in the <INSERTNAME> package estimate the instantaneous
frequency at sample-level resolution. Most methods will suffer from edge effects
which cause the estimated instantaneous frequency to spike especially at the start and end of
 the sound or due to noise. 

The typical way these spikes are dealt with is to calculate an aboslute frequency 
accelaration profile along the frequency profile. Any regions above a certain
threshold are considered anomalous, and an (sort of) extrapolation is attempted 
using the nearest non-anomalous regions. 

An example frequency profile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's create an example sound, and use the PWVD method to track the instantaneous 
frequency over time. 

"""

import numpy as np 
from measure_horseshoe_bat_calls.frequency_tracking import generate_pwvd_frequency_profile, frequency_spike_detection
from measure_horseshoe_bat_calls.simulate_calls import make_fm_chirp
import matplotlib.pyplot as plt
from measure_horseshoe_bat_calls.view_horseshoebat_call import plot_accelaration_profile, time_plot

# %%
# Let's create a hyperbolic chirp, this is a nice example because the
# the hyperbolic chirp shows a nice variation in frequeny velocity over time. 
# This means the accelaration varies from low-->high. But what is an 'acceptable'
# value of accelaration to allow. Let's inspect the accelaration profile itself
# to understand what accelaration values are 'normal' and which values correspond
# to the spikes caused by the edge effects and noise.

fs = 22100
chirp = make_fm_chirp(500, 5000, 0.100, fs, 'logarithmic')

raw_fp, frequency_index = generate_pwvd_frequency_profile(chirp,
                                                              fs, percentile=99)
plt.figure()
time_plot(raw_fp,fs)

# %% 
# The spikes caused by edge effects are there here too- even without noise. Let's
# check out the typical accelaration profile of this sound, and pay special
# attention to the values towards the ends. 

acc_plot  = plot_accelaration_profile(raw_fp, fs)
acc_plot.set_ylim(0,0.5) # show a limited y-axis, because the frequency spikes mess up the display


# %% 
# Remember that the accelaration of the frequency is calcualted at a per-sample resolution and thus may not show 
# too much variation -- but the profile still shows outliers! Looking at this
# plot we can see that a value :math:`\geq` 0.1 kHz/ms :math:`^{2}` is likely to be an outlier. 

# %% 
# Now, we know a way to set sensible `max_acc` values for our own recordings - let's see 
# how this translates to outlier detection in the frequency profile:

spikey_regions, acc_profile = frequency_spike_detection(raw_fp, fs, max_acc=0.1)

plt.figure()
a = plt.subplot(211)
time_plot(raw_fp, fs)
plt.plot( np.argwhere(spikey_regions)/fs, raw_fp[spikey_regions], 
         '*', label='Anomalous spikes in frequency profile')
plt.legend()
a.set_title('Detected spikes in frequency profile')
a.set_ylabel('Frequency, Hz')
a.set_xticks([])
b = plt.subplot(212)
time_plot(acc_profile, fs)
plt.plot( np.argwhere(spikey_regions)/fs, acc_profile[spikey_regions], '*')
b.set_ylim(0,0.5)
b.set_title('Frequency accelaration profile')
b.set_ylabel('Frequency accelaration, $kHz/ms^{2}$')
