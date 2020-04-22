# -*- coding: utf-8 -*-
"""
Finding the right parameter setting with the call zoo
=====================================================
The 'call zoo' is an inbuilt collection of sounds which were made for testing 
the package. It has a variety of sounds to assess the accuracy of the 
segmentation and measuring capabilities of the pacakge. 
"""

import matplotlib.pyplot as plt
from measure_horseshoe_bat_calls.view_horseshoebat_call import *
from measure_horseshoe_bat_calls.simulate_calls import make_call_zoo, add_noise
from measure_horseshoe_bat_calls.segment_horseshoebat_call import segment_call_into_cf_fm

fs=22100

freq_profile, call_zoo = make_call_zoo(fs=fs, gap=0.1)
add_noise(call_zoo, -40)

plt.figure()
plt.specgram(call_zoo, Fs=fs, NFFT=128, noverlap=127);

# %%
# Now, let's run the segmentation on this sound 

cf, fm, info = segment_call_into_cf_fm(call_zoo, fs, method='pwvd')

plot_cffm_segmentation(cf, fm, call_zoo, fs);

# %%
# Now, the results show that some sounds are being recognised, but a closer 
# look the results indicate there's too much silence on either side of the 
# sounds, and the FM sweeps at the end have been mis-classified as CF sounds. Why is this happening?
# This kind of apparent errors typically come from a bad match between the recordings properties and the
# default parameter values in place. The 'issues' can be sorted out most of the time 
# by playing around with the parameter values. 

# %%
# Fixing 'wide' sound selections
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A valid sound is 'recognised' when a region of the audio has an rms :math:`\geq` the `signal_level`
# parameter. The rms over the audio is calcualted by running a moving window. the size of the window
# and the threshold signal leve will thus decide how accurate the 'width' of the sound element selection is.
# 
# We know in our recordings that the signal level is actually pretty high, and so, let's increase the 
# signal level to see if things get better. The :code:`signal_level` is in dB rms with reference value of 1. 

cf, fm, info = segment_call_into_cf_fm(call_zoo, fs, method='pwvd',
                                       signal_level=-15,
                                       window_size=50)
plot_cffm_segmentation(cf, fm, call_zoo, fs);

## Oops, now we've set the 

plot_movingdbrms(call_zoo,fs)

