# -*- coding: utf-8 -*-
"""
Finding the right parameter setting with the call zoo
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The 'call zoo' is an inbuilt collection of sounds which were made for testing 
the package. It has a variety of sounds to assess the accuracy of the 
segmentation and measuring capabilities of the pacakge. 
"""

import matplotlib.pyplot as plt
import numpy as np 
np.random.seed(82319)
import itsfm
from itsfm.simulate_calls import make_call_zoo, add_noise
from itsfm.segment import segment_call_into_cf_fm


fs=30000

freq_profile, call_zoo = make_call_zoo(fs=fs, gap=0.1)
add_noise(call_zoo, -40)

itsfm.visualise_sound(call_zoo, fs, fft_size=128)
itsfm.plot_movingdbrms(call_zoo,fs, window_size=int(fs*0.001))

# %%
# Now, let's run the segmentation on this sound 
segment_parameters = {'window_size' : int(fs*0.001),
                      'segment_method':'pwvd',
                      'signal_level': -30,
                      'sample_every':0.25*10**-3}
segment_out = segment_call_into_cf_fm(call_zoo, fs, **segment_parameters)
cf, fm, info = segment_out 
itsfm.visualise_cffm_segmentation(cf,fm,call_zoo,fs, fft_size=128)

# %%
# Now, the results show that some sounds are being recognised, but a closer 
# look the results indicate there's too much silence on either side of the 
# sounds, and the FM sweeps at the end have been mis-classified as CF sounds. Why is this happening?
# This kind of apparent errors typically come from a bad match between the recordings properties and the
# default parameter values in place. The 'issues' can be sorted out most of the time 
# by playing around with the parameter values. 
