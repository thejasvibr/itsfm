"""
The peak-percentage method
^^^^^^^^^^^^^^^^^^^^^^^^^^

The peak percentage method works if the constant frequency portion of a sound segment 
is the highest frequency. For instance, in CF-FM bat calls, the calls typically have 
a CF and one or two FM segments connected. 

This method is loosely based on the spectrogram based CF-FM segmentation in [1], but most importantly
it differs because it is implemented completely in the time-domain. 


How does it work?
>>>>>>>>>>>>>>>>>
A constant frequency segment in any sound leads to a peak in the power spectrum. The same audio 
is high-passed and low-passed at a threshold frequency that's very close (eg. 99% of the peak frequency)
 and just below the peak frequency. This creates two versions of the same sound, one with an emphasis on the CF, and one with the emphasis on 
the FM. By comparing the two sounds, the segmentation proceeds to detect CF and FM parts. 


References

[1]Schoeppler, D., Schnitzler, H. U., & Denzinger, A. (2018). Precise Doppler shift compensation in the hipposiderid bat, 
   Hipposideros armiger. Scientific reports, 8(1), 1-11.

"""

# %%
# Let's begin by making a synthetic CF-FM call which looks a lot like a horseshoe/leaf nosed bat's call 

import matplotlib.pyplot as plt
import scipy.signal as signal 
from measure_horseshoe_bat_calls.simulate_calls import make_cffm_call
from measure_horseshoe_bat_calls.view_horseshoebat_call import visualise_call
from measure_horseshoe_bat_calls.segment_horseshoebat_call import segment_call_into_cf_fm 
fs = 44100 
call_props = {'cf':(8000, 0.01), 
			 'upfm':(2000,0.002),
			 'downfm':(100,0.003)} 

cffm_call, freq_profile = make_cffm_call(call_props, fs)
cffm_call *= signal.tukey(cffm_call.size, 0.1)

w,s = visualise_call(cffm_call, fs, fft_size=64)

# %% 
# Now, let's proceed to run the peak-percentage based segmentation. 

cf, fm, info = segment_call_into_cf_fm(cffm_call, fs, segment_method='peak_percentage',
										window_size=50)

# %% 
# The `segment_call_into_cf_fm` provides the estimates of which samples are CF and FM. The `info` object is a 
# dictionary with content that varies according to the segmentation method used. For instance:

info.keys()

# %%
#  To illustrate how exactly the method works, let's check out the output from the `info` dictionary. 

plt.figure()
plt.plot(info['cf_dbrms'], label='CF emphasised')
plt.plot(info['fm_dbrms'],  label='FM emphasised')
plt.ylabel('Signal level, dB rms')
plt.legend()

# %%
# You can see the FM dBrms peaks towards the end and the start, while the CF peaks at the middle. The peak percentage method 
# relies on subtracting the two from each other to see which parts of the call the FM and CF are dominant. We thus get this 
# from the previous dBrms profiles

plt.figure()
plt.plot(info['cf_re_fm'], label='relative CF')
plt.plot(info['fm_re_cf'], label='relative FM')
plt.legend()
plt.ylabel('CF/FM relative level')

# %%
# And thus, we can see that wherever the relative FM/CF is >0, we can safely assign it to a segment of that type. 
# Compare the relative levels and the final segmented values below. 

plt.figure()
plt.subplot(211) 
plt.plot(cf, label='segmented CF')
plt.plot(fm, label='segmented FM')
plt.legend()
plt.subplot(212)
plt.plot(info['cf_re_fm'], label='relative CF')
plt.plot(info['fm_re_cf'], label='relative FM')
plt.legend()
plt.ylabel('CF/FM relative level')
