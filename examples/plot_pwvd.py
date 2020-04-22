"""
Segmenting CF and FM using the PWVD method
==========================================
The 'PWVD' method stands for the Pseudo Wigner-Ville Distribution. It is a class
of time-frequency representations that can be used to be gain very high spectro-
temporal resolution of a sound [1,2]. 

How does it work?
~~~~~~~~~~~~~~~~~
The PWVD is made by performing a local auto-correlation at each sample in the
audio signal, with a window applied onto it later. The FFT of this windowed-
auto correlation reveals the local spectro-temporal content. The 'tftb' package [3] 
is used to generate the PWVD representation in this package. The website is also a
great place to see more examples and great graphics of the PWVD and alternate
time-frequency distributions!.

The PWVD thus produces spectrogram like visualisations of the sound, albeit with
its own quirks sometimes. 

References
----------
[1] Cohen, L. (1995). Time-frequency analysis (Vol. 778). Prentice hall.

[2] Boashash, B. (2015). Time-frequency signal analysis and processing: a
    comprehensive reference. Academic Press.
    
[3] Jaidev Deshpande, tftb 0.1.1,  https://tftb.readthedocs.io/en/latest/auto_examples/index.html

"""

# %%
# Let's begin by making a synthetic CF-FM call which looks a lot like a horseshoe/leaf nosed bat's call 

import matplotlib.pyplot as plt
import scipy.signal as signal 
from measure_horseshoe_bat_calls.frequency_tracking import generate_pwvd_frequency_profile
from measure_horseshoe_bat_calls.frequency_tracking import pwvd_transform
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
# The PWVD is a somewhat new representation to most people, so let's just check out an example
# 

pwvd = pwvd_transform(cffm_call, fs)

# %%
# The output is an NsamplesxNsamples matrix, where Nsamples is the number
# of samples in the original audio. 

plt.figure()
plt.imshow(abs(pwvd), origin='lower')

# %% 
# The dominant frequency at each sample can be tracked to see how the 
# the frequency changes over time.  Let's not get into the details right away, 
# and proceed with the segmentation first. 

cf, fm, info = segment_call_into_cf_fm(cffm_call, fs, segment_method='pwvd',
										window_size=50)

# %% 
# The `segment_call_into_cf_fm` provides the estimates of which samples are CF and FM. The `info` object is a 
# dictionary with content that varies according to the segmentation method used. For instance:

info.keys()

# %%
#  To illustrate how exactly the method works, let's check out the output from the `info` dictionary. 

#plt.figure()
#plt.plot(info['cf_dbrms'], label='CF emphasised')
#plt.plot(info['fm_dbrms'],  label='FM emphasised')
#plt.ylabel('Signal level, dB rms')
#plt.legend()

# %%
# You can see the FM dBrms peaks towards the end and the start, while the CF peaks at the middle. The peak percentage method 
# relies on subtracting the two from each other to see which parts of the call the FM and CF are dominant. We thus get this 
# from the previous dBrms profiles

#plt.figure()
#plt.plot(info['cf_re_fm'], label='relative CF')
#plt.plot(info['fm_re_cf'], label='relative FM')
#plt.legend()
#plt.ylabel('CF/FM relative level')

# %%
# And thus, we can see that wherever the relative FM/CF is >0, we can safely assign it to a segment of that type. 
# Compare the relative levels and the final segmented values below. 
#
#plt.figure()
#plt.subplot(211) 
#plt.plot(cf, label='segmented CF')
#plt.plot(fm, label='segmented FM')
#plt.legend()
#plt.subplot(212)
#plt.plot(info['cf_re_fm'], label='relative CF')
#plt.plot(info['fm_re_cf'], label='relative FM')
#plt.legend()
#plt.ylabel('CF/FM relative level')
