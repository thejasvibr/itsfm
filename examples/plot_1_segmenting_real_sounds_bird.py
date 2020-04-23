# -*- coding: utf-8 -*-
"""
Analysing real sounds : bird calls
==================================
Here we'll use the <insertname> package to identify elements of a dolphin song

*Warning* : this section is not yet complete -- work in progress. The dolphin
calls are actually very horribly tracked, and super slow. 

"""
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000
import numpy as np 
import scipy.signal as signal 
import measure_horseshoe_bat_calls as mhbc
from measure_horseshoe_bat_calls.data import example_calls, all_wav_files,folder_with_audio_files
#
#audio, fs = example_calls[76]
#
##audio = audio[int(0.9*fs):]# take out a part which works well as an example!
#
#w,s = mhbc.visualise_call(audio, fs, fft_size=512)
#s.set_ylim(0,10000)
##
#
###  bandpass the recording
#b,a = signal.butter(2, np.array([5000])/fs*0.5, 'highpass')
#audio_bp = signal.lfilter(b,a,audio)
##
### see how much the recording improved after filtering. 
##w,s = mhbc.visualise_call(audio_bp, fs, fft_size=512)
##s.set_ylim(0,5000)
#
## make a movind dbrms plot to see which a sensible signal threshold to set
#plt.figure()
#a = plt.subplot(211)
#mhbc.plot_movingdbrms(audio_bp, fs, window_size=int(0.005*fs))
#plt.subplot(212, sharex=a)
#out = plt.specgram(audio_bp, Fs=fs, NFFT=256, noverlap=255)
#a.grid()
#
## -50 dB rms seems to be a sensible enough `signal_level` estimate
#
#output = mhbc.segment_and_measure_call(audio_bp, fs, segment_method='pwvd',
#                                       signal_level=-40,
#                                       window_size=int(fs*0.005),
#                                       pwvd_window=0.010,
#                                       medianfilter_length=0.01,
#                                       sample_every=0.0025,
#                                       percentile=99
#                                       )
#
#seg_out, call_parts, measurements, _ = output
#cf,fm,info = seg_out
#
#measurements
#
#w,s=mhbc.plot_cffm_segmentation(cf, fm, audio_bp, fs, fft_size=512)
#s.plot()
#mhbc.time_plot(info['fitted_fp'], fs)
#
## %%
## Let's check out the FM rate over the sound. 
#
#plt.figure()
#mhbc.plot_fmrate_profile(info['fitted_fp'], fs)
#
#
## see how much the recording improved after filtering. 
#w,s = mhbc.visualise_call(audio_bp, fs, fft_size=256)
#s.set_ylim(0,8000)
#mhbc.time_plot(info['fitted_fp'], fs)
