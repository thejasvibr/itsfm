# -*- coding: utf-8 -*-
"""
CF-FM call segmentation accuracy
================================
This page will illustrate the accuracy with which `itsfm` can segment CF-FM parts of a 
CF-FM call. To see what a CF-FM call looks like check out the bat-call example 
in the 'Basic Examples' page. 

The synthetic data has already been generated and run with the :code:`segment_and_measure`
function, and now we'll compare the accuracy with which it has all happened.

A CF-FM bat call typically has three parts to it, 1) an 'up' FM, where the  
frequency of the call increases, 2) a 'CF' part, where the frequency is 
stable, and then 3) a 'down' FM, where the frequency drops. The synthetic
data is basically a set of CF-FM calls with a combination of upFM, downFM
and CF part durations, bandwidths,etc. 

Here we will only be seeing if the durations of each of the segment parts have been picked
up properly or not. We will *not* be performing any accuracy assessments on 
the exact parameters (eg. peak frequency, rms, etc) because it is assumed that 
if the call parts can be identified by their durations then the measurements will 
in turn be as expected. 

There is no silence in the synthetic calls, and no noise too. This is the 
situation which should provide the highest accuracy. 

What happened before
~~~~~~~~~~~~~~~~~~~~
To see more on the details of the generation and running of the synthetic data 
see the modules `CF/FM call segmentation` and `Generating the CF-FM synthetic calls`

"""
import h5py
import itsfm
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000
import numpy as np
import pandas as pd 
import seaborn as sns
import tqdm

obtained = pd.read_csv('obtained_horseshoe_sim.csv')
synthesised = pd.read_csv('horseshoe_test_parameters.csv')

# %% 
# Let's look at the obtained regions and their durations
obtained

# %% 
# We can see the output has each CF/FM region labelled by the order in which
# they're found. Let's re-label these to match the names of the synthesised
# call parameter dataframe. 'upfm' is fm1, 'downfm' is fm2. 

obtained.columns = ['call_number','cf_duration',
                    'upfm_duration', 'downfm_duration', 'other']

# %% 
# Let's look at the synthetic call parameters. There's a bunch of parameters
# that're not interesting for this accuracy exercise and so let's remove them 

synthesised

synthesised.columns

synth_regions = synthesised.loc[:,['cf_duration', 'upfm_duration','downfm_duration']]
synth_regions['other'] = np.nan
synth_regions['call_number'] = obtained['call_number']

# %% 
# Comparing the synthetic and the obtained results
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We have the two datasets formatted properly, now let's compare the 
# accuracy of `itsfm`.

accuracy = obtained/synth_regions
accuracy['call_number'] = obtained['call_number']
# %% 
# Overall accuracy of segmentation:
accuracy_reformat = accuracy.melt(id_vars=['call_number'], 
                                            var_name='Region type',
                                            value_name='Accuracy')
    
ax = sns.boxplot(x='Region type', y = 'Accuracy',
                         data=accuracy_reformat)

ax = sns.swarmplot(x='Region type', y = 'Accuracy',
                         data=accuracy_reformat,
                         alpha=0.5)

# %% 
# Some bad identifications
# ~~~~~~~~~~~~~~~~~~~~~~~~
# As we can see there are a few regions where the accuracy is very low, let's
# investigate which of these calls are doing badly. 

poor_msmts = accuracy[accuracy['cf_duration']<0.5].index

# %% 
# Now, let's troubleshooot this particular set of poor measurements fully.

simcall_params = pd.read_csv('horseshoe_test_parameters.csv')
obtained_params = pd.read_csv('obtained_horseshoe_sim.csv')

obtained_params.loc[poor_msmts,:]

# %% 
# There are two CF regions being recognised, one of them is just extremely short.
# Where is this coming from? Let's take a look at the actual frequency tracking output,
# by re-running the ```itsfm``` routine once more:


f = h5py.File('horseshoe_test.hdf5', 'r')

fs = float(f['fs'][:])

parameters = {}
parameters['segment_method'] = 'pwvd'
parameters['window_size'] = int(fs*0.001)
parameters['fmrate_threshold'] = 2.0
parameters['max_acc'] = 10
parameters['extrap_window'] = 75*10**-6

raw_audio = {}

synthesised = pd.read_csv('horseshoe_test_parameters.csv')
for call_num in tqdm.tqdm(poor_msmts.to_list()):
    synthetic_call = f[str(call_num)][:]
    raw_audio[str(call_num)] = synthetic_call
    output = itsfm.segment_and_measure_call(synthetic_call, fs, **parameters)
                                    
    seg_output, call_parts, measurements= output
    
    # # save the long format output into a wide format output to
    # # allow comparison
    # sub = measurements[['region_id', 'duration']]
    # sub['call_number'] = call_num
    # region_durations = sub.pivot(index='call_number',
    #                              columns='region_id', values='duration')
    # obtained.append(region_durations)

f.close()

call_num = str(poor_msmts[0])

plt.figure()
plt.specgram(raw_audio[call_num], Fs=fs)
plt.plot(np.linspace(0,raw_audio[call_num].size/fs,raw_audio[call_num].size),
         seg_output[2]['cleaned_fp'])
plt.plot(np.linspace(0,raw_audio[call_num].size/fs,raw_audio[call_num].size),
         seg_output[0]*4000,'w')
plt.plot(np.linspace(0,raw_audio[call_num].size/fs,raw_audio[call_num].size),
         seg_output[1]*4000,'k')


plt.figure()
plt.subplot(211)
plt.specgram(raw_audio[call_num], Fs=fs)
plt.plot(np.linspace(0,raw_audio[call_num].size/fs,raw_audio[call_num].size),
         seg_output[2]['raw_fp'])
plt.plot(np.linspace(0,raw_audio[call_num].size/fs,raw_audio[call_num].size),
         seg_output[0]*4000,'w')
plt.plot(np.linspace(0,raw_audio[call_num].size/fs,raw_audio[call_num].size),
         seg_output[1]*4000,'k')
plt.subplot(212)
plt.plot(raw_audio[call_num])



parameters = {}
parameters['segment_method'] = 'peak_percentage'
parameters['window_size'] = int(fs*0.001)
parameters['fmrate_threshold'] = 2.0
parameters['max_acc'] = 10
parameters['extrap_window'] = 50*10**-6
output = itsfm.segment_and_measure_call(synthetic_call, fs, **parameters)




