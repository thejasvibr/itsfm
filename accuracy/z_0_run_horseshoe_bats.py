# -*- coding: utf-8 -*-
"""
Running CF-FM call segmentation
===============================
Here we will run the :code:`segment_and_measure` function and store the 
results of how long each Cf/FM segment is. 

Dataset creation
~~~~~~~~~~~~~~~~
The synthetic dataset has already been created in a separate module. 
See 'Generating the CF-FM synthetic calls' in the main page. 

It can take long
~~~~~~~~~~~~~~~~
We're running a few hundred synthetic audio clips with a few seconds (1-10s)
needed per iteration. This could mean, it might take a while(5,10 or more minutes)!
"""

import h5py
import itsfm
import pandas as pd
from tqdm import tqdm

# %% 
# Now, let's load each synthetic call and proceed to save the 
# results from the PWVD and peak-percentage based methods.


# %% 
# FM rate based segmentation 
# --------------------------

obtained = []

f = h5py.File('horseshoe_test.hdf5', 'r')
synthesised = pd.read_csv('horseshoe_test_parameters.csv')


fs = float(f['fs'][:])

parameters = {}
parameters['segment_method'] = 'pwvd'
parameters['window_size'] = int(fs*0.001)
parameters['fmrate_threshold'] = 2.0
parameters['max_acc'] = 10
parameters['extrap_window'] = 75*10**-6

for call_num in tqdm(range(synthesised.shape[0])):
    synthetic_call = f[str(call_num)][:]
    output = itsfm.segment_and_measure_call(synthetic_call, fs, **parameters)
                                    
    seg_output, call_parts, measurements= output
    # save the long format output into a wide format output to
    # allow comparison
    sub = measurements[['region_id', 'duration']]
    sub['call_number'] = call_num
    region_durations = sub.pivot(index='call_number',
                                 columns='region_id', values='duration')
    obtained.append(region_durations)

all_obtained = pd.concat(obtained)

all_obtained.to_csv('obtained_pwvd_horseshoe_sim.csv')

# %% 
# Peak-percentage based segmentation
# ----------------------------------

pkpctg_parameters = {}
pkpctg_parameters['segment_method'] = 'peak_percentage'
pkpctg_parameters['peak_percentage'] = 0.99
pkpctg_parameters['window_size'] = 125
pkpctg_parameters['double_pass'] = True

pkpct_obtained = []

for call_num in tqdm(range(synthesised.shape[0])):
    synthetic_call = f[str(call_num)][:]
    output = itsfm.segment_and_measure_call(synthetic_call, fs, **pkpctg_parameters)
                                    
    seg_output, call_parts, measurements= output
    # save the long format output into a wide format output to
    # allow comparison
    sub = measurements[['region_id', 'duration']]
    sub['call_number'] = call_num
    region_durations = sub.pivot(index='call_number',
                                 columns='region_id', values='duration')
    pkpct_obtained.append(region_durations)


f.close()

pk_pctage = pd.concat(pkpct_obtained)

pk_pctage.to_csv('obtained_pkpct_horseshoe_sim.csv')
