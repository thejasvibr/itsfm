# -*- coding: utf-8 -*-
"""
Generating the CF-FM synthetic calls
====================================
Module that creates the data for accuracy testing horseshoe bat type calls

"""

import h5py
from itsfm.simulate_calls import make_cffm_call
import numpy as np 
import pandas as pd
import scipy.signal as signal 
from tqdm import tqdm

cf_durations = [0.005, 0.010, 0.015]
cf_peakfreq = [40000, 60000, 90000]
fm_durations = [0.001, 0.002]
fm_bw = [5000, 10000, 20000]

all_combinations = np.array(np.meshgrid(cf_peakfreq, cf_durations,
                                        fm_bw,fm_durations,
                                        np.flip(fm_bw),np.flip(fm_durations)))
all_params = all_combinations.flatten().reshape(6,-1).T

col_names = ['cf_peak_frequency', 'cf_duration',
             'upfm_bw', 'upfm_duration',
             'downfm_bw', 'downfm_duration']

parameter_space = pd.DataFrame(all_params, columns=col_names)
parameter_space['upfm_terminal_frequency'] = parameter_space['cf_peak_frequency'] - parameter_space['upfm_bw']
parameter_space['downfm_terminal_frequency'] = parameter_space['cf_peak_frequency'] - parameter_space['downfm_bw']

parameter_columns = ['cf_peak_frequency', 'cf_duration',
                     'upfm_terminal_frequency', 'upfm_duration',
                     'downfm_terminal_frequency', 'downfm_duration']

all_calls = {}
for row_number, parameters in tqdm(parameter_space.iterrows(),
                                   total=parameter_space.shape[0]):

    cf_peak, cf_durn, upfm_terminal, upfm_durn, downfm_terminal, downfm_durn = parameters[parameter_columns]
    call_parameters = {'cf':(cf_peak, cf_durn),
                        'upfm':(upfm_terminal, upfm_durn),
                        'downfm':(downfm_terminal, downfm_durn),
                        }

    fs = 250*10**3 # 500kHz sampling rate
    synthetic_call, _ = make_cffm_call(call_parameters, fs)
    synthetic_call *= signal.tukey(synthetic_call.size, 0.1)
    all_calls[row_number] = synthetic_call


# now save the data into an hdf5 file 
with h5py.File('horseshoe_test.hdf5','w') as f:
    for index, audio in all_calls.items():
        f.create_dataset(str(index), data=audio)
    f.create_dataset('fs', data=np.array([fs]))
parameter_space.to_csv('horseshoe_test_parameters.csv')
        