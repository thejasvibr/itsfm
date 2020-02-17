#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""tests for measure_single_horseshoe_bat_call
Created on Wed Feb 12 17:53:58 2020

@author: tbeleyur
"""
from measure_a_horseshoe_bat_call import *

import unittest     


class CheckIfMeasurementsCorrect(unittest.TestCase):

    def setUp(self):
        self.fm_durn = 0.01
        self.cf_durn = 0.2
        self.fs =250000
        self.test_call = make_one_CFcall(self.cf_durn+self.fm_durn,
                                         self.fm_durn, 90000,
                                    self.fs, call_shape='staplepin',
                                    fm_bandwidth=10*10**3.0)
        self.test_call *= 0.9
        self.test_call *= signal.tukey(self.test_call.size, 0.1)
        self.backg_noise = np.random.normal(0,10**-80.0/20, self.test_call.size+500)
        self.backg_noise[250:-250] += self.test_call
        
    def test_duration(self):
        '''checks if the duration output is ~correct '''
        
        sounds, msmts = measure_hbc_call(self.backg_noise, fs=self.fs)
        print(msmts, msmts.columns)
        print(msmts['call_duration'])
        print(msmts['upfm_end_time']-msmts['upfm_start_time'])
        

if __name__ == '__main__':
    unittest.main()