# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 21:42:54 2020

@author: tbeleyur
"""
import unittest
from measure_horseshoe_bat_calls.frequency_tracking import *

class PWVDTracking(unittest.TestCase):
    
    def test_simple(self):
        input_signal = np.random.normal(0,1,1000)
        fs = 1000
        freqs, inds = generate_pwvd_frequency_profile(input_signal, fs)
        
        self.assertEqual(freqs.size, input_signal.size)






if __name__ == '__main__':
    unittest.main()