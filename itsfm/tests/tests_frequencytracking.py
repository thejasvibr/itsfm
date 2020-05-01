# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 21:42:54 2020

@author: tbeleyur
"""
import unittest
from itsfm.frequency_tracking import *

class PWVDTracking(unittest.TestCase):
    
    def test_simple(self):
        input_signal = np.random.normal(0,1,1000)
        fs = 1000
        freqs, inds = generate_pwvd_frequency_profile(input_signal, fs)
        
        self.assertEqual(freqs.size, input_signal.size)


class GeneratePWVDProfile(unittest.TestCase):
    def setUp(self):
        self.test_signal = np.random.normal(0,1,10000)
        self.fs = 2000

    def test_Works(self):
        get_pwvd_frequency_profile(self.test_signal, self.fs)






if __name__ == '__main__':
    unittest.main()