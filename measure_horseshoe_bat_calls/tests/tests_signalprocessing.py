# -*- coding: utf-8 -*-
"""Tests for signal processing 
Created on Sat Apr 18 14:37:13 2020

@author: tbeleyur
"""

import unittest 
import numpy as np 
from measure_horseshoe_bat_calls.signal_processing import *

class TestFormConsensusRMS(unittest.TestCase):
    
    def setUp(self):
        self.fwd = np.concatenate((np.linspace(0,6,7), np.tile(np.nan, 4)))
        self.bkwd = np.concatenate((np.linspace(10,4,7), np.tile(np.nan, 4)))
        
        two_rows = np.column_stack((self.fwd, self.bkwd[::-1]))
        self.expected  = np.apply_along_axis(np.nanmean, 1, two_rows )
    
    def test_basic(self):
        obtained = form_consensus_moving_rms(self.fwd, self.bkwd)
        print(obtained, self.expected)
        matching = np.array_equal(obtained, self.expected)
        self.assertTrue(matching)

    
    
    
    

if __name__ == '__main__':
    unittest.main()
        
        
        
        
        