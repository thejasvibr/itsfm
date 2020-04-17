# -*- coding: utf-8 -*-
"""Tests for signal cleaning
Created on Thu Apr 16 08:43:47 2020

@author: tbeleyur
"""
import unittest
import numpy as np 
from measure_horseshoe_bat_calls.signal_cleaning import *

class CheckSmoothOverPothole(unittest.TestCase):
    '''
    
    '''

    def setUp(self):
        self.level_value = 10
        self.one_pothole = np.concatenate((np.tile(self.level_value,10), 
                                           np.zeros(3),
                                           np.tile(self.level_value,10)))
        self.pothole_samples = np.argwhere(self.one_pothole==0).flatten()
    
        self.fs = 1.0
        
        self.two_potholes_asymm = np.concatenate((self.one_pothole, np.zeros(2),
                                                      np.tile(self.level_value,2)))
    def test_basic(self):
        '''everything should be the same value. 
        '''
    
        pothole_covered, _ = smooth_over_potholes(self.one_pothole, self.fs, 
                                               max_stepsize=2,
                                               pothole_inspection_window=4)
        
        pothole_levelling_success = np.all(self.level_value==pothole_covered)
        self.assertTrue(pothole_levelling_success)
    
    def test_asymm(self):
        pothole_covered, _ = smooth_over_potholes(self.two_potholes_asymm, self.fs, 
                                               max_stepsize=2,
                                               pothole_inspection_window=4)
        
        pothole_levelling_success = np.all(self.level_value==pothole_covered)
        self.assertTrue(pothole_levelling_success)
    

class IdentifyPotholeSamples(unittest.TestCase):
    
    def setUp(self):
        self.one_pothole = np.concatenate((np.tile(10,10), 
                                           np.zeros(3),
                                           np.tile(10,10)))
        self.pothole_samples = np.argwhere(self.one_pothole==0).flatten()

        self.fs = 1.0
        
        self.two_potholes_asymm = np.concatenate((self.one_pothole, np.zeros(2),
                                                      np.tile(10,2)))
        
    def test_basic(self):
        potholes = identify_pothole_samples(self.one_pothole, self.fs,
                                 max_stepsize=2,
                                 pothole_inspection_window=4)
        pothole_output = np.argwhere(potholes).flatten()
        
        same_samples = np.array_equal(pothole_output, self.pothole_samples)
        self.assertTrue(same_samples)
        
    def test_asymm_w_shortandlong_potholes(self):
        potholes = identify_pothole_samples(self.two_potholes_asymm, self.fs,
                                 max_stepsize=2,
                                 pothole_inspection_window=4)
        
        expected_potholes = np.argwhere(self.two_potholes_asymm==0).flatten()
        obtained_pothols = np.argwhere(potholes).flatten()
        same_samples = np.array_equal(expected_potholes, obtained_pothols)
        self.assertTrue(same_samples)
        
        


class DetectLocalPotholes(unittest.TestCase):
    
    def setUp(self):
        self.eg_uniform = np.tile(2, 10)
        self.eg_all_above = np.arange(10)
        self.eg_all_below = np.random.choice(np.arange(0,2,0.1), 10)
        self.threshold = 0.1
        self.eg_just_threshold = np.arange(0,10,self.threshold)

    def test_all_below_threshold(self):
        output = np.invert(detect_local_potholes(self.eg_uniform, 1))
        self.assertTrue(np.all(output))
    def test_all_above(self):
        output = np.invert(detect_local_potholes(self.eg_all_above, 0.5))
        second_index_on = output[1:]
        all_above = np.all(second_index_on)
        self.assertTrue(all_above)
    def test_all_below(self):
        output = detect_local_potholes(self.eg_all_below, 2)
        
        all_below = np.invert(output)
        self.assertTrue(np.all(all_below))
    def test_just_at_max_step_size(self):
        output = detect_local_potholes(self.eg_just_threshold, self.threshold)
        all_below = np.invert(output)
        self.assertTrue(np.all(all_below))
        
        
    
        
    

if __name__  == '__main__':
    unittest.main()
        
