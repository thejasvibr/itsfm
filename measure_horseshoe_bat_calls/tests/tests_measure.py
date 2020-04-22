# -*- coding: utf-8 -*-
"""Tests for measure

@author: tbeleyur
"""

import unittest
from measure_horseshoe_bat_calls.measure_a_horseshoe_bat_call import *
from measure_horseshoe_bat_calls.measurement_functions import *

class ParseCFFM(unittest.TestCase):
    '''
    '''
    
    def setUp(self):
        self.cf = np.array([0,1,1,0,0,0,1,1,0]).astype('bool')
        self.fm = np.array([0,0,0,1,1,1,0,0,0]).astype('bool')
    
        self.empty_cf = np.zeros(self.cf.size)
    
    def test_parsecffm_works(self):
        ordered_cffm = parse_cffm_segments(self.cf, self.fm)

    def test_only_one_sundtype(self):
        ordered_cffm = parse_cffm_segments(self.empty_cf, self.fm)
        expected = [('fm1', slice(3,6,None))]
        
        correct = np.array_equal(np.array(expected),
                                 np.array(ordered_cffm))
        self.assertTrue(correct)

    def test_no_types(self):
        ordered_cffm = parse_cffm_segments(self.empty_cf, self.empty_cf)
    
class CheckMeasurementsWork(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(111) # to ensure the same random numbers are generated
        self.fs = 1.0
        self.call = np.random.normal(0,1,100)
        self.cf = np.concatenate((np.tile(0, 50), np.tile(1,50))).astype('bool')
        self.fm = np.invert(self.cf)
    def test_simple(self):
        # Get the default measurements by not specifying any measurements explicitly
        sound_segments, measures = measure_hbc_call(self.call, self.fs,
                                                        self.cf, self.fm)
    def test_custom_functions(self):
        sound_segments, measures = measure_hbc_call(self.call, self.fs,
                                                self.cf, self.fm,
                                                measurements=[measure_peak_amplitude,
                                                              measure_peak_frequency])
    
    def a_custom_measure_function(self,x,y,z,**kwargs):
        custom_key = kwargs.get('custom_key', 1000)
        return {'custom': custom_key}
    
    def test_custom_fn_withkwargs(self):
        sound_segments, measures = measure_hbc_call(self.call, self.fs,
                                                self.cf, self.fm,
                                                measurements=[self.a_custom_measure_function],
                                                custom_key='abracadabrs')
        

    def test_custom_terminal_freq(self):
        sound_segments, measures = measure_hbc_call(self.call, self.fs,
                                                self.cf, self.fm,
                                                measurements=[measure_terminal_frequency],
                                                terminal_frequency_threshold=-20)















if __name__ == '__main__':
    unittest.main()