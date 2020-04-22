#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""tests for measure_single_horseshoe_bat_call
Created on Wed Feb 12 17:53:58 2020

@author: tbeleyur
"""
import scipy.signal as signal 
from measure_horseshoe_bat_calls.measure_a_horseshoe_bat_call import *
from measure_horseshoe_bat_calls.segment_horseshoebat_call import *
from measure_horseshoe_bat_calls.signal_processing import *
from measure_horseshoe_bat_calls.simulate_calls import make_cffm_call,make_FM_with_joint
import unittest     

class TestMovingRMS(unittest.TestCase):
    
    def setUp(self):
        self.test_signal = np.ones(1250)
        self.start_value = 0.5
        self.end_value = 0.8
        self.test_signal[:100] = np.ones(100)*self.start_value
        self.test_signal[-100:] =np.ones(100)*self.end_value

    def test_rms_for_edges(self):
        small_window_size = 25
        mov_rms = moving_rms_edge_robust(self.test_signal, 
                                        window_size=small_window_size)
        
        rms_start_end = np.array([mov_rms[0], mov_rms[-1]]).flatten()
        expected_values = np.array([self.start_value,
                                    self.end_value])

        values_match = np.array_equal(rms_start_end, 
                        expected_values)
        self.assertTrue(values_match)

class TestIdentifyContiguousRegions(unittest.TestCase):
    def setUp(self):
        self.eg_conditions = np.array([0,1,0,1,1,0,1,1,1], dtype='bool')
        self.null_conditions = np.array([0,0,0,0,0,0], dtype='bool')
        self.full_conditions = np.ones(10, dtype='bool')
    
    def test_basic(self):
        region_ids, region_info = identify_maximum_contiguous_regions(self.eg_conditions, 3)
        obtained_regions, obtained_samples = np.unique(region_info[:,0], return_counts=True)

        expected_regions = np.array([1,2,3]) 
        expected_number_samples = np.array([1,2,3])
        expected_regions_and_samples = np.concatenate((expected_regions, 
                                                       expected_number_samples))
        
        obtained = np.concatenate((obtained_regions,
                                   obtained_samples))
        self.assertTrue(np.array_equal(expected_regions_and_samples, 
                                       obtained))
    def test_null(self):
        with self.assertRaises(ValueError) as error:
            region_ids, region_info = identify_maximum_contiguous_regions(self.null_conditions, 1)
        self.assertEqual(str(error.exception), 
                         'No regions satisfying the condition found: all entries are False')
    def test_full(self):
        region_ids, region_info = identify_maximum_contiguous_regions(self.full_conditions, 
                                                                      1)
        regions, region_counts = np.unique(region_info[:,0], return_counts=True)
        
        obtained_regions_and_counts = np.concatenate((regions, region_counts))
        expected = np.concatenate(([0], [self.full_conditions.size]))
        self.assertTrue(np.array_equal(obtained_regions_and_counts, expected))


class TestGetFMRegions(unittest.TestCase):
    def setUp(self):
        self.good = np.array([1,1,1,0,0,0,0,1,1,1], dtype='bool')
        self.short_and_longfm = np.array([1,0,0,0,0,1,1,1], dtype='bool')
        self.short_and_shortfm = np.array([1,0,0,0,0,1], dtype='bool')
        self.only_1_fm = np.array([1,0,0,0,0,1], dtype='bool')
   
    def test_2fms(self):
        valid_fm = get_fm_regions(self.good, fs=1.0, min_fm_duration=3.0)
        input_and_output_same = np.array_equal(valid_fm, self.good)
        self.assertTrue(input_and_output_same)
    
    def test_invalid_and_onefms(self):
        valid_fm = get_fm_regions(self.short_and_longfm, 
                                  fs=1.0, min_fm_duration=3.0)
        expected = np.bool8(np.concatenate((np.zeros(5),np.ones(3))))
        input_and_output_same = np.array_equal(valid_fm, expected)
        self.assertTrue(input_and_output_same)
    
    def test_nofm(self):
        valid_fm = get_fm_regions(self.short_and_shortfm, 
                                  fs=1.0, min_fm_duration=3.0)
        expected = np.zeros(self.short_and_shortfm.size, dtype='bool')
        input_and_output_same = np.array_equal(valid_fm, expected)
        self.assertTrue(input_and_output_same)

    
       

#### Tesing the segment_call_into_cf_fm
class CheckingDifferentSegmentationMethods(unittest.TestCase):
    
    def setUp(self):
        call_properties = {'upfm':(98000, 0.001),
                           'cf':(100000, 0.005),
                           'downfm':(85000, 0.002)}
        self.fs = 500000
        self.cffm_call, _ = make_cffm_call(call_properties, self.fs)
    
    def test_peak_percentage_method_works(self):
        '''Check that perform_segmentation doesn't raise an error
        '''
        cf, fm, _ = perform_segmentation['peak_percentage'](self.cffm_call,
                                                self.fs)
        

        
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
