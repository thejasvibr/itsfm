# -*- coding: utf-8 -*-
"""Tests for signal cleaning
Created on Thu Apr 16 08:43:47 2020

@author: tbeleyur
"""
import unittest
import numpy as np
from itsfm.signal_cleaning import *

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
        
class TestAnomalyExtrapolation(unittest.TestCase):
    
    def setUp(self):
        self.actual = np.linspace(1,10,10)
        self.weird_region = [slice(0,3)]
        self.fs = 1
    def test_basic(self):
        self.extrapolated = np.zeros(self.actual.size)
        self.extrapolated[3:] = self.actual[3:]

        extrap =anomaly_extrapolation(self.weird_region, self.actual,3)
        self.extrapolated[self.weird_region[0]] = extrap
        
        output_match = np.array_equal(self.extrapolated, self.actual)
        self.assertTrue(output_match)

class TestAnomalyInterpolation(unittest.TestCase):
    
    def setUp(self):
        self.actual = np.linspace(1,10,10)
        self.weird_region = [slice(4,7)]
        self.fs = 1
    
    def test_basic(self):
        self.interpolated = np.zeros(self.actual.size)
        self.interpolated[:self.weird_region[0].start] = self.actual[:self.weird_region[0].start]
        self.interpolated[self.weird_region[0].stop:] = self.actual[self.weird_region[0].stop:]
        
        intp_values = anomaly_interpolation(self.weird_region, self.actual)
        
        self.interpolated[self.weird_region[0]] = intp_values
        
        same_output = np.array_equal(self.interpolated, self.actual)
        self.assertTrue(same_output)
        

class TestAnomalyType(unittest.TestCase):
    
    def setUp(self):
        self.input = np.linspace(0,100,100)
    
    def test_edge(self):
        edge_region = [slice(0,10)]
        anom_type = anomaly_type(edge_region, self.input)
        self.assertEqual(anom_type, 'edge')
    
    def test_2edge(self):
        edge_region = [slice(99,100)]
        anom_type = anomaly_type(edge_region, self.input)
        self.assertEqual(anom_type, 'edge')
    
    def test_island(self):
        edge_region = [slice(2,5)]
        anom_type = anomaly_type(edge_region, self.input)
        self.assertEqual(anom_type, 'island')
    
    def test_island2(self):
        edge_region = [slice(2,99)]
        anom_type = anomaly_type(edge_region, self.input)
        self.assertEqual(anom_type, 'island')
        
        
class TestExterpolateOverAnomalies(unittest.TestCase):
    
    def setUp(self):
        self.element = np.linspace(0,10,5000)
        self.input = np.concatenate((self.element[::-1], self.element))
        self.fs = 20000
        self.anomalous = np.concatenate((np.zeros(10, dtype='bool'),
                                         np.ones(4980, dtype='bool'),
                                         np.zeros(10, dtype='bool'))
                                        )

    def test_check_outputsize(self):
        extp = exterpolate_over_anomalies(self.input, self.fs, self.anomalous)
        self.assertEqual(extp.size, self.input.size)

    def test_check_correct_exterpolation(self):
        self.anomalous[30:40] = False
        extp = exterpolate_over_anomalies(self.input, self.fs, self.anomalous)
        same_output = np.array_equal(extp, self.input)


class TestClipTFR(unittest.TestCase):
    
    def setUp(self):
        self.tfr = np.ones((10,10))
        
    
    def test_lower_half_clipped(self):
        self.tfr[5:,:] *= 0.4
        
        clipped = clip_tfr(self.tfr, tfr_cliprange=6)
        
        tfr_exp = self.tfr.copy()
        tfr_exp[5:,:] = 10**(-6/20.0)
        match = np.allclose(tfr_exp, clipped)    
        
        self.assertTrue(match)
    def test_None_cliprange(self):
        '''when no cliprange is given the input tfr is
        returned as it is '''
        
        output = clip_tfr(self.tfr)
        match = np.array_equal(output, self.tfr)
        self.assertTrue(match)
    
class TestConditionallyInvert(unittest.TestCase):
    
    def setUp(self):
        self.x = np.array([True, True, False, False, True])
        self.y = np.array([0,0,10,10,10])
    
    def test_simple(self):
        cond_inv = conditionally_set_to(self.x, self.y<10, False)
        exp = np.array([False, False, False, False, True])
        match = np.array_equal(exp, cond_inv)
        self.assertTrue(match)
    
    def test_no_True(self):
        cond_inv = conditionally_set_to(self.x, self.y>100, False)
        exp = self.x.copy()
        match = np.array_equal(exp, cond_inv)
        self.assertTrue(match)
        
class TestGetLeftRightOfX(unittest.TestCase):
    
    def setUp(self):
        self.x = np.linspace(0,20,21)
        self.target = (slice(3,5,None),)
        self.ref_region = 10 
        
    
    def test_simple(self):
        left_right = get_neighbouring_regions(self.x, self.target, self.ref_region)
        exp_left_right = [np.arange(3), np.arange(5,15)]
        match = np.array_equal(np.concatenate(exp_left_right), 
                               np.concatenate(left_right))
        self.assertTrue(match)
                               
        
        
    def test_nextsimple(self):
        self.target = [slice(10,12,None)]
        self.ref_region = 3 
        left_right = get_neighbouring_regions(self.x, self.target, self.ref_region)
        exp_left_right = [np.arange(7,10), np.arange(12,15)]
        match = np.array_equal(np.concatenate(exp_left_right), 
                               np.concatenate(left_right))
        self.assertTrue(match)
        
class TestFixIslandAnomaly(unittest.TestCase):
    
    def setUp(self):
        self.fs = 250000
        parts = [np.zeros(5), np.arange(102,110)*1000]
        self.X = np.concatenate(parts)
        self.ref_size = 3
        self.anomaly = (slice(3,6,None),)
        self.anomaly_size = self.anomaly[0].stop - self.anomaly[0].start
        
    def test_call_start(self):
        output = fix_island_anomaly(self.X, self.fs, self.anomaly,
                           self.ref_size)
        exp = np.arange(100,103)*1000
        match = np.array_equal(exp, output)
        self.assertTrue(match, exp)
    
    
    def test_call_end(self):
        self.X = self.X[::-1]
        self.anomaly = (slice(6,10,None),)
        output = fix_island_anomaly(self.X, self.fs, self.anomaly,
                           self.ref_size)
        
        exp = np.arange(103,99,-1)*1000
        match = np.array_equal(exp, output)
        self.assertTrue(match, exp)
    
    def test_size_matches(self):
        output = fix_island_anomaly(self.X, self.fs, self.anomaly,
                           self.ref_size)
        self.assertEqual(self.anomaly_size, output.size)
        
        
    
    
    
        
        
        
        


if __name__  == '__main__':
    unittest.main()
        