# -*- coding: utf-8 -*-
"""Tests to check the CF-FM segmentation.

@author: tbeleyur
"""
import matplotlib.pyplot as plt
import unittest
from measure_horseshoe_bat_calls.segment_horseshoebat_call import *
from measure_horseshoe_bat_calls.simulate_calls import make_cffm_call

class TestSegmentationMethods(unittest.TestCase):
    '''Suite of tests that makes sure all the implemented CF-FM segmentation
    methods aren't broken
    '''
    
    def setUp(self):
        self.fs = 25000
        call_props = {'upfm':(8000, 0.002),
                      'cf':(9000, 0.01),
                      'downfm':(6000, 0.003)}
        self.call,_ = make_cffm_call(call_props, self.fs)
        self.kwargs = {}
        
        
    def test_peakpercentage_working(self):
        method = 'peak_percentage'
        cf_candidates, fm_candidates, info = perform_segmentation[method](self.call,
                                                                  self.fs,
                                                                  **self.kwargs)
    def test_pwvd_working(self):
        method = 'pwvd'
        cf_candidates, fm_candidates, info = perform_segmentation[method](self.call,
                                                                  self.fs,
                                                                  **self.kwargs
                                                                     )

    def test_inst_freq_working(self):
        method = 'inst_freq'
        with self.assertRaises(NotImplementedError):
            cf_candidates, fm_candidates, info = perform_segmentation[method](self.call,
                                                                  self.fs,
                                                                  **self.kwargs
                                                                    )



class TestFMrateCalculation(unittest.TestCase):
    '''All FM rate calculates are done with a tolerance of 
    10**-3 kHz/ms or lesser.
    
    TODO :
        1) implement fmrate calculation tests with noisy frequency profiles.
    '''
    
    
    def setUp(self):
        self.fs = 2000.0
        self.durn = 5
        self.start, self.stop = 10000, 20000
        self.frequency_profile = np.linspace(self.start, self.stop,
                                             int(self.durn*self.fs) ).flatten()
        
        self.fmrate = np.abs(np.gradient(self.frequency_profile))/(1/self.fs)
        self.fmrate_khz_per_ms = self.fmrate*10**-6
        self.max_fmrate_tolerance = 10**-3 # kHz/ms
        
        cf = np.tile(self.frequency_profile[-1], 1000)
        self.fm_and_cf = np.concatenate((self.frequency_profile, cf))
        
    def test_simple_sweep(self):
        fmrate_profile, _ = calculate_fm_rate(self.frequency_profile, self.fs, 
                                      medianfilter_length=0.002)

        fmr_profiles_same = np.allclose(fmrate_profile, self.fmrate_khz_per_ms,
                                        atol=self.max_fmrate_tolerance)
        self.assertTrue(fmr_profiles_same)
    def test_sweep_and_cf(self):
       
        
        fmcf_rate = np.abs(np.gradient(self.fm_and_cf))/(1/self.fs)
        fmcf_rate *= 10**-6
        
        fmrate_profile, _ = calculate_fm_rate(self.fm_and_cf, self.fs, 
                                      medianfilter_length=0.0015,
                                      sample_every=0.001)
        
        
        fmr_profiles_same = np.allclose(fmrate_profile, fmcf_rate,
                                        atol=self.max_fmrate_tolerance)
        self.assertTrue(fmr_profiles_same)
        
    def test_fmcf_with_noise(self):
        pass
        


class TestCustomRefinement(unittest.TestCase):
    
    def setUp(self):
        self.fm = np.ones(10).astype('bool')
        self.cf = self.fm.copy()
        self.fs = 1.0
        self.method = 'miaow'
        
    def test_string_input(self):
        with self.assertRaises(AttributeError):
            cf, fm = refine_cf_fm_candidates(self.method,
                                             [self.cf, self.fm],
                                             self.fs, {})
                         
    def test_donothing_string_input(self):
        self.method = 'do_nothing'
        cf, fm = refine_cf_fm_candidates(self.method,
                                         [self.cf, self.fm],
                                         self.fs, {})
        
    def random_custom_function(in1, in2, in3, in4):
        return in2, in3

    def test_refine_with_custom_function(self):
        self.method = self.random_custom_function
        cf, fm = refine_cf_fm_candidates(self.method,
                                         [self.cf, self.fm],
                                         self.fs, {})
        self.assertEqual(cf, [self.cf, self.fm])
        
        
class NoiseSuppressionCFFMSamples(unittest.TestCase):
    
    def setUp(self):
        self.audio = np.concatenate((np.tile(0.9, 1000),
                                np.tile(0.001, 1000))).flatten()
        self.m_rms = moving_rms_edge_robust(self.audio)
        self.cf_candidate = np.tile(True, self.audio.size)
        
        self.noise_level_dB = -6
        self.expected = self.m_rms > 10**(self.noise_level_dB/20.0)
        
        
    def test_simple_suppression(self):
        '''
        '''
        noise_suppressed = suppress_background_noise(self.cf_candidate, self.audio,
                                      signal_level=self.noise_level_dB)
        
        expected_match = np.array_equal(noise_suppressed, self.expected)
        plt.plot(self.expected)
        self.assertTrue(expected_match)
        


if __name__ == '__main__':
    unittest.main()