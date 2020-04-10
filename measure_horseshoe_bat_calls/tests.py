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

class check_2_5dB_accuracy_of_cffm(unittest.TestCase):
    '''Make sure that the accuracy of 
    FM and CF segmentation is within +/- 2.5 dB (1.33-0.75 times accuracy)

    '''
    def setUp(self):
        self.permitted_error_range = (1/0.75, 0.75)
        self.whole_call = 0.05
        self.fm_durn = 0.005
        self.cf_durn = self.get_cf_durn()
        self.fs = 250000
        
    def get_cf_durn(self):
        return self.whole_call - 2*self.fm_durn

    def test_basic(self):
        eg_call = make_one_CFcall(self.whole_call, self.fm_durn, 100000,
                                  self.fs, call_shape='staplepin',
                                  fm_bandwidth=15000)
        
        
        cf, fm, info = segment_call_into_cf_fm(eg_call, self.fs,
                                               peak_percentage=0.99,
                                               min_fm_duration=0.0001)
        
        halfway =int(fm.size/2.0)
        
        obtained_cf_durn = np.round(np.sum(cf)/float(self.fs), 4)
        
        obtained_upfm_durn = np.round(np.sum(fm[:halfway])/float(self.fs),4)
        obtained_downfm_durn = np.round(np.sum(fm[halfway:])/float(self.fs),4)
        
        obtained = np.array([obtained_cf_durn, 
                                   obtained_upfm_durn,
                                   obtained_downfm_durn])
        expected = np.array([self.cf_durn,
                            self.fm_durn,
                            self.fm_durn])
        accuracy = obtained/expected
        within_error_range =  np.logical_and(accuracy<self.permitted_error_range[0],
                                       accuracy>self.permitted_error_range[1])
        self.assertTrue(np.all(within_error_range))
    
    def test_short_fm(self):
        self.whole_call = 0.025
        self.fm_durn = 0.001
        self.cf_durn = self.get_cf_durn()
        
        eg_call = make_one_CFcall(self.whole_call, self.fm_durn, 100000,
                                  self.fs, call_shape='staplepin',
                                  fm_bandwidth=15000)
        
        
        cf, fm, info = segment_call_into_cf_fm(eg_call, self.fs,
                                               peak_percentage=0.99)
        
        halfway =int(fm.size/2.0)
        
        obtained_cf_durn = np.round(np.sum(cf)/float(self.fs), 4)
        
        obtained_upfm_durn = np.round(np.sum(fm[:halfway])/float(self.fs),4)
        obtained_downfm_durn = np.round(np.sum(fm[halfway:])/float(self.fs),4)
        
        obtained = np.array([obtained_cf_durn, 
                                   obtained_upfm_durn,
                                   obtained_downfm_durn])
        expected = np.array([self.cf_durn,
                            self.fm_durn,
                            self.fm_durn])
        accuracy = obtained/expected
        within_error_range =  np.logical_and(accuracy<self.permitted_error_range[0],
                                       accuracy>self.permitted_error_range[1])
        self.assertTrue(np.all(within_error_range))
    
        

class call_background_segmentation(unittest.TestCase):
    '''Check that the wavelet method of call-background 
    segmentation is working as expected.
    
    '''
    def setUp(self):
        self.whole_call = 0.05
        self.fm_durn = 0.005
        self.fs = 500000
        self.permitted_range = np.array([1.05, 0.95])

    def check_if_call_duration_matches(self,call, call_w_noise, **kwargs):
        
        actual_call_duration = call.size/float(self.fs)
        main_call, _ = segment_call_from_background(call_w_noise, self.fs,
                                                 window_size=50,
                                                 background_frequency=50000,
                                                 **kwargs)

        obtained_call_duration = np.sum(main_call)/float(self.fs)
        relative_error = obtained_call_duration/actual_call_duration
        self.assertTrue(np.logical_and(relative_error<=self.permitted_range[0],
                                       relative_error>=self.permitted_range[1])
                        )

    def test_basic_call_segmentation(self):
        '''
        '''
        call = make_one_CFcall(self.whole_call, self.fm_durn,
                               fs=self.fs,
                               cf_freq=100000,
                               call_shape='staplepin',
                               fm_bandwidth=15000)
        
        one_ms = int(0.001*self.fs)
        call_w_noise = np.random.normal(0,10**-60/20.0,
                                 call.size + 10*one_ms )
        call_w_noise[5*one_ms:-one_ms*5] += call
        self.check_if_call_duration_matches(call, call_w_noise)
    
    def test_noisy_call_segmentation(self):
        '''Make the call-background SNR just 10dB 
        and see if it can recover the correct duration
        '''
        call = make_one_CFcall(self.whole_call, self.fm_durn,
                               fs=self.fs,
                               cf_freq=100000,
                               call_shape='staplepin',
                               fm_bandwidth=15000)
        
        call_dbrms = dB(rms(call))
        one_ms = int(0.001*self.fs)
        call_w_noise = np.random.normal(0,10**(call_dbrms-10)/20,
                                 call.size + 2*one_ms )
        call_w_noise[one_ms:-one_ms] += call
        self.check_if_call_duration_matches(call, call_w_noise,
                                            background_threshold=-10)
        
        
class TestGetFMSnippets(unittest.TestCase):
    
    def setUp(self):
        fm_duration = 0.001
        cf_duration = 0.01
        
        self.cf_startstop = (fm_duration, fm_duration+cf_duration)
        self.fs = 50000.0
        self.num_fm_samples = int(self.fs*fm_duration)
        self.num_cf_samples = int(self.fs*cf_duration)
        
        t_fm = np.linspace(0, fm_duration, self.num_fm_samples)
        upfm_sweep = signal.chirp(t_fm, 0, t_fm[-1], 15000)
        downfm_sweep = np.flip(upfm_sweep)
        cf_audio = np.zeros(self.num_cf_samples)
        
        fm_mask = np.tile(True, int(self.fs*fm_duration))
        
        self.wholecall = np.concatenate((upfm_sweep, cf_audio, downfm_sweep))
        
        self.fm_2segment = np.concatenate((fm_mask,
                                           np.tile(False,self.num_cf_samples),
                                           fm_mask))
        
        self.fm_1segment_down = np.concatenate((np.tile(False, self.num_cf_samples+self.num_fm_samples),
                             fm_mask)         )

        self.fm_1segment_up = np.flip(self.fm_1segment_down)
        

    def check_type_and_index_match(self,expected_types, expected_snippets,
                                    fm_types, fm_snippets):
        
        types_match = fm_types == expected_types
        indices_match = np.array_equal( np.concatenate((fm_snippets)),
                                       np.concatenate(expected_snippets))
        type_and_index_match = np.all([types_match, indices_match])
        self.assertTrue(type_and_index_match)
        
    
    def test_simple_get2fms(self):
        fm_types, fm_snippets, fmstartstop = get_fm_snippets(self.wholecall,
                                                self.fm_2segment,
                                                self.fs,
                                                self.cf_startstop)
        
        expected_types = ['upfm_', 'downfm_']
        expected_snippets= [
                            self.wholecall[:self.num_fm_samples],
                            self.wholecall[-self.num_fm_samples:]
                            ]

        self.check_type_and_index_match(expected_types, expected_snippets,
                                        fm_types, fm_snippets)
    
    def test_simple_1fmdown(self):
        fm_types, fm_snippets, fmstartstop = get_fm_snippets(self.wholecall, 
                                                self.fm_1segment_down,
                                                self.fs,
                                                self.cf_startstop)
        
        expected_types = ['downfm_']
        expected_snippets= [self.wholecall[-self.num_fm_samples:]]
        
        self.check_type_and_index_match(expected_types, expected_snippets,
                                        fm_types, fm_snippets)
    def test_simple_1fmup(self):
        fm_types, fm_snippets, fmstartstop = get_fm_snippets(self.wholecall, 
                                                self.fm_1segment_up,
                                                self.fs,
                                                self.cf_startstop)
        
        expected_types = ['upfm_']
        expected_snippets= [self.wholecall[:self.num_fm_samples]]
        
        self.check_type_and_index_match(expected_types, expected_snippets,
                                        fm_types, fm_snippets)



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
