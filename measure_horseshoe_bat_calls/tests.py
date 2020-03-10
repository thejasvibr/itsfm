#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""tests for measure_single_horseshoe_bat_call
Created on Wed Feb 12 17:53:58 2020

@author: tbeleyur
"""
from measure_a_horseshoe_bat_call import *
from segment_horseshoebat_call import *
import unittest     
#
#
#class CheckIfMeasurementsCorrect(unittest.TestCase):
#
#    def setUp(self):
#        self.fm_durn = 0.01
#        self.cf_durn = 0.2
#        self.fs =250000
#        self.test_call = make_one_CFcall(self.cf_durn+self.fm_durn,
#                                         self.fm_durn, 90000,
#                                    self.fs, call_shape='staplepin',
#                                    fm_bandwidth=10*10**3.0)
#        self.test_call *= 0.9
#        self.test_call *= signal.tukey(self.test_call.size, 0.1)
#        self.backg_noise = np.random.normal(0,10**-80.0/20, self.test_call.size+500)
#        self.backg_noise[250:-250] += self.test_call
#        
#    def test_duration(self):
#        '''checks if the duration output is ~correct '''
#        
#        sounds, msmts = measure_hbc_call(self.backg_noise, fs=self.fs)
#        print(msmts, msmts.columns)
#        print(msmts['call_duration'])
#        print(msmts['upfm_end_time']-msmts['upfm_start_time'])
#        

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
        
class TestIdentifyMainCall(unittest.TestCase):
    def setUp(self):
        self.fs = 250000
        self.call_durn = 0.050
        call = make_one_CFcall(0.050, 0.010, 100000,
                                   self.fs, call_shape='staplepin',
                                   fm_bandwidth=20000)
        call *= 10**(-20/20.0)
        noise = np.random.normal(0,10**-(40/20.0),int(self.call_durn*self.fs*3))
        lp = signal.butter(4, 12000/self.fs*0.5, 'lowpass')
        noise_lp = signal.filtfilt(lp[0],lp[1], noise)
        
        
        self.call_w_noise = noise_lp.copy()
        self.call_w_noise[int(self.call_durn*self.fs):int(2*self.call_durn*self.fs)] += call

    def test_clear_main_call_duration(self):
        '''a 150ms snippet with a 50ms call. 
        The call has 6dB SNR overall.
        '''
        main_call, difference_profile = identify_call_from_background(self.call_w_noise,
                                                                      self.fs)
        approximate_duration = np.round(np.sum(main_call)/float(self.fs),3)
        self.assertEqual(self.call_durn, approximate_duration)
    
    def test_one_single_main_call(self):
        '''
        '''
        main_call, difference_profile = identify_call_from_background(self.call_w_noise,
                                                                  self.fs)
        try:
            main_call_regions = identify_valid_regions(main_call, 2)
            self.fail('Main call has 2 segments - something weird going on')
        except:
            main_call_regions = identify_valid_regions(main_call, 1)
            pass

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
        print('Éasy:', accuracy)
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
        print('Tricky:', accuracy)
        within_error_range =  np.logical_and(accuracy<self.permitted_error_range[0],
                                       accuracy>self.permitted_error_range[1])
        self.assertTrue(np.all(within_error_range))
    
        
    
           
        
        
        
if __name__ == '__main__':
    unittest.main()
#    fs = 250000
#    call_durn = 0.050
#    call = make_one_CFcall(0.050, 0.0025, 100000,
#                               fs, call_shape='staplepin',
#                               fm_bandwidth=20000)
#    call *= 10**(-0/20.0)
#    noise = np.random.normal(0,10**-(20/20.0),int(call_durn*fs*3))
#    
#    call_w_noise = noise.copy()
#    call_w_noise[int(call_durn*fs):int(2*call_durn*fs)] += call
#
#    main_call, difference_profile = identify_call_from_background(call_w_noise,
#                                                                  fs,
#                                                                  background_frequency=20000)
#    print('MIAAAOW', np.sum(main_call)/fs)