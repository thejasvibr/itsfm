# -*- coding: utf-8 -*-
"""
Tests for the view module

@author: tbeleyur
"""
import unittest
from measure_horseshoe_bat_calls.view_horseshoebat_call import itsFMInspector
from measure_horseshoe_bat_calls.simulate_calls import make_cffm_call
from measure_horseshoe_bat_calls.user_interface import segment_and_measure_call

class TestInspector(unittest.TestCase):
    
    def setUp(self):
        fm_props = (80000, 0.003)
        call_parameters = {'cf':(100000, 0.01),
                            'upfm':fm_props,
                            'downfm':fm_props
                            }
    
        fs = 500*10**3 # 500kHz sampling rate
        synthetic_call, _ = make_cffm_call(call_parameters, fs)
        common_parameters = {
        'segment_method':'pwvd',
        'fmrate_threshold':2,
        'percentile' : 99.5,
        'max_acc':10}

        outputs = segment_and_measure_call(synthetic_call, fs, common_parameters)
        
        self.viewer = itsFMInspector(outputs, synthetic_call, fs)
        
    def test_visualise_audio(self):
        self.viewer.visualise_call()
    def test_fmrate(self):
        self.viewer.visualise_fmrate()
    def test_acc(self):
        self.viewer.visualise_accelaration()
    def test_cffm_seg(self):
        self.viewer.visualise_cffm_segmentation()
    def test_fp(self):
        self.viewer.visualise_frequency_profiles()
    def test_geq_siglevel(self):
        self.viewer.visualise_geq_signallevel()          
        
        
        
        
        