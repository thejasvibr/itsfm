# -*- coding: utf-8 -*-
"""Tests to check the CF-FM segmentation.

@author: tbeleyur
"""
import unittest
from measure_horseshoe_bat_calls.segment_horseshoebat_call import perform_segmentation
from measure_horseshoe_bat_calls.simulate_calls import make_cffm_call

class TestSegmentationMethods(unittest.TestCase):
    '''Suite of tests that makes sure all the implemented CF-FM segmentation
    methods aren't broken
    '''
    
    def setUp(self):
        self.fs = 250000
        call_props = {'upfm':(80000, 0.002),
                      'cf':(90000, 0.01),
                      'downfm':(60000, 0.003)}
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
        cf_candidates, fm_candidates, info = perform_segmentation[method](self.call,
                                                                  self.fs,
                                                                  **self.kwargs
                                                                     )

    


if __name__ == '__main__':
    unittest.main()