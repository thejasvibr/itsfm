# -*- coding: utf-8 -*-
"""User-friendly top-level functions which allow the user to handle 
    
#. Call-background segmentation
#. CF-FM call part segmentation
#. Measurement of CF-FM audio parts 

Let's take a look at an example where we [TO BE COMPLETED!!!]

.. code-block:: python

    import scipy.signal as signal 
    from measure_horseshoe_bat_calls.user_interface import segment_and_measure_call
    from measure_horseshoe_bat_calls.view_horseshoebat_call import *
    from measure_horseshoe_bat_calls.simulate_calls import make_cffm_call

    # create synthetic call 
    call_parameters = {'cf':(100000, 0.01),
                        'upfm':(80000, 0.002),
                        'downfm':(60000, 0.003),
                        }
    
    fs = 500*10**3 # 500kHz sampling rate 
    synthetic_call, freq_profile = make_cffm_call(call_parameters, fs) 

    # window and reduce overall signal level
    synthetic_call *= signal.tukey(synthetic_call.size, 0.1)
    synthetic_call *= 0.75
    
    # measuring a well-selected call (without silent background)
    
    
    
    # measuing a call with a silent background
    
    # and add 2ms of additional background_noise of ~ -60dBrms
    samples_1ms = int(fs*0.001)
    final_size = synthetic_call.size + samples_1ms*2
    call_with_noise = np.random.normal(0,10**(-60/20.0),final_size)
    call_with_noise[samples_1ms:-samples_1ms] += synthetic_call
    
    # 
    
    seg_and_msmts = segment_and_measure_call(call_with_noise, fs,
                                             segment_from_background=True)
    call_segmentation, call_parts, measurements, backg_segment = seg_and_msmts

"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import measure_horseshoe_bat_calls.segment_horseshoebat_call 
from measure_horseshoe_bat_calls.segment_horseshoebat_call import segment_call_from_background
from measure_horseshoe_bat_calls.segment_horseshoebat_call import segment_call_into_cf_fm
from measure_horseshoe_bat_calls.measure_a_horseshoe_bat_call import measure_hbc_call


def segment_and_measure_call(main_call, fs,
                             segment_from_background=False, 
                             **kwargs):
    '''Segments the CF and FM parts of a call and then 
    proceeds to measure their characteristics. If required, 
    also segments call from background. 

    Parameters
    ----------
    main_call : np.array
    fs : float>0
        sampling rate in Hz
    segment_from_background : boolean
        Whether to segment the call in the main_call audio. 
        Defaults to False.
    

    Keyword Arguments
    -----------------
    For further keyword arguments see segment_call_from_background,
    segment_call_into_cf_fm and measure_hbc_call

    Returns
    -------
    segmentation_outputs : tuple
        The outputs of segment_call_into_cf_fm in a tuple
    call_parts : dictionary
        Dictionary with 'cf' and 'fm' entries and corresponding 
        audio. 
    measurements : dictionary
        All the measurements from the FM and CF parts. 
        For details see measure_a_horseshoe_bat_call
    backg_segmentation_info : dictionary
        Contains the output from segment_call_from_background + a copy of 
        main_call, the raw un-segmented audio with background. The objects are
        accessed with the following keys:
        'raw_audio', 'call_mask', 'call_backg_info'.

    '''
    backg_segmentation_info = {}
    if segment_from_background:
        raw_audio = main_call.copy()
        call_portion, call_backg_info = segment_call_from_background(raw_audio, 
                                                              fs, **kwargs)
        main_call = main_call[call_portion]
        backg_segmentation_info['raw_audio'] = raw_audio
        backg_segmentation_info['call_mask'] = call_portion
        backg_segmentation_info['call_backg_info'] = call_backg_info

    cf, fm, info = segment_call_into_cf_fm(main_call, fs, 
                                                           **kwargs)

    call_parts, measurements = measure_hbc_call(main_call, fs, cf, fm, 
                                                        **kwargs)
    
    return (cf, fm, info), call_parts, measurements, backg_segmentation_info

def save_overview_graphs(all_subplots, analysis_name, file_name, index,
                         **kwargs):
    '''Saves overview graphs. 

    Parameters
    ----------
    all_subplots : list
        List with plt.subplot objects in them. 
        For each figure to be saved, one subplot object is enough.
    analysis_name : str
        The name of the analysis. If this funciton is called 
        through a batchfile, then it becomes the name of the 
        batchfile
    file_name : str
    index : int, optional
        A numeric identifier for each graph. This is especially relevant
        for analyses driven by batch files as there may be cases where the 
        calls are selected from the same audio file but in different parts. 

    Returns
    -------
    None
    
    Notes
    -----
    This function has the main side effect of saving all the input figures
    into a pdf file with >1 pages (one page per plot) for the user to inspect 
    the results.
    
    Example
    ---------
    import numpy as np 
    
    # 1st plot
    plt.figure()
    a = plt.subplot(211)
    plt.plot([1,2,3])
    b = plt.subplot(212)
    plt.plot([5,4,3])
    
    #2nd plot
    plt.figure()
    c = plt.subplot(121)
    plt.plot(np.random.normal(0,1,100))
    d = plt.subplot(122)
    plt.plot(np.random.normal(0,1,10))
    
    save_overview_graphs([a,c], 'example_plots', 'example_file',0)
    '''
    
    final_file_name = analysis_name+'_'+file_name+'_'+str(index)
        
    # thanks to J0e3gan : https://stackoverflow.com/a/17788764
    pdf = matplotlib.backends.backend_pdf.PdfPages(final_file_name+".pdf")
    
    for one_subplot in all_subplots: 
        pdf.savefig(one_subplot.figure)
    pdf.close()

