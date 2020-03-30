# -*- coding: utf-8 -*-
"""User-friendly higher level functions
Created on Sat Mar 28 10:40:46 2020

@author: tbeleyur
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import measure_horseshoe_bat_calls.segment_horseshoebat_call 
from measure_horseshoe_bat_calls.segment_horseshoebat_call import segment_call_from_background
from measure_horseshoe_bat_calls.segment_horseshoebat_call import segment_call_into_cf_fm
from measure_horseshoe_bat_calls.measure_a_horseshoe_bat_call import measure_hbc_call


def segment_and_measure_call(main_call, fs, **kwargs):
    '''
    Parameters
    ----------
    main_call : np.array
    fs : float>0
        sampling rate in Hz

    Keyword Arguments
    -----------------
    see segment_call_into_cf_fm and measure_hbc_call

    Returns
    -------
    segmentation_outputs : tuple
        The outputs of segment_call_into_cf_fm in a tuple
    call_parts : dictionary
        Dictionary with 'cf' and 'fm' entries and corresponding 
        audio. 
    measurements : pd.DataFrame
        A single row with all the measurements. 
   
    '''
    cf, fm, info = segment_call_into_cf_fm(main_call, fs, 
                                                           **kwargs)

    call_parts, measurements = measure_hbc_call(main_call, fs, cf, fm, 
                                                        **kwargs)
    
    return (cf, fm, info), call_parts, measurements

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
    file_name : str. 
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

