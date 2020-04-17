# -*- coding: utf-8 -*-
"""Module that resolves the initial CF FM candidate regions according to 
different rules


All CF-FM candidate refinement functions follow the same pattern:

Parameters
----------
* cfm_candidates : list with 2 boolean np.arrays
    The first array is the CF candidate regions
    The second array si the FM candidate regions
* fs: float>0
    The sampling rate in Hz. 
* info : dictionary
    Segmentation method specific content. 

Keyword Arguments 
-----------------
as required by the refinement function.

Returns
-------
CF_refined, FM_refined : np.array
    Each refinement function can only output two np.arrays as
    two separate objects. 




"""




def do_nothing(cfm_candidates, fs, info, **kwargs):
    '''
    '''
    return cfm_candidates[0], cfm_candidates[1]
