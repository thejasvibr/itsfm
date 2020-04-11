# -*- coding: utf-8 -*-
"""Once the frequency profile have been calculated, its
'velocity' must be measured to get the rate of frequency modulation. Once 
the FM rate can be calculated, the CF and FM parts of the sound can be 
segmented accurately. 

The primary problems with estimated frequency profiles are :
    
#. Spikiness due to edge effects : when the background wasn't succesfully isolated 
   from the signal. This appears especially at the edges of the sound segments. 

#. Repeated back-and-forth between somewhat disparate frequencies. Over the course of one samples,
   the instantaneous frequency can switch upto a few kHz. This is typically caused
   by harmonics or noise causing the well-known interference terms in the PWVD. 
   This typically occurs in the middle of the signal itself. 

"""



def correct_for_end_spikes():
    pass


def correct_for_repeated_spikes():
    pass


def correct_for_miaow():
    pass


