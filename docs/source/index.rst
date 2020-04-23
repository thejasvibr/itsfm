<NAMETOBEDECIDED> : Documentation
==================================

The <NAMETOBEDECIDED> package identifies regions of sound with and without frequency modulation, 
and allows custom measurements to be made on them. The sounds could be bird, bat, whale, human, artifical 
sounds - it should hopefully work :P. 

The basic workflow involves the tracking of a sounds frequency over time, and then calculating the 
rate of frequency modulation (FM), which is then used to decide which parts of a sound are frequency
modulated, and which are not. Here are some examples to show the capabilities of the package. 

.. include:: gallery_dir/index.rst


What the package `does`:
~~~~~~~~~~~~~~~~~~~~~~~~

#. Identify sounds as being constant frequency or frequency modulated
#. The 'pwvd' segmentation method allows a sample-level frequency estimation, the 'frequency profile' of the sound
#. Generates an FM rate profile over the sound 
#. Performs *basic* outlier detection 

What the package `does not`:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Perform any kind of pattern detection/classification. The frequency profile of a sound is generated using 
   a percentile based threshold on each slice of the underlying Pseudo Wigner-Ville distribution. 
#. Handle complex and reverberant sounds well. Sounds that are multi-component, ie, with multiple harmonics or 
   with variation in intensity of harmonics across the recording won't fare very well. 
#. Separate overlapping sounds


API References
##############

.. toctree::
   :maxdepth: 2
  
   userinterface	
   segmentation
   measurement
   view
   supportmodules

