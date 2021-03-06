itsfm : Identify, Track and Segment sound (by) Frequency (and its) Modulation
============================================================================

Introduction
~~~~~~~~~~~~
The `itsfm` package identifies regions of sound with and without frequency modulation, 
and allows custom measurements to be made on them. It's all in the name. Each of the 
task behind the identification, tracking and segmenting of a sound can be done independently.

The sounds could be bird, bat, whale, artifical sounds - it should hopefully work,
however be aware that this is an alpha version package at the moment. 

The basic workflow involves the tracking of a sounds frequency over time, and then calculating the 
rate of frequency modulation (FM), which is then used to decide which parts of a sound are frequency
modulated, and which are not. Here are some examples to show the capabilities of the package. 

The broad idea of this package is to achieve a loose coupling between the I,T, S in the package name. 
`itsfm` can do all or one of the below.

* I : Identify sounds by frequency modulation. An input audio can have multiple sounds in it, separated by silence or fainter regions. 
* T : Track the sound's frequency over time. The PWVD method allows tracking a sound's frequency with high temporal resolution. 
* S : Segment according to the frequency modulation. Calculates the local rate of frequency modulation over a sound and classifies parts of it 
  as frequency modulated (FM) or constant frequency (CF)

*Warning : The docs are constantly under construction, and is likely to change fairly regularly like the stairs in Hogwarts.
Do not be surprised by dramatic changes, but do come back regularly to see improvements!*



   
Let's cut to the chase : some examples *NOW*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
	:maxdepth:4
	:caption: Examples!
	
	gallery_dir/index.rst
	gallery_detailed/index.rst

.. toctree::
   :caption: Using itsfm without coding:
   
   no_coding.rst

.. toctree::
   :caption: itsfm Accuracy
   
   gallery_accuracy/index.rst

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
#. Handle complex and reverberant sounds. Sounds that are multi-component, ie, with multiple harmonics or 
   with variation in intensity of harmonics across the recording won't fare very well. 
#. Separate overlapping sounds

Installation
~~~~~~~~~~~~
This is a pre-PyPi version of the package. The easiest way to install the package is to head to this `page <https://github.com/thejasvibr/itsfm.git>`_, and 
download/clone the repository. Go into the downloaded folder and type :code:`python setup.py install`.


What the package could do with (future feature ideas):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. A sensible way to deal with edges of the signals. Right now the instantaneous frequencies suffer from spikes caused
   by bad instantaneous frequency estimates at the edges in the pseudo-wigner ville distribution method. 	
 
#. Informed frequency tracking (eg. Viterbi path or similar) in multi-harmonic sounds. Right now 
   the frequency profile of a sound is selected by independently choosing the first peak in the time-frequency
   slice. This prevents a sensible tracking of frequency because even slight variations in harmonic intensities
   over a sound can cause the peak frequency to jump almost an octave sometimes!

#. More time-frequency representation implementations and the signal cleaning methods associated with them. 

Why is everything in this codebase a function? Have you heard of classes?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is the author's first Python package, and the author admits it may not 
be the most elegant implementation. The author's previous experience (or lack thereof)  
working with classes may have left some bad memories :P.However the author also admits 
that many things in the package might have been less cumbersome with the use of classes,
and plans to implement it in due time.  

Where to get help
~~~~~~~~~~~~~~~~~
.. toctree::
	:maxdepth:2
	:caption: Errors

	common_errors.rst	
	

Hopefully this web page has enough information. Use the search bar to check if the error/issue 
you're encountering has already been documented. Also do check the examples to see
if the same error messages have been explained. If something's not clear or 
there's something not covered do write to me : thejasvib@gmail.com. I'll try to answer
within a week. 



I found a bug and/or have fixed something
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please raise an issue or pull request on Github 


Acknowledgements
~~~~~~~~~~~~~~~~

The PWVD transforms in itsFM rely on the `tftb <https://tftb.readthedocs.io/en/latest/index.html>`_ package by Jaidev Deshpande. 
I'd like to thank the Neetash MR, Aditya Krishna and Holger R. Goerlitz for helpful discussions that eventually lead down this path, and Diana 
Schoeppler for discussions that inspired the peak percentage method in this package. I'd also like to thank all the people who happily sent 
me example data and gave feedback whenever asked! 

License
~~~~~~~

MIT License

Copyright (c) 2020 Thejasvi Beleyur

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


.. toctree::
   :maxdepth:1
   :caption: API reference:

   userinterface.rst
   segmentation
   measurement
   view
   supportmodules

