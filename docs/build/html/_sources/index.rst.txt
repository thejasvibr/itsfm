

measure horseshoe bat call : Documentation
=======================================================

.. include:: ../README.rst

.. toctree::
   :maxdepth: 2
   :caption: Contents:

How the package works
~~~~~~~~~~~~~~~~~~~~
The package assumes that any given sound could have a set of three elements

* Constant Frequency (CF) :  a pure tone like element
* Frequency Modulated (FM) :  an element with a varying frequency
* Gap : a silence between two elements that is neither a CF nor an FM. 

Any input audio goes through the following steps:

#. Time frequency localisation within audio: either one of STFT, WignerVille, instantaneous frequency 
#. FM rate calculation : regions with modulation below the set limit are considered CF, and those above the FM rate are considered FM. 
#. Rough candidate region assignment into CF and FM regions 
#. Refined sample-level assignment into CF,FM and Gap regions based on user-input constraints (max number of elements, presence/absence of gaps, etc)
#. Segmentation of assigned elements and measurements of various parameters (peak frequency, energy, duration,...)

High-level API
~~~~~~~~~~~~~~
.. automodule:: measure_horseshoe_bat_calls.user_interface
   :members:

Batch processing from batchfiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: measure_horseshoe_bat_calls.batch_processing
   :members:

Call -background and call-part segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: measure_horseshoe_bat_calls.segment_horseshoebat_call
   :members:

Call background and call-part segmentation: part 2 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: measure_horseshoe_bat_calls.pwvd.frequency_tracking_pwvd
	:members:

Call-parts measurements
~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: measure_horseshoe_bat_calls.measure_a_horseshoe_bat_call
   :members:

Call segmentation and measurements visualisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: measure_horseshoe_bat_calls.view_horseshoebat_call
   :members:

Checking accuracy
~~~~~~~~~~~~~~~~~
.. automodule:: measure_horseshoe_bat_calls.simulate_calls
	:members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
