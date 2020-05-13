Common Errors
~~~~~~~~~~~~~
Here are the most common errors and the probable causes for them. When I use the word 'bad' here, I mean it 
in the sense of `bad` for that particular signal! Especially while analysing bioacoustic recordings, a parameter
value that works for one recording may not necessarily work for another one! 


1. Bad `signal_level`
>>>>>>>>>>>>>>>>>>>>>

.. code::bash
    $ ValueError: No regions above signal level found!

Easy, reduce the `signal_level` and try again. 

2. Bad `signal_level`
>>>>>>>>>>>>>>>>>>>>>

.. code::bash

    $ IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices

This region is caused by a very small region of the signal being selected. The PWVD transform works by choosing a small window of samples
on the left and right of the current sample. If the region above `signal_level` is very small, and not greater than this small window
of samples this error is raised. By default, the `isfm` window size is set to the numebr of samples corresponding to 1ms. 

Alter `signal_level` or `window_size` to get a more continuous moving dB rms profile. See below also. 


3. Bad `signal_level` or `window_size`
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. code:: bash

    $ ValueError: Shape of array too small to calculate a numerical gradient, at least (edge_order + 1) elements are required



The actual signal in an audio file is detected by the segment of audio that's above a user-defined `signal_level`. When the 
`signal_level` is set poorly or results in very short chunks of audio (<3 samples), then typically this error is thrown:


This means there's a very short audio segment that's above the `signal_level`. This typically happens because the moving dB rms profile 
is too spiky, which means the signal level fluctuates very quickly above and below the threshold. The new `signal_level` is best re-set 
after inspecting the moving dB rms profile. 

The two options to fix this error are:

#.  increase `window_size` to get a smoother moving dB rms profile 

#. set a new `signal_level` which will make sure the moving dB rms profile is above it and matches the duration of the original signal 


Anomaly spans whole array
>>>>>>>>>>>>>>>>>>>>>>>>>

.. code:: bash

    $ ValueError: The anomaly spans the whole array - please check again

"Anomalies" in the `itsfm` package are regions in the frequency profile which are particularly rough. This means the 
accelaration of the frequency profile has gone beyond the `max_acc` threshold value. Most of the time anomalies
are small parts of the original signal. However, there may be times when an anomalous region spans the whole 
signal -- and thus this warning. 

A closer inspection of this particular audio file may reveal more.

#. Reduce the `signal_level` for this particular audio. When the `signal_level` is set too high, the frequency 
profile of irrelevant parts may be getting analysed, leading to odd and rough frequency profiles. 




