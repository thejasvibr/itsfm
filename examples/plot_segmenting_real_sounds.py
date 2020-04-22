"""
Analysing real recordings
============================================================
The <INSERTNEWNAME> package has many example recordings of bat calls thanks to
the generous contributions of bioacousticians around the world:
"""

from measure_horseshoe_bat_calls.run_example_analysis import contributors
print(contributors)

# %% Let's load the example data from the `data` module of the package

from measure_horseshoe_bat_calls.data import example_calls
from measure_horseshoe_bat_calls.view_horseshoebat_call import visualise_call

# %% `The example_calls` object is a list with many recording snippets. Each
# recording snippet is comprised of a tuple with the audio as a numpy array
# and the sampling rate of the audio. 

# take an example R. mehelyi?/euryale? call recording. 

first_call = example_calls[10]
audio, fs = first_call[0], first_call[1]
w,s = visualise_call(audio,fs, fft_size=128)

# set the ylim of the spectrogram narrow to check out the call in more detail
s.set_ylim(60000, 125000)

# %%
# Now, let's segment and get some basic measurements from this call

