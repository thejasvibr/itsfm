# -*- coding: utf-8 -*-
"""
CF-FM call segmentation accuracy
================================
This page will illustrate the accuracy with which `itsfm` can segment CF-FM parts of a 
CF-FM call. To see what a CF-FM call looks like check out the bat-call example 
in the 'Basic Examples' page. 

The synthetic data has already been generated and run with the :code:`segment_and_measure`
function, and now we'll compare the accuracy with which it has all happened. Here we 
will only be seeing if the durations of each of the segment parts have been picked
up properly or not. We will *not* be performing any accuracy assessments on 
the exact parameters (eg. peak frequency, rms, etc) because it is assumed that 
if the call parts can be identified properly then the measurements will 
in turn be as expected. 

What happened before
~~~~~~~~~~~~~~~~~~~~
To see more on the details of the generation and running of the synthetic data 
see the modules `CF/FM call segmentation` and `Generating the CF-FM synthetic calls`

"""
import numpy as np
import pandas as pd 
import seaborn as sns

obtained = pd.read_csv('obtained_horseshoe_sim.csv')
synthesised = pd.read_csv('horseshoe_test_parameters.csv')

# %% 
# Let's look at the obtained regions and their durations
obtained.head()

# %% 
# We can see the output has each CF/FM region labelled by the order in which
# they're found. Let's re-label these to match the names of the synthesised
# call parameter dataframe. 'upfm' is fm1, 'downfm' is fm2. 

obtained.columns = ['call_number','cf_duration',
                    'upfm_duration', 'downfm_duration', 'other']

# %% 
# Let's look at the synthetic call parameters. There's a bunch of parameters
# that're not interesting for this accuracy exercise and so let's remove them 

synthesised.head()

synthesised.columns

synth_regions = synthesised.loc[:,['cf_duration', 'upfm_duration','downfm_duration']]
synth_regions['other'] = np.nan
synth_regions['call_number'] = obtained['call_number']

# %% 
# Comparing the synthetic and the obtained results
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We have the two datasets formatted properly, now let's compare the 
# accuracy of `itsfm`.

accuracy = obtained/synth_regions
accuracy['call_number'] = obtained['call_number']
# %% 
# Overall accuracy of segmentation:
accuracy_reformat = accuracy.melt(id_vars=['call_number'], 
                                            var_name='Region type',
                                            value_name='Accuracy')
    
ax = sns.boxplot(x='Region type', y = 'Accuracy',
                         data=accuracy_reformat)

ax = sns.swarmplot(x='Region type', y = 'Accuracy',
                         data=accuracy_reformat,
                         alpha=0.5)



# %% 
# Some bad identifications
# ~~~~~~~~~~~~~~~~~~~~~~~~
# As we can see there are a few regions where the accuracy is very low, let's
# investigate which of these calls are doing badly. 

accuracy[accuracy['cf_duration']<0.5]





