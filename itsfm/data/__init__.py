# -*- coding: utf-8 -*-
"""script which loads the raw data into numpy arrays

Created on Fri Mar 27 15:45:51 2020

@author: tbeleyur
"""
import glob
import numpy as np
import os 
import warnings
try:
	import soundfile as sf
except:
    msg1 = '\n \n The package soundfile could not be imported properly. Check your installation.'
    msg2 = 'Using the scipy.io package for now.'
    warnings.warn(msg1+msg2)
    import scipy.io.wavfile as wav

folder_with_audio_files, file_name =os.path.split(os.path.abspath(__file__))
search_path = os.path.join(folder_with_audio_files)
all_wav_files = glob.glob(search_path+'/*.WAV') + glob.glob(search_path+'/*.wav')
example_calls = []

def has_positive_and_negative(X):
    '''
    '''
    return np.logical_and(np.sum(X<0)>0, np.sum(X>0)>0)



def normalise_to_pm1(X):
    '''
    '''
    dtype_X = X.dtype
    
    permitted_dtypes = [ np.dtype(each)  for each in ['float32', 'float16','float64']]
    
    
    if dtype_X  in permitted_dtypes:
        return X
    else:    
        signed = has_positive_and_negative(X)
        
        if signed:
            max_possible_value = np.iinfo(dtype_X).max
            Y = np.float32(X)
            Y /= max_possible_value
            return Y
        else:
            raise ValueError('Unsigned audio not yet handled')
            X_values = [np.min(X), np.max(X)]
            msg1 = 'Cannot handle this audio dtype: %s with value ranges'%(str(dtype_X))
            msg2 = '.  The array has min and max values %f %f'%(X_values[0], X_values[1])
            raise ValueError(msg1+msg2)

        
        
    return X

for each in all_wav_files:
    try:
        audio, fs_original = sf.read(each)
    except:
        fs_original, audio = wav.read(each)
    audio = normalise_to_pm1(audio)
    audio_and_fs = (audio, fs_original)
    example_calls.append(audio_and_fs)
       
   

