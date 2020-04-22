# -*- coding: utf-8 -*-
"""script which loads the raw data into numpy arrays

Created on Fri Mar 27 15:45:51 2020

@author: tbeleyur
"""
import glob
import os 
try:
	import soundfile as sf
except:
	print('There was an issue importing the soundfile library - please resolve this! Using alternate package (scipy)..')
	import scipy.io.wavfile as wav

folder_with_audio_files, file_name =os.path.split(os.path.abspath(__file__))
search_path = os.path.join(folder_with_audio_files,'*.WAV')
all_wav_files = glob.glob(search_path)
example_calls = []

for each in all_wav_files:
	try:
		audio, fs_original = sf.read(each)
	except:
		fs_original, audio = wav.read(each)
	file_name = os.path.split(each)[-1]
	audio_and_fs = (audio, fs_original)
	example_calls.append(audio_and_fs)

