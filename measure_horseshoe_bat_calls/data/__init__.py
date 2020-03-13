import glob
import os 
import scipy.signal as signal 
try:
	import soundfile as sf
except:
	print('There was an issue importing the soundfile library - please resolve this! Using alternate package (scipy)..')
	import scipy.io.wavfile as wav

all_wav_files = glob.glob('measure_horseshoe_bat_calls/data/'+'*.WAV')
example_calls = []

# all raw audio files here were recorded by a Avisoft DAC running at 
# 250kHz - this doesn't give very good results in segmentation. 
# To overcome this in the example dataset - resample all audio to 
# 500 kHz
for each in all_wav_files:
	try:
		audio, fs_original = sf.read(each)
	except:
		fs_original, audio = wav.read(each)
	# resample on the spot
	audio_ups = signal.resample(audio, audio.size*2)
	file_name = os.path.split(each)[-1]
	example_calls.append(audio_ups)
fs = fs_original*2
