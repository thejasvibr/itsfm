# -*- coding: utf-8 -*-
"""Code which runs an example segmentation workflow for 
a bunch of CF-FM bat calls

Created on Fri Mar 27 15:04:49 2020

@author: tbeleyur
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import  measure_horseshoe_bat_calls.data as calls
import measure_horseshoe_bat_calls.view_horseshoebat_call as view

folder, file_path = os.path.split(os.path.abspath(__file__))

print(folder)
people_file_path = os.path.join(folder, 'data_contributors.csv')
print(people_file_path)

contributors = pd.read_csv(people_file_path)['people']

def run_example_data(example_calls=calls, people=contributors):
    # choose the first 10 calls
    some_calls = example_calls.example_calls[:10]
    fs = calls.fs
    plt.ioff()
    # save the spectrogram of the current call
    # REMEMBER TO ADD ANY OTHER PEOPLE!!! 
    print('This example data run was possible by the generous contributions of:')
    people_alphabetical = people.sort_values()
    print(people_alphabetical.to_string(index=False))

    for i, each in enumerate(tqdm(some_calls)):
            waveform, specgram = view.visualise_call(each, fs);
            waveform.set_title(str(i)+' the plot!')
            plt.savefig('TESTRUN'+str(i)+'.png')
     
    print('If you can see this message - it looks like you have a succesful installation!')