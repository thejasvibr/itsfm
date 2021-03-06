# -*- coding: utf-8 -*-
"""script which runs the command line stuff for 
itsfm
Created on Fri Mar 27 14:30:51 2020

@author: tbeleyur
"""
import argparse
import pandas as pd
import itsfm.run_example_analysis as run_eg
import itsfm.batch_processing as batch

#import measure_hor
parser = argparse.ArgumentParser(description='measure some CF-FM calls!')

parser.add_argument('-run_example', action="store_true",
                                dest="run_example", 
                                help='Runs example if the -run_example argument is included')

parser.add_argument('-batchfile', 
                    action="store", dest="batchfile", 
                    default=None,
                    help='Path to the batch file. A .csv file is expected')

parser.add_argument('-one_row', 
                    action="store", dest="one_row", 
                    default=None,
                    type=int,
                    help='A specific row to be loaded from the batch file. Integer>=0')

parser.add_argument('-from', 
                    action="store", dest="_from", 
                    default=None,
                    type=int,
                    help='A specific row to start the batchfile run from. Row numbers start with 0.')

parser.add_argument('-till', 
                    action="store", dest="_till", 
                    default=None,
                    type=int,
                    help='A specific row to run the batchfile till. Row numbers end at Nrows-1.')

parser.add_argument('-del_measurement', 
                    action="store", dest="del_measurement", 
                    default=False,
                    type=bool,
                    help='Whether to delete the measurement file if it already exists')

def main(arg_parser):
    '''
    '''
    args = arg_parser.parse_args()
    if args.run_example:
        run_eg.run_example_data()

    if args.batchfile:
        print('batch file detected, starting batch analyses now...')
        batch.run_from_batchfile(args.batchfile, **args.__dict__)       
        pass    


if __name__ == '__main__':
	main(parser)