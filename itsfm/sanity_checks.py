# -*- coding: utf-8 -*-
"""Module that checks the user-input values make sense for each variable. 

Created on Mon Mar 30 12:36:50 2020

@author: tbeleyur
"""
import os 

def make_sure_its_positive(value, **kwargs):
    '''

    Parameters
    ----------
    value : float/int
        The variable value tobe checked
    variable: str, optional
        Name of the variabel to be checkeds

    Raises
    ------
    ValueError
        If the variable value is <0

    Returns
    -------
    None.

    '''
    variable = kwargs.get('variable', 'this variable')
    if value <0:
        msg = f'The entered value for {variable}: {value} cannot be negative. Check entry.'
        raise ValueError(msg)

def make_sure_its_negative(value, **kwargs):
    '''

    Parameters
    ----------
    value : float/int
        The variable value tobe checked
    variable: str, optional
        Name of the variabel to be checkeds

    Raises
    ------
    ValueError
        If the variable value is >0

    Returns
    -------
    None.

    '''
    variable = kwargs.get('variable', 'this variable')
    if value >0:
        msg = f'The entered value for {variable}: {value} cannot be positive. Check entry.'
        raise ValueError(msg)

def check_preexisting_file(file_name):
    '''
    Raises
    ------
    ValueError : if the target file name already exists in the current directory
    '''
    exists = os.path.exists(file_name)

    if exists:
        mesg = 'The file: '+file_name+' already exists- please move it elsewhere or rename it!'
        raise ValueError(mesg)
