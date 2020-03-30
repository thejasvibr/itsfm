# -*- coding: utf-8 -*-
"""Module that checks the user-input values make sense for each variable. 

Created on Mon Mar 30 12:36:50 2020

@author: tbeleyur
"""


def make_sure_its_positive(value, **kwargs):
    variable = kwargs.get('variable', 'this variable')
    if value <0:
        msg = 'The entered value for '+variable+' :'+str(value)+' cannot be negative. Check entry.'
        raise ValueError(msg)
