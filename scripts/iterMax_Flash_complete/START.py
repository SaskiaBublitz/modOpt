# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 18:28:06 2019

@author: Sassi
"""

fileNames = ['Flash1.py',
             'Flash2.py',
             'Flash3.py',
             'Flash4.py',
             'Flash5.py',
             'Flash6.py',
             'Flash7.py',
             'Flash8.py',
             'Flash9.py',
             'Flash10.py',
             'Flash20.py',
             'Flash30.py',
             'Flash50.py',
             'Flash100.py',
             'Flash300.py',
             'Flash500.py',
             'Flash1000.py'
             ]

for fileName in fileNames:
    print "Started: ", fileName
    execfile(fileName)