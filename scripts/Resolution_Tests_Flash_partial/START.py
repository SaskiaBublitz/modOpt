# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 18:28:06 2019

@author: Sassi
"""

fileNames = ['Flash3.py',
             'Flash10.py',
             'Flash20.py',
             'Flash30.py',
             'Flash40.py',
             'Flash50.py',
             'Flash60.py',
             'Flash70.py',
             'Flash80.py',
             'Flash90.py',
             'Flash100.py',
             'Flash200.py',
             'Flash300.py',
             'Flash400.py',
             'Flash500.py',
             'Flash1000.py'
             ]

for fileName in fileNames:
    print "Started: ", fileName
    execfile(fileName)