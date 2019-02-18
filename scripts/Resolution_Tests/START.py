# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 18:28:06 2019

@author: Sassi
"""

fileNames = ['vanDerWaals3.py',
             'vanDerWaals20.py',
             'vanDerWaals30.py',
             'vanDerWaals40.py',
             'vanDerWaals50.py',
             'vanDerWaals60.py',
             'vanDerWaals70.py',
             'vanDerWaals80.py',
             'vanDerWaals90.py',
             'vanDerWaals100.py',
             'vanDerWaals200.py',
             'vanDerWaals300.py',
             'vanDerWaals400.py',
             'vanDerWaals500.py',
             'vanDerWaals1000.py',
             ]

for fileName in fileNames:
    print "Started: ", fileName
    execfile(fileName)