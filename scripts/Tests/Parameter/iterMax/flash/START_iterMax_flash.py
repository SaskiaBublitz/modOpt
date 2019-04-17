# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 13:22:04 2019

@author: sassibub
"""

#directoryNames = ["resolution_1",
                  #"resolution_2",
                  #"resolution_3"
#                  ]

modelNames = ["Flash",          
              ]

shortcut = "_i"


parValues = [3, 10, 50, 100, 500]
#parValues = [20, 30, 40, 60, 70, 80, 90]

for modName in modelNames:
    for parVal in parValues:    

        try:
            print "%s%s%s is currently running"%(modName, 
                                                   shortcut, 
                                                   parVal)

            execfile("%s%s%s.py"%(modName, shortcut, parVal))
           
        except:
            print "Error in %s%s%s"%(modName, shortcut, parVal)
            pass
                