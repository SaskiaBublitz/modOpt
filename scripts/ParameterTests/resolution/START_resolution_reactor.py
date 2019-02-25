# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 13:22:04 2019

@author: sassibub
"""

directoryNames = ["resolution_1",
                  #"resolution_2",
                  #"resolution_3"
                  ]

modelNames = ["Reactor",          
              ]

shortcut = "_r"


parValues = [10, 50, 100, 500]

for dirName in directoryNames:
    for modName in modelNames:
        for parVal in parValues:    

            try:
                print "%s%s%s in %s is currently running"%(modName, 
                                                       shortcut, 
                                                       parVal,
                                                       dirName)

                execfile("%s/%s%s%s.py"%(dirName, modName, shortcut, parVal))
               
            except:
                print "Error in %s/%s%s%s"%(dirName, modName, shortcut, parVal)
                pass
                