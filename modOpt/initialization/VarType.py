'''
Created on Aug 17, 2018

@author: e.esche
'''

class VarType:
    def __init__(self, globalID=-1, varName='', value=0.0, lowerBound=-1.0e9, upperBound=1.0e9, modelVarID='', engUnit='-'):
        self.globalID = globalID
        self.varName = varName
        self.value = value
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.engUnit = engUnit
        self.modelVarID = modelVarID
    def __str__(self): 
        return "ID: {0}, vN: {1}, val: {2}, lB: {3}, uB: {4}, engUnit: {5}\n".format(self.globalID,self.varName,self.value,self.lowerBound,self.upperBound,self.engUnit)

class ControlType:
    def __init__(self, globalID=-1, varName='', value=[], time=[], lowerBound=-1.0e9, upperBound=1.0e9, modelVarID='', engUnit='-'):
        self.globalID = globalID
        self.varName = varName
        self.value = value
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.engUnit = engUnit
        self.modelVarID = modelVarID
        self.time = time
    def __str__(self): 
        return "ID: {0}, vN: {1}, lB: {2}, uB: {3}, engUnit: {4}\n".format(self.globalID,self.varName,self.lowerBound,self.upperBound,self.engUnit)

class StateType:
    def __init__(self, globalID=-1, varName='', value=[], modelVarID ='', engUnit='-'):
        self.globalID = globalID
        self.varName = varName
        self.value = value
        self.engUnit = engUnit
        self.modelVarID = modelVarID
    def __str__(self): 
        return "ID: {0}, vN: {1}, engUnit: {2}\n".format(self.globalID,self.varName,self.engUnit)