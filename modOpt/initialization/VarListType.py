'''
Created on Aug 17, 2018

@author: e.esche
'''
from numpy import float
import numpy as np
from modOpt.initialization import VarType

TYPE_VARIABLE = 'variable'
TYPE_CONTROL = 'control'
TYPE_STATE = 'state'

class VarListType:
    def __init__(self, type=TYPE_VARIABLE, performSampling=False, numberOfSamples=0, samplingMethod='NDS', sampleData=[], time=[], snapshotDict={},
                 model = None, boxID=0):
        
        if type == 'state':
            self.varlist = []
            self.globalID = -1
            self.type = type
            self.time = time
            self.snapshotDict = snapshotDict
        else:
            self.varlist = []
            self.globalID = -1
            self.type = type
            self.performSampling = performSampling
            self.numberOfSamples =numberOfSamples
            self.samplingMethod = samplingMethod
            self.sampleData = sampleData
        if model != None:
            self.addVarsFromModel(model, boxID)
            
    
    def __str__(self):
        outputStr = ''
        for item in self.varlist:
            outputStr += str(item)
        return outputStr
    
    def getTime(self,requestedVarName):
        for currVar in self.varlist:
            if currVar.varName == requestedVarName:
                return currVar.time
        return None
    
    def getValues(self,requestedVarName):
        for currVar in self.varlist:
            if currVar.varName == requestedVarName:
                return currVar.value
        return None
    
    def getValueAtTime(self,requestedVarName,requestedTime):
        if (type(requestedTime['values']) == float):
            currValues = self.getValues(requestedVarName)
            currTime = self.getTime(requestedVarName)
            if currTime['engUnit'] == requestedTime['engUnit']:
                currTimeValues = currTime['values']
                for i in range(0,len(currTimeValues)):
                    if currTimeValues[i] == requestedTime['values']:
                        return currValues[i]
        print ("ERROR: There is no applicable value for your requested time.")
        return None
       
    def getValueAtIndex(self,requestedVarName,requestedIndex):
        for currVar in self.varlist:
            if currVar.varName == requestedVarName:
                return currVar.value[requestedIndex]
        return None
       
    def add(self, varName, value=[], lowerBound=-1.0e9, upperBound=1.0e9, modelVarID ='', engUnit='-', time=[], controlTime=None):
        if self.type == TYPE_VARIABLE:
            self.globalID += 1
            listItem = VarType.VarType(self.globalID,varName,value,lowerBound,upperBound,modelVarID,engUnit)
            self.varlist.append(listItem)
        elif self.type == TYPE_CONTROL:
            self.globalID += 1
            listItem = VarType.ControlType(self.globalID,varName,value,time,lowerBound,upperBound,modelVarID,engUnit)
            self.varlist.append(listItem)
        elif self.type == TYPE_STATE:
            self.globalID += 1
            listItem = VarType.StateType(self.globalID,varName,value,modelVarID,engUnit)
            self.varlist.append(listItem)
        else:
            print ('ERROR in VarListType: Unknown Type')
    
    def getLowerBound(self,globalID):
        for listItem in self.varlist:
            if listItem.globalID == globalID:
                return listItem.lowerBound
    
    def getUpperBound(self,globalID):
        for listItem in self.varlist:
            if listItem.globalID == globalID:
                return listItem.upperBound
    
    def getArrayOfStates(self, stateID=None, runID=0):
        '''Create array of time series of states of size (number timepoints, number states)

            stateID: tuple of integers with ID of requested state variables

            runID: integer with ID of snapshot run
        '''
        if self.type != 'state':
            print ("ERROR: method only valid for variable type 'state'")
            exit()
        else:
            if stateID == None:
                fullArray = []
                for i in range(len(self.time['values'])):
                    row = []
                    for j in range(self.globalID+1):
                        row.append(self.varlist[j].value[i][runID])
                    fullArray.append(row)
                fullArray = np.asarray(fullArray)
            else:
                fullArray = []
                for i in range(len(self.time['values'])):
                    row = []
                    for j in stateID:
                        row.append(self.varlist[j].value[i][runID])
                    fullArray.append(row)
                fullArray = np.asarray(fullArray)
        return fullArray


    def addVarsFromModel(self, model, boxID):   
        varNames = model.xSymbolic
        for glbID in range(0, len(varNames)):
            self.add(varName = varNames[glbID], 
                         lowerBound = model.xBounds[boxID][glbID, 0], 
                         upperBound = model.xBounds[boxID][glbID, 1], engUnit='-')