'''
Created on Aug 17, 2018

@author: e.esche + j.weigert
'''
from numpy import float
import numpy as np
import mpmath

from modOpt.initialization import VarType 

class VarListType(object):

    '''
    definition of variable lists

    Parameters:
    -----------

    type: string, optional
        defines the type of the variable list
    performSampling: bool, optional
        sampling, yes or no
    numberOfSamples: int, optional
        number of samples if sampling is performed
    samplingMethod: string, optional
        'hammersley', 'sobol', 'latin_hypercube' --> method for uniform sample generation
    samplingDistribution: string, optional
        method from scipy.stats --> distibution applied on chosen sampling method
    distributionParams: tuple, optional
        additional params to loc and scale in defined sampling distribution (see scipy.stats)
    seed: int, optional
        from 0 to 10^32 if None random seed is used --> seed for uniform sample generation
        (remark: if 'hammersley' or 'sobol' sampling time increases with increasing seed')
    time: list, optional
        list with time points used in type=TYPE_STATE

    '''

    def __init__(self):
        self.varlist = []
        self.globalID = -1
    
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
        print("ERROR: There is no applicable value for your requested time.")
        return None
       
    def getValueAtIndex(self,requestedVarName,requestedIndex):
        for currVar in self.varlist:
            if currVar.varName == requestedVarName:
                return currVar.value[requestedIndex]
        return None
    
    def getLowerBound(self,globalID):
        for listItem in self.varlist:
            if listItem.globalID == globalID:
                return listItem.lowerBound
    
    def getUpperBound(self,globalID):
        for listItem in self.varlist:
            if listItem.globalID == globalID:
                return listItem.upperBound
    

class StateVarList(VarListType):

    def __init__(self, time=[]):
        super().__init__()
        self.time = time
        self.snapshotDict = {}
    
    def add(self, varName, value=[], modelVarID ='', engUnit='-'):
        self.globalID += 1
        listItem = VarType.StateType(self.globalID,varName,value,modelVarID,engUnit)
        self.varlist.append(listItem)

    def getArrayOfStates(self, stateID=None, runID=0):
        '''Create array of time series of states of size (number timepoints, number states)
            stateID: tuple of integers with ID of requested state variables
            runID: integer with ID of snapshot run
        '''
        if stateID == None:
            fullArray = []
            for time in range(len(self.time['values'])):
                fullArray.append([x.value[runID][time] for x in self.varlist])
        else:
            fullArray = []
            for time in range(len(self.time['values'])):
                fullArray.append([x.value[runID][time] for x in self.varlist if x.globalID in stateID])
        return fullArray

class ControlVarList(VarListType):

    def __init__(self, performSampling=False, numberOfSamples=1, samplingMethod='hammersley', samplingDistribution='uniform', distributionParams=(), seed=None):
        super().__init__()
        self.performSampling = performSampling
        self.numberOfSamples =numberOfSamples
        self.samplingMethod = samplingMethod
        self.sampleData = []
        self.seed = seed
        self.samplingDistribution = samplingDistribution
        self.distributionParams = distributionParams
    
    def add(self, varName, value=[], loc=0, scale=1, modelVarID ='', engUnit='-', time=[], samplingDistribution=None, distributionParams=(), binarySamplingSeed=None):
        self.globalID += 1
        listItem = VarType.ControlType(self.globalID,varName,value,time,loc,scale,modelVarID,engUnit,samplingDistribution,distributionParams,binarySamplingSeed)
        self.varlist.append(listItem)

class InitialVarList(VarListType):

    def __init__(self, performSampling=False, numberOfSamples=1, samplingMethod='hammersley', samplingDistribution='uniform', distributionParams=(), seed=None, restartFile=''):
        super().__init__()
        
        if restartFile:
            self.performSampling = False
        else:
            self.performSampling = performSampling
        
        self.numberOfSamples =numberOfSamples
        self.samplingMethod = samplingMethod
        self.sampleData = []
        self.seed = seed
        self.samplingDistribution = samplingDistribution
        self.distributionParams = distributionParams
        self.restartFile = restartFile
    
    def add(self, varName, value=[], loc=0, scale=1, modelVarID ='', engUnit='-', samplingDistribution=None, distributionParams=()):
        self.globalID += 1
        listItem = VarType.VarType(self.globalID,varName,value,loc,scale,modelVarID,engUnit,samplingDistribution,distributionParams)
        self.varlist.append(listItem)

class VariableList(VarListType):

    def __init__(self, performSampling=False, numberOfSamples=1, samplingMethod='hammersley', samplingDistribution='uniform', distributionParams=(), seed=None, restartFile='', model=None, boxID=None, block=None):
        super().__init__()
        
        if restartFile:
            self.performSampling = False
        else:
            self.performSampling = performSampling
        self.numberOfSamples =numberOfSamples
        self.samplingMethod = samplingMethod
        self.sampleData = []
        self.seed = seed
        self.samplingDistribution = samplingDistribution
        self.distributionParams = distributionParams
        if model != None:
            self.add_vars_from_model(model, boxID)
        if block != None:
            self.add_vars_from_block(block, boxID)
    
    def add(self, varName, value=[], loc=0, scale=1, modelVarID ='', engUnit='-', samplingDistribution=None, distributionParams=()):
        self.globalID += 1
        listItem = VarType.VarType(self.globalID,varName,value,loc,scale,modelVarID,engUnit,samplingDistribution,distributionParams)
        self.varlist.append(listItem)
    
    def add_vars_from_block(self, block, boxID):
        varNames = block.x_sym_tot
        
        for glbID in block.colPerm:
            if isinstance(block.xBounds_tot[glbID], mpmath.ctx_iv.ivmpf):
                scale = float(mpmath.mpf((block.xBounds_tot[glbID].a - block.xBounds_tot[glbID].b).mid))/6.0 # standard deviation
                loc = float(mpmath.mpf(block.xBounds_tot[glbID].mid))
            else:
                scale = (block.xBounds_tot[glbID, 1] - block.xBounds_tot[glbID, 0])/6.0 # standard deviation
                loc = 0.5 * (block.xBounds_tot[glbID, 0] +block.xBounds_tot[glbID, 1]) # Mean value 
            self.add(varName = varNames[glbID], 
                     loc = loc,
                     scale = scale,
                     samplingDistribution=self.samplingDistribution,
                     distributionParams=self.distributionParams)
    
    
    def add_vars_from_model(self, model, boxID):   
        varNames = model.xSymbolic
        
        for glbID in range(0, len(varNames)):
            if isinstance(model.xBounds[boxID][glbID], mpmath.ctx_iv.ivmpf):
                scale = float(mpmath.mpf((model.xBounds[boxID][glbID].a - model.xBounds[boxID][glbID].b).mid))/6.0 # standard deviation
                loc = float(mpmath.mpf(model.xBounds[boxID][glbID].mid))
            else:
                scale = (model.xBounds[boxID][glbID, 1] - model.xBounds[boxID][glbID, 0])/6.0 # standard deviation
                loc = 0.5 * (model.xBounds[boxID][glbID, 0] +model.xBounds[boxID][glbID, 1]) # Mean value 
            self.add(varName = varNames[glbID], 
                     loc = loc,
                     scale = scale,
                     samplingDistribution=self.samplingDistribution,
                     distributionParams=self.distributionParams)

class ParameterList(VarListType):

    def __init__(self, adaptiveSampling=False, performSampling=False, numberOfSamples=1, samplingMethod='hammersley', samplingDistribution='norm', distributionParams=(), seed=None, initialNumberOfSamples=1, addedNumberOfSamples=1):
        super().__init__()
        self.adaptiveSampling = adaptiveSampling
        self.performSampling = performSampling
        self.numberOfSamples =numberOfSamples
        self.samplingMethod = samplingMethod
        self.sampleData = []
        self.seed = seed
        self.samplingDistribution = samplingDistribution
        self.distributionParams = distributionParams
        if adaptiveSampling:
            self.initialNumberOfSamples = initialNumberOfSamples
            self.addedNumberOfSamples = addedNumberOfSamples
    
    def add(self, varName, value=[], loc=0, scale=1, modelVarID ='', engUnit='-', samplingDistribution=None, distributionParams=()):
        self.globalID += 1
        listItem = VarType.VarType(self.globalID,varName,value,loc,scale,modelVarID,engUnit,samplingDistribution,distributionParams)
        self.varlist.append(listItem)





