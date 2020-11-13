'''
Created on Aug 17, 2018

@author: e.esche + j.weigert
'''

class VarType:
    '''
    definition of variables

    Parameters
    ----------

    globalID: int, optional
        ID of variable in variable list
    varName: string, optional
        name of defined variable
    value: float, optional
        value of variable if not used for sampling
    loc: float, optional
        location parameter for distributions from scipy.stats
    scale: float, optional
        scale patameter for distributions from scipy.stats
    modelVarID: int, optional
        can be used for defining the variable in a model
    engUnit: string, optional
        engeneering unit of defined variable
    samplingDistribution: string, optional
        method from scipy.stats --> distibution applied on chosen sampling method
        if defined this method is used for sampling this variable instead of the method
        defined in the variable list
    distributionParams: string, optional
        additional params to loc and scale in defined sampling distribution (see scipy.stats)
    
    '''
    def __init__(self, globalID=-1, varName='', value=0.0, loc=0.0, scale=1.0, modelVarID='', engUnit='-', samplingDistribution=None, distributionParams=()):
        self.globalID = globalID
        self.varName = varName
        self.value = value
        self.loc = loc
        self.scale = scale
        self.engUnit = engUnit
        self.modelVarID = modelVarID
        self.samplingDistribution = samplingDistribution
        self.distributionParams = distributionParams
    def __str__(self): 
        return "ID: {0}, vN: {1}, val: {2}, loc: {3}, scale: {4}, engUnit: {5}\n".format(self.globalID,self.varName,self.value,self.loc,self.scale,self.engUnit)

class ControlType:
    def __init__(self, globalID=-1, varName='', value=[], time=[], loc=0.0, scale=1.0, modelVarID='', engUnit='-', samplingDistribution=None, distributionParams=(), binarySamplingSeed=None):
        self.globalID = globalID
        self.varName = varName
        self.value = value
        self.loc = loc
        self.scale = scale
        self.engUnit = engUnit
        self.modelVarID = modelVarID
        self.time = time
        self.samplingDistribution = samplingDistribution
        self.distributionParams = distributionParams
        self.binarySamplingSeed = binarySamplingSeed
    def __str__(self): 
        return "ID: {0}, vN: {1}, loc: {2}, scale: {3}, engUnit: {4}\n".format(self.globalID,self.varName,self.loc,self.scale,self.engUnit)

class StateType:
    def __init__(self, globalID=-1, varName='', value=[], modelVarID ='', engUnit='-'):
        self.globalID = globalID
        self.varName = varName
        self.value = value
        self.engUnit = engUnit
        self.modelVarID = modelVarID
    def __str__(self): 
        return "ID: {0}, vN: {1}, engUnit: {2}\n".format(self.globalID,self.varName,self.engUnit)