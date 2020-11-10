""" Sample selection """

"""
***************************************************
Import packages
***************************************************
"""
import numpy
import modOpt.initialization as moi
import modOpt.storage as mostge

"""
********************************************************
All methods to specify a specific sample point selection
********************************************************
"""
__all__ = ['get_samples_with_n_lowest_residuals', 'doSampling']


def doSampling(model, dict_options, sampling_options):
    fileName = dict_options["fileName"]\
    +"_r"+str(dict_options["redStep"])\
    +"_s"+str(sampling_options["number of samples"])\
    +"_smin"+str(sampling_options["sampleNo_min_resiudal"])\
    +".npz"
    
    for boxID in range(0, len(model.xBounds)):
    
        #iterVars = moi.VarListType(performSampling=True, 
        #                               numberOfSamples=sampling_options["number of samples"],
        #                               samplingMethod=sampling_options["sampling method"], 
        #                               model=model, boxID=boxID)
        iterVars = moi.VariableList(performSampling=True, 
                                      numberOfSamples=sampling_options["number of samples"],
                                      samplingMethod=sampling_options["sampling method"], 
                                      samplingDistribution='uniform',
                                      seed=None,
                                      distributionParams=(),
                                      model=model, 
                                      boxID=boxID)
        #iterVarsSampler = moi.Variable_Sampling(samplingMethod=iterVars.samplingMethod, numberOfSamples=iterVars.numberOfSamples, seed=0, inputSpace=iterVars)
        iterVarsSampler = moi.Variable_Sampling(iterVars, number_of_samples=iterVars.numberOfSamples)

        iterVars.sampleData = numpy.array(iterVarsSampler.create_samples())
        sampleNo = sampling_options['sampleNo_min_resiudal']   
        sampleData = moi.get_samples_with_n_lowest_residuals(model, sampleNo, iterVars.sampleData)
        mostge.store_list_in_npz_dict(fileName, sampleData, boxID) 


def get_samples_with_n_lowest_residuals(model, n, sampleData):
    """ tests the first n samples that have the lowest function residuals from 
    sampleData
    
    Args:
        :model:             instance of class model
        :n:                 integer with real number of tested samples
        :sampleData:        numpy array sampling points 
    
    """
    
    residuals = []
    
    # Calc residuals:
    for curSample in sampleData:
        residuals.append(sum(abs(numpy.array(model.fSymCasadi(*curSample)))))
    
    # Sort samples by minimum residuals
    residuals = numpy.array(residuals)
    sample_index = numpy.argsort(residuals)
    residuals = residuals[sample_index]
    print ("Function residuals of sample points:\t", residuals[0:n])
    
    return sampleData[sample_index][0:n]