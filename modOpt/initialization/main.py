""" Sample selection """

"""
***************************************************
Import packages
***************************************************
"""
import time
import numpy
from modOpt.initialization import parallelization, VarListType, Sampling
import modOpt.storage as mostge

"""
********************************************************
All methods to specify a specific sample point selection
********************************************************
"""
__all__ = ['get_samples_with_n_lowest_residuals', 'doSampling', 'sample_box']


def doSampling(model, dict_options, sampling_options):
    """ samples multiple boxes with selected method from sampling_options and stores
    the ones with the minimum functional residual.
    
    Args:
        :model:             instance of class model
        :dict_options:      dictionary with user-settings regarding parallelization
                            and box reduction steps applied before
        :sampling_options:  dictionary with number of samples to generate and 
                            sampleNo_min_residual which is the number of candidates
                            with lowest residuals that are stored

    Returns:                None

    """
    
    res = {}
    tic = time.time()
    fileName = dict_options["fileName"]\
    +"_r"+str(dict_options["redStep"])\
    +"_s"+str(sampling_options["number of samples"])\
    +"_smin"+str(sampling_options["sampleNo_min_resiudal"])\
    +".npz"
    
    
    if dict_options["parallelization"]: 
        res =parallelization.sample_box(model, sampling_options, dict_options)
    else:
        for boxID in range(0, len(model.xBounds)):
            res = sample_box(model, boxID, sampling_options, dict_options, res)
            
    for boxID in range(0,len(model.xBounds)):     
        mostge.store_list_in_npz_dict(fileName, res[boxID], boxID)

    if dict_options["timer"]:  
        print("Time: ", time.time() - tic, " sec.")
        mostge.store_time(fileName, [time.time() - tic], len(model.xBounds))


def sample_box(model, boxID, sampling_options, dict_options, res):
    """ samples one box with ID boxID by selected method from sampling_options and 
    stores the samples with the minimum functional residual.
    
    Args:
        :model:             instance of class model
        :boxID:             ID of current box as integer
        :sampling_options:  dictionary with number of samples to generate and 
                            sampleNo_min_residual which is the number of candidates
                            with lowest residuals that are stored
        :dict_options:      dictionary with user-settings regarding parallelization
                            and box reduction steps applied before
        :res:               dictionary for storage of samples with minimum residual
                            for all boxes

    Returns:                updated dictionary res by samples of current box

    """
    iterVars = VarListType.VariableList(performSampling=True, 
                                  numberOfSamples=sampling_options["number of samples"],
                                  samplingMethod=sampling_options["sampling method"], 
                                  samplingDistribution='uniform',
                                  seed=None,
                                  distributionParams=(),
                                  model=model, 
                                  boxID=boxID)
    iterVarsSampler = Sampling.Variable_Sampling(iterVars, number_of_samples=iterVars.numberOfSamples)

    iterVars.sampleData = numpy.array(iterVarsSampler.create_samples())
    sampleNo = sampling_options['sampleNo_min_resiudal']   
    
    res[boxID] = get_samples_with_n_lowest_residuals(model, sampleNo, iterVars.sampleData)
    return res
    
    


def get_samples_with_n_lowest_residuals(model, n, sampleData):
    """ tests the first n samples that have the lowest function residuals from 
    sampleData
    
    Args:
        :model:             instance of class model
        :n:                 integer with real number of tested samples
        :sampleData:        numpy array sampling points 
    
    Returns:                numpy array with n samples with lowest residual
    
    """
    residuals = []
    
    # Calc residuals:
    #if dict_options != parallel: 
    for curSample in sampleData:
        model.stateVarValues=[curSample]
        
        residuals.append(numpy.linalg.norm(model.getFunctionValues()))
    
    # Sort samples by minimum residuals
    residuals = numpy.array(residuals)
    sample_index = numpy.argsort(residuals)
    residuals = residuals[sample_index]
    print ("Function residuals of sample points:\t", residuals[0:n])
    
    return sampleData[sample_index][0:n]