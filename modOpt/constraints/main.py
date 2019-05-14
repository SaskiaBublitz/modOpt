""" Interval union Newton method """

"""
***************************************************
Import packages
***************************************************
"""
import time
from modOpt.decomposition.dM  import doDulmageMendelsohn
import copy
import numpy
import parallelization
import iNes_procedure


"""
***************************************************
Main that invokes methods for variable constraints reduction
***************************************************
"""
__all__ = ['reduceVariableBounds', 'nestBlocks']

def reduceVariableBounds(model, options):
    """ variable bounds are reduced based on user-defined input
    
    Args: 
        :model:       object of class model in modOpt.model that contains all
                      information of the NLE-evaluation from MOSAICm. 
        :options:     dictionary with user-specified information
        
    Return:
        :reducedModel:  model with reduced XBounds
        :iterNo:        number of outter iteration Steps
                        is returned, otherwise an empty list is returned
        :t:             measured time
    
    """
    
    if options['method'] == 'complete':
        model.blocks = [range(0, len(model.xSymbolic))]
            
    if options['method'] == 'partial':
        # Decomposition:
        jacobian = model.getJacobian()
        dict_permutation = doDulmageMendelsohn(jacobian)
        model.updateToPermutation(dict_permutation["Row Permutation"],
                                     dict_permutation["Column Permutation"],
                                     dict_permutation["Number of Row Blocks"])
        
    if options['timer'] == True: 
        tic = time.clock()
        reducedModel, iterNo = doIntervalNesting(model, options)
        toc = time.clock()
        t = toc - tic
        return reducedModel, iterNo, t
        
    else:
        reducedModel, iterNo = doIntervalNesting(model, options)
        return reducedModel, iterNo, []


def doIntervalNesting(model, dict_options):
    """ iterates the state variable intervals related to model using the
    Gauss-Seidel Operator combined with an interval nesting strategy
    
    Args:
        :model:           object of class model
        :dict_options:    dictionary with solver options
    
    Return:
        :newModel:        model object with reduced state variable bounds
        :iterNo:          number of iteration steps in Newton algorithm
        
    """
    
    iterNo = 0
    xBounds = model.xBounds
    xSymbolic = model.xSymbolic
    parameter = model.parameter
    blocks = model.blocks
    newModel = copy.deepcopy(model)
    dimVar = len(xSymbolic)
    boundsAlmostEqual = False * numpy.ones(dimVar, dtype=bool)


    for l in range(0, dict_options["iterMaxNewton"]): 
         
        iterNo = l + 1
        
        if dict_options["Parallel Branches"]:
            newXBounds, xAlmostEqual, boundsAlmostEqual = parallelization.reduceMultipleXBounds(xBounds, 
                                        boundsAlmostEqual, model, blocks, dimVar,
                                        xSymbolic, parameter, dict_options)
                    
        else:
            newXBounds, xAlmostEqual, boundsAlmostEqual = iNes_procedure.reduceMultipleXBounds(xBounds, 
                                                        xSymbolic, parameter, model, dimVar, blocks, 
                                                        boundsAlmostEqual, dict_options)

        if xAlmostEqual.all():
            newModel.setXBounds(xBounds)

            return newModel, iterNo
          
        if newXBounds != []: xBounds = newXBounds
        else: 
            print "NoSolutionError: No valid solution space was found. Please check consistency of initial constraints"
            
    newModel.setXBounds(xBounds)
    
    return newModel, iterNo


def nestBlocks(model):
    """ creates list with nested blocks for complete interval nesting procedurce

    Args:
        :model:         instance of class Model 
    
    """
    
    nestedBlocks = []
    
    for item in model.blocks:
        nestedBlocks.append([item])
    model.blocks = nestedBlocks
    
    