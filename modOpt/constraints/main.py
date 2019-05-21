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
        :res_solver:    dictionary with resulting model of procedure, iteration 
                        number and time (if time measurement is chosen)
    
    """
    
    res_solver = {}
    
    if options['method'] == 'complete':
        model.blocks = [range(0, len(model.xSymbolic))]
            
    if options['method'] == 'partial':
        # Decomposition:
        jacobian = model.getJacobian()
        dict_permutation = doDulmageMendelsohn(jacobian)
        model.updateToPermutation(dict_permutation["Row Permutation"],
                                     dict_permutation["Column Permutation"],
                                     dict_permutation["Number of Row Blocks"])
    res_solver["Model"] = model
    
    if options['timer'] == True: 
        tic = time.clock()
        doIntervalNesting(res_solver, options)
        toc = time.clock()
        t = toc - tic
        res_solver["time"] = t
        return res_solver
        
    else:
        doIntervalNesting(res_solver, options)
        res_solver["time"] = []
        return res_solver


def doIntervalNesting(res_solver, dict_options):
    """ iterates the state variable intervals related to model using the
    Gauss-Seidel Operator combined with an interval nesting strategy
    
    Args:
        :res_solver:      dictionary for storing procedure output
        :dict_options:    dictionary with solver options
            
    """
    
    model = res_solver["Model"]
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
            output = parallelization.reduceMultipleXBounds(xBounds, 
                                        boundsAlmostEqual, model, blocks, dimVar,
                                        xSymbolic, parameter, dict_options)
               
        else: 
            output = iNes_procedure.reduceMultipleXBounds(xBounds, xSymbolic, parameter, 
                                                          model, dimVar, blocks, boundsAlmostEqual, dict_options)
        
        xAlmostEqual = output["xAlmostEqual"]
        newXBounds = output["newXBounds"]
        
        if output.has_key("noSolution"):
                       
            newModel.setXBounds(xBounds)
            newModel.failed = True
 
            res_solver["Model"] = newModel
            res_solver["iterNo"] = iterNo
            res_solver["noSolution"] = output["noSolution"]
            return True
        
        elif xAlmostEqual.all():
            newModel.setXBounds(xBounds)
            res_solver["Model"] = newModel
            res_solver["iterNo"] = iterNo
            
            return True
          
        else: xBounds = newXBounds
        
    # Terminates with reaching iterMax:
    newModel.setXBounds(xBounds)
    res_solver["Model"] = newModel
    res_solver["iterNo"] = iterNo
    
    return True


def nestBlocks(model):
    """ creates list with nested blocks for complete interval nesting procedurce

    Args:
        :model:         instance of class Model 
    
    """
    
    nestedBlocks = []
    
    for item in model.blocks:
        nestedBlocks.append([item])
    model.blocks = nestedBlocks
    
    