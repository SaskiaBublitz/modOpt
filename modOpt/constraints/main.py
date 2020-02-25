""" Interval union Newton method """

"""
***************************************************
Import packages
***************************************************
"""
import time
import copy
from modOpt.constraints import iNes_procedure, parallelization 
from modOpt.constraints.function import Function


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
    
    model.blocks = [range(0, len(model.xSymbolic))]       
    res_solver["Model"] = copy.deepcopy(model)
    
    if options['timer'] == True: 
        tic = time.time() # time.clock() measures only CPU which is regarding parallelized programms not the time determining step 
        doIntervalNestingNew(res_solver, options)
        toc = time.time()
        t = toc - tic
        res_solver["time"] = t
        return res_solver
        
    else:
        doIntervalNestingNew(res_solver, options)
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
    #boundsAlmostEqual = False * numpy.ones(dimVar, dtype=bool)


    for l in range(0, dict_options["iterMaxNewton"]): 
         
        iterNo = l + 1
       
        if dict_options["Parallel Branches"]:
            output = parallelization.reduceMultipleXBounds(xBounds, model, blocks, dimVar,
                                        xSymbolic, parameter, dict_options)
               
        else: 
            output = iNes_procedure.reduceMultipleXBounds(xBounds, xSymbolic, parameter, 
                                                          model, dimVar, blocks, dict_options)
        
        xAlmostEqual = output["xAlmostEqual"]
        newXBounds = output["newXBounds"]
        
        if output.__contains__("noSolution"):
                       
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



def doIntervalNestingNew(res_solver, dict_options):
    """ iterates the state variable intervals related to model using the
    Gauss-Seidel Operator combined with an interval nesting strategy
    
    Args:
        :res_solver:      dictionary for storing procedure output
        :dict_options:    dictionary with solver options
            
    """
    
    model = res_solver["Model"]
    dict_options["maxBoxNo"] = 100 + int((len(model.xBounds[0]))**0.5)
    iterNo = 0
    #xBounds = model.xBounds
    #xSymbolic = model.xSymbolic
    #parameter = model.parameter
    #blocks = model.blocks
    newModel = copy.deepcopy(model)
    #dimVar = len(xSymbolic)
    #boundsAlmostEqual = False * numpy.ones(dimVar, dtype=bool)
    functions = []
    
    for f in model.fSymbolic:    
        functions.append(Function(f, model.xSymbolic))
    
     
    for l in range(0, dict_options["iterMaxNewton"]): 
        
        iterNo = l + 1
       
        if dict_options["Parallel Branches"]:
            output = parallelization.reduceMultipleXBounds(model, functions, dict_options)
        
        else: 
            output = iNes_procedure.reduceMultipleXBounds(model, functions, dict_options)
            #output = iNes_procedure.reduceMultipleXBounds(xBounds, xSymbolic, parameter, 
            #                                              model, dimVar, blocks, dict_options)

        xAlmostEqual = output["xAlmostEqual"]

        if dict_options["boxNo"] >= dict_options["maxBoxNo"]:
            print("Warning: Maximum number of boxes reached!")
            break      
        
        if output.__contains__("noSolution"):

            newModel.failed = True
            res_solver["noSolution"] = output["noSolution"]
            break
        
        elif xAlmostEqual.all():
            break
        
        else: continue
        
    # Updating model:
    newModel.setXBounds(model.xBounds)
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
    
    