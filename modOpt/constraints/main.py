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
        doIntervalNesting(res_solver, options)
        toc = time.time()
        t = toc - tic
        res_solver["time"] = t
        return res_solver
        
    else:
        doIntervalNesting(res_solver, options)
        res_solver["time"] = []
        return res_solver


def sort_fId_to_varIds(fId, varIds, dict_varId_fIds):
    for varId in varIds:
        if not varId in dict_varId_fIds: dict_varId_fIds[varId]=[fId]
        else: dict_varId_fIds[varId].append(fId)


def doIntervalNesting(res_solver, dict_options):
    """ iterates the state variable intervals related to model using the
    Gauss-Seidel Operator combined with an interval nesting strategy
    
    Args:
        :res_solver:      dictionary for storing procedure output
        :dict_options:    dictionary with solver options
            
    """
    
    if "iterMaxNewton" in dict_options: dict_options["redStepMax"] = dict_options["iterMaxNewton"]
    
    model = res_solver["Model"]
    #dict_options["maxBoxNo"] =  int((len(model.xBounds[0]))**0.5)
    iterNo = 0
    dict_options["tear_id"] = 0
    newModel = copy.deepcopy(model)
    functions = []
    dict_varId_fIds = {}
    
    for i in range(0, len(model.fSymbolic)):
        functions.append(Function(model.fSymbolic[i], model.xSymbolic))
        sort_fId_to_varIds(i, functions[i].glb_ID, dict_varId_fIds)
    
     
    for l in range(0, dict_options["redStepMax"]): 
        
        iterNo = l + 1
       
        if dict_options["Parallel Branches"]:
            output = parallelization.reduceMultipleXBounds(model, functions, dict_varId_fIds, dict_options)
        
        else: 
            output = iNes_procedure.reduceMultipleXBounds(model, functions, dict_varId_fIds, dict_options)

        xSolved = output["xSolved"]

              
        if output.__contains__("noSolution"):

            newModel.failed = True
            res_solver["noSolution"] = output["noSolution"]
            break
        
        elif xSolved.all():
            break
        
        else: continue
        
    # Updating model:
    # TODO: if some boxes share same lb or ub they should be unified
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
    
    