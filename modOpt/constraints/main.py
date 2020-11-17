""" Interval union Newton method """

"""
***************************************************
Import packages
***************************************************
"""
import time
import copy
import sympy
import numpy
from modOpt import storage
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
    tic = time.time()
    model.blocks = [range(0, len(model.xSymbolic))]       
    res_solver["Model"] = copy.deepcopy(model)
    res_solver["time"] = []

    doIntervalNesting(res_solver, options)
    toc = time.time()
    res_solver["time"] = toc - tic
    return res_solver
        
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
    #iterNo = 0
    dict_options["tear_id"] = 0
    npzFileName = dict_options["fileName"] + "_r" + str(dict_options["redStepMax"]) + "_boxes.npz"
    newModel = copy.deepcopy(model)
    functions = []
    dict_varId_fIds = {}
    timeMeasure = []
    tic = time.time()
    
    createNewtonSystem(model)
    
    for i in range(0, len(model.fSymbolic)):
        functions.append(Function(model.fSymbolic[i], model.xSymbolic))
        sort_fId_to_varIds(i, functions[i].glb_ID, dict_varId_fIds)
    
    storage.store_newBoxes(npzFileName, model, 0) 
    
    for iterNo in range(1, dict_options["redStepMax"]+1): 

        if dict_options["Parallel Branches"]:
            output = parallelization.reduceBoxes(model, functions, dict_varId_fIds, dict_options)
        
        else: 
            output = iNes_procedure.reduceBoxes(model, functions, dict_varId_fIds, dict_options)

        xSolved = output["xSolved"]
        xAlmostEqual = output["xAlmostEqual"]
        timeMeasure.append(time.time() - tic)

              
        if output.__contains__("noSolution"):

            newModel.failed = True
            res_solver["noSolution"] = output["noSolution"]
            storage.store_time(npzFileName, timeMeasure, iterNo)
            break
        
        
        storage.store_newBoxes(npzFileName, model, iterNo)
        
        if xSolved.all():
            break
        
        elif xAlmostEqual.all():
            dict_options["maxBoxNo"] +=1
                
        else: 
            continue
        
    # Updating model:
    validXBounds = []
    for xBounds in model.xBounds:
        if iNes_procedure.solutionInFunctionRange(model, xBounds, dict_options):
            validXBounds.append(xBounds)
       
    if validXBounds == []: 
        model.failed = True
        res_solver["Model"] = model
    else:
      newModel.setXBounds(validXBounds)
      res_solver["Model"] = newModel
      
    storage.store_time(npzFileName, timeMeasure, iterNo) 
    res_solver["iterNo"] = iterNo
    
    return True


def sort_fId_to_varIds(fId, varIds, dict_varId_fIds):
    """ writes current function's global id to a dictionary whose keys equal
    the global id's of the variables. Hence, all variables that occur in the 
    current function get an additional value (fId).
    
    Args:
        :fId:               integer with global id of current function
        :varIds:            list with global id of variables that occur in the 
                            current function
        :dict_varId_fIds:   dictionary that stores global id of current function 
                            (value) to global id of the variables (key(s))

    """
    
    for varId in varIds:
        if not varId in dict_varId_fIds: dict_varId_fIds[varId]=[fId]
        else: dict_varId_fIds[varId].append(fId)

  
def nestBlocks(model):
    """ creates list with nested blocks for complete interval nesting procedurce

    Args:
        :model:         instance of class Model 
    
    """
    
    nestedBlocks = []
    
    for item in model.blocks:
        nestedBlocks.append([item])
    model.blocks = nestedBlocks


def createNewtonSystem(model):
    '''creates lambdified functions for Newton Interval reduction and saves it as model parameter
    Args:
        :model: model of equation system
    '''
    
    model.fLamb = sympy.lambdify(model.xSymbolic, model.fSymbolic)
    model.jacobianSympy = model.getSympySymbolicJacobian()
    model.jacobianLambNumpy = sympy.lambdify(model.xSymbolic, model.jacobianSympy,'numpy')
    model.jacobianLambMpmath = iNes_procedure.lambdifyToMpmathIvComplex(model.xSymbolic,
                                                    list(numpy.array(model.jacobianSympy)))
    
    