""" Interval union Newton method """

"""
***************************************************
Import packages
***************************************************
"""
import os
import operator
import time
import copy
import sympy
import numpy
from modOpt import storage
from modOpt.constraints import iNes_procedure, parallelization, results, analysis
numpy.seterr(all="ignore")

"""
***************************************************
Main that invokes methods for variable constraints reduction
***************************************************
"""
__all__ = ['reduceVariableBounds', 'nestBlocks','sort_fId_to_varIds']


def reduceVariableBounds(model, bxrd_options, sampling_options=None, 
                         solv_options=None):
    """ variable bounds are reduced based on user-defined input
    
    Args: 
        :model:            object of class model in modOpt.model that contains all
                           information of the NLE-evaluation from MOSAICm. 
        :bxrd_options:     dictionary with user-specified information
        :sampling_options: dictionary with sampling settings
        :solv_options:     dicionary with settings for numerical solver
        
    Return:
        :res_solver:      dictionary with resulting model of procedure, iteration 
                            number and time (if time measurement is chosen)
    
    """   
    res_solver = {}
    tic = time.time()     
    res_solver["Model"] = model
    res_solver["time"] = []
    res_solver["init_box"] = list(model.xBounds)
    # Currently not used or prespecified
    bxrd_options["parallelVariables"]=False
    
    doIntervalNesting(res_solver, bxrd_options, sampling_options, solv_options)
    toc = time.time()
    res_solver["time"] = toc - tic
    
    return res_solver
  
      
def doIntervalNesting(res_solver, bxrd_options, sampling_options=None, 
                      solv_options=None):
    """ iterates the state variable intervals related to model using the
    Gauss-Seidel Operator combined with an interval nesting strategy
    
    Args:
        :res_solver:      dictionary for storing procedure output
        :bxrd_options:    dictionary with solver options
        :sampling_options: dictionary with sampling settings
        :solv_options:     dicionary with settings for numerical solver
            
    """
    
    if "iterMaxNewton" in bxrd_options: 
        bxrd_options["redStepMax"] = bxrd_options["iterMaxNewton"]
    
    model = res_solver["Model"]
    bxrd_options["mean_residual"] = [0.0]
    iterNo = 0
    bxrd_options["tear_id"] = 0
    bxrd_options["splitvar_id"] = -1
    #bxrd_options["maxBoxNo"] = len(model.xBounds)
    bxrd_options["disconti"] = [False] * len(model.xBounds)
    bxrd_options["xAlmostEqual"] = [False] * len(model.xBounds)
    bxrd_options["cut"] = [True] * len(model.xBounds)
    bxrd_options["xSolved"] = [False] * len(model.xBounds)
    os.makedirs(bxrd_options["savePath"], exist_ok=True)
    bxrd_options["matlabName"] = bxrd_options["fileName"]
    bxrd_options["fileName"] = os.path.join(bxrd_options["savePath"],
                                            results.get_file_name(bxrd_options, 
                                                                  sampling_options, 
                                                                  solv_options))
    npzFileName = bxrd_options["fileName"] + "_boxes.npz"
    
    newModel = copy.copy(model)
    timeMeasure = []
    tic = time.time()
    num_solved = []
    update_complete_parent_boxes(model, 0)
        
    createNewtonSystem(model)
    print(npzFileName)
    storage.store_newBoxes(npzFileName, model, 0) 
    
    model.xBounds[0] = numpy.array(
        [iNes_procedure.remove_zero_and_max_value_bounds(x) 
         for x in model.xBounds[0]])
     
    if bxrd_options["redStepMax"] == 0:
            num_solved = [iNes_procedure.lookForSolutionInBox(model, 0, 
                                                         bxrd_options, 
                                                         sampling_options, 
                                                         solv_options)  ]  
    for iterNo in range(1, bxrd_options["redStepMax"]+1): 

        bxrd_options["iterNo"] = iterNo
        print(f'Red. Step {iterNo}')
        
        if bxrd_options["parallelBoxes"] and len(model.xBounds)>1:
            output = parallelization.reduceBoxes(model, bxrd_options, 
                                                 sampling_options, solv_options)
        
        else: 
            output = iNes_procedure.reduceBoxes(model, bxrd_options, 
                                                sampling_options, solv_options)
            

        if True in output["xSolved"]: 
            for k, solved in enumerate(output["xSolved"]):
                if solved: print("Box ", k, " is solved.")
                
        if not output.__contains__("noSolution"):
            model, output = identify_empty_boxes(model, output, bxrd_options)
            
        if output.__contains__("noSolution"):
            newModel.failed = True
            res_solver["noSolution"] = output["noSolution"]
            storage.store_time(npzFileName, timeMeasure, iterNo)
            break
                        
        if len(model.xBounds) > 1: 
            residuals = analysis.calc_residual(model)
        else: 
            residuals = [0.0]
            
        bxrd_options["xAlmostEqual"]= output["xAlmostEqual"]
        bxrd_options["xSolved"] = output["xSolved"]
        bxrd_options["disconti"] = output["disconti"] 
        bxrd_options["cut"] = output["cut"]       
        bxrd_options["maxBoxNo"] = len(model.xBounds)
        timeMeasure.append(time.time() - tic)

        num_solved.append(output["num_solved"])                 
        storage.store_newBoxes(npzFileName, model, iterNo)
        
        if (solv_options != None and "FoundSolutions" in bxrd_options.keys()):
            if ("termination" in solv_options.keys()):
                if (solv_options["termination"] == "one_solution"):
                    print("Solver terminates because it has found", 
                          " one solution :)")
                    break
        
        if all(bxrd_options["xSolved"]):
            print("All solutions have been found.")
            break

        elif (numpy.array(bxrd_options["xAlmostEqual"]).all() and not model.cut 
              and not bxrd_options["maxBoxNo"] > len(model.xBounds)):
            if bxrd_options["parallelBoxes"]:
                bxrd_options["maxBoxNo"] += bxrd_options["cpuCountBoxes"]
            else:
                bxrd_options["maxBoxNo"] += 1
            model.complete_parent_boxes = output["complete_parent_boxes"]
            print("Can increase maxBoxNo. Current MaxBoxNo is: ", 
                  bxrd_options["maxBoxNo"])
            if residuals[-1] == bxrd_options["mean_residual"][-1]:
                residuals[-1] = 0.0
        else:
            if bxrd_options["debugMode"]: print("Complete parent boxes: ", 
                                                  output["complete_parent_boxes"])
            if bxrd_options["debugMode"]: print("xSolved: ", 
                                                  bxrd_options["xSolved"])            
            model.complete_parent_boxes = output["complete_parent_boxes"]
        bxrd_options["mean_residual"] = residuals
        if len(model.xBounds) > 1: 
            change_order_of_boxes_residual(model, output, bxrd_options)                
        continue
                
    # Updating model:    
    validXBounds = [x for x in model.xBounds if (
        iNes_procedure.solutionInFunctionRange(model.functions, x, bxrd_options))]
    solved = [bxrd_options["xSolved"][i] for i, x in enumerate(model.xBounds) if (
        iNes_procedure.solutionInFunctionRange(model.functions, x, bxrd_options))]
    if validXBounds != [] and solved !=[]:
        solved_boxes = [x for i,x in enumerate(validXBounds) if (solved[i])]   
        solved_boxes = filter_out_discontinuous_boxes(solved_boxes, model)
        unsolved_boxes = [x for i,x in enumerate(validXBounds) if not (solved[i])]  
        validXBounds = solved_boxes + unsolved_boxes
        
    if validXBounds == []: 
        model.failed = True
        res_solver["Model"] = model
        if not "noSolution" in res_solver.keys():
            res_solver = iNes_procedure.identify_function_with_no_solution(res_solver, 
                                                                       model.functions, 
                                                                       model.xBounds[0], 
                                                                       bxrd_options)

    else:
      if all(bxrd_options["xSolved"]): 
          validXBounds, res_solver["unified"] = iNes_procedure.unify_boxes(validXBounds,
                                                                           bxrd_options)  
      storage.store_newBoxes(npzFileName, model, iterNo)    
      newModel.setXBounds(validXBounds)
      res_solver["Model"] = newModel
    storage.store_time(npzFileName, timeMeasure, iterNo)
    storage.store_solved(npzFileName, num_solved, iterNo+1) 
    res_solver["iterNo"] = iterNo
    
    return True


def filter_out_discontinuous_boxes(xBounds, model):
    validXBounds = []
    for j,box in enumerate(xBounds):
        for f in model.functions:
            sub_box = [box[i] for i in f.glb_ID]
            f_iv = f.f_mpmath[0](*sub_box)
            if f_iv.a < -1e100 or f_iv.b > 1e100:
                break
        if f_iv.a < -1e100 or f_iv.b > 1e100: continue
        else: validXBounds.append(box)
    return validXBounds



def update_complete_parent_boxes(model, iterNo):
    """ updates reduction step number box_ID of current complete boxes 
    (used for example in splitting strategy)
    
    Args:
        :model:         instance of type model
        :iterNo:        integer with current reduction step

    """    
    model.complete_parent_boxes = [[iterNo, l] for l 
                                   in range(0,len(model.xBounds))]


def change_order_of_boxes_residual(model, output, bxrd_options):
    """ changes boxes by residual value of midpoints of the current boxes by 
    increasing order. Doing so, the boxes that are the most feasible are 
    reduced first and solutions might be found quickly.
    
    Args: 
        :model:            object of class model in modOpt.model that contains all
                           information of the NLE-evaluation from MOSAICm. 
        :output:           dictionary with output variables from box reduction
        :sampling_options: dictionary with sampling settings
        :bxrd_options:     dicionary with user settings for box reduction
            
    """
    sorted_residual = enumerate(list(bxrd_options["mean_residual"]))
    sorted_index_value = sorted(sorted_residual, key=operator.itemgetter(1))
    #print(sorted_index_value)  
    order_all = [index for index, value in sorted_index_value]
    model.xBounds = [model.xBounds[new_pos] for new_pos in order_all]
    bxrd_options["disconti"] = [bxrd_options["disconti"][new_pos] 
                                for new_pos in order_all]
    model.complete_parent_boxes = [output["complete_parent_boxes"][new_pos] 
                                   for new_pos in order_all]
    bxrd_options["xAlmostEqual"] = [bxrd_options["xAlmostEqual"][new_pos] 
                                    for new_pos in order_all]
    bxrd_options["xSolved"] = [bxrd_options["xSolved"][new_pos] 
                               for new_pos in order_all]    
    bxrd_options["cut"] = [bxrd_options["cut"][new_pos] 
                               for new_pos in order_all]   

def change_order_of_boxes(model, output, bxrd_options):
    """ changes boxes order so that discontinuous boxes come first because their
    gap can be used for splitting.
    
    Args: 
        :model:            object of class model in modOpt.model that contains all
                           information of the NLE-evaluation from MOSAICm. 
        :output:           dictionary with output variables from box reduction
        :sampling_options: dictionary with sampling settings
        :bxrd_options:     dicionary with user settings for box reduction
            
    """    
    id_disconti_boxes = [l for l in range(0,len(output["disconti"])) 
                         if output["disconti"][l]==True] 
    if "complete_boxes" in output.keys(): 
        id_complete_boxes = [l for l in range(0,len(output["complete_boxes"])) 
                             if output["complete_boxes"][l]==False and 
                             not l in id_disconti_boxes] 
        id_disconti_boxes += id_complete_boxes
    if id_disconti_boxes:
        order_all =  id_disconti_boxes

        for l, x in enumerate(model.xBounds):          
            if not l in id_disconti_boxes: order_all.append(l)    
            
        model.xBounds = [model.xBounds[new_pos] for new_pos in order_all]
        bxrd_options["disconti"] = [output["disconti"][new_pos] 
                                    for new_pos in order_all]
        model.complete_parent_boxes = [output["complete_parent_boxes"][new_pos] 
                                       for new_pos in order_all]
        bxrd_options["xAlmostEqual"] = [bxrd_options["xAlmostEqual"][new_pos] 
                                        for new_pos in order_all]
        bxrd_options["xSolved"] = [bxrd_options["xSolved"][new_pos] 
                                   for new_pos in order_all]
    
    
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
    model.blocks = [[item] for item in model.blocks]


def createNewtonSystem(model):
    '''creates lambdified functions for Newton Interval reduction and saves 
    it as model parameter
    
    Args:
        :model: model of equation system
        
    '''
    model.jacobianSympy = model.getSympySymbolicJacobian()
    model.jacobianLambNumpy = sympy.lambdify(model.xSymbolic, 
                                             model.jacobianSympy,'numpy')
    model.jacobianLambMpmath = iNes_procedure.lambdifyToMpmathIvComplex(model.xSymbolic,
                                                    list(numpy.array(model.jacobianSympy)))   
    
def identify_empty_boxes(model, output, bxrd_options):
    for l, box in enumerate(model.xBounds):
        new_x = iNes_procedure.check_box_for_eq_consistency(model, box, 
                                                            bxrd_options) 
        #if new_x != []: 
        #    new_x = iNes_procedure.check_box_for_disconti_iv(model, 
        #                                                     new_x, 
        #                                                     bxrd_options)                
        if new_x != []:
            model.xBounds[l] = new_x
        else:
            model.xBounds[l] = []
            output["xSolved"][l] = []
            output["xAlmostEqual"][l] = []
            output["disconti"][l] = []
            output["cut"][l] = []
            output["complete_parent_boxes"][l] = []
     
    model.xBounds = [x for x in model.xBounds if x!=[]]  
    if not model.xBounds: 
        model.xBounds = [box]
        iNes_procedure.saveFailedSystem(output, model.functions[0], 
                                        model, 0)
        return model, output
    else:
        output["xAlmostEqual"] = [x for x in output["xAlmostEqual"] if x!=[]] 
        output["disconti"] = [x for x in output["disconti"] if x!=[]] 
        output["cut"] = [x for x in output["cut"] if x!=[]] 
        output["xSolved"] = [x for x in output["xSolved"] if x!=[]] 
        output["complete_parent_boxes"] = [x for x in 
                                           output["complete_parent_boxes"]
                                           if x!=[]] 
    return model, output