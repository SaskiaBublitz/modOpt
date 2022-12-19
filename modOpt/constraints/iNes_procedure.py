"""
***************************************************
Import packages
***************************************************
"""
import copy
import numpy
import sympy
import mpmath
import pyibex
import itertools
from modOpt.constraints import affineArithmetic,parallelization, analysis
from modOpt.constraints.FailedSystem import FailedSystem
from modOpt.decomposition import MC33
from modOpt.decomposition import dM
import modOpt.solver as mos
import modOpt.constraints.realIvPowerfunction # redefines __power__ (**) for ivmpf
import modOpt.storage as mostg


__all__ = ['reduceBoxes', 'reduceXIntervalByFunction', 'setOfIvSetIntersection',
           'checkWidths', 'getPrecision', 'saveFailedSystem', 
            'variableSolved', 'contractBox', 'reduceConsistentBox','updateSetOfBoxes',
            'do_HC4', 'checkIntervalAccuracy', 'do_bnormal', 'roundValue']

"""
***************************************************
Algorithm for interval Nesting procedure
***************************************************
"""        
def reduceBoxes(model, bxrd_options, sampling_options=None, solv_options=None):
    """ reduction of multiple boxes
    Args:    
        :model:                 object of type model   
        :bxrd_options:          dictionary with user specified algorithm settings
        :sampling_options:      dictionary with sampling settings
        :solv_options:          dicionary with settings for numerical solver

    Return:
        :results:               dictionary with newXBounds and a list of booleans
                                that marks all non reducable variable bounds with 
                                True. If solver terminates because of a NoSolution 
                                case the critical equation is also stored in results 
                                for the error analysis.

    """
    if (bxrd_options["cutBox"] in {"all", "tear", True}): model.cut = True
    model.interval_jac = None  
    model.jac_center = None
    results = {"num_solved": False, "disconti": [], "complete_parent_boxes": [],
        "xSolved": [], "xAlmostEqual": [], "cut": [],
        }
    allBoxes = []
    emptyBoxes = []
    bxrd_options["boxNo"] = len(model.xBounds) 
    bxrd_options["ready_for_reduction"] = get_index_of_boxes_for_reduction(bxrd_options["xSolved"], 
                                                                           bxrd_options["cut"], 
                                                                           bxrd_options["maxBoxNo"]  )
    for k in range(len(model.xBounds)):       
        emptyBoxes = reduce_box(model, allBoxes, emptyBoxes, k, results, 
                                bxrd_options, sampling_options, solv_options)
        
    check_results_reduction_step(model, allBoxes, emptyBoxes, results)      

    return results

def reduce_box(model, allBoxes, emptyBoxes, k, results, bxrd_options,
                      sampling_options=None, solv_options=None):
    xBounds = model.xBounds[k]
    """ reduction of one box by contraction, cutting,  splitting and numerical
    iteration
    
    Args:    
        :model:                 object of type model   
        :allBoxes:              list with reduced boxes
        :emptyBoxes:            list with empty boxes
        :k:                     index as integer of currently reduced box
        :results:               dictionary for storing results
        :bxrd_options:          dictionary with user specified algorithm settings
        :sampling_options:      dictionary with sampling settings
        :solv_options:          dicionary with settings for numerical solver

    Return:
        :results:               dictionary with newXBounds and a list of booleans
                                that marks all non reducable variable bounds with 
                                True. If solver terminates because of a NoSolution 
                                case the critical equation is also stored in results 
                                for the error analysis.

    """
    
    if bxrd_options["debugMode"]: print("Current box index: ", k)

    if bxrd_options["xSolved"][k]: 
        prepare_results_constant_x(model, k, results, allBoxes, bxrd_options)
        return emptyBoxes
    
    elif bxrd_options["xAlmostEqual"][k] and not bxrd_options["disconti"][k]:
        output = reduceConsistentBox(model, bxrd_options, k, 
                                     bxrd_options["boxNo"])             

        prepare_results_splitted_x(model, k, results, output, 
                                   bxrd_options)
            
    elif (bxrd_options["xAlmostEqual"][k] and bxrd_options["disconti"][k] 
          and bxrd_options["boxNo"]  >= bxrd_options["maxBoxNo"] or 
          (not bxrd_options["ready_for_reduction"][k])):
        

        prepare_results_constant_x(model, k, results, allBoxes, bxrd_options) 
        #cut += [False]
        return emptyBoxes

    else:
        if bxrd_options["debugMode"]: print(f'Box {k}')
        if (not bxrd_options["xAlmostEqual"][k] and 
            bxrd_options["disconti"][k] and bxrd_options["considerDisconti"]): 
            
            store_boxNo = bxrd_options["boxNo"]
            bxrd_options["boxNo"] = bxrd_options["maxBoxNo"]
            output = contractBox(xBounds, model, bxrd_options["boxNo"] , 
                                 bxrd_options)     
            bxrd_options["boxNo"]  = store_boxNo             
        else:                                
            output = contractBox(xBounds, model, bxrd_options["boxNo"] , 
                                 bxrd_options)
        prepare_results_inconsistent_x(model, k, results, output, 
                                       bxrd_options)    
                                                                                                                        
        # if (all(output["xAlmostEqual"]) and not all(output["xSolved"]) 
        #     and not solv_options == None and len(output["xNewBounds"])==1):  
        #     model.xBounds[k] = output["xNewBounds"][0]
        #     bxrd_options["parent_box_r"] = model.complete_parent_boxes[k][0]
        #     bxrd_options["parent_box_ID"] = model.complete_parent_boxes[k][1]
        #     results["num_solved"] = lookForSolutionInBox(model, k, 
        #                                                  bxrd_options, 
        #                                                  sampling_options, 
        #                                                  solv_options)
        if (all(output["xAlmostEqual"]) and not all(output["xSolved"]) 
              and not solv_options == None ):
            bxrd_options["parent_box_r"] = model.complete_parent_boxes[k][0]
            bxrd_options["parent_box_ID"] = model.complete_parent_boxes[k][1]
            solved = []
            for box in output["xNewBounds"]:
                model.xBounds[k] = box
                solved.append(lookForSolutionInBox(model, k, bxrd_options, 
                                                          sampling_options, 
                                                          solv_options))
            results["num_solved"] = any(solved)
            

    emptyBoxes = prepare_general_resluts(model, k, allBoxes, results, 
                                         output, bxrd_options)
    return emptyBoxes


def check_results_reduction_step(model, allBoxes, emptyBoxes, results):
    """ checks results for no solution at all in intial box or if consistent
    boxes from contraction cannot be further reduced through cutting. If this
    is true for all consistent boxes than model.cut is False. If it is False
    then the maximum allowed number of boxes is increased.
    
    Args:
        :model:         instance of type model 
        :allBoxes:      list with currently reduced boxes             
        :emptyBoxes:    dictionary with entries for error analysis in case
                        all boxes are empty
        :results:       dictionary with reduction step's reults
    
    """
    if results["cut"] != []: 
        model.cut = any(results["cut"])
    if allBoxes == []: results["noSolution"] = emptyBoxes
    else: model.xBounds = allBoxes       
 
    
def prepare_general_resluts(model, k, allBoxes, results, output, bxrd_options):
    """ writes results that depends on either contraction, cutting or splitting
    into dictionary results
    
    Args:
        :model:         instance of type model
        :k:             integer with curremt box index   
        :allBoxes:      list with currently reduced boxes             
        :results:       dictionary with results from reduction step
        :output:        dictionary with output from splitting and cutting
        :bxrd_options:  dictionary with results from former reduction step
    
    """      
    if output.__contains__("noSolution") :
        saveFailedIntervalSet = output["noSolution"]
        bxrd_options["boxNo"] = len(allBoxes) + (len(model.xBounds) - (k+1))
        return saveFailedIntervalSet
    
    # Successful unique solution test + solution numerically found:    
    if output.__contains__("uniqueSolutionInBox"):   
        results["xSolved"]+= [True]#output["xSolved"][0]
    else: 
        results["xSolved"] += output["xSolved"]
        
    bxrd_options["boxNo"] = (len(allBoxes) + len(output["xNewBounds"]) + 
                                 (len(model.xBounds) - (k+1)))          
    updateSetOfBoxes(model, allBoxes, model.xBounds[k], output, 
                     bxrd_options["boxNo"], k, bxrd_options, results)
    return []


def prepare_results_inconsistent_x(model, k, results, output, bxrd_options):
    """ writes results of box after contraction into dictionary results 
    
    Args:
        :model:         instance of type model
        :k:             integer with curremt box index                
        :results:       dictionary with results from reduction step
        :output:        dictionary with output from splitting and cutting
        :bxrd_options:  dictionary with reduction step
    
    """
    if len(output["xNewBounds"]) > 1:
        results["complete_parent_boxes"] += (len(output["xNewBounds"]) 
                                             * [[bxrd_options["iterNo"]-1, k]])
    else:
        results["complete_parent_boxes"] += (len(output["xNewBounds"]) * 
                                         [model.complete_parent_boxes[k]])
    if bxrd_options["cutBox"]=="tear" or bxrd_options["cutBox"]=="all":
        results["cut"] += len(output["xNewBounds"]) * [True]
    else: results["cut"] += len(output["xNewBounds"]) * [False]
    
    #cut += len(output["xNewBounds"]) * [True]


def prepare_results_splitted_x(model, k, results, output, bxrd_options):
    """ writes results of box after splitting and cutting into dictionary results 
    
    Args:
        :model:         instance of type model
        :k:             integer with curremt box index                
        :results:       dictionary with results from reduction step
        :output:        dictionary with output from splitting and cutting
        :bxrd_options:  dictionary with quantities from former reduction step
    
    """
    results["cut"] += output["cut"]
    #cut += output["cut"] 
    if model.teared: 
        results["complete_parent_boxes"] += (len(output["xNewBounds"]) 
                                             * [[bxrd_options["iterNo"]-1, k]])
        model.teared = False
    else: 
        results["complete_parent_boxes"] += (len(output["xNewBounds"]) * 
                                             [model.complete_parent_boxes[k]])


def prepare_results_constant_x(model, k, results, allBoxes, bxrd_options):
    """ writes results of already solved and currently not reducible boxes
    into results dictionary
    
    Args:
        :model:         instance of type model
        :k:             integer with curremt box index        
        :results:       dictionary with results from reduction step
        :allBoxes:      list with currently reduced Boxes
        :bxrd_options:  dictionary with quantities from former reduction step
    
    """
    results["xSolved"] += [bxrd_options["xSolved"][k]]
    results["xAlmostEqual"] += [True]
    allBoxes.append(model.xBounds[k])
    results["disconti"] += [bxrd_options["disconti"][k]]
    results["complete_parent_boxes"]  += [model.complete_parent_boxes[k]]
    results["cut"] += [False]    


def roundValue(val, digits):
    """ generates tightest interval around value val in accuracy of its last 
    digit so that its actual value is not lost because of round off errors.

    Args:
        :val:         sympy.Float value
        :digit:       integer number of digits

    Return: 
        tightest interval in mpmath.mpi formate
    
    """
    rounded_val = round(val, digits)
    if rounded_val == val:
        return mpmath.mpi(val)
    elif rounded_val > val:
        return mpmath.mpi(rounded_val - 10**(-digits), rounded_val)
    
    return mpmath.mpi(rounded_val, rounded_val + 10**(-digits))


def updateSetOfBoxes(model, allBoxes, xBounds, output, boxNo, k, bxrd_options, 
                     results):
    """ updates set of boxes with reduced boxes from current step. If the maximum
    number of boxes is exceeded the former will be put into the set instead.
    
    Args:
    :model:         instance of type Model
    :allBoxes:      list with already reduced boxes
    :xBounds:       numpy.array with former box
    :output:        dictionary with new box(es) and xAlmostEqual check for box
    :boxNo:         integer with current number of boxes (including new boxes)
    :k:             inter with index of former box
    :bxrd_options:  dictionary with user-specified maximum number of boxes
    :results:       dictionary for storage of results

    """    
    if boxNo <= bxrd_options["maxBoxNo"]:
        for box in output["xNewBounds"]: allBoxes.append(numpy.array(box, 
                                                                     dtype=object))
        results["disconti"] += output["disconti"]
        results["xAlmostEqual"] += output["xAlmostEqual"]
 
    else:# boxNo > bxrd_options["maxBoxNo"]:
        print("Warning: Algorithm stops the current box reduction because the",
              " current number of boxes is ", boxNo, "and exceeds the maximum", 
              "number of boxes that is ",
              bxrd_options["maxBoxNo"], "." )
        allBoxes.append(xBounds)
        results["xAlmostEqual"] += [True] 
        results["disconti"] += bxrd_options["disconti"][k]#output["disconti"]


def contractBox(xBounds, model, boxNo, bxrd_options):
    """ general contraction step that contains newton, HC4 and box reduction method    
    with and without parallelization. The combined algorithm is there for finding 
    an "efficient" alternation strategy between the contraction step methods
    
    Args:
    :xBounds:               numpy.array with current box
    :model:                 instance of class model
                            with function's glb id they appear in
    :boxNo:                 current number of boxes
    :bxrd_options:          dictionary with user specified algorithm settings
    
    Return:
        :output:            dictionary with results
    
    """    
    if not bxrd_options["parallelVariables"]:
        output = reduceBox(xBounds, model, boxNo, bxrd_options)
    else:
        output = parallelization.reduceBox(xBounds, model, boxNo, bxrd_options)

    return output


def reduceConsistentBox(model, bxrd_options, k, boxNo):
    """ reduces a consistent box after the contraction step
    
    Args:
        :model:              instance of class Model
        :bxrd_options:       dictionary with user-specifications
        :k:                  index of currently reduced box
        :boxNo:              integer with number of boxes
        :newtonMethods:     dictionary with netwon method names
        
    Return:
        :output:             modified dictionary with new split or cut box(es)

    """
    output = {"xNewBounds": [model.xBounds[k]],
              "xAlmostEqual": [bxrd_options["xAlmostEqual"][k]],
              "xSolved": [bxrd_options["xSolved"][k]],                      
                }   
     
    newBox = output["xNewBounds"]
    possibleCutOffs = False
    # if cutBox is chosen,parts of the box are now tried to cut off 
    if bxrd_options["cutBox"] in {"tear", "all", True} and bxrd_options["cut"][k]:
        print("Now box ", k, "is cutted")
        if bxrd_options["cutBox"] == "tear": 
            if model.tearVarsID == []: getTearVariables(model)
            newBox, possibleCutOffs = cut_off_box(model, newBox, bxrd_options,
                                                     model.tearVarsID)
            #newBox, possibleCutOffs = cutOffBox_tear(model, newBox, bxrd_options)
        if bxrd_options["cutBox"] == "all" or bxrd_options["cutBox"] == True : 
            #newBox, possibleCutOffs = cutOffBox(model, newBox, bxrd_options)
            newBox, possibleCutOffs = cut_off_box(model, newBox, bxrd_options)
        if bxrd_options["cutBox"] == "leastChanged":
            leastChanged_Id = split_least_changed_variable(newBox, model, k, bxrd_options)
            newBox, possibleCutOffs = cut_off_box(model, newBox, bxrd_options,
                                                     leastChanged_Id)
        if newBox == [] or newBox ==[[]]: 
            saveFailedSystem(output, model.functions[0], model, 0)
            output["disconti"]=[]
            output["cut"] = [True]  
            return output
        
        # If box could be cut it is returned for another contraction:        
        if possibleCutOffs:
            output["xNewBounds"] = [numpy.array(newBox[0])]
            output["xSolved"] = [variableSolved(newBox[0], bxrd_options)]
            output["xAlmostEqual"] = [False]
            output["disconti"] = [False]
            output["cut"] = [True]
            return output
    #if ((not possibleCutOffs or output["xAlmostEqual"][0]) and 
    #    (not "uniqueSolutionInBox" in output.keys())):  # if cutBox was not successful or it didn't help to reduce the box, then the box is now splitted      
        
    # Check if consistent box is allowed to be split:
    output["cut"] = [possibleCutOffs]
    
    if not bxrd_options["splitBox"] or bxrd_options["splitBox"]=="None":
       output["xAlmostEqual"] = [True]
       output["disconti"] = [False]
       return output   
   
    if "ready_for_reduction" in bxrd_options.keys(): 
       if not bxrd_options["ready_for_reduction"][k]: 
           output["disconti"] = [False] 
           return output
    boxNo_split = bxrd_options["maxBoxNo"] - boxNo
    
    if boxNo_split > 0:            
        if bxrd_options["debugMode"]: print("Now box", k, " is teared")
        model.teared = True
        output["xNewBounds"] = splitBox(newBox, model, bxrd_options, k, 
                                        boxNo_split)
        if output["xNewBounds"] == []:
            saveFailedSystem(output, model.functions[0], model, 0)
            return output        
        box = [convert_mpi_iv_float(iv) for iv in output["xNewBounds"][0]]
        if len( output["xNewBounds"]) == 1 and solved(box , bxrd_options):
            
            output["xSolved"] = [True]
        else:
            output["xSolved"] = [False] * len(output["xNewBounds"]) 
        output["xAlmostEqual"] = [False] * len(output["xNewBounds"])   
        output["cut"] = [True] * len(output["xNewBounds"])
        output["disconti"] = [False] * len(output["xNewBounds"]) 
    else:
       output["xAlmostEqual"] = [True]
       output["disconti"] = [False]
    return output


def checkBoxesForRootInclusion(functions, boxes, bxrd_options):
    """ checks if boxes are non-empty for EQS given by functions
    
    Args:
        :functions:        list with function objects
        :boxes:            list of boxes that are checked in mpmath.mpi formate
        :bxrd_options:     dictionary with box redcution settings
        
    Return:
        :nonEmptyboxes:    list with non-empty boxes with mpmath.mpi intervals

    """
    if boxes == []: return []
    nonEmptyboxes = [box for box in boxes if 
                    solutionInFunctionRange(functions, box, bxrd_options)]
    #if nonEmptyboxes:
    #    nonEmptyboxes = [box for box in boxes if 
    #                solutionInFunctionRangePyibex(functions, box, bxrd_options)]
    return nonEmptyboxes


def splitBox(consistentBox, model, bxrd_options, k, boxNo_split):
    """ box splitting algorithm should be invoked if contraction doesn't work 
    anymore because of consistency. The user-specified splitting method is 
    executed.
    
    Args:
       :xNewBounds:         numpy.array with consistent box
       :model:              instance of class model
       :bxrd_options:       dictionary with user-specifications
       :k:                  index of currently reduced box
       :boxNo_split:        number of possible splits (maxBoxNo-boxNo) could be 
                            used for multisection too
    
    Return:
        list with split boxes
        
    """
    if bxrd_options["splitBox"]=="tearVar": 
        # splits box by tear variables  
        if model.tearVarsID == []: getTearVariables(model)  
        tearId = getCurrentVarToSplit(model.tearVarsID, consistentBox[0], 
                                      bxrd_options)
        if tearId !=[]: split_var_id = [model.tearVarsID[tearId]]
        else: split_var_id = None
    elif bxrd_options["splitBox"] =="forecastTear":
        if model.tearVarsID == []: getTearVariables(model)  
        split_var_id = model.tearVarsID
    elif bxrd_options["splitBox"] =="forecastSplit":
        split_var_id = None
    elif bxrd_options["splitBox"] =="leastChanged":
        split_var_id = split_least_changed_variable(consistentBox, model, k, 
                                                    bxrd_options)
    elif bxrd_options["splitBox"]=="largestDer":  
        split_var_id = getTearVariablelargestDerivative(model, k)
    else:  split_var_id = None
        
    new_box = get_best_split_new(consistentBox, model, k, bxrd_options, split_var_id)

    new_box: new_box = checkBoxesForRootInclusion(model.functions, new_box, 
                                                     bxrd_options)    
    return new_box


def check_box_for_eq_consistency(model, box, bxrd_options):
    abs_old = bxrd_options["absTol"]
    relTol_old = bxrd_options["relTol"]
    bxrd_options["absTol"] *=100
    bxrd_options["relTol"] *=100     

    unsolved = [i for i in model.colPerm 
    if not checkVariableBound(box[i], bxrd_options)] 
    
    if len(unsolved) <= model.max_block_dim:
        bxrd_options["absTol"] *=1e-4
        bxrd_options["relTol"] *=1e-4 
        valid_box, cutoff = cut_off_box(model, [box], bxrd_options)
        bxrd_options["absTol"] = abs_old
        bxrd_options["relTol"] = relTol_old
        if not cutoff: return box
        if not valid_box: return []
        else: return valid_box[0]
    else:
        bxrd_options["absTol"] = abs_old
        bxrd_options["relTol"] = relTol_old        
        return box


def cutOffBox_tear(model, xBounds, bxrd_options):
    """ trys to cut off all empty sides of the box spanned by the tear variables
    to reduce the box without splitting-

    Args:
        :model:         instance of type model
        :xBounds:       current boxbounds in iv.mpmath
        :bxrd_options:  dictionary of options
    Return:
        :xNewBounds:    new xBounds with cut off sides
        :cutOff:        boolean if any cut offs are possible
    
    """
    if model.tearVarsID == []: getTearVariables(model)
    xNewBounds = list(xBounds[0])
    #tear_box = list(xBounds[0][i] for i in model.tearVarsID)
    cutOff=False
    xChanged = numpy.array([True]*len(model.tearVarsID))
    while xChanged.any(): 
        for u in model.tearVarsID:
            i=1
            while i<100: #number of cut offs are limited to 100 due to step sie of 0.01*delta
                CutBoxBounds = list(xNewBounds)
                xu = CutBoxBounds[u]
                if (mpmath.mpf(xu.delta) <= 
                    mpmath.mpf(xBounds[0][u].delta)*0.02*i): break #if total box is too small for further cut offs
                cur_x = (float(mpmath.mpf(xu.b)) - 
                         float(mpmath.mpf(xBounds[0][u].delta)*0.01*i))
                CutBoxBounds[u] = mpmath.mpi(cur_x, xu.b) #define small box to cut
                
                if not solutionInFunctionRangePyibex(model.functions, numpy.array(CutBoxBounds), bxrd_options): #check,if small box is empty
                #if not solutionInFunctionRange(functions, numpy.array(CutBoxBounds), bxrd_options):
                    xNewBounds[u] = mpmath.mpi(xu.a, cur_x)
                    cutOff = True
                    i+=1
                    continue                 
                else:
                    break
            if i == 1: xChanged[model.tearVarsID.index(u)] = False
            #try to cut off lower part
            i=1
            while i<100:
                CutBoxBounds = list(xNewBounds)
                xu = CutBoxBounds[u]
                if (mpmath.mpf(xu.delta) <= 
                    mpmath.mpf(xBounds[0][u].delta)*0.02*i): break
                cur_x = (float(mpmath.mpf(xu.a)) + 
                         float(mpmath.mpf(xBounds[0][u].delta)*0.01*i))
                CutBoxBounds[u] = mpmath.mpi(xu.a, cur_x)
                
                if not solutionInFunctionRangePyibex(model.functions, 
                                                     numpy.array(CutBoxBounds), 
                                                     bxrd_options): #check,if small box is empty
                    xNewBounds[u] = mpmath.mpi(cur_x, xu.b)
                    cutOff = True
                    i=i+1
                    continue
                else:
                    break
            if not xChanged[model.tearVarsID.index(u)] and not i ==1: 
                xChanged[model.tearVarsID.index(u)]=True
                
    return [list(xNewBounds)], cutOff


def cut_off_box(model, box, bxrd_options, cut_var_id=None):
    """ cuts off edge boxes if they are identifed as having no solution to 
    f(x)=0. The variables that are cut can be preselected by their global ids 
    in cut_var_id or all variables are cut otherwise.
    
    Args:
        :model:         instance of class model
        :box:           list with numpy array that contains box which needs to 
                        be cut
        :bxrd_options:  dictionary with absolute and relative tolerances
        :cut_var_id:    list with global indices referring to variables that
                        shall be cut, if not specified all variables are cut
                        
    Return:
        :new_box:       reduced box
        :cut_off:       boolean that is set true as soon as one empty edge box
                        could be removed
        
    """
    new_box = list(list(box[0]))
    cut_off = False    
    
    if not cut_var_id: 
        cut_var_id = [i for i in model.colPerm 
                      if not checkVariableBound(new_box[i], bxrd_options)] 
    else:       
        cut_var_id = [i for i in cut_var_id 
                        if not checkVariableBound(new_box[i], bxrd_options)]      
    cutBox = [new_box[i] for i in cut_var_id]
    
    xChanged = numpy.array([True]*len(cut_var_id))
    rstep_min = 0.01 
    rstep = rstep_min                      
    step = [rstep_min * float(mpmath.mpf(iv.delta)) for iv in cutBox]
    
    while xChanged.any():
        for cut_id, i in enumerate(cut_var_id):
            while(rstep <= 1.0):
                edge_box = list(new_box)
                xi = float(mpmath.mpf(edge_box[i].b)) - step[cut_id]
                edge_box[i] = mpmath.mpi(xi, edge_box[i].b)  
                if edge_box[i].delta == 0: break
                           
                (has_solution, 
                 rstep) = check_solution_in_edge_box(model, i, cut_id, 
                                                     float(mpmath.mpf(new_box[i].a)), 
                                                     xi, edge_box, new_box, 
                                                     step, rstep, bxrd_options)
                if not has_solution: 
                    cut_off = True
                    continue
                else: break
            if (rstep == rstep_min): xChanged[cut_id] = False
            elif rstep >= 1.0 and not has_solution: 
                return [], cut_off
            else: 
                rstep = rstep_min
                step[cut_id] = float(mpmath.mpf(new_box[i].delta)) * rstep
            
            while(rstep <= 1.0):
                edge_box = list(new_box)
                xi = float(mpmath.mpf(edge_box[i].a)) + step[cut_id]
                edge_box[i] = mpmath.mpi(edge_box[i].a, xi)   
                if edge_box[i].delta == 0: break
                           
                (has_solution, 
                 rstep) = check_solution_in_edge_box(model, i, cut_id, xi, 
                                                     float(mpmath.mpf(new_box[i].b)),
                                                     edge_box, new_box, step,
                                                     rstep, bxrd_options)
                if not has_solution: 
                    cut_off = True
                    continue
                else: break            

            if not xChanged[cut_id] and not rstep == rstep_min: 
                xChanged[cut_id] = True
            elif rstep >= 1.0 and not has_solution:
                return [], cut_off
            
            rstep = rstep_min
            step[cut_id] = (float(mpmath.mpf(new_box[i].delta)) * rstep)
                        
    return [tuple(new_box)], cut_off  


def check_solution_in_edge_box(model, i, cut_id, a, b, cur_box, new_box, step,
                               rstep, bxrd_options):
    """ checks if edge box is empty or not during the cutting process and 
    returns True if it has a solution and False otherwise. If edge box is empty 
    the remaining box (new_box) is updated and also the absolute and relative 
    step sizes
    
    Args:
        :model:         instance of class model
        :i:             integer with currently cutted variable's global index
        :cut_id:        integer with currently cutted variable's index of 
                        preselection
        :a:             lower bound of new box as float from current variable
        :b:             uper bound of new box as float from current variable
        :cur_box:       list of currently checked box for root inclusion
        :new_box:       list of non-empty box
        :step:          current step size as float for cutting
        :rstep:         current relative step size as float for cutting
        :bxrd_options:  dictionary with absTol and relTol for root inclusion 
                        test based on 3 HC4 steps
                        
    Return:
        :True/False:    True if cur_box is not emtpy and False otherwise
        :rstep:         updated relative step size
        
    """
    if not solutionInFunctionRangePyibex(model.functions, numpy.array(cur_box), 
                                         bxrd_options): 
        if b < a: return False, 1.1
        new_box[i] = mpmath.mpi(a, b)
        rstep = ((rstep*100.0)**0.5 + 1)**2/100.0
        step[cut_id] = (b - a) * rstep
        
        if rstep >= 1:
           return solutionInFunctionRangePyibex(model.functions, 
                                                numpy.array(new_box), 
                                                bxrd_options), rstep
        return False, rstep
    else:
        return True, rstep


def cutOffBox(model, xBounds, bxrd_options):
    '''trys to cut off all empty sides of the box, to reduce the box without splitting

    Args:
        :model:         instance of type model
        :xBounds:       current boxbounds in iv.mpmath
        :bxrd_options:  dictionary of options
        
    Return:
        :xNewBounds:    new xBounds with cut off sides
        :cutOff:        boolean if any cut offs are possible
        
    '''
    xNewBounds = list(list(xBounds[0]))
    
    cutOff=False
    xChanged = numpy.array([True]*len(model.xSymbolic))
    #while xChanged.any(): 
    for u in range(len(model.xSymbolic)):
        #try to cut off upper variable parts
        i=1
        while i<10: #number of cutt offs are limited to 100
            CutBoxBounds = list(list(xNewBounds))
            xu = CutBoxBounds[u]
            if (mpmath.mpf(xu.delta) <= 
                mpmath.mpf(xBounds[0][u].delta)*0.02*i): break #if total bounds is too small for further cut offs
            cur_x = (float(mpmath.mpf(xu.b)) - 
                     float(mpmath.mpf(xBounds[0][u].delta)*0.01*i))
            CutBoxBounds[u] = mpmath.mpi(cur_x, xu.b) #define small box to cut
        
            if not solutionInFunctionRangePyibex(model.functions, 
                                                 numpy.array(CutBoxBounds), 
                                                 bxrd_options): #check,if small box is empty
                xNewBounds[u] = mpmath.mpi(xu.a, cur_x)
                cutOff = True
                i=i+1
                continue
            else:
                break
        if i == 1: xChanged[u] = False
        #try to cut off lower part
        i=1
        while i<10:
            CutBoxBounds = list(list(xNewBounds))
            xu = CutBoxBounds[u]
            if (mpmath.mpf(xu.delta) <= 
                mpmath.mpf(xBounds[0][u].delta)*0.02*i): break
            cur_x = (float(mpmath.mpf(xu.a)) + 
                     float(mpmath.mpf(xBounds[0][u].delta)*0.01*i))
            CutBoxBounds[u] = mpmath.mpi(xu.a, cur_x)
            
            if not solutionInFunctionRangePyibex(model.functions, 
                                                 numpy.array(CutBoxBounds), 
                                                 bxrd_options):
                xNewBounds[u] = mpmath.mpi(cur_x, xu.b)
                cutOff = True
                i=i+1
                continue
            else:
                break
            if not xChanged[u] and not i ==1: xChanged[u]=True
    return [tuple(xNewBounds)], cutOff


def getTearVariables(model):
    """ identifies tear variables of system based on MC33 algorithm
    
    Args:
        :model:     instance of type model
        
    """
    model.jacobian = dM.getCasadiJandF(model.xSymbolic, model.fSymbolic)[0]
    jacobian = model.getCasadiJacobian()
    res_permutation = MC33.doMC33(jacobian)  
    tearsCount = max(res_permutation["Border Width"],1)
    model.tearVarsID =res_permutation["Column Permutation"][-tearsCount:]  


def getTearVariablelargestDerivative(model, boxNo):
    '''finds variable with highest derivative*equation_appearance for splitting

    Args:
        :model:     instance of type model
        :boxNo:     index of current box
        
    Return:
        :splitVar:  list with index of variable to split
        
    '''    
    subset = numpy.arange(len(model.xBounds[boxNo]))
    
    if model.VarFrequency==[]:
        model.VarFrequency = numpy.zeros((len(model.xBounds[boxNo])))
        for i in range(len(model.xSymbolic)):
            #frequency of equation apperances
            for f in model.fSymbolic:
                if model.xSymbolic[i] in f.free_symbols:
                    model.VarFrequency[i] = model.VarFrequency[i] + 1
    ''' frequency i jacobian
    for j in jacobian:
        model.VarFrequency[i] = model.VarFrequency[i] + j.count(model.xSymbolic[i])
    '''  
    jaclamb = model.jacobianLambNumpy
    
    #finds largest derivative of smallest, mid and largest boxpoint
    maxJacpoint = []
    for p in ['a','mid','b']:
        PointIndicator = len(model.xBounds[boxNo])*[p]
        Boxpoint = getPointInBox(model.xBounds[boxNo], PointIndicator)
        Jacpoint = jaclamb(*Boxpoint)
        Jacpoint = numpy.nan_to_num(Jacpoint)
        maxJacpoint.append(numpy.max(abs(Jacpoint), axis=0))
    
    #multiplies largest jacobian value of each component with its equation frequency   
    maxJacpoint = model.VarFrequency*numpy.max(maxJacpoint, axis=0)

    #sum of derivatives*frequency
    largestJacIVVal = -numpy.inf
    largestJacIVVarID = [0]
    for i in subset:
           if abs(maxJacpoint[i])>largestJacIVVal and float(
                   mpmath.convert(model.xBounds[boxNo][i].delta))>0.0001:
               largestJacIVVal = abs(maxJacpoint[i])
               largestJacIVVarID = [i]
    splitVar = largestJacIVVarID
    
    return splitVar


def get_best_split(box, model, boxNo, bxrd_options, split_var_id=None):
    '''finds variable, which splitting causes the best reduction

    Args:
        :box:           variable bounds of class momath.iv
        :model:         instance of type model
        :boxNo:         integer with number of boxes
        :bxrd_options:  dictionary of options
        :split_var_id:  list with global ids of potential split variables
      
    Return:
        :new_box:    best reduced two variable boxes
        
    '''      
    old_box = list(numpy.array(box)[0])  
    old_box_float = [[convert_mpi_float(iv.a), convert_mpi_float(iv.b)]  
                       for iv in old_box]
    smallestAvrSide = 2.0
    new_box = []
    # if split_var_id is availabe, solved variables from this set are removed
    if split_var_id:     
        split_var_id = [i for i in split_var_id 
                        if not checkVariableBound(old_box[i], bxrd_options)]
        
    # if all variables from split_var_id are solved or all variables shall be
    # tested for splitting, split_var_id then contains all unsovled variable ids
    if not split_var_id:
        split_var_id = model.colPerm
        split_var_id = [i for i in split_var_id 
                        if not checkVariableBound(old_box[i], bxrd_options)]
    # if split_var_id is still empty then all variables in this box are solved
    # and the box is returned    
    if not split_var_id:
        return [tuple(box[0])]
    
    # bisection and contraction is tested on all unsolved variable intervals from
    # split_var_id             
    for i, t in enumerate(split_var_id): 
        box = list(old_box)
        #splitBox = bisect_box(box, t)
        splitBox = bisect_box_adv(model, box, t, bxrd_options)
        #reduce both boxes
        output0, output1 = reduceHC4_orNewton(splitBox, model, boxNo, 
                                              bxrd_options)
        
        # if only one variable is split then no best split needs to be found
        # and the split boxes are directly returned
        if not new_box and i+1 == len(split_var_id): 
            return output0["xNewBounds"] + output1["xNewBounds"]

        # if first split box is not empty average length is calculated
        elif output0["xNewBounds"] != [] and output0["xNewBounds"] != [[]]:
            new_box_float = [[convert_mpi_float(iv.a), convert_mpi_float(iv.b)]  
                       for iv in output0["xNewBounds"][0]]            
            avrSide0 = analysis.identify_box_reduction(new_box_float, old_box_float)
        else:
            # if first split box is empty the second box is returned
            if bxrd_options["debugMode"]:
                print("This is the current best splitted variable by an empty box ", 
                  model.xSymbolic[t])
            return [tuple(splitBox[1])]
        # if second split box is not empty average length is calculated
        if output1["xNewBounds"] != [] and output1["xNewBounds"] != [[]]:
            new_box_float = [[convert_mpi_float(iv.a), convert_mpi_float(iv.b)]  
                       for iv in output1["xNewBounds"][0]]            
            avrSide1 = analysis.identify_box_reduction(new_box_float, old_box_float)
            
        else:
            # if second split box is empty the first box is returned
            if bxrd_options["debugMode"]:
                print("This is the current best splitted variable by an empty box: ", 
                  model.xSymbolic[t])
            return [tuple(splitBox[0])]     
        # sum of both boxreductions
        avrSide = avrSide0 + avrSide1
        # find best overall boxredution
        if avrSide < smallestAvrSide:
            smallestAvrSide = avrSide
            if bxrd_options["debugMode"]: 
                print("variable ", model.xSymbolic[t], " is splitted")
            new_box = output0["xNewBounds"] + output1["xNewBounds"]
            
    return new_box


def get_best_split_new(box, model, boxNo, bxrd_options, split_var_id=None):
    '''finds variable, which splitting causes the best reduction

    Args:
        :box:           variable bounds of class momath.iv
        :model:         instance of type model
        :boxNo:         integer with number of boxes
        :bxrd_options:  dictionary of options
        :split_var_id:  list with global ids of potential split variables
      
    Return:
        :new_box:    best reduced two variable boxes
        
    '''      
    old_box = list(numpy.array(box)[0])  
    old_box_float = [[convert_mpi_float(iv.a), convert_mpi_float(iv.b)]  
                       for iv in old_box]
    smallestAvrSide = 2.0
    new_box = []
    two_boxes = False
    
    # bisection and contraction is tested on all unsolved variable intervals from
    # split_var_id  
    while not two_boxes:   
        if split_var_id:     
            split_var_id = [i for i in split_var_id 
                            if not checkVariableBound(old_box[i], bxrd_options)]
            
        # if all variables from split_var_id are solved or all variables shall be
        # tested for splitting, split_var_id then contains all unsovled variable ids
        if not split_var_id:
            split_var_id = model.colPerm
            split_var_id = [i for i in split_var_id 
                            if not checkVariableBound(old_box[i], bxrd_options)]
        # if split_var_id is still empty then all variables in this box are solved
        # and the box is returned    
        if not split_var_id:
            return [old_box]
        two_boxes = True  
        #split_var_id = [i for i in split_var_id 
        #                if not checkVariableBound(old_box[i], bxrd_options)]        
        for i, t in enumerate(split_var_id): 
            box = list(old_box)
            splitBox = bisect_box_adv(model, box, t, bxrd_options)
            #reduce both boxes
            output0, output1 = reduceHC4_orNewton(splitBox, model, boxNo, 
                                                  bxrd_options)
            
            # if only one variable is split then no best split needs to be found
            # and the split boxes are directly returned
            if not new_box and i+1 == len(split_var_id): 
                return output0["xNewBounds"] + output1["xNewBounds"]

            box0_exists = (output0["xNewBounds"] != [] and output0["xNewBounds"] != [[]])
            box1_exists = (output1["xNewBounds"] != [] and output1["xNewBounds"] != [[]])
            if not  box0_exists and not box1_exists: return []  
            elif not box0_exists:
                old_box = output1["xNewBounds"][0]
                smallestAvrSide = 2.0
                two_boxes = False
                break
            elif not box1_exists:
                old_box = output0["xNewBounds"][0]
                smallestAvrSide = 2.0
                two_boxes = False
                break
            else:    
                new_box_float = [[convert_mpi_float(iv.a), convert_mpi_float(iv.b)]  
                           for iv in output0["xNewBounds"][0]] 
                avrSide = analysis.identify_box_reduction(new_box_float, old_box_float)
                new_box_float = [[convert_mpi_float(iv.a), convert_mpi_float(iv.b)]  
                           for iv in output1["xNewBounds"][0]] 
                avrSide += analysis.identify_box_reduction(new_box_float, old_box_float)
            
            # if first split box is not empty average length is calculated

                if avrSide < smallestAvrSide:
                    smallestAvrSide = avrSide
                    if bxrd_options["debugMode"]: 
                        print("variable ", model.xSymbolic[t], " is splitted")
                    new_box = output0["xNewBounds"] + output1["xNewBounds"]
            
    return new_box


def reduceHC4_orNewton(splittedBox, model, boxNo, bxrd_options):
    '''reduces both side of the splitted box with detNewton or HC4

    Args:
        :splittedBox:   list of two boxes with variable bounds of class momath.iv
        :model:         instance of type model
        :boxNo:         integer with number of boxes
        :bxrd_options:  dictionary of options
        
    Return:
        :output0:    reduced box 1
        :output1:    reduced box 2
    '''    
    bxrd_options_temp = bxrd_options.copy()
    bxrd_options_temp.update({"hcMethod":'HC4',#bxrd_options["hcMethod"], 
                                  "bcMethod":'None',
                                  "newtonMethod": 'None',#bxrd_options["newtonMethod"],
                                  "InverseOrHybrid": 'None', 
                                  "affineArithmetic": False})       
    output0 = reduceBox(numpy.array(splittedBox[0]), model, boxNo, bxrd_options_temp)
    output1 = reduceBox(numpy.array(splittedBox[1]), model, boxNo, bxrd_options_temp)
              
    return output0, output1
    

# def identify_interval_reduction(box_new,box_old):
#     '''calculates the side length ratio of all intervals from a new box
#     to an old box

#     Args:
#         :box_new:        new variable bounds of class momath.iv
#         :box_old:        old variable bounds of class momath.iv
        
#     Return:
#         :sideLengthsRatio:    list with side length ratio for all interval
        
#     '''
#     sideLengthsRatio = []
    
#     for i, iv in enumerate(box_old):
#         if (iv[1]-iv[0])>0: 
#             sideLengthsRatio.append((box_new[i][1] - box_new[i][0]) / (iv[1]-iv[0]))
#         else:
#             sideLengthsRatio.append(0)
                                             
#     return sideLengthsRatio
 
       
# def identifyReduction(newBox,oldBox):
#     '''calculates the average side length reduction from old to new Box

#     Args:
#         :newBox:        new variable bounds of class momath.iv
#         :oldBox:        old variable bounds of class momath.iv
        
#     Return:
#         :avrSideLength/len(oldBox):    average sidelength reduction
        
#     '''
#     avrSideLength = 0
    
#     for i, iv in enumerate(oldBox): 
#         if (iv[1] -iv[0])>0:
#             avrSideLength += (newBox[i][1] - newBox[i][0])/(iv[1] - iv[0])    
                                                 
#     return avrSideLength/len(oldBox)


def splitTearVars(tearVarIds, box, boxNo_max, bxrd_options):
    """ splits unchanged box by one of its alternating tear variables
    
    Args:
        :tearVarIds:    list with global id's of tear variables
        :box:           numpy array intervals in mpmath.mpi formate
        :boxNo_max:     currently available number of boxes to maximum
        :bxrd_options:  dictionary with user specific settings
    
    Return: two sub boxes bisected by alternating tear variables from bxrd_options
        
    """
    
    if tearVarIds == [] or boxNo_max <= 0 : return [box], bxrd_options["tear_id"]
    iN = getCurrentVarToSplit(tearVarIds, box, bxrd_options)
    
    if iN == []: return [box], bxrd_options["tear_id"]
    print("Variable ", tearVarIds[iN], " is now splitted.")
    
    return bisect_box(box, tearVarIds[iN]), iN + 1


def getCurrentVarToSplit(tearVarIds, box, bxrd_options):
    """ returns current tear variable id in tearVarIds for bisection. Only tear
    variables with nonzero widths are selected. 
    
    Args:
        :tearVarIds:    list with global id's of tear variables
        :box:           numpy array intervals in mpmath.mpi formate   
        :bxrd_options:  dictionary with user specific settings  
        
    Return:
        :i:             current tear variable for bisection
        
    """
    i = bxrd_options["tear_id"]
    
    if i  > len(tearVarIds) - 1: i = 0   
    if checkIntervalWidth([box[i] for i in tearVarIds], bxrd_options["absTol"],
                            bxrd_options["relTol"]) == []:
        return []
         
    while checkIntervalWidth([box[tearVarIds[i]]], bxrd_options["absTol"],
                             bxrd_options["relTol"]) == []:
        if i  < len(tearVarIds) - 1: i+=1
        else: i = 0
    
    else: 
        bxrd_options["tear_id"] = i
        return i
    
        
# def separateBox(box, varID):
#     """ bi/multisects a box by variables with globalID in varID
    
#     Args:
#         :box:       numpy.array with variable bounds
#         :varID:     list with globalIDs of variables chosen for bisection        
        
#     Returns:
#         numpy.array wit sub boxes
        
#     """     
#     for i, interval in enumerate(box):
#         if i in varID:
#           box[i]=[mpmath.mpi(interval.a, interval.mid), mpmath.mpi(interval.mid, 
#                                                                    interval.b)]
#         else:box[i]=[interval]
        
#     return list(itertools.product(*box))

def bisect_box_adv(model, box, varID, bxrd_options):
    """ bisects a box by the variable with the global index varID
    
    Args:
        :box:       numpy.array with variable bounds
        :varID:     integer with global index of variable chosen for bisection        
        
    Returns:
        list with subboxes
        
    """
    box_low = list(box)
    box_up = list(box)
    if box[varID].mid == 0:
        box_mid = list(box)
        box_mid[varID] = bxrd_options["absTol"] * mpmath.mpi(-1,1)    
        empty = not solutionInFunctionRangePyibex(model.functions, numpy.array(box_mid), 
                                                bxrd_options)
        if empty:
            box_low[varID] = mpmath.mpi(box[varID].a, -bxrd_options["absTol"])
            box_up[varID] = mpmath.mpi(bxrd_options["absTol"], box[varID].b)
            return [box_low, box_up]
        
    if (0 <= box[varID].a and box[varID].a < 1.0e-6 and 
        box[varID].b > 1.0e-6):
        box_low[varID] = mpmath.mpi(box[varID].a, 1.0e-6)  
        box_up[varID] = mpmath.mpi(1.0e-6, box[varID].b) 
    if (0 >= box[varID].b and box[varID].b > -1.0e-6 and 
        box[varID].a < -1.0e-6):
        box_low[varID] = mpmath.mpi(box[varID].a, -1.0e-6)  
        box_up[varID] = mpmath.mpi(-1.0e-6, box[varID].b)
    else:
    # a = convert_mpi_float(box[varID].a)
    # b =  convert_mpi_float(box[varID].b)
    
    # if a != 0 and b != 0: 
    #     oa = math.floor(math.log(abs(a), 10))
    #     ob = math.floor(math.log(abs(b), 10))
    #     od = ob - oa
    #     if od > 0:
    #         if oa < ob and a >= 0: 
    #             mid = a + 10**(oa + od/2.0)
    #             box_low[varID] = mpmath.mpi(a, mid)   
    #             box_up[varID] = mpmath.mpi(mid, b)     
    #         if oa > ob and b <=0: 
    #             mid = b - 10**(ob + od/2.0)
    #             box_low[varID] = mpmath.mpi(a, mid)   
    #             box_up[varID] = mpmath.mpi(mid, b)
    #         else:
        box_low[varID] = mpmath.mpi(box[varID].a, box[varID].mid)   
        box_up[varID] = mpmath.mpi(box[varID].mid,box[varID].b)
    
    return [box_low, box_up]
    

def bisect_box(box, varID):
    """ bisects a box by the variable with the global index varID
    
    Args:
        :box:       numpy.array with variable bounds
        :varID:     integer with global index of variable chosen for bisection        
        
    Returns:
        list with subboxes
        
    """
    box_low = list(box)
    box_up = list(box)
    box_low[varID] = mpmath.mpi(box[varID].a, box[varID].mid)
    
    
    box_up[varID] = mpmath.mpi(box[varID].mid,box[varID].b)
    
    return [box_low, box_up]


def getPointInBox(xBounds, pointIndicator):
    '''returns lowest(a), highest(b) or Midpoint(mid)
    out of the Box 
    
    Args:
        :xBounds: current Bounds as iv.mpi
        :pointIndicator: a, b or mid as String
        Return:
        :Boxpoint: chosen Boxpoint
        
    '''  
    Boxpoint = numpy.zeros(len(xBounds), dtype=float)
    for i in range(len(xBounds)):
        if pointIndicator[i]=='a':
            Boxpoint[i] = sympy.Float(xBounds[i].a)
        elif pointIndicator[i]=='b':
            Boxpoint[i] = sympy.Float(xBounds[i].b)
        else:
            Boxpoint[i] = sympy.Float(xBounds[i].mid)	   	    
    return Boxpoint


def removeInfAndConvertToFloat(array, subs):
    '''removes inf in 2-dimensional array
    
    Args:
        :array:     2-dimensional array
        :subs:      value, the inf-iv is substituded with

    Return:
        :array:     numpy array as float
        
    '''
    for l in range(0, len(array)):
        for n, iv in enumerate(array[l]):
            if iv == float('inf'):
                array[l][n] = numpy.nan_to_num(numpy.inf)
            if iv == float('-inf'):
            	array[l][n] = -numpy.nan_to_num(numpy.inf)
            if iv == mpmath.mpi('-inf','+inf'):
                array[l][n] = subs
            elif isinstance(iv, mpmath.ctx_iv.ivmpf):
                if iv.a == '-inf': iv = mpmath.mpi(iv.b, iv.b)
                if iv.b == '+inf': iv = mpmath.mpi(iv.a, iv.a)
                array[l][n] = float(mpmath.mpmathify(iv.mid))
    
    array = numpy.array(array, dtype='float')
    
    return array


# def get_failed_output(f, varBounds):
#     """ collects information about variable bound reduction in function f
#     Args:
#         :f:             instance of class function
#         :varBounds:     dictionary with informaiton about failed variable bound

#     Return:
#         :output:        dictionary with information about failed variable bound
#                         reduction

#     """

#     output = {}
#     output["xNewBounds"] = []
#     failedSystem = FailedSystem(f.f_sym, f.x_sym[varBounds['Failed_xID']])
#     output["noSolution"] = failedSystem
#     output["xAlmostEqual"] = False
#     output["xSolved"] = False
    
#     return output
       

def getPrecision(xBounds):
    """ calculates precision for intervalnesting procedure (when intervals are
    joined to one interval)
    Args:
        :xBounds:         list with iteration variable bounds in mpmath.mpi formate

    Return:
        :precision:       as float value

    """
    allValuesOfx = []
    for x in xBounds:
        allValuesOfx.append(numpy.abs(float(mpmath.mpf(x.a))))
        allValuesOfx.append(numpy.abs(float(mpmath.mpf(x.b))))
    minValue = min(filter(None, allValuesOfx))
    
    return 5*10**(numpy.floor(numpy.log10(minValue))-2)


def reduceBox(xBounds, model, boxNo, bxrd_options):
    """ reduce box spanned by current intervals of xBounds.
     
    Args: 
        :xBounds:            numpy array with box
        :model:              instance of class Model
                             with function's glb id they appear in   
        :boxNo:              number of boxes as integer  
        :bxrd_options:       dictionary with user specified algorithm settings
            
    Returns:
        :output:             dictionary with new boxes in a list and
                             eventually an instance of class failedSystem if
                             the procedure failed.
                        
    """  
    subBoxNo = 1
    output = {}
    output["disconti"] = [False]         
    xSolved = True
    consistent = False # relevant for newton, bnormal and hc4
    bxrd_options_temp = bxrd_options.copy()
    
    # Prepare settings for unique solution test ref. to applied contraction methods
    nwt_enabled = (bxrd_options['newtonMethod'] 
                   in {'newton', 'detNewton', '3PNewton', 'mJNewton'})
    hc_enabled = (bxrd_options['hcMethod'] in {'HC4'})                                                 
    bc_enabled = (bxrd_options['bcMethod'] in {'bnormal'})
    bxrd_options_temp["x_old"] = list(xBounds)
                                         
    # cycling through contraction methods until they are all consistent:                                  
    while not consistent: 

        x_last = list(xBounds) # Store box for consistency check
        [bxrd_options_temp["unique_nwt"], 
         bxrd_options_temp["unique_hc"], 
         bxrd_options_temp["unique_bc"]] = set_unique_solution_true(nwt_enabled, 
                                                                    hc_enabled, 
                                                                    bc_enabled)
        
        # if HC4 is active
        if hc_enabled:
            (output, empty, x_HC4) = do_HC4(model, xBounds, output, 
                                            bxrd_options_temp)        
            if empty: return output
            if "uniqueSolutionInBox" in output.keys():
                return prepare_output(bxrd_options, output, True, [x_HC4], 
                                      True)
            xBounds = list(x_HC4)

        # Preparation for contraction with iv_newton or bnormal
        xNewBounds = list(xBounds)
        subBoxNo = 1  
        # update current old x to see progress in each contraction method
        bxrd_options_temp["x_old"]=list(xBounds) 
                              
        for i in model.colPerm:
            y = [xBounds[i]] # currently reduced variable
            if bxrd_options["debugMode"]: print(i)
            # if y is already solved tolerances are decreased to contract y 
            #further because other variables may rely on y
            checkIntervalAccuracy(xBounds, i, bxrd_options_temp)

            # if any newton method is active       
            if (not variableSolved(y, bxrd_options_temp) and nwt_enabled): 
                # Iv Newton step:
                y_new = iv_newton(model, xBounds, i, bxrd_options_temp)
                if y_new == [] or y_new ==[[]]: 
                    subBox_failed = [xBounds[xid] for xid in 
                                     model.functions[i].glb_ID]
                    saveFailedSystem(output, model.functions[i], model, i,
                                     subBox_failed)
                    return output
                else: 
                    y = check_contracted_set(model, y_new, i, xBounds, bxrd_options)
   
            # if bnormal is active
            if (not variableSolved(y, bxrd_options_temp) and bc_enabled):                
                if bxrd_options_temp["unique_bc"]: unique_test_bc = True
                else: unique_test_bc = False            
                f_for_unique_test_bc = False

                # Bnormal step:
                for j in model.dict_varId_fIds[i]:   
                    
                    y_new = do_bnormal(model.functions[j], xBounds, y, i, 
                                       bxrd_options_temp)  
                    if unique_test_bc: # unique solution test
                        (fbc, ubc) = update_for_unique_test(j, 
                                                            model.dict_varId_fIds[i][-1],
                                                            f_for_unique_test_bc,
                                                            bxrd_options_temp["unique_bc"])
                        f_for_unique_test_bc = fbc
                        bxrd_options_temp["unique_bc"] = ubc                            
                    # if discontinuity has been detected during reduction 
                    # it could be used for split order by discontinuities:                                     
                    if ("disconti_iv" in bxrd_options_temp.keys() and 
                        bxrd_options["considerDisconti"]):
                        output["disconti"] = [True]
                        del bxrd_options_temp["disconti_iv"]  
                    if y_new == [] or y_new ==[[]]: 
                        subBox_failed = [xBounds[xid] for xid in 
                                         model.functions[j].glb_ID]
                        saveFailedSystem(output, model.functions[j], model, i, 
                                         subBox_failed)
                        return output
                    else: 
                        y = check_contracted_set(model, y_new, i, xBounds, 
                                             bxrd_options)
                    if variableSolved(y, bxrd_options_temp): 
                        if f_for_unique_test_bc: 
                            bxrd_options_temp["unique_bc"] = True
                        break                     
                # Turns unique_bc to true if function for unique solution 
                # criterion in interval has been found:
                if f_for_unique_test_bc: 
                    bxrd_options_temp["unique_bc"] = True
  
            # Update quantities:
            if ((boxNo-1) + subBoxNo * len(y)) > bxrd_options["maxBoxNo"]:  
                y = [mpmath.mpi(min([yi.a for yi in y]),max([yi.b for yi in y]))]
                output["disconti"]=[True]

            (xSolved, 
             subBoxNo) = update_quantities(y, i, subBoxNo, xNewBounds, xBounds,
                                           xSolved, bxrd_options, 
                                           bxrd_options_temp)
        
        xNewBounds = check_uniqueness(output, xNewBounds, bxrd_options_temp) 
        if "uniqueSolutionInBox" in output.keys(): 
                return prepare_output(bxrd_options, output, True, xNewBounds, 
                                      True)    
        # Check for consistency:                             
        consistent = compare_with_last_box(xBounds, x_last, bxrd_options)    

    # Update contraction data:    
    xNewBounds = list(itertools.product(*xNewBounds))

    return prepare_output(bxrd_options, output, xSolved, xNewBounds, consistent)


def compare_with_last_box(box, box_old, options):
    """ checks two boxes regarding consistency and returns True if they are
    consistent.

    Args:
    :box:               list or numpy array with variable intervals in 
                        mpmath.mpi formate
    :box_old:           list or numpy array with variable intervals in 
                        mpmath.mpi formate from former reduction step
    :options:           dictionary with tolerances

    Returns:
    :True/False:        True if consistent and False otherwise
                        
    """
    for i,iv in enumerate(box_old): 
            if not checkXforEquality(iv, box[i], True, options): return False 
    return True


def check_uniqueness(output, xNewBounds, options):
    """ checks boxes that fulfill unique solution criterion for containing
    already numiercally found solutions
  
    Args:
    :output:            dictionary with contraction results
    :xNewBounds:        list with contracted intervals
    :options:           dictionary with tolerances and numerical solutions

    Returns
    :xNewBounds:        either newly reduced box if numerical solution in box
                        was found so that solved box can directly be returned
                        or the list with contracted intervals

    """
    new_box = list(itertools.product(*xNewBounds))
    if ((options["unique_nwt"] or options["unique_bc"]) 
        and "FoundSolutions" in options.keys()):     
        if test_for_root_inclusion(new_box[0], options["FoundSolutions"], 
                               options["absTol"]):
            output["uniqueSolutionInBox"] = True
            return new_box
            
        else: output["box_has_unique_solution"] = True
        
    elif (options["unique_nwt"] or 
      options["unique_bc"]):
        output["box_has_unique_solution"] = True
        
    return xNewBounds
    

def check_contracted_set(model, y, i, box, bxrd_options):
    """ referring to the number of sub-intervals y consists, box is directly
    updated if dim(y) = 1, y is checked for root inclusion in the entire system
    if dim(y) > 1, and if dimension is still > 1 after the check the box is 
    updated wit h y's min/max values, i.e. the hull of all subintervals

    Args:
    :model:             instance of type model
    :y:                 list with contracted intervals of i-th variable
    :i:                 integer with global id of variable
    :box:               list or numpy array with variable intervals in 
                        mpmath.mpi formate
    :bxrd_options:      dictionary with tolerances and numerical solutions

    Returns:
    :y:                 updated list of contracted intervals of i-th variable
                        
    """
    if len(y) > 1: 
        y = prove_list_for_root_inclusion(y, box, i, model.functions, 
                                          bxrd_options)
        if len(y)>1:
            y_float = [[convert_mpi_float(iv.a),convert_mpi_float(iv.b)] for iv in y]
            if not isinstance(box, list): box = list(box)
            box[i] = mpmath.mpi(min(min(y_float)), max(max(y_float)))
    if len(y)==1 and y[0]!=box[i]: 
        #box = list(box)
        if not isinstance(box, list): box = list(box)
        box[i] = y[0]
    return y


# def reduceBox_old(xBounds, model, boxNo, bxrd_options):
#     """ reduce box spanned by current intervals of xBounds.
     
#     Args: 
#         :xBounds:            numpy array with box
#         :model:              instance of class Model
#                              with function's glb id they appear in   
#         :boxNo:              number of boxes as integer  
#         :bxrd_options:       dictionary with user specified algorithm settings
            
#     Returns:
#         :output:             dictionary with new boxes in a list and
#                              eventually an instance of class failedSystem if
#                              the procedure failed.
                        
#     """  
#     subBoxNo = 1
#     output = {}
#     output["disconti"] = [False]   
#     bxrd_options_temp = bxrd_options.copy()      
#     xNewBounds = list(xBounds)
#     bxrd_options_temp["x_old"] = list(xBounds)
#     xUnchanged = True
#     xSolved = True
    
#     # Prepare settings for unique solution test ref. to applied contraction methods
#     newtonMethods = {'newton', 'detNewton', '3PNewton', 'mJNewton'}
#     hcMethods = {'HC4'}
#     bcMethods = {'bnormal'}
#     (bxrd_options_temp["unique_nwt"], 
#      bxrd_options_temp["unique_hc"], 
#      bxrd_options_temp["unique_bc"]) = set_unique_solution_true(bxrd_options, 
#                                                                 newtonMethods, 
#                                                                 hcMethods, 
#                                                                 bcMethods)
                                      
#     # if HC4 is active
#     if bxrd_options['hcMethod']=='HC4':
#         (output, empty, 
#          xNewBounds) = do_HC4(model, xBounds, output, bxrd_options_temp)        
#         if empty: return output
#         if "uniqueSolutionInBox" in output.keys(): 
#             return prepare_output(bxrd_options, output, xSolved, [xNewBounds], 
#                                   xUnchanged)
#         # Update processed bounds to HC4 results     
#         xBounds = list(xNewBounds)
    
#     # use decomposed order to avoid structural zero in iv_newton         
#     for i in model.colPerm: 
#         y = [xNewBounds[i]]
#         if bxrd_options["debugMode"]: print(i)
#         # Adjust tolerances for solved variables so that they can be further 
#         # reduced in order to solve other variables
#         checkIntervalAccuracy(xNewBounds, i, bxrd_options_temp)
   
#         # if any newton method is active       
#         if not variableSolved(y, bxrd_options_temp) and bxrd_options['newtonMethod'] in newtonMethods:
#             y = iv_newton(model, xBounds, i, bxrd_options_temp)
#             #if not unique: 
#             #    bxrd_options_temp["unique_nwt"] = False
               
#             if len(y) > 1: 
#                 y = prove_list_for_root_inclusion(y, xBounds, i, model.functions, 
#                                                   bxrd_options)
#             if y == [] or y ==[[]]: 
#                 saveFailedSystem(output, model.functions[0], model, i)
#                 return output
#             elif len(y)==1 and y[0]!=xBounds[i]: xBounds[i] = y[0]
           
#         # if bnormal is active
#         if not variableSolved(y, bxrd_options_temp) and bxrd_options['bcMethod']=='bnormal':
#             if bxrd_options_temp["unique_bc"]: unique_test_bc = True
#             else: unique_test_bc = False            
#             f_for_unique_test_bc = False
           
#             for j in model.dict_varId_fIds[i]:
                 
#                 y = do_bnormal(model.functions[j], xBounds, y, i, 
#                                    bxrd_options_temp)
#                 if unique_test_bc:
#                     (f_for_unique_test_bc, 
#                      bxrd_options_temp["unique_bc"]) = update_for_unique_test(j, 
#                                                                               model.dict_varId_fIds[i][-1],
#                                                                               f_for_unique_test_bc,
#                                                                               bxrd_options_temp["unique_bc"])
#                 if "disconti_iv" in bxrd_options_temp.keys() and bxrd_options["considerDisconti"]:
#                     output["disconti"] = [True]
#                     del bxrd_options_temp["disconti_iv"]  
#                 elif len(y) > 1: 
#                     y = prove_list_for_root_inclusion(y, xBounds, i, model.functions, 
#                                                   bxrd_options)
#                 if y == [] or y ==[[]]: 
#                     saveFailedSystem(output, model.functions[j], model, i)
#                     return output
#                 elif len(y)==1 and y[0]!=xBounds[i]: xBounds[i] = y[0]
                
#                 if variableSolved(y, bxrd_options_temp): 
#                     if f_for_unique_test_bc: bxrd_options_temp["unique_bc"] = True
#                     break
#             if f_for_unique_test_bc: 
#                 bxrd_options_temp["unique_bc"] = True
#                 #f_for_unique_test_bc = False
        
#         if ((boxNo-1) + subBoxNo * len(y)) > bxrd_options["maxBoxNo"]:
            
#             y = [mpmath.mpi(min([yi.a for yi in y]),max([yi.b for yi in y]))]
#             output["disconti"]=[True]
#         # Update quantities
#         xSolved, xUnchanged, subBoxNo = update_quantities(y, i, subBoxNo, xNewBounds, xBounds,
#                                                           xUnchanged, xSolved, 
#                                                           bxrd_options, bxrd_options_temp)
#     # Uniqueness test of solution:
#     xNewBounds = list(itertools.product(*xNewBounds))
#     if (bxrd_options_temp["unique_nwt"] or bxrd_options_temp["unique_hc"] or 
#         bxrd_options_temp["unique_bc"]) and "FoundSolutions" in bxrd_options.keys(): 
#         if test_for_root_inclusion(xNewBounds[0], bxrd_options["FoundSolutions"], 
#                                    bxrd_options_temp["absTol"]):
#             output["uniqueSolutionInBox"] = True
#             xSolved = True
#             xUnchanged = True
#         else: output["box_has_unique_solution"] = True
#     elif (bxrd_options_temp["unique_nwt"] or bxrd_options_temp["unique_hc"] or 
#           bxrd_options_temp["unique_bc"]):# and not xUnchanged:
#         output["box_has_unique_solution"] = True

#     return prepare_output(bxrd_options, output, xSolved, xNewBounds, xUnchanged)


def test_for_root_inclusion(box, solutions, absTol):
    """ tests if there is any solution in solutions that is insight of box or
    outsight but in a small distance given by absTol to one of the bounds.
    
    Args:
        :box:           numpy array of current bounds
        :solutions:     list with current solution points from numerical 
                        iteration
        :absTol:        float with maximum absolute distance outsight of bounds
        
    Return:
        True if such a solution exists and False otherwhise

    """
    for solution in solutions:
        solved = True
        for i, x in enumerate(solution):
            if x in box[i]: continue
            elif (isclose_ordered(float(mpmath.mpf(box[i].a)),x, 0.0, absTol) or
                  isclose_ordered(float(mpmath.mpf(box[i].b)),x, 0.0, absTol) ):
                print("Warning: Inaccuracy possible.")
                continue
            else: 
                solved = False 
                break
        if solved: 
            print("Successful Root Inclusion test by numerical iteration")
            return True
    return False
    

def prove_list_for_root_inclusion(interval_list, box, i, functions, bxrd_options):
    """ checks a list of intervals of a certain variable if they are non-empty
    in the constrained problem given by functions and box.
    
    Args:
        :interval_list:         list with a specific variable's intervals
        :box:                   box as list with mpmath.mpi intervals
        :i:                     id of currently reduced variable as int
        :functions:             list with function objects
        :bxrd_options:          dictionary with settings of box reduction
        
    Return:
        :interval_list:         list with non-empty intervals of the variable
        
    """
    cur_box = list(box)
    for iv in interval_list:
        cur_box[i] = iv
        if not solutionInFunctionRangePyibex(functions, cur_box, bxrd_options):
            interval_list.remove(iv)         
    return interval_list


def set_unique_solution_true(nwt_enabled, hc_enabled, bc_enabled):
    """ initializes the list unique_solution for root inclusion tests in
    a box. It is set to False if method is not used in the user-specified run.
    
    Args:
        :nwt_enabled:     boolean true if interval newton is used
        :hc_enabled:      boolean true if hc method is used
        :bc_enabled:      boolean true if hc method is used
        
    Returns:
        :unique_solution_enabled:     boolean if root inclusion test in HC-method
        
    """    
    unique_solution_test_enabled = []
    for enabled in [nwt_enabled, hc_enabled, bc_enabled]:
        if enabled:  unique_solution_test_enabled.append(True)
        else: unique_solution_test_enabled.append(False)
        
    return unique_solution_test_enabled


def prepare_output(bxrd_options, output, xSolved, xNewBounds, consistent):   
    """ prepares dictionary with output of box reduction step
    
    Args:
        :bxrd_options:      dictionary with box reduction settings
        :output:            dictionary with output quantities of box reduction
                            step
        :xSolved:           list with boolean if box intervals are solved
        :xNewBounds:        list with reduced variable bounds
        :consistent:        list with boolean if box intervals still change
                            
    Return
        :output:            dictionary with output

    """
    if "uniqueSolutionInBox" in output.keys():
        xSolved = True
        consistent = True 
    output["xNewBounds"] = xNewBounds
    if len(output["xNewBounds"])>1: 
        output["xAlmostEqual"] = [True] * len(output["xNewBounds"])
        output["xSolved"] = [xSolved] * len(output["xNewBounds"])   
        output["disconti"] = output["disconti"] * len(output["xNewBounds"]) 
    else: 
        output["xAlmostEqual"] = [consistent]
        output["xSolved"] = [xSolved]

    return output


def update_quantities(y, i, subBoxNo, xNewBounds, xBounds, xSolved,
                      bxrd_options, bxrd_options_temp):
    """ updates all quantities after box reductions
    
    Args:
        :y:                 list will currently reduced variable's intervals
        :i:                 id of currently reduced variable as int
        :subBoxNo:          current number of sub-boxes as int
        :xNewBounds:        list with already reduced variable bounds
        :xUnchanged:        list with boolean if box intervals still change
        :xSolved:           list with boolean if box intervals are solved
        :bxrd_options:      dictionary with box reduction settings
        :bxrd_options_temp: dictionary with modified box reduction setting 
                            for solved intervals that need to be further 
                            reduced to tighten other non-degenerate intervals
                            
    Returns:
        :xSolved:           updated list for current variable interval
        :xUnchanged:        updated list for current variable interval
        :subBoxNo:          updated number of sub-boxes

    """
    subBoxNo = subBoxNo * len(y)
    xNewBounds[i] = y
    if not variableSolved(y, bxrd_options): xSolved = False
    #if xUnchanged: xUnchanged = checkXforEquality(bxrd_options_temp["x_old"][i], 
    #                                              xBounds[i], xUnchanged, 
    #                               {"absTol":bxrd_options["absTol"], 
    #                                'relTol':bxrd_options["relTol"]})  
    
    bxrd_options_temp["relTol"] = bxrd_options["relTol"]
    bxrd_options_temp["absTol"] = bxrd_options["absTol"]  
    return xSolved, subBoxNo


def do_bnormal(f, xBounds, y, i, bxrd_options):
    """ excecutes box consistency method for a variable with global index i in
    function f and intersects is with its former interval. 
    
    Args:
        :f:             instance of type function
        :xBounds:       numpy array with currently reduced box
        :y:             currently reduced interval in mpmath.mpi formate
        :i:             global index of the current variable
        :bxrd_options:  dictionary with user-settings such as tolerances
    
    Returns:
        :y:             current interval after reduction in mpmath.mpi formate

    """
    box = [xBounds[j] for j in f.glb_ID]
    x_new = reduceXIntervalByFunction(box, f,f.glb_ID.index(i), bxrd_options)
    y = setOfIvSetIntersection([y, x_new])         
    return y


# def doHC4(model, xBounds, xNewBounds, output, bxrd_options):
#     """ excecutes HC4revise hull consistency method and returns output with
#     failure information in case of an empty box. Otherwise the initial output
#     dictionary is returned.
    
#     Args:
#         :model:         instance of type model
#         :xBounds:       numpy array with currently reduced box
#         :xNewBounds:    numpy array for reduced box
#         :output:        dictionary that stores information of current box reduction
#         :dict_otpions:  dictionary with tolerances
    
#     Returns:
#         :output:        unchanged dictionary (successful reduction) or dictionary
#                         with failure outpot (unsuccessful reduction)
#         :empty:         boolean, that is true for empty boxes

#     """
#     empty = False
#     HC4_IvV = HC4(model.functions, xBounds, bxrd_options)
#     if HC4_IvV.is_empty():
#         saveFailedSystem(output, bxrd_options["failed_function"], model, 
#                          bxrd_options["failed_function"].glb_ID[0])
#         empty = True
#     else:
#         for i in range(len(model.xSymbolic)):
#             HC4IV_mpmath = mpmath.mpi(HC4_IvV[i][0],(HC4_IvV[i][1]))
#             #else: HC4IV_mpmath = mpmath.mpi(HC4_IvV[i][0]-1e-17,(HC4_IvV[i][1])+1e-17)
#             y = ivIntersection(xBounds[i], HC4IV_mpmath)
#             if y == [] or y == [[]]:

#                 if (isclose_ordered(float(mpmath.mpf(xBounds[i].b)), 
#                                  HC4_IvV[i][0],0.0, bxrd_options["absTol"]) or 
#                     isclose_ordered(float(mpmath.mpf(xBounds[i].a)), 
#                                   HC4_IvV[i][1], 0.0, bxrd_options["absTol"])):    
#                     y = mpmath.mpi(min(xBounds[i].a,HC4IV_mpmath.a), 
#                                    max(xBounds[i].b,HC4IV_mpmath.b))                                      
#             xNewBounds[i] = y
#             if  xNewBounds[i]  == [] or  xNewBounds[i]  ==[[]]:                 
#                 if "failed_function" in bxrd_options.keys(): 
#                     saveFailedSystem(output, bxrd_options["failed_function"], 
#                                      model, bxrd_options["failed_function"].glb_ID[0])
#                     del bxrd_options['failed_function']
#                 else:
#                     saveFailedSystem(output, model.functions[0], model, 0)
#                 empty = True
#                 break  
            
#     return output, empty


def do_HC4(model, xBounds, output, bxrd_options):
    """ excecutes HC4revise hull consistency method and returns output with
    failure information in case of an empty box. Otherwise the initial output
    dictionary is returned.
    
    Args:
        :model:         instance of type model
        :xBounds:       numpy array with currently reduced box
        :output:        dictionary that stores information of current box reduction
        :dict_otpions:  dictionary with tolerances
    
    Returns:
        :output:        unchanged dictionary (successful reduction) or dictionary
                        with failure outpot (unsuccessful reduction)
        :empty:         boolean, that is true for empty boxes
        :box:           numpy.array with reduced box

    """
    empty = False
    consistent = False 
    box = [[convert_mpi_float(iv.a),convert_mpi_float(iv.b)] for iv in list(xBounds)]
    while not consistent:
        box_old = list(box)
        box = HC4_float(model.functions, box, bxrd_options)

        if box.is_empty():
            saveFailedSystem(output, bxrd_options["failed_function"], model, 
                             bxrd_options["failed_function"].glb_ID[0],
                              bxrd_options["failed_subbox"])
            empty = True
            break   
        elif bxrd_options["unique_hc"] and "FoundSolutions" in bxrd_options.keys():
            box_mpmath = numpy.array([mpmath.mpi(iv[0], iv[1]) for iv in box]) 
            if test_for_root_inclusion(box_mpmath, bxrd_options["FoundSolutions"], 
                                   bxrd_options["absTol"]):
                
                output["uniqueSolutionInBox"] = True
                return output, empty, box_mpmath   
        elif solved(box, bxrd_options):
            box_mpmath = numpy.array([mpmath.mpi(iv[0], iv[1]) for iv in box]) 
            output["uniqueSolutionInBox"] = True
            return output, empty, box_mpmath  
        elif bxrd_options["unique_hc"]: 
            output["box_has_unique_solution"] = True         
        
        consistent = check_consistency(box_old, box, bxrd_options)
 
    box = numpy.array([mpmath.mpi(iv[0], iv[1]) for iv in box])      
    return output, empty, box


def solved(box, bxrd_options):
    for iv in box:
        if not isclose_ordered(iv[0],iv[1],bxrd_options["relTol"], 
                           bxrd_options["absTol"]): return False
    return True

def check_consistency(box_old, box, bxrd_options):
    """ checks if two boxes are consistent
    
    Args:
        :box_old:          list with intervals in mpmath.mpi formate
        :box:              list with intervals in mpmath.mpi formate
        :bxrd_options:     dictionary with tolerances for equality criterion
        
    Returns:
        :True/False:    boolean that is True if boxes are consistent and False 
                        otherwise
                
    """    
    for i, x in enumerate(box):
        if not isclose_ordered(box_old[i][0], x[0], bxrd_options["relTol"], 
                           bxrd_options["absTol"]):
            return False
        elif not isclose_ordered(x[1], box_old[i][1], bxrd_options["relTol"], 
                           bxrd_options["absTol"]):
            return False
    return True
    
       
def checkIntervalAccuracy(xNewBounds, i, bxrd_options):
    """ checks the accuracy of the current box before reduction creates a nested 
    box for all intervals that are already degenerate. If some variable's 
    interval widths are degenerate in the current tolerance but they are not below
    1e-15 the tolerance for the width is stepwise decreased until only widths
    below 1e-15 are degenerate. This is sometimes necessary to decrease "solved"
    variables further on as other variables might be highly sensitive to their value
    and their interval would otherwise be not further reduced. Splitting would not
    help in this case as all sub-intervals of the sensitive variable would still be 
    solutions.
    
    Args:
    :xNewBounds:    numpy. array for new reduced intervals
    :i:             integer with current variable's global index
    :bxrd_options:  dictionary with current relative and absolute tolerance of 
                    variable
                    
    Returns:        False if not solved yet and true otherwise
    
    """  
    if xNewBounds[i].delta == 0: return True
    else:
        if isinstance(xNewBounds[i], mpmath.ctx_iv.ivmpf):
            iv = [convert_mpi_float(xNewBounds[i].a),convert_mpi_float(xNewBounds[i].b)]
        else: iv = xNewBounds[i]
        accurate = variableSolved([xNewBounds[i]], bxrd_options)
        notdegenerate = iv[1]-iv[0] > 1.0e-15
        if accurate and notdegenerate:
            bxrd_options["relTol"] = min(0.9*(iv[1] - iv[0])/
                                         (max(abs(iv[0]),abs(iv[1]))),
                                         0.1*bxrd_options["relTol"])
            bxrd_options["absTol"] = min(0.9 * abs(iv[1]- iv[0]), 
                                         0.1*bxrd_options["absTol"])   
                
        #if isinstance(bxrd_options["relTol"], mpmath.ctx_iv.ivmpf):
        #    bxrd_options["relTol"] = float(mpmath.mpf(bxrd_options["relTol"].a))
        #if isinstance(bxrd_options["absTol"], mpmath.ctx_iv.ivmpf):
        #    bxrd_options["absTol"] = float(mpmath.mpf(bxrd_options["absTol"].a))             
                
        return False


def variableSolved(BoundsList, bxrd_options):
    """ checks, if variable is solved in all Boxes in BoundsList
    Args:
        :BoundsList:      List of mpi Bounds for single variable
        :bxrd_options:    dictionary with tolerances for equality criterion
    
    Returns:
        :variableSolved:  boolean that is True if all variables have been solved
        
    """
    variableSolved = True
    for bound in BoundsList:
        if not checkVariableBound(bound, bxrd_options):
            variableSolved = False
    
    return variableSolved


def checkXforEquality(xBound, xNewBound, xUnchanged, bxrd_options):
    """ changes variable xUnchanged to false if new variable interval xNewBound
    is different from former interval xBound
    
    Args:
        :xBound:          interval in mpmath.mpi formate
        :xNewBound:       interval in mpmath.mpi formate
        :xUnchanged:      boolean
        :bxrd_options:    dictionary with tolerances for equality criterion
        
    Returns:
        :xUnchanged:     boolean that is True if interval has not changed
        
    """
    absEpsX = bxrd_options["absTol"]
    relEpsX = bxrd_options["relTol"]
    lb = isclose_ordered(float(mpmath.mpf(xNewBound.a)), 
                       float(mpmath.mpf(xBound.a)), relEpsX, absEpsX)
    ub = isclose_ordered(float(mpmath.mpf(xNewBound.b)), 
                       float(mpmath.mpf(xBound.b)), relEpsX, absEpsX)
        
    if not lb or not ub and xUnchanged: 
        xUnchanged = False   
    
    return xUnchanged
        
            
def assignIvsWithoutSplit(output, i, xUnchanged, xBounds, xNewBounds):
    """ assigns former varibale intervals to list of new intervals if maximum 
    number of boxes is reached.
    
    Args:
        :output:        dictionary with output data
        :i:             index of variable
        :xUnchanged:    boolean that is True as long as no variable interval 
                        could be reduced
        :xBounds:       numpy array with former bounds in mpmath.mpi formate
        :xNewBounds:    list with reduced variable bounds in mpmath.mpi formate
    
    """
    for resti in range(i, len(xBounds)): xNewBounds[resti] = [xBounds[resti]]
    output["xAlmostEqual"] = xUnchanged     
    output["xSolved"] = False
    output["xNewBounds"] = list(itertools.product(*xNewBounds))
                
                               
def saveFailedSystem(output, f, model, i, sub_box_failed=None):
    """ saves output of failed box reduction 
    
    Args:
        :output:        dictionary with output data
        :f:             instance of class Function
        :model:         instance of class Model
        :i:             index of variable
   
    """     
    output["xNewBounds"] = []
    failedSystem = FailedSystem(f.f_sym, model.xSymbolic[i],sub_box_failed)
    output["noSolution"] = failedSystem
    output["xAlmostEqual"] = [False] 
    output["xSolved"] = [False]

    
def checkVariableBound(newXInterval, bxrd_options):
    """ if lower and upper bound of a variable are almost equal the boolean 
    boundsAlmostEqual is set to true.

    Args:
        :newXInterval:      variable interval in mpmath.mpi logic
        :bxrd_options:      dictionary with tolerance limits
        
    Returns:                True, if lower and upper variable bound are almost
                            equal.

    """
    absEpsX = bxrd_options["absTol"]
    relEpsX = bxrd_options["relTol"]   
    iv = convertIntervalBoundsToFloatValues(newXInterval)

    if isclose_ordered(iv[0], iv[1], relEpsX, absEpsX):
        return True
    else: 
        return False

    
def reduceXIntervalByFunction(xBounds, f, i, bxrd_options):
    """ reduces variable interval by either solving a linear function directly
    with Gap-operator or finding the reduced variable interval(s) of a
    nonlinear function by interval nesting
     
    Args: 
        :xBounds:            one set of variable interavls as numpy array
        :f:                  instance of class Function
        :i:                  index for iterated variable interval
        :bxrd_options:       dictionary with solving settings

    Returns:                 list with new set of variable intervals
                        
    """       
    xUnchanged = True
    xBounds = list(xBounds)
    
    gxInterval, dgdxInterval, bInterval = calculateCurrentBounds(f, i, xBounds, 
                                                                 bxrd_options)
    if (gxInterval == [] or dgdxInterval == [] or 
        bInterval == [] or gxInterval in bInterval): 
        bxrd_options["unique_bc"] = False
        return [xBounds[i]]
    xUnchanged = checkXforEquality(gxInterval, dgdxInterval*xBounds[i], 
                                   xUnchanged, bxrd_options)
    # Linear Case -> solving system directly f.x_sym[i] in f.dgdx_sym[i].free_symbols:
    if f.deriv_is_constant[i] and xUnchanged : 
        x_new,si = getReducedIntervalOfLinearFunction(dgdxInterval, i, xBounds, 
                                                  bInterval)
        if si: x_new = getReducedIntervalOfNonlinearFunction(f, dgdxInterval, i, 
                                                          xBounds, bInterval, 
                                                          bxrd_options)            
             
    else: # Nonlinear Case -> solving system by interval nesting:
        x_new = getReducedIntervalOfNonlinearFunction(f, dgdxInterval, i, 
                                                     xBounds, bInterval, 
                                                     bxrd_options)
    # Unique solution check:    
    if bxrd_options["unique_bc"]:
        if len(x_new)== 1 and x_new != [[]]:
            unique = checkUniqueness(x_new, bxrd_options["x_old"][i],
                                     bxrd_options["relTol"],
                                     bxrd_options["absTol"])
            bxrd_options["unique_bc"] = unique
        else: 
            bxrd_options["unique_bc"] = False
            
    return x_new


def lambdifyToAffapyAffine(x, f):
    """Converting operations of symoblic equation system f (simpy) to
    affine arithmetic values (ref. to affapy.aa slightly modified in 
    affineArithmetic module to count for minimum range and Chebyshev's 
    approximation

    Args:
        :x:      set with symbolic variables in sympy formate
        :f:      list with symbolic functions in sympy formate
        
    Return:     lambdified symbolic function in affapy formate   

    """
    affapyhIv = {"exp" : affineArithmetic.Affine.exp,
            "sin" : affineArithmetic.Affine.sin,
            "sinh" : affineArithmetic.Affine.sinh,
            "cos" : affineArithmetic.Affine.cos,
            "cosh" : affineArithmetic.Affine.cosh,
            "tan" : affineArithmetic.Affine.tan,
            "tanh" : affineArithmetic.Affine.tanh,
            "log" : affineArithmetic.Affine.log,
            "sqrt": affineArithmetic.Affine.sqrt,
            "Pow": affineArithmetic.Affine.__pow__}
    
    return sympy.lambdify(x, f, affapyhIv)


def lambdifyToMpmathIv(x, f):
    """Converting operations of symoblic equation system f (simpy) to
    arithmetic interval functions (mpmath.iv), able to filter out complex 
    intervals
    
    Args:
        :x:      set with symbolic variables in sympy formate
        :f:      list with symbolic functions in sympy formate
        
    Return:     lambdified symbolic function in mpmath.mpi formate   

    """
    mpmathIv = {"exp" : mpmath.iv.exp,
            "sin" : mpmath.iv.sin,
            "cos" : mpmath.iv.cos,
            "acos": mpmath.iv.cos,
            "asin": mpmath.iv.sin,
            "atan": mpmath.iv.tan,
            "log" : mpmath.log,
            "sqrt": mpmath.ivsqrt}

    return sympy.lambdify(x, f, mpmathIv)


def lambdifyToMpmathIvComplex(x, f):
    """Converting operations of symoblic equation system f (simpy) to
    arithmetic interval functions (mpmath.iv)

    Args:
        :x:      set with symbolic variables in sympy formate
        :f:      list with symbolic functions in sympy formate
        
    Return:     lambdified symbolic function in mpmath.mpi formate   

    """

    mpmathIv = {"exp" : mpmath.iv.exp,
            "sin" : mpmath.iv.sin,
            "cos" : mpmath.iv.cos,
            "acos": ivacos,
            "asin": ivasin,
            "tan": ivtan,
            "log" : ivlog,
            "sqrt": ivsqrt,
            "Pow": modOpt.constraints.realIvPowerfunction.ivRealPower}

    return sympy.lambdify(x, f, mpmathIv)


def ivsqrt(iv):
    """calculates the square root of an interval iv, stripping it from the imaginary part"""

    if iv.a >= 0 and iv.b >= 0: return mpmath.iv.sqrt(iv)# sqrtiv = mpmath.mpi(mpmath.sqrt(iv.a), mpmath.sqrt(iv.b))
    elif iv.a < 0 and iv.b >= 0:return mpmath.iv.sqrt(mpmath.mpi(0.0, iv.b))
    else:
        # this case should not occur, the solution can not be in this interval
        return numpy.nan_to_num(numpy.inf) * mpmath.mpi(-1.0, 1.0)


def ivlog(iv):
    """calculates the ln root of an interval iv, stripping it from the imaginary part"""
    
    if iv.a > 0 and iv.b > 0: return mpmath.iv.log(iv) #mpmath.mpi(mpmath.log(iv.a),mpmath.log(iv.b))
    elif iv.a <= 0 and iv.b > 0: return mpmath.iv.log(mpmath.mpi(0.0, iv.b))#mpmath.mpi('-inf',mpmath.iv.log(iv.b))
    elif iv.a <= 0 and iv.b <= 0:
        #this case should not occur, the solution can not be in this interval
        #print('Negative ln! Solution can not be in this Interval!')
        return numpy.nan_to_num(numpy.inf) *  mpmath.mpi(-1.0, 1.0)


def ivacos(iv):
    """calculates the acos of an interval iv, stripping it from the imaginary part"""

    if iv.a>=-1 and iv.b<=1: return mpmath.mpi(mpmath.acos(iv.b),mpmath.acos(iv.a))
    elif iv.a<-1 and iv.b<=1 and iv.b>=-1: return mpmath.mpi(mpmath.acos(iv.b),mpmath.pi)
    elif iv.a>=-1 and iv.a<=1 and iv.b>1: return mpmath.mpi(0, mpmath.acos(iv.a))
    else: return mpmath.mpi(0, mpmath.pi)


def ivasin(iv):
    """calculates the asin of an interval iv, stripping it from the imaginary part"""

    if iv.a>=-1 and iv.b<=1: return mpmath.mpi(mpmath.asin(iv.a),mpmath.asin(iv.b))
    elif iv.a<-1 and iv.b<=1 and iv.b>=-1: return mpmath.mpi(mpmath.asin(-1),mpmath.asin(iv.b))
    elif iv.a>=-1 and iv.a<=1 and iv.b>1: return mpmath.mpi(mpmath.asin(iv.a),mpmath.asin(1))
    else: return mpmath.mpi(mpmath.asin(-1), mpmath.asin(1))

        
def ivtan(iv):
    """calculates the atan of an interval iv, stripping it from the imaginary part"""

    if iv.a>-mpmath.pi/2 and iv.b<mpmath.pi/2:
        return mpmath.mpi(mpmath.tan(iv.a),mpmath.tan(iv.b))
    elif iv.a<=-mpmath.pi/2 and iv.b<mpmath.pi/2 and iv.b>-mpmath.pi/2:
        return mpmath.mpi(numpy.nan_to_num(-numpy.inf),mpmath.tan(iv.b))
    elif iv.a>-mpmath.pi/2 and iv.a<mpmath.pi/2 and iv.b>=mpmath.pi/2:
        return mpmath.mpi(mpmath.tan(iv.a),numpy.nan_to_num(numpy.inf))
    else: return numpy.nan_to_num(numpy.inf) * mpmath.mpi(-1.0, 1.0)


def calculateCurrentBounds(f, i, xBounds, bxrd_options):
    """ calculates bounds of function gx, the residual b, first derrivative of 
    function gx with respect to variable x (dgdx).
    
    Args:
        :f:                  instance of class Function
        :i:                  index of current variable 
        :xBounds:            numpy array with variable bounds
        :bxrd_options:       dictionary with entries about stop-tolerances
       
    Returns:
        :bInterval:          residual interval in mpmath.mpi logic
        :gxInterval:         function interval in mpmath.mpi logic
        :dfdxInterval:       Interval of first derrivative in mpmath.mpi logic 
    
    """
    #bxrd_options["tightBounds"] = True
    try:
        if bxrd_options["affineArithmetic"]:
            bInterval = eval_fInterval(f, f.b_mpmath[i], xBounds, f.b_aff[i],
                                       bxrd_options["tightBounds"],
                                       bxrd_options["resolution"])
        else: 
            bInterval = eval_fInterval(f, f.b_mpmath[i], xBounds, False, 
                                       bxrd_options["tightBounds"],
                                       bxrd_options["resolution"])

    except: return [], [], []
    try:
        if bxrd_options["affineArithmetic"]:
            gxInterval = eval_fInterval(f, f.g_mpmath[i], xBounds, f.g_aff[i],
                                        bxrd_options["tightBounds"],
                                        bxrd_options["resolution"])
        else: 
            gxInterval = eval_fInterval(f, f.g_mpmath[i], xBounds, False, 
                                        bxrd_options["tightBounds"],
                                        bxrd_options["resolution"])

    except: return [], [], []
    try:
        if bxrd_options["affineArithmetic"]:
            dgdxInterval = eval_fInterval(f, f.dgdx_mpmath[i], xBounds, f.dgdx_aff[i],
                                          bxrd_options["tightBounds"],
                                          bxrd_options["resolution"])
        else: 
            dgdxInterval = eval_fInterval(f, f.dgdx_mpmath[i], xBounds, False,
                                          bxrd_options["tightBounds"],
                                          bxrd_options["resolution"])
    except: return [], [], []
   
    return gxInterval, dgdxInterval, bInterval  
   
         
def eval_function_tight_mpmath(f, fInterval, f_mpmath, f_sym, box, resolution):
    """ evaluate mpmath function with discretized intervals of box to reduce
    interval dependency issues
    
    Args:
        :f:             object of class Function
        :fInterval:     function interval from classical interval arithmetic
        :f_mpmath:      mpmath function
        :f_sym:         symbolic function in sympy logic
        :box:           list or numpy.array with variable intervals
        :resoltion:     integer with number of refinements

    Return:
        :fInterval:     tightened function interval in mpmath.mpi formate

    """
    y_sym = list(f_sym.free_symbols)
    if not resolution: resolution = 8
    for j, var in enumerate(y_sym):
        k = f.x_sym.index(var)
        #f_sym.count(var)
        if f.var_count[k] > 1 and box[k].delta > 1.0e-15: 
            #print("Before: ", fInterval)
            iv = convertIntervalBoundsToFloatValues(box[k])
            ivs = numpy.linspace(iv[0], iv[1], int(resolution)+1)
            f_low, f_up = getFunctionValuesIntervalsOfXList(ivs, f_mpmath, 
                                                            list(box), k)
            f_new = [min(f_low), max(f_up)]
            fInterval = mpmath.mpi(max(fInterval.a, f_new[0]), 
                                   min(fInterval.b, f_new[1]))
        else: continue
    
    return fInterval            
            
    
def eval_fInterval(f, f_mpmath, box, f_aff=None, f_tight=None, resolution=None):
    """ evaluate mpmath function f_mpmath in list box with mpmath.mpi intervals
    
    Args:
        :f:             object of class Function
        :f_mpmath:      mpmath function
        :box:           list or numpy.array with variable intervals
        :f_aff:         boolean that is true if affine arithmetic shall be used
        :f_tight:       boolean that is true if tight arithmetic shall be used 
    
    Return:
        :fInterval:     function interval in mpmath.mpi formate

    """
    fInterval = f.eval_mpmath_function(box, f_mpmath)
    if f_aff:
        try: 
            newIv = ivIntersection(fInterval, f.eval_aff_function(box, f_aff))
            if newIv != []: fInterval = newIv
        except: pass
    
    if f_tight:
        try: 
            newIv = eval_function_tight_mpmath(f, fInterval, f_mpmath, 
                                               f.f_sym, box, resolution)  
            if newIv != []: fInterval = newIv
        except: pass      
        
    return fInterval
                
                
def checkIntervalWidth(intervals, absEpsX, relEpsX):
    """ checks if width of intervals is smaller than a given absolute and 
    relative tolerance by numpy.isclose(a,b, relTol, absTol) being True for 
    abs(a-b)<=absTol + abs(b)*relTol

    Args:
        :intervals:           set of intervals in mpmath.mpi-logic
        :absEpsX:            absolute x tolerance
        :relEpsX:            relative x tolerance

    Return: list of intervals with a higher width than absEpsX and relEpsX
    
    """
    return [iv for iv in intervals if not (isclose_ordered(float(mpmath.mpf(iv.a)), 
                            float(mpmath.mpf(iv.b)), relEpsX, absEpsX))]


def getReducedIntervalOfLinearFunction(a, i, xBounds, bi):
    """ returns reduced interval of variable X if f is linear in X. The equation
    is solved directly by the use of the hansenSenguptaOperator.
    
    Args: 
        :a:                  mpmath.mpi interval
        :i:                  integer with current iteration variable index
        :xBounds:            numpy array with set of variable bounds
        :bi:                 current function residual bounds

    Return:                  reduced x-Interval(s)   
    
    """        
    # If this is the case, there is no solution in xBounds
    if bool(0 in mpmath.mpi(bi) - a * xBounds[i]) == False: return [], False
    # If this is the case, bi/aInterval would return [-inf, +inf]. 
    # Hence the approximation of x is already smaller
    if bool(0 in mpmath.mpi(bi)) and bool(0 in mpmath.mpi(a)): return [xBounds[i]], True
    else: 
        return hansenSenguptaOperator(mpmath.mpi(a), mpmath.mpi(bi), xBounds[i]),False # bi/aInterval  


# def checkAndRemoveComplexPart(interval):
#     """ creates a warning if a complex interval occurs and keeps only the real
#     part.

#     """
#     if interval.imag != 0:
#         print("Warning: A complex interval: ", interval.imag," occured.\n",
#         "For further calculations only the real part: ", interval.real, " is used.")
#         interval = interval.real


def hansenSenguptaOperator(a, b, x):
    """ Computation of the Gauss-Seidel-Operator [1] to get interval for x
    for given intervals for a and b from the 1-dimensional linear system:

                                    a * x = b
        Args:
            :a:     interval of mpi format from mpmath library
            :b:     interval of mpi format from mpmath library
            :x:     interval of mpi format from mpmath library
                    (initially guessed interval of x)

        Returns:
            :interval:   interval(s) of mpi format from mpmath library where
                         solution for x can be in, if interval remains [] there
                         is no solution within the initially guessed interval of x

    """
    interval = []
    u = ivDivision(b, a)

    for cu in u:
        intersection = ivIntersection(cu, x)
        if intersection !=[]: interval.append(intersection)

    return interval


def ivDivision(i1, i2):
    """ calculates the result of the divion of two intervals i1, i2: i1 / i2

    Args:
        :i1:     interval of mpi format from mpmath library
        :i2:     interval of mpi format from mpmath library

    Returns:
        :mpmath.mpi(a,b):    resulting interval of division [a,b],
                             this is empty if i2 =[0,0] and returns []

    """
    if bool(0 in i2)== False: return [i1 * mpmath.mpi(1/i2.b, 1/i2.a)]
    if bool(0 in i1) and bool(0 in i2): return [i1 / i2]
    if i1.b < 0 and i2.a != i2.b and i2.b == 0: 
        return [mpmath.mpi(i1.b / i2.a, i1.a / i2.b)]
    if i1.b < 0 and i2.a < 0 and i2.b > 0: 
        return [mpmath.mpi(-numpy.nan_to_num(numpy.inf), 
                           max(i1.b / i2.b, mpmath.mpi(-numpy.nan_to_num(numpy.inf)))), 
                mpmath.mpi(min(i1.b / i2.a,  numpy.nan_to_num(numpy.inf)), 
                           numpy.nan_to_num(numpy.inf))]
    if i1.b < 0 and i2.a == 0 and i2.b > 0: 
        return [mpmath.mpi(-numpy.nan_to_num(numpy.inf), 
                           max(i1.b / i2.b, mpmath.mpi(-numpy.nan_to_num(numpy.inf))))]
    if i1.a > 0 and i2.a < 0 and i2.b == 0: 
        return [mpmath.mpi(-numpy.nan_to_num(numpy.inf), 
                           max(i1.a / i2.a, mpmath.mpi(-numpy.nan_to_num(numpy.inf))))]
    if i1.a > 0 and i2.a < 0 and i2.b > 0: 
        return [mpmath.mpi(-numpy.nan_to_num(numpy.inf), 
                           max(i1.a / i2.a, mpmath.mpi(-numpy.nan_to_num(numpy.inf)))), 
                mpmath.mpi(min(i1.a / i2.b, mpmath.mpi(numpy.nan_to_num(numpy.inf))), 
                           numpy.nan_to_num(numpy.inf))]
    if i1.a > 0 and i2.a == 0 and i2.b > 0: 
        return [mpmath.mpi(min(i1.a / i2.b, mpmath.mpi(numpy.nan_to_num(numpy.inf))),
                           numpy.nan_to_num(numpy.inf))]

    if bool(0 in i1) == False and i2.a == 0 and i2.b == 0: return []


def ivIntersection(i1, i2):
    """ returns intersection of two intervals i1 and i2

    Args:
        :i1:     interval of mpi format from mpmath library
        :i2:     interval of mpi format from mpmath library

    Returns:
        :mpmath.mpi(a,b):    interval of intersection [a,b],
                             if empty [] is returned

    """
    if i1.a <= i2.a and i1.b <= i2.b and i2.a <= i1.b: 
        return mpmath.mpi(i2.a, i1.b)
    if i1.a <= i2.a and i1.b >= i2.b: return i2
    if i1.a >= i2.a and i1.b <= i2.b: return i1
    if i1.a >= i2.a and i1.b >= i2.b and i1.a <= i2.b: 
        return mpmath.mpi(i1.a, i2.b)
    else: return []
    
    
def check_capacities(nested_interval_list, f, b, i, box, bxrd_options):
    """ checks if there is enough space on the stack for any further splitting
    due to removing a discontinuity
    
    Args:
        :nested_interval_list:      list with intervals of currently reduced x
        :f:                         object of class Function
        :b:                         constant interval b in mpmath.mpi formate
        :i:                         index of currently reduced variable as int
        :box:                       list of f's variable intervals
        :bxrd_options:              dictionary with box reduction settings
    
    Returns:
        :boolean:                   True if there is capacity for the gap and 
                                    False otherwise
        :nested_interval_list:      cleaned list of possible intervals that have
                                    an intersection with b (are not empty)
        
    """  
    ivNo = 0
    for iv_list in nested_interval_list:    
        for iv in iv_list:
            box[i] = iv
            if not ivIntersection(f.g_mpmath[i](*box), b): iv_list.remove(iv)
        ivNo += len(iv_list)
    if ivNo <= (bxrd_options["maxBoxNo"] - bxrd_options["boxNo"])+1: 
        return True, nested_interval_list    
    else: return False, nested_interval_list


def getReducedIntervalOfNonlinearFunction(f, dgdXInterval, i, xBounds, bi, bxrd_options):
    """ checks function for monotone sections in x and reduces them one after the other.

    Args:
        :f:                  object of class Function
        :dgdXInterval:       first derivative of function f with respect to x at xBounds
        :i:                  integer with current iteration variable index
        :xBounds:            numpy array with set of variable bounds
        :bi:                 current function residual bounds
        :bxrd_options:       for function and variable interval tolerances in the used
                             algorithms     

    Return:                reduced x-Interval(s) and list of monotone x-intervals

    """
    relEpsX = bxrd_options["relTol"]
    absEpsX = bxrd_options["absTol"]
    increasingZones = []
    decreasingZones = []
    nonMonotoneZones = []
    reducedIntervals = []
    orgXiBounds = [xBounds[i]]
    curXiBounds = orgXiBounds
    
    if dgdXInterval == []: return []
      
    if (numpy.nan_to_num(-numpy.inf) in dgdXInterval or 
        numpy.nan_to_num(numpy.inf) in dgdXInterval) : # condition for discontinuities
        curXiBounds, nonMonotoneZones = getContinuousFunctionSections(f, i, 
                                                                      xBounds, 
                                                                      bxrd_options)
        if len(curXiBounds) + len(nonMonotoneZones) > 1:
            cap, intervals = check_capacities([curXiBounds, nonMonotoneZones], 
                                              f, bi, i, xBounds, bxrd_options)
            if cap: curXiBounds, nonMonotoneZones = intervals           
            else:
                if bxrd_options["considerDisconti"]: bxrd_options["disconti_iv"] = True
                return orgXiBounds
           
    if curXiBounds != []:
        for curInterval in curXiBounds:
            
            xBounds[i] = curInterval
            iz, dz, nmz = getMonotoneFunctionSections(f, i, xBounds, bxrd_options)
            if iz != []:  increasingZones += iz
            if dz != []:  decreasingZones += dz
            if nmz !=[]: nonMonotoneZones += nmz 

        if len(nonMonotoneZones)>1: 
           nonMonotoneZones = joinIntervalSet(nonMonotoneZones, relEpsX, absEpsX)
              
    if increasingZones !=[]:
            reducedIntervals = reduceMonotoneIntervals(increasingZones, 
                                                       reducedIntervals, f,
                                                       xBounds, i, bi, 
                                                       bxrd_options, 
                                                       increasing = True)
    if decreasingZones !=[]:               
            reducedIntervals = reduceMonotoneIntervals(decreasingZones, 
                                                       reducedIntervals, f, 
                                                       xBounds, i, bi, 
                                                       bxrd_options, 
                                                       increasing = False)  
    if nonMonotoneZones !=[]:
        reducedIntervals = reduceNonMonotoneIntervals({"0":nonMonotoneZones, 
                                   "1": reducedIntervals, 
                                   "2": f, 
                                   "3": i, 
                                   "4": xBounds, 
                                   "5": bi, 
                                   "6": bxrd_options})

        if reducedIntervals == False: 
            print("Warning: Reduction in non-monotone Interval took too long.")
            return orgXiBounds
    #reducedIntervals = joinIntervalSet(reducedIntervals, relEpsX, absEpsX)
    reducedIntervals = setOfIvSetIntersection([reducedIntervals, orgXiBounds])
    return reducedIntervals


def getContinuousFunctionSections(f, i, xBounds, bxrd_options):
    """filters out discontinuities which either have a +/- inf derrivative.

    Args:
        :f:                   object of type Function
        :i:                   index of differential variable
        :xBounds:             numpy array with variable bounds
        :bxrd_options:        dictionary with variable and function interval tolerances

    Return:
        :continuousZone:      list with continuous sections
        :discontiZone:        list with discontinuous sections
    
    """
    maxIvNo = bxrd_options["maxBoxNo"]
    absEpsX = bxrd_options["absTol"]
    relEpsX = bxrd_options["relTol"]
    if checkIntervalWidth([xBounds[i]], absEpsX, 0.1*relEpsX) == []: 
        return [],[]
    
    continuousZone = []
    orgXiBounds = xBounds[i]
    interval = [xBounds[i]]
    
    while interval != [] and len(interval) <= maxIvNo:
        discontinuousZone = []

        for curInterval in interval:
            newContinuousZone = testIntervalOnContinuity(f, curInterval, xBounds, 
                                                         i, discontinuousZone)
            if newContinuousZone == False: return (continuousZone, 
                                                   joinIntervalSet(interval, 
                                                                   relEpsX, absEpsX))
            elif len(newContinuousZone)>1: continuousZone += newContinuousZone
            else: continuousZone = addIntervaltoZone(newContinuousZone, 
                                                     continuousZone, bxrd_options)  
               
        interval = checkIntervalWidth(discontinuousZone, absEpsX, 0.1*relEpsX)
        if not len(interval) <= maxIvNo: return (continuousZone, 
                                                 joinIntervalSet(interval, 
                                                                 relEpsX, absEpsX))
    if interval == [] and continuousZone == []: return [], [orgXiBounds]

    return continuousZone, []


# def removeListInList(listInList):
#     """changes list with the shape: [[a], [b,c], [d], ...] to [a, b, c, d, ...]

#     """
#     return [value for sublist in listInList for value in sublist]


def reduceMonotoneIntervals(monotoneZone, reducedIntervals, f,
                                      xBounds, i, bi, bxrd_options, increasing):
    """ reduces interval sets of one variable by interval nesting

    Args:
        :monotoneZone        list with monotone increasing or decreasing set of intervals
        :reducedIntervals    list with already reduced set of intervals
        :fx:                 symbolic x-depending part of function f
        :xBounds:            numpy array with set of variable bounds
        :i:                  integer with current iteration variable index
        :bi:                 current function residual bounds
        :bxrd_options:       dictionary with function and variable interval tolerances
        :increasing:         boolean, True for increasing function intervals,
                             False for decreasing intervals
    
    Returns:
        :reducedIntervals:  list with reduced intervals
        
    """  
    old_x = list(xBounds)
    for curMonZone in monotoneZone: #TODO: Parallelizing
        xBounds[i] = curMonZone
        dgdXInterval = eval_fInterval(f, f.dgdx_mpmath[i], xBounds)
        non_mon = (dgdXInterval.a < 0 and dgdXInterval.b > 0 
                   and old_x[i] == curMonZone)
        if increasing and non_mon:
            curReducedInterval = bisect_mon_inc_interval(f, xBounds, i, bi, bxrd_options)
        

        elif increasing and not non_mon: 
            curReducedInterval = reduce_mon_inc_newton(f, xBounds, i, 
                                                                  bi, bxrd_options)
        else:
            if non_mon:
                curReducedInterval = bisect_mon_dec_interval(f, xBounds, i, bi, bxrd_options)  
            else:    
                curReducedInterval = reduce_mon_dec_newton(f, xBounds, i, bi, 
                                                         bxrd_options)
        if curReducedInterval !=[] and reducedIntervals != []:
            reducedIntervals.append(curReducedInterval)
            reducedIntervals = joinIntervalSet(reducedIntervals, 
                                               bxrd_options["relTol"], 
                                               bxrd_options["absTol"])
        elif curReducedInterval !=[]: reducedIntervals.append(curReducedInterval)

    return reducedIntervals


def bisect_mon_dec_interval(f, xBounds, i, bi, bxrd_options):
    tb = bxrd_options["tightBounds"]
    reso = bxrd_options["resolution"]

    fxInterval = eval_fInterval(f, f.g_mpmath[i], xBounds, f.g_aff[i], tb, reso)
    # first check if xBounds can be further reduced:
    if fxInterval in bi: return xBounds[i]
    if ivIntersection(fxInterval, bi)==[]: return []
    x_old = list(xBounds)
    xBounds[i] = xBounds[i].a
    xBounds_1 = list(xBounds)
    cur_iv = x_old[i]
    
    # Iteration of lower bound:
    if not 0.0 in eval_fInterval(f, f.g_mpmath[i], xBounds, f.g_aff[i], tb, reso)-bi:
        
        while(cur_iv.delta > bxrd_options["absTol"]):
            xBounds[i]= cur_iv.a
            xBounds_1[i]= cur_iv.mid
            if (not 0.0 in mpmath.mpi(eval_fInterval(f, f.g_mpmath[i], 
                                                     xBounds_1, f.g_aff[i], 
                                                     tb, reso).a-bi.b,
                                      eval_fInterval(f, f.g_mpmath[i], 
                                                     xBounds, f.g_aff[i], 
                                                     tb, reso).a-bi.b)):
                cur_iv=mpmath.mpi(cur_iv.mid, cur_iv.b)
            else: 
                cur_iv=mpmath.mpi(cur_iv.a, cur_iv.mid)
    x_low = cur_iv.a
    xBounds[i] = x_old[i].b

    cur_iv = x_old[i]
    
    # Iteration of upper bound:
    if not 0.0 in eval_fInterval(f, f.g_mpmath[i], xBounds, f.g_aff[i], tb, reso)-bi:
        while(cur_iv.delta > bxrd_options["absTol"]):
            xBounds[i]= cur_iv.a
            xBounds_1[i]= cur_iv.mid
            if (not 0.0 in mpmath.mpi(eval_fInterval(f, f.g_mpmath[i], 
                                                     xBounds_1, f.g_aff[i], 
                                                     tb, reso).b-bi.a,
                                      eval_fInterval(f, f.g_mpmath[i], 
                                                     xBounds, f.g_aff[i], 
                                                     tb, reso).b-bi.a)):
                cur_iv=mpmath.mpi(cur_iv.mid, cur_iv.b)
            else: 
                cur_iv=mpmath.mpi(cur_iv.a, cur_iv.mid)
    xBounds[i] = x_old[i]
    return mpmath.mpi(x_low.a, cur_iv.b)


def bisect_mon_inc_interval(f, xBounds, i, bi, bxrd_options):
    tb = bxrd_options["tightBounds"]
    reso = bxrd_options["resolution"]

    fxInterval = eval_fInterval(f, f.g_mpmath[i], xBounds, f.g_aff[i], tb, reso)
    # first check if xBounds can be further reduced:
    if fxInterval in bi: return xBounds[i]
    if ivIntersection(fxInterval, bi)==[]: return []
    x_old = list(xBounds)
    xBounds[i] = xBounds[i].a
    xBounds_1 = list(xBounds)
    cur_iv = x_old[i]
    
    # Iteration of lower bound:
    if not 0.0 in eval_fInterval(f, f.g_mpmath[i], xBounds, f.g_aff[i], tb, reso)-bi:
        
        while(cur_iv.delta > bxrd_options["absTol"]):
            xBounds[i]= cur_iv.a
            xBounds_1[i]= cur_iv.mid
            if (not 0.0 in mpmath.mpi(eval_fInterval(f, f.g_mpmath[i], 
                                                     xBounds, f.g_aff[i], 
                                                     tb, reso).b-bi.a,
                                      eval_fInterval(f, f.g_mpmath[i], 
                                                     xBounds_1, f.g_aff[i], 
                                                     tb, reso).b-bi.a)):
                cur_iv=mpmath.mpi(cur_iv.mid, cur_iv.b)
            else: 
                cur_iv=mpmath.mpi(cur_iv.a, cur_iv.mid)
    x_low = cur_iv.a
    xBounds[i] = x_old[i].b

    cur_iv = x_old[i]
    
    # Iteration of upper bound:
    if not 0.0 in eval_fInterval(f, f.g_mpmath[i], xBounds, f.g_aff[i], tb, reso)-bi:
        while(cur_iv.delta > bxrd_options["absTol"]):
            xBounds[i]= cur_iv.a
            xBounds_1[i]= cur_iv.mid
            if (not 0.0 in mpmath.mpi(eval_fInterval(f, f.g_mpmath[i], 
                                                     xBounds, f.g_aff[i], 
                                                     tb, reso).a-bi.b,
                                      eval_fInterval(f, f.g_mpmath[i], 
                                                     xBounds_1, f.g_aff[i], 
                                                     tb, reso).a-bi.b)):
                cur_iv=mpmath.mpi(cur_iv.mid, cur_iv.b)
            else: 
                cur_iv=mpmath.mpi(cur_iv.a, cur_iv.mid)
    xBounds[i] = x_old[i]
    return mpmath.mpi(x_low.a, cur_iv.b)
   


def reduce_mon_inc_newton(f, xBounds, i, bi, bxrd_options):
    """ Specific Interval-Newton method to reduce intervals in b_normal method
    
    Args:
        :f:             object of class Function
        :xBounds:       currently reduced box as list or numpy.array
        :bi:            constant interval for reduction in mpmath.mpi formate
        :bxrd_options:  dictionary with box reduction algorithm settings

    Returns:
        reduced interval in mpmath.mpi formate
        
    """
    tb = bxrd_options["tightBounds"]
    reso = bxrd_options["resolution"]
    nosuccess = False
    fxInterval = eval_fInterval(f, f.g_mpmath[i], xBounds, f.g_aff[i], tb, reso)
    # first check if xBounds can be further reduced:
    if fxInterval in bi: return xBounds[i]
    if ivIntersection(fxInterval, bi)==[]: return []
    x_old = list(xBounds)
    
    # Otherwise, iterate each bound of bi:
    relEpsX = bxrd_options["relTol"]
    absEpsX = bxrd_options["absTol"]
    curInterval = xBounds[i]    
    fIntervalxLow, fIntervalxUp = getFIntervalsFromXBounds(f, curInterval, 
                                                           xBounds, i)
    if fIntervalxLow.b < bi.a:
        while (not check_bound_and_interval_accuracy(
                convertIntervalBoundsToFloatValues(curInterval),
                [convert_mpi_float(fxInterval.b), convert_mpi_float(bi.a)],
                relEpsX, absEpsX)):

            x = curInterval.mid #+ curInterval.delta/2.0
            xBounds[i] = mpmath.mpi(x)
            fxInterval = eval_fInterval(f, f.g_mpmath[i], xBounds, 
                                        f.g_aff[i], tb, reso).b - bi.a
            xBounds[i] = curInterval
            quotient = ivDivision(fxInterval, 
                                         eval_fInterval(f, f.dgdx_mpmath[i], 
                                                        xBounds))
            if len(quotient)==1: 
                newInterval = x - quotient[0]
                #intersection = ivIntersection(curInterval, newInterval)
            else: 
                newInterval = x - mpmath.mpi([min(
                    [float(mpmath.mpf(element.a)) for element in quotient]),
                    max([float(mpmath.mpf(element.b)) for element in quotient])])
            intersection = ivIntersection(curInterval, newInterval)
            if intersection == curInterval: break
            if intersection == []: 
                curInterval = check_accuracy_newton_step(curInterval, 
                                                         newInterval, 
                                                         bxrd_options)
                if not curInterval: nosuccess = True
                break
            else: curInterval = intersection
            
    if (not curInterval or curInterval.a > x_old[i].b or 
        curInterval.b < x_old[i].a): 
        x_low = x_old[i].a
    else: 
        x_low = max(curInterval.a, x_old[i].a) 
    curInterval = x_old[i]
    
    fxInterval = eval_fInterval(f, f.g_mpmath[i], x_old, f.g_aff[i], tb, reso)
    
    if fIntervalxUp.a > bi.b:
        while (not check_bound_and_interval_accuracy(
                convertIntervalBoundsToFloatValues(curInterval),
                [convert_mpi_float(bi.b), convert_mpi_float(fxInterval.a)],
                relEpsX, absEpsX)):               

            x = curInterval.mid
            xBounds[i] = mpmath.mpi(x)
            fxInterval = eval_fInterval(f, f.g_mpmath[i], xBounds, 
                                        f.g_aff[i]).a - bi.b
            xBounds[i] = curInterval
            #if not const_deriv:
            quotient = ivDivision(fxInterval, 
                                         eval_fInterval(f, f.dgdx_mpmath[i], 
                                                        xBounds))   
            if len(quotient)==1: 
                newInterval = x - quotient[0]
                #intersection = ivIntersection(curInterval, newInterval)
            else: 
                newInterval = x - mpmath.mpi([min(
                    [float(mpmath.mpf(element.a)) for element in quotient]),
                    max([float(mpmath.mpf(element.b)) for element in quotient])])
            intersection = ivIntersection(curInterval, newInterval)
            if intersection == curInterval: break
            if intersection == []: 
                curInterval = check_accuracy_newton_step(curInterval, 
                                                         newInterval, 
                                                         bxrd_options)
                break
            else: curInterval = intersection
      
    if (not curInterval or curInterval.a > x_old[i].b or 
        curInterval.b < x_old[i].a):
        if nosuccess: return []
        else: return mpmath.mpi(x_low, x_old[i].b)
    else: 
        return mpmath.mpi(x_low, min(curInterval.b, x_old[i].b))


def reduce_mon_dec_newton(f, xBounds, i, bi, bxrd_options):
    """ Specific Interval-Newton method to reduce intervals in b_normal method
    
    Args:
        :f:             object of class Function
        :xBounds:       currently reduced box as list or numpy.array
        :bi:            constant interval for reduction in mpmath.mpi formate
        :bxrd_options:  dictionary with box reduction algorithm settings

    Returns:
        reduced interval in mpmath.mpi formate
        
    """
    tb = bxrd_options["tightBounds"]
    reso = bxrd_options["resolution"]  
    nosuccess = False
    fxInterval = eval_fInterval(f, f.g_mpmath[i], xBounds, f.g_aff[i], tb, reso)
    # first check if xBounds can be further reduced:
    if fxInterval in bi: return xBounds[i]
    if ivIntersection(fxInterval, bi)==[]: return []
    x_old = list(xBounds)
    # Otherwise, iterate each bound of bi:
    relEpsX = bxrd_options["relTol"]
    absEpsX = bxrd_options["absTol"]
    curInterval = xBounds[i]    
    fIntervalxLow, fIntervalxUp = getFIntervalsFromXBounds(f, curInterval, 
                                                           xBounds, i)
    if fIntervalxLow.a > bi.b:
        while (not check_bound_and_interval_accuracy(
                convertIntervalBoundsToFloatValues(curInterval),
                [convert_mpi_float(bi.b), convert_mpi_float(fxInterval.a)],
                relEpsX, absEpsX)):        
        
            x = curInterval.mid# + curInterval.delta/2.0
            xBounds[i]=mpmath.mpi(x)
            fxInterval = eval_fInterval(f, f.g_mpmath[i], xBounds, 
                                        f.g_aff[i], tb, reso).a - bi.b
            xBounds[i] = curInterval
            quotient = ivDivision(fxInterval, 
                                         eval_fInterval(f, f.dgdx_mpmath[i], 
                                                        xBounds))  
            if len(quotient)==1: 
                newInterval = x - quotient[0]
                #intersection = ivIntersection(curInterval, newInterval)
            else: 
                newInterval = x - mpmath.mpi([min(
                    [float(mpmath.mpf(element.a)) for element in quotient]),
                    max([float(mpmath.mpf(element.b)) for element in quotient])])
            intersection = ivIntersection(curInterval, newInterval)
            if intersection == curInterval: break
            if intersection == []: 
                curInterval = check_accuracy_newton_step(curInterval, 
                                                         newInterval, 
                                                         bxrd_options)
                if not curInterval: nosuccess = True
                break
            else: curInterval = intersection
                   
    if (curInterval == [] or curInterval.a > x_old[i].b or 
        curInterval.b < x_old[i].a): 
        x_low = x_old[i].a
    else:
        x_low = max(curInterval.a, x_old[i].a)   
    curInterval = x_old[i]
    fxInterval = eval_fInterval(f, f.g_mpmath[i], x_old, f.g_aff[i], tb, reso)
    
    if fIntervalxUp.b < bi.a:
        while (not check_bound_and_interval_accuracy(
                convertIntervalBoundsToFloatValues(curInterval),
                [convert_mpi_float(fxInterval.b), convert_mpi_float(bi.a)],
                relEpsX, absEpsX)):

            x = curInterval.mid#b - curInterval.delta/2.0
            xBounds[i]=mpmath.mpi(x)
            fxInterval = eval_fInterval(f, f.g_mpmath[i], xBounds, 
                                        f.g_aff[i], tb, reso).b - bi.a
            xBounds[i] = curInterval
            quotient = ivDivision(fxInterval, 
                                         eval_fInterval(f, f.dgdx_mpmath[i], 
                                                        xBounds))  
            if len(quotient)==1: 
                newInterval = x - quotient[0]
                #intersection = ivIntersection(curInterval, newInterval)
            else: 
                newInterval = x - mpmath.mpi([min(
                    [float(mpmath.mpf(element.a)) for element in quotient]),
                    max([float(mpmath.mpf(element.b)) for element in quotient])])
            intersection = ivIntersection(curInterval, newInterval)
            if intersection == curInterval: break
            if intersection == []: 
                curInterval = check_accuracy_newton_step(curInterval, 
                                                         newInterval, 
                                                         bxrd_options)
                break
            else: curInterval = intersection
            
    if (curInterval == [] or curInterval.a > x_old[i].b or 
        curInterval.b < x_old[i].a):
        if nosuccess: return []
        else: return mpmath.mpi(x_low, x_old[i].b)    
    else: return mpmath.mpi(x_low, min(curInterval.b, x_old[i].b))
   
    
def convert_mpi_float(mpi):
    """ converts single mpi value to float value
    
    Args:   mpmath.mpi value such as x.a or x.b
    
    Returns: corresponding float value

    """
    return float(mpmath.mpf(mpi))

    
def check_bound_and_interval_accuracy(x, val, relEpsX, absEpsX):
    """ checks if two bounds in bnormal are close to each other or the
    corresponding interval x is degenerate with respect to the tolerances
    
    Args:
        :x:         list with interval float values
        :val:       list with float values
        :relEpsX:   relative tolerance
        :absEpsX:   absolute tolerance
        
    Returns:
        :True/False: True if x or difference between values is close and False
                     otherwise

    """
    if val[0] <= val[1] and isclose_ordered(val[0], val[1], relEpsX, absEpsX):
        return True
    elif isclose_ordered(x[0], x[1], relEpsX, absEpsX):
        return True
    else:
        return False


def isclose_ordered(a, b, relTol, absTol):
    """ sorts two float values by their value and checks them for equality
    
    Args:
        :a:         first float value
        :a:         second float value
        :relTol:    relative tolerance
        :absTol:    absolute tolerance
        
    Returns:
        :True/False: True if x or difference between values is close and False
                     otherwise

    """    
    
    if abs(a) < abs(b): 
        return numpy.isclose(a, b, relTol, absTol)
    else:
        return numpy.isclose(b, a, relTol, absTol)
    
    
# def monotoneIncreasingIntervalNesting(f, xBounds, i, bi, bxrd_options):
#     """ reduces variable intervals of monotone increasing functions fx
#     by interval nesting

#         Args:
#             :f:                  object of type Function
#             :xBounds:            numpy array with set of variable bounds
#             :i:                  integer with current iteration variable index
#             :bi:                 current function residual bounds
#             :bxrd_options:       dictionary with function and variable interval 
#                                  tolerances

#         Return:                  list with one entry that is the reduced interval
#                                  of the variable with the index i

#     """
#     fxInterval = eval_fInterval(f, f.g_mpmath[i], xBounds, f.g_aff[i])
#     # first check if xBounds can be further reduced:
#     if fxInterval in bi: return xBounds[i]
#     if ivIntersection(fxInterval, bi)==[]: return []
    
#     # Otherwise, iterate each bound of bi:
#     relEpsX = bxrd_options["relTol"]
#     absEpsX = bxrd_options["absTol"]
#     curInterval = xBounds[i]    
#     fIntervalxLow, fIntervalxUp = getFIntervalsFromXBounds(f, curInterval, xBounds, i)

#     if fIntervalxLow.b < bi.a:
#         while (not numpy.isclose(float(mpmath.mpf(fxInterval.a)), 
#                                  float(mpmath.mpf(fxInterval.b)),relEpsX, absEpsX) 
#                and not numpy.isclose(float(mpmath.mpf(curInterval.a)), 
#                                      float(mpmath.mpf(curInterval.b)),relEpsX, absEpsX)):  
        
#                          curInterval, fxInterval = iteratefBound(f, curInterval, 
#                                                                  xBounds, i, bi,
#                                                                  increasing = True,
#                                                                  lowerXBound = True)
#                          if curInterval == [] or fxInterval == []: return []

#     lowerBound = curInterval.a
#     curInterval  = xBounds[i]    
    
#     fxInterval = eval_fInterval(f, f.g_mpmath[i], xBounds, f.g_aff[i])
    
#     if fIntervalxUp.a > bi.b:
#         while (not numpy.isclose(float(mpmath.mpf(fxInterval.a)), 
#                                  float(mpmath.mpf(fxInterval.b)),relEpsX, absEpsX) 
#                and not numpy.isclose(float(mpmath.mpf(curInterval.a)), 
#                                      float(mpmath.mpf(curInterval.b)),relEpsX, 
#                                      absEpsX)): 

#             curInterval, fxInterval = iteratefBound(f, curInterval, 
#                                                     xBounds, i, bi,
#                                                     increasing = True,
#                                                     lowerXBound = False)
#             if curInterval == [] or fxInterval == []: return []
#     upperBound = curInterval.b
    
#     return mpmath.mpi(lowerBound, upperBound)


# def monotoneDecreasingIntervalNesting(f, xBounds, i, bi, bxrd_options):
#     """ reduces variable intervals of monotone decreasing functions fx
#     by interval nesting

#         Args:
#             :f:                  object of type function
#             :xBounds:            numpy array with set of variable bounds
#             :i:                  integer with current iteration variable index
#             :bi:                 current function residual bounds
#             :bxrd_options:       dictionary with function and variable interval tolerances

#         Return:                  list with one entry that is the reduced interval
#                                  of the variable with the index i

#     """
#     relEpsX = bxrd_options["relTol"]
#     absEpsX = bxrd_options["absTol"]
#     fxInterval =  eval_fInterval(f, f.g_mpmath[i], xBounds, f.g_aff[i])
#     curInterval = xBounds[i]

#     if ivIntersection(fxInterval, bi)==[]: return []

#     # first check if xBounds can be further reduced:
#     if fxInterval in bi: return curInterval

#     # Otherwise, iterate each bound of bi:
#     fIntervalxLow, fIntervalxUp = getFIntervalsFromXBounds(f, curInterval, 
#                                                            xBounds, i)
#     if fIntervalxLow.a > bi.b:
#         while (not numpy.isclose(float(mpmath.mpf(fxInterval.a)), 
#                                  float(mpmath.mpf(fxInterval.b)),relEpsX, absEpsX) 
#                and not numpy.isclose(float(mpmath.mpf(curInterval.a)), 
#                                      float(mpmath.mpf(curInterval.b)),relEpsX, 
#                                      absEpsX)):         
#                          curInterval, fxInterval = iteratefBound(f, curInterval, 
#                                                                  xBounds, i, bi,
#                                                                  increasing = False,
#                                                                  lowerXBound = True)
#                          if curInterval == [] or fxInterval == []: return []
        
#     lowerBound = curInterval.a  
#     curInterval  = xBounds[i]        
#     fxInterval =  eval_fInterval(f, f.g_mpmath[i], xBounds, f.g_aff[i])
    
#     if fIntervalxUp.b < bi.a:
#         while (not numpy.isclose(float(mpmath.mpf(fxInterval.a)), 
#                                  float(mpmath.mpf(fxInterval.b)),relEpsX, absEpsX) 
#                and not numpy.isclose(float(mpmath.mpf(curInterval.a)), 
#                                      float(mpmath.mpf(curInterval.b)),relEpsX, 
#                                      absEpsX)): 
#             curInterval, fxInterval = iteratefBound(f, curInterval, 
#                                                     xBounds, i, bi,
#                                                     increasing = False,
#                                                     lowerXBound = False)
#             if curInterval == [] or fxInterval == []: return []
#     upperBound = curInterval.b
    
#     return mpmath.mpi(lowerBound, upperBound)


# def iteratefBound(f, curInterval, xBounds, i, bi, increasing, lowerXBound):
#     """ returns the half of curInterval that contains the lower or upper
#     bound of bi (biLimit)

#     Args:
#         :f:                  object of type function
#         :curInterval:        X-Interval that contains the solution to f(x) = biLimit
#         :xBounds:            numpy array with set of variable bounds
#         :i:                  integer with current iteration variable index
#         :bi:                 current function residual
#         :increasing:         boolean: True = function is monotone increasing,
#                              False = function is monotone decreasing
#         :lowerXBound:        boolean: True = lower Bound is iterated
#                              False = upper bound is iterated

#     Return:                  reduced curInterval (by half) and bounds of in curInterval

#     """

#     biBound = residualBoundOperator(bi, increasing, lowerXBound)

#     curlowerXInterval = mpmath.mpi(curInterval.a, curInterval.mid)
#     fIntervalxLow, fIntervalxUp = getFIntervalsFromXBounds(f, curlowerXInterval,
#                                                            xBounds, i)

#     fxInterval = fBoundsOperator(fIntervalxLow, fIntervalxUp, increasing, 
#                                  lowerXBound)
#     if biBound in fxInterval: return curlowerXInterval, fxInterval

#     else:
#         curUpperXInterval = mpmath.mpi(curInterval.mid, curInterval.b)
#         fIntervalxLow, fIntervalxUp = getFIntervalsFromXBounds(f, curUpperXInterval,
#                                                                xBounds, i)

#         fxInterval = fBoundsOperator(fIntervalxLow, fIntervalxUp, increasing, 
#                                      lowerXBound)
#         if biBound in fxInterval: return curUpperXInterval, fxInterval
        # else: return [], []


def getFIntervalsFromXBounds(f, curInterval, xBounds, i):
    """ returns function interval for lower variable bound and upper variable
    bound of variable interval curInterval.

    Args:
        :f:              object of type Function
        :curInterval:    current variable interval in mpmath logic
        :xBounds:        set of variable intervals in mpmath logic
        :i:              index of currently iterated variable interval

    Returns:             function interval for lower variable bound and upper 
                         variable bound

    """
    curXBoundsLow = list(xBounds)
    curXBoundsUp = list(xBounds)
    curXBoundsLow[i]  = curInterval.a
    curXBoundsUp[i] = curInterval.b

    return (eval_fInterval(f, f.g_mpmath[i], curXBoundsLow, f.g_aff[i]), 
            eval_fInterval(f, f.g_mpmath[i], curXBoundsUp, f.g_aff[i]))


# def fBoundsOperator(fIntervalxLow, fIntervalxUp, increasing, lowerXBound):
#     """ returns the relevant fInterval bounds for iterating the certain bi
#     bound

#     Args:
#         :fIntervalxLow:   function interval of lower variable bound in mpmath
#                           logic
#         :fIntervalxUp:    function interval of upper variable bound in mpmath
#                           logic
#         :increasing:      boolean: True = monotone increasing, False = monotone
#                           decreasing function
#         :lowerXBound:     boolean: True = lower variable bound, False = upper
#                           variable bound

#     Return:               relevant function interval for iterating bi bound in 
#                           mpmath logic

#     """
#     if increasing and lowerXBound: return mpmath.mpi(fIntervalxLow.b, fIntervalxUp.b)
#     if increasing and not lowerXBound: return mpmath.mpi(fIntervalxLow.a, fIntervalxUp.a)

#     if not increasing and lowerXBound: return mpmath.mpi(fIntervalxUp.a, fIntervalxLow.a)
#     if not increasing and not lowerXBound: return mpmath.mpi(fIntervalxUp.b, fIntervalxLow.b)


# def residualBoundOperator(bi, increasing, lowerXBound):
#     """ returns the residual bound that is iterated in the certain case

#     Args:
#         :bi:              function residual interval in mpmath logic
#         :increasing:      boolean: True = monotone increasing, False = monotone
#                           decreasing function
#         :lowerXBound:     boolean: True = lower variable bound, False = upper
#                           variable bound

#     Return:               lower or upper bound of function residual interval in 
#                          mpmath logic

#     """
#     if increasing and lowerXBound: return bi.a
#     if increasing and not lowerXBound: return bi.b
#     if not increasing and lowerXBound: return bi.b
#     if not increasing and not lowerXBound: return bi.a


def getMonotoneFunctionSections(f, i, xBounds, bxrd_options):
    """seperates variable interval into variable interval sets where a function
    with derivative dfdx is monontoneous

    Args:
        :f:                   object of type Function
        :i:                   index of differential variable
        :xBounds:             numpy array with variable bounds
        :bxrd_options:        dictionary with function and variable interval
                              tolerances

    Returns:
        :monIncreasingZone:   monotone increasing intervals
        :monDecreasingZone:   monotone decreasing intervals
        :interval:            non monotone zone if  function interval can not be
                              reduced to monotone increasing or decreasing section

    """
    relEpsX = bxrd_options["relTol"]
    absEpsX = bxrd_options["absTol"]
    maxIvNo = bxrd_options["resolution"]
    monIncreasingZone = []
    monDecreasingZone = []
    org_xiBounds = xBounds[i]
    interval = [xBounds[i]]

    while(interval != [] and len(interval) <= maxIvNo):
        curIntervals = []

        for xc in interval:
            (newIntervals, newMonIncreasingZone, 
             newMonDecreasingZone) = testIntervalOnMonotony(f, xc, list(xBounds), i)
            monIncreasingZone = addIntervaltoZone(newMonIncreasingZone,
                                                          monIncreasingZone, 
                                                          bxrd_options)
            monDecreasingZone = addIntervaltoZone(newMonDecreasingZone,
                                                          monDecreasingZone, 
                                                          bxrd_options)
            curIntervals += newIntervals

        #if checkIntervalWidth(curIntervals, absEpsX, 0.1*relEpsX) == interval:
        joined_interval = joinIntervalSet(curIntervals, relEpsX, absEpsX)     
        if (joined_interval == interval or 
            checkWidths(joined_interval, relEpsX, absEpsX)):
            interval = joined_interval
            break    
        interval = curIntervals#checkIntervalWidth(curIntervals, absEpsX, 0.1*relEpsX)

    if not len(interval) <= maxIvNo:
        interval = joinIntervalSet(interval, relEpsX, absEpsX)

    if interval == [] and monDecreasingZone == [] and monIncreasingZone ==[]:
        return [], [], [org_xiBounds]
    return monIncreasingZone, monDecreasingZone, interval


def convertIntervalBoundsToFloatValues(interval):
    """ converts mpmath.mpi intervals to list with bounds as float values

    Args:
        :interval:              interval in math.mpi logic

    Returns:                     list with bounds as float values

    """
    return [float(mpmath.mpf(interval.a)), float(mpmath.mpf(interval.b))]


def testIntervalOnContinuity(f, interval, xBounds, i, discontinuousZone):
    """ splits interval into 2 halfs and orders them regarding their continuity
   in the first derrivative.

    Args:
        :f:                 object of type function
        :interval:          x interval in mpmath.mpi-logic
        :xBounds:           numpy array with variable bounds in mpmath.mpi-logic
        :i:                 variable index
        :discontinuousZone: list with current discontinuous intervals in
                            mpmath.mpi-logic
        
    Returns:
        :continuousZone:    list with new continuous intervals in
                            mpmath.mpi-logic

    """ 
    continuousZone = []
    curXBoundsLow = mpmath.mpi(interval.a, interval.mid)
    xBounds[i] = curXBoundsLow

    if isinstance(f.dgdx_mpmath[i], mpmath.ctx_iv.ivmpf): dgdxLow = f.dgdx_mpmath[i]
    else: dgdxLow = f.dgdx_mpmath[i](*xBounds)
             
    curXBoundsUp = mpmath.mpi(interval.mid, interval.b)
    xBounds[i] = curXBoundsUp

    if isinstance(f.dgdx_mpmath[i], mpmath.ctx_iv.ivmpf): dgdxUp = f.dgdx_mpmath[i]
    else: dgdxUp = f.dgdx_mpmath[i](*xBounds)
    
    if dgdxLow == False: discontinuousZone.append(curXBoundsLow)    
    if dgdxUp == False : discontinuousZone.append(curXBoundsUp)       
    if dgdxLow == False or dgdxUp == False: return False

    if (not numpy.nan_to_num(-numpy.inf) in dgdxLow and 
        not numpy.nan_to_num(numpy.inf) in dgdxLow): continuousZone.append(curXBoundsLow)
    else: discontinuousZone.append(curXBoundsLow) 
        
    if (not numpy.nan_to_num(-numpy.inf) in dgdxUp and 
        not numpy.nan_to_num(numpy.inf) in dgdxUp): continuousZone.append(curXBoundsUp)
    else: discontinuousZone.append(curXBoundsUp) 

    return continuousZone


def testIntervalOnMonotony(f, interval, xBounds, i):
    """ splits interval into 2 halfs and orders concering their monotony
    behaviour of f (first derivative dfdx):
        1. monotone increasing function in interval of x
        2. monotone decreasing function in interval of x
        3. non monotone function in interval of x

    Args:
        :f:              object of type function
        :interval:       x interval in mpmath.mpi-logic
        :xBounds:        numpy array with variable bounds in mpmath.mpi-logic
        :i:              variable index
                         that have an x indepenendent derrivate constant interval 
                         of [-inf, +inf] for example: f=x/y-1 and y in [-1,1]  
            
    Returns:
        3 lists nonMonotoneZone, monotoneIncreasingZone, monotoneDecreasingZone
        and one updated count of [-inf,inf] dfdxIntervals as integer

    """
    nonMonotoneZone = []
    monotoneIncreasingZone = []
    monotoneDecreasingZone = []
    
    curXBoundsLow = mpmath.mpi(interval.a, interval.mid)
    curXBoundsUp = mpmath.mpi(interval.mid, interval.b)

    xBounds[i] = curXBoundsLow
    if isinstance(f.dgdx_mpmath[i], mpmath.ctx_iv.ivmpf): 
        dgdxLow =f.dgdx_mpmath[i]
        if bool(dgdxLow >= 0): monotoneIncreasingZone.append(interval)
        elif bool(dgdxLow <= 0): monotoneDecreasingZone.append(interval)
        else: monotoneIncreasingZone.append(interval)
        return nonMonotoneZone, monotoneIncreasingZone, monotoneDecreasingZone 
    else: dgdxLow = f.dgdx_mpmath[i](*xBounds)
    
    if bool(dgdxLow >= 0): monotoneIncreasingZone.append(curXBoundsLow)
    elif bool(dgdxLow <= 0): monotoneDecreasingZone.append(curXBoundsLow)
    else: nonMonotoneZone.append(curXBoundsLow)

    xBounds[i] = curXBoundsUp
    if isinstance(f.dgdx_mpmath[i], mpmath.ctx_iv.ivmpf): 
        dgdxUp =f.dgdx_mpmath[i]
    else: dgdxUp = f.dgdx_mpmath[i](*xBounds)
    
    if bool(dgdxUp >= 0): monotoneIncreasingZone.append(curXBoundsUp)
    elif bool(dgdxUp <= 0): monotoneDecreasingZone.append(curXBoundsUp)
    else: nonMonotoneZone.append(curXBoundsUp)

    return nonMonotoneZone, monotoneIncreasingZone, monotoneDecreasingZone


def addIntervaltoZone(newInterval, monotoneZone, bxrd_options):
    """ adds one or two monotone intervals newInterval to list of other monotone
    intervals. Function is related to function testIntervalOnMonotony, since if  the
    lower and upper part of an interval are identified as monotone towards the same direction
    they are joined and both parts are added to monotoneZone. If monotoneZone contains
    an interval that shares a bound with newInterval they are joined. Intersections
    should not occur afterwards.

    Args:
        :newInterval:         list with interval(s) in mpmath.mpi logic
        :monotoneZone:        list with intervals from mpmath.mpi logic
        :bxrd_options:        dictionary with variable interval specified tolerances
                              absolute = absTol, relative = relTol

    Return:
        :monotoneZone:        monotoneZone including newInterval

    """
    absEpsX = bxrd_options["absTol"]
    relEpsX = bxrd_options["relTol"]
    red_disconti = 0.1   # To ensure that interval is not joined when discontinuity is present

    if newInterval != []:

        if len(newInterval) > 1:
            newInterval = [mpmath.mpi(newInterval[0].a, newInterval[1].b)]

        if monotoneZone == []:
            monotoneZone += newInterval
            return monotoneZone

        else:
            for iv in newInterval:
                monotoneZone.append(iv)

            return joinIntervalSet(monotoneZone, 
                                   relEpsX*red_disconti, absEpsX*red_disconti)
    return monotoneZone


def joinIntervalSet(ivSet, relEpsX, absEpsX):
    """joins all intervals in an interval set ivSet that intersec or share the
    same bound

    Args:
        :ivSet:              set of intervals in mpmath.mpi logic
        :relEps:             relative tolerance of variable intervals
        :absEps:             absolute tolerance of variable intervals

    Returns:
        :newIvSet:           new set of joint intervals

    """
    newIvSet = ivSet
    noOldIvSet = len(ivSet) + 1

    while noOldIvSet != len(newIvSet) and len(newIvSet)!=1:
        ivSet = newIvSet
        noOldIvSet = len(ivSet)
        newIvSet = []
        noIv = len(ivSet)

        while noIv != 0: #len(newIvSet) != noIv:

            for i in range(1, noIv):
                if ivSet[0] in ivSet[i]:
                    newIvSet.append(ivSet[i])
                    ivSet.remove(ivSet[0])
                    ivSet.remove(ivSet[i-1])
                    break

                elif ivSet[i] in ivSet[0]:
                    newIvSet.append(ivSet[0])
                    ivSet.remove(ivSet[0])
                    ivSet.remove(ivSet[i-1])
                    break

                elif ivIntersection(ivSet[0], ivSet[i])!=[]:
                    newIvSet.append(mpmath.mpi(min(ivSet[i].a, ivSet[0].a),
                                               max(ivSet[i].b, ivSet[0].b)))
                    ivSet.remove(ivSet[0])
                    ivSet.remove(ivSet[i-1])
                    break

                elif i == noIv-1:
                    newIvSet.append(ivSet[0])
                    ivSet.remove(ivSet[0])
                    break

            if len(ivSet) == 1:
                newIvSet.append(ivSet[0])
                ivSet.remove(ivSet[0])

            noIv = len(ivSet)

    return newIvSet


def addIntervalToNonMonotoneZone(newIntervals, curIntervals):
    """adds copy of newInterval(s) to already stored ones in list curIntervals

    """
    if newIntervals != []: curIntervals.append(list(newIntervals))


def reduceNonMonotoneIntervals(args):
    """ reduces non monotone intervals by simply calculating function values for
    interval segments of a discretized variable interval and keeps the hull of 
    those segments that intersect with bi. The discretization resolution is 
    defined in bxrd_options.

    Args:
        :nonMonotoneZone:    list with non monotone variable intervals
        :reducedIntervals:   lits with reduced non monotone variable intervals
        :f:                  object of type Function
        :i:                  integer with current iteration variable index
        :xBounds:            numpy array with set of variable bounds
        :bi:                 current function residual bounds
        :bxrd_options:       for function and variable interval tolerances in the 
                             used algorithms and resolution of the discretization

    Returns:                  reduced x-Interval(s) and list of monotone x-intervals
        
 """   
    nonMonotoneZone = args["0"]
    reducedIntervals = args["1"]
    f = args["2"]
    i = args["3"]
    xBounds = args["4"]
    bi = args["5"]
    bxrd_options = args["6"]
    
    relEpsX = bxrd_options["relTol"]
    precision = getPrecision(xBounds)
    resolution = bxrd_options["resolution"]

    for curNonMonZone in nonMonotoneZone:
        curInterval = convertIntervalBoundsToFloatValues(curNonMonZone)
        x = numpy.linspace(curInterval[0], curInterval[1], int(resolution)+1)
        x_low = None
        x_up = None
        fLowValues, fUpValues = getFunctionValuesIntervalsOfXList(x, 
                                                                  f.g_mpmath[i], 
                                                                  xBounds, i)       
        for k, fLow_val in enumerate(fLowValues):
            if ivIntersection(mpmath.mpi(fLow_val, fUpValues[k]), bi):
                x_low = x[k]
                break
        for k, fLow_val in enumerate(reversed(fLowValues)):
            k_up = len(fLowValues)-1-k
            if ivIntersection(mpmath.mpi(fLow_val, fUpValues[k_up]), bi):
                x_up = x[k_up+1]
                break
        if x_low!=None and x_up!=None:  reducedIntervals.append(mpmath.mpi(x_low,
                                                                           x_up))   
    return joinIntervalSet(reducedIntervals, relEpsX, precision)


def getFunctionValuesIntervalsOfXList(x, f_mpmath, xBounds, i):
    """ calculates lower and upper function value bounds for segments that are
    members of a list and belong to a discretized variable interval.

    Args:
        :x:          numpy list with segments for iteration variable xi
        :f_mpmath:   mpmath function
        :xBounds:    numpy array with variable bounds in mpmath.mpi.logic
        :i:          index of currently reduced variable

    Returns:         list with lower function value bounds within x and upper
                    function value bounds within x

    """
    funValuesLow = []
    funValuesUp = []
    if isinstance(f_mpmath, mpmath.ctx_iv.ivmpf): 
        funValuesLow = [float(mpmath.mpf(f_mpmath.a))]*len(x)-1
        funValuesUp = [float(mpmath.mpf(f_mpmath.b))]*len(x)-1       
    else:
        for j in range(len(x)-1):
            xBounds[i] = mpmath.mpi(x[j], x[j+1])         
            curfunValue = f_mpmath(*xBounds)   
            funValuesLow.append(float(mpmath.mpf(curfunValue.a)))
            funValuesUp.append(float(mpmath.mpf(curfunValue.b)))
        
    return funValuesLow, funValuesUp


def reduceTwoIVSets(ivSet1, ivSet2):
    """ reduces two interval sets to one and keeps only the resulting interval
    when elements of both intervals intersect. Each element of the longer 
    interval set is compared to the list of the shorter interval set.

    Args:
        :ivSet1:          list 1 with intervals in mpmath.mpi logic
        :ivSet2:          list 2 with intervals in mpmath.mpi logic

    Returns:
        :list:            with reduced set of intervals

    """
    ivReduced = []
    if len(ivSet1) >= len(ivSet2):
        ivLong = ivSet1
        ivShort = ivSet2
    else:
        ivLong = ivSet2
        ivShort = ivSet1
    
    for iv in ivLong:
        curIV = compareIntervalToIntervalSet(iv, ivShort)
        if curIV != []: ivReduced.append(curIV)

    return ivReduced


def setOfIvSetIntersection(setOfIvSets):
    """ intersects elements of a set of sets with disjoint intervals and returns 
    a list with the intersecting intervals.
    
    Args:
        :setOfIvSets: list with lists of disjoint intervals in mpmath.mpi logic
    
    Returns:          
        :ivSet:     list with intersected, disjoint intervals in mpmath.mpi logic
        
    """
    if len(setOfIvSets) <= 1: return setOfIvSets
    
    ivSet = setOfIvSets.pop(0)
       
    for curIvSet in setOfIvSets:
        ivSet = ivSetIntersection(ivSet, curIvSet)
        if ivSet == []: return []
    return ivSet
              
  
def ivSetIntersection(ivSet1, ivSet2):
    """ intersects two sets of intervals with each other.
    
    Args:
        :ivSet1:    first list with intervals in mpmath.mpi logic
        :ivSet2:    second list with intervals in mpmath.mpi logic
    
    Returns:          
        :ivSetIntersected:  set with intersecting intervals in mpmath.mpi logic
        
    """
    
    ivSetIntersected = []
    for iv in ivSet1:     
        ivWithIvSetIntersection(iv, ivSet2, ivSetIntersected)
                    
    return ivSetIntersected  
 
    
def ivWithIvSetIntersection(iv1, ivSet, ivSetIntersected):
    """ intersects an interval with a set of intervals.
    
    Args:
        :iv1:               interval  in mpmath.mpi logic
        :ivSet:             list with intervals in mpmath.mpi logic
        :ivSetIntersected:  set with intersecting intervals in mpmath.mpi logic
    
    Returns: True for successful method termination     
        
    """   
    for iv2 in ivSet:
        intersection = ivIntersection(iv1, iv2)
        if intersection !=[]: ivSetIntersected.append(intersection)
    return True
            
            
def compareIntervalToIntervalSet(iv, ivSet):
    """ checks if there is an intersection betweeen interval iv and a list of
    intervals ivSet. If there is one the intersection is returned.

    Args:
        :iv:         interval in mpmath.mpi logic
        :ivSet:      list with intervals in mpmath.mpi logic

    Return:          intersection or empty list if there is no intersection

    """
    for iv2 in ivSet:
        try:
            newIV = ivIntersection(iv, iv2)
        except: return []

        if newIV != []: return newIV
        
    return []


def checkWidths(X, relEps, absEps):
    """ returns True if all intervals are degenerate

    Args:
        :X:             list with set of intervals
        :relEps:        relative tolerance
        :absEps:        absolute tolerance

    Returns:
        :boolean:       True if all intervals are degenerate

    """
    almostEqual = False * numpy.ones(len(X), dtype = bool)
    
    for i, x in enumerate(X):    
         if isclose_ordered(float(mpmath.mpf(x.a)), float(mpmath.mpf(x.b)),
                          relEps, absEps): almostEqual[i] = True

    return almostEqual.all()


def iv_newton(model, box, i, bxrd_options):
    """ Computation of the Interval-Newton Method to reduce the single 
    interval box[i]
     
    Args: 
        :model:         instance of model class
        :box:           list with current bounds in mpmath.mpi formate
        :i:             index of currently reduced interval as integer
        :bxrd_options:  dictionary with user settings
            
    Returns:
        :y_new:     list with reduced intervals of box[i]
        :unique:    boolean, true if y_new lies in the interior(box[i])    
         
    """  
    x_c=None
    
    if bxrd_options["newtonPoint"]=="center": 
        x_c = [float(mpmath.mpf(iv.mid)) for iv in box]
    elif bxrd_options["newtonPoint"]=="condJ":
        x_all = [[float(mpmath.mpf(iv.a)) for iv in box],
                 [float(mpmath.mpf(iv.mid)) for iv in box],
                 [float(mpmath.mpf(iv.b)) for iv in box]]    
        condNo = []
        for x in x_all:
            try: condNo.append(numpy.linalg.cond(model.jacobianLambNumpy(*x))) 
            except: condNo.append(numpy.inf)
        x_c = [float(mpmath.mpf(iv.mid)) for iv in box]
              
    if bxrd_options["preconditioning"] == "inverseCentered":
        G_i, r_i, n_x_c, n_box, n_i = get_precondition_centered(model, box, i,
                                                                x_c)
    elif bxrd_options["preconditioning"] == "inversePoint":
        G_i, r_i, n_x_c, n_box, n_i = get_precondition_point(model, box, i, 
                                                             x_c)
    elif bxrd_options["preconditioning"] == "diagInverseCentered":
        G_i, r_i, n_x_c, n_box, n_i = get_diag_precondition_centered(model, 
                                                                   box, i, x_c)   
    elif bxrd_options["preconditioning"] == "pivotAll":     
        y_new = get_best_from_pivotAll(model, box, i, bxrd_options, x_c)
        return y_new
    else:
        j = model.rowPerm[model.colPerm.index(i)]
        G_i, r_i, n_x_c, n_box, n_i = get_org_newton_system(model, box, i, j, 
                                                            x_c)
    if bxrd_options["newtonPoint"] == "3P":
        x_low = newton_step(r_i, G_i, n_x_c[0], n_box, n_i, bxrd_options)
        x_up = newton_step(r_i, G_i, n_x_c[1], n_box, n_i, bxrd_options)
        y_new = setOfIvSetIntersection([mpmath.mpi(x_low.a, x_up.b), y_new])
    else:
        y_new = newton_step(r_i, G_i, n_x_c, n_box, n_i, bxrd_options)
        
    return y_new


def get_best_from_pivotAll(model, box, i, bxrd_options, x_c=None):
    """ reduces variable i in all functions it appears in and intersects
    their results. If no gap occurs box[i] is automatically updated before the 
    next function.
     
    Args:
        :model:         instance of model class
        :box:           list with current bounds in mpmath.mpi formate
        :i:             index of currently reduced bound as integer
        :bxrd_options:  dictionary with solver settings
        :x_c:           list with currently used point
            
    Returns:
        :y_new:         list with reduced interval(s) of variable i
        
    """ 
    if bxrd_options["unique_nwt"]:
        unique_test = True
    else: unique_test = False
    f_for_unique_test = False
    y_old = [box[i]]
    y_new = [box[i]]
        
    for j in model.fWithX[i]:     
        y_old = y_new
        (G_i, r_i, 
         n_x_c, n_box, n_i) = get_org_newton_system(model, box, i, j, x_c,
                                                    bxrd_options["newtonPoint"])      
        if bxrd_options["newtonPoint"] == "3P":
            x_low, unique = newton_step(r_i[0], G_i, n_x_c[0], n_box, n_i, bxrd_options)
            x_up, unique = newton_step(r_i[1], G_i, n_x_c[1], n_box, n_i, bxrd_options)
            if x_low == [] or x_up == []:
                y_new = []
                break
            y_new = setOfIvSetIntersection([[mpmath.mpi(x_low[0].a, x_up[0].b)], 
                                            y_new])
        else:
            y_new = newton_step(r_i, G_i, n_x_c, n_box, 
                                                     n_i, bxrd_options)
            
            if unique_test:
                (f_for_unique_test, 
                 bxrd_options["unique_nwt"]) = update_for_unique_test(j, model.fWithX[i][-1],
                                                                      f_for_unique_test,
                                                                      bxrd_options["unique_nwt"])
        if y_new == [] or y_new==[[]]: break
        y_new = setOfIvSetIntersection([y_new, y_old])   
        if len(y_new)==1:
            box = list(box)
            box[i] = y_new[0]
            x_c[i] = convert_mpi_float(y_new[0].mid)
    if f_for_unique_test: bxrd_options["unique_nwt"] = True
        
    return y_new
        

def update_for_unique_test(j, nj, f_for_unique_test, unique):
    """ this method turns the unique parameter to true in functions 
    so that all functions' reduction steps will be investigated for fulfilling 
    the unqiue solution criterion
    
    Args:
        :j:                 index of currently reduced function
        :nj:                index of last function that is used for variable 
                            reduction
        :f_for_unique_test: boolean that is true if a for the current variable
                            a function for a successful unqiue solution test 
                            has already been found, to save work unique is
                            then turned False
        :unique:            unique paramter that determines
                            if uniqueness check shall be done in reduction step
                            or not
        
    Returns:
        :f_for_unique_test: updated boolean
        :unqiue:            updated boolean
        
    """
    if (not unique and not j==nj and not f_for_unique_test):
                unique = True
    elif unique:
                f_for_unique_test = True
                unique = False
    return f_for_unique_test,unique

def get_diag_precondition_centered(model, box, i, x_c):
    """ preconditions the i-th jacobian row and the i-th resiudal vector entry
    only with the center point of the interval j_ii. If m(j_ii) is 0 the entry 
    is set to 1 and no preconditioning is used
     
    Args:
        :model:         instance of model class
        :box:           list with current bounds in mpmath.mpi formate
        :i:             index of currently reduced bound as integer
        :x_c:           list with currently used point
            
    Returns:
        :G_i[i_perm]:   row of preconditioned jacobian
        :r_i:           scalar value of preconditioned residual
        :x_c_sub:       list with currently used point changed to permutation
        :box_sub:       list with permuted box values
        :function.glb_ID.index(i):        permuted index of i as integer
        
    """     
    j = model.rowPerm[model.colPerm.index(i)]
    function = model.functions[j]
    box_sub = [box[k] for k in model.functions[j].glb_ID] 
    x_c_sub = [x_c[i] for i in function.glb_ID]   
    G_i_row = numpy.zeros(len(function.glb_ID),dtype=object)
    Y = get_function_value_as_iv(function, 
                                 function.dgdx_mpmath[function.glb_ID.index(i)], 
                                 box_sub)  
    Y = float(mpmath.mpf(Y.mid))
    if Y == 0: Y = 1.0
    try: r_i = function.f_numpy(*x_c_sub) / Y
    except: r_i = numpy.inf
    
    for k in range(len(box_sub)):
        G_i_row[k] = get_function_value_as_iv(function, function.dgdx_mpmath[k], 
                                              box_sub) / Y
                        
    return G_i_row, r_i, x_c_sub, box_sub, function.glb_ID.index(i)


def get_function_value_as_iv(f,f_mpmath, box):
    """ returns current interval extended function value in box
    
    Args:
        :f:         function object of class Function
        :f_mpmath:  mpmath function that needs to be evluated
        :box:       list or numpy.array of variable bounds
        
    Returns:
        interval extended function value in mpmath.mpi formate

    """
    if isinstance(f_mpmath, mpmath.ctx_iv.ivmpf): return f_mpmath
    elif isinstance(f_mpmath, float) or isinstance(f_mpmath, int): 
        return mpmath.mpi(f_mpmath)
    else: return f.eval_mpmath_function(box, f_mpmath) #mpmath.mpi(f_mpmath(*box))


def get_precondition_point(model, box, i, x_c):
    """ preconditions the jacobian and resiudal vector with inverse of jacobian
    evaluated at newton point and returns the currently reduced row values. 
    If the inverse is singular, the method continues without preconditioning.
     
    Args:
        :model:         instance of model class
        :box:           list with current bounds in mpmath.mpi formate
        :i:             index of currently reduced bound as integer
        :x_c:           list with currently used point
            
    Returns: see function get_org_newton_system(model, box, i, j, x_c)
        
    """       
    i_perm = model.colPerm.index(i)
    fval = numpy.array(model.fLamb(*x_c)) 

    if not isinstance(model.jac_center, numpy.ndarray): 
        model.jac_center = model.jacobianLambNumpy(*x_c)
    
    if not (numpy.isnan(numpy.linalg.cond(model.jac_center)) or 
        numpy.isinf(numpy.linalg.cond(model.jac_center))):
        if not isinstance(model.interval_jac, numpy.ndarray):
            model.interval_jac = numpy.array(model.jacobianLambMpmath(*box))
             
        jac_center_inv = numpy.linalg.inv(model.jac_center)
        r_i = numpy.dot(jac_center_inv[i_perm], fval[model.rowPerm])        
        G_i = real_interval_matrix_product([jac_center_inv[i_perm]], 
                                           permute_matrix(model.interval_jac,
                                                          model.rowPerm, 
                                                          model.colPerm))
        x_c_perm = [x_c[k] for k in model.colPerm]
        box_perm = [box[k] for k in model.colPerm]
        return G_i, r_i, x_c_perm, box_perm, i_perm

    else:     
        j = model.rowPerm[model.colPerm.index(i)]
        return get_org_newton_system(model, box, i, j, x_c)


def get_precondition_centered(model, box, i, x_c):
    """ preconditions the jacobian and resiudal vector with inverse centred
    procedure and returns the currently reduced row values. If the inverse
    is singular, the method continues without preconditioning.
     
    Args:
        :interval_jac:  numpy.array with interval jacobian (original shape)
        :model:         instance of model class
        :box:           list with current bounds in mpmath.mpi formate
        :i:             index of currently reduced bound as integer
        :x_c:           list with currently used point
            
    Returns: see function get_org_newton_system(model, box, i, j, x_c)
        
    """       
    i_perm = model.colPerm.index(i)
    fval = numpy.array(model.fLamb(*x_c))
    
    if not isinstance(model.interval_jac, numpy.ndarray):
        model.interval_jac = numpy.array(model.jacobianLambMpmath(*box))
        model.jac_center = get_permuted_centered_jacobian(model.interval_jac, 
                                                          model)    
    if not (numpy.isnan(numpy.linalg.cond(model.jac_center)) or 
        numpy.isinf(numpy.linalg.cond(model.jac_center))):
        jac_center_inv = numpy.linalg.inv(model.jac_center)
        r_i = numpy.dot(jac_center_inv[i_perm], fval[model.rowPerm])        
        G_i = real_interval_matrix_product([jac_center_inv[i_perm,:]], 
                                           permute_matrix(model.interval_jac,
                                                          model.rowPerm, 
                                                          model.colPerm))
        x_c_perm = [x_c[k] for k in model.colPerm]
        box_perm = [box[k] for k in model.colPerm]
        return G_i, r_i, x_c_perm, box_perm, i_perm

    else:
        j = model.rowPerm[model.colPerm.index(i)]
        return  get_org_newton_system(model, box, i, j, x_c)


def get_permuted_centered_jacobian(interval_jac, model):
    """ gets the centers of the jacobian interval matrix and sorts due to the 
    permutation order
     
    Args:
        :interval_jac:  numpy.array with interval jacobian (original shape)
        :model:         instance of model class
            
    Returns:
        :jac_center:    numpy.array with matrix that consist of jacobian entry
                        centers with permuted shape     
                        
    """ 
    jac_center = numpy.zeros([len(model.rowPerm), len(model.colPerm)])
    
    for i, var_id in enumerate(model.xInF):
        for j in var_id:
            if isinstance(interval_jac[i][j], mpmath.ctx_iv.ivmpf): 
                jac_center[model.rowPerm.index(i), 
                           model.colPerm.index(j)] = float(mpmath.mpf(
                               interval_jac[i][j].mid))
            else:
                 jac_center[model.rowPerm.index(i), 
                           model.colPerm.index(j)] = float(interval_jac[i][j])                   
    return jac_center

    
def get_org_newton_system(model, box, i, j, x_c=None, nP=None):
    """ gets jacobian row nonzero entries and the referring function residual 
    without preconditioning
     
    Args:
        :model:             instance of typ model
        :box:               list with current box (original shape)
        :i:                 index of currently reduced variable as integer
        :j:                 index of currently reduced function as integer
        :x_c:               list with currently used point (original shape)
        :nP:                string that equals 3P for 3P-preconditioning
            
    Returns:
        :G_i_row:            numpy.array with nonzero jacobian entries      
        :r_i:                scalar value with function point as float
        :x_c_sub:            list with currently used point for function
        :box_sub:            list with current sub-box related to function
        :function.glb_ID.index(i): function's internal index of variable i'
        
    """
    function = model.functions[j]
    box_sub = [box[k] for k in model.functions[j].glb_ID] 
    if nP=="3P": 
        x_low = len(function.glb_ID) * [1]
        x_up = len(function.glb_ID) * [1]
    else:
        x_c_sub = [x_c[i] for i in function.glb_ID]   
        try: r_i = function.f_numpy(*x_c_sub)
        except: r_i = function.eval_mpmath_function(function.f_mpmath, [
            mpmath.mpi(x) for x in x_c_sub])
        
    G_i_row = numpy.zeros(len(function.glb_ID),dtype=object)  
    
    for k in range(len(box_sub)):
        G_i_row[k] = get_function_value_as_iv(function,function.dgdx_mpmath[k], 
                                              box_sub)
        if nP=="3P": 
            if G_i_row[k] > 0:
               x_low[k] = float(mpmath.mpf(box_sub[k].b))
               x_up[k] = float(mpmath.mpf(box_sub[k].a))              
            elif G_i_row[k] < 0:
               x_low[k] = float(mpmath.mpf(box_sub[k].a))
               x_up[k] = float(mpmath.mpf(box_sub[k].b))
            else:
               x_low[k] = float(mpmath.mpf(box_sub[k].mid))
               x_up[k] = float(mpmath.mpf(box_sub[k].mid))
    
    if nP=="3P":  
        x_c_sub=[x_low, x_up] 
        r_i = [function.f_numpy(*x) for x in x_c_sub]
          
    return G_i_row, r_i, x_c_sub, box_sub, function.glb_ID.index(i)
    

def newton_step(r_i, G_i, x_c, box, i, bxrd_options):
    """ calculates one newton step in interval-arithmetic
     
    Args: 
        :r_i:           float value with function residual 
        :G_i:           numpy.array with row entries of interval jacobian
        :x_c:           list with currently used point
        :box:           list with current box in mpmath.mpi formate
        :i:             index of currently reduced variable as integer
        :bxrd_options:  dictionary with user settings
            
    Returns:
        :y_new:         list with interval(s) after reduction of current 
                        variable                  
    """             
    iv_sum = sum([G_i[j] * (box[j] - x_c[j]) for j in range(len(G_i)) if j!=i])
    if numpy.isinf(r_i) or numpy.isnan(r_i) or r_i == []: 
        bxrd_options["unique_nwt"] = False
        return [box[i]]
    try:
        if abs(x_c[i]) > 1.0e8: tol = r_i/x_c[i] * mpmath.mpi(-1,1)
        else: tol = 0.0
        quotient = ivDivision(mpmath.mpi(r_i + iv_sum), G_i[i])
        #N = [x_c[i] - l for l in quotient] 
        N = [(x_c[i]* (1 - l/x_c[i])+tol)for l in quotient] # because of round off errors
        if bxrd_options["unique_nwt"]:
            bxrd_options["unique_nwt"] = checkUniqueness(N, 
                                                         bxrd_options["x_old"][i], 
                                                         bxrd_options["relTol"],
                                                         bxrd_options["absTol"])
        
        y_new = setOfIvSetIntersection([N, [box[i]]])
        
    except: 
        bxrd_options["unique_nwt"] = False
        return [box[i]]
    
    if y_new == []: 
        #print(N[0])
        return [check_accuracy_newton_step(box[i], N[0], bxrd_options)]
    
    return y_new

def convert_mpi_iv_float(iv):
    return [float(mpmath.mpf(iv.a)), float(mpmath.mpf(iv.b))]


def check_accuracy_newton_step(old_iv, new_iv, bxrd_options):
    """ to prevent discarding almost equal intervals
     
    Args: 
        :old_iv:           interval in mpmath.mpi logic
        :new_iv:           interval in mpmath.mpi logic
        :bxrd_options:     dictionary with user settings
            
    Returns:                 if lower/upper or upper/lower bounds of both 
                            intervals are almost equal they are joined to
                            one interval instead of discarded   
                            
    """     
    if (isinstance(old_iv, mpmath.ctx_iv.ivmpf)):
        mpi = True
        old_float = convert_mpi_iv_float(old_iv)
    else: 
        old_float = list(old_iv)
        mpi = False
    if (isinstance(new_iv, mpmath.ctx_iv.ivmpf)):
        new_float = convert_mpi_iv_float(new_iv)
    else: new_float = list(new_iv)
    if new_float[0] >= old_float[1]:
        if(isclose_ordered(old_float[1], new_float[0], bxrd_options["relTol"], 
                           bxrd_options["absTol"])):
            if mpi: return mpmath.mpi(old_float[0], new_float[1])
            else: return [old_float[0], new_float[1]]
        else: return []
    elif(new_float[1] <= old_float[0]):    
        if(isclose_ordered(new_float[1], old_float[0], bxrd_options["relTol"], 
                         bxrd_options["absTol"])):
            if mpi: return mpmath.mpi(new_float[0], old_float[1])
            else: return [new_float[0], old_float[1]]    
        else: return []
       

def permute_matrix(A, rowPerm, colPerm):
    """permutes matrix based on permutation order 
    
    Args:
        :A:             numpy.array with matrix
        :rowPerm:       list with row order as integers
        :colPerm:       list with column order as integers
    
    Returns:            permuted matrix as numpy.array
    
    """
    A = A[rowPerm,:]
    return A[:, colPerm]
    

def real_interval_matrix_product(r_A, iv_B):
    """multiplication of real-valued matrix with interval matrix 
    
    Args:
        :r_A:       numpy.array with real matrix
        :iv_B:      numpy.array with interval matrix (mpmath.mpi entries)
    
    Returns:     
        :product:   numpy.array with resulting interval matrix
        
    """
    product = numpy.zeros([len(r_A), len(iv_B[0])],dtype=object)
    for i,a in enumerate(r_A):
        for j,b in enumerate(numpy.transpose(iv_B)):
            product[i,j] = real_interval_vector_product(a, b)
    return product
 
       
def real_interval_vector_product(r_vec, iv_vec):
    """inner product of real-valued vector with interval vector 
    
    Args:
        :r_vec:     numpy.array with real vector
        :iv_vec:    numpy.array with interval vector (mpmath.mpi entries)
    
    Returns:        inner product of both vectors as interval vector 
        
    """
    return sum([r_vec[i] * iv for i, iv in enumerate(iv_vec)])

            
def identify_function_with_no_solution(output, functions, xBounds, bxrd_options):
    """ checks which function has no solution in the current function range
    for error report
    
    Args:
        :output:        dictionary with box reduction results
        :functions:     list with function objects
        :xBounds:       numpy.array with mpmath.mpi intervals of variables
        :bxrd_options:  dictionary with box reduction settings
        
    Returns:
        :output:        updated dictionary about failed equation

    """
    for f in functions:
        if not 0 in eval_fInterval(f, f.f_mpmath[0], xBounds, f.f_aff[0],
                                   bxrd_options["tightBounds"],
                                   bxrd_options["resolution"]):         
            output["noSolution"] = FailedSystem(f.f_sym, f.x_sym[0])
            output["xAlmostEqual"] = False 
            output["xSolved"] = False
            return output
        else:
            return output


def solutionInFunctionRange(functions, xBounds, bxrd_options):
    """checks, if the solution (0-vector) can lie in these Bounds and returns 
    true or false 
    
    Args: 
        :model:             instance of class-Model
        :xBounds:           current Bounds of Box
        :bxrd_options:      options with absTolerance for deviation from the solution
        
    Returns:
        :solutionRange:     boolean that is true if solution in the range
    """
    if xBounds == []: return False 
    absTol = bxrd_options["absTol"]
     
    for f in functions:        
        if bxrd_options["affineArithmetic"]: 
            fInterval = eval_fInterval(f, f.f_mpmath[0], [xBounds[i] for i in f.glb_ID], f.f_aff[0],
                                       bxrd_options["tightBounds"],
                                       bxrd_options["resolution"])
        else:
            fInterval = eval_fInterval(f, f.f_mpmath[0], [xBounds[i] for i in f.glb_ID],False, 
                                       bxrd_options["tightBounds"],
                                       bxrd_options["resolution"])

        if not(fInterval.a<=0+absTol and fInterval.b>=0-absTol):
            return False

    return True


def solutionInFunctionRangePyibex(functions, xBounds, bxrd_options):
    """checks, if box is empty by reducing it three times with HC4 method
    
    Args: 
        :functions:         instance of class function
        :xBounds:           current Bounds of Box
        :bxrd_options:      options with absTolerance for deviation from the 
                            solution
        
    Returns: 
        boolean that is True if solution is in function range and False otherwise

    """
    if xBounds == []: return False 
    xNewBounds = list(xBounds)
    for i in range(3):
        Intersection = HC4(functions, xNewBounds, bxrd_options)
        if Intersection.is_empty(): return False 
        else:
            xNewBounds = [mpmath.mpi(Intersection[j][0],Intersection[j][1]) 
                          for j in range(0, len(xBounds))] 
 
    return True


def solutionInFunctionRangeNewton(model, xBounds, bxrd_options):
    """checks, if box is empty by reducing it three times with HC4 method
    Args: 
        :model:             instance of class-Model
        :xBounds:           current Bounds of Box
        :bxrd_options:      options with absTolerance for deviation from the 
                            solution
        
    Returns: 
        boolean that is True if solution is in function range and False otherwise
        
    """
    xOld = [list(xBounds)]
    
    for i in range(3):
        xNewBounds = []
        for x in xOld:
            output = reduceBox(numpy.array(x), model, 4, bxrd_options)
            if not output["xNewBounds"] == []: xNewBounds += output["xNewBounds"]     
        if xNewBounds == []: return False
        
        if xOld == xNewBounds: break       
        else: xOld = xNewBounds
            
    return True


def HC4_float(functions, box, bxrd_options):
    """reduces the bounds of all variables in every model function based on HC4 
    hull-consistency
    
    Args:
        :functions:     list with function instances
        :box:           current bounds of box as nested lists in float formate
        :bxrd_options:  dictionary with tolerances
        
    Returns: 
        :pyibex IntervalVector with reduced bounds 
        
    """  
    x_HC4 = []
    unique_x = [False]*len(box)
    tol = bxrd_options["absTol"]
    for i, iv in enumerate(box):
        if (iv[1] - iv[0]) < tol: 
            x_HC4.append([iv[0]-0.1*tol, iv[1]+0.1*tol])
        else:x_HC4.append(list(iv))
        
    box_old = list(x_HC4)
    
    for f in functions:
        sub_box = pyibex.IntervalVector([x_HC4[i] for i in f.glb_ID])   

        currentIntervalVector = pyibex.IntervalVector(sub_box)
        f.f_pibex.contract(currentIntervalVector)

        if (bxrd_options.__contains__("unique_hc") and 
            not all(unique_x)):
            sub_box_old = [box_old[i] for i in f.glb_ID]             
            checkUniqueness_HC4(currentIntervalVector,sub_box_old, 
                                         f.glb_ID,
                                         unique_x,
                                         bxrd_options["relTol"],
                                         bxrd_options["absTol"])
            
        sub_box = sub_box & currentIntervalVector

        if sub_box.is_empty():           
            bxrd_options["failed_subbox"] = pyibex.IntervalVector([x_HC4[i] for 
                                                                   i in f.glb_ID])
            bxrd_options["failed_function"] = f
            bxrd_options["unique_hc"] = all(unique_x)
            return sub_box
        else:
            for i,val in enumerate(f.glb_ID): x_HC4[val] =  list(sub_box[i])
    bxrd_options["unique_hc"] = all(unique_x)
    return pyibex.IntervalVector(x_HC4)


def HC4(functions, xBounds, bxrd_options):
    """reduces the bounds of all variables in every model function based on HC4 
    hull-consistency
    
    Args:
        :functions:     list with function instances
        :xBounds:       current Bounds of Box
        :bxrd_options:  dictionary with tolerances
        
    Returns: 
        :pyibex IntervalVector with reduced bounds 
        
    """  
    x_HC4 = []
    unique_x = [False]*len(xBounds)
    
    for x in xBounds:
        if x.delta < bxrd_options["absTol"]: tol = bxrd_options["absTol"]
        else: tol = 0.0
               
        x_HC4.append([float(mpmath.mpf(x.a))-0.1*tol, 
                    float(mpmath.mpf(x.b))+0.1*tol])  #keep Bounds in max tolerance to prevent rounding error

    for f in functions:
        sub_box = pyibex.IntervalVector([x_HC4[i] for i in f.glb_ID])
        
        currentIntervalVector = pyibex.IntervalVector(sub_box)
        f.f_pibex.contract(currentIntervalVector)
        #cur_IV_mpi = [mpmath.mpi(iv) for iv in currentIntervalVector]
        if (bxrd_options.__contains__("unique_hc") and 
            bxrd_options.__contains__("x_old") and not all(unique_x)):
            sub_box_old = [convertIntervalBoundsToFloatValues(
                bxrd_options["x_old"][i]) for i in f.glb_ID]
                
            checkUniqueness_HC4(currentIntervalVector,sub_box_old, 
                                         f.glb_ID,
                                         unique_x,
                                         bxrd_options["relTol"],
                                         bxrd_options["absTol"])
  
        sub_box = sub_box & currentIntervalVector
        if sub_box.is_empty(): # TODO: store currentIntervalVector and f in bxrd_options for error analysis
            bxrd_options["failed_subbox"] = pyibex.IntervalVector([x_HC4[i] for 
                                                                   i in f.glb_ID])
            bxrd_options["failed_function"] = f
            bxrd_options["unique_hc"] = all(unique_x)
            return sub_box
        else:
            for i,val in enumerate(f.glb_ID): x_HC4[val] =  list(sub_box[i])
    bxrd_options["unique_hc"] = all(unique_x)
    return pyibex.IntervalVector(x_HC4)


def checkUniqueness_HC4(new_x, old_x, glb_ID,  unique_x, relEpsX, absEpsX):
    """ checks if the condition for a unique solution in a box is fulfilled which
    states that the new_x must be a in the interior of old_x. This 
    has to be met by all x of new_x. 
    
    Args:
        :new_x:      list with subbox in mpmath.mpi formate after contraction
        :old_x:      list with subbox in mpmath.mpi formate before contraction
        :glb_ID:     list with global indices of subbox variables
        :unique_x:   list with booleans that are true if new interval is in 
                     the interior of the old one or solved and false otherwise
        :relEpsX:    relative tolerance
        :absEpsX:    absolute tolerance
        
    Returns: 
        :-: 
        
    """     
    for i,x in enumerate(new_x):
        if x[0] > old_x[i][0] and x[1] < old_x[i][1] and not unique_x[glb_ID[i]]:
            unique_x[glb_ID[i]] = True
        elif isclose_ordered(x[0], x[1], relEpsX,absEpsX) and not unique_x[glb_ID[i]]:
            unique_x[glb_ID[i]] = True
    return True

        
def checkUniqueness(new_x, old_x,relEpsX,absEpsX):
    """ checks if the condition for a unique solution in a box is fulfilled which
    states that the new_x must be a subinterval of the interior of old_x. This 
    has to be met by all x of a box. 
    
    Args:
        :new_x:     list with intervals in the formate mpmath.mpi
        :old_x:     interval in the formate mpmath.mpi
        :relEpsX:   relative tolerance
        :absEpsX:   absolute tolerance
    
    Returns: 
        boolean true if the criterion is fulfilled and false otherwise
    
    """
    for i,x in enumerate(new_x):
        if x.a > old_x.a and x.b < old_x.b:
            continue
        elif isclose_ordered(float(mpmath.mpf(x.a)), float(mpmath.mpf(x.b)), 
                           relEpsX, absEpsX):
            continue
        else: return False
    return True


def lookForSolutionInBox(model, boxID, bxrd_options, sampling_options, solv_options):
    """Uses Matlab File and tries to find Solution with initial points in the box samples by HSS.
     Writes Results in File, if one is found: 
    
    Args: 
        :model:            instance of class model
        :boxID:            id of current Box
        :bxrd_options:     dictionary of options  
        :sampling_options: dictionary with sampling settings
        :solv_options:     dicionary with settings for numerical solver
        
    Returns:
        :solved:           boolean that is true for successful iteration
                                     
    """
    if bxrd_options["debugMode"]: print("Box no. ", boxID, 
                                          "is now numerically iterated.")
    
    if solv_options.__contains__("scaling"): 
        bxrd_options["scaling"] = solv_options["scaling"]
    else: bxrd_options["scaling"] = "None"
    if solv_options.__contains__("scalingProcedure"): 
        bxrd_options["scalingProcedure"] = solv_options["scalingProcedure"]
    else: bxrd_options["scalingProcedure"] = "None"
    solved = False
    allBoxes = list(model.xBounds)
    model.xBounds = ConvertMpiBoundsToList(model.xBounds,boxID)
    if sampling_options == None: 
        sampling_options = {"smplNo": 0, "smplBest": 1,
                            "smplMethod": "None"}
    bxrd_options["sampling"] =True
    
    results = mos.solveBlocksSequence(model, solv_options, bxrd_options, 
                                      sampling_options)
   #print("Condition met:",results != {} and not results["Model"].failed)
    if (results != {} and not results["Model"].failed):# and solution_solved_in_tolerance(results["Model"],solv_options) !=[]):
        solved = True 
        if not "FoundSolutions" in bxrd_options.keys():  
            bxrd_options["FoundSolutions"] = copy.deepcopy(results["Model"].stateVarValues)
            solv_options["sol_id"] = 1
            mos.results.write_successful_results({0: results}, bxrd_options, 
                                                 sampling_options, solv_options) 
        else:    
            for new_solution in copy.deepcopy(results["Model"].stateVarValues):
                sol_exist = False
                for solution in bxrd_options["FoundSolutions"]: 
                    if numpy.allclose(numpy.array(new_solution), 
                                      numpy.array(solution),
                                      bxrd_options["relTol"],
                                      bxrd_options["absTol"]):
                        sol_exist = True
                        break
                if not sol_exist: 
                    bxrd_options["FoundSolutions"].append(new_solution)  
                    if not solv_options.__contains__("sol_id"): 
                        solv_options["sol_id"]=len(bxrd_options["FoundSolutions"])
                    else: solv_options["sol_id"] += 1
                    mos.results.write_successful_results({0: results}, bxrd_options, 
                                                 sampling_options, solv_options) 
             
    if model.failed: 
        model.failed = False
    #else: solved = True 
    model.xBounds = allBoxes
    return solved


def solution_solved_in_tolerance(model,solv_options):
    old_solutions = list(model.stateVarValues)
    solutions = []
    for solution in old_solutions:
         model.stateVarValues = [solution]
         if not (numpy.linalg.norm(model.getScaledFunctionValues()) <= 
                    solv_options["FTOL"]):
                continue
         else:
             solutions += [solution]
    return solutions
  
def ConvertMpiBoundsToList(xBounds, boxID):
    """Converts the xBounds, containing mpi to a list for sampling method
    
    Args: 
        :xBounds:   array of bounds as mpmath.mpi
        :boxID:   id of current Box      
        
    Returns:
        :[xBoxBounds.astype(numpy.float)]: Bounds in List format    
            
    """
    xBoxBounds = numpy.zeros((len(xBounds[boxID]),2))

    for x in range(len(xBounds[boxID])):
        xBoxBounds[x][0] = float(mpmath.convert(xBounds[boxID][x].a))
        xBoxBounds[x][1] = float(mpmath.convert(xBounds[boxID][x].b))
            
    return [xBoxBounds.astype(numpy.float)]


def remove_zero_and_max_value_bounds(x):
    """ removes zeros and +/- inf values from variable domain by very small/
    large value
    
    Args:
        :x:     interval in mpmath.mpi formate
        
    Returns:
        :x:     zero- or inf- free x-interval

    """    
    if x.a > numpy.nan_to_num(numpy.inf) : return x
    if x.b < -numpy.nan_to_num(numpy.inf) : return x
    if (0.0 <= x.a < 1.0/numpy.nan_to_num(numpy.inf) and 
        x.b < 1.0/numpy.nan_to_num(numpy.inf)): return x
    if (0.0 >= x.b > -1.0/numpy.nan_to_num(numpy.inf) and 
        x.a > -1.0/numpy.nan_to_num(numpy.inf)): return x
    if (0.0 <= x.a < 1.0/numpy.nan_to_num(numpy.inf) and 
        x.b < 1.0/numpy.nan_to_num(numpy.inf)): return x
    elif (0.0 > x.a > -1.0/numpy.nan_to_num(numpy.inf)): 
        x = mpmath.mpi(-1/numpy.nan_to_num(numpy.inf), x.b)
    if (0.0 <= x.b < 1.0/numpy.nan_to_num(numpy.inf)): 
        x = mpmath.mpi(x.a, 1/numpy.nan_to_num(numpy.inf))
    elif (0.0 > x.b > -1.0/numpy.nan_to_num(numpy.inf)): 
        x = mpmath.mpi(x.a, -1/numpy.nan_to_num(numpy.inf))
    if x.b > numpy.nan_to_num(numpy.inf) : 
        x = mpmath.mpi(x.a, numpy.nan_to_num(numpy.inf))
    if x.a < -numpy.nan_to_num(numpy.inf) : 
        x = mpmath.mpi(-numpy.nan_to_num(numpy.inf),x.b)
    return x


def saveSolutions(model, bxrd_options):
    """Writes all found results to a txt file 
    
        Args: 
            :model:   instance of class model
            :bxrd_options:   dictionary of options      
            
        Returns:
             
        """
    file=open(f'{bxrd_options["fileName"]}{len(model.FoundSolutions)}Solutions.txt', 
              'a')
    for fs in model.FoundSolutions:
        for var in range(len(fs)):
            file.write(str(model.xSymbolic[var])+' '+str(fs[var]))
            file.write('  \n')
        file.write('\n')
    file.close()

    
def split_least_changed_variable(box_new, model, k, bxrd_options):
    """ splits interval of the variable that has least changed to the complete 
    interval before the last split. This is measured by the width ratio w. If 
    the maximum w is the same for multiple variables (e.g. w =1 no change at all)
    at most three of them are tested for the best split with the forecastSplit 
    method. If there are more than three intervals the variables with the least 
    change are sorted by there occurence in different equations. Only the three 
    with the maximum occurences are forwared to the forcastSplit method.
    
    Args:
        :box_new:           list with box that shall be splitted as numpy.array
        :model:             instance of type model
        :k:                 id of current box
        :bxrd_options:      dictionary with user-specified reduction settings
    
    Returns:
        :w_max_ids:         list with global indices of least changed variables
               
    """
    w_ratio = []
    r = model.complete_parent_boxes[k][0]
    box_ID = model.complete_parent_boxes[k][1]
    box_new_float=[[convert_mpi_float(iv.a), convert_mpi_float(iv.b)] 
                   for iv in box_new[0]]  
    box_old = mostg.get_entry_from_npz_dict(bxrd_options["fileName"]+"_boxes.npz", 
                                            r, allow_pickle=True)[box_ID]
    
    w_ratio = analysis.identify_interval_reduction(box_new_float, box_old)
    w_remove = [ i for i, j in enumerate(w_ratio) if(j == 1
                                                     and variableSolved([box_new[0][i]],
                                                                            bxrd_options
                                                                            )
                                                     )]
    w_ratio = [i for i,j in enumerate(w_ratio) if not i in w_remove]
    w_max_ids = [i for i, j in enumerate(w_ratio) if j == max(w_ratio)]
    
    return w_max_ids


def get_index_of_boxes_for_reduction(xSolved, cut, maxBoxNo):
    """ creates list for all boxes that can still be reduced
    Args:
        :cut:           list with boolean that is true for incomplete boxes
        :maxBoxNo:      integer with current maximum number of boxes
        
    Returns:
        :ready_for_reduction:   list with boolean that is true if box is ready 
                                for reduction
                                
    """
    complete = []
    incomplete = []
    solved =[]
    not_solved_but_complete=[]
    ready_for_reduction = len(cut) * [False]
    
    for i, val in enumerate(xSolved):
        if val: solved +=[i]
    
    for i,val in enumerate(cut):
        if not val: 
            complete += [i]
            if not i in solved: not_solved_but_complete +=[i] 
        else: incomplete += [i]
   
    nl = max(0, min(len(not_solved_but_complete), maxBoxNo - len(cut)))
    
    for i in range(nl): ready_for_reduction[not_solved_but_complete[i]] = True
    for i in incomplete: ready_for_reduction[i] = True
    
    return ready_for_reduction


def unify_boxes(boxes, bxrd_options):
    """ checks boxes if they are neighbors, i.e. differ only in one variable 
    interval and the respecitve intervals share one bound but are disjoint 
    otherwise through rounding errors a slight intersection of the boxes is
    also allowed, therefore relative and absolute tolerances are used. Next
    to the unification an alternative tolerance is calcluated. This is useful
    if they all correspond to one solution interval. To keep a low number of
    solution intervals.
    
    Args:
        :boxes:         list with boxes in mpmath.mpi formate
        :bxrd_options:  dictionary with relative and absolute tolerances
    
    Returns:
        :boxes:         list with unified boxes
        :results:       stores ids of unified boxes and the variable index plus
                        the tolerance in that the unified box fulfills.

    """   
    results = {
        "boxes_unified": len(boxes) * [False],
        "epsilon_uni": [],
        "var_id": []}
    
    boxes_to_check = list(boxes) 
    len_old = len(boxes)
    
    while boxes_to_check:   
        for k,box in enumerate(boxes_to_check):
            for j,box_2 in enumerate(boxes):
                if list(box) == list(box_2): continue
            
                elif all([box[i] in box_2[i] for i in range(len(box))]):
                    continue
                else:
                    identical=[box[i]==box_2[i] for i in range(len(box))]
                    if identical.count(False)==1:
                        i = identical.index(False)
                        if box[i].a in mpmath.mpi(box_2[i].b - bxrd_options["absTol"],
                                                  box_2[i].b): 
                            index = [l for l, cur_box in enumerate(boxes) 
                                     if list(cur_box) == list(box)]
                            boxes[j][i]=mpmath.mpi(box_2[i].a, box[i].b)
                            results["boxes_unified"][j] = True
                            if index: results["boxes_unified"].pop(index[0])
                            if index: boxes.pop(index[0])
  
                        elif box[i].b in mpmath.mpi(box_2[i].a - bxrd_options["absTol"], 
                                                    box_2[i].a):
                            index = [l for l, cur_box in enumerate(boxes) 
                                     if list(cur_box) == list(box)]
                            boxes[j][i]=mpmath.mpi(box[i].a, box_2[i].b) 
                            results["boxes_unified"][j] = True
                            if index: results["boxes_unified"].pop(index[0])
                            if index: boxes.pop(index[0])
                         
        if len(boxes) != len_old: 
            boxes_to_check = list(boxes)
            len_old = len(boxes)
        else: boxes_to_check = []
    unified_boxes = [boxes[i] for i, box_unified in 
                     enumerate(results["boxes_unified"]) if box_unified]    
    for box in unified_boxes:
        epsilon_uni = [iv.delta/(1+abs(iv).b) for iv in box]
        results["epsilon_uni"].append(convert_mpi_float(max(epsilon_uni).b))
        results["var_id"].append(epsilon_uni.index(max(epsilon_uni)))
      
    return boxes, results              


def check_box_for_disconti_iv(model, new_x, bxrd_options):
    for f in model.functions:
        box = [new_x[i] for i in f.glb_ID] 
        if variableSolved(box, bxrd_options):
            f_iv = f.f_mpmath[0](*box)
            if (f_iv.a < -1.0e15  or f_iv.b > 1.0e15):
                return []
        else: continue
    return new_x
    

        