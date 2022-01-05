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
from modOpt.constraints import affineArithmetic
from modOpt.constraints import parallelization
#from modOpt.constraints import function
from modOpt.constraints.FailedSystem import FailedSystem
from modOpt.decomposition import MC33
from modOpt.decomposition import dM
import modOpt.solver as mos
import modOpt.constraints.realIvPowerfunction # redefines __power__ (**) for ivmpf
import modOpt.storage as mostg


__all__ = ['reduceBoxes', 'reduceXIntervalByFunction', 'setOfIvSetIntersection',
           'checkWidths', 'getPrecision', 'saveFailedSystem', 
            'variableSolved', 'contractBox', 'reduceConsistentBox','updateSetOfBoxes',
            'doHC4', 'checkIntervalAccuracy', 'do_bnormal', 'roundValue']

"""
***************************************************
Algorithm for interval Nesting procedure
***************************************************
"""

def reduceBoxes(model, dict_options, sampling_options=None, solv_options=None):
    """ reduction of multiple boxes
    Args:    
        :model:                 object of type model   
        :dict_options:          dictionary with user specified algorithm settings
        :sampling_options:      dictionary with sampling settings
        :solv_options:          dicionary with settings for numerical solver

    Return:
        :results:               dictionary with newXBounds and a list of booleans
                                that marks all non reducable variable bounds with True.
                                If solver terminates because of a NoSolution case the
                                critical equation is also stored in results for the error
                                analysis.

    """
    if (dict_options["cut_Box"] in {"all", "tear", True}): model.cut = True
    model.interval_jac = None  
    model.jac_center = None
    results = {}
    results["num_solved"] = False
    results["disconti"] = []
    results["complete_parent_boxes"] = []
    allBoxes = []
    newtonMethods = {'newton', 'detNewton', '3PNewton','mJNewton'}

    xAlmostEqual = dict_options["xAlmostEqual"]
    disconti = dict_options["disconti"]
    xSolved = dict_options["xSolved"]

    cut = []
    dict_options["boxNo"] = len(model.xBounds)
    #boxNo = len(model.xBounds)
    nl = len(model.xBounds)
    dict_options["ready_for_reduction"] = get_index_of_boxes_for_reduction(xSolved, 
                                                                           xAlmostEqual, 
                                                                           dict_options["maxBoxNo"]  )

    for k in range(nl):
        xBounds = model.xBounds[k]
        if dict_options["Debug-Modus"]: print("Current box index: ", k)

        if xSolved[k]: 
            output = {}
            xSolved[k] = [True]
            xAlmostEqual[k] = [True]
            allBoxes.append(xBounds)
            results["disconti"] += [disconti[k]]
            results["complete_parent_boxes"]  += [model.complete_parent_boxes[k]]
            cut += [False]
            continue    
        
        elif xAlmostEqual[k] and not disconti[k]:
            output = {"xNewBounds": [xBounds],
                      "xAlmostEqual": [xAlmostEqual[k]],
                      "xSolved": [xSolved[k]],                      
                }
            output = reduceConsistentBox(output, model, dict_options, k, 
                                         dict_options["boxNo"],
                                         newtonMethods)  
            cut += [output["cut"]]
            if model.teared: 
                results["complete_parent_boxes"] += (len(output["xNewBounds"]) 
                                                     * [[dict_options["iterNo"]-1, k]])
                model.teared = False
            else: 
                results["complete_parent_boxes"] += len(output["xNewBounds"])*[model.complete_parent_boxes[k]]
                
        elif xAlmostEqual[k] and disconti[k] and dict_options["boxNo"]  >= dict_options["maxBoxNo"]:
            xSolved[k] = [False]
            xAlmostEqual[k] = [True]
            allBoxes.append(xBounds)
            results["disconti"] += [disconti[k]] 
            results["complete_parent_boxes"]  += [model.complete_parent_boxes[k]]
            cut += [False]
            continue

        else:
            if dict_options["Debug-Modus"]: print(f'Box {k}')
            if not xAlmostEqual[k] and disconti[k] and dict_options["consider_disconti"]: 
                store_boxNo = dict_options["boxNo"]
                
                dict_options["boxNo"] = dict_options["maxBoxNo"]
                output = contractBox(xBounds, model, dict_options["boxNo"] , 
                                     dict_options)     
                dict_options["boxNo"]  = store_boxNo             
            else:                                
                output = contractBox(xBounds, model, dict_options["boxNo"] , 
                                     dict_options)
                
            results["complete_parent_boxes"] += (len(output["xNewBounds"]) * 
                                                 [ model.complete_parent_boxes[k]])
            if (output["xSolved"][0] and len(output["xNewBounds"]) == 1 
                and dict_options["hybrid_approach"]):
                if not "FoundSolutions" in dict_options.keys():  
                     model.xBounds[k] = output["xNewBounds"][0]
                     num_solved = lookForSolutionInBox(model, k, dict_options, 
                                                      sampling_options, solv_options)
                else: num_solved = True
                if num_solved:    
                    if not test_for_root_inclusion(output["xNewBounds"][0], 
                                            dict_options["FoundSolutions"], 
                                            dict_options["absTol"]):
                        num_solved = False
                #if not num_solved: 
                #    output["xSolved"][0] = False
                                                                                                              
            elif (all(output["xAlmostEqual"]) and not all(output["xSolved"]) 
                  and dict_options["hybrid_approach"]):  

                if not sampling_options ==None and not solv_options == None:
                    num_solved = lookForSolutionInBox(model, k, dict_options, 
                                                      sampling_options, solv_options)
                    if num_solved: results["num_solved"] = True
                                
        if output.__contains__("noSolution") :
            saveFailedIntervalSet = output["noSolution"]
            dict_options["boxNo"] = len(allBoxes) + (nl - (k+1))
            xAlmostEqual[k]=[]
            xSolved[k]=[]
            continue
        
        if output.__contains__("uniqueSolutionInBox"):
            xSolved[k] = output["xSolved"]
    
        dict_options["boxNo"] = (len(allBoxes) + len(output["xNewBounds"]) + 
                                 (nl - (k+1)))          
        updateSetOfBoxes(model, allBoxes, xBounds, output, dict_options["boxNo"], 
                         k, dict_options, results)
        xAlmostEqual[k] = output["xAlmostEqual"]
        xSolved[k] = output["xSolved"]
        
    if cut != []: model.cut = any(cut)            
    if allBoxes == [] and output.__contains__("uniqueSolutionInBox"):
        print("All solutions have been found. The last boxes before proof by",
              " Interval-Newton are returned.")
        xSolved = numpy.array([True])
    elif allBoxes == []: results["noSolution"] = saveFailedIntervalSet
      
    else:
        model.xBounds = allBoxes       
    results["xAlmostEqual"] = xAlmostEqual
    results["xSolved"] = xSolved
    
    return results


def roundValue(val, digits):
    """ generates tightest interval around value val in accuracy of its last digit
    so that its actual value is not lost because of round off errors

    Args:
        :val:         sympy.Float value
        :digit:       integer number of digits

    Return: tightest interval in mpmath.mpi formate
    

    """
    rounded_val = round(val, digits)
    if rounded_val == val:
        return mpmath.mpi(val)
    elif rounded_val > val:
        return mpmath.mpi(rounded_val - 10**(-digits), rounded_val)
    
    return mpmath.mpi(rounded_val, rounded_val + 10**(-digits))


def updateSetOfBoxes(model, allBoxes, xBounds, output, boxNo, k, dict_options, results):
    """ updates set of boxes with reduced boxes from current step. If the maximum
    number of boxes is exceeded the former will be put into the set instead.
    
    Args:
    :model:         instance of type Model
    :allBoxes:      list with already reduced boxes
    :xBounds:       numpy.array with former box
    :output:        dictionary with new box(es) and xAlmostEqual check for box
    :boxNo:         integer with current number of boxes (including new boxes)
    :k:             inter with index of former box
    :dict_options:  dictionary with user-specified maximum number of boxes
    :results:       dictionary for storage of results

    """    
    if boxNo <= dict_options["maxBoxNo"]:
        for box in output["xNewBounds"]: allBoxes.append(numpy.array(box, 
                                                                     dtype=object))
        results["disconti"] += len(output["xNewBounds"])*[output["disconti"]]
      
 
    else:# boxNo > dict_options["maxBoxNo"]:
        print("Warning: Algorithm stops the current box reduction because the current number of boxes is ",
              boxNo, "and exceeds the maximum number of boxes that is ",
              dict_options["maxBoxNo"], "." )
        allBoxes.append(xBounds)
        output["xAlmostEqual"] = True 
        results["disconti"] = output["disconti"]


def contractBox(xBounds, model, boxNo, dict_options):
    """ general contraction step that contains newton, HC4 and box reduction method    

    with and without parallelization. The combined algorithm is there for finding 
    an "efficient" alternation strategy between the contraction step methods
    
    Args:
    :xBounds:               numpy.array with current box
    :model:                 instance of class model
                            with function's glb id they appear in
    :boxNo:                 current number of boxes
    :dict_options:          dictionary with user specified algorithm settings
    
    Return:
        :output:            dictionary with results
    
    """    
    if not dict_options["Parallel Variables"]:
        output = reduceBox(xBounds, model, boxNo, dict_options)
    else:
        output = parallelization.reduceBox(xBounds, model, boxNo, dict_options)

    return output


def reduceConsistentBox(output, model, dict_options, k, boxNo, 
                        newtonMethods):
    """ reduces a consistent box after the contraction step
    
    Args:
        :output:             dictionary quantities from consistent box
        :model:              instance of class Model
        :dict_options:       dictionary with user-specifications
        :k:                  index of currently reduced box
        :boxNo:              integer with number of boxes
        :newtonMethods:     dictionary with netwon method names
        
    Return:
        :output:             modified dictionary with new split or cut box(es)

    """
    newBox = output["xNewBounds"]
    possibleCutOffs = False
       
    if dict_options["cut_Box"] in {"tear", "all", True}: # if cut_Box is chosen,parts of the box are now tried to cut off 
        if dict_options["Debug-Modus"]: print("Now box ", k, "is cutted")
        if dict_options["cut_Box"] == "tear": 
            newBox, possibleCutOffs = cutOffBox_tear(model, newBox, dict_options)
        if dict_options["cut_Box"] == "all" or dict_options["cut_Box"] == True : 
            newBox, possibleCutOffs = cutOffBox(model, newBox, dict_options)
        if newBox == []: possibleCutOffs = False     
        if possibleCutOffs:  # if cut_Box was successful,the box is now tried to be reduced again
            output = contractBox(numpy.array(newBox[0]), model, boxNo, 
                                 dict_options)           
            newBox = output["xNewBounds"]
            
        if newBox == [] or newBox ==[[]]: 
            saveFailedSystem(output, model.functions[0], model, 0)
            output["disconti"]=[]
            output["cut"] = True  
            return output
                         
        if output["xAlmostEqual"][0] and dict_options["Debug-Modus"]: 
            print("box ", k, " is still complete.")
    output["cut"] = True  

    if ((not possibleCutOffs or output["xAlmostEqual"][0]) and 
        (not "uniqueSolutionInBox" in output.keys())):  # if cut_Box was not successful or it didn't help to reduce the box, then the box is now splitted      
        output["cut"] = False
        if "ready_for_reduction" in dict_options.keys(): 
           if not dict_options["ready_for_reduction"][k]: 
               output["disconti"] = False 
               return output
        boxNo_split = dict_options["maxBoxNo"] - boxNo
        
        if boxNo_split > 0:
            if dict_options["Debug-Modus"]: print("Now box", k, " is teared")
            model.teared = True
            output["xNewBounds"] = splitBox(newBox, model, dict_options, k, 
                                            boxNo_split)
            if output["xNewBounds"] == []:
                saveFailedSystem(output, model.functions[0], model, 0)
            output["xAlmostEqual"] = [False] * len(output["xNewBounds"])   
            output["xSolved"] = [False] * len(output["xNewBounds"]) 
            output["cut"] = True
            
        else:
           output["xAlmostEqual"] =  [True]
    output["disconti"] = False
        
    return output


def checkBoxesForRootInclusion(functions, boxes, dict_options):
    """ checks if boxes are non-empty for EQS given by functions
    
    Args:
        :functions:        list with function objects
        :boxes:            list of boxes that are checked in mpmath.mpi formate
        :dict_options:     dictionary with box redcution settings
        
    Return:
        :nonEmptyboxes:    list with non-empty boxes with mpmath.mpi intervals

    """
    if boxes == []: return []
    nonEmptyboxes = [box for box in boxes if solutionInFunctionRange(functions, 
                                                                     box, dict_options)]

    return nonEmptyboxes


def splitBox(consistentBox, model, dict_options, k, boxNo_split):
    """ box splitting algorithm should be invoked if contraction doesn't work anymore because of consistency.
    The user-specified splitting method is executed.
    
    Args:
       :xNewBounds:         numpy.array with consistent box
       :model:              instance of class model
       :dict_options:       dictionary with user-specifications
       :k:                  index of currently reduced box
       :boxNo_split:        number of possible splits (maxBoxNo-boxNo) could be 
                            used for multisection too
    
    Return:
        list with split boxes
        
    """
    if dict_options["split_Box"]=="TearVar": 
        # splits box by tear variables  
        if model.tearVarsID == []: getTearVariables(model)  
        
        xNewBounds, dict_options["tear_id"] = splitTearVars(model.tearVarsID, 
                           numpy.array(consistentBox[0]), boxNo_split, 
                           dict_options)
        if numpy.isclose(float(mpmath.mpf(xNewBounds[0][model.tearVarsID[dict_options["tear_id"]-1]].a)),
                           float(mpmath.mpf(xNewBounds[0][model.tearVarsID[dict_options["tear_id"]-1]].b)), 
                           dict_options["absTol"]):
            dict_options["split_Box"] ="forecastSplit"
            xNewBounds = getBestSplit(consistentBox, model, k, dict_options)

    elif dict_options["split_Box"]=="LargestDer":  
        #splits box by largest derivative
        splitVar = getTearVariableLargestDerivative(model, k)
        xNewBounds, dict_options["tear_id"] = splitTearVars(splitVar, 
                           numpy.array(consistentBox[0]), boxNo_split, 
                           dict_options)
    elif dict_options["split_Box"]=="LeastChanged":  
        xNewBounds = split_least_changed_variable(consistentBox, model, k, 
                                                  dict_options)
                
    elif (dict_options["split_Box"]=="forecastSplit" or 
          dict_options["split_Box"]=="forecast_HC4" or 
          dict_options["split_Box"]=="forecast_newton"):  
        # splits box by best variable
        xNewBounds = getBestSplit(consistentBox, model, k, dict_options)
    
    elif dict_options["split_Box"]=="forecastTear":
        if model.tearVarsID == []: getTearVariables(model)
        xNewBounds = getBestTearSplit(consistentBox, model, k, dict_options)

    
    xNewBounds = checkBoxesForRootInclusion(model.functions, xNewBounds, 
                                            dict_options)
    return xNewBounds


def cutOffBox_tear(model, xBounds, dict_options):
    """ trys to cut off all empty sides of the box spanned by the tear variables
    to reduce the box without splitting-

    Args:
        :model:         instance of type model
        :xBounds:       current boxbounds in iv.mpmath
        :dict_options:  dictionary of options
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
            while i<100: #number of cutt offs are limited to 100 due to step sie of 0.01*delta
                CutBoxBounds = list(xNewBounds)
                xu = CutBoxBounds[u]
                if (mpmath.mpf(xu.delta) <= 
                    mpmath.mpf(xBounds[0][u].delta)*0.02*i): break #if total box is to small for further cutt offs
                cur_x = (float(mpmath.mpf(xu.b)) - 
                         float(mpmath.mpf(xBounds[0][u].delta)*0.01*i))
                CutBoxBounds[u] = mpmath.mpi(cur_x, xu.b) #define small box to cut
                
                if not solutionInFunctionRangePyibex(model.functions, numpy.array(CutBoxBounds), dict_options): #check,if small box is empty
                #if not solutionInFunctionRange(functions, numpy.array(CutBoxBounds), dict_options):
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
                                                     dict_options): #check,if small box is empty
                    xNewBounds[u] = mpmath.mpi(cur_x, xu.b)
                    cutOff = True
                    i=i+1
                    continue
                else:
                    break
            if not xChanged[model.tearVarsID.index(u)] and not i ==1: 
                xChanged[model.tearVarsID.index(u)]=True
                
    return [list(xNewBounds)], cutOff


def cutOffBox(model, xBounds, dict_options):
    '''trys to cut off all empty sides of the box, to reduce the box without splitting

    Args:
        :model:         instance of type model
        :xBounds:       current boxbounds in iv.mpmath
        :dict_options:  dictionary of options
        
    Return:
        :xNewBounds:    new xBounds with cut off sides
        :cutOff:        boolean if any cut offs are possible
        
    '''
    xNewBounds = list(list(xBounds[0]))
    
    cutOff=False
    xChanged = numpy.array([True]*len(model.xSymbolic))
    while xChanged.any(): 
        for u in range(len(model.xSymbolic)):
            #try to cut off upper variable parts
            i=1
            while i<100: #number of cutt offs are limited to 100
                CutBoxBounds = list(list(xNewBounds))
                xu = CutBoxBounds[u]
                if (mpmath.mpf(xu.delta) <= 
                    mpmath.mpf(xBounds[0][u].delta)*0.02*i): break #if total bounds is too small for further cut offs
                cur_x = (float(mpmath.mpf(xu.b)) - 
                         float(mpmath.mpf(xBounds[0][u].delta)*0.01*i))
                CutBoxBounds[u] = mpmath.mpi(cur_x, xu.b) #define small box to cut
            
                if not solutionInFunctionRangePyibex(model.functions, 
                                                     numpy.array(CutBoxBounds), 
                                                     dict_options): #check,if small box is empty
                    xNewBounds[u] = mpmath.mpi(xu.a, cur_x)
                    cutOff = True
                    i=i+1
                    continue
                else:
                    break
            if i == 1: xChanged[u] = False
            #try to cut off lower part
            i=1
            while i<100:
                CutBoxBounds = list(list(xNewBounds))
                xu = CutBoxBounds[u]
                if (mpmath.mpf(xu.delta) <= 
                    mpmath.mpf(xBounds[0][u].delta)*0.02*i): break
                cur_x = (float(mpmath.mpf(xu.a)) + 
                         float(mpmath.mpf(xBounds[0][u].delta)*0.01*i))
                CutBoxBounds[u] = mpmath.mpi(xu.a, cur_x)
                
                if not solutionInFunctionRangePyibex(model.functions, 
                                                     numpy.array(CutBoxBounds), 
                                                     dict_options):
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
    # TODO: For blocks
    model.jacobian = dM.getCasadiJandF(model.xSymbolic, model.fSymbolic)[0]
    jacobian = model.getCasadiJacobian()
    res_permutation = MC33.doMC33(jacobian)  
    tearsCount = max(res_permutation["Border Width"],1)
    model.tearVarsID =res_permutation["Column Permutation"][-tearsCount:]  


def getTearVariableLargestDerivative(model, boxNo):
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


def getBestTearSplit(xBounds,model, boxNo, dict_options, w_max_ids=None):
    '''finds variable, which splitting causes the best reduction

    Args:
        :xBounds:           variable bounds of class momath.iv
        :model:             instance of type model
        :boxNo:             integer with number of boxes
        :dict_options:      dictionary of options
        :w_max_ids:         ids of intervals with maximum widths
        
    Return:
        :xNewBounds:    best reduced two variable boxes
        
    '''
    oldBounds = list(numpy.array(xBounds)[0])  
    smallestAvrSide = numpy.Inf
    if w_max_ids: a = w_max_ids
    else: a = model.tearVarsID
    
    #try all splits
    for i, value in enumerate(a):
        BoundsToSplit = list(numpy.array(xBounds)[0])
        splittedBox = separateBox(BoundsToSplit, [value])
        
        #reduce both boxes

        output0, output1 = reduceHC4_orNewton(splittedBox, model, boxNo, 
                                              dict_options)

        if output0["xNewBounds"] != [] and output0["xNewBounds"] != [[]]:
            avrSide0 = identifyReduction(output0["xNewBounds"], oldBounds)
        else:
            #if one of splitted boxes is empty always prefer this split
            print("This is the current best splitted varID by an empty box ", 
                  model.xSymbolic[value])
            return [tuple(splittedBox[1])]
        
        if output1["xNewBounds"] != [] and output1["xNewBounds"] != [[]]:
            avrSide1 = identifyReduction(output1["xNewBounds"], oldBounds)
        else:
            #if one of splitted boxes is empty always prefer this split
            print("This is the current best splitted varID by an empty box ", 
                  model.xSymbolic[value])
            return [tuple(splittedBox[0])]
        
        # sum of both boxreductions
        avrSide = avrSide0 + avrSide1
        # find best overall boxredution
        if avrSide<smallestAvrSide:
            smallestAvrSide = avrSide
            print("variable ", model.xSymbolic[value], " is splitted")
            xNewBounds = [output0["xNewBounds"][0], output1["xNewBounds"][0]]
            
    return xNewBounds


def getBestSplit(xBounds,model, boxNo, dict_options):
    '''finds variable, which splitting causes the best reduction

    Args:
        :xBounds:       variable bounds of class momath.iv
        :model:         instance of type model
        :boxNo:         integer with number of boxes
        :dict_options:  dictionary of options
      
    Return:
        :xNewBounds:    best reduced two variable boxes
        
    '''      
    print(dict_options["split_Box"])
    oldBounds = list(numpy.array(xBounds)[0])  
    smallestAvrSide = numpy.Inf
    
    #try all splits
    for i,x in enumerate(model.xSymbolic):
        BoundsToSplit = list(numpy.array(xBounds)[0])
        splittedBox = separateBox(BoundsToSplit, [i])
        
        #reduce both boxes
        output0, output1 = reduceHC4_orNewton(splittedBox, model, boxNo, 
                                              dict_options)
        if output0["xNewBounds"] != [] and output0["xNewBounds"] != [[]]:
            avrSide0 = identifyReduction(output0["xNewBounds"], oldBounds)
        else:
            #if one of splitted boxes is empty always prefer this split
            print("This is the current best splitted varID by an empty box ", i)
            return [tuple(splittedBox[1])]
        
        if output1["xNewBounds"] != [] and output1["xNewBounds"] != [[]]:
            avrSide1 = identifyReduction(output1["xNewBounds"], oldBounds)
        else:
            #if one of splitted boxes is empty always prefer this split
            print("This is the current best splitted varID by an empty box: ", 
                  model.xSymbolic[i])
            return [tuple(splittedBox[0])]
        
        # sum of both boxreductions
        avrSide = avrSide0 + avrSide1
        # find best overall boxredution
        if avrSide<smallestAvrSide:
            smallestAvrSide = avrSide
            print("variable ", model.xSymbolic[i], " is splitted")
            xNewBounds = [output0["xNewBounds"][0], output1["xNewBounds"][0]]
            
    return xNewBounds


def reduceHC4_orNewton(splittedBox, model, boxNo, dict_options):
    '''reduces both side of the splitted box with detNewton or HC4

    Args:
        :splittedBox:   list of two boxes with variable bounds of class momath.iv
        :model:         instance of type model
        :boxNo:         integer with number of boxes
        :dict_options:  dictionary of options
        
    Return:
        :output0:    reduced box 1
        :output1:    reduced box 2
    '''    
    dict_options_temp = dict_options.copy()
    
    #if newton for split
    if (dict_options["split_Box"] =="forecast_newton" or 
        dict_options["split_Box"] =="forecastSplit"or 
        dict_options["split_Box"]=="forecastTear" or 
        dict_options["split_Box"]=="LeastChanged"):
        dict_options_temp.update({"hc_method":dict_options["hc_method"], 
                                  "bc_method":'None',
                                  "newton_method": dict_options["newton_method"],
                                  "InverseOrHybrid": 'none', 
                                  "Affine_arithmetic": False})       
        output0 = reduceBox(numpy.array(splittedBox[0]), model, boxNo, dict_options_temp)
        output1 = reduceBox(numpy.array(splittedBox[1]), model, boxNo, dict_options_temp)
        
    #if HC4 for split    
    elif dict_options["split_Box"]=="forecast_HC4":
        dict_options_temp.update({"hc_method":'HC4', "bc_method":'none',
                                  "newton_method":'none', 
                                  "InverseOrHybrid": 'both', 
                                  "Affine_arithmetic": False})
        
        output0 = reduceBox(numpy.array(splittedBox[0]), model, boxNo, dict_options_temp)
        output1 = reduceBox(numpy.array(splittedBox[1]), model, boxNo, dict_options_temp)
      
    return output0, output1
    
        
def identifyReduction(newBox,oldBox):
    '''calculates the average side length reduction from old to new Box

    Args:
        :newBox:        new variable bounds of class momath.iv
        :oldBox:        old variable bounds of class momath.iv
        
    Return:
        :avrSideLength/len(oldBox):    average sidelength reduction
        
    '''
    avrSideLength = 0
    for i, box in enumerate(oldBox):
        if (float(mpmath.convert(box.b))-float(mpmath.convert(box.a)))>0:
            avrSideLength = avrSideLength + (float(mpmath.convert(newBox[0][i].b))-
                                             float(mpmath.convert(newBox[0][i].a)))/(
                                             float(mpmath.convert(oldBox[i].b))-
                                             float(mpmath.convert(oldBox[i].a)))
                                             
    return avrSideLength/len(oldBox)


def splitTearVars(tearVarIds, box, boxNo_max, dict_options):
    """ splits unchanged box by one of its alternating tear variables
    
    Args:
        :tearVarIds:    list with global id's of tear variables
        :box:           numpy array intervals in mpmath.mpi formate
        :boxNo_max:     currently available number of boxes to maximum
        :dict_options:  dictionary with user specific settings
    
    Return: two sub boxes bisected by alternating tear variables from dict_options
        
    """
    
    if tearVarIds == [] or boxNo_max <= 0 : return [box], dict_options["tear_id"]
    iN = getCurrentVarToSplit(tearVarIds, box, dict_options)
    
    if iN == []: return [box], dict_options["tear_id"]
    print("Variable ", tearVarIds[iN], " is now splitted.")
    
    return separateBox(box, [tearVarIds[iN]]), iN + 1


def getCurrentVarToSplit(tearVarIds, box, dict_options):
    """ returns current tear variable id in tearVarIds for bisection. Only tear
    variables with nonzero widths are selected. 
    
    Args:
        :tearVarIds:    list with global id's of tear variables
        :box:           numpy array intervals in mpmath.mpi formate   
        :dict_options:  dictionary with user specific settings  
        
    Return:
        :i:             current tear variable for bisection
        
    """
    i = dict_options["tear_id"]
    
    if i  > len(tearVarIds) - 1: i = 0   
    if checkIntervalWidth(box[tearVarIds], dict_options["absTol"],
                            dict_options["relTol"]) == []:
        return []
         
    while checkIntervalWidth([box[tearVarIds[i]]], dict_options["absTol"],
                             dict_options["relTol"]) == []:
        if i  < len(tearVarIds) - 1: i+=1
        else: i = 0

    else: return i
    
        
def separateBox(box, varID):
    """ bi/multisects a box by variables with globalID in varID
    
    Args:
        :box:       numpy.array with variable bounds
        :varID:     list with globalIDs of variables chosen for bisection        
        
    Return:
        numpy.array wit sub boxes
        
    """     
    for i, interval in enumerate(box):
        if i in varID:
          box[i]=[mpmath.mpi(interval.a, interval.mid), mpmath.mpi(interval.mid, 
                                                                   interval.b)]
        else:box[i]=[interval]
        
    return list(itertools.product(*box))

# def getNewtonIntervalSystem(xBounds, model, options):
#     '''calculates all necessary System matrices for the Newton-Interval reduction
#     Args:
#         :xBounds:           list with variable bounds in mpmath.mpi formate
#         :model:             instance of type model
#         :options:           dictionary options

#     Return:
#         :Boxpoint:          list with Boxpoint/s values
#         :fpoint:            functions of Boxpoint/s in numpy array
#         :Jacpoint:          jacobian of Boxpoint/s in numpy array
#         :JacInterval:       jacobian of ivmpf intervals in numpy array
#         :JacInv:            inverse of Jacpoint in numpy array, stripped of inf Parts
#     '''
    
#     flamb = model.fLamb
#     jaclamb = model.jacobianLambNumpy
#     jacIvLamb = model.jacobianLambMpmath
#     noJacpoint = False
    
#     if options['newton_method'] == "mJNewton": noJacpoint = True
#     if options['newton_method'] == "detNewton":
#         Boxpoint, Jacpoint, fpoint = findPointWithHighestDeterminant(jaclamb, flamb, xBounds, noJacpoint)
#     elif options['newton_method'] == "3PNewton":
#         Boxpoint, Jacpoint, fpoint = getFunctionPoint(jaclamb, flamb, xBounds, ['a','mid','b'], noJacpoint)
#     else:
#         Boxpoint, Jacpoint, fpoint = getFunctionPoint(jaclamb, flamb, xBounds, ['mid'], noJacpoint)
     
#     fpoint = [fp[model.rowPerm] for fp in fpoint]
#     Boxpoint = [bp[model.colPerm] for bp in Boxpoint]
#     #if not len(Jacpoint)==1: 
#     Jacpoint = [jp[model.rowPerm,:][:,model.colPerm] for jp in Jacpoint]
#     JacInterval = numpy.array(jacIvLamb(*xBounds))[model.rowPerm,:][:,model.colPerm]
#     Jac_c = numpy.zeros([len(xBounds), len(xBounds)])

#     infRows = []           
#     for l in range(len(JacInterval)):
#         for n, iv in enumerate(JacInterval[l]):
#             if isinstance(iv, mpmath.ctx_iv.ivmpf):
#                 if iv == mpmath.mpi('-inf','+inf'):
#                     JacInterval[l][n] = mpmath.mpi(numpy.nan_to_num(-numpy.inf), numpy.nan_to_num(numpy.inf))
#                     if l not in infRows:
#                         infRows.append(l)
#                 elif iv.a == '-inf': 
#                     JacInterval[l][n] = mpmath.mpi(numpy.nan_to_num(-numpy.inf), iv.b)
#                 elif iv.b == '+inf':
#                     JacInterval[l][n] = mpmath.mpi(iv.a, numpy.nan_to_num(numpy.inf))
#             if options['newton_method'] == "mJNewton": 
#                 if not isinstance(JacInterval[l][n], mpmath.ctx_iv.ivmpf):
#                     Jac_c[l][n] = float(JacInterval[l][n])
#                 else:
#                     Jac_c[l][n] = numpy.float(mpmath.mpf(JacInterval[l][n].mid))
                    
#     if options['newton_method'] == "mJNewton": Jacpoint = [Jac_c]
#     JacInv = []
#     if options["InverseOrHybrid"]!='Hybrid':
#         for J in Jacpoint:
#             try:
#                 JacInv.append(numpy.linalg.inv(J))
#             except: 
#                 print('singular point')
#                 JacInv.append(numpy.array(numpy.matrix(J).getH()))
        
#         #remove inf parts in inverse
#         for inf in infRows:
#             for Ji in JacInv:
#                 Ji[:,inf] = numpy.zeros(len(Ji))
       
        
#     return {'Boxpoint':Boxpoint, 'f(Boxpoint)':fpoint, 'J(Boxpoint)':Jacpoint,
#             'J(Box)':JacInterval, 'J(Boxpoint)-1':JacInv,
#             'infRows': infRows}


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

# def findPointWithHighestDeterminant(jacLamb, fLamb, xBounds, noJacpoint=None): 
#     '''finds Boxpoint with highest determinant of J(Boxpoint)
#     Args:
#         :jacLamb: 		lambdifyed Jacobian 
#         :fLamb: 		lambdifyed function
#         :xBounds: 		current xBounds as iv.mpi
#     Return:
#         :chosenPoint: 	List of variable Point in Box
#         :chosenJacobi: 	Jacobian evaluated at chosen Point
#         :chosenFunc: 	function evaluated at chosen Point
#     '''
    
#     lowerPoint = getPointInBox(xBounds, len(xBounds)*['a'])
#     midPoint = getPointInBox(xBounds, len(xBounds)*['mid'])
#     upperPoint = getPointInBox(xBounds, len(xBounds)*['b'])
#     TestingPoints = [lowerPoint, midPoint, upperPoint]
    
#     biggestDValue = -numpy.inf    
#     for tp in TestingPoints:
#         warnings.filterwarnings("ignore", category=RuntimeWarning)
#         currentJacpoint = jacLamb(*tp)
#         currentFunc = numpy.array([fLamb(*tp)])
#         warnings.filterwarnings("default", category=RuntimeWarning)
#         currentJacpoint = removeInfAndConvertToFloat(currentJacpoint, 1)
#         currentFunc = removeInfAndConvertToFloat(currentFunc, 1)
#         currentAbsDet = numpy.absolute(numpy.linalg.det(currentJacpoint))
#         decisionValue = currentAbsDet
#         if decisionValue>=biggestDValue:
#             biggestDValue = decisionValue
#             chosenPoint = tp
#             chosenJacobi = currentJacpoint
#             chosenFunc = currentFunc
     
#     return [chosenPoint], [chosenJacobi], [chosenFunc[0]]


# def getFunctionPoint(jacLamb, fLamb, xBounds, points, noJacpoint=None):
#     '''finds Boxpoint with highest determinant of J(Boxpoint)
#     Args:
#         :jacLamb:       lambdifyed Jacobian 
#         :fLamb:         lambdifyed function
#         :xBounds:       current xBounds as iv.mpi
#         :points:        list of strings with "a" and/or "mid" and/or "b"
#     Return:
#         :Boxpoint:  variable values of points
#         :Jacpoint:  Jacobian values of chosen points
#         :fpoint:    function values of chosen points
#     '''
    
#     warnings.filterwarnings("ignore", category=RuntimeWarning)
#     Boxpoint = []
#     Jacpoint = []
#     fpoint = []
#     for p in points:
#         PointIndicator = len(xBounds)*[p]
#         singlePoint = getPointInBox(xBounds, PointIndicator)
#         Boxpoint.append(singlePoint)
#         if not noJacpoint: Jacpoint.append(removeInfAndConvertToFloat(jacLamb(*singlePoint), 1))
#         fpoint.append(removeInfAndConvertToFloat(numpy.array([fLamb(*singlePoint)]), 1)[0])
#     warnings.filterwarnings("default", category=RuntimeWarning)
    
#     return Boxpoint, Jacpoint, fpoint


def removeInfAndConvertToFloat(array, subs):
    '''removes inf in 2-dimensional arra
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


def get_failed_output(f, varBounds):
    """ collects information about variable bound reduction in function f
    Args:
        :f:             instance of class function
        :varBounds:     dictionary with informaiton about failed variable bound

    Return:
        :output:        dictionary with information about failed variable bound
                        reduction

    """

    output = {}
    output["xNewBounds"] = []
    failedSystem = FailedSystem(f.f_sym, f.x_sym[varBounds['Failed_xID']])
    output["noSolution"] = failedSystem
    output["xAlmostEqual"] = False
    output["xSolved"] = False
    
    return output
       

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


# def reduceBoxHC4Bnormal(xBounds, model, boxNo, dict_options):
#     subBoxNo = 1
#     output = {} 
#     output["disconti"]=False
#     xNewBounds = list(xBounds)
#     xNewListBounds = list(xBounds)
#     xUnchanged = True
#     xSolved = True
#     dict_options_temp = dict_options.copy()
#     dict_options_temp.update({"newton_method":"3PNewton","InverseOrHybrid":"both"})
#     output, empty = doHC4(model, xNewBounds, xNewBounds, output, dict_options)
#     if empty: return output
   
#     for i in range(0, len(model.xSymbolic)):
#         y = [xNewBounds[i]]
#         if dict_options["Debug-Modus"]: print(i)
#         checkIntervalAccuracy(xNewBounds, i, dict_options_temp)
#         for j in model.dict_varId_fIds[i]:
#             if model.functions[j].f_sym.count(model.xSymbolic[i])>1:
#                 y = do_bnormal(model.functions[j], xNewBounds, y, i, 
#                                dict_options_temp)
#                 if y == [] or y ==[[]]: 
#                     saveFailedSystem(output, model.functions[j], model, i)
#                     return output

#                 if ((boxNo-1) + subBoxNo * len(y)) > dict_options["maxBoxNo"]:
#                     y = [xBounds[i]]
#                     output["disconti"] = True
#                 if variableSolved(y, dict_options_temp): break
#                 else: xSolved = False
#                 xUnchanged = checkXforEquality(xBounds[i], y, xUnchanged, 
#                                        {"absTol":0.001, 'relTol':0.001})
#         # Update quantities
#         subBoxNo = subBoxNo * len(y) 
#         xNewBounds[i] = y[0]
#         xNewListBounds[i] = y
        
#         if not variableSolved(y, dict_options): xSolved = False
#         xUnchanged = checkXforEquality(xBounds[i], y, xUnchanged, 
#                                        {"absTol":0.001, 'relTol':0.001})               
   
#     output["xSolved"] = xSolved    
#     output["xNewBounds"] = list(itertools.product(*xNewListBounds))
#     if len(output["xNewBounds"])>1: 
#         output["xAlmostEqual"] = [False]*len(output["xNewBounds"])
#     else: 
#         output["xAlmostEqual"] = xUnchanged
                                        
#     return output


# def reduceBoxHC43PNewtonBnormal(xBounds, model, boxNo, dict_options):
#     """ reduce box spanned by current intervals of xBounds. First HC4 is tried.
#     If reduction is not sufficient, 3PNewton and bnormal are applied. If Affine-Arithmetic
#     is active, bnormal and 3PNewton are repeated with affine.
     
#     Args: 
#         :xBounds:            numpy array with box
#         :model:              instance of class Model
#         :dict_options:       dictionary with user specified algorithm settings
            
#         Returns:
#         :output:             dictionary with new boxes in a list and
#                              eventually an instance of class failedSystem if
#                              the procedure failed.
                        
#     """ 
#     subBoxNo = 1
#     output = {} 
#     output["disconti"] = False
#     xNewBounds = list(xBounds)
#     xNewListBounds = list(xBounds)
#     xUnchanged = True
#     xSolved = True
#     dict_options_temp = dict_options.copy()
#     dict_options_temp.update({"newton_method":"3PNewton","InverseOrHybrid":"both"})
 
#     # first it is tried to reduce bounds by HC4
#     output, empty = doHC4(model, xNewBounds, xNewBounds, output, dict_options)
#     if empty: return output
#     else:
#         for i in range(0, len(model.xSymbolic)):
#             if dict_options["Debug-Modus"]: print(i)
#             xNewListBounds[i] = [xNewBounds[i]]
#             xUnchanged = checkXforEquality(xBounds[i], xNewListBounds[i], xUnchanged, {"absTol":0.001, 'relTol':0.01})
#             if not variableSolved(xNewListBounds[i], dict_options): xSolved = False
    
#     # if HC4 could not reduce box sufficiently, now newton is used
#     if xUnchanged and not xSolved:
#         xSolved = True

#         for i in range(0, len(model.xSymbolic)):
#             # apply 3PNewton Hybrid and Inverse for every variable
#             y = [xNewBounds[i]]
#             if dict_options["Debug-Modus"]: print(i)
#             checkIntervalAccuracy(xNewBounds, i, dict_options_temp)
#             y, unique = iv_newton(model, xBounds, i, dict_options_temp)
#             if y == [] or y ==[[]]: 
#                 saveFailedSystem(output, model.functions[0], model, 0)
#                 return output
            
#             # Update quantities
#             subBoxNo = subBoxNo * len(y) 
#             xNewBounds[i] = y[0]
#             xNewListBounds[i] = y
#             if not variableSolved(y, dict_options): xSolved = False
#             xUnchanged = checkXforEquality(xBounds[i], y, xUnchanged, 
#                                            {"absTol":0.001, 'relTol':0.001})
            
#     # if reduction not suffiecient, apply bnormal
#     if xUnchanged and not xSolved:
#         for i in range(0, len(model.xSymbolic)):
#             y = [xNewBounds[i]]
#             if dict_options["Debug-Modus"]: print(i)
#             checkIntervalAccuracy(xNewBounds, i, dict_options_temp)
#             for j in model.dict_varId_fIds[i]:
                
#                 y = do_bnormal(model.functions[j], xNewBounds, y, i, 
#                                    dict_options_temp)
#                 if y == [] or y ==[[]]: 
#                     saveFailedSystem(output, model.functions[j], model, i)
#                     return output
    
#                 if ((boxNo-1) + subBoxNo * len(y)) > dict_options["maxBoxNo"]:
#                     y = xBounds[i]
#                     output["disconti"] = True
                    
    
#                 if variableSolved(y, dict_options_temp): break
#                 else: xSolved = False
                
#         # Update quantities
#         subBoxNo = subBoxNo * len(y) 
#         xNewBounds[i] = y
#         if not variableSolved(y, dict_options): xSolved = False
#         xUnchanged = checkXforEquality(xBounds[i], y, xUnchanged, 
#                                        {"absTol":0.001, 'relTol':0.001})   
#         dict_options_temp["relTol"] = dict_options["relTol"]
#         dict_options_temp["absTol"] = dict_options["absTol"]  
        
#     # Prepare output dictionary for return    
#     output["xSolved"] = xSolved       
#     output["xNewBounds"] = list(itertools.product(*xNewBounds))
#     if len(output["xNewBounds"])>1: 
#         output["xAlmostEqual"] = [False]*len(output["xNewBounds"])
#     else: 
#         output["xAlmostEqual"] = xUnchanged
#     return output


# def reduceBoxDetNewtonHC4(xBounds, model, dict_options):
#     """ reduce box spanned by current intervals of xBounds. First detNewton is tried.
#     If reduction is not sufficient, HC4 is applied.
     
#     Args: 
#         :xBounds:            numpy array with box
#         :model:              instance of class Model
#         :dict_options:       dictionary with user specified algorithm settings
            
#         Returns:
#         :output:             dictionary with new boxes in a list and
#                              eventually an instance of class failedSystem if
#                              the procedure failed.
                        
#     """ 
    
#     subBoxNo = 1
#     output = {}  
#     xUnchanged = True
#     xSolved = True
#     xNewBounds = list(xBounds)
#     xNewListBounds = list(xBounds)
#     dict_options_temp = dict_options.copy()
    
#     dict_options_temp.update({"newton_method":"detNewton","InverseOrHybrid":"both"})
    
#     for i in range(0, len(model.xSymbolic)):
#         # apply detNewton Hybrid and Inverse for every variable
#         y = [xNewBounds[i]]
#         if dict_options["Debug-Modus"]: print(i)
#         if not checkIntervalAccuracy(xNewBounds, i, dict_options_temp):
#             y, unique = iv_newton(model, xBounds, i, dict_options_temp)
#         if y == [] or y ==[[]]: 
#             saveFailedSystem(output, model.functions[0], model, 0)
#             return output
        
#         # Update quantities
#         subBoxNo = subBoxNo * len(y) 
#         xNewBounds[i] = y[0]
#         xNewListBounds[i] = y
#         if not variableSolved(y, dict_options): xSolved = False
#         xUnchanged = checkXforEquality(xBounds[i], y, xUnchanged, 
#                                        {"absTol":0.001, 'relTol':0.001})
    
#     # if reduction not suffiecient, apply HC4
#     if xUnchanged and not xSolved:
#         xSolved = True
#         output, empty = doHC4(model, xNewBounds, xNewBounds, output, dict_options)
#         if empty: return output
#         else:
#             for i in range(0, len(model.xSymbolic)):
#                 if dict_options["Debug-Modus"]: print(i)
#                 xNewListBounds[i] = [xNewBounds[i]]
#                 xUnchanged = checkXforEquality(xBounds[i], xNewListBounds[i], xUnchanged, {"absTol":0.001, 'relTol':0.01})
#                 if not variableSolved(xNewListBounds[i], dict_options): xSolved = False
                
                   
        
#     # Prepare output dictionary for return
#     output["xAlmostEqual"] = xUnchanged
#     output["xSolved"] = xSolved       
#     output["xNewBounds"] = list(itertools.product(*xNewListBounds))
#     return output


def reduceBox(xBounds, model, boxNo, dict_options):
    """ reduce box spanned by current intervals of xBounds.
     
    Args: 
        :xBounds:            numpy array with box
        :model:              instance of class Model
                             with function's glb id they appear in   
        :boxNo:              number of boxes as integer  
        :dict_options:       dictionary with user specified algorithm settings
            
        Returns:
        :output:             dictionary with new boxes in a list and
                             eventually an instance of class failedSystem if
                             the procedure failed.
                        
    """  
    subBoxNo = 1
    output = {}
    output["disconti"] = False          
    xNewBounds = list(xBounds)
    x_old = list(xBounds)
    xUnchanged = True
    xSolved = True
    dict_options_temp = dict_options.copy()
    newtonMethods = {'newton', 'detNewton', '3PNewton', 'mJNewton'}
    hc_methods = {'HC4'}
    u_solution_newton, u_solution_hc = set_unique_solution_true(dict_options, 
                                                                newtonMethods, 
                                                                hc_methods)

    # if HC4 is active
    if dict_options['hc_method']=='HC4':
        output, empty = doHC4(model, xBounds, xNewBounds, output, dict_options)
        
        for i, x_new in enumerate(xNewBounds):
            if not checkUniqueness([x_new], x_old[i], dict_options["relTol"],
                                  dict_options["absTol"]):
                u_solution_hc = False
                break
        
        if empty: return output
        xBounds = list(xNewBounds)
                 
    for i in model.colPerm: #range(0, len(model.xSymbolic)):
        y = [xNewBounds[i]]
        if dict_options["Debug-Modus"]: print(i)

        checkIntervalAccuracy(xNewBounds, i, dict_options_temp)
   
        # if any newton method is active       
        if not variableSolved(y, dict_options_temp) and dict_options['newton_method'] in newtonMethods:
            y, unique = iv_newton(model, xBounds, i, dict_options_temp)
            if not unique: 
                u_solution_newton = False
               
            if len(y) > 1: 
                y = prove_list_for_root_inclusion(y, xBounds, i, model.functions, 
                                                  dict_options)
            if y == [] or y ==[[]]: 
                saveFailedSystem(output, model.functions[0], model, i)
                return output
            elif len(y)==1 and y[0]!=xBounds[i]: xBounds[i] = y[0]
            
        # if bnormal is active
        if not variableSolved(y, dict_options_temp) and dict_options['bc_method']=='bnormal':
            for j in model.dict_varId_fIds[i]:
                
                y = do_bnormal(model.functions[j], xBounds, y, i, 
                                   dict_options_temp)
                if "disconti_iv" in dict_options_temp.keys() and dict_options["consider_disconti"]:
                    output["disconti"] = True
                    del dict_options_temp["disconti_iv"]  
                elif len(y) > 1: 
                    y = prove_list_for_root_inclusion(y, xBounds, i, model.functions, 
                                                  dict_options)
                if y == [] or y ==[[]]: 
                    saveFailedSystem(output, model.functions[j], model, i)
                    return output
                elif len(y)==1 and y[0]!=xBounds[i]: xBounds[i] = y[0]
                
                if variableSolved(y, dict_options_temp): break

        if ((boxNo-1) + subBoxNo * len(y)) > dict_options["maxBoxNo"]:
            y = [mpmath.mpi(min([yi.a for yi in y]),max([yi.b for yi in y]))]
            output["disconti"]=True
        # Update quantities
        xSolved, xUnchanged, subBoxNo = update_quantities(y, i, subBoxNo, xNewBounds, 
                                                          x_old, xUnchanged, xSolved, 
                                                          dict_options, dict_options_temp)
    # for i, x_new in enumerate(xNewBounds):
    #     if len(x_new) > 1: break
    #     else:
    #         if not checkUniqueness([x_new[0]], xBounds[i], dict_options["relTol"],
    #                                dict_options["absTol"]): 
    #             u_solution_bc = False
    #             break
        
    # Uniqueness test of solution:
    xNewBounds = list(itertools.product(*xNewBounds))
    if (u_solution_newton or u_solution_hc) and "FoundSolutions" in dict_options.keys(): 
        if test_for_root_inclusion(xNewBounds[0], dict_options["FoundSolutions"], 
                                   dict_options_temp["absTol"]):
            output["uniqueSolutionInBox"] = True
            xSolved = True
            xUnchanged = True

        # for solution in dict_options["FoundSolutions"]:
        #     xFound = numpy.array([False]*len(solution))
        #     for i in range(len(solution)):
        #         if solution[i] in xNewBounds[0][i]: xFound[i] = True
        #         elif (numpy.isclose(float(mpmath.mpf(xNewBounds[0][i].a)),
        #                                  solution[i], dict_options["absTol"])
        #                    or numpy.isclose(float(mpmath.mpf(xNewBounds[0][i].b)),
        #                                  solution[i], dict_options["absTol"])):
        #             xFound[i] = True
        #             print("Warning: Inaccuracy possible.")
        #         else: break

        #     if xFound.all():    
        #         print("Successful Root Inclusion test")                
        #         output["uniqueSolutionInBox"] = True
        #         xSolved = True
        #         xUnchanged = True
        #         break       
    # Prepare output dictionary for return
    return prepare_output(dict_options, output, xSolved, xNewBounds, xUnchanged)


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
            elif (numpy.isclose(float(mpmath.mpf(box[i].a)),x, absTol) or
                  numpy.isclose(float(mpmath.mpf(box[i].b)),x, absTol) ):
                print("Warning: Inaccuracy possible.")
                continue
            else: 
                solved = False 
                break
        if solved: 
            print("Successful Root Inclusion test")
            return True
    return False
    


def prove_list_for_root_inclusion(interval_list, box, i, functions, dict_options):
    """ checks a list of intervals of a certain variable if they are non-empty
    in the constrained problem given by functions and box.
    
    Args:
        :interval_list:         list with a specific variable's intervals
        :box:                   box as list with mpmath.mpi intervals
        :i:                     id of currently reduced variable as int
        :functions:             list with function objects
        :dict_options:          dictionary with settings of box reduction
        
    Return:
        :interval_list:         list with non-empty intervals of the variable
        
    """
    cur_box = list(box)
    for iv in interval_list:
        cur_box[i] = iv
        if not solutionInFunctionRangePyibex(functions, cur_box, dict_options):
            interval_list.remove(iv)
            
    return interval_list


def set_unique_solution_true(dict_options, newtonMethods, hc_methods):
    """ initializes the parameter unique_solution for root inclusion tests in
    a box. It is set to False if it is not used in the user-specified run.
    
    Args:
        :dict_options:      dictionary with chosed reduction algorithms
        :newtonMethods:     dictionary with all possible I-Newton methods
        :hc_methods:        dictionary with all possible HC-methods
        
    Returns:
        :unique_solution_newton: boolean if root inclusion test in I-Newton
        :unique_solution_hc:     boolean if root inclusion test in HC-method
        
    """    
    if dict_options['newton_method'] in newtonMethods: 
        unique_solution_newton = True
    else: unique_solution_newton = False
    if dict_options['hc_method'] in hc_methods: unique_solution_hc = True
    else: unique_solution_hc = False
    #if dict_options['bc_method'] == "bnormal": unique_solution_bc = True
    #else: unique_solution_bc = False    
    return unique_solution_newton, unique_solution_hc


def prepare_output(dict_options, output, xSolved, xNewBounds, xUnchanged):   
    """ prepares dictionary with output of box reduction step
    
    Args:
        :dict_options:      dictionary with box reduction settings
        :output:            dictionary with output quantities of box reduction
                            step
        :xSolved:           list with boolean if box intervals are solved
        :xNewBounds:        list with reduced variable bounds
        :xUnchanged:        list with boolean if box intervals still change
                            
    Return
        :output:            dictionary with output

    """
    output["xNewBounds"] = xNewBounds
    if len(output["xNewBounds"])>1: 
        output["xAlmostEqual"] = [False] * len(output["xNewBounds"])
        output["xSolved"] = [xSolved] * len(output["xNewBounds"])   
    else: 
        output["xAlmostEqual"] = [xUnchanged]
        output["xSolved"] = [xSolved]

    return output


def update_quantities(y, i, subBoxNo, xNewBounds, xBounds, xUnchanged, xSolved,
                      dict_options, dict_options_temp):
    """ updates all quantities after box reductions
    
    Args:
        :y:                 list will currently reduced variable's intervals
        :i:                 id of currently reduced variable as int
        :subBoxNo:          current number of sub-boxes as int
        :xNewBounds:        list with already reduced variable bounds
        :xBounds:           list with former box bounds
        :xUnchanged:        list with boolean if box intervals still change
        :xSolved:           list with boolean if box intervals are solved
        :dict_options:      dictionary with box reduction settings
        :dict_options_temp: dictionary with modified box reduction setting 
                            for solved intervals that need to be further 
                            reduced to tighten other non-degenerate intervals
                            
    Return
        :xSolved:           updated list for current variable interval
        :xUnchanged:        updated list for current variable interval
        :subBoxNo:          updated number of sub-boxes

    """
    subBoxNo = subBoxNo * len(y) 
    xNewBounds[i] = y
    if not variableSolved(y, dict_options): xSolved = False
    if xUnchanged: xUnchanged = checkXforEquality(xBounds[i], y, xUnchanged, 
                                   {"absTol":0.001, 'relTol':0.001})  
    
    dict_options_temp["relTol"] = dict_options["relTol"]
    dict_options_temp["absTol"] = dict_options["absTol"]  
    return xSolved, xUnchanged, subBoxNo


def do_bnormal(f, xBounds, y, i, dict_options):
    """ excecutes box consistency method for a variable with global index i in
    function f and intersects is with its former interval. 
    
    Args:
        :f:             instance of type function
        :xBounds:       numpy array with currently reduced box
        :y:             currently reduced interval in mpmath.mpi formate
        :i:             global index of the current variable
        :dict_options:  dictionary with user-settings such as tolerances
    
    Returns:
        :y:             current interval after reduction in mpmath.mpi formate

    """
    box = [xBounds[j] for j in f.glb_ID]
    y = setOfIvSetIntersection([y, reduceXIntervalByFunction(box, f,
                                                          f.glb_ID.index(i),
                                                          dict_options)])         
    return y


def doHC4(model, xBounds, xNewBounds, output, dict_options):
    """ excecutes HC4revise hull consistency method and returns output with
    failure information in case of an empty box. Otherwise the initial output
    dictionary is returned.
    
    Args:
        :model:         instance of type model
        :xBounds:       numpy array with currently reduced box
        :xNewBounds:    numpy array for reduced box
        :output:        dictionary that stores information of current box reduction
        :dict_otpions:  dictionary with tolerances
    
    Returns:
        :output:        unchanged dictionary (successful reduction) or dictionary
                        with failure outpot (unsuccessful reduction)
        :empty:         boolean, that is true for empty boxes

    """
    empty = False
    HC4_IvV = HC4(model.functions, xBounds, dict_options)
    if HC4_IvV.is_empty():
        saveFailedSystem(output, dict_options["failed_function"], model, 
                         dict_options["failed_function"].glb_ID[0])
        empty = True
    else:
        for i in range(len(model.xSymbolic)):
            HC4IV_mpmath = mpmath.mpi(HC4_IvV[i][0],(HC4_IvV[i][1]))
            #else: HC4IV_mpmath = mpmath.mpi(HC4_IvV[i][0]-1e-17,(HC4_IvV[i][1])+1e-17)
            y = ivIntersection(xBounds[i], HC4IV_mpmath)
            if y == [] or y == [[]]:

                if (numpy.isclose(float(mpmath.mpf(xBounds[i].b)), 
                                 HC4_IvV[i][0], dict_options["absTol"]) or 
                    numpy.isclose(float(mpmath.mpf(xBounds[i].a)), 
                                  HC4_IvV[i][1], dict_options["absTol"])):    
                    y = mpmath.mpi(min(xBounds[i].a,HC4IV_mpmath.a), 
                                   max(xBounds[i].b,HC4IV_mpmath.b))                                      
            xNewBounds[i] = y
            if  xNewBounds[i]  == [] or  xNewBounds[i]  ==[[]]:                 
                if "failed_function" in dict_options.keys(): 
                    saveFailedSystem(output, dict_options["failed_function"], 
                                     model, dict_options["failed_function"].glb_ID[0])
                    del dict_options['failed_function']
                else:
                    saveFailedSystem(output, model.functions[0], model, 0)
                empty = True
                break  
            
    return output, empty

       
def checkIntervalAccuracy(xNewBounds, i, dict_options):
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
    :dict_options:  dictionary with current relative and absolute tolerance of 
                    variable
    Return:         False if not solved yet and true otherwise
    
    """  
    if xNewBounds[i].delta == 0: return True
    else:
        accurate = variableSolved([xNewBounds[i]], dict_options)
        notdegenerate = xNewBounds[i].delta > 1.0e-15
        if accurate and notdegenerate:
            if isinstance(xNewBounds[i], mpmath.ctx_iv.ivmpf):
                dict_options["relTol"] = 0.1 * xNewBounds[i].delta
                dict_options["absTol"] = min(abs(0.1 * float(mpmath.mpf(xNewBounds[i].a))), 
                                             dict_options["absTol"])
            else:  
                dict_options["relTol"] = 0.1 * (xNewBounds[i][1] - xNewBounds[i][0])
                dict_options["absTol"] = min(0.1 * abs(xNewBounds[i][0]), 
                                                  dict_options["absTol"])   
                
        if isinstance(dict_options["relTol"], mpmath.ctx_iv.ivmpf):
            dict_options["relTol"] = float(mpmath.mpf(dict_options["relTol"].a))
        if isinstance(dict_options["absTol"], mpmath.ctx_iv.ivmpf):
            dict_options["absTol"] = float(mpmath.mpf(dict_options["absTol"].a))             
                
        return False


def variableSolved(BoundsList, dict_options):
    """ checks, if variable is solved in all Boxes in BoundsList
    Args:
        :BoundsList:      List of mpi Bounds for single variable
        :dict_options:    dictionary with tolerances for equality criterion
    
    Return:
        :variableSolved:  boolean that is True if all variables have been solved
        
    """
    variableSolved = True
    for bound in BoundsList:
        if not checkVariableBound(bound, dict_options):
            variableSolved = False
    
    return variableSolved


def checkXforEquality(xBound, xNewBound, xUnchanged, dict_options):
    """ changes variable xUnchanged to false if new variable interval xNewBound
    is different from former interval xBound
    
    Args:
        :xBound:          interval in mpmath.mpi formate
        :xNewBound:       interval in mpmath.mpi formate
        :xUnchanged:      boolean
        :dict_options:    dictionary with tolerances for equality criterion
        
    Return:
        :xUnchanged:     boolean that is True if interval has not changed
        
    """
    absEpsX = dict_options["absTol"]
    relEpsX = dict_options["relTol"]
    lb = numpy.isclose(float(mpmath.mpf(xNewBound[0].a)), 
                       float(mpmath.mpf(xBound.a)), relEpsX, absEpsX)
    ub = numpy.isclose(float(mpmath.mpf(xNewBound[0].b)), 
                       float(mpmath.mpf(xBound.b)), relEpsX, absEpsX)
        
    if not lb or not ub and xUnchanged: xUnchanged = False   
    
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
                
                               
def saveFailedSystem(output, f, model, i):
    """ saves output of failed box reduction 
    
    Args:
        :output:        dictionary with output data
        :f:             instance of class Function
        :model:         instance of class Model
        :i:             index of variable
   
    """     
    output["xNewBounds"] = []
    failedSystem = FailedSystem(f.f_sym, model.xSymbolic[i])
    output["noSolution"] = failedSystem
    output["xAlmostEqual"] = [False] 
    output["xSolved"] = [False]

    
def checkVariableBound(newXInterval, dict_options):
    """ if lower and upper bound of a variable are almost equal the boolean 
    boundsAlmostEqual is set to true.

    Args:
        :newXInterval:      variable interval in mpmath.mpi logic
        :dict_options:      dictionary with tolerance limits
        
    Return:                 True, if lower and upper variable bound are almost
                            equal.

    """
    absEpsX = dict_options["absTol"]
    relEpsX = dict_options["relTol"]    
    if numpy.isclose(float(mpmath.mpf(newXInterval.a)), 
                     float(mpmath.mpf(newXInterval.b)), relEpsX, absEpsX):
        return True

    
def reduceXIntervalByFunction(xBounds, f, i, dict_options):
    """ reduces variable interval by either solving a linear function directly
    with Gap-operator or finding the reduced variable interval(s) of a
    nonlinear function by interval nesting
     
    Args: 
        :xBounds:            one set of variable interavls as numpy array
        :f:                  instance of class Function
        :i:                  index for iterated variable interval
        :dict_options:       dictionary with solving settings

    Returns:                 list with new set of variable intervals
                        
    """       
    xUnchanged = True
    xBounds = list(xBounds)
    
    gxInterval, dgdxInterval, bInterval = calculateCurrentBounds(f, i, xBounds, 
                                                                 dict_options)
    if (gxInterval == [] or dgdxInterval == [] or 
        bInterval == [] or gxInterval in bInterval): return [xBounds[i]]
    xUnchanged = checkXforEquality(gxInterval, [dgdxInterval*xBounds[i]], 
                                   xUnchanged, dict_options)
    
    if f.deriv_is_constant[i] and xUnchanged : # Linear Case -> solving system directly f.x_sym[i] in f.dgdx_sym[i].free_symbols
        return getReducedIntervalOfLinearFunction(dgdxInterval, i, xBounds, 
                                                  bInterval)
             
    else: # Nonlinear Case -> solving system by interval nesting
        return getReducedIntervalOfNonlinearFunction(f, dgdxInterval, i, 
                                                     xBounds, bInterval, 
                                                     dict_options)


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


def calculateCurrentBounds(f, i, xBounds, dict_options):
    """ calculates bounds of function gx, the residual b, first derrivative of 
    function gx with respect to variable x (dgdx).
    
    Args:
        :f:                  instance of class Function
        :i:                  index of current variable 
        :xBounds:            numpy array with variable bounds
        :dict_options:       dictionary with entries about stop-tolerances
       
    Returns:
        :bInterval:          residual interval in mpmath.mpi logic
        :gxInterval:         function interval in mpmath.mpi logic
        :dfdxInterval:       Interval of first derrivative in mpmath.mpi logic 
    
    """
    #dict_options["tight_bounds"] = True
    try:
        if dict_options["Affine_arithmetic"]:
            bInterval = eval_fInterval(f, f.b_mpmath[i], xBounds, f.b_aff[i],
                                       dict_options["tight_bounds"])
        else: 
            bInterval = eval_fInterval(f, f.b_mpmath[i], xBounds, False, 
                                       dict_options["tight_bounds"])

    except: return [], [], []
    try:
        if dict_options["Affine_arithmetic"]:
            gxInterval = eval_fInterval(f, f.g_mpmath[i], xBounds, f.g_aff[i],
                                        dict_options["tight_bounds"])
        else: 
            gxInterval = eval_fInterval(f, f.g_mpmath[i], xBounds, False, 
                                        dict_options["tight_bounds"])

    except: return [], [], []
    try:
        if dict_options["Affine_arithmetic"]:
            dgdxInterval = eval_fInterval(f, f.dgdx_mpmath[i], xBounds, f.dgdx_aff[i],
                                          dict_options["tight_bounds"])
        else: 
            dgdxInterval = eval_fInterval(f, f.dgdx_mpmath[i], xBounds, False,
                                          dict_options["tight_bounds"])
    except: return [], [], []
   
    return gxInterval, dgdxInterval, bInterval  
   
         
def eval_function_tight_mpmath(f, fInterval, f_mpmath, f_sym, box):
    """ evaluate mpmath function with discretized intervals of box to reduce
    interval dependency issues
    
    Args:
        :f:             object of class Function
        :fInterval:     function interval from classical interval arithmetic
        :f_mpmath:      mpmath function
        :f_sym:         symbolic function in sympy logic
        :box:           list or numpy.array with variable intervals

    Return:
        :fInterval:     tightened function interval in mpmath.mpi formate

    """
    y_sym = list(f_sym.free_symbols)

    for j, var in enumerate(y_sym):
        k = f.x_sym.index(var)
        if f_sym.count(var) > 1 and box[k].delta > 1.0e-15: 
            #print("Before: ", fInterval)
            iv = convertIntervalBoundsToFloatValues(box[k])
            ivs = numpy.linspace(iv[0], iv[1], 8)
            f_low, f_up = getFunctionValuesIntervalsOfXList(ivs, f_mpmath, 
                                                            list(box), k)
            f_new = [min(f_low), max(f_up)]
            fInterval = mpmath.mpi(max(fInterval.a, f_new[0]), 
                                   min(fInterval.b, f_new[1]))
        else: continue
    
    return fInterval            
            
    
def eval_fInterval(f, f_mpmath, box, f_aff=None, f_tight=None):
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
                                               f.f_sym, box)  
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
    return [iv for iv in intervals if not (numpy.isclose(float(mpmath.mpf(iv.a)), 
                            float(mpmath.mpf(iv.b)), relEpsX, absEpsX))]


def checkAbsoluteTolerance(intervals, absEpsX):
    """ checks if width of intervals is smaller than a given absolute tolerance 
    absEpsX

    Args:
        :intervals:           set of intervals in mpmath.mpi-logic
        :absEpsX:             absolute x tolerance

    Return:
        :reducedInterval:    list of intervals with a higher width than absEps

    """
    return [iv for iv in intervals if iv.delta > absEpsX]


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
    if bool(0 in mpmath.mpi(bi) - a * xBounds[i]) == False: return [] # if this is the case, there is no solution in xBounds

    if bool(0 in mpmath.mpi(bi)) and bool(0 in mpmath.mpi(a)):  # if this is the case, bi/aInterval would return [-inf, +inf]. Hence the approximation of x is already smaller
                return [xBounds[i]]
    else: 
        return hansenSenguptaOperator(mpmath.mpi(a), mpmath.mpi(bi), xBounds[i]) # bi/aInterval  


def checkAndRemoveComplexPart(interval):
    """ creates a warning if a complex interval occurs and keeps only the real
    part.

    """
    if interval.imag != 0:
        print("Warning: A complex interval: ", interval.imag," occured.\n",
        "For further calculations only the real part: ", interval.real, " is used.")
        interval = interval.real


def hansenSenguptaOperator(a, b, x):
    """ Computation of the Gauss-Seidel-Operator [1] to get interval for x
    for given intervals for a and b from the 1-dimensional linear system:

                                    a * x = b
        Args:
            :a:     interval of mpi format from mpmath library
            :b:     interval of mpi format from mpmath library
            :x:     interval of mpi format from mpmath library
                    (initially guessed interval of x)

        Return:
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

        Return:
            :mpmath.mpi(a,b):    resulting interval of division [a,b],
                                 this is empty if i2 =[0,0] and returns []

    """
    if bool(0 in i2)== False: return [i1 * mpmath.mpi(1/i2.b, 1/i2.a)]
    if bool(0 in i1) and bool(0 in i2): return [i1 / i2]
    if i1.b < 0 and i2.a != i2.b and i2.b == 0: return [mpmath.mpi(i1.b / i2.a, i1.a / i2.b)]
    if i1.b < 0 and i2.a < 0 and i2.b > 0: return [mpmath.mpi(-numpy.nan_to_num(numpy.inf), max(i1.b / i2.b, mpmath.mpi(-numpy.nan_to_num(numpy.inf)))), 
                                                   mpmath.mpi(min(i1.b / i2.a,  numpy.nan_to_num(numpy.inf)), numpy.nan_to_num(numpy.inf))]
    if i1.b < 0 and i2.a == 0 and i2.b > 0: return [mpmath.mpi(-numpy.nan_to_num(numpy.inf), max(i1.b / i2.b, mpmath.mpi(-numpy.nan_to_num(numpy.inf))))]
    if i1.a > 0 and i2.a < 0 and i2.b == 0: return [mpmath.mpi(-numpy.nan_to_num(numpy.inf), max(i1.a / i2.a, mpmath.mpi(-numpy.nan_to_num(numpy.inf))))]
    if i1.a > 0 and i2.a < 0 and i2.b > 0: return [mpmath.mpi(-numpy.nan_to_num(numpy.inf), max(i1.a / i2.a, mpmath.mpi(-numpy.nan_to_num(numpy.inf)))), 
                                                   mpmath.mpi(min(i1.a / i2.b, mpmath.mpi(numpy.nan_to_num(numpy.inf))), numpy.nan_to_num(numpy.inf))]
    if i1.a > 0 and i2.a == 0 and i2.b > 0: return [mpmath.mpi(min(i1.a / i2.b, mpmath.mpi(numpy.nan_to_num(numpy.inf))),numpy.nan_to_num(numpy.inf))]

    if bool(0 in i1) == False and i2.a == 0 and i2.b == 0: return []


def ivIntersection(i1, i2):
    """ returns intersection of two intervals i1 and i2

        Args:
            :i1:     interval of mpi format from mpmath library
            :i2:     interval of mpi format from mpmath library

        Return:
            :mpmath.mpi(a,b):    interval of intersection [a,b],
                                 if empty [] is returned

    """
    if i1.a <= i2.a and i1.b <= i2.b and i2.a <= i1.b: return mpmath.mpi(i2.a, i1.b)
    if i1.a <= i2.a and i1.b >= i2.b: return i2
    if i1.a >= i2.a and i1.b <= i2.b: return i1
    if i1.a >= i2.a and i1.b >= i2.b and i1.a <= i2.b: return mpmath.mpi(i1.a, i2.b)

    else: return []
    
    
def check_capacities(nested_interval_list, f, b, i, box, dict_options):
    """ checks if there is enough space on the stack for any further splitting
    due to removing a discontinuity
    
    Args:
        :nested_interval_list:      list with intervals of currently reduced x
        :f:                         object of class Function
        :b:                         constant interval b in mpmath.mpi formate
        :i:                         index of currently reduced variable as int
        :box:                       list of f's variable intervals
        :dict_options:              dictionary with box reduction settings
    
    Return:
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
    if ivNo <= (dict_options["maxBoxNo"] - dict_options["boxNo"])+1: 
        return True, nested_interval_list    
    else: return False, nested_interval_list


def getReducedIntervalOfNonlinearFunction(f, dgdXInterval, i, xBounds, bi, dict_options):
    """ checks function for monotone sections in x and reduces them one after the other.

    Args:
        :f:                  object of class Function
        :dgdXInterval:       first derivative of function f with respect to x at xBounds
        :i:                  integer with current iteration variable index
        :xBounds:            numpy array with set of variable bounds
        :bi:                 current function residual bounds
        :dict_options:       for function and variable interval tolerances in the used
                             algorithms     

    Return:                reduced x-Interval(s) and list of monotone x-intervals

    """
    relEpsX = dict_options["relTol"]
    absEpsX = dict_options["absTol"]
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
                                                                      dict_options)
        if len(curXiBounds) + len(nonMonotoneZones) > 1:
            cap, intervals = check_capacities([curXiBounds, nonMonotoneZones], 
                                              f, bi, i, xBounds, dict_options)
            if cap: curXiBounds, nonMonotoneZones = intervals           
            else:
                if dict_options["consider_disconti"]: dict_options["disconti_iv"] = True
                return orgXiBounds
           
    if curXiBounds != []:
        for curInterval in curXiBounds:
            xBounds[i] = curInterval
            iz, dz, nmz = getMonotoneFunctionSections(f, i, xBounds, dict_options)
            if iz != []: increasingZones += iz
            if dz != []: decreasingZones += dz
            if nmz !=[]: nonMonotoneZones += nmz 

        if len(nonMonotoneZones)>1: 
           nonMonotoneZones = joinIntervalSet(nonMonotoneZones, relEpsX, absEpsX)
              
    if increasingZones !=[]:
            reducedIntervals = reduceMonotoneIntervals(increasingZones, 
                                                       reducedIntervals, f,
                                                       xBounds, i, bi, 
                                                       dict_options, 
                                                       increasing = True)
    if decreasingZones !=[]:               
            reducedIntervals = reduceMonotoneIntervals(decreasingZones, 
                                                       reducedIntervals, f, 
                                                       xBounds, i, bi, 
                                                       dict_options, 
                                                       increasing = False)  
    if nonMonotoneZones !=[]:
        reducedIntervals = reduceNonMonotoneIntervals({"0":nonMonotoneZones, 
                                   "1": reducedIntervals, 
                                   "2": f, 
                                   "3": i, 
                                   "4": xBounds, 
                                   "5": bi, 
                                   "6": dict_options})

        if reducedIntervals == False: 
            print("Warning: Reduction in non-monotone Interval took too long.")
            return orgXiBounds
    #reducedIntervals = joinIntervalSet(reducedIntervals, relEpsX, absEpsX)
    reducedIntervals = setOfIvSetIntersection([reducedIntervals, orgXiBounds])
    
    return reducedIntervals


def getContinuousFunctionSections(f, i, xBounds, dict_options):
    """filters out discontinuities which either have a +/- inf derrivative.

    Args:
        :f:                   object of type Function
        :i:                   index of differential variable
        :xBounds:             numpy array with variable bounds
        :dict_options:        dictionary with variable and function interval tolerances

    Return:
        :continuousZone:      list with continuous sections
        :discontiZone:        list with discontinuous sections
    
    """
    maxIvNo = dict_options["maxBoxNo"]
    absEpsX = dict_options["absTol"]
    relEpsX = dict_options["relTol"]
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
                                                     continuousZone, dict_options)  
               
        interval = checkIntervalWidth(discontinuousZone, absEpsX, 0.1*relEpsX)
        if not len(interval) <= maxIvNo: return (continuousZone, 
                                                 joinIntervalSet(interval, 
                                                                 relEpsX, absEpsX))
    if interval == [] and continuousZone == []: return [], [orgXiBounds]

    return continuousZone, []


def removeListInList(listInList):
    """changes list with the shape: [[a], [b,c], [d], ...] to [a, b, c, d, ...]

    """
    return [value for sublist in listInList for value in sublist]


def reduceMonotoneIntervals(monotoneZone, reducedIntervals, f,
                                      xBounds, i, bi, dict_options, increasing):
    """ reduces interval sets of one variable by interval nesting

    Args:
        :monotoneZone        list with monotone increasing or decreasing set of intervals
        :reducedIntervals    list with already reduced set of intervals
        :fx:                 symbolic x-depending part of function f
        :xBounds:            numpy array with set of variable bounds
        :i:                  integer with current iteration variable index
        :bi:                 current function residual bounds
        :dict_options:       dictionary with function and variable interval tolerances
        :increasing:         boolean, True for increasing function intervals,
                             False for decreasing intervals
    
    Return:
        :reducedIntervals:  list with reduced intervals
        
    """  
    for curMonZone in monotoneZone: #TODO: Parallelizing
        xBounds[i] = curMonZone

        if increasing: curReducedInterval = reduce_mon_inc_newton(f, xBounds, i, 
                                                                  bi, dict_options)
        else: curReducedInterval = reduce_mon_dec_newton(f, xBounds, i, bi, 
                                                         dict_options)
        if curReducedInterval !=[] and reducedIntervals != []:
            reducedIntervals.append(curReducedInterval)
            reducedIntervals = joinIntervalSet(reducedIntervals, 
                                               dict_options["relTol"], 
                                               dict_options["absTol"])
        elif curReducedInterval !=[]: reducedIntervals.append(curReducedInterval)

    return reducedIntervals


def reduce_mon_inc_newton(f, xBounds, i, bi, dict_options):
    """ Specific Interval-Newton method to reduce intervals in b_normal method
    
    Args:
        :f:             object of class Function
        :xBounds:       currently reduced box as list or numpy.array
        :bi:            constant interval for reduction in mpmath.mpi formate
        :dict_options:  dictionary with box reduction algorithm settings

    Return:
        reduced interval in mpmath.mpi formate
        
    """
    fxInterval = eval_fInterval(f, f.g_mpmath[i], xBounds, f.g_aff[i], False)
    # first check if xBounds can be further reduced:
    if fxInterval in bi: return xBounds[i]
    if ivIntersection(fxInterval, bi)==[]: return []
    x_old = list(xBounds)

    # Otherwise, iterate each bound of bi:
    relEpsX = dict_options["relTol"]
    absEpsX = dict_options["absTol"]
    curInterval = xBounds[i]    
    fIntervalxLow, fIntervalxUp = getFIntervalsFromXBounds(f, curInterval, xBounds, i)
    if fIntervalxLow.b < bi.a:
        while (not check_bound_and_interval_accuracy(
                convertIntervalBoundsToFloatValues(curInterval),
                [convert_mpi_float(fxInterval.b), convert_mpi_float(bi.a)],
                relEpsX, absEpsX)):
       # while (not numpy.isclose(float(mpmath.mpf(bi.a)), 
       #                          float(mpmath.mpf(fxInterval.b)),relEpsX, absEpsX) 
       #        and not numpy.isclose(float(mpmath.mpf(curInterval.a)), 
       #                              float(mpmath.mpf(curInterval.b)),relEpsX, 
       #                              absEpsX)):
            x = curInterval.a + curInterval.delta/2.0
            xBounds[i] = mpmath.mpi(x)
            fxInterval = eval_fInterval(f, f.g_mpmath[i], xBounds, 
                                        f.g_aff[i]).b - bi.a
            xBounds[i] = curInterval
            quotient = ivDivision(fxInterval, 
                                  eval_fInterval(f, f.dgdx_mpmath[i], xBounds))
            curInterval = x - quotient[-1]
    if curInterval.a > x_old[i].b or curInterval.b < x_old[i].a: x_low = x_old[i].mid
    #if curInterval.b < x_old[i].a: x_low = x_old[i].a
    else: x_low = max(curInterval.a, x_old[i].a)   
    fxInterval = eval_fInterval(f, f.g_mpmath[i], x_old, f.g_aff[i])
    curInterval = x_old[i]
    
    if fIntervalxUp.a > bi.b:
        while (not check_bound_and_interval_accuracy(
                convertIntervalBoundsToFloatValues(curInterval),
                [convert_mpi_float(bi.b), convert_mpi_float(fxInterval.a)],
                relEpsX, absEpsX)):               
        #while (not numpy.isclose(float(mpmath.mpf(fxInterval.a)), 
        #                         float(mpmath.mpf(bi.b)),relEpsX, absEpsX) 
        #       and not numpy.isclose(float(mpmath.mpf(curInterval.a)), 
        #                             float(mpmath.mpf(curInterval.b)),relEpsX, 
        #                             absEpsX)): 

            x = curInterval.b - curInterval.delta/2.0
            xBounds[i] = mpmath.mpi(x)
            fxInterval = eval_fInterval(f, f.g_mpmath[i], xBounds, 
                                        f.g_aff[i]).a - bi.b
            xBounds[i] = curInterval
            curInterval = x - ivDivision(fxInterval, 
                                         eval_fInterval(f, f.dgdx_mpmath[i], 
                                                        xBounds))[0]
    if curInterval.a > x_old[i].b or curInterval.b < x_old[i].a:
        if x_low == x_old[i].mid: return []
        else: return mpmath.mpi(x_low, x_old[i].mid)
    #if curInterval.a > x_old[i].b:
    #    return mpmath.mpi(x_low, x_old[i].b)
    else: 
        return mpmath.mpi(x_low, min(curInterval.b, x_old[i].b))


def reduce_mon_dec_newton(f, xBounds, i, bi, dict_options):
    """ Specific Interval-Newton method to reduce intervals in b_normal method
    
    Args:
        :f:             object of class Function
        :xBounds:       currently reduced box as list or numpy.array
        :bi:            constant interval for reduction in mpmath.mpi formate
        :dict_options:  dictionary with box reduction algorithm settings

    Return:
        reduced interval in mpmath.mpi formate
        
    """
    fxInterval = eval_fInterval(f, f.g_mpmath[i], xBounds, f.g_aff[i])
    # first check if xBounds can be further reduced:
    if fxInterval in bi: return xBounds[i]
    if ivIntersection(fxInterval, bi)==[]: return []
    x_old = list(xBounds)
    
    # Otherwise, iterate each bound of bi:
    relEpsX = dict_options["relTol"]
    absEpsX = dict_options["absTol"]
    curInterval = xBounds[i]    
    fIntervalxLow, fIntervalxUp = getFIntervalsFromXBounds(f, curInterval, 
                                                           xBounds, i)
    if fIntervalxLow.a > bi.b:
        while (not check_bound_and_interval_accuracy(
                convertIntervalBoundsToFloatValues(curInterval),
                [convert_mpi_float(bi.b), convert_mpi_float(fxInterval.a)],
                relEpsX, absEpsX)):        
        
        #while (not numpy.isclose(float(mpmath.mpf(fxInterval.a)), 
        #                         float(mpmath.mpf(bi.b)),relEpsX, absEpsX) 
        #       and not numpy.isclose(float(mpmath.mpf(curInterval.a)), 
        #                             float(mpmath.mpf(curInterval.b)),relEpsX, 
        #                             absEpsX)):
            x = curInterval.a + curInterval.delta/2.0
            xBounds[i]=mpmath.mpi(x)
            fxInterval = eval_fInterval(f, f.g_mpmath[i], xBounds, 
                                        f.g_aff[i]).a - bi.b
            xBounds[i] = curInterval
            quotient = ivDivision(fxInterval, 
                                  eval_fInterval(f, f.dgdx_mpmath[i], xBounds))
            curInterval = x - quotient[-1]
            
    if curInterval.a > x_old[i].b or curInterval.b < x_old[i].a: x_low = x_old[i].mid
    #if curInterval.b < x_old[i].a: x_low = x_old[i].a
    else: x_low = max(curInterval.a, x_old[i].a)       
    fxInterval = eval_fInterval(f, f.g_mpmath[i], x_old, f.g_aff[i])
    curInterval = x_old[i]
    
    if fIntervalxUp.b < bi.a:
        while (not check_bound_and_interval_accuracy(
                convertIntervalBoundsToFloatValues(curInterval),
                [convert_mpi_float(fxInterval.b), convert_mpi_float(bi.a)],
                relEpsX, absEpsX)):
        #while (not numpy.isclose(float(mpmath.mpf(bi.a)), 
        #                         float(mpmath.mpf(fxInterval.b)),relEpsX, absEpsX) 
        #       and not numpy.isclose(float(mpmath.mpf(curInterval.a)), 
        #                             float(mpmath.mpf(curInterval.b)),relEpsX, 
        #                             absEpsX)): 

            x = curInterval.b - curInterval.delta/2.0
            xBounds[i]=mpmath.mpi(x)
            fxInterval = eval_fInterval(f, f.g_mpmath[i], xBounds, 
                                        f.g_aff[i]).b - bi.a
            xBounds[i] = curInterval
            curInterval = ivIntersection(curInterval, x - ivDivision(fxInterval, 
                                         eval_fInterval(f, f.dgdx_mpmath[i], 
                                                        xBounds))[0] )
    if curInterval.a > x_old[i].b or curInterval.b < x_old[i].a:
        if x_low == x_old[i].mid: return []
        else: return mpmath.mpi(x_low, x_old[i].mid)   
    #if curInterval.a > x_old[i].b:
    #    return mpmath.mpi(x_low, x_old[i].b)    
    else: return mpmath.mpi(x_low, min(curInterval.b, x_old[i].b))
    
def convert_mpi_float(mpi):
    return float(mpmath.mpf(mpi))

    
def check_bound_and_interval_accuracy(x, val, relEpsX, absEpsX):
    
    if val[0] <= val[1] and numpy.isclose(val[0], val[1], relEpsX, absEpsX):
        return True
    elif numpy.isclose(x[0], x[1], relEpsX, absEpsX):
        return True
    else:
        return False
    
    
def monotoneIncreasingIntervalNesting(f, xBounds, i, bi, dict_options):
    """ reduces variable intervals of monotone increasing functions fx
    by interval nesting

        Args:
            :f:                  object of type Function
            :xBounds:            numpy array with set of variable bounds
            :i:                  integer with current iteration variable index
            :bi:                 current function residual bounds
            :dict_options:       dictionary with function and variable interval 
                                 tolerances

        Return:                  list with one entry that is the reduced interval
                                 of the variable with the index i

    """
    fxInterval = eval_fInterval(f, f.g_mpmath[i], xBounds, f.g_aff[i])
    # first check if xBounds can be further reduced:
    if fxInterval in bi: return xBounds[i]
    if ivIntersection(fxInterval, bi)==[]: return []
    
    # Otherwise, iterate each bound of bi:
    relEpsX = dict_options["relTol"]
    absEpsX = dict_options["absTol"]
    curInterval = xBounds[i]    
    fIntervalxLow, fIntervalxUp = getFIntervalsFromXBounds(f, curInterval, xBounds, i)

    if fIntervalxLow.b < bi.a:
        while (not numpy.isclose(float(mpmath.mpf(fxInterval.a)), 
                                 float(mpmath.mpf(fxInterval.b)),relEpsX, absEpsX) 
               and not numpy.isclose(float(mpmath.mpf(curInterval.a)), 
                                     float(mpmath.mpf(curInterval.b)),relEpsX, absEpsX)):  
        
                         curInterval, fxInterval = iteratefBound(f, curInterval, 
                                                                 xBounds, i, bi,
                                                                 increasing = True,
                                                                 lowerXBound = True)
                         if curInterval == [] or fxInterval == []: return []

    lowerBound = curInterval.a
    curInterval  = xBounds[i]    
    
    fxInterval = eval_fInterval(f, f.g_mpmath[i], xBounds, f.g_aff[i])
    
    if fIntervalxUp.a > bi.b:
        while (not numpy.isclose(float(mpmath.mpf(fxInterval.a)), 
                                 float(mpmath.mpf(fxInterval.b)),relEpsX, absEpsX) 
               and not numpy.isclose(float(mpmath.mpf(curInterval.a)), 
                                     float(mpmath.mpf(curInterval.b)),relEpsX, 
                                     absEpsX)): 

            curInterval, fxInterval = iteratefBound(f, curInterval, 
                                                    xBounds, i, bi,
                                                    increasing = True,
                                                    lowerXBound = False)
            if curInterval == [] or fxInterval == []: return []
    upperBound = curInterval.b
    
    return mpmath.mpi(lowerBound, upperBound)


def monotoneDecreasingIntervalNesting(f, xBounds, i, bi, dict_options):
    """ reduces variable intervals of monotone decreasing functions fx
    by interval nesting

        Args:
            :f:                  object of type function
            :xBounds:            numpy array with set of variable bounds
            :i:                  integer with current iteration variable index
            :bi:                 current function residual bounds
            :dict_options:       dictionary with function and variable interval tolerances

        Return:                  list with one entry that is the reduced interval
                                 of the variable with the index i

    """
    relEpsX = dict_options["relTol"]
    absEpsX = dict_options["absTol"]
    fxInterval =  eval_fInterval(f, f.g_mpmath[i], xBounds, f.g_aff[i])
    curInterval = xBounds[i]

    if ivIntersection(fxInterval, bi)==[]: return []

    # first check if xBounds can be further reduced:
    if fxInterval in bi: return curInterval

    # Otherwise, iterate each bound of bi:
    fIntervalxLow, fIntervalxUp = getFIntervalsFromXBounds(f, curInterval, 
                                                           xBounds, i)
    if fIntervalxLow.a > bi.b:
        while (not numpy.isclose(float(mpmath.mpf(fxInterval.a)), 
                                 float(mpmath.mpf(fxInterval.b)),relEpsX, absEpsX) 
               and not numpy.isclose(float(mpmath.mpf(curInterval.a)), 
                                     float(mpmath.mpf(curInterval.b)),relEpsX, 
                                     absEpsX)):         
                         curInterval, fxInterval = iteratefBound(f, curInterval, 
                                                                 xBounds, i, bi,
                                                                 increasing = False,
                                                                 lowerXBound = True)
                         if curInterval == [] or fxInterval == []: return []
        
    lowerBound = curInterval.a  
    curInterval  = xBounds[i]        
    fxInterval =  eval_fInterval(f, f.g_mpmath[i], xBounds, f.g_aff[i])
    
    if fIntervalxUp.b < bi.a:
        while (not numpy.isclose(float(mpmath.mpf(fxInterval.a)), 
                                 float(mpmath.mpf(fxInterval.b)),relEpsX, absEpsX) 
               and not numpy.isclose(float(mpmath.mpf(curInterval.a)), 
                                     float(mpmath.mpf(curInterval.b)),relEpsX, 
                                     absEpsX)): 
            curInterval, fxInterval = iteratefBound(f, curInterval, 
                                                    xBounds, i, bi,
                                                    increasing = False,
                                                    lowerXBound = False)
            if curInterval == [] or fxInterval == []: return []
    upperBound = curInterval.b
    
    return mpmath.mpi(lowerBound, upperBound)


def iteratefBound(f, curInterval, xBounds, i, bi, increasing, lowerXBound):
    """ returns the half of curInterval that contains the lower or upper
    bound of bi (biLimit)

    Args:
        :f:                  object of type function
        :curInterval:        X-Interval that contains the solution to f(x) = biLimit
        :xBounds:            numpy array with set of variable bounds
        :i:                  integer with current iteration variable index
        :bi:                 current function residual
        :increasing:         boolean: True = function is monotone increasing,
                             False = function is monotone decreasing
        :lowerXBound:        boolean: True = lower Bound is iterated
                             False = upper bound is iterated

    Return:                  reduced curInterval (by half) and bounds of in curInterval

    """

    biBound = residualBoundOperator(bi, increasing, lowerXBound)

    curlowerXInterval = mpmath.mpi(curInterval.a, curInterval.mid)
    fIntervalxLow, fIntervalxUp = getFIntervalsFromXBounds(f, curlowerXInterval,
                                                           xBounds, i)

    fxInterval = fBoundsOperator(fIntervalxLow, fIntervalxUp, increasing, 
                                 lowerXBound)
    if biBound in fxInterval: return curlowerXInterval, fxInterval

    else:
        curUpperXInterval = mpmath.mpi(curInterval.mid, curInterval.b)
        fIntervalxLow, fIntervalxUp = getFIntervalsFromXBounds(f, curUpperXInterval,
                                                               xBounds, i)

        fxInterval = fBoundsOperator(fIntervalxLow, fIntervalxUp, increasing, 
                                     lowerXBound)
        if biBound in fxInterval: return curUpperXInterval, fxInterval
        else: return [], []


def getFIntervalsFromXBounds(f, curInterval, xBounds, i):
    """ returns function interval for lower variable bound and upper variable
    bound of variable interval curInterval.

    Args:
        :f:              object of type Function
        :curInterval:    current variable interval in mpmath logic
        :xBounds:        set of variable intervals in mpmath logic
        :i:              index of currently iterated variable interval

    Return:              function interval for lower variable bound and upper 
                         variable bound

    """
    curXBoundsLow = list(xBounds)
    curXBoundsUp = list(xBounds)
    curXBoundsLow[i]  = curInterval.a
    curXBoundsUp[i] = curInterval.b

    return (eval_fInterval(f, f.g_mpmath[i], curXBoundsLow, f.g_aff[i]), 
            eval_fInterval(f, f.g_mpmath[i], curXBoundsUp, f.g_aff[i]))


def fBoundsOperator(fIntervalxLow, fIntervalxUp, increasing, lowerXBound):
    """ returns the relevant fInterval bounds for iterating the certain bi
    bound

    Args:
        :fIntervalxLow:   function interval of lower variable bound in mpmath
                          logic
        :fIntervalxUp:    function interval of upper variable bound in mpmath
                          logic
        :increasing:      boolean: True = monotone increasing, False = monotone
                          decreasing function
        :lowerXBound:     boolean: True = lower variable bound, False = upper
                          variable bound

    Return:               relevant function interval for iterating bi bound in 
                          mpmath logic

    """
    if increasing and lowerXBound: return mpmath.mpi(fIntervalxLow.b, fIntervalxUp.b)
    if increasing and not lowerXBound: return mpmath.mpi(fIntervalxLow.a, fIntervalxUp.a)

    if not increasing and lowerXBound: return mpmath.mpi(fIntervalxUp.a, fIntervalxLow.a)
    if not increasing and not lowerXBound: return mpmath.mpi(fIntervalxUp.b, fIntervalxLow.b)


def residualBoundOperator(bi, increasing, lowerXBound):
    """ returns the residual bound that is iterated in the certain case

    Args:
        :bi:              function residual interval in mpmath logic
        :increasing:      boolean: True = monotone increasing, False = monotone
                          decreasing function
        :lowerXBound:     boolean: True = lower variable bound, False = upper
                          variable bound

    Return:               lower or upper bound of function residual interval in 
                         mpmath logic

    """
    if increasing and lowerXBound: return bi.a
    if increasing and not lowerXBound: return bi.b
    if not increasing and lowerXBound: return bi.b
    if not increasing and not lowerXBound: return bi.a


def getMonotoneFunctionSections(f, i, xBounds, dict_options):
    """seperates variable interval into variable interval sets where a function
    with derivative dfdx is monontoneous

    Args:
        :f:                   object of type Function
        :i:                   index of differential variable
        :xBounds:             numpy array with variable bounds
        :dict_options:        dictionary with function and variable interval
                              tolerances

    Return:
        :monIncreasingZone:   monotone increasing intervals
        :monDecreasingZone:   monotone decreasing intervals
        :interval:            non monotone zone if  function interval can not be
                              reduced to monotone increasing or decreasing section

    """
    relEpsX = dict_options["relTol"]
    absEpsX = dict_options["absTol"]
    maxIvNo = dict_options["resolution"]
    monIncreasingZone = []
    monDecreasingZone = []
    org_xiBounds = xBounds[i]
    interval = [xBounds[i]]

    while interval != [] and len(interval) < maxIvNo:
        curIntervals = []

        for xc in interval:
            (newIntervals, newMonIncreasingZone, 
             newMonDecreasingZone) = testIntervalOnMonotony(f, xc, xBounds, i)
            monIncreasingZone = addIntervaltoZone(newMonIncreasingZone,
                                                          monIncreasingZone, 
                                                          dict_options)
            monDecreasingZone = addIntervaltoZone(newMonDecreasingZone,
                                                          monDecreasingZone, 
                                                          dict_options)
            curIntervals += newIntervals

        if checkIntervalWidth(curIntervals, absEpsX, 0.1*relEpsX) == interval:
            interval = joinIntervalSet(interval, relEpsX, absEpsX)
            break
        interval = checkIntervalWidth(curIntervals, absEpsX, 0.1*relEpsX)

    if not len(interval) <= maxIvNo:
        interval = joinIntervalSet(interval, relEpsX, absEpsX)

    if interval == [] and monDecreasingZone == [] and monIncreasingZone ==[]:
        return [], [], [org_xiBounds]
    return monIncreasingZone, monDecreasingZone, interval


def convertIntervalBoundsToFloatValues(interval):
    """ converts mpmath.mpi intervals to list with bounds as float values

    Args:
        :interval:              interval in math.mpi logic

    Return:                     list with bounds as float values

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
        
    Return:
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


# TODO: remove def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
#     import signal

#     class TimeoutError(Exception):
#         pass

#     def handler(signum, frame):
#         raise TimeoutError()

#     # set the timeout handler
#     signal.signal(signal.SIGALRM, handler) 
#     signal.alarm(timeout_duration)
#     try:
#         result = func(*args, **kwargs)
#     except TimeoutError: #as exc:
#         result = False
#         print("Warning: TimeOut")
#     finally:
#         signal.alarm(0)
#     return result


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
            
    Return:
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
        elif bool(dgdxLow <= 0): monotoneIncreasingZone.append(interval)
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


def addIntervaltoZone(newInterval, monotoneZone, dict_options):
    """ adds one or two monotone intervals newInterval to list of other monotone
    intervals. Function is related to function testIntervalOnMonotony, since if  the
    lower and upper part of an interval are identified as monotone towards the same direction
    they are joined and both parts are added to monotoneZone. If monotoneZone contains
    an interval that shares a bound with newInterval they are joined. Intersections
    should not occur afterwards.

    Args:
        :newInterval:         list with interval(s) in mpmath.mpi logic
        :monotoneZone:        list with intervals from mpmath.mpi logic
        :dict_options:        dictionary with variable interval specified tolerances
                              absolute = absTol, relative = relTol

    Return:
        :monotoneZone:        monotoneZone including newInterval

    """
    absEpsX = dict_options["absTol"]
    relEpsX = dict_options["relTol"]
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

    Return:
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
    interval segments of a discretized variable interval and keeps those segments
    that intersect with bi. The discretization resolution is defined in dict_options.

    Args:
        :nonMonotoneZone:    list with non monotone variable intervals
        :reducedIntervals:   lits with reduced non monotone variable intervals
        :f:                  object of type Function
        :i:                  integer with current iteration variable index
        :xBounds:            numpy array with set of variable bounds
        :bi:                 current function residual bounds
        :dict_options:       for function and variable interval tolerances in the 
                             used algorithms and resolution of the discretization

    Return:                  reduced x-Interval(s) and list of monotone x-intervals
        
 """   
    nonMonotoneZone = args["0"]
    reducedIntervals = args["1"]
    f = args["2"]
    i = args["3"]
    xBounds = args["4"]
    bi = args["5"]
    dict_options = args["6"]
    
    relEpsX = dict_options["relTol"]
    precision = getPrecision(xBounds)
    resolution = dict_options["resolution"]

    for curNonMonZone in nonMonotoneZone:
        curInterval = convertIntervalBoundsToFloatValues(curNonMonZone)
        x = numpy.linspace(curInterval[0], curInterval[1], int(resolution))

        fLowValues, fUpValues = getFunctionValuesIntervalsOfXList(x, f.g_mpmath[i], 
                                                                  xBounds, i)       
        for k, fLow_val in enumerate(fLowValues):
            if ivIntersection(mpmath.mpi(fLow_val, fUpValues[k]), bi):
                reducedIntervals.append(mpmath.mpi(x[k], x[k+1]))
    reducedIntervals = joinIntervalSet(reducedIntervals, relEpsX, precision)

    return reducedIntervals


def getFunctionValuesIntervalsOfXList(x, f_mpmath, xBounds, i):
    """ calculates lower and upper function value bounds for segments that are
    members of a list and belong to a discretized variable interval.

    Args:
        :x:          numpy list with segments for iteration variable xi
        :f_mpmath:   mpmath function
        :xBounds:    numpy array with variable bounds in mpmath.mpi.logic
        :i:          index of currently reduced variable

    Return:         list with lower function value bounds within x and upper
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

    Return:
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
    
    Return:          
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
    
    Return:          
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
    
    Return: True for successful method termination     
        
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

        Return:
            :boolean:       True if all intervals are degenerate

    """
    almostEqual = False * numpy.ones(len(X), dtype = bool)
    
    for i, x in enumerate(X):    
         if numpy.isclose(float(mpmath.mpf(x.a)), float(mpmath.mpf(x.b)),
                          relEps, absEps): almostEqual[i] = True

    return almostEqual.all()


def iv_newton(model, box, i, dict_options):
    """ Computation of the Interval-Newton Method to reduce the single 
    interval box[i]
     
    Args: 
        :model:         instance of model class
        :box:           list with current bounds in mpmath.mpi formate
        :i:             index of currently reduced interval as integer
        :dict_options:  dictionary with user settings
            
    Return:
        :y_new:     list with reduced intervals of box[i]
        :unique:    boolean, true if y_new lies in the interior(box[i])    
         
    """  
    x_c=None
    
    if dict_options["newton_point"]=="center": 
        x_c = [float(mpmath.mpf(iv.mid)) for iv in box]
    elif dict_options["newton_point"]=="condJ":
        x_all = [[float(mpmath.mpf(iv.a)) for iv in box],
                 [float(mpmath.mpf(iv.mid)) for iv in box],
                 [float(mpmath.mpf(iv.b)) for iv in box]]    
        condNo = []
        for x in x_all:
            try: condNo.append(numpy.linalg.cond(model.jacobianLambNumpy(*x))) 
            except: condNo.append(numpy.inf)
        #condNo = [numpy.linalg.cond(model.jacobianLambNumpy(*x)) for x in x_all]
        x_c = x_all[condNo.index(min(condNo))]
              
    if dict_options["preconditioning"] == "inverse_centered":
        G_i, r_i, n_x_c, n_box, n_i = get_precondition_centered(model, box, i, x_c)
    elif dict_options["preconditioning"] == "inverse_point":
        G_i, r_i, n_x_c, n_box, n_i = get_precondition_point(model, box, i, x_c)
    elif dict_options["preconditioning"] == "diag_inverse_centered":
        G_i, r_i, n_x_c, n_box, n_i = get_diag_precondition_centered(model, 
                                                                   box, i, x_c)   
    elif dict_options["preconditioning"] == "all_functions":
        y_old = box[i]
        y_new = get_best_from_all_functions(model, box, i, dict_options, x_c)
        return y_new, checkUniqueness(y_new, y_old, dict_options["relTol"],
                                  dict_options["absTol"])
    else:
        j = model.rowPerm[model.colPerm.index(i)]
        G_i, r_i, n_x_c, n_box, n_i = get_org_newton_system(model, box, i, j, x_c)
    
    if dict_options["newton_point"] == "3P":
        x_low = newton_step(r_i, G_i, n_x_c[0], n_box, n_i, dict_options)
        x_up = newton_step(r_i, G_i, n_x_c[1], n_box, n_i, dict_options)
        y_new = setOfIvSetIntersection([mpmath.mpi(x_low.a, x_up.b), y_new])
    else:
        y_new = newton_step(r_i, G_i, n_x_c, n_box, n_i, dict_options)
        
    return y_new, checkUniqueness(y_new, box[i],dict_options["relTol"],
                                  dict_options["absTol"])


def get_best_from_all_functions(model, box, i, dict_options, x_c=None):
    """ reduces variable i in all functions it appears in and intersects
    their results. If no gap occurs box[i] is automatically updated before the 
    next function.
     
    Args:
        :model:         instance of model class
        :box:           list with current bounds in mpmath.mpi formate
        :i:             index of currently reduced bound as integer
        :dict_options:  dictionary with solver settings
        :x_c:           list with currently used point
            
    Return:
        :y_new:         list with reduced interval(s) of variable i
        
    """ 
    y_new = [box[i]]
    for j in model.fWithX[i]:           
        G_i, r_i, n_x_c, n_box, n_i = get_org_newton_system(model, box, i, j, x_c,
                                                            dict_options["newton_point"])      
        if dict_options["newton_point"] == "3P":
            x_low = newton_step(r_i[0], G_i, n_x_c[0], n_box, n_i, dict_options)
            x_up = newton_step(r_i[1], G_i, n_x_c[1], n_box, n_i, dict_options)
            if x_low == [] or x_up == []:
                y_new = []
                break
            y_new = setOfIvSetIntersection([[mpmath.mpi(x_low[0].a, x_up[0].b)], 
                                            y_new])
        else:    
            y_new = setOfIvSetIntersection([newton_step(r_i, G_i, n_x_c, n_box, 
                                                     n_i, dict_options), y_new])
        if y_new == []: break
        elif len(y_new)==1: box[i] = y_new[0]
        
    return y_new
        

def get_diag_precondition_centered(model, box, i, x_c):
    """ preconditions the i-th jacobian row and the i-th resiudal vector entry
    only with the center point of the interval j_ii. If m(j_ii) is 0 the entry 
    is set to 1 and no preconditioning is used
     
    Args:
        :model:         instance of model class
        :box:           list with current bounds in mpmath.mpi formate
        :i:             index of currently reduced bound as integer
        :x_c:           list with currently used point
            
    Return:
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
    Y = get_function_value_as_iv(function, function.dgdx_mpmath[function.glb_ID.index(i)], 
                                 box_sub)  
    Y = float(mpmath.mpf(Y.mid))
    if Y == 0: Y = 1.0
    r_i = function.f_numpy(*x_c_sub) / Y
    
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
        
    Return:
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
            
    Return: see function get_org_newton_system(model, box, i, j, x_c)
        
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
            
    Return: see function get_org_newton_system(model, box, i, j, x_c)
        
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
            
    Return:
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
            
    Return:
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
    

def newton_step(r_i, G_i, x_c, box, i, dict_options):
    """ calculates one newton step in interval-arithmetic
     
    Args: 
        :r_i:           float value with function residual 
        :G_i:           numpy.array with row entries of interval jacobian
        :x_c:           list with currently used point
        :box:           list with current box in mpmath.mpi formate
        :i:             index of currently reduced variable as integer
        :dict_options:  dictionary with user settings
            
    Return:
        :y_new:         list with interval(s) after reduction of current 
                        variable                  
    """             
    
    iv_sum = sum([G_i[j] * (box[j] - x_c[j]) for j in range(len(G_i)) if j!=i])
    if numpy.isinf(r_i) or numpy.isnan(r_i): return [box[i]]  
    try:
        quotient = ivDivision(mpmath.mpi(r_i + iv_sum), G_i[i])
        #N = [x_c[i] - l for l in quotient] 
        N = [x_c[i]*(1 - l/x_c[i]) for l in quotient] # because of round off errors
        y_new = setOfIvSetIntersection([N, [box[i]]])
    except: return [box[i]]
    
    if y_new == []: return check_accuracy_newton_step(box[i], N, dict_options) 
    
    return y_new


def check_accuracy_newton_step(old_iv, new_iv, dict_options):
    """ to prevent discarding almost equal intervals
     
    Args: 
        :old_iv:           interval in mpmath.mpi logic
        :new_iv:           interval in mpmath.mpi logic
        :dict_options:     dictionary with user settings
            
    Return:                 if lower/upper or upper/lower bounds of both 
                            intervals are almost equal they are joined to
                            one interval instead of discarded   
                            
    """      
    if (isinstance(old_iv, mpmath.ctx_iv.ivmpf) and 
        isinstance(new_iv, mpmath.ctx_iv.ivmpf)):    
        if(numpy.isclose(float(mpmath.mpf(old_iv.b)), 
                         float(mpmath.mpf(new_iv.a)), dict_options["absTol"]) or
           numpy.isclose(float(mpmath.mpf(old_iv.a)), 
                         float(mpmath.mpf(new_iv.b)), dict_options["absTol"])):   
            return [mpmath.mpi(min(old_iv.a, new_iv.a), max(old_iv.b, 
                                                            new_iv.b))] 
    return []


def permute_matrix(A, rowPerm, colPerm):
    """permutes matrix based on permutation order 
    
    Args:
        :A:             numpy.array with matrix
        :rowPerm:       list with row order as integers
        :colPerm:       list with column order as integers
    
    Return:             permuted matrix as numpy.array
    
    """
    A = A[rowPerm,:]
    return A[:, colPerm]
    

def real_interval_matrix_product(r_A, iv_B):
    """multiplication of real-valued matrix with interval matrix 
    
    Args:
        :r_A:       numpy.array with real matrix
        :iv_B:      numpy.array with interval matrix (mpmath.mpi entries)
    
    Return:     
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
    
    Return:         inner product of both vectors as interval vector 
        
    """
    return sum([r_vec[i] * iv for i, iv in enumerate(iv_vec)])

            
def identify_function_with_no_solution(output, functions, xBounds, dict_options):
    """ checks which function has no solution in the current function range
    for error report
    
    Args:
        :output:        dictionary with box reduction results
        :functions:     list with function objects
        :xBounds:       numpy.array with mpmath.mpi intervals of variables
        :dict_options:  dictionary with box reduction settings
        
    Return:
        :output:        updated dictionary about failed equation

    """
    for f in functions:
        if not 0 in eval_fInterval(f, f.f_mpmath[0], xBounds, f.f_aff[0],
                                   dict_options["tight_bounds"]):         
            output["noSolution"] = FailedSystem(f.f_sym, f.x_sym[0])
            output["xAlmostEqual"] = False 
            output["xSolved"] = False
            return output
        else:
            return output


def solutionInFunctionRange(functions, xBounds, dict_options):
    """checks, if the solution (0-vector) can lie in these Bounds and returns true or false 
    Args: 
        :model:             instance of class-Model
        :xBounds:           current Bounds of Box
        :dict_options:      options with absTolerance for deviation from the solution
        
    Returns:
        :solutionRange:     boolean that is true if solution in the range
    """

    absTol = dict_options["absTol"]
      
    for f in functions:
        
        if dict_options["Affine_arithmetic"]: 
            fInterval = eval_fInterval(f, f.f_mpmath[0], [xBounds[i] for i in f.glb_ID], f.f_aff[0],
                                       dict_options["tight_bounds"])
        else:
            fInterval = eval_fInterval(f, f.f_mpmath[0], [xBounds[i] for i in f.glb_ID],False, 
                                       dict_options["tight_bounds"])

        if not(fInterval.a<=0+absTol and fInterval.b>=0-absTol):
            return False

    return True


def solutionInFunctionRangePyibex(functions, xBounds, dict_options):
    """checks, if box is empty by reducing it three times with HC4 method
    Args: 
        :functions:             instance of class function
        :xBounds:           current Bounds of Box
        :dict_options:      options with absTolerance for deviation from the solution
        
    Return: boolean that is true if solution is in function range and false otherwise

    """
    xNewBounds = list(xBounds)
    for i in range(3):
        Intersection = HC4(functions, xNewBounds, dict_options)
        if Intersection.is_empty(): return False 
        else:
            xNewBounds = [mpmath.mpi(Intersection[j][0],Intersection[j][1]) 
                          for j in range(0, len(xBounds))] 
 
    return True


def solutionInFunctionRangeNewton(model, xBounds, dict_options):
    """checks, if box is empty by reducing it three times with HC4 method
    Args: 
        :model:             instance of class-Model
        :xBounds:           current Bounds of Box
        :dict_options:      options with absTolerance for deviation from the solution
        
    Return: boolean that is true if solution is in function range and false otherwise
        
    """
    xOld = [list(xBounds)]
    
    for i in range(3):
        xNewBounds = []
        for x in xOld:
            output = reduceBox(numpy.array(x), model, 4, dict_options)
            if not output["xNewBounds"] == []: xNewBounds += output["xNewBounds"]     
        if xNewBounds == []: return False
        
        if xOld == xNewBounds: break       
        else: xOld = xNewBounds
            
    return True


def HC4(functions, xBounds, dict_options):
    """reduces the bounds of all variables in every model function based on HC4 
    hull-consistency
    
    Args:
        :functions:     list with function instances
        :xBounds:       current Bounds of Box
        :dict_options:  dictionary with tolerances
        
    Return: 
        :pyibex IntervalVector with reduced bounds 
        
    """  
    x_HC4 = []
    for x in xBounds:
        if x.delta < dict_options["absTol"]: tol = dict_options["absTol"]
        else: tol = 0.0
               
        x_HC4.append([float(mpmath.mpf(x.a))-0.1*tol, 
                    float(mpmath.mpf(x.b))+0.1*tol])  #keep Bounds in max tolerance to prevent rounding error

    for f in functions:
        sub_box = pyibex.IntervalVector([x_HC4[i] for i in f.glb_ID])
        currentIntervalVector = pyibex.IntervalVector(sub_box)
        f.f_pibex.contract(currentIntervalVector)
        sub_box = sub_box & currentIntervalVector
        if sub_box.is_empty(): # TODO: store currentIntervalVector and f in dict_options for error analysis
            dict_options["failed_subbox"] = pyibex.IntervalVector([x_HC4[i] for 
                                                                   i in f.glb_ID])
            dict_options["failed_function"] = f
            return sub_box
        else:
            for i,val in enumerate(f.glb_ID): x_HC4[val] =  list(sub_box[i])
    return pyibex.IntervalVector(x_HC4) 
        

def checkUniqueness(new_x, old_x,relEpsX,absEpsX):
    """ checks if the condition for a unique solution in a box is fulfilled which
    states that the new_x must be a subinterval of the interior of old_x. This 
    has to be met by all x of a box. 
    
    Args:
        :new_x:     list with intervals in the formate mpmath.mpi
        :old_x:     interval in the formate mpmath.mpi
        :relEpsX:   relative tolerance
        :absEpsX:   absolute tolerance
    
    Return: 
        boolean true if the criterion is fulfilled and false otherwise
    
    """
    for i,x in enumerate(new_x):
        if x.a > old_x.a and x.b < old_x.b:
            continue
        elif numpy.isclose(float(mpmath.mpf(x.a)), float(mpmath.mpf(x.b)), 
                           relEpsX, absEpsX):
            continue
        else: return False
    return True


def lookForSolutionInBox(model, boxID, dict_options, sampling_options, solv_options):
    """Uses Matlab File and tries to find Solution with initial points in the box samples by HSS.
     Writes Results in File, if one is found: 
    
    Args: 
        :model:            instance of class model
        :boxID:            id of current Box
        :dict_options:     dictionary of options  
        :sampling_options: dictionary with sampling settings
        :solv_options:     dicionary with settings for numerical solver
        
    Return:
        :solved:           boolean that is true for successful iteration
                                     
    """
    if dict_options["Debug-Modus"]: print("Box no. ", boxID, "is now numerically iterated.")
    dict_options["sampling"]=True
    dict_options["scaling"]= "None"
    dict_options["scaling procedure"]="None"
    solved = False
    allBoxes = list(model.xBounds)
    
    model.xBounds = ConvertMpiBoundsToList(model.xBounds,boxID)
    results = mos.solveBlocksSequence(model, solv_options, dict_options, 
                                      sampling_options)
    
    if results != {} and not results["Model"].failed:
        if not "FoundSolutions" in dict_options.keys():  
            dict_options["FoundSolutions"] = copy.deepcopy(model.stateVarValues)
            solv_options["sol_id"] = 1
            mos.results.write_successful_results({0: results}, dict_options, 
                                                 sampling_options, solv_options) 
        else:    
            for new_solution in copy.deepcopy(model.stateVarValues):
                sol_exist = False
                for solution in dict_options["FoundSolutions"]: 
                    if numpy.allclose(numpy.array(new_solution), 
                                      numpy.array(solution),
                                      dict_options["relTol"],
                                      dict_options["absTol"]):
                        sol_exist = True
                        break
                if not sol_exist: 
                    dict_options["FoundSolutions"].append(new_solution)  
                    solv_options["sol_id"] += 1
                    mos.results.write_successful_results({0: results}, dict_options, 
                                                 sampling_options, solv_options) 
            
    if model.failed: model.failed = False
    else: solved = True 
    model.xBounds = allBoxes
    return solved

  
def ConvertMpiBoundsToList(xBounds, boxID):
    """Converts the xBounds, containing mpi to a list for sampling methods
    
    Args: 
        :xBounds:   array of bounds as mpmath.mpi
        :boxID:   id of current Box      
        
    Return:
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
        
    Return:
        :x:     zero- or inf- free x-interval

    """    
    if x.a > numpy.nan_to_num(numpy.inf) : return x
    if x.b < -numpy.nan_to_num(numpy.inf) : return x
    if 0.0 <= x.a < 1.0/numpy.nan_to_num(numpy.inf) and x.b < 1.0/numpy.nan_to_num(numpy.inf): return x
    if 0.0 >= x.b > -1.0/numpy.nan_to_num(numpy.inf) and x.a > -1.0/numpy.nan_to_num(numpy.inf): return x
    if 0.0 <= x.a < 1.0/numpy.nan_to_num(numpy.inf) and x.b < 1.0/numpy.nan_to_num(numpy.inf): return x
    elif 0.0 > x.a > -1.0/numpy.nan_to_num(numpy.inf) : x = mpmath.mpi(-1/numpy.nan_to_num(numpy.inf), x.b)
    if 0.0 <= x.b < 1.0/numpy.nan_to_num(numpy.inf) : x = mpmath.mpi(x.a, 1/numpy.nan_to_num(numpy.inf))
    elif 0.0 > x.b > -1.0/numpy.nan_to_num(numpy.inf) : x = mpmath.mpi(x.a, -1/numpy.nan_to_num(numpy.inf))
    if x.b > numpy.nan_to_num(numpy.inf) : x = mpmath.mpi(x.a, numpy.nan_to_num(numpy.inf))
    if x.a < -numpy.nan_to_num(numpy.inf) : x = mpmath.mpi(-numpy.nan_to_num(numpy.inf),x.b)
    return x


def saveSolutions(model, dict_options):
    """Writes all found results to a txt file 
    
        Args: 
            :model:   instance of class model
            :dict_options:   dictionary of options      
            
        Return:
             
        """
    file=open(f'{dict_options["fileName"]}{len(model.FoundSolutions)}Solutions.txt', 
              'a')
    for fs in model.FoundSolutions:
        for var in range(len(fs)):
            file.write(str(model.xSymbolic[var])+' '+str(fs[var]))
            file.write('  \n')
        file.write('\n')
    file.close()

    
def split_least_changed_variable(box_new, model, k, dict_options):
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
        :dict_options:      dictionary with user-specified reduction settings
    
    Return: see getBestTearSplit(box_new, model, boxNo, dict_options, w_max_ids) 
               
    """
    w_ratio = []
    r = model.complete_parent_boxes[k][0]
    box_ID = model.complete_parent_boxes[k][1]
    boxNo = len(model.xBounds)
    
    box_old = mostg.get_entry_from_npz_dict(dict_options["fileName"]+"_boxes.npz", 
                                            r, allow_pickle=True)[box_ID]
    
    for i in range(0, len(box_new[0])):
        if (mpmath.mpf(box_new[0][i].delta)>= dict_options["absTol"]):
                                   w_ratio.append(
                                       float(mpmath.mpf(box_new[0][i].delta))
                                       / (box_old[i][1] - box_old[i][0])) 
        else: w_ratio.append(0.0) 
    w_max_ids = [i for i, j in enumerate(w_ratio) if j == max(w_ratio)]
    if model.fCounts == []: model.fCounts = [len(model.dict_varId_fIds[i]) 
                                             for i in range(0, len(box_new[0]))]
    if len(w_max_ids) > 3:
        splitVar = []
        least_changed_intervals = [ model.fCounts[i] for i in w_max_ids]
        for i in range(3):
            splitVar.append(w_max_ids.pop(
                least_changed_intervals.index(max(least_changed_intervals))))
            least_changed_intervals.pop(
                least_changed_intervals.index(max(least_changed_intervals)))
        w_max_ids = splitVar

    return getBestTearSplit(box_new, model, boxNo, dict_options, w_max_ids) 


def get_index_of_boxes_for_reduction(xSolved, xAlmostEqual, maxBoxNo):
    """ creates list for all boxes that can still be reduced
    Args:
        :xAlmostEqual:      list with boolean that is true for complete boxes
        :maxBoxNo:          integer with current maximum number of boxes
        
    Return:
        :ready_for_reduction:   list with boolean that is true if box is ready 
                                for reduction
                                
    """
    complete = []
    incomplete = []
    solved =[]
    not_solved_but_complete=[]
    ready_for_reduction = len(xAlmostEqual) * [False]
    
    for i, val in enumerate(xSolved):
        if val: solved +=[i]
    
    for i,val in enumerate(xAlmostEqual):
        if val: 
            complete += [i]
            if not i in solved: not_solved_but_complete +=[i] 
        else: incomplete += [i]
   
    nl = max(0, min(len(not_solved_but_complete), maxBoxNo - len(xAlmostEqual)))
    
    for i in range(nl): ready_for_reduction[not_solved_but_complete[i]] = True
    for i in incomplete: ready_for_reduction[i] = True
    
    return ready_for_reduction