"""
***************************************************
Import packages
***************************************************
"""
import time
import copy
import numpy
import sympy
import mpmath
import pyibex
import itertools
import warnings
from modOpt.constraints import parallelization
from modOpt.constraints.FailedSystem import FailedSystem
from modOpt.decomposition import MC33
from modOpt.decomposition import dM
import modOpt.constraints.realIvPowerfunction # redefines __power__ (**) for ivmpf


__all__ = ['reduceMultipleXBounds', 'reduceXIntervalByFunction', 'setOfIvSetIntersection',
           'checkWidths', 'getPrecision', 'getNewtonIntervalSystem', 'saveFailedSystem', 
            'variableSolved']

"""
***************************************************
Algorithm for interval Nesting procedure
***************************************************
"""

def reduceMultipleXBounds(model, functions, dict_varId_fIds, dict_options):
    """ reduction of multiple solution interval sets  
    Args:    
        :model:                 object of type model
        :functions:             list with function instances
        :dict_varId_fIds:       dictionary with variable's glb id (key) and list 
                                with function's glb id they appear in    
        :dict_options:          dictionary with user specified algorithm settings

    Return:
        :results:               dictionary with newXBounds and a list of booleans
                                that marks all non reducable variable bounds with True.
                                If solver terminates because of a NoSolution case the
                                critical equation is also stored in results for the error
                                analysis.

    """
    results = {}
    allBoxes = []
    xAlmostEqual = False * numpy.ones(len(model.xBounds), dtype=bool)
    xSolved = False * numpy.ones(len(model.xBounds), dtype=bool)

    boxNo = len(model.xBounds)
    nl = len(model.xBounds)
    for k in range(0, len(model.xBounds)):
        
        xBounds = model.xBounds[k]

        newtonMethods = {'newton', 'detNewton', '3PNewton'}
        if dict_options['newton_method'] in newtonMethods and dict_options['combined_algorithm']==False:
            newtonSystemDic = getNewtonIntervalSystem(xBounds, model, dict_options)  
            if nl==len(xBounds)-1: nl = 0
            else: nl = nl+1 
        else:
            newtonSystemDic = {}
        
        #if dict_options['method'] == 'b_tight':
        #    output = reduceXbounds_b_tight(functions, xBounds, boxNo, dict_options)
        #else:
        if not dict_options["Parallel Variables"]:
            if dict_options["combined_algorithm"]==True:
                output = reduceBoxCombined(xBounds, model, functions, dict_options)
            else:             
                output = reduceBox(xBounds, model, functions, dict_varId_fIds, boxNo, dict_options, newtonSystemDic)
        else: 
            output = parallelization.reduceBox(xBounds, model, functions, 
                                                       dict_varId_fIds, boxNo, dict_options, newtonSystemDic)

        xNewBounds = output["xNewBounds"]
        xAlmostEqual[k] = output["xAlmostEqual"]
        xSolved[k] = output["xSolved"]


        if output["xAlmostEqual"] and not output["xSolved"]:
            boxNo_split = dict_options["maxBoxNo"] - boxNo
            #if model.tearVarsID == []: getTearVariables(model)
            #xNewBounds = separateBox(model.xBounds[k], model.tearVarsID, boxNo_split)
            splitVar = getTearVariableLargestDerivative(model, k)
            xNewBounds, dict_options["tear_id"] = splitTearVars(splitVar, 
                                       model.xBounds[k], boxNo_split, dict_options)
            #xAlmostEqual[k] = False


        if output.__contains__("noSolution") :
            saveFailedIntervalSet = output["noSolution"]
            boxNo = len(allBoxes) + (nl - (k+1))
            continue
        
        for box in xNewBounds: allBoxes.append(numpy.array(box, dtype=object))
     
        boxNo = len(allBoxes) + (nl - (k+1))
  
        if boxNo >= dict_options["maxBoxNo"]:
            print("Note: Algorithm stops because the current number of boxes is ",
                  boxNo,
                  "and exceeds or equals the maximum number of boxes that is ",
                  dict_options["maxBoxNo"], "." )
            for l in range(k+1, nl):
                allBoxes.append(model.xBounds[l])
            break
        
    if allBoxes == []: 
        results["noSolution"] = saveFailedIntervalSet

    else:
        model.xBounds = allBoxes    
        
    results["xAlmostEqual"] = xAlmostEqual
    results["xSolved"] = xSolved
    return results

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
    
    maxJacpoint = []
    for p in ['a','mid','b']:
        PointIndicator = len(model.xBounds[boxNo])*[p]
        Boxpoint = getPointInBox(model.xBounds[boxNo], PointIndicator)
        Jacpoint = jaclamb(*Boxpoint)
        Jacpoint = numpy.nan_to_num(Jacpoint)
        maxJacpoint.append(numpy.max(abs(Jacpoint), axis=0))
   
    maxJacpoint = model.VarFrequency*numpy.max(maxJacpoint, axis=0)

    #sum of derivatives
    largestJacIVVal = -numpy.inf
    largestJacIVVarID = [0]
    for i in subset:
           if abs(maxJacpoint[i])>largestJacIVVal and float(
                   mpmath.convert(model.xBounds[boxNo][i].delta))>0.0001:
               largestJacIVVal = abs(maxJacpoint[i])
               largestJacIVVarID = [i]
    splitVar = largestJacIVVarID
    
    return splitVar


def splitTearVars(tearVarIds, box, boxNo_max, dict_options):
    """ splits unchanged box by one of its alternating tear variables
    
    Args:
        :tearVarIds:    list with global id's of tear variables
        :box:           numpy array intervals in mpmath.mpi formate
        :boxNo_max:     currently available number of boxes to maximum
        :dic_options:   dictionary with user specific settings
    
    Return: two sub boxes bisected by alternating tear variables from dict_options
        
    """
    
    if tearVarIds == [] or boxNo_max <= 0 : return [box], dict_options["tear_id"]
    
    iN = getCurrentVarToSplit(tearVarIds, box, dict_options)
    if iN == []: return [box], dict_options["tear_id"]
    
    return separateBox(box, [tearVarIds[iN]]), iN + 1


def getCurrentVarToSplit(tearVarIds, box, dict_options):
    """ returns current tear variable id in tearVarIds for bisection. Only tear
    variables with nonzero widths are selectd. 
    
    Args:
        :tearVarIds:    list with global id's of tear variables
        :box:           numpy array intervals in mpmath.mpi formate   
        :dic_options:   dictionary with user specific settings  
        
    Return:
        :i:             current tear variable for bisection
        
    """
    
    i = dict_options["tear_id"]
    while checkIntervalWidth([box[i]], dict_options["absTol"],
                             dict_options["relTol"]) == [] and \
                            i  <= len(tearVarIds) - 1:
                                i+=1
    if i > len(tearVarIds)-1:     
        nondeg_intervals = checkIntervalWidth(box[tearVarIds], dict_options["absTol"],
                            dict_options["relTol"])
        if nondeg_intervals == []: return []
        else: return tearVarIds.index(list(box).index(nondeg_intervals[0]))
    else: return i
    
        
def separateBox(box, varID):
    """ bisects a box by variables with globalID in varID
    
    Args:
        :box:       numpy.array with variable bounds
        :varID:     list with globalIDs of variables chosen for bisection        
        
    Return:
        numpy.array wit sub boxes
        
    """   
   
    for i in range(0,len(box)):
        if i in varID:
          box[i]=[mpmath.mpi(box[i].a, box[i].mid), mpmath.mpi(box[i].mid, box[i].b)]
        else:box[i]=[box[i]]
        
    return list(itertools.product(*box))


def getNewtonIntervalSystem(xBounds, model, options):
    '''calculates all necessary System matrices for the Newton-Interval reduction
    Args:
        :xBounds:           list with variable bounds in mpmath.mpi formate
        :model:             instance of type model
        :options:           dictionary options

    Return:
        :Boxpoint:          list with Boxpoint/s values
        :fpoint:            functions of Boxpoint/s in numpy array
        :Jacpoint:          jacobian of Boxpoint/s in numpy array
        :JacInterval:       jacobian of ivmpf intervals in numpy array
        :JacInv:            inverse of Jacpoint in numpy array, stripped of inf Parts
    '''
    
    flamb = model.fLamb
    jaclamb = model.jacobianLambNumpy
    jacIvLamb = model.jacobianLambMpmath
    

    if options['newton_method'] == "detNewton":
        Boxpoint, Jacpoint, fpoint = findPointWithHighestDeterminant(jaclamb, flamb, xBounds)
    elif options['newton_method'] == "3PNewton":
        Boxpoint, Jacpoint, fpoint = getFunctionPoint(jaclamb, flamb, xBounds, ['a','mid','b'])
    else:
        Boxpoint, Jacpoint, fpoint = getFunctionPoint(jaclamb, flamb, xBounds, ['mid'])
    
    JacInterval = numpy.array(jacIvLamb(*xBounds)) 

    # converts 'inf' in numpy float-inf   
    infRows = []           
    for l in range(0, len(JacInterval)):
        for n, iv in enumerate(JacInterval[l]):
            if isinstance(iv, mpmath.ctx_iv.ivmpf):
                if iv == mpmath.mpi('-inf','+inf'):
                    JacInterval[l][n] = mpmath.mpi(numpy.nan_to_num(-numpy.inf), numpy.nan_to_num(numpy.inf))
                    if l not in infRows:
                        infRows.append(l)
                elif iv.a == '-inf': 
                    JacInterval[l][n] = mpmath.mpi(numpy.nan_to_num(-numpy.inf), iv.b)
                elif iv.b == '+inf':
                    JacInterval[l][n] = mpmath.mpi(iv.a, numpy.nan_to_num(numpy.inf))
    
    JacInv = []
    if options["InverseOrHybrid"]!='Hybrid':
        for J in Jacpoint:
            try:
                JacInv.append(numpy.linalg.inv(J))
            except: 
                print('singular point')
                JacInv.append(numpy.array(numpy.matrix(J).getH())) # if singular return adjunct 
        
        #remove inf parts in inverse
        for inf in infRows:
            for Ji in JacInv:
                Ji[:,inf] = numpy.zeros(len(Ji))
          
    return {'Boxpoint':Boxpoint, 'f(Boxpoint)':fpoint, 'J(Boxpoint)':Jacpoint,
            'J(Box)':JacInterval, 'J(Boxpoint)-1':JacInv,
            'infRows': infRows}


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

def findPointWithHighestDeterminant(jacLamb, fLamb, xBounds): 
    '''finds Boxpoint with highest determinant of J(Boxpoint)
    Args:
        :jacLamb: 		lambdifyed Jacobian 
        :fLamb: 		lambdifyed function
        :xBounds: 		current xBounds as iv.mpi
    Return:
        :chosenPoint: 	List of variable Point in Box
        :chosenJacobi: 	Jacobian evaluated at chosen Point
        :chosenFunc: 	function evaluated at chosen Point
    '''
    
    lowerPoint = getPointInBox(xBounds, len(xBounds)*['a'])
    midPoint = getPointInBox(xBounds, len(xBounds)*['mid'])
    upperPoint = getPointInBox(xBounds, len(xBounds)*['b'])
    TestingPoints = [lowerPoint, midPoint, upperPoint]
    
    biggestDValue = -numpy.inf    
    for tp in TestingPoints:
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        currentJacpoint = jacLamb(*tp)
        currentFunc = numpy.array([fLamb(*tp)])
        warnings.filterwarnings("default", category=RuntimeWarning)
        currentJacpoint = removeInfAndConvertToFloat(currentJacpoint, 1)
        currentFunc = removeInfAndConvertToFloat(currentFunc, 1)
        currentAbsDet = numpy.absolute(numpy.linalg.det(currentJacpoint))
        decisionValue = currentAbsDet - numpy.sum(numpy.absolute(currentFunc[0]))
        if decisionValue>=biggestDValue:
            biggestDValue = decisionValue
            chosenPoint = tp
            chosenJacobi = currentJacpoint
            chosenFunc = currentFunc
     
    return [chosenPoint], [chosenJacobi], [chosenFunc[0]]


def getFunctionPoint(jacLamb, fLamb, xBounds, points):
    
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    Boxpoint = []
    Jacpoint = []
    fpoint = []
    for p in points:
        PointIndicator = len(xBounds)*[p]
        singlePoint = getPointInBox(xBounds, PointIndicator)
        Boxpoint.append(singlePoint)
        Jacpoint.append(removeInfAndConvertToFloat(jacLamb(*singlePoint), 1))
        fpoint.append(removeInfAndConvertToFloat(numpy.array([fLamb(*singlePoint)]), 1)[0])
    warnings.filterwarnings("default", category=RuntimeWarning)
    
    return Boxpoint, Jacpoint, fpoint


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
            if iv == float('inf') or iv == float('-inf'):
                array[l][n] = subs
            if iv == mpmath.mpi('-inf','+inf'):
                array[l][n] = subs
            elif isinstance(iv, mpmath.ctx_iv.ivmpf):
                if iv.a == '-inf': iv = mpmath.mpi(iv.b, iv.b)
                if iv.b == '+inf': iv = mpmath.mpi(iv.a, iv.a)
                array[l][n] = float(mpmath.mpmathify(iv.mid))
    
    array = numpy.array(array, dtype='float')
    
    return array

def reduceXbounds_b_tight(functions, xBounds, boxNo, dict_options):
    """ main function to reduce all variable bounds with all equations

    Args:
        :functions:         list with instances of class function
        :xBounds:           list with variable bounds in mpmath.mpi formate
        :boxNo:             number of current boxes
        :dict_options:      dictionary with tolerance options

    Return:
        :output:            dictionary with new interval sets(s) in a list and
                            eventually an instance of class failedSystem if
                            the procedure failed.

    """

    varBounds = {}
    maxBoxNo = dict_options["maxBoxNo"]

    for k in range(0, len(functions)):

        f = functions[k]
        if dict_options["Debug-Modus"]: print(k)

        if not dict_options["Parallel Variables"]:
            reduceXBounds_byFunction(f, numpy.array(xBounds, dtype=None)[f.glb_ID], dict_options, varBounds)
            #if varBounds.__contains__('3'): print(varBounds['3'])
        else:
            parallelization.reduceXBounds_byFunction(f, numpy.array(xBounds, dtype=None)[f.glb_ID], dict_options, varBounds)

        if varBounds.__contains__('Failed_xID'):
            return get_failed_output(f, varBounds)
        
    output = get_newXBounds(varBounds, boxNo, maxBoxNo, xBounds, dict_options)

    #if len(output["intervalsPerm"]) > dict_options["maxBoxNo"]:
    #    print("Note: Algorithm stops because the current number of boxes is ",
    #    len(output["intervalsPerm"]),
    #    "and exceeds the maximum number of boxes that is ",
    #    dict_options["maxBoxNo"], "." )
    #    output["xAlmostEqual"] = True

    return output


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


def get_newXBounds(varBounds, boxNo, maxBoxNo, xBounds, dict_options):
    """ creates output dictionary with new variable bounds in a successful
    reduction process. Also checks if variable bounds change by xUnchanged.

    Args:
        :varBounds:     dictionary with reduced variable bounds
        :boxNo:         number of current boxes in stack
        :maxBoxNo:      maximum number of boxes allowed
        :xBounds:       numpy array with variable bounds in mpmath.mpi formate       
    
    Return:
        :output:        dictionary with information about variable reduction

    """
    output = {}
    xNewBounds = []
    xUnchanged = True
    xSolved = True
    subBoxNo = 1

    for i in range(0, len(xBounds)):

        if ((boxNo-1) + subBoxNo * len(varBounds['%d' % i])) > maxBoxNo:
            for restj in range(i,len(xBounds)):
                xNewBounds.append([xBounds[restj]])
            output["xAlmostEqual"] = xUnchanged
            output["xSolved"] = False
            output["xNewBounds"] = list(itertools.product(*xNewBounds))
            return output

        subBoxNo = subBoxNo * len(varBounds['%d' % i])
        xNewBounds.append(varBounds['%d' % i])

        if varBounds['%d' % i] != [xBounds[i]] and xUnchanged:
            xUnchanged = False
            
        for varB in varBounds['%d' % i]:
            if not checkVariableBound(varB, dict_options): xSolved = False

    output["xAlmostEqual"] = xUnchanged
    output["xSolved"] = xSolved
    output["xNewBounds"] = list(itertools.product(*xNewBounds))
    return output


def reduceXBounds_byFunction(f, xBounds, dict_options, varBounds):
    """ reduces all n bounds of variables (xBounds) that occur in a certain
    function f and stores it in varBounds.

    Args:
        :f:             instance of class function
        :xBounds:       list with n variable bounds
        :dict_options:  dictionary with user-specified settings
        :varBounds:     dictionary with n reduced variable bounds. The key 
                        'Failed_xID' is used to store a variable's global ID in
                        case  a reduced interval is empty.

    """
    dict_options_temp = copy.deepcopy(dict_options)
    #TODO
    for x_id in range(0, len(f.glb_ID)): # get g(x) and b(x),y
    
        if xBounds[x_id].delta <= 1.0e-15: 
            store_reduced_xBounds(f, x_id, [xBounds[x_id]], varBounds)
            continue     
        
        if variableSolved([xBounds[x_id]], dict_options) and xBounds[x_id].delta > 1.0e-15:
            dict_options_temp["relTol"] = 0.1 * xBounds[x_id].delta
            dict_options_temp["absTol"] = 0.1 * xBounds[x_id].delta

        #if mpmath.almosteq(xBounds[x_id].a, xBounds[x_id].b,
        #                   dict_options["absTol"],
        #                   dict_options["relTol"]):
        #    store_reduced_xBounds(f, x_id, [xBounds[x_id]], varBounds)
        #    continue
        if dict_options["Parallel b's"]:
            b = parallelization.get_tight_bBounds(f, x_id, xBounds, dict_options)

        else:
            b = get_tight_bBounds(f, x_id, xBounds, dict_options) # TODO: Parallel
            if b == mpmath.mpi('-inf','inf') or b == []: reduced_xBounds = [xBounds[x_id]]
            else: reduced_xBounds = get_reducedxBounds(f, b, x_id, copy.deepcopy(xBounds), dict_options_temp)

            #if reduced_xBounds == [xBounds[x_id]] and f.b_sym[x_id].free_symbols!=set():
                #if x_id==7: print ("Before ", f.x_sym[7], " ", b)
                #b = get_b_from_branching(f, x_id, xBounds, dict_options) 
                #if x_id==7: print ("Behind ", f.x_sym[7], " ", b)
                #if not b == mpmath.mpi('-inf','inf') and b != []:
                #    reduced_xBounds = get_reducedxBounds(f, b, x_id, copy.deepcopy(xBounds), dict_options)
            
        store_reduced_xBounds(f, x_id, reduced_xBounds, varBounds)
 

def get_b_from_branching(f, x_id, xBounds, dict_options):
    """ splits variable intervals of right hand side of an equation so that sub
    interval are evaluated
    
    Args:
        :f:             instance of class Function
        :xBounds:       numpy array with current variable bounds in 
                        mpmath.mpi formate
        :dict_options:  dictionary with user-specified settings
    
    Return:             b interval in mpmath.mpi formate
    """
    
    xBounds_disc = []
    b_intervals = []
    
    resolution = int(numpy.ceil(dict_options["maxBoxNo"]**(1/float(len(xBounds)))))
    noBox = int(dict_options["maxBoxNo"] / resolution)
    oneBox = dict_options["maxBoxNo"] - noBox
    j = 0
    b = lambdifyToMpmathIvComplex(f.x_sym, f.b_sym[x_id])
    for interval in xBounds:
        iv_disc = []
        #if interval == xBounds[x_id]:
        #    xBounds_disc.append([interval])
        #    continue
        el = mpmath.mpf(interval.delta)/resolution
        lb = interval.a
        if j <= noBox:
            for i in range(0, noBox):      
                iv_disc.append(mpmath.mpi(lb, lb + el))
                lb = lb + el
                j+=1
        else: iv_disc = [interval]                       
        xBounds_disc.append(iv_disc)
    
    for i in range(oneBox, len(xBounds)):
        xBounds_disc.append([xBounds[i]])
    
    boxes = list(itertools.product(*xBounds_disc))
    
    for box in boxes:
        try: b_intervals.append(mpmath.mpi(b(*box)))
        except: continue   
    
    if b_intervals == []: return []
    return mpmath.mpi(min(b_intervals).a, max(b_intervals).b)
                    
           
def get_reducedxBounds(f, b, x_id, xBounds, dict_options):
    """ reduces an variable interval from xBounds with global id x_id
    
    Args:
    :f:             instance of class Function
    :b:             interval of equation's right-hand side in mpmath.mpi
    :x_id:          global id of currently reduced variable interval
    :xBounds:       numpy array with current variable intervals
    :dict_options:  dictionary with user-specified settings

    Return:     List with reduced variable intervals
    
    """
    
    if b == []: 
        return [xBounds[x_id]]
    
    else: 
        return reduce_x_by_gb(f.g_sym[x_id], f.dgdx_sym[x_id],
                                                     b, f.x_sym, x_id,
                                                     xBounds,
                                                     dict_options)      


def store_reduced_xBounds(f, x_id, reduced_interval, varBounds):
    """ stores reduced interval of x in dictionary varBounds

    Args:
        :f:                 instance of class function
        :x_id:              index (integer) of current variable bound that is reduced
        :reduced_interval:  list with reduced x intervals in mpmath.mpi formate
        :varBounds:         dictionary with n reduced variable bounds. The key
                            'Failed_xID' is used to store a variable's global ID in
                            case  a reduced interval is empty.

    """


    if not varBounds.__contains__('%d' % f.glb_ID[x_id]):
        varBounds['%d' % f.glb_ID[x_id]] = reduced_interval
    else: 
        varBounds['%d' % f.glb_ID[x_id]] = setOfIvSetIntersection([varBounds['%d' % f.glb_ID[x_id]],
                 reduced_interval])
    
    if varBounds['%d' % f.glb_ID[x_id]] == [] or varBounds['%d' % f.glb_ID[x_id]] == [[]]: 
                varBounds['Failed_xID'] = x_id


def get_tight_bBounds(f, x_id, xBounds, dict_options):
    """ returns tight b bound interval based on all variables y that are evaluated
    separatly for all other y intervals beeing constant.

    Args:
        :f:                 instance of class function
        :x_id:              index (integer) of current variable bound that is reduced
        :xBounds:           list with variable bonunds in mpmath.mpi formate        
        :dict_options:      dictionary with tolerance options

    Return:
        b interval in mpmath.mpi formate and [] if error occured (check for complex b)

    """
    
    b_max = []
    b_min = []
    #digits = int(abs(numpy.floor(numpy.log10(dict_options['absTol']))))
    b = getBoundsOfFunctionExpression(f.b_sym[x_id], f.x_sym, xBounds)

    if b == []: return []

    #if mpmath.almosteq(b.a, b.b, dict_options["relTol"], dict_options["absTol"]):
    #    return b

    if len(f.glb_ID)==1: # this is import if b is interval but there is only one variable in f (for design var intervals in future)
        return b

    for y_id in range(0, len(f.glb_ID)): # get b(y)

        if  x_id != y_id:
            get_tight_bBounds_y(f, x_id, y_id, xBounds, b_min, b_max, dict_options)

    if b_min !=[] and b_max != []:

        b_lb = max(b_min) #round_off(max(b_min), digits)
        b_ub = min(b_max) #round_up(min(b_max), digits)
        return mpmath.mpi(b_lb, b_ub)
    else: return []


def round_off(n, decimals=0): 
    """rounds number n off by decimals
    """
    
    multiplier = 10 ** decimals 
    return numpy.floor(n * multiplier) / multiplier


def round_up(n, decimals=0): 
    """rounds number n up by decimals
    """
    
    multiplier = 10 ** decimals 
    return numpy.ceil(n * multiplier) / multiplier


def get_tight_bBounds_y(f, x_id, y_id, xBounds, b_min, b_max, dict_options):
    """ reduces bounds of b as function of y.

    Args:
        :f:                 instance of class function
        :x_id:              index (integer) of current x variable bound that is reduced
        :y_id:              index (integer) of current y variable bound that is reduced
        :xBounds:           list with variable bonunds in mpmath.mpi formate
        :b_min:             list with minimum values of b
        :b_max:             list with maximum values of b
        :dict_options:      dictionary with tolerance options

    """

    if f.dbdx_sym[x_id][y_id] == 0:
        b = getBoundsOfFunctionExpression(f.b_sym[x_id], f.x_sym, xBounds)
        if not b == []:
            b_min.append(float(mpmath.mpf(b.a)))
            b_max.append(float(mpmath.mpf(b.b)))      
    else:        
        incr_zone, decr_zone, nonmon_zone = get_conti_monotone_intervals(f.dbdx_sym[x_id][y_id], 
                                                                         f.x_sym, 
                                                                         y_id, 
                                                                         copy.deepcopy(xBounds), 
                                                                         dict_options)    
        add_b_min_max(f, incr_zone, decr_zone, nonmon_zone, x_id, y_id, 
                      copy.deepcopy(xBounds), b_min, b_max, dict_options)    
    

def add_b_min_max(f, incr_zone, decr_zone, nonmon_zone, x_id, y_id, xBounds, b_min, b_max, dict_options):
    """ gets minimum and maximum b(x,y) values for continuous monotone-increasing/-decreasing or 
    non-monotone section. If section b is non-monotone the complete interval is evaluated with
    common interval arithmetic. The upper bound of b is put to b_max and th lower bound to b_min.

    Args:
        :f:                 instance of class function
        :incr_zone:         list with monotone-increasing intervals in mpmath.mpi
                            formate
        :decr_zone:         list with monotone-decreasing intervals in mpmath.mpi
                            formate
        :nonmon_zone:       list with non-monotone intervals in mpmath.mpi formate
        :x_id:              index (integer) of current x-variable
        :y_id:              index (integer) of current y-variable
        :xBounds:           list with current bounds of variables in f in mpmath.mpi
                            formate
        :b_min:             list with lower bounds of b as float numbers
        :b_max:             list with upper bounds of b as float numbers
        :dict_options:      dictonary with user-specified settings
    """
    
    cur_b_min = []
    cur_b_max = []
    if incr_zone != []: get_bounds_incr_zone(f.b_sym[x_id],
                                             f.x_sym,
                                             y_id,
                                             copy.deepcopy(xBounds),
                                             incr_zone,
                                             cur_b_min,
                                             cur_b_max)

    if decr_zone != []: get_bounds_decr_zone(f.b_sym[x_id],
                                             f.x_sym,
                                             y_id,
                                             copy.deepcopy(xBounds),
                                             decr_zone,
                                             cur_b_min,
                                             cur_b_max)

    if nonmon_zone != []: get_bounds_nonmon_zone(f.b_sym[x_id],
                                                 f.x_sym,
                                                 y_id,
                                                 copy.deepcopy(xBounds),
                                                 nonmon_zone,
                                                 cur_b_min,
                                                 cur_b_max,
                                                 dict_options)                                                      
    
    if cur_b_max != []:  b_max.append(max(cur_b_max))
    if cur_b_min != []:  b_min.append(min(cur_b_min))


def get_bounds_incr_zone(b_sym, x_sym, i, xBounds, incr_zone, b_min, b_max):
    """ stores maximum and minimum value of b(y) in b_max and b_min for an
    interval y = x_sym[i], where b is increasing. If y occurs multiple times
    in b for example b(y) = a*y + c/y it is important to evaluate b at lower
    and upper bound independently to get tighter bounds for b(y).

    Args:
        :b_sym:         function in sympy logic
        :x_sym:         list with n variables of b_sym in sympy logic
        :i:             index of variable y in x_sym
        :xBounds:       list with n variable bounds in mpmath.mpi logic
        :incr_zone:     list with increasing intervals of b(y) in mpmath.mpi logic
        :b_min:         list with minimum values of b(y) for different y of b(x,y)
        :b_max:         list with maximun values of b(y) for different y of b(x,y)

    """
    cur_max = []
    cur_min = []
    for interval in incr_zone:
        xBounds[i] = mpmath.mpi(interval.b)
        cur_max.append(float(mpmath.mpf(getBoundsOfFunctionExpression(b_sym, x_sym, xBounds).b)))
        xBounds[i] = mpmath.mpi(interval.a)
        cur_min.append(float(mpmath.mpf(getBoundsOfFunctionExpression(b_sym, x_sym, xBounds).a)))

    if cur_max != []: b_max.append(max(cur_max))
    if cur_min != []: b_min.append(min(cur_min))



def get_bounds_decr_zone(b_sym, x_sym, i, xBounds, decr_zone, b_min, b_max):
    """ stores maximum and minimum value of b(y) in b_max and b_min for an
    interval y = x_sym[i], where b is decreasing. If y occurs multiple times
    in b for example b(y) = a*y + c/y it is important to evaluate b at lower
    and upper bound independently to get tighter bounds for b(y).

    Args:
        :b_sym:         function in sympy logic
        :x_sym:         list with n variables of b_sym in sympy logic
        :i:             index of variable y in x_sym
        :xBounds:       list with n variable bounds in mpmath.mpi logic
        :decr_zone:     list with decreasing intervals of b(y) in mpmath.mpi logic
        :b_min:         list with minimum values of b(y) for different y of b(x,y)
        :b_max:         list with maximun values of b(y) for different y of b(x,y)

    """
    cur_max = []
    cur_min = []
    for interval in decr_zone:
        xBounds[i] = mpmath.mpi(interval.a)
        cur_max.append(float(mpmath.mpf(getBoundsOfFunctionExpression(b_sym, x_sym, xBounds).b)))
        xBounds[i] = mpmath.mpi(interval.b)
        cur_min.append(float(mpmath.mpf(getBoundsOfFunctionExpression(b_sym, x_sym, xBounds).a)))

    if cur_max != []: b_max.append(max(cur_max))
    if cur_min != []: b_min.append(min(cur_min))
    

def get_bounds_nonmon_zone(b_sym, x_sym, i, xBounds, nonmon_zone, b_min, b_max, dict_options):
    """ stores maximum and minimum value of b(y) in b_max and b_min for an 
    interval y = x_sym[i], where b is non-monotone.

    Args:
        :b_sym:         function in sympy logic
        :x_sym:         list with n variables of b_sym in sympy logic
        :i:             index of variable y in x_sym
        :xBounds:       list with n variable bounds in mpmath.mpi logic
        :nonmon_zone:   list with non-monotone intervals of b(y) in mpmath.mpi logic
        :b_min:         list with minimum values of b(y) for different y of b(x,y)
        :b_max:         list with maximun values of b(y) for different y of b(x,y)
        :dict_options:  dictionary with user-specified settings
    """
    
    resolution = dict_options["resolution"]
    b = lambdifyToMpmathIvComplex(x_sym, b_sym)
    
    cur_max = []
    cur_min = []
    
    for interval in nonmon_zone:
        xBounds[i] = interval
        #b_bounds = getBoundsOfFunctionExpression(b_sym, x_sym, xBounds)   
        
        curInterval= convertIntervalBoundsToFloatValues(interval)
        x = numpy.linspace(curInterval[0], curInterval[1], resolution)
        bLowValues, bUpValues = getFunctionValuesIntervalsOfXList(x, b, xBounds, i)
        
        
        #if type(b_bounds) == mpmath.iv.mpc: continue        
        if bUpValues != []: cur_max.append(max(bUpValues))#float(mpmath.mpf(b_bounds.b)))
        if bLowValues != []: cur_min.append(min(bLowValues))#float(mpmath.mpf(b_bounds.a)))
    
    if cur_max != []: b_max.append(max(cur_max))
    if cur_min != []: b_min.append(min(cur_min))
    

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


def reduceBoxCombined(xBounds, model, functions, dict_options):


    subBoxNo = 1
    output = {} 
    xNewBounds = copy.deepcopy(xBounds)
    xNewListBounds = copy.deepcopy(xBounds)
    xUnchanged = True
    xSolved = True
    dict_options_temp = copy.deepcopy(dict_options)
    eps = dict_options["relTol"]
    
    
    HC4_IvV = HC4(model, xBounds)
    if HC4_IvV.is_empty():
        saveFailedSystem(output, functions[0], model, 0)
        return output 
    else:
        for i in range(0, len(model.xSymbolic)):
            xNewListBounds[i] = [mpmath.mpi(HC4_IvV[i][0],(HC4_IvV[i][1]))]
            xNewBounds[i] = mpmath.mpi(HC4_IvV[i][0],(HC4_IvV[i][1]))
            xUnchanged = checkXforEquality(xBounds[i], xNewListBounds[i], xUnchanged, {"absTol":eps, 'relTol':0.1})
            if not variableSolved(xNewListBounds[i], dict_options): xSolved = False
    
    if xUnchanged:
        xSolved = True
        dict_options_temp.update({"newton_method":"3PNewton","InverseOrHybrid":"both"})
        newtonSystemDic = getNewtonIntervalSystem(xNewBounds, model, dict_options_temp)
        for i in range(0, len(model.xSymbolic)):
            y = xNewListBounds[i]
            if dict_options["Debug-Modus"]: 
                print(i)

            
            if xBounds[i].delta == 0:
                xNewBounds[i] = xBounds[i]
                continue
            
            if variableSolved(y, dict_options) and y[0].delta > 1.0e-15:
                dict_options_temp["relTol"] = 0.1 * y[0].delta
                dict_options_temp["absTol"] = 0.1 * y[0].delta
                
            if not variableSolved(y, dict_options_temp):        
                y = setOfIvSetIntersection([y, NewtonReduction(newtonSystemDic, xNewBounds, i, dict_options_temp)])
                if y == [] or y ==[[]]: 
                    saveFailedSystem(output, functions[0], model, 0)
                    return output 
            if not variableSolved(y, dict_options_temp):  
                    y = setOfIvSetIntersection([y, HybridGS(newtonSystemDic, xNewBounds, i, dict_options_temp)])
                    if y == [] or y ==[[]]: 
                        saveFailedSystem(output, functions[0], model, 0)
                        return output 
                    
            if not variableSolved(y, dict_options): xSolved = False
            subBoxNo = subBoxNo * len(y) 
            xNewListBounds[i] = y
            if len(y)==1: xNewBounds[i] = y[0]
            xUnchanged = checkXforEquality(xBounds[i], xNewListBounds[i], xUnchanged, 
                                           {"absTol":0.001, 'relTol':0.001})
        
        
        
    output["xAlmostEqual"] = xUnchanged 
    output["xSolved"] = xSolved    
    output["xNewBounds"] = list(itertools.product(*xNewListBounds))
    
    return output


def reduceBox(xBounds, model, functions, dict_varId_fIds, boxNo, dict_options, newtonSystemDic):
    """ reduce box spanned by current intervals of xBounds.
     
    Args: 
        :xBounds:            numpy array with box
        :model:              instance of class Model
        :functions:          list with instances of class Function
        :dict_varId_fIds:    dictionary with variable's glb id (key) and list 
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
    xNewBounds = copy.deepcopy(xBounds)
    xUnchanged = True
    xSolved = True
    dict_options_temp = copy.deepcopy(dict_options)


    if dict_options['hc_method']=='HC4':
        HC4_IvV = HC4(model, xBounds)
        if HC4_IvV.is_empty():
            saveFailedSystem(output, functions[0], model, 0)
            return output 
        else:
            for i in range(0, len(model.xSymbolic)):
                xNewBounds[i] = mpmath.mpi(HC4_IvV[i][0],(HC4_IvV[i][1]))

    
    for i in range(0, len(model.xSymbolic)):
        y = [xNewBounds[i]]
     
        if dict_options["Debug-Modus"]: print(i)
        
        #if checkVariableBound(xBounds[i], dict_options):
        if xBounds[i].delta == 0:
            xNewBounds[i] = [xBounds[i]]
            continue

        if variableSolved(y, dict_options) and y[0].delta > 1.0e-15:
            dict_options_temp["relTol"] = 0.1 * y[0].delta
            dict_options_temp["absTol"] = 0.1 * y[0].delta
       

        newtonMethods = {'newton', 'detNewton', '3PNewton'}
        if not variableSolved(y, dict_options_temp) and dict_options['newton_method'] in newtonMethods:   
            if dict_options['InverseOrHybrid']!='Hybrid':     
                y = setOfIvSetIntersection([y, NewtonReduction(newtonSystemDic, xBounds, i, dict_options_temp)])
                if y == [] or y ==[[]]: 
                    saveFailedSystem(output, functions[0], model, 0)
                    return output 
            if dict_options['InverseOrHybrid']=='Hybrid' or dict_options['InverseOrHybrid']=='both' and not variableSolved(y, dict_options_temp):  
                y = setOfIvSetIntersection([y, HybridGS(newtonSystemDic, xBounds, i, dict_options_temp)])
                if y == [] or y ==[[]]: 
                    saveFailedSystem(output, functions[0], model, 0)
                    return output 


        if not variableSolved(y, dict_options_temp) and dict_options['bc_method']=='b_normal':
            for j in dict_varId_fIds[i]:
                f = functions[j]
                y = setOfIvSetIntersection([y, 
                                            reduceXIntervalByFunction(xBounds[f.glb_ID],
                                                                      f,
                                                                      f.glb_ID.index(i),
                                                                      dict_options_temp)]) 
   
                if y == [] or y ==[[]]: 
                    saveFailedSystem(output, f, model, i)
                    return output    
        
                if ((boxNo-1) + subBoxNo * len(y)) > dict_options["maxBoxNo"]:
                    assignIvsWithoutSplit(output, xUnchanged, i, xBounds, xNewBounds)
                    return output 
    
                if variableSolved(y, dict_options): break
                else: xSolved = False 

  
        subBoxNo = subBoxNo * len(y) 
        xNewBounds[i] = y
        if not variableSolved(y, dict_options): xSolved = False
        xUnchanged = checkXforEquality(xBounds[i], xNewBounds[i], xUnchanged, 
                                       {"absTol":0.001, 'relTol':0.001})
        
        dict_options_temp["relTol"] = dict_options["relTol"]
        dict_options_temp["absTol"] = dict_options["absTol"]  

    output["xAlmostEqual"] = xUnchanged
    output["xSolved"] = xSolved       
    output["xNewBounds"] = list(itertools.product(*xNewBounds))
    return output


def variableSolved(BoundsList, dict_options):
    """ checks, if variable is solved in all Boxes in BoundsList
    Args:
        :BoundsList:      List of mpi Bounds for single variable
        :dict_options:    dictionary with tolerances for equality criterion
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
        
    """
    
    absEpsX = dict_options["absTol"]
    relEpsX = dict_options["relTol"]
    lb = mpmath.almosteq(xNewBound[0].a, xBound.a, relEpsX, absEpsX)
    ub = mpmath.almosteq(xNewBound[0].b, xBound.b, relEpsX, absEpsX)
        
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
    
    for resti in range(i, len(xBounds)):
        xNewBounds[resti] = [xBounds[resti]]
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
    output["xAlmostEqual"] = False 
    output["xSolved"] = False

    
def checkVariableBound(newXInterval, dict_options):
    """ if lower and upper bound of a variable are almost equal the boolean 
    boundsAlmostEqual is set to true.

    Args:
        :newXInterval:      variable interval in mpmath.mpi logic
        :dict_options:      dictionary with tolerance limits
        
    Return:                True, if lower and upper variable bound are almost
                            equal.

    """
    
    absEpsX = dict_options["absTol"]
    relEpsX = dict_options["relTol"]    
    if mpmath.almosteq(newXInterval.a, newXInterval.b, relEpsX, absEpsX):
        return True


def reduce_x_by_gb(g_sym, dgdx_sym, b, x_sym, i, xBounds, dict_options):
    """ reduces x=x_sym[i] by matching g_sym(x,y) with b(y).

    Args:
        :g_sym:         function in symy logic
        :dgdx_sym:      derrivative of g_sym with respect to x in sympy logic
        :b:             bounds of b(y) in mpmath.mpi formate
        :x_sym:         list with n symbolic variables in sympy formate
        :i:             index (integer) of current variable x in x_sym
        :xBounds:       list with n variable bounds in mpmath.mpi formate
        :dict_options:  dictionary with user-specified interval tolerances
        
    Return:             reduced x interval in mpmath.mpi formate 
    """

    g = getBoundsOfFunctionExpression(g_sym, x_sym, xBounds)
    dgdx = getBoundsOfFunctionExpression(dgdx_sym, x_sym, xBounds)
    
    g_sym = lambdifyToMpmathIvComplex(x_sym, g_sym)

    if isinstance(g, mpmath.iv.mpc) or isinstance(dgdx, mpmath.iv.mpc) or g==[] or dgdx==[]: return [xBounds[i]] # TODO: Complex case

    if not x_sym[i] in dgdx_sym.free_symbols and g == dgdx*xBounds[i]: # Linear Case -> solving system directly
        #dgdx_sym = lambdifyToMpmathIvComplex(x_sym, dgdx_sym)
        return getReducedIntervalOfLinearFunction(dgdx, i, xBounds, b)   

    else: # Nonlinear Case -> solving system by interval nesting
        dgdx_sym = lambdifyToMpmathIvComplex(x_sym, dgdx_sym)
        
        return getReducedIntervalOfNonlinearFunction(g_sym, 
                                                     dgdx_sym, 
                                                     dgdx, 
                                                     i, 
                                                     xBounds, 
                                                     b, 
                                                     dict_options)

    
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
    
    xBounds = copy.deepcopy(xBounds)

    gxInterval, dgdxInterval, bInterval = calculateCurrentBounds(f, i, xBounds, dict_options)
    
    if gxInterval == [] or dgdxInterval == [] or bInterval == []: return [xBounds[i]]
    
    if not f.x_sym[i] in f.dgdx_sym[i].free_symbols and gxInterval == dgdxInterval*xBounds[i]: # Linear Case -> solving system directly
        return getReducedIntervalOfLinearFunction(dgdxInterval, i, xBounds, bInterval)
             
    else: # Nonlinear Case -> solving system by interval nesting
        g_sym = lambdifyToMpmathIvComplex(f.x_sym, f.g_sym[i])
        dgdx_sym = lambdifyToMpmathIvComplex(f.x_sym, f.dgdx_sym[i])
        return getReducedIntervalOfNonlinearFunction(g_sym, dgdx_sym, 
                                                     dgdxInterval, i, 
                                                     xBounds, bInterval, dict_options)


def splitFunctionByVariableDependency(f, x):
    """ a sympy function expression f is splitted into a sum from x depended terms
    and a sum from x independent terms. If the expressions consists of a product or
    quotient, it is only checked if this one contains x or not. If one or both of the
    resulting parts are empty they are returned as 0.

    Args:
        f:              mathematical expression in sympy logic
        x:              symbolic variable x

        Return:
            fvar:           sum of x dependent arguments
            fWithoutVar:    sum of x independent arguments

    """

    allArguments = f.args
    allArgumentsWithVariable = []
    allArgumentsWithoutVariable = []

    if f.func.class_key()[2] in ['Mul', 'sin', 'cos', 'exp', 'log', 'Pow'] :
        #f = f + numpy.finfo(numpy.float).eps/2
        return f, 0.0

    if f.func.class_key()[2]=='Add':

        for i in range(0, len(allArguments)):
            if x in allArguments[i].free_symbols:
                allArgumentsWithVariable.append(allArguments[i])
            else: allArgumentsWithoutVariable.append(allArguments[i])

        fvar = sympy.Add(*allArgumentsWithVariable)
        fWithoutVar = sympy.Add(*allArgumentsWithoutVariable)
        return fvar, fWithoutVar

    #if f.func.class_key()[2]=='Mul':
    #    if x in f.free_symbols: return f, 0
    #    else: return 0, f

    else:
        print("Problems occured during function parsing")
        return 0, 0


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
            "atan": ivatan,
            "log" : ivlog,
            "sqrt": ivsqrt}

    return sympy.lambdify(x, f, mpmathIv)


def ivsqrt(iv):
    """calculates the square root of an interval iv, stripping it from the imaginary part"""

    if iv.a >= 0 and iv.b >= 0:
        sqrtiv = mpmath.mpi(mpmath.sqrt(iv.a), mpmath.sqrt(iv.b))
    elif iv.a < 0 and iv.b >= 0:
        sqrtiv = mpmath.mpi(0., mpmath.sqrt(iv.b))
    else:
        # this case should not occur, the solution can not be in this interval
        sqrtiv = mpmath.mpi('-inf', '+inf')

    return sqrtiv


def ivlog(iv):
    """calculates the ln root of an interval iv, stripping it from the imaginary part"""
    
    if iv.a > 0 and iv.b > 0:
        ivlog=mpmath.mpi(mpmath.log(iv.a),mpmath.log(iv.b))
    elif iv.a <= 0 and iv.b > 0:
        ivlog=mpmath.mpi('-inf',mpmath.log(iv.b))
    elif iv.a <= 0 and iv.b <= 0:
        #this case should not occur, the solution can not be in this interval
        #print('Negative ln! Solution can not be in this Interval!')
        ivlog=mpmath.mpi('-inf', '+inf')

    return ivlog


def ivacos(iv):
    """calculates the acos of an interval iv, stripping it from the imaginary part"""

    if iv.a>=-1 and iv.b<=1:
        ivacos = mpmath.mpi(mpmath.acos(iv.b),mpmath.acos(iv.a))
    elif iv.a<-1 and iv.b<=1 and iv.b>=-1:
        ivacos = mpmath.mpi(mpmath.acos(iv.b),mpmath.pi)
    elif iv.a>=-1 and iv.a<=1 and iv.b>1:
        ivacos = mpmath.mpi(0, mpmath.acos(iv.a))
    else:
        ivacos = mpmath.mpi(0, mpmath.pi)
        
    return ivacos


def ivasin(iv):
    """calculates the asin of an interval iv, stripping it from the imaginary part"""

    if iv.a>=-1 and iv.b<=1:
        ivasin = mpmath.mpi(mpmath.asin(iv.a),mpmath.asin(iv.b))
    elif iv.a<-1 and iv.b<=1 and iv.b>=-1:
        ivasin = mpmath.mpi(mpmath.asin(-1),mpmath.asin(iv.b))
    elif iv.a>=-1 and iv.a<=1 and iv.b>1:
        ivasin = mpmath.mpi(mpmath.asin(iv.a),mpmath.asin(1))
    else:
        ivasin = mpmath.mpi(mpmath.asin(-1), mpmath.asin(1))
        
    return ivasin


def ivatan(iv):
    """calculates the atan of an interval iv, stripping it from the imaginary part"""

    if iv.a>=-mpmath.pi/2 and iv.b<=mpmath.pi/2:
        ivatan = mpmath.mpi(mpmath.atan(iv.a),mpmath.atan(iv.b))
    elif iv.a<-mpmath.pi/2 and iv.b<=mpmath.pi/2 and iv.b>=-mpmath.pi/2:
        ivatan = mpmath.mpi(mpmath.atan(-mpmath.pi/2),mpmath.atan(iv.b))
    elif iv.a>=-mpmath.pi/2 and iv.a<=mpmath.pi/2 and iv.b>mpmath.pi/2:
        ivatan = mpmath.mpi(mpmath.atan(iv.a),mpmath.atan(mpmath.pi/2))
    else:
        ivatan = mpmath.mpi(mpmath.atan(-mpmath.pi/2), mpmath.atan(mpmath.pi/2))
        
    return ivatan


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
    
    if dict_options["bc_method"] == "b_tight":
        try: bInterval = get_tight_bBounds(f, i, xBounds, dict_options)
           # b_max = []
           # b_min = []
           # box = copy.deepcopy(xBounds)
           # newBoxes = separateBox(box, [f.glb_ID.index(id) for id in f.glb_ID if id != i])
           # for sb in newBoxes:
           #     curB = getBoundsOfFunctionExpression(f.b_sym[i], f.x_sym, sb)  
           #     b_max.append(float(mpmath.mpf(curB.b)))
           #     b_min.append(float(mpmath.mpf(curB.a)))
           # bInterval = mpmath.mpi(min(b_min), max(b_max))
            
        except: return [], [], []
        
    else:
        try:
            bInterval = getBoundsOfFunctionExpression(f.b_sym[i], 
                                                      f.x_sym, xBounds)
        except: return [], [], []
      
    try:
       gxInterval = getBoundsOfFunctionExpression(f.g_sym[i], 
                                                  f.x_sym, xBounds)
       if type(gxInterval) == mpmath.iv.mpc: gxInterval = []
      
    except: return [], [], bInterval

    try:
       dgdxInterval = getBoundsOfFunctionExpression(f.dgdx_sym[i], 
                                                    f.x_sym, xBounds)
       if type(dgdxInterval) == mpmath.iv.mpc: dgdxInterval = []

    except: return gxInterval, [], bInterval
       
    return gxInterval, dgdxInterval, bInterval  
    
    
def getBoundsOfFunctionExpression(f, xSymbolic, xBounds):
    """ evaluates function expression f for variable bounds xBounds

    Args:
        :f:                  scalar function in sympy logic
        :xSymbolic:          list with symbolic variables in sympy logic
        :xBounds:            numpy array with variable bounds


    Return:
        :interval:           of function expression at xBounds in mpmath.mpi logic

    """

    if isinstance(f, sympy.Float) and len(str(f)) > 15:
        return roundValue(f, 16)
    fMpmathIV = lambdifyToMpmathIvComplex(xSymbolic, f)
    try:         
        fInterval = fMpmathIV(*xBounds)#timeout(fMpmathIV, xBounds)
        if fInterval == False: return mpmath.mpi('-inf','inf')
        return mpmath.mpi(str(fInterval))
    except:
        return []


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


def reformulateComplexExpressions(f):
    """ to avoid complex intervals this function reformulates arguments of log or
    pow functions, so that for example:

        log(a) = log((a**2)**0.5)
        b**0.25 = ((b**2)**0.5)**0.25

    In this way, only real numbers remain.

    Args:
        :f:                       scalar function in sympy logic

    Return:
        :f:                       reformulated scalar function in sympy logic

    """

    allArguments = []

    for arg in sympy.preorder_traversal(f):
        allArguments.append(arg)

    for i in range(0, len(allArguments)):
        if allArguments[i].func.class_key()[2] == 'Pow':
            f= f.subs(allArguments[i+1], ((allArguments[i+1])**2)**0.5)
        if allArguments[i].func.class_key()[2] == 'log':
            f= f.subs(allArguments[i+1], ((allArguments[i+1])**2)**0.5)
    return f


def reactOnComplexError(f, xSymbolic, i, xBounds, dict_options):
    """ starts interval nesting procedure to get rid of complex intervals

    Args:
        :f:                  scalar function in mpmath logic
        :xSymbolic:          list with symbolic variables in sympy logic
        :i:                  index of current iteration variable
        :xBounds:            numpy array with variable bounds
        :dict_options:       dictionary with variable interval tolerances

    Return:
        :realSection:        real xBounds (so far this is only valid if there is
                             only one complex and one real interval,
                             TODO: return set of real intervals for x[i])

    """
    tmax = dict_options["tmax"]
    absEpsX = dict_options["absTol"]
    realSection = []
    curXBounds = copy.deepcopy(xBounds)
    problematicSection = [xBounds[i]]
    timeout = False
    t0 = time.clock()

    while problematicSection != [] and timeout == False:
        curProblematicSection =[]
        for j in range(0, len(problematicSection)):

            curXBounds[i] = problematicSection[j].a
            complexA = testOneBoundOnComplexity(f, xSymbolic, curXBounds, i)
            curXBounds[i] = problematicSection[j].mid
            complexMid = testOneBoundOnComplexity(f, xSymbolic, curXBounds, i)
            curXBounds[i] = problematicSection[j].b
            complexB = testOneBoundOnComplexity(f, xSymbolic, curXBounds, i)

            problematicInterval, realInterval = complexOperator(complexA, complexMid, complexB,
                                                        problematicSection[j])

            if problematicInterval != []:
                if curProblematicSection == []: curProblematicSection.append(problematicInterval)
                else:
                    addIntervaltoZone(problematicInterval, curProblematicSection, dict_options)

            if realInterval !=[]:
                if realSection == []: realSection = realInterval
                else:
                    realSection = addIntervaltoZone(realInterval, realSection, dict_options)

        problematicSection = checkAbsoluteTolerance(removeListInList(curProblematicSection), absEpsX)
        timeout = checkTimeout(t0, tmax, timeout)
    if realSection ==[]: return []
    else: return realSection[0]


def checkTimeout(t0, tmax, timeout):
    """ sets timeout variable true if current time of loop tf exceeds maximum loop time

    Args:
        :t0:        integer with start time of loop
        :tmax:      integer with maximum time of loop
        :timeout:   boolean true if tf-t0 > tmax

    """

    tf = time.clock()

    if (tf-t0) > tmax:
        timeout = True
        print("Warning: Timeout of process.")
    return timeout

def checkIntervalWidth(interval, absEpsX, relEpsX):
    """ checks if width of intervals is smaller than a given absolute or relative tolerance

    Args:
        :interval:           set of intervals in mpmath.mpi-logic
        :absEpsX:            absolute x tolerance
        :relEpsX:            relative x tolerance

    Return:
        :interval:    set of intervals with a higher width than absEps
    """
    reduced_interval = []
    for curInterval in interval:
        if not mpmath.almosteq(curInterval.a, curInterval.b, absEpsX, relEpsX):
            reduced_interval.append(curInterval)
            #interval.remove(curInterval)
    return reduced_interval


def checkAbsoluteTolerance(interval, absEpsX):
    """ checks if width of intervals is smaller than a given absolute tolerance absEpsX

    Args:
        :interval:           set of intervals in mpmath.mpi-logic
        :absEpsX:             absolute x tolerance

    Return:
        :reducedInterval:    set of intervals with a higher width than absEps

    """

    reducedInterval = []

    for i in range(0, len(interval)):
        if interval[i].delta  > absEpsX:
                reducedInterval.append(interval[i])

    return reducedInterval


def testOneBoundOnComplexity(f, xSymbolic, xBounds, i):
    """ returns False if the evlauation of f fails, assumption is that this is
    due to complex variable intervals

    Args:
        :f:                  scalar function in mpmath logic
        :xSymbolic:          list with symbolic variables in sympy logic
        :xBounds:            numpy array with variable bounds
        :i:                  index of current iteration variable

    Return:
        :True:               Complex number
        :False:              Real number

    """

    try:
        mpmath.mpi(str(f(*xBounds)))
        return False
    except:
        return True


def complexOperator(complexA, complexMid, complexB, xInterval):
    """ splits xInterval into complex/real-xInterval and real-xInterval

    Args:
        :complexA:           boolean: True = complex lower bound of xInterval
                             False = real lower bound of xInterval
        :complexB:           boolean: True = complex upper bound of xInterval
                             False = real upper bound of xInterval
        :complexC:           boolean: True = complex midpoint of xInterval
                             False = real midpoint of xInterval
        :xInterval:          interval of variable x in mpmath.mpi logic

    Return:
        :list1:              list with complex-real-xInterval
        :list2:              list with real xInterval

    """

    if complexA and complexMid and complexB:
        return [], []
    if complexA and complexMid and not complexB:
        return [mpmath.mpi(xInterval.mid, xInterval.b)], []
    if complexA and not complexMid and not complexB:
        return [mpmath.mpi(xInterval.a, xInterval.mid)], [mpmath.mpi(xInterval.mid, xInterval.b)]
    if not complexA and not complexMid and not complexB:
        return [], [] # non monotone function
    if not complexA and not complexMid and complexB:
        return [mpmath.mpi(xInterval.mid, xInterval.b)], [mpmath.mpi(xInterval.a, xInterval.mid)]
    if not complexA and complexMid and complexB:
        return [mpmath.mpi(xInterval.a, xInterval.mid)], []
    else: return [], []


def getReducedIntervalOfLinearFunction(a, i, xBounds, bi):
    """ returns reduced interval of variable X if f is linear in X. The equation
    is solved directly by the use of the gaussSeidelOperator.
    
    Args: 
        :a:                  mpmath.mpi interval
        :i:                  integer with current iteration variable index
        :xBounds:            numpy array with set of variable bounds
        :bi:                 current function residual bounds

    Return:                  reduced x-Interval(s)   
    """        
    
    if bool(0 in bi - a * xBounds[i]) == False: return [] # if this is the case, there is no solution in xBoundsi

    if bool(0 in bi) and bool(0 in a):  # if this is the case, bi/aInterval would return [-inf, +inf]. Hence the approximation of x is already smaller
                return [xBounds[i]]
    else: 
        return gaussSeidelOperator(a, bi, xBounds[i]) # bi/aInterval  


def checkAndRemoveComplexPart(interval):
    """ creates a warning if a complex interval occurs and keeps only the real
    part.

    """

    if interval.imag != 0:
        print("Warning: A complex interval: ", interval.imag," occured.\n",
        "For further calculations only the real part: ", interval.real, " is used.")
        interval = interval.real


def gaussSeidelOperator(a, b, x):
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

    for j in range(0,len(u)):
        intersection = ivIntersection(u[j], x)
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

    # Different cases:
    if bool(0 in i2)== False: return [i1 * mpmath.mpi(1/i2.b, 1/i2.a)]
    if bool(0 in i1) and bool(0 in i2): return [i1 / i2]
    if i1.b < 0 and i2.a != i2.b and i2.b == 0: return [mpmath.mpi(i1.b / i2.a, i1.a / i2.b)]
    if i1.b < 0 and i2.a < 0 and i2.b > 0: return [mpmath.mpi('-inf', i1.b / i2.b), mpmath.mpi(i1.b / i2.a, 'inf')]
    if i1.b < 0 and i2.a == 0 and i2.b > 0: return [mpmath.mpi('-inf', i1.b / i2.b)]
    if i1.a > 0 and i2.a < 0 and i2.b == 0: return [mpmath.mpi('-inf', i1.a / i2.a)]
    if i1.a > 0 and i2.a < 0 and i2.b > 0: return [mpmath.mpi('-inf', i1.a / i2.a), mpmath.mpi(i1.a / i2.b, 'inf')]
    if i1.a > 0 and i2.a == 0 and i2.b > 0: return [mpmath.mpi(i1.a / i2.b,'inf')]

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

    # Different cases:
    if i1.a <= i2.a and i1.b <= i2.b and i2.a <= i1.b: return mpmath.mpi(i2.a, i1.b)
    if i1.a <= i2.a and i1.b >= i2.b: return i2
    if i1.a >= i2.a and i1.b <= i2.b: return i1
    if i1.a >= i2.a and i1.b >= i2.b and i1.a <= i2.b: return mpmath.mpi(i1.a, i2.b)

    else: return []


def get_conti_monotone_intervals(dfdx_sym, x_sym, i, xBounds, dict_options):
    """ splits interval into monotone-increasing, monotone-decreasing and non-monotone
    sections by first derrivative dfx_sym referring to x_sym[i]

    Args:
        :dfdx_sym:      symbolic first derrivative of function f in sympy logic
        :x_sym:         list with n symbolic variables of function f in sympy logic
        :i:             index (integer) of independent variable in x_sym
        :xBounds:       list with n variable bounds in mpmath.mpi logic
        :dict_options:  dicionary with user-specified interval tolerances

    Return:
        :increasingZones:   list with bounds of monotone-increasing intervals in
                            mpmath.mpi formate
        :decreasingZones:   list with bounds of monotone-decreasing intervals in
                            mpmath.mpi formate
        :nonMonotoneZones:  list with bounds of non-monotone intervals in
                            mpmath.mpi formate

    """
    relEpsX = dict_options["relTol"]
    absEpsX = dict_options["absTol"]
    increasingZones = []
    decreasingZones = []
    nonMonotoneZones = []
    cur_xiBounds = [xBounds[i]]

    dfdx = getBoundsOfFunctionExpression(dfdx_sym, x_sym, xBounds)

    if type(dfdx) == mpmath.iv.mpc or dfdx == []: return [], [], cur_xiBounds
    
    dfdx_sym = lambdifyToMpmathIvComplex(x_sym, dfdx_sym)
    
    if '-inf' in dfdx or '+inf' in dfdx: 
        cur_xiBounds, nonMonotoneZones = getContinuousFunctionSections(dfdx_sym, i, xBounds, dict_options)

    if  cur_xiBounds != [] :

        for curInterval in cur_xiBounds:
            xBounds[i] = curInterval
            increasingZone, decreasingZone, nonMonotoneZone = getMonotoneFunctionSections(dfdx_sym, i, xBounds, dict_options)

            if increasingZone !=[]:
                for interval in increasingZone:
                    increasingZones.append(interval)
            if decreasingZone !=[]:
                for interval in decreasingZone:
                    decreasingZones.append(interval)
            if nonMonotoneZone !=[]:
                for interval in nonMonotoneZone:
                    nonMonotoneZones.append(interval)
                nonMonotoneZones = joinIntervalSet(nonMonotoneZones, relEpsX, absEpsX)
    return increasingZones, decreasingZones, nonMonotoneZones


def getReducedIntervalOfNonlinearFunction(fx, dfdX, dfdXInterval, i, xBounds, bi, dict_options):
    """ checks function for monotone sections in x and reduces them one after the other.

    Args:
        :fx:                 symbolic x-depending part of function f in mpmath.mpi logic
        :dfdX:               first symbolic derivative of function f with respect to x
                             in mpmath.mpi logic
        :dfdXInterval:       first derivative of function f with respect to x at xBounds
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
    orgXiBounds = [copy.deepcopy(xBounds[i])]
    curXiBounds = orgXiBounds

    if dfdXInterval == []: return []

    if '-inf' in dfdXInterval or '+inf' in dfdXInterval: # condition for discontinuities
        curXiBounds, nonMonotoneZones = getContinuousFunctionSections(dfdX, i, xBounds, dict_options)
    if curXiBounds != []:
        for curInterval in curXiBounds:
            xBounds[i] = curInterval
            increasingZone, decreasingZone, nonMonotoneZone = getMonotoneFunctionSections(dfdX, i, xBounds, dict_options)
            if increasingZone !=[]: increasingZones.append(increasingZone)
            if decreasingZone !=[]: decreasingZones.append(decreasingZone)
            if nonMonotoneZone !=[]:
                for interval in nonMonotoneZone:
                    nonMonotoneZones.append(interval)
                nonMonotoneZones = joinIntervalSet(nonMonotoneZones, relEpsX, absEpsX)

    if increasingZones !=[]:
            increasingZones = removeListInList(increasingZones)
            reducedIntervals = reduceMonotoneIntervals(increasingZones, reducedIntervals, fx,
                                    xBounds, i, bi, dict_options, increasing = True)

    if decreasingZones !=[]:
            decreasingZones = removeListInList(decreasingZones)                
            reducedIntervals = reduceMonotoneIntervals(decreasingZones, reducedIntervals, fx, 
                                    xBounds, i, bi, dict_options, increasing = False)  
       
    if nonMonotoneZones !=[]:
        reducedIntervals = reduceNonMonotoneIntervals({"0":nonMonotoneZones, 
                                   "1": reducedIntervals, 
                                   "2": fx, 
                                   "3": i, 
                                   "4": xBounds, 
                                   "5": bi, 
                                   "6": dict_options})
        #reducedIntervals = timeout(reduceNonMonotoneIntervals, [{"0":nonMonotoneZones, 
    #                               "1": reducedIntervals, 
    #                               "2": fx, 
    #                               "3": i, 
    #                               "4": xBounds, 
    #                               "5": bi, 
    #                               "6": dict_options}]) 
        if reducedIntervals == False: 
            print("Warning: Reduction in non-monotone Interval took too long.")
            return orgXiBounds
        #reducedIntervals = reduceNonMonotoneIntervals(nonMonotoneZones, reducedIntervals, 
        #                                                  fx, i, xBounds, bi, 
        #                                                  dict_options)
    #reducedIntervals = reduceTwoIVSets(reducedIntervals, orgXiBounds)
    reducedIntervals = setOfIvSetIntersection([reducedIntervals, orgXiBounds])
    return reducedIntervals


def getContinuousFunctionSections(dfdx, i, xBounds, dict_options):
    """filters out discontinuities which either have a +/- inf derrivative.

    Args:
        :dfdx:                scalar first derivate of function in mpmath.mpi logic
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
    orgXiBounds = copy.deepcopy(xBounds[i])
    interval = [xBounds[i]]

    while interval != [] and len(interval) <= maxIvNo:
        discontinuousZone = []

        for curInterval in interval:
            newContinuousZone = testIntervalOnContinuity(dfdx, curInterval, xBounds, i, discontinuousZone)
            if newContinuousZone == False: return continuousZone, joinIntervalSet(interval, relEpsX, absEpsX)
            continuousZone = addIntervaltoZone(newContinuousZone, continuousZone, dict_options)  
               
        interval = checkIntervalWidth(discontinuousZone, absEpsX, relEpsX)
        if not len(interval) <= maxIvNo: return continuousZone, joinIntervalSet(interval, relEpsX, absEpsX)
    if interval == [] and continuousZone == []: return [], [orgXiBounds]

    return continuousZone, []


def removeListInList(listInList):
    """changes list with the shape: [[a], [b,c], [d], ...] to [a, b, c, d, ...]

    """

    newList = []
    for i in range(0, len(listInList)):
        for j in range(0, len(listInList[i])):
            newList.append(listInList[i][j])
    return newList


def reduceMonotoneIntervals(monotoneZone, reducedIntervals, fx,
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

    relEpsX = dict_options["relTol"]
    absEpsX = dict_options["absTol"]

    for curMonZone in monotoneZone: #TODO: Parallelizing
        xBounds[i] = curMonZone

        if increasing: curReducedInterval = monotoneIncreasingIntervalNesting(fx, xBounds, i, bi, dict_options)
        else: curReducedInterval = monotoneDecreasingIntervalNesting(fx, xBounds, i, bi, dict_options)

        if curReducedInterval !=[] and reducedIntervals != []:
            reducedIntervals.append(curReducedInterval)
            reducedIntervals = joinIntervalSet(reducedIntervals, relEpsX, absEpsX)
        elif curReducedInterval !=[]: reducedIntervals.append(curReducedInterval)

    return reducedIntervals


def monotoneIncreasingIntervalNesting(fx, xBounds, i, bi, dict_options):
    """ reduces variable intervals of monotone increasing functions fx
    by interval nesting

        Args:
            :fx:                 symbolic xi-depending part of function fi
            :xBounds:            numpy array with set of variable bounds
            :i:                  integer with current iteration variable index
            :bi:                 current function residual bounds
            :dict_options:       dictionary with function and variable interval tolerances

        Return:                  list with one entry that is the reduced interval
                                 of the variable with the index i

    """

    relEpsX = dict_options["relTol"]
    absEpsX = dict_options["absTol"]

    fxInterval = fx(*xBounds)
    curInterval = xBounds[i]

    if ivIntersection(fxInterval, bi)==[]: return []

    # first check if xBounds can be further reduced:
    if fxInterval in bi: return curInterval

    # Otherwise, iterate each bound of bi:
    fIntervalxLow, fIntervalxUp = getFIntervalsFromXBounds(fx, curInterval, xBounds, i)

    if fIntervalxLow.b < bi.a:
         while not mpmath.almosteq(mpmath.mpf(fxInterval.a), mpmath.mpf(fxInterval.b),
                                   relEpsX, absEpsX) and not mpmath.almosteq(curInterval.a,
                                                 curInterval.b, relEpsX, absEpsX):
                         curInterval, fxInterval = iteratefBound(fx, curInterval, xBounds, i, bi,
                                                increasing = True,
                                                lowerXBound = True)
                         if curInterval == [] or fxInterval == []: return []

    lowerBound = curInterval.a
    curInterval  = xBounds[i]    
    fxInterval = fx(*xBounds)
    if fIntervalxUp.a > bi.b:

        while not mpmath.almosteq(mpmath.mpf(fxInterval.a), mpmath.mpf(fxInterval.b),
                                  relEpsX, absEpsX) and not mpmath.almosteq(curInterval.a,
                                                curInterval.b, relEpsX, absEpsX):

            curInterval, fxInterval = iteratefBound(fx, curInterval, xBounds, i, bi,
                                                increasing = True,
                                                lowerXBound = False)
            if curInterval == [] or fxInterval == []: return []

    upperBound = curInterval.b
    return mpmath.mpi(lowerBound, upperBound)


def monotoneDecreasingIntervalNesting(fx, xBounds, i, bi, dict_options):
    """ reduces variable intervals of monotone decreasing functions fx
    by interval nesting

        Args:
            :fx:                 symbolic xi-depending part of function fi
            :xBounds:            numpy array with set of variable bounds
            :i:                  integer with current iteration variable index
            :bi:                 current function residual bounds
            :dict_options:       dictionary with function and variable interval tolerances

        Return:                  list with one entry that is the reduced interval
                                 of the variable with the index i

    """

    relEpsX = dict_options["relTol"]
    absEpsX = dict_options["absTol"]

    fxInterval = fx(*xBounds)
    curInterval = xBounds[i]

    if ivIntersection(fxInterval, bi)==[]: return []

    # first check if xBounds can be further reduced:
    if fxInterval in bi: return curInterval

    # Otherwise, iterate each bound of bi:
    fIntervalxLow, fIntervalxUp = getFIntervalsFromXBounds(fx, curInterval, xBounds, i)


    if fIntervalxLow.a > bi.b:
         while not mpmath.almosteq(mpmath.mpf(fxInterval.a), mpmath.mpf(fxInterval.b),
                                   relEpsX, absEpsX) and not mpmath.almosteq(curInterval.a,
                                                 curInterval.b, relEpsX, absEpsX):
                         curInterval, fxInterval = iteratefBound(fx, curInterval, xBounds, i, bi,
                                                increasing = False,
                                                lowerXBound = True)
                         if curInterval == [] or fxInterval == []: return []
        
    lowerBound = curInterval.a  
    curInterval  = xBounds[i]        
    fxInterval = fx(*xBounds)
    
    if fIntervalxUp.b < bi.a:

        while not mpmath.almosteq(mpmath.mpf(fxInterval.a), mpmath.mpf(fxInterval.b),
                                  relEpsX, absEpsX) and not mpmath.almosteq(curInterval.a,
                                                curInterval.b, relEpsX, absEpsX):

            curInterval, fxInterval = iteratefBound(fx, curInterval, xBounds, i, bi,
                                                increasing = False,
                                                lowerXBound = False)
            if curInterval == [] or fxInterval == []: return []

    upperBound = curInterval.b
    return mpmath.mpi(lowerBound, upperBound)


def iteratefBound(fx, curInterval, xBounds, i, bi, increasing, lowerXBound):
    """ returns the half of curInterval that contains the lower or upper
    bound of bi (biLimit)

    Args:
        :fx:                 symbolic xi-depending part of function fi
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
    fIntervalxLow, fIntervalxUp = getFIntervalsFromXBounds(fx, curlowerXInterval,
                                                           xBounds, i)

    fxInterval = fBoundsOperator(fIntervalxLow, fIntervalxUp, increasing, lowerXBound)

    if biBound in fxInterval: return curlowerXInterval, fxInterval

    else:
        curUpperXInterval = mpmath.mpi(curInterval.mid, curInterval.b)
        fIntervalxLow, fIntervalxUp = getFIntervalsFromXBounds(fx, curUpperXInterval,
                                                               xBounds, i)

        fxInterval = fBoundsOperator(fIntervalxLow, fIntervalxUp, increasing, lowerXBound)

        if biBound in fxInterval: return curUpperXInterval, fxInterval
        else: return [], []


def getFIntervalsFromXBounds(fx, curInterval, xBounds, i):
    """ returns function interval for lower variable bound and upper variable
    bound of variable interval curInterval.

    Args:
        :fx:             symbolic function in mpmath.mpi logic
        :curInterval:    current variable interval in mpmath logic
        :xBounds:        set of variable intervals in mpmath logic
        :i:              index of currently iterated variable interval

    Return:              function interval for lower variable bound and upper variable bound

    """

    curXBoundsLow = copy.deepcopy(xBounds)
    curXBoundsUp = copy.deepcopy(xBounds)

    curXBoundsLow[i]  = curInterval.a
    curXBoundsUp[i] = curInterval.b

    return fx(*curXBoundsLow), fx(*curXBoundsUp)


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

    Return:               relevant function interval for iterating bi bound in mpmath logic

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

    Return:               lower or upper bound of function residual interval in mpmath logic

    """

    if increasing and lowerXBound: return bi.a
    if increasing and not lowerXBound: return bi.b
    if not increasing and lowerXBound: return bi.b
    if not increasing and not lowerXBound: return bi.a


def getMonotoneFunctionSections(dfdx, i, xBounds, dict_options):
    """seperates variable interval into variable interval sets where a function
    with derivative dfdx is monontoneous

    Args:
        :dfdx:                scalar function in mpmath.mpi logic
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
    
    #tmax = dict_options["tmax"]
    relEpsX = dict_options["relTol"]
    absEpsX = dict_options["absTol"]
    maxIvNo = dict_options["resolution"]
    monIncreasingZone = []
    monDecreasingZone = []
    org_xiBounds = copy.deepcopy(xBounds[i])
    interval = [xBounds[i]]

    while interval != [] and len(interval) < maxIvNo: #and timeout == False: # #and dfdXconst == False:

        curIntervals = []

        for xc in interval:
            newIntervals, newMonIncreasingZone, newMonDecreasingZone = testIntervalOnMonotony(dfdx,
                                                    xc, xBounds, i)

            monIncreasingZone = addIntervaltoZone(newMonIncreasingZone,
                                                          monIncreasingZone, dict_options)
            monDecreasingZone = addIntervaltoZone(newMonDecreasingZone,
                                                          monDecreasingZone, dict_options)

            curIntervals.append(newIntervals)
        curIntervals = removeListInList(curIntervals)


        if checkIntervalWidth(curIntervals, absEpsX, relEpsX) == interval:
            interval = joinIntervalSet(interval, relEpsX, absEpsX)
            break

        interval = checkIntervalWidth(curIntervals, absEpsX, relEpsX)

    if not len(interval) <= maxIvNo:
        interval = joinIntervalSet(interval, relEpsX, absEpsX)

    if interval == [] and monDecreasingZone == [] and monIncreasingZone ==[]:
        return [], [], [org_xiBounds]
    return monIncreasingZone, monDecreasingZone, interval


def discretizeAndEvaluateIntervals(dfdX, xBounds, i, intervals, monIncreasingZone,
                                   monDecreasingZone, dict_options):
    """ if a certain interval width is undershot it is useful to discretize it and
    check all segments on their monotony.

    Args:
        :dfdX:                scalar function in mpmath.mpi logic
        :xBounds:             numpy array with variable bounds
        :i:                   index of differential variable
        :intervals:           new unchecked variable intervals
        :monIncreasingZone:   already identified monotone increasing intervals
        :monDecreasingZone:   already identified monotone decreasing intervals
        :dict_options:        dictionary with function and variable interval
                             tolerances and discretization resolution

    Return:
        :newIntervals:        non monotone intervals if  function interval can not be
                             reduced to monotone increasing or decreasing section
        :monIncreasingZone:   monotone increasing intervals
        :monDecreasingZone:   monotone decreasing intervals

    """

    newIntervals = []

    for interval in intervals:
        newIntervals, monIncreasingZone, monDecreasingZone = discretizeAndEvaluateInterval(dfdX, 
                                        xBounds, i, interval, newIntervals, monIncreasingZone, 
                                        monDecreasingZone, dict_options)

    return newIntervals, monIncreasingZone, monDecreasingZone


def discretizeAndEvaluateInterval(dfdX, xBounds, i, interval, newIntervals,
                                   monIncreasingZone, monDecreasingZone, dict_options):
    """ if a certain interval width is undershot it is useful to discretize it and
    check all segments on their monotony.

    Args:
        :dfdX:                scalar function in mpmath.mpi logic
        :xBounds:             numpy array with variable bounds
        :i:                   index of differential variable
        :interval:            unchecked variable interval
        :newIntervals:        already identified non monotone intervals
        :monIncreasingZone:   already identified monotone increasing intervals
        :monDecreasingZone:   already identified monotone decreasing intervals
        :dict_options:        dictionary with function and variable interval
                              tolerances and discretization resolution

    Return:
        :newIntervals:        non monotone intervals if  function interval can not be
                              reduced to monotone increasing or decreasing section
        :monIncreasingZone:   monotone increasing intervals
        :monDecreasingZone:   monotone decreasing intervals

    """

    resolution = dict_options["resolution"]*2

    intervalBounds = convertIntervalBoundsToFloatValues(interval)
    intervalPoints = numpy.linspace(intervalBounds[0], intervalBounds[1], resolution)

    for j in range(0, (len(intervalPoints)-1)):

        xBounds[i] = mpmath.mpi(min(intervalPoints[j], intervalPoints[j+1]),
               max(intervalPoints[j], intervalPoints[j+1]))

        if dfdX(*xBounds) > 0:
            monIncreasingZone = addIntervaltoZone([xBounds[i]], monIncreasingZone, dict_options)

        elif dfdX(*xBounds) < 0:
            monDecreasingZone = addIntervaltoZone([xBounds[i]], monDecreasingZone, dict_options)
        else:
            newIntervals = addIntervaltoZone([xBounds[i]], newIntervals, dict_options)

    return newIntervals, monIncreasingZone, monDecreasingZone


def convertIntervalBoundsToFloatValues(interval):
    """ converts mpmath.mpi intervals to list with bounds as float values

    Args:
        :interval:              interval in math.mpi logic

    Return:                     list with bounds as float values

    """

    return [float(mpmath.mpf(interval.a)), float(mpmath.mpf(interval.b))]


def testIntervalOnContinuity(dfdx, interval, xBounds, i, discontinuousZone):
    """ splits interval into 2 halfs and orders them regarding their continuity
   in the first derrivative.

    Args:
        :dfdx:              scalar derivative of f with respect to x in
                            mpmath.mpi-logic
        :interval:          x interval in mpmath.mpi-logic
        :xBounds:           numpy array with variable bounds in mpmath.mpi-logic
        :i:                 variable index
        :discontinuousZone: list with current discontinuous intervals in
                            mpmath.mpi-logic

    Reutrn:
        :continuousZone:    list with new continuous intervals in
                            mpmath.mpi-logic

    """
    
    continuousZone = []
    curXBoundsLow = mpmath.mpi(interval.a, interval.mid)
    xBounds[i] = curXBoundsLow
    dfdxLow = dfdx(*xBounds)#timeout(dfdx, xBounds)  
         
    curXBoundsUp = mpmath.mpi(interval.mid, interval.b)
    xBounds[i] = curXBoundsUp
    dfdxUp = dfdx(*xBounds)#timeout(dfdx, xBounds)
    
    if dfdxLow == False: discontinuousZone.append(curXBoundsLow)    
    if dfdxUp == False : discontinuousZone.append(curXBoundsUp)       
    if dfdxLow == False or dfdxUp == False: return False

    if not '-inf' in dfdxLow and not '+inf' in dfdxLow: continuousZone.append(curXBoundsLow)
    else: discontinuousZone.append(curXBoundsLow) 
        
    if not '-inf' in dfdxUp and not '+inf' in dfdxUp: continuousZone.append(curXBoundsUp)
    else: discontinuousZone.append(curXBoundsUp) 

    return continuousZone


def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    import signal

    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    # set the timeout handler
    signal.signal(signal.SIGALRM, handler) 
    signal.alarm(timeout_duration)
    try:
        result = func(*args, **kwargs)
    except TimeoutError: #as exc:
        result = False
        print("Warning: TimeOut")
    finally:
        signal.alarm(0)

    return result


def testIntervalOnMonotony(dfdx, interval, xBounds, i):
    """ splits interval into 2 halfs and orders concering their monotony
    behaviour of f (first derivative dfdx):
        1. monotone increasing function in interval of x
        2. monotone decreasing function in interval of x
        3. non monotone function in interval of x

    Args:
        :dfdx:           scalar derivative of f with respect to x in mpmath.mpi-logic
        :interval:       x interval in mpmath.mpi-logic
        :xBounds:        numpy array with variable bounds in mpmath.mpi-logic
        :i:              variable index
                         that have an x indepenendent derrivate constant interval of
                         [-inf, +inf] for example: f=x/y-1 and y in [-1,1]
    Reutrn:
        3 lists nonMonotoneZone, monotoneIncreasingZone, monotoneDecreasingZone
        and one updated count of [-inf,inf] dfdxIntervals as integer

    """
    nonMonotoneZone = []
    monotoneIncreasingZone = []
    monotoneDecreasingZone = []

    curXBoundsLow = mpmath.mpi(interval.a, interval.mid)
    curXBoundsUp = mpmath.mpi(interval.mid, interval.b)

    xBounds[i] = curXBoundsLow
    dfdxLow = dfdx(*xBounds)

    if bool(dfdxLow >= 0): monotoneIncreasingZone.append(curXBoundsLow)
    elif bool(dfdxLow <= 0): monotoneDecreasingZone.append(curXBoundsLow)
    else: nonMonotoneZone.append(curXBoundsLow)

    xBounds[i] = curXBoundsUp
    dfdxUp = dfdx(*xBounds)

    if bool(dfdxUp >= 0): monotoneIncreasingZone.append(curXBoundsUp)
    elif bool(dfdxUp <= 0): monotoneDecreasingZone.append(curXBoundsUp)
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

        if monotoneZone == [] :
            monotoneZone.append(newInterval)
            return removeListInList(monotoneZone)

        else:
            for i in range(0, len(newInterval)):
                monotoneZone.append(newInterval[i])

            return joinIntervalSet(monotoneZone, relEpsX*red_disconti, absEpsX*red_disconti)

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

    if newIntervals != []: curIntervals.append(copy.deepcopy(newIntervals))


def checkTolerance(interval, relEpsX):
    """ checks if width of intervals is smaller than a given relative tolerance relEpsX

    Args:
        :interval:           set of intervals in mpmath.mpi-logic
        :relEpsX:            relative x tolerance

    Return:
        :reducedInterval:    set of intervals with a higher width than absEps

    """

    reducedInterval = []

    for i in range(0, len(interval)):
        if interval[i].mid != 0:
            if interval[i].delta/ abs(interval[i].mid) > relEpsX:
                reducedInterval.append(interval[i])
        elif interval[i].a != 0:
            if interval[i].delta / abs(interval[i].a) > relEpsX:
                reducedInterval.append(interval[i])
        elif interval[i].b != 0:
            if interval[i].delta / abs(interval[i].b) > relEpsX:
                reducedInterval.append(interval[i])

    return reducedInterval


def reduceNonMonotoneIntervals(args):
    """ reduces non monotone intervals by simply calculating function values for
    interval segments of a discretized variable interval and keeps those segments
    that intersect with bi. The discretization resolution is defined in dict_options.

    Args:
        :nonMonotoneZone:    list with non monotone variable intervals
        :reducedIntervals:   lits with reduced non monotone variable intervals
        :fx:                 variable-dependent function in mpmath.mpi logic
        :i:                  integer with current iteration variable index
        :xBounds:            numpy array with set of variable bounds
        :bi:                 current function residual bounds
        :dict_options:       for function and variable interval tolerances in the used
                             algorithms and resolution of the discretization

    Return:                  reduced x-Interval(s) and list of monotone x-intervals
        
 """   
    nonMonotoneZone = args["0"]
    reducedIntervals = args["1"]
    fx = args["2"]
    i = args["3"]
    xBounds = args["4"]
    bi = args["5"]
    dict_options = args["6"]
    
    relEpsX = dict_options["relTol"]
    precision = getPrecision(xBounds)
    resolution = dict_options["resolution"]

    for curNonMonZone in nonMonotoneZone:
        curInterval = convertIntervalBoundsToFloatValues(curNonMonZone)
        x = numpy.linspace(curInterval[0], curInterval[1], resolution)

        fLowValues, fUpValues = getFunctionValuesIntervalsOfXList(x, fx, xBounds, i)
        for k in range(0, len(fLowValues)):
            if ivIntersection(mpmath.mpi(fLowValues[k], fUpValues[k]), bi):
                reducedIntervals.append(mpmath.mpi(x[k], x[k+1]))
                reducedIntervals = joinIntervalSet(reducedIntervals, relEpsX, precision)

    return reducedIntervals


def getFunctionValuesIntervalsOfXList(x, f, xBounds, i):
    """ calculates lower and upper function value bounds for segments that are
    members of a list and belong to a discretized variable interval.

    Args:
        :x:          numpy list with segments for iteration variable xi
        :f:          x-dependent function in mpmath.mpi logic
        :xBounds:    numpy array with variable bounds in mpmath.mpi.logic
        :i:          current iteration variable index

    Return:         list with lower function value bounds within x and upper
                    function value bounds within x

    """

    funValuesLow = []
    funValuesUp = []

    for j in range(0, len(x)-1):
        xBounds[i] = mpmath.mpi(x[j], x[j+1])
        curfunValue = f(*xBounds)

        funValuesLow.append(float(mpmath.mpf(curfunValue.a)))
        funValuesUp.append(float(mpmath.mpf(curfunValue.b)))
        
    return funValuesLow, funValuesUp


def reduceTwoIVSets(ivSet1, ivSet2):
    """ reduces two interval sets to one and keeps only the resulting interval
    when elements of both intervals intersect. Each element of the longer interval
    set is compared to the list of the shorter interval set.

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
    """ intersects elements of a set of sets with disjoint intervals and returns a list with
    the intersecting intervals.
    
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

    for i in range(0, len(ivSet)):
        try:
            newIV = ivIntersection(iv, ivSet[i])
        except: return []

        if newIV != []: return newIV
    return []


def checkWidths(X, relEps, absEps):
    """ returns the maximum interval width of a set of intervals X

        Args:
            :X:                     list with set of intervals

        Return:
            :mpmath.mpi interval:   of maximum width

    """

    almostEqual = False * numpy.ones(len(X), dtype = bool)
    for i in range(0,len(X)):
         if(mpmath.almosteq(X[i].a, X[i].b, relEps, absEps)):
             almostEqual[i] = True

    return almostEqual.all()


def NewtonReduction(newtonSystemDic, xBounds, i, dict_options):
    """ Computation of the Interval-Newton Method to reduce the single interval xBounds[i]:
     
        Args: 
            :Boundsmid:     center of intervals
            :fmid:          evaluated functions with midpoint
            :Jacmid:        evaluated jacobimatrix with midpoint
            :JacInterval:   evaluated jacobimatrix with Bounds
            :JacmidInv:     inverse of Jacmid
            :xBounds:       current Bounds
            :i:             index of reducing Bound
            
        Return:
            :interval:   interval(s) of mpi format from mpmath library where
                         solution for x can be in, if interval remains [] there
                         is no solution within the initially guessed interval of x                  
    """
    interval=[]
             
    Boundspoint = newtonSystemDic['Boxpoint']
    fpoint = newtonSystemDic['f(Boxpoint)']
    JacInterval = newtonSystemDic['J(Box)']
    Y = newtonSystemDic['J(Boxpoint)-1']

    intersection = xBounds[i]
    for bp in range(len(Boundspoint)):
        Yfmid = numpy.dot(Y[bp][i],fpoint[bp])
        D = numpy.dot(Y[bp][i],JacInterval[:,i])
        ivsum=0
        for j in range(0, len(JacInterval[i])):
            if j!=i:
                ivsum = ivsum + numpy.dot(numpy.dot(Y[bp][i], JacInterval[:,j]), (xBounds[j]-Boundspoint[bp][j]))
                
        N = Boundspoint[bp][i] - ivDivision((Yfmid+ivsum),mpmath.mpi(D))[0]
    
        if N.a == '-inf' or N.b=='+inf':
            N = xBounds[i]
            
        intersection = ivIntersection(N, intersection)
        if intersection == []:
            return intersection
        if checkVariableBound(intersection, dict_options):
                        break


    if intersection !=[]: interval.append(intersection)
    return interval



def HC4(model, xBounds):
    """reduces the bounds of all variables in every model function based on HC4 hull-consistency
    Args:
        :model: instance of class-Model
        :xBounds:   current Bounds of Box
    Return: 
        :pyibex IntervalVector with reduced bounds 
    """

    HC4reduced_IvV = pyibex.IntervalVector(eval(mpmath.nstr(xBounds.tolist())))
    currentIntervalVector = HC4reduced_IvV
    for f in model.fSymbolic: 
        stringF = str(f).replace('log', 'ln').replace('**', '^')
        for i,s in enumerate(tuple(reversed(tuple(sympy.ordered(model.xSymbolic))))):
            if s in f.free_symbols:
                stringF = stringF.replace(str(s), 'x['+str(model.xSymbolic.index(s))+']')
           
        pyibexFun = pyibex.Function('x['+str(len(xBounds))+']', stringF)
        ctc = pyibex.CtcFwdBwd(pyibexFun)
        ctc.contract(currentIntervalVector)
        HC4reduced_IvV = HC4reduced_IvV & currentIntervalVector

        if HC4reduced_IvV.is_empty():
            return HC4reduced_IvV

    return HC4reduced_IvV



def HybridGS(newtonSystemDic, xBounds, i, dict_options):
        """ Computation of the Interval-Newton Method with hybrid approach to reduce the single interval xBounds[i]: 
        Args: 
            :newtonSystemDic:   dictionary, containing:
                :Boxpoint:      Point in IntervalBox
                :f(Boxpoint):   functionvalues at Boxpoint
                :JacInterval:   evaluated jacobimatrix with Bounds
            :xBounds:       current Bounds
            :i:             index of reducing Bound
            
        Return:
            :interval:   interval(s) of mpi format from mpmath library where
                         solution for x can be in, if interval remains [] there
                         is no solution within the initially guessed interval of x                  
        """

        interval=[]
                 
        Boundspoint = newtonSystemDic['Boxpoint']
        fpoint = newtonSystemDic['f(Boxpoint)']
        JacInterval = newtonSystemDic['J(Box)']
        #Y = newtonSystemDic['J(Boxpoint)-1']

        intersection = xBounds[i]
        for bp in range(len(Boundspoint)):
            for y in range(len(xBounds)):
                Yfmid = fpoint[bp][y]
                D = JacInterval[y,i]
                ivsum=0
                if D!=0 and D!=mpmath.mpi('-inf','+inf'):
                    for j in range(0, len(JacInterval[i])):
                        if j!=i:
                            ivsum = ivsum + numpy.dot(JacInterval[y,j], (xBounds[j]-Boundspoint[bp][j]))
                    
                    try:    N = Boundspoint[bp][i] - ivDivision((Yfmid+ivsum),mpmath.mpi(D))[0]
                    except: N = mpmath.mpi('-inf', '+inf')
                
                    if N.a == '-inf' or N.b=='+inf':
                        N = xBounds[i]
                        
                    intersection = ivIntersection(N, intersection)
                    if intersection == []:
                        return intersection
                    if checkVariableBound(intersection, dict_options):
                        return [intersection]

        if intersection !=[]: interval.append(intersection)
        return interval