"""
***************************************************
Import packages
***************************************************
"""
import copy
import numpy
import sympy
import mpmath
import itertools
from modOpt.constraints import parallelization
import time
from modOpt.constraints.FailedSystem import FailedSystem

__all__ = ['reduceMultipleXBounds', 'reduceXIntervalByFunction', 'reduceTwoIVSets',
           'checkWidths', 'getPrecision']

"""
***************************************************
Algorithm for interval Nesting procedure
***************************************************
"""

def reduceMultipleXBounds(model, functions, dict_options):
    """ reduction of multiple solution interval sets
    
    Args:
        :xBounds:               list with list of interval sets in mpmath.mpi logic
        :xSymbolic:             list with symbolic iteration variables in sympy logic
        :parameter:             list with parameter values of equation system        
        :model:                 object of type model
        :dimVar:                integer that equals iteration variable dimension        
        :blocks:                list with block indices of equation system       
        :dict_options:          dictionary with user specified algorithm settings
    
    Return:
        :results:               dictionary with newXBounds and a list of booleans
                                that marks all non reducable variable bounds with True.
                                If solver terminates because of a NoSolution case the
                                critical equation is also stored in results for the error
                                analysis.
            
    """
    dimVar = len(model.xBounds[0])
    results = {}
    newXBounds = []
    xAlmostEqual = False * numpy.ones(len(model.xBounds), dtype=bool)

    for k in range(0, len(model.xBounds)):
        
        if dict_options['method'] == 'b_tight':
            output = reduceXbounds_b_tight(functions, model.xBounds[k], dict_options) 

        else:
            if not dict_options["Parallel Variables"]:
                output = reduceXBounds(model.xBounds[k], model.xSymbolic, model.fSymbolic,
                                  model.blocks, dict_options)
            else: 
                output = parallelization.reduceXBounds(model.xBounds[k], model.xSymbolic, model.fSymbolic,
                                  model.blocks, dict_options)
        
        intervalsPerm = output["intervalsPerm"]
        xAlmostEqual[k] = output["xAlmostEqual"]
        
        if output.__contains__("noSolution") :
            saveFailedIntervalSet = output["noSolution"]
            break
        
        for m in range(0, len(intervalsPerm)):          
            x = numpy.empty(dimVar, dtype=object)     
            x[model.colPerm]  = numpy.array(intervalsPerm[m])           
            newXBounds.append(x)
              
    
    
    if newXBounds == []: 
        results["noSolution"] = saveFailedIntervalSet
    
    else:
        model.xBounds = newXBounds    
        
    results["xAlmostEqual"] = xAlmostEqual
    return results


def reduceXbounds_b_tight(functions, xBounds, dict_options):
    """ main function to reduce all variable bounds with all equations
    
    Args:
        :functions:         list with instances of class function
        :xBounds:           list with variable bounds in mpmath.mpi formate
        :dict_options:      dictionary with tolerance options
        
    Return:
        :output:            dictionary with new interval sets(s) in a list and
                            eventually an instance of class failedSystem if
                            the procedure failed.
        
    """
    
    varBounds = {}
            
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
                  
    return get_newXBounds(varBounds, xBounds)


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
    output["intervalsPerm"] = []
    failedSystem = FailedSystem(f.f_sym, f.x_sym[varBounds['Failed_xID']])
    output["noSolution"] = failedSystem
    output["xAlmostEqual"] = False 
    return output    
    
   
def get_newXBounds(varBounds, xBounds):
    """ creates output dictionary with new variable bounds in a successful 
    reduction process. Also checks if variable bounds change by xUnchanged.
    
    Args:
        :varBounds:     dictionary with reduced variable bounds
        :xBounds:       numpy array with variable bounds in mpmath.mpi formate       
    
    Return:
        :output:        dictionary with information about variable reduction
        
    """   
    output = {}
    xNewBounds = []
    xUnchanged = True
    
    for i in range(0, len(xBounds)):
        xNewBounds.append(varBounds['%d' % i])               
        
        if varBounds['%d' % i] != xBounds[i] and xUnchanged:
            xUnchanged = False
                
    output["xAlmostEqual"] = xUnchanged                   
    output["intervalsPerm"] = list(itertools.product(*xNewBounds))
    return output


def reduceXBounds_byFunction(f, xBounds, dict_options, varBounds):
    """ reduces all n bounds of variables (xBounds) that occur in a certain 
    function f and stores it in varBounds.
    
    Args:
        :f:             instance of class function
        :xBounds:       list with n variable bounds
        :varBounds:     dictionary with n reduced variable bounds. The key 
                        'Failed_xID' is used to store a variable's global ID in
                        case  a reduced interval is empty.
                        
    """    
    
    for x_id in range(0, len(f.glb_ID)): # get g(x) and b(x),y 
        if mpmath.almosteq(xBounds[x_id].a, xBounds[x_id].b, 
                           dict_options["absTol"],
                           dict_options["relTol"]): 
            store_reduced_xBounds(f, x_id, [xBounds[x_id]], varBounds)
            continue
        if dict_options["Parallel b's"]:
            b = parallelization.get_tight_bBounds(f, x_id, xBounds, dict_options)
            
        else:    
            b = get_tight_bBounds(f, x_id, xBounds, dict_options) # TODO: Parallel
            
        if b == []: reduced_xBounds = [xBounds[x_id]]
        else:
            reduced_xBounds = reduce_x_by_gb(f.g_sym[x_id], f.dgdx_sym[x_id],
                                                     b, f.x_sym, x_id,
                                                     copy.deepcopy(xBounds),
                                                     dict_options)                                                     
                          
        store_reduced_xBounds(f, x_id, reduced_xBounds, varBounds)


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
        varBounds['%d' % f.glb_ID[x_id]] = reduceTwoIVSets(varBounds['%d' % f.glb_ID[x_id]],
                 reduced_interval)
    
    if varBounds['%d' % f.glb_ID[x_id]] == [] or varBounds['%d' % f.glb_ID[x_id]] == [[]]: 
                varBounds['Failed_xID'] = x_id
                

def get_tight_bBounds(f, x_id, xBounds, dict_options):
    """ returns tight b bound interval based on all variables y that are evaluated
    separatly for all other y intervals beeing constant.
    
    Args:
        :f:                 instance of class function
        :xBounds:           list with variable bonunds in mpmath.mpi formate
        :x_id:              index (integer) of current variable bound that is reduced
        :dict_options:      dictionary with tolerance options
        
    Return:
        b interval in mpmath.mpi formate and [] if error occured (check for complex b) 
    
    """
    b_max = []
    b_min = []
    
    b = getBoundsOfFunctionExpression(f.b_sym[x_id], f.x_sym, xBounds)  
    if mpmath.almosteq(b.a, b.b, dict_options["relTol"], dict_options["absTol"]):
        return b
    
    if len(f.glb_ID)==1: # this is import if b is interval but there is only one variable in f (for design var intervals in future)
        return b
    
    for y_id in range(0, len(f.glb_ID)): # get b(y)
          
        if  x_id != y_id:
            get_tight_bBounds_y(f, x_id, y_id, xBounds, b_min, b_max, dict_options)
        
    if b_min !=[] and b_max != []:
        return mpmath.mpi(max(b_min), min(b_max))
    else: return []


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
        b_min.append(b.a)
        b_max.append(b.b)
        
    else:        
        incr_zone, decr_zone, nonmon_zone = get_conti_monotone_intervals(f.dbdx_sym[x_id][y_id], 
                                                                         f.x_sym, 
                                                                         y_id, 
                                                                         copy.deepcopy(xBounds), 
                                                                         dict_options)    
        add_b_min_max(f, incr_zone, decr_zone, nonmon_zone, x_id, y_id, 
                      copy.deepcopy(xBounds), b_min, b_max)    
    


def add_b_min_max(f, incr_zone, decr_zone, nonmon_zone, x_id, y_id, xBounds, b_min, b_max):
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
    
    """
    
    if incr_zone != []: get_bounds_incr_zone(f.b_sym[x_id], 
                    f.x_sym,
                    y_id, 
                    copy.deepcopy(xBounds), 
                    incr_zone,
                    b_max,
                    b_min) 
 
    if decr_zone != []: get_bounds_decr_zone(f.b_sym[x_id],
                    f.x_sym,
                    y_id, 
                    copy.deepcopy(xBounds),
                    decr_zone,
                    b_max,
                    b_min) 
                              
    if nonmon_zone != []: get_bounds_nonmon_zone(f.b_sym[x_id],
                      f.x_sym,
                      y_id,
                      copy.deepcopy(xBounds),
                      nonmon_zone,
                      b_max,
                      b_min)     


def get_bounds_incr_zone(b_sym, x_sym, i, xBounds, incr_zone, b_max, b_min):
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
    

def get_bounds_decr_zone(b_sym, x_sym, i, xBounds, decr_zone, b_max, b_min):
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
    

def get_bounds_nonmon_zone(b_sym, x_sym, i, xBounds, nonmon_zone, b_max, b_min):
    """ stores maximum and minimum value of b(y) in b_max and b_min for an 
    interval y = x_sym[i], where b is non-monotone.
    
    Args:
        :b_sym:         function in sympy logic
        :x_sym:         list with n variables of b_sym in sympy logic
        :i:             index of variable y in x_sym
        :xBounds:       list with n variable bounds in mpmath.mpi logic
        :nonmon_zone:   list with non-monotone intervals of b(y) in mpmath.mpi logic
        :b_max:         list with maximun values of b(y) for different y of b(x,y)
        
    """
    cur_max = []
    cur_min = []
    for interval in nonmon_zone:
        xBounds[i] = interval
        b_bounds = getBoundsOfFunctionExpression(b_sym, x_sym, xBounds)   
        
        if type(b_bounds) == mpmath.iv.mpc: continue        
        cur_max.append(float(mpmath.mpf(b_bounds.b)))
        cur_min.append(float(mpmath.mpf(b_bounds.a)))
    
    if cur_max != []: b_max.append(max(cur_max))
    if cur_min != []: b_min.append(min(cur_min))
    

def reduceMultipleXBounds_old(xBounds, xSymbolic, parameter, model, dimVar, blocks,
                          dict_options):
    """ reduction of multiple solution interval sets
    
    Args:
        :xBounds:               list with list of interval sets in mpmath.mpi logic
        :xSymbolic:             list with symbolic iteration variables in sympy logic
        :parameter:             list with parameter values of equation system        
        :model:                 object of type model
        :dimVar:                integer that equals iteration variable dimension        
        :blocks:                list with block indices of equation system       
        :dict_options:          dictionary with user specified algorithm settings
    
    Return:
        :results:               dictionary with newXBounds and a list of booleans
                                that marks all non reducable variable bounds with True.
                                If solver terminates because of a NoSolution case the
                                critical equation is also stored in results for the error
                                analysis.
            
    """
    
    results = {}
    newXBounds = []
    xAlmostEqual = False * numpy.ones(len(xBounds), dtype=bool)

    for k in range(0,len(xBounds)):

        FsymPerm, xSymbolicPerm, xBoundsPerm = model.getBoundsOfPermutedModel(xBounds[k], 
                                                                              xSymbolic, 
                                                                              parameter)
        #dict_options["precision"] = getPrecision(xBoundsPerm)
        
        if not dict_options["Parallel Variables"]:
            output = reduceXBounds(xBoundsPerm, xSymbolicPerm, FsymPerm,
                                  blocks, dict_options)#, boundsAlmostEqual)
        else:
            output = parallelization.reduceXBounds(xBoundsPerm, xSymbolicPerm, FsymPerm,
                                                  blocks, dict_options)#, boundsAlmostEqual)
        
        intervalsPerm = output["intervalsPerm"]
        xAlmostEqual[k] = output["xAlmostEqual"]
        
        if output.__contains__("noSolution") :
            saveFailedIntervalSet = output["noSolution"]
            break
        
        for m in range(0, len(intervalsPerm)):          
            x = numpy.empty(dimVar, dtype=object)     
            x[model.colPerm]  = numpy.array(intervalsPerm[m])           
            newXBounds.append(x)
            
            #if checkWidths(xBounds[k], x, dict_options["relTolX"], 
            #               dict_options["absTolX"]): 
            #    xAlmostEqual[k] = True
            #    break
            
    results["newXBounds"] = newXBounds
    
    if newXBounds == []: 
        results["noSolution"] = saveFailedIntervalSet
        
    results["xAlmostEqual"] = xAlmostEqual
    return results


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


def reduceXBounds(xBounds, xSymbolic, f, blocks, dict_options):
    """ solves an equation system blockwise. For block dimensions > 1 each 
    iteration variable interval of the block is reduced sequentially by all 
    equations of the block. The narrowest bounds from this procedure are taken
    over.
     
        Args: 
            xBounds:            One set of variable interavls as numpy array
            xSymbolic:          list with symbolic variables in sympy logic
            f:                  list with symbolic equation system in sympy logic
            blocks:             List with blocklists, whereas the blocklists contain
                                the block elements with index after permutation
            dict_options:       dictionary with solving settings
        Returns:
            :output:            dictionary with new interval sets(s) in a list and
                                eventually an instance of class failedSystem if
                                the procedure failed.
                        
    """    
    absEpsX = dict_options["absTol"]
    relEpsX = dict_options["relTol"]
    output = {}
    xNewBounds = copy.deepcopy(xBounds)
    xUnchanged = True
    
    for b in range(0, len(blocks)):
        blockDim = len(blocks[b])
        for n in range(0, blockDim):
            y = [] 
            j = blocks[b][n]
                     
            if dict_options["Debug-Modus"]: print(j)
            
            if checkVariableBound(xBounds[j], relEpsX, absEpsX):
                        xNewBounds[j] = [xBounds[j]]
                        continue
                    
            for m in range(0, blockDim):
                i = blocks[b][m]
                if xSymbolic[j] in f[i].free_symbols:                    
                    if y == []: y = reduceXIntervalByFunction(xBounds, xSymbolic, 
                           f[i], j, dict_options)
                    else: 
                        y = reduceTwoIVSets(y, reduceXIntervalByFunction(xBounds, 
                                                    xSymbolic, f[i], j, dict_options))
                    if y == [] or y ==[[]]:
                        output["intervalsPerm"] = []
                        failedSystem = FailedSystem(f[i], xSymbolic[j])
                        output["noSolution"] = failedSystem
                        output["xAlmostEqual"] = False 
                        return output

            xNewBounds[j] = y

#            if len(xNewBounds[j]) == 1: 
#                boundsAlmostEqual[j] = checkVariableBound(xNewBounds[j][0], relEpsX, absEpsX)
            
            if xNewBounds[j] != [xBounds[j]] and xUnchanged:
                xUnchanged = False
                
    output["xAlmostEqual"] = xUnchanged     
    output["intervalsPerm"] = list(itertools.product(*xNewBounds))
    return output


def checkVariableBound(newXInterval, relEpsX, absEpsX):
    """ if lower and upper bound of a variable are almost equal the boolean 
    boundsAlmostEqual is set to true.
    
    Args:
        newXInterval:       variable interval in mpmath.mpi logic
        relEpsX:            relative variable interval tolerance
        absEpsX:            absolute variable interval tolerance
        
    Returns:                True, if lower and upper variable bound are almost
                            equal.
       
    """
    
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
        
    Return:
        reduced x interval in mpmath.mpi formate 
    
    """
    
    g = getBoundsOfFunctionExpression(g_sym, x_sym, xBounds)
    dgdx = getBoundsOfFunctionExpression(dgdx_sym, x_sym, xBounds)
    
    g_sym = lambdifyToMpmathIv(x_sym, g_sym)

    
    if type(g) == mpmath.iv.mpc or type(dgdx) == mpmath.iv.mpc: return [xBounds[i]] # TODO: Complex case
    
    if not x_sym[i] in dgdx_sym.free_symbols and g == dgdx*xBounds[i]: # Linear Case -> solving system directly
        dgdx_sym = lambdifyToMpmathIv(x_sym, dgdx_sym)
        return getReducedIntervalOfLinearFunction(dgdx_sym, i, xBounds, b)   

    else: # Nonlinear Case -> solving system by interval nesting
        dgdx_sym = lambdifyToMpmathIv(x_sym, dgdx_sym)
        return getReducedIntervalOfNonlinearFunction(g_sym, 
                                                     dgdx_sym, 
                                                     dgdx, 
                                                     i, 
                                                     xBounds, 
                                                     b, 
                                                     dict_options)

    
def reduceXIntervalByFunction(xBounds, xSymbolic, f, i, dict_options): # One function that gets fi, xi and does the procedure
    """ reduces variable interval by either solving a linear function directly
    with Gauss-Seidl-Operator or finding the reduced variable interval(s) of a
    nonlinear function by interval nesting
     
        Args: 
            xBounds:            One set of variable interavls as numpy array
            xSymbolic:          list with symbolic variables in sympy logic
            f:                  symbolic equation in sympy logic
            i:                  index for iterated variable interval
            dict_options:       dictionary with solving settings

        Returns:                list with new set of variable intervals
                        
    """       
    xBounds = copy.deepcopy(xBounds)
    fx, fWithoutX = splitFunctionByVariableDependency(f, xSymbolic[i])
    dfdX_sympy = sympy.diff(fx, xSymbolic[i]) 
    fx, dfdX, bi, fxInterval, dfdxInterval, xBounds = calculateCurrentBounds(fx, 
                    fWithoutX, dfdX_sympy, xSymbolic, i, xBounds, dict_options)
    
    if bi == [] or fxInterval == [] or dfdxInterval == []: 
        return [xBounds[i]]
    
    #if(df2dxInterval == 0 and fxInterval == dfdxInterval*xBounds[i]): # Linear Case -> solving system directly
    if not xSymbolic[i] in dfdX_sympy.free_symbols and fxInterval == dfdxInterval*xBounds[i]: # Linear Case -> solving system directly
        return getReducedIntervalOfLinearFunction(dfdX, i, xBounds, bi)
             
    else: # Nonlinear Case -> solving system by interval nesting
        return getReducedIntervalOfNonlinearFunction(fx, dfdX, dfdxInterval, 
                                                     i, xBounds, bi, dict_options)

    
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
    arithmetic interval functions (mpmath.iv)
    
    """
    
    mpmathIv = {"exp" : mpmath.iv.exp,
            "sin" : mpmath.iv.sin,
            "cos" : mpmath.iv.cos,
            "log" : mpmath.iv.log}
    
    return sympy.lambdify(x,f, mpmathIv)


def calculateCurrentBounds(fx, fWithoutX, dfdX, xSymbolic, i, xBounds, dict_options):
    """ calculates bounds of function fx, the residual fWithoutX, first and second
    derrivative of function fx with respect to variable xSymbolic[i] (dfX, df2dX).
    It uses the tolerance values from dict_options in the complex case to remove 
    complex intervals by interval nesting.
    
    Args:
        :fx:                 symbolic function in sympy logic (contains current 
                             iteration variable X)
        :fWithoutX:          symbolic function in sympy logic (does not contain 
                             current iteration variable X)                            
        :xSymbolic:          list with symbolic variables in sympy logic
        :i:                  index of current iteration variable
        :xBounds:            numpy array with variable bounds
        :dict_options:       dictionary with entries about stop-tolerances

        
    Returns:
        :dfdX:               first derrivative of function fx with respect to 
                             variable xSymbolic[i] in mpmath.mpi-logic
        :bi:                 residual interval in mpmath.mpi logic
        :fxInterval:         function interval in mpmath.mpi logic
        :dfdxInterval:       Interval of first derrivative in mpmath.mpi logic 
        :df2dxInterval:      Interval of second derrivative in mpmath.mpi logic 
        :xBounds:            real set of variable intervals in mpmath.mpi logic 
    
    """
       
    #dfdX = sympy.diff(fx, xSymbolic[i]) 
    #df2dX = sympy.diff(dfdX, xSymbolic[i]) 
    
    fxBounds = lambdifyToMpmathIv(xSymbolic, fx)
    dfdxBounds = lambdifyToMpmathIv(xSymbolic, dfdX)

    #df2dxBounds = lambdifyToMpmathIv(xSymbolic, df2dX)
    
    try:
        bi = getBoundsOfFunctionExpression(-fWithoutX, xSymbolic, xBounds)

    except:
        fWithoutX = reformulateComplexExpressions(fWithoutX)
        try : bi = getBoundsOfFunctionExpression(-fWithoutX, xSymbolic, xBounds)
        except: return fxBounds, dfdxBounds, [], [], [], xBounds
      
    try:
       fxInterval = getBoundsOfFunctionExpression(fx, xSymbolic, xBounds)
       if type(fxInterval) == mpmath.iv.mpc: fxInterval = []
       # if type(fxInterval) == mpmath.iv.mpc:
       #    newXBounds = reactOnComplexError(fxBounds, xSymbolic, i, xBounds, dict_options)
       #    xBounds[i] = newXBounds
       #    fxInterval = getBoundsOfFunctionExpression(fx, xSymbolic, xBounds)
    except: return fxBounds, dfdxBounds, bi, [], [], xBounds

    try:
       dfdxInterval = getBoundsOfFunctionExpression(dfdX, xSymbolic, xBounds)
       if type(dfdxInterval) == mpmath.iv.mpc: dfdxInterval = []
       #if type(dfdxInterval) == mpmath.iv.mpc:
       #    newXBounds = reactOnComplexError(dfdxBounds, xSymbolic, i, xBounds, dict_options)
       #    xBounds[i] = newXBounds
       #    dfdxInterval = getBoundsOfFunctionExpression(dfdX, xSymbolic, xBounds)
    except: return fxBounds, dfdxBounds, bi, fxInterval, [], xBounds
       
    #try:    
    #    df2dxInterval = getBoundsOfFunctionExpression(df2dX, xSymbolic, xBounds)
        
    #except:
    #    newXBounds = reactOnComplexError(df2dxBounds, xSymbolic, i, xBounds, dict_options)
    #    if newXBounds == []: 
    #        return fxBounds, dfdxBounds, [], [], [], [], xBounds
    #    else: 

    return fxBounds, dfdxBounds, bi, fxInterval, dfdxInterval, xBounds   
    
    
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
    fMpmathIV = lambdifyToMpmathIv(xSymbolic, f)
    return  mpmath.mpi(fMpmathIV(*xBounds))
    #return  mpmath.mpi(str(fMpmathIV(*xBounds)))
            
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
    
    for curInterval in interval:
        if mpmath.almosteq(curInterval.a, curInterval.b, absEpsX, relEpsX):
                interval.remove(curInterval)
    return interval
                
    
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
        :a:                  first symbolic derivative of function f with respect to x
        :i:                  integer with current iteration variable index
        :xBounds:            numpy array with set of variable bounds
        :bi:                 current function residual bounds
    
    Return:                  reduced x-Interval(s)
    
    """
    
    aInterval = mpmath.mpi(str(a(*xBounds)))
    
    checkAndRemoveComplexPart(aInterval)
        
    if bool(0 in bi - aInterval * xBounds[i]) == False: return [] # if this is the case, there is no solution in xBoundsi

    if bool(0 in bi) and bool(0 in aInterval):  # if this is the case, bi/aInterval would return [-inf, +inf]. Hence the approximation of x is already smaller
                return [xBounds[i]]
    else: 
        return gaussSeidelOperator(aInterval, bi, xBounds[i]) # bi/aInterval  


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
    
    increasingZones = []
    decreasingZones = []
    nonMonotoneZones = []
    old_xiBounds = xBounds[i]
    cur_xiBounds = [xBounds[i]]
    
    dfdx = getBoundsOfFunctionExpression(dfdx_sym, x_sym, xBounds)
    
    if type(dfdx) == mpmath.iv.mpc: return [], [], [old_xiBounds]
    
    dfdx_sym = lambdifyToMpmathIv(x_sym, dfdx_sym)
    
    if '-inf' in dfdx or '+inf' in dfdx: 
        cur_xiBounds = getContinuousFunctionSections(dfdx_sym, i, xBounds, dict_options)
        
        if cur_xiBounds == []: return [], [], [old_xiBounds]
        
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

    increasingZones = []
    decreasingZones = []
    nonMonotoneZones = []
    reducedIntervals = []
    orgXiBounds = [copy.deepcopy(xBounds[i])]
    curXiBounds = orgXiBounds
    
    if dfdXInterval == []: return []
    
    if '-inf' in dfdXInterval or '+inf' in dfdXInterval: # condition for discontinuities
        curXiBounds = getContinuousFunctionSections(dfdX, i, xBounds, dict_options)
        if curXiBounds == []: return orgXiBounds
    
    for curInterval in curXiBounds:
        xBounds[i] = curInterval
        increasingZone, decreasingZone, nonMonotoneZone = getMonotoneFunctionSections(dfdX, i, xBounds, dict_options)
        if increasingZone !=[]: increasingZones.append(increasingZone)                                                                            
        if decreasingZone !=[]: decreasingZones.append(decreasingZone)    
        if nonMonotoneZone !=[]: nonMonotoneZones.append(nonMonotoneZone)
    
    if increasingZones !=[]:
            increasingZones = removeListInList(increasingZones)                
            reducedIntervals = reduceMonotoneIntervals(increasingZones, reducedIntervals, fx, 
                                    xBounds, i, bi, dict_options, increasing = True)  
 
    if decreasingZones !=[]:
            decreasingZones = removeListInList(decreasingZones)                
            reducedIntervals = reduceMonotoneIntervals(decreasingZones, reducedIntervals, fx, 
                                    xBounds, i, bi, dict_options, increasing = False)  
       
    if nonMonotoneZones !=[]:
        nonMonotoneZones = removeListInList(nonMonotoneZones)   
        reducedIntervals = reduceNonMonotoneIntervals(nonMonotoneZones, reducedIntervals, 
                                                          fx, i, xBounds, bi, 
                                                          dict_options)
    reducedIntervals = reduceTwoIVSets(reducedIntervals, orgXiBounds)
    return reducedIntervals

    
#def getReducedIntervalOfNonlinearFunction_old(fx, dfdX, df2dX, dfdXInterval, df2dxInterval, xSymbolic, i, xBounds, bi, dict_options):
#    """ checks function for monotone sections in x and reduces them one after the other.
#    
#    Args: 
#        :fx:                 symbolic x-depending part of function f in mpmath.mpi logic
#        :dfdX:               first symbolic derivative of function f with respect to x
#                             in mpmath.mpi logic
#        :df2dX:              second symbolic derivative of function f with respect to x
#                             in mpmath.mpi logic
#        :dfdXInterval:       first derivative of function f with respect to x at xBounds
#        :df2dxInterval:      second derivative of function f with respect to x at xBounds        
#        :xSymbolic:          list with symbolic variables in sympy logic
#        :i:                  integer with current iteration variable index
#        :xBounds:            numpy array with set of variable bounds
#        :bi:                 current function residual bounds
#        :dict_options:       for function and variable interval tolerances in the used
#                            algorithms
#
#    Return:                reduced x-Interval(s) and list of monotone x-intervals
#    
#    """
#    
#    orgXiBounds = copy.deepcopy(xBounds[i])
#
#    if (dfdXInterval == [] or df2dxInterval == []): return [orgXiBounds]
#    if (dfdXInterval >= 0):
#        
#        if bool(df2dxInterval >= 0) == False and bool(df2dxInterval <= 0) == False:
#            
#            incrZoneIndf2dX, decrZoneIndf2dX = getContinuousFunctionSections(df2dX, xSymbolic, i, xBounds, dict_options)
#            increasingZone = removeListInList([incrZoneIndf2dX, decrZoneIndf2dX])
#            if increasingZone == []: return [orgXiBounds]
#            
#            reducedIntervals = []
#            reducedIntervals = reduceMonotoneIntervals(increasingZone, reducedIntervals, fx, xSymbolic, 
#                                    xBounds, i, bi, dict_options, increasing = True)    
#            return reducedIntervals
#        
#        return [monotoneIncreasingIntervalNesting(fx, xSymbolic, xBounds, i, bi, dict_options)]
#
#    
#    if (dfdXInterval <= 0):
#
#        if bool(df2dxInterval >= 0) == False and bool(df2dxInterval <= 0) == False:
#            
#            incrZoneIndf2dX, decrZoneIndf2dX = getContinuousFunctionSections(df2dX, xSymbolic, i, xBounds, dict_options)
#            decreasingZone = removeListInList([incrZoneIndf2dX, decrZoneIndf2dX])
#            if decreasingZone == []: return [orgXiBounds]
#            
#            reducedIntervals = []
#            reducedIntervals = reduceMonotoneIntervals(decreasingZone, reducedIntervals, fx, xSymbolic, 
#                                    xBounds, i, bi, dict_options, increasing = False)    
#            return reducedIntervals
#        
#        return [monotoneDecreasingIntervalNesting(fx, xSymbolic, xBounds, i, bi, dict_options)]
#        
#    else:
#        reducedIntervals = []
#        increasingZone, decreasingZone, nonMonotoneZone = getMonotoneFunctionSections(dfdX, xSymbolic, i, xBounds, 
#                                                                                      dict_options)
#
#        if increasingZone == [] and decreasingZone == [] and nonMonotoneZone == []: return [orgXiBounds]
#
#        reducedIntervals = reduceMonotoneIntervals(increasingZone, reducedIntervals, fx, xSymbolic, 
#                                 xBounds, i, bi, dict_options, increasing = True)
#        
#        reducedIntervals = reduceMonotoneIntervals(decreasingZone, reducedIntervals, fx, xSymbolic, 
#                                 xBounds, i, bi, dict_options, increasing = False)
#        if nonMonotoneZone != []:
#            reducedIntervals = reduceNonMonotoneIntervals(nonMonotoneZone, reducedIntervals, 
#                                                          fx, xSymbolic, i, xBounds, bi, 
#                                                          dict_options)
#        return reducedIntervals


def getContinuousFunctionSections(dfdx, i, xBounds, dict_options):
    """filters out discontinuities which either have a +/- inf derrivative.
    
    Args:
        :dfdx:                scalar first derivate of function in mpmath.mpi logic
        :i:                   index of differential variable
        :xBounds:             numpy array with variable bounds
        :dict_options:        dictionary with variable and function interval tolerances
    
    Return:
        :continuousZone:      continuous interval
    
    """

    tmax = dict_options["tmax"]
    absEpsX = dict_options["absTol"]
    relEpsX = dict_options["relTol"]   
    continuousZone = []
    
    interval = [xBounds[i]] 
    timeout = False
    t0 = time.clock()
      
    while interval != [] and timeout == False:    
        discontinuousZone = []
               
        for curInterval in interval:
            newContinuousZone = testIntervalOnContinuity(dfdx, curInterval, xBounds, i, discontinuousZone)
            continuousZone = addIntervaltoZone(newContinuousZone, continuousZone, dict_options)  
               
        interval = checkIntervalWidth(discontinuousZone, absEpsX, relEpsX)
        timeout = checkTimeout(t0, tmax, timeout)
        
        if timeout: continuousZone = []
        
    return continuousZone

    
#def getContinuousFunctionSections_old(df2dx, xSymbolic, i, xBounds, dict_options):
#    """seperates variable interval into variable interval sets where a function
#    with derivative dfdx is monontoneous
#    
#    Args:
#        :df2dx:               scalar second derivate of function in mpmath.mpi logic
#        :xSymbolic:           symbolic variables in derivative function
#        :i:                   index of differential variable
#        :xBounds:             numpy array with variable bounds
#        :dict_options:        dictionary with variable and function interval tolerances
#    
#    Return:
#        :monIncreasingZone:   monotone increasing intervals 
#        :monDecreasingZone:   monotone decreasing intervals 
#    
#    """
#    tmax = dict_options["tmax"]
#    relEpsX = dict_options["relTolX"]
#    relEpsF = dict_options["relTolF"]
#    absEpsF = dict_options["absTolF"]
#
#    lmax = dict_options["NoOfNonChangingValues"] 
#        
#    monIncreasingZone = []
#    monDecreasingZone = []
#    
#    interval = [xBounds[i]] 
#    l = 0
#    timeout = False
#    t0 = time.clock()
#    
#    while interval != [] and timeout == False and l < lmax:    
#        curIntervals = []
#               
#        for k in range(0, len(interval)):
# 
#            newIntervals, newMonIncreasingZone, newMonDecreasingZone, l = testIntervalOnMonotony(df2dx, interval[k], 
#                                                                                                 xBounds, i, l,
#                                                                                                 relEpsF, absEpsF)
#                        
#            monIncreasingZone = addIntervaltoZone(newMonIncreasingZone, monIncreasingZone, dict_options)            
#            monDecreasingZone = addIntervaltoZone(newMonDecreasingZone, monDecreasingZone, dict_options)
#            addIntervalToNonMonotoneZone(newIntervals, curIntervals)    
#               
#        interval = checkTolerance(removeListInList(curIntervals), relEpsX)
#        timeout = checkTimeout(t0, tmax, timeout)
#    
#    manipulateLowerUpperBounds(monIncreasingZone, df2dx, i, xBounds)
#    manipulateLowerUpperBounds(monDecreasingZone, df2dx, i, xBounds)
#    
#    return monIncreasingZone, monDecreasingZone


#def manipulateLowerUpperBounds(interval, fx, i, xBounds):
#    """ shifts interval bounds that create +inf/-inf-function value bounds
#    minimally to the side to remove this singular bound.
#    
#    Args:
#        :interval:      interval in mpmath.mpi logic
#        :fx:            function that is evaluated in mpmath.mpi logic
#        :i:             iteration variable index as integer
#        :xBounds:       set of variable bounds
#    
#    """
#    
#    xBounds = copy.deepcopy(xBounds)    
#    for j in range(0, len(interval)):
#        xBounds[i] = interval[j].a
#        
#        curdf2dx = fx(*xBounds) 
#        if curdf2dx.a == '-inf' or curdf2dx.b == '+inf': 
#            interval[j] = mpmath.mpi(interval[j].a + numpy.finfo(numpy.float).eps,
#            interval[j].b)
#            
#        xBounds[i] = interval[j].b
#        curdf2dx = fx(*xBounds) 
#        if curdf2dx.a == '-inf' or curdf2dx.b == '+inf': 
#            b = interval[j].b - numpy.finfo(numpy.float).eps
#            interval[j] = mpmath.mpi(interval[j].a, b.a)


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
    tmax = dict_options["tmax"]
    relEpsX = dict_options["relTol"]
    absEpsX = dict_options["absTol"]
    #maxIvNo = dict_options["maxIvNo"]   
    monIncreasingZone = []
    monDecreasingZone = []
    interval = [xBounds[i]] 
    
    timeout = False
    t0 = time.clock()
    
    while interval != [] and timeout == False: #and len(interval) < maxIvNo: #and dfdXconst == False:  
        
        curIntervals = []
               
        for xc in interval:
            newIntervals, newMonIncreasingZone, newMonDecreasingZone = testIntervalOnMonotony(dfdx, 
                                                    xc, xBounds, i)
            
            #if (newIntervals != [] and checkTolerance(newIntervals, relEpsX)==[]):
            #    newIntervals, monIncreasingZone, monDecreasingZone = discretizeAndEvaluateIntervals(dfdx, 
            #                                        xBounds, i, newIntervals, monIncreasingZone, 
            #                                        monDecreasingZone, dict_options)
            
            monIncreasingZone = addIntervaltoZone(newMonIncreasingZone, 
                                                          monIncreasingZone, dict_options)            
            monDecreasingZone = addIntervaltoZone(newMonDecreasingZone, 
                                                          monDecreasingZone, dict_options)
       
            #curIntervals = addIntervaltoZone(newIntervals, curIntervals, dict_options)
            curIntervals.append(newIntervals)
        curIntervals = removeListInList(curIntervals)
        #if interval == curIntervals: break
        interval = checkIntervalWidth(curIntervals, absEpsX, relEpsX)       
        timeout = checkTimeout(t0, tmax, timeout)
        if timeout: interval = joinIntervalSet(interval, relEpsX, absEpsX)
            
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
        newIntervals, monIncreasingZone, monDecreasingZone = discretizeAndEvaluataInterval(dfdX, 
                                        xBounds, i, interval, newIntervals, monIncreasingZone, 
                                        monDecreasingZone, dict_options)
    
    return newIntervals, monIncreasingZone, monDecreasingZone  
        

def discretizeAndEvaluataInterval(dfdX, xBounds, i, interval, newIntervals,
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
    
    resolution = dict_options["resolution"]
    
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
    dfdxLow = dfdx(*xBounds)    
          
    if not '-inf' in dfdxLow and not '+inf' in dfdxLow: continuousZone.append(curXBoundsLow)
    else: discontinuousZone.append(curXBoundsLow) 
       
    curXBoundsUp = mpmath.mpi(interval.mid, interval.b)
    xBounds[i] = curXBoundsUp
    dfdxUp = dfdx(*xBounds)
    
    if not '-inf' in dfdxUp and not '+inf' in dfdxUp: continuousZone.append(curXBoundsUp)
    else: discontinuousZone.append(curXBoundsUp) 
    return continuousZone


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
    
    newIvSet = []
    k = 0
    noIv = len(ivSet)
    
    while len(newIvSet) != noIv:
        
        if k != 0: 
            ivSet = newIvSet
            noIv = len(ivSet)
            newIvSet = []
        
        for iv in ivSet:
            for i in range(0, len(ivSet)): 
                if iv == ivSet[i]: 
                    continue
                elif iv in ivSet[i]:
                    newIvSet.append(ivSet[i])
                    ivSet.remove(ivSet[i])
                    ivSet.remove(iv)
                    break
                elif ivSet[i] in iv:
                    newIvSet.append(iv)
                    ivSet.remove(ivSet[i])
                    ivSet.remove(iv)
                    break
                elif mpmath.almosteq(iv.b, ivSet[i].a, relEpsX, absEpsX): 
                    newIvSet.append(mpmath.mpi(iv.a, ivSet[i].b))
                    ivSet.remove(ivSet[i])
                    ivSet.remove(iv)
                    break
                elif mpmath.almosteq(iv.a, ivSet[i].b, relEpsX, absEpsX): 
                    newIvSet.append(mpmath.mpi(ivSet[i].a, iv.b))
                    ivSet.remove(ivSet[i])
                    ivSet.remove(iv)
                    break
                elif ivIntersection(iv, ivSet[i])!=[]:
                    newIvSet.append(mpmath.mpi(min(ivSet[i].a, iv.a), max(ivSet[i].b, iv.b)))
                    ivSet.remove(ivSet[i])
                    ivSet.remove(iv)
                    break
                
        if ivSet != []: 
            for j in range(0, len(ivSet)):
                newIvSet.append(ivSet[j])
        k=1
                
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


def reduceNonMonotoneIntervals(nonMonotoneZone, reducedIntervals, fx, i, xBounds, bi, 
                               dict_options):
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
        
        if curfunValue.a != '+inf' and curfunValue.a !='-inf':
            funValuesLow.append(float(mpmath.mpf(curfunValue.a)))
        elif curfunValue.a == '+inf': 
            funValuesLow.append(1/numpy.finfo(float).eps)
        else: 
            funValuesLow.append(-1/numpy.finfo(float).eps)
        
        if curfunValue.b != '+inf' and curfunValue.b !='-inf':
            funValuesUp.append(float(mpmath.mpf(curfunValue.b)))
        elif curfunValue.b == '+inf': 
            funValuesUp.append(1/numpy.finfo(float).eps)
        else: 
            funValuesUp.append(-1/numpy.finfo(float).eps)
        
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
    
    for i in range(0, len(ivLong)):
        curIV = compareIntervalToIntervalSet(ivLong[i], ivShort)
        if curIV != []: ivReduced.append(curIV)
    
    return ivReduced


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


def checkWidths(X, Y, relEps, absEps):
    """ returns the maximum interval width of a set of intervals X
     
        Args: 
            :X:                     list with set of intervals    

        Return:
            :mpmath.mpi interval:   of maximum width
                        
    """
    
    almostEqual = False * numpy.ones(len(X), dtype = bool)
    for i in range(0,len(X)):
         if(mpmath.almosteq(X[i].delta, Y[i].delta, relEps, absEps)):
             almostEqual[i] = True
             
    return almostEqual.all()