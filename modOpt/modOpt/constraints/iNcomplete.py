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


__all__ = ['doIntervalNesting'
        ]

"""
***************************************************
Algorithm for complete interval nesting procedure
***************************************************
"""

def doIntervalNesting(model, dict_options):
    """ iterates the state variable intervals related to model using the
    Gauss-Seidel Operator combined with an interval nesting strategy
    
    Args:
        model           object of class model
        dict_options    dictionary with solver options
    
    Return:
        model           model object with reduced state variable bounds
        iterNo          number of iteration steps in Newton algorithm
    """
    
    iterNo = 0
    xBounds = model.getXBounds()
    xSymbolic = model.getXSymbolic()
    parameter = model.getParameter()
    blocks = [range(0, len(xSymbolic))] # block decompostion is not used here
    newModel = copy.deepcopy(model)
    dimVar = len(xBounds[0])
    boundsAlmostEqual = False * numpy.ones(dimVar, dtype=bool)


    for l in range(0, dict_options["iterMaxNewton"]): 
        
        iterNo = l + 1
        newXBounds = []
        
        for k in range(0,len(xBounds)): #TODO: Parallelizing
            
            xAlmostEqual = False * numpy.ones(len(xBounds), dtype=bool)
            FsymPerm, xSymbolicPerm, xBoundsPerm = model.getBoundsOfPermutedModel(xBounds[k], 
                                                                                  xSymbolic, 
                                                                                  parameter)
            dict_options["precision"] = getPrecision(xBoundsPerm)

            intervalsPerm, boundsAlmostEqual = reduceXBounds(xBoundsPerm, xSymbolicPerm, FsymPerm,
                                                             blocks, dict_options, boundsAlmostEqual)
            
            if intervalsPerm == []:
                break
            
            for m in range(0, len(intervalsPerm)): #TODO: Parallelizing and in new Function
                
                x = numpy.empty(dimVar, dtype=object)     
                x[newModel.colPerm]  = numpy.array(intervalsPerm[m])           
                newXBounds.append(x)
                
                if checkWidths(xBounds[k], x, dict_options["machEpsRelNewton"], 
                               dict_options["machEpsAbsNewton"]): 
                    xAlmostEqual[k] = True
                    break
    
            if xAlmostEqual.all():
                newModel.setXBounds(xBounds)

                return newModel, iterNo
          
        if newXBounds != []: xBounds = newXBounds
        else: 
            print "NoSolutionError: No valid solution space was found. Please check consistency of initial constraints"
            
    newModel.setXBounds(xBounds)
    
    return newModel, iterNo


def getPrecision(xBounds):
    """
    calculates precision for intervalnesting procedure (when intervals are
    joined to one interval)
    Args:
        xBounds         list with iteration variable bounds in mpmath.mpi formate
    
    Return:
        precision as float value
        
    """
    
    allValuesOfx = []
    for x in xBounds:
        allValuesOfx.append(numpy.abs(float(mpmath.mpf(x.a))))
        allValuesOfx.append(numpy.abs(float(mpmath.mpf(x.b))))
    
    minValue = min(filter(None, allValuesOfx))
    return 5*10**(numpy.floor(numpy.log10(minValue))-2)


def reduceXBounds(xBounds, xSymbolic, f, blocks, dict_options, boundsAlmostEqual):
    """ Solves an equation system blockwise. For block dimensions > 1 each 
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
            boundsAlmostEqual:  boolean list, dimension equals number of variables.
                                if variable bounds can't be further reduced, the
                                variable entry in the list equals true. The variable
                                order is given by the global index.
        Returns:                list with new set of variable bounds
                        
    """    
    
    xNewBounds = copy.deepcopy(xBounds)
    relEpsX = dict_options["relTolX"]
    absEpsX = dict_options["absTolX"]
    
    
    for b in range(0, len(blocks)):
        blockDim = len(blocks[b])
        for n in range(0, blockDim):
            y = [] 
            j = blocks[b][n]
            
            if boundsAlmostEqual[j] == True: 
                xNewBounds[j] = [xNewBounds[j]]
                continue           
            if dict_options["Debug-Modus"]: print j
            
            for m in range(0, blockDim): #TODO: Parallelizing
                i = blocks[b][m]
                if xSymbolic[j] in f[i].free_symbols:
                    
                    if y == []: y = reduceXIntervalByFunction(xBounds, xSymbolic, 
                           f[i], j, dict_options)
                    else: 
                        y = reduceTwoIVSets(y, reduceXIntervalByFunction(xBounds, 
                                                    xSymbolic, f[i], j, dict_options))
                        #if y == []: return [], boundsAlmostEqual
            if y != []: xNewBounds[j] = y
            else: return [], boundsAlmostEqual
            #else: xNewBounds[j] =[xBounds[j]], boundsAlmostEqual

            if len(xNewBounds[j]) == 1: 
                boundsAlmostEqual[j] = checkVariableBound(xNewBounds[j][0], relEpsX, absEpsX)
            
    return list(itertools.product(*xNewBounds)), boundsAlmostEqual


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
    
    fx, dfdX, df2dX, bi, fxInterval, dfdxInterval, df2dxInterval, xBounds = calculateCurrentBounds(fx, 
                    fWithoutX, xSymbolic, i, xBounds, dict_options)
    
    if bi == [] or fxInterval == [] or dfdxInterval == [] or df2dxInterval == []: 
        return [xBounds[i]]
    
    if(df2dxInterval == 0 and fxInterval == dfdxInterval*xBounds[i]): # Linear Case -> solving system directly
        return getReducedIntervalOfLinearFunction(dfdX, xSymbolic, i, xBounds, bi)
             
    else: # Nonlinear Case -> solving system by interval nesting
        return getReducedIntervalOfNonlinearFunction(fx, dfdX, df2dX, dfdxInterval, 
                            df2dxInterval, xSymbolic, i, xBounds, bi, dict_options)


def splitFunctionByVariableDependency(f, x):
    """a sympy function expression f is splitted into a sum from x depended terms
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
    
    #if allArguments == ():
    #    if x in f.free_symbols: return f, 0
    #    else: return 0, f
        
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
        print "Problems occured during function parsing"
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


def calculateCurrentBounds(fx, fWithoutX, xSymbolic, i, xBounds, dict_options):
    """ calculates bounds of function fx, the residual fWithoutX, first and second
    derrivative of function fx with respect to variable xSymbolic[i] (dfX, df2dX).
    It uses the tolerance values from dict_options in the complex case to remove 
    complex intervals by interval nesting.
    
    Args:
        fx:                 symbolic function in sympy logic (contains current 
                            iteration variable X)
        fWithoutX:          symbolic function in sympy logic (does not contain 
                            current iteration variable X)                            
        xSymbolic:          list with symbolic variables in sympy logic
        i:                  index of current iteration variable
        xBounds:            numpy array with variable bounds
        dict_options:       dictionary with entries about stop-tolerances

        
    Returns:
        dfdX:               first derrivative of function fx with respect to 
                            variable xSymbolic[i] in mpmath.mpi-logic
        df2dX:              second derrivative of function fx with respect to 
                            variable xSymbolic[i] in mpmath.mpi-logic
        bi:                 residual interval in mpmath.mpi logic
        fxInterval:         function interval in mpmath.mpi logic
        dfdxInterval:       Interval of first derrivative in mpmath.mpi logic 
        df2dxInterval:      Interval of second derrivative in mpmath.mpi logic 
        xBounds:            real set of variable intervals in mpmath.mpi logic 
    
    """
    #TODO: Parallelizing and Output       
    dfdX = sympy.diff(fx, xSymbolic[i]) 
    df2dX = sympy.diff(dfdX, xSymbolic[i]) 
    
    fxBounds = lambdifyToMpmathIv(xSymbolic, fx)
    dfdxBounds = lambdifyToMpmathIv(xSymbolic, dfdX)
    df2dxBounds = lambdifyToMpmathIv(xSymbolic, df2dX)
    
    try:
        bi = getBoundsOfFunctionExpression(-fWithoutX, xSymbolic, xBounds)

    except:
        fWithoutX = reformulateComplexExpressions(fWithoutX)
        try : bi = getBoundsOfFunctionExpression(-fWithoutX, xSymbolic, xBounds)
        except: return fxBounds, dfdxBounds, df2dxBounds, [], [], [], [], xBounds
      
    try:
       fxInterval = getBoundsOfFunctionExpression(fx, xSymbolic, xBounds)
        
    except:
        newXBounds = reactOnComplexError(fxBounds, xSymbolic, i, xBounds, dict_options)
        if newXBounds == []: 
            return fxBounds, dfdxBounds, df2dxBounds, [], [], [], [], xBounds
        else: 
            xBounds[i] = newXBounds
            fxInterval = getBoundsOfFunctionExpression(fx, xSymbolic, xBounds)
      
    try:
       dfdxInterval = getBoundsOfFunctionExpression(dfdX, xSymbolic, xBounds)
        
    except:
        newXBounds = reactOnComplexError(dfdxBounds, xSymbolic, i, xBounds, dict_options)
        if newXBounds == []: 
            return fxBounds, dfdxBounds, df2dxBounds, [], [], [], xBounds
        else: 
            xBounds[i] = newXBounds
            dfdxInterval = getBoundsOfFunctionExpression(dfdX, xSymbolic, xBounds)            
 
    try:    
        df2dxInterval = getBoundsOfFunctionExpression(df2dX, xSymbolic, xBounds)
        
    except:
        newXBounds = reactOnComplexError(df2dxBounds, xSymbolic, i, xBounds, dict_options)
        if newXBounds == []: 
            return fxBounds, dfdxBounds, df2dxBounds, [], [], [], xBounds
        else: 
            xBounds[i] = newXBounds
            dfdxInterval = getBoundsOfFunctionExpression(dfdX, xSymbolic, xBounds)

    return fxBounds, dfdxBounds, df2dxBounds, bi, fxInterval, dfdxInterval, df2dxInterval, xBounds   
    
            
def getBoundsOfFunctionExpression(f, xSymbolic, xBounds):
    """ evaluates function expression f for variable bounds xBounds
    
    Args:
        f:                  scalar function in mpmath logic
        xSymbolic:          list with symbolic variables in sympy logic
        xBounds:            numpy array with variable bounds

        
    Returns: mpmath.mpi interval of function expression in xBounds
    
    """
    fMpmathIV = lambdifyToMpmathIv(xSymbolic, f)
    return  mpmath.mpi(str(fMpmathIV(*xBounds)))
            

def reformulateComplexExpressions(f):
    """ to avoid complex intervals this function reformulates arguments of log or
    pow functions, so that for example:
        
        log(a) = log((a**2)**0.5)
        b**0.25 = ((b**2)**0.5)**0.25
        
    In this way, only real numbers remain.
    
    Args:
        f                       scalar function in sympy logic
    
    Returns:
        f                       reformulated scalar function in sympy logic
    
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
        f:                  scalar function in mpmath logic
        xSymbolic:          list with symbolic variables in sympy logic
        i:                  index of current iteration variable
        xBounds:            numpy array with variable bounds
        dict_options:       dictionary with variable interval tolerances 
    
    Returns:
        realSection:        real xBounds (so far this is only valid if there is
                            only one complex and one real interval, 
                            TODO: return set of real intervals for x[i])

    """
    absEpsX = dict_options["absTolX"] 
    realSection = []
    curXBounds = copy.deepcopy(xBounds)
    problematicSection = [xBounds[i]]
    
    while problematicSection != []:
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
                    addIntervaltoMonotoneZone(problematicInterval, curProblematicSection, dict_options)
            
            if realInterval !=[]:
                if realSection == []: realSection = realInterval
                else: 
                    realSection = addIntervaltoMonotoneZone(realInterval, realSection, dict_options)
                    
        problematicSection = checkAbsoluteTolerance(removeListInList(curProblematicSection), absEpsX)
    if realSection ==[]: return []
    else: return realSection[0]


def checkAbsoluteTolerance(interval, absEpsX):
    """
    checks if width of intervals is smaller than a given relative tolerance relEpsX
    
    Args:
        interval:           set of intervals in mpmath.mpi-logic
        absEpsX:             absolute x tolerance
    
    Returns:
        reducedInterval:    set of intervals with a higher width than absEps
        
    """
    
    reducedInterval = []
    
    for i in range(0, len(interval)):
        if interval[i].delta > absEpsX:
                reducedInterval.append(interval[i])
                
    return reducedInterval

def testOneBoundOnComplexity(f, xSymbolic, xBounds, i):
    """ returns False if the evlauation of f fails, assumption is that this is
    due to complex variable intervals
    
    Args:
        f:                  scalar function in mpmath logic
        xSymbolic:          list with symbolic variables in sympy logic
        xBounds:            numpy array with variable bounds
        i:                  index of current iteration variable
        
    Returns:
        True:               Complex number
        False:              Real number
    
    """
    
    try:
        mpmath.mpi(str(f(*xBounds)))
        return False
    except:
        return True
    

def complexOperator(complexA, complexMid, complexB, xInterval):
    """ splits xInterval into complex/real-xInterval and real-xInterval 
    
    Args:
        complexA:           boolean: True = complex lower bound of xInterval
                            False = real lower bound of xInterval
        complexB:           boolean: True = complex upper bound of xInterval
                            False = real upper bound of xInterval
        complexC:           boolean: True = complex midpoint of xInterval
                            False = real midpoint of xInterval
        xInterval:          interval of variable x in mpmath.mpi logic
        
    Returns:
        list1:              list with complex-real-xInterval
        list2:              list with real xInterval
    
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


def getReducedIntervalOfLinearFunction(a, xSymbolic, i, xBounds, bi):
    """returns reduced interval of variable X if f is linear in X. The equation
    is solved directly by the use of the gaussSeidelOperator.
    
    Args: 
        a:                  first symbolic derivative of function f with respect to x
        xSymbolic:          list with symbolic variables in sympy logic
        i:                  integer with current iteration variable index
        xBounds:            numpy array with set of variable bounds
        bi:                 current function residual bounds
    
    Returns:                reduced x-Interval(s)
    
    """
    
    aInterval = mpmath.mpi(str(a(*xBounds)))
    
    checkAndRemoveComplexPart(aInterval)
        
    if bool(0 in bi - aInterval * xBounds[i]) == False: return [] # if this is the case, there is no solution in xBoundsi

    if bool(0 in bi) and bool(0 in aInterval):  # if this is the case, bi/aInterval would return [-inf, +inf]. Hence the approximation of x is already smaller
                return [xBounds[i]]
    else: 
        return gaussSeidelOperator(aInterval, bi, xBounds[i]) # bi/aInterval  


def checkAndRemoveComplexPart(interval):
    """ Creates a warning if a complex interval occurs and keeps only the real
    part.
    
    """
    if interval.imag != 0:
        print "Warning: A complex interval: ", interval.imag," occured.\n",
        "For further calculations only the real part: ", interval.real, " is used."
        interval = interval.real
        

def gaussSeidelOperator(a, b, x):
    """ Computation of the Gauss-Seidel-Operator [1] to get interval for x
    for given intervals for a and b from the 1-dimensional linear system:
     
                                    a * x = b
        Args: 
            a:     interval of mpi format from mpmath library 
            b:     interval of mpi format from mpmath library
            x:     interval of mpi format from mpmath library 
                    (initially guessed interval of x)
            
        Returns:
            interval:   interval(s) of mpi format from mpmath library where
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
    """Calculates the result of the divion of two intervals i1, i2: i1 / i2
    
        Args: 
            i1:     interval of mpi format from mpmath library
            i2:     interval of mpi format from mpmath library 
            
        Returns:
            mpmath.mpi(a,b):    resulting interval of division [a,b], 
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
    """Returns intersection of two intervals i1 and i2
    
        Args: 
            i1:     interval of mpi format from mpmath library
            i2:     interval of mpi format from mpmath library 
            
        Returns:
            mpmath.mpi(a,b):    interval of intersection [a,b], 
                                if empty [] is returned
                                
        """
        
    # Different cases:    
    if i1.a <= i2.a and i1.b <= i2.b and i2.a <= i1.b: return mpmath.mpi(i2.a, i1.b)
    if i1.a <= i2.a and i1.b >= i2.b: return i2
    if i1.a >= i2.a and i1.b <= i2.b: return i1
    if i1.a >= i2.a and i1.b >= i2.b and i1.a <= i2.b: return mpmath.mpi(i1.a, i2.b)
    
    else: return []


def getReducedIntervalOfNonlinearFunction(fx, dfdX, df2dX, dfdXInterval, df2dxInterval, xSymbolic, i, xBounds, bi, dict_options):
    """ Checks function for monotone sections in x and reduces them one after the other.
    
    Args: 
        fx:                 symbolic x-depending part of function f in mpmath.mpi logic
        dfdX:               first symbolic derivative of function f with respect to x
                            in mpmath.mpi logic
        dfdX:               second symbolic derivative of function f with respect to x
                            in mpmath.mpi logic
        dfdXInterval:       first derivative of function f with respect to x at xBounds
        xSymbolic:          list with symbolic variables in sympy logic
        i:                  integer with current iteration variable index
        xBounds:            numpy array with set of variable bounds
        bi:                 current function residual bounds
        dict_options:       for function and variable interval tolerances in the used
                            algorithms

    Returns:                reduced x-Interval(s) and list of monotone x-intervals
    
    """
    orgXiBounds = copy.deepcopy(xBounds[i])

    if (dfdXInterval == [] or df2dxInterval == []): return [orgXiBounds]
    if (dfdXInterval >= 0):
        
        if bool(df2dxInterval >= 0) == False and bool(df2dxInterval <= 0) == False:
            
            incrZoneIndf2dX, decrZoneIndf2dX = getContinuousFunctionSections(df2dX, xSymbolic, i, xBounds, dict_options)
            increasingZone = removeListInList([incrZoneIndf2dX, decrZoneIndf2dX])
            if increasingZone == []: return [orgXiBounds]
            
            reducedIntervals = []
            reducedIntervals = reduceMonotoneIntervals(increasingZone, reducedIntervals, fx, xSymbolic, 
                                    xBounds, i, bi, dict_options, increasing = True)    
            return reducedIntervals
        
        return [monotoneIncresingIntervalNesting(fx, xSymbolic, xBounds, i, bi, dict_options)]

    
    if (dfdXInterval <= 0):
        
        if bool(df2dxInterval >= 0) == False and bool(df2dxInterval <= 0) == False:
            
            incrZoneIndf2dX, decrZoneIndf2dX = getContinuousFunctionSections(df2dX, xSymbolic, i, xBounds, dict_options)
            decreasingZone = removeListInList([incrZoneIndf2dX, decrZoneIndf2dX])
            if decreasingZone == []: return [orgXiBounds]
            
            reducedIntervals = []
            reducedIntervals = reduceMonotoneIntervals(decreasingZone, reducedIntervals, fx, xSymbolic, 
                                    xBounds, i, bi, dict_options, increasing = False)    
            return reducedIntervals
        
        return [monotoneDecreasingIntervalNesting(fx, xSymbolic, xBounds, i, bi, dict_options)]
        
    else:
        reducedIntervals = []
        increasingZone, decreasingZone, nonMonotoneZone = getMonotoneFunctionSections(dfdX, xSymbolic, i, xBounds, 
                                                                                      dict_options)
        if increasingZone == [] and decreasingZone == [] and nonMonotoneZone == []: return [orgXiBounds]
        
        reducedIntervals = reduceMonotoneIntervals(increasingZone, reducedIntervals, fx, xSymbolic, 
                                 xBounds, i, bi, dict_options, increasing = True)
        
        reducedIntervals = reduceMonotoneIntervals(decreasingZone, reducedIntervals, fx, xSymbolic, 
                                 xBounds, i, bi, dict_options, increasing = False)
        if nonMonotoneZone != []:
            reducedIntervals = reduceNonMonotoneIntervals(nonMonotoneZone, reducedIntervals, 
                                                          fx, xSymbolic, i, xBounds, bi, 
                                                          dict_options)
        return reducedIntervals


def getContinuousFunctionSections(df2dx, xSymbolic, i, xBounds, dict_options):
    """seperate variable interval into variable interval sets where a function
    with derivative dfdx is monontoneous
    
    Args:
        df2dx:               scalar second derivate of function in mpmath.mpi logic
        xSymbolic:           symbolic variables in derivative function
        i:                   index of differential variable
        xBounds:             numpy array with variable bounds
        dict_options:        dictionary with variable and function interval tolerances
    
    Return:
        monIncreasingZone:   monotone increasing intervals 
        monDecreasingZone:   monotone decreasing intervals 
    
    """
    
    relEpsX = dict_options["relTolX"]
    relEpsF = dict_options["relTolF"]
    absEpsF = dict_options["absTolF"]

    lmax = dict_options["NoOfNonChangingValues"] 
        
    monIncreasingZone = []
    monDecreasingZone = []
    
    interval = [xBounds[i]] 
    l = 0
    while interval != [] and l < lmax:
        
        curIntervals = []
               
        for k in range(0, len(interval)):
 
            newIntervals, newMonIncreasingZone, newMonDecreasingZone, l = testIntervalOnMonotony(df2dx, interval[k], 
                                                                                                 xBounds, i, l,
                                                                                                 relEpsF, absEpsF)
                        
            monIncreasingZone = addIntervaltoMonotoneZone(newMonIncreasingZone, monIncreasingZone, dict_options)            
            monDecreasingZone = addIntervaltoMonotoneZone(newMonDecreasingZone, monDecreasingZone, dict_options)
            addIntervalToNonMonotoneZone(newIntervals, curIntervals)    
               
        interval = checkTolerance(removeListInList(curIntervals), relEpsX)
    
    return monIncreasingZone, monDecreasingZone


def removeListInList(listInList):
    """Changes list with the shape: [[a], [b,c], [d], ...] to [a, b, c, d, ...]
    
    """
    newList = []
    for i in range(0, len(listInList)):
        for j in range(0, len(listInList[i])):
            newList.append(listInList[i][j])
    return newList


def reduceMonotoneIntervals(monotoneZone, reducedIntervals, fx, xSymbolic, 
                                      xBounds, i, bi, dict_options, increasing):
    """reduces interval sets of one variable by interval nesting
    
    Args:
        monotoneZone        list with monotone increasing or decreasing set of intervals
        reducedIntervals    list with already reduced set of intervals
        fx:                 symbolic x-depending part of function f
        xSymbolic:          list with symbolic variables in sympy logic
        xBounds:            numpy array with set of variable bounds
        i:                  integer with current iteration variable index        
        bi:                 current function residual bounds
        dict_options:       dictionary with function and variable interval tolerances
        increasing:         boolean, True for increasing function intervals, 
                            False for decreasing intervals

    """
    
    relEpsX = dict_options["relTolX"]
    precision = dict_options["precision"]

    for j in range(0, len(monotoneZone)): #TODO: Parallelizing
        xBounds[i] = monotoneZone[j] 
        
        if increasing: curReducedInterval = monotoneIncresingIntervalNesting(fx, xSymbolic, xBounds, i, bi, dict_options)
        else: curReducedInterval = monotoneDecreasingIntervalNesting(fx, xSymbolic, xBounds, i, bi, dict_options)
        
        if curReducedInterval !=[] and reducedIntervals != []:
            reducedIntervals.append(curReducedInterval)
            reducedIntervals = joinIntervalSet(reducedIntervals, relEpsX, precision)
        elif curReducedInterval !=[]: reducedIntervals.append(curReducedInterval)
        
    return reducedIntervals


def monotoneIncresingIntervalNesting(fx, xSymbolic, xBounds, i, bi, dict_options):
    """ reduces variable intervals of monotone increasing functions fx
    by by interval nesting
     
        Args: 
            fx:                 symbolic xi-depending part of function fi
            xSymbolic:          list with symbolic variables in sympy logic
            xBounds:            numpy array with set of variable bounds
            i:                  integer with current iteration variable index
            bi:                 current function residual bounds
            dict_options:       dictionary with function and variable interval tolerances

        Returns:                list with one entry that is the reduced interval 
                                of the variable with the index i
                        
    """  
    relEpsX = dict_options["relTolX"]
    absEpsX = dict_options["absTolX"]
    relEpsF = dict_options["relTolF"]
    absEpsF = dict_options["absTolF"]
    
    
    fxInterval = fx(*xBounds)
    curInterval = xBounds[i]
    
    if ivIntersection(fxInterval, bi)==[]: return []
    
    # first check if xBounds can be further reduced:
    if fxInterval in bi: return curInterval
        
    # Otherwise, iterate each bound of bi:
    fIntervalxLow, fIntervalxUp = getFIntervalsFromXBounds(fx, curInterval, xBounds, i)
    #if fIntervalxLow.b > bi.b:
        
    if fIntervalxLow.b < bi.a:   
         while not mpmath.almosteq(mpmath.mpf(fxInterval.a), mpmath.mpf(bi.a), 
                                   relEpsF, absEpsF) and not mpmath.almosteq(curInterval.a, 
                                                 curInterval.b, relEpsX, absEpsX):
                         curInterval, fxInterval = iteratefBound(fx, curInterval, xBounds, i, bi, 
                                                increasing = True, 
                                                lowerXBound = True)
    
        
    lowerBound = curInterval.a
    curInterval  = xBounds[i]    
    
    #if fIntervalxUp.a < bi.a: return []
    if fIntervalxUp.a > bi.b:
        
        while not mpmath.almosteq(mpmath.mpf(fxInterval.b), mpmath.mpf(bi.b), 
                                  relEpsF, absEpsF) and not mpmath.almosteq(curInterval.a, 
                                                curInterval.b, relEpsX, absEpsX):
            
            curInterval, fxInterval = iteratefBound(fx, curInterval, xBounds, i, bi, 
                                                increasing = True, 
                                                lowerXBound = False)
                
    upperBound = curInterval.b                 
    return mpmath.mpi(lowerBound, upperBound)


def monotoneDecreasingIntervalNesting(fx, xSymbolic, xBounds, i, bi, dict_options):
    """ reduces variable intervals of monotone decreasing functions fx
    by by interval nesting
     
        Args: 
            fx:                 symbolic xi-depending part of function fi
            xSymbolic:          list with symbolic variables in sympy logic
            xBounds:            numpy array with set of variable bounds
            i:                  integer with current iteration variable index
            bi:                 current function residual bounds
            dict_options:       dictionary with function and variable interval tolerances
            
        Returns:                list with one entry that is the reduced interval 
                                of the variable with the index i
                        
    """     
    relEpsX = dict_options["relTolX"]
    absEpsX = dict_options["absTolX"]
    relEpsF = dict_options["relTolF"]
    absEpsF = dict_options["absTolF"]
    
    fxInterval = fx(*xBounds)
    curInterval = xBounds[i]
    
    if ivIntersection(fxInterval, bi)==[]: return []
    
    # first check if xBounds can be further reduced:
    if fxInterval in bi: return curInterval
    
    # Otherwise, iterate each bound of bi:
    fIntervalxLow, fIntervalxUp = getFIntervalsFromXBounds(fx, curInterval, xBounds, i)
    #if fIntervalxLow.a < bi.a: return []
    
    if fIntervalxLow.a > bi.b:   
         while not mpmath.almosteq(mpmath.mpf(fxInterval.b), mpmath.mpf(bi.b), 
                                   relEpsF, absEpsF) and not mpmath.almosteq(curInterval.a, 
                                                 curInterval.b, relEpsX, absEpsX):
                         curInterval, fxInterval = iteratefBound(fx, curInterval, xBounds, i, bi, 
                                                increasing = False, 
                                                lowerXBound = True)
    
        
    
    lowerBound = curInterval.a  
    curInterval  = xBounds[i]        
    #if fIntervalxUp.b > bi.b: return []
    if fIntervalxUp.b < bi.a:
        
        while not mpmath.almosteq(mpmath.mpf(fxInterval.a), mpmath.mpf(bi.a), 
                                  relEpsF, absEpsF) and not mpmath.almosteq(curInterval.a, 
                                                curInterval.b, relEpsX, absEpsX):
            
            curInterval, fxInterval = iteratefBound(fx, curInterval, xBounds, i, bi, 
                                                increasing = False, 
                                                lowerXBound = False)
                
    upperBound = curInterval.b               
    return mpmath.mpi(lowerBound, upperBound)


def iteratefBound(fx, curInterval, xBounds, i, bi, increasing, lowerXBound):
    """ Returns the half of curInterval that contains the lower or upper
    bound of bi (biLimit)
    
    Args:
        fx:                 symbolic xi-depending part of function fi
        curInterval:        X-Interval that contains the solution to f(x) = biLimit
        xBounds:            numpy array with set of variable bounds
        i:                  integer with current iteration variable index
        bi:                 current function residual 
        increasing:         boolean: True = function is monotone increasing, 
                            False = function is monotone decreasing
        lowerXBound:        boolean: True = lower Bound is iterated
                            False = upper bound is iterated
        
    Returns:    
        reduced curInterval (by half) and bounds of in curInterval
        
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
        return curUpperXInterval, fxInterval


def getFIntervalsFromXBounds(fx, curInterval, xBounds, i):
    """ returns function interval for lower variable bound and upper variable 
    bound of variable interval curInterval.
    
    Args:
        fx:             symbolic function in mpmath.mpi logic
        curInterval:    current variable interval in mpmath logic
        xBounds:        set of variable intervals in mpmath logic
        i:              index of currently iterated variable interval
                         
    Returns: function interval for lower variable bound and upper variable bound
    
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
        fIntervalxLow:   function interval of lower variable bound in mpmath 
                         logic
        fIntervalxUp:    function interval of upper variable bound in mpmath 
                         logic
        increasing:      boolean: True = monotone increasing, False = monotone
                         decreasing function
        lowerXBound:     boolean: True = lower variable bound, False = upper 
                         variable bound
                         
    Returns: relevant function interval for iterating bi bound in mpmath logic
    
    """
    
    if increasing and lowerXBound: return mpmath.mpi(fIntervalxLow.b, fIntervalxUp.b)
    if increasing and not lowerXBound: return mpmath.mpi(fIntervalxLow.a, fIntervalxUp.a)
    
    if not increasing and lowerXBound: return mpmath.mpi(fIntervalxUp.a, fIntervalxLow.a)
    if not increasing and not lowerXBound: return mpmath.mpi(fIntervalxUp.b, fIntervalxLow.b)
 
    
def residualBoundOperator(bi, increasing, lowerXBound):
    """ returns the residual bound that is iterated in the certain case
    
    Args:
        bi:              function residual interval in mpmath logic
        increasing:      boolean: True = monotone increasing, False = monotone
                         decreasing function
        lowerXBound:     boolean: True = lower variable bound, False = upper 
                         variable bound
                         
    Returns: lower or upper bound of function residual interval in mpmath logic
    
    """
    
    if increasing and lowerXBound: return bi.a
    if increasing and not lowerXBound: return bi.b
    if not increasing and lowerXBound: return bi.b
    if not increasing and not lowerXBound: return bi.a 


def getMonotoneFunctionSections(dfdx, xSymbolic, i, xBounds, dict_options):
    """seperate variable interval into variable interval sets where a function
    with derivative dfdx is monontoneous
    
    Args:
        dfdx:                scalar function in mpmath.mpi logic
        xSymbolic :          symbolic variables in derivative function
        i:                   index of differential variable
        xBounds:             numpy array with variable bounds
        dict_options:        dictionary with function and variable interval 
                             tolerances
    
    Return:
        monIncreasingZone:   monotone increasing intervals 
        monDecreasingZone:   monotone decreasing intervals 
        interval:            non monotone zone if  function interval can not be
                             reduced to monotone increasing or decreasing section
    
    """
    
    relEpsX = dict_options["relTolX"]
    absEpsX = dict_options["absTolX"]
    relEpsF = dict_options["relTolF"]
    absEpsF = dict_options["absTolF"]
    lmax = dict_options["NoOfNonChangingValues"]     
    
    relEpsdFdX = relEpsF/relEpsX
    absEpsdFdX = absEpsF/absEpsX
        
    monIncreasingZone = []
    monDecreasingZone = []
    l=0
    interval = [xBounds[i]] 
    
    while interval != [] and l < lmax:
        
        curIntervals = []
               
        for k in range(0, len(interval)):
            newIntervals, newMonIncreasingZone, newMonDecreasingZone, l = testIntervalOnMonotony(dfdx, 
                                                    interval[k], xBounds, i, l, relEpsdFdX, absEpsdFdX)
            
            if newIntervals != [] and checkTolerance(newIntervals, relEpsX)==[]:
                newIntervals, monIncreasingZone, monDecreasingZone = discretizeAndEvaluateIntervals(dfdx, 
                                                    xBounds, i, newIntervals, monIncreasingZone, 
                                                    monDecreasingZone, dict_options)
            
            monIncreasingZone = addIntervaltoMonotoneZone(newMonIncreasingZone, 
                                                          monIncreasingZone, dict_options)            
            monDecreasingZone = addIntervaltoMonotoneZone(newMonDecreasingZone, 
                                                          monDecreasingZone, dict_options)
       
            addIntervalToNonMonotoneZone(newIntervals, curIntervals)
            
        interval = checkTolerance(removeListInList(curIntervals), relEpsX)
    
    return monIncreasingZone, monDecreasingZone, interval


def discretizeAndEvaluateIntervals(dfdX, xBounds, i, intervals, monIncreasingZone, 
                                   monDecreasingZone, dict_options):
    """ if a certain interval width is undershot it is useful to discretize it and
    check all segments on their monotony.
    
    Args:
        dfdX:                scalar function in mpmath.mpi logic
        xBounds:             numpy array with variable bounds
        i:                   index of differential variable
        intervals:           new unchecked variable intervals
        monIncreasingZone:   already identified monotone increasing intervals
        monDecreasingZone:   already identified monotone decreasing intervals 
        dict_options:        dictionary with function and variable interval 
                             tolerances and discretization resolution
    
    Returns:
        newIntervals:        non monotone intervals if  function interval can not be
                             reduced to monotone increasing or decreasing section
        monIncreasingZone:   monotone increasing intervals 
        monDecreasingZone:   monotone decreasing intervals 

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
        dfdX:                scalar function in mpmath.mpi logic
        xBounds:             numpy array with variable bounds
        i:                   index of differential variable
        interval:            unchecked variable interval
        newIntervals:        already identified non monotone intervals 
        monIncreasingZone:   already identified monotone increasing intervals
        monDecreasingZone:   already identified monotone decreasing intervals 
        dict_options:        dictionary with function and variable interval 
                             tolerances and discretization resolution
    
    Returns:
        newIntervals:        non monotone intervals if  function interval can not be
                             reduced to monotone increasing or decreasing section
        monIncreasingZone:   monotone increasing intervals 
        monDecreasingZone:   monotone decreasing intervals 

    """
    
    resolution = dict_options["resolution"]
    
    intervalBounds = convertIntervalBoundsToFloatValues(interval)
    intervalPoints = numpy.linspace(intervalBounds[0], intervalBounds[1], resolution)
    
    for j in range(0, (len(intervalPoints)-1)):
        
        xBounds[i] = mpmath.mpi(min(intervalPoints[j], intervalPoints[j+1]), 
               max(intervalPoints[j], intervalPoints[j+1]))
        
        if dfdX(*xBounds) > 0:
            monIncreasingZone = addIntervaltoMonotoneZone([xBounds[i]], monIncreasingZone, dict_options)
            
        elif dfdX(*xBounds) < 0:
            monDecreasingZone = addIntervaltoMonotoneZone([xBounds[i]], monDecreasingZone, dict_options)
        else: 
            newIntervals = addIntervaltoMonotoneZone([xBounds[i]], newIntervals, dict_options)
        
    return newIntervals, monIncreasingZone, monDecreasingZone
  
          
def convertIntervalBoundsToFloatValues(interval):
    """ converts mpmath.mpi intervals to list with bounds as float values
    
    Args:
        interval                interval in math.mpi logic
        
    Returns:                    list with bounds as float values
    
    """
    
    return [float(mpmath.mpf(interval.a)), float(mpmath.mpf(interval.b))]


def testIntervalOnMonotony(dfdx, interval, xBounds, i, l, relEpsdFdX, absEpsdFdX):
    """splits interval into 2 halfs and orders concering their monotony 
    behaviour of f (first derivative dfdx):
        1. monotone increasing function in interval of x 
        2. monotone decreasing function in interval of x
        3. non monotone function in interval of x
        
    Args:
        dfdx:           scalar derivative of f with respect to x in mpmath.mpi-logic
        interval:       x interval in mpmath.mpi-logic
        xBounds:        numpy array with variable bounds in mpmath.mpi-logic
        i:              variable index
        l:              counts [-inf,inf] dfdxIntervals to filter out functions
                        that have an x indepenendent derrivate constant interval of 
                        [-inf, +inf] for example: f=x/y-1 and y in [-1,1]
        relEpsdFdX:     relative tolerance of first derivative
        absEpsdFdX:     absolute tolerance of first derivative
    Reutrn:
        3 lists nonMonotoneZone, monotoneIncreasingZone, monotoneDecreasingZone
        and one updated count of [-inf,inf] dfdxIntervals as integer
        
    """

    nonMonotoneZone = []    
    monotoneIncreasingZone = []
    monotoneDecreasingZone = []
       
    xBoundsLow = copy.deepcopy(xBounds)
    xBoundsUp = copy.deepcopy(xBounds)
    
    curXBoundsLow = mpmath.mpi(interval.a, interval.mid)
    curXBoundsUp = mpmath.mpi(interval.mid, interval.b)
    
    
    xBoundsLow[i] = curXBoundsLow
    xBoundsUp[i] = curXBoundsUp

    dfdxLow = dfdx(*xBoundsLow)    
    dfdxUp = dfdx(*xBoundsUp)
  
    if bool(dfdxLow >= 0): monotoneIncreasingZone.append(curXBoundsLow)
    elif bool(dfdxLow <= 0): monotoneDecreasingZone.append(curXBoundsLow)
    else: nonMonotoneZone.append(curXBoundsLow)

    if bool(dfdxUp >= 0): monotoneIncreasingZone.append(curXBoundsUp)
    elif bool(dfdxUp <= 0): monotoneDecreasingZone.append(curXBoundsUp)
    else: nonMonotoneZone.append(curXBoundsUp)
    
    if dfdxUp == mpmath.mpi('-inf', '+inf'): l = l+1
    if dfdxLow == mpmath.mpi('-inf', '+inf'): l = l+1
    if mpmath.almosteq(dfdxUp.a, dfdxLow.a, relEpsdFdX, absEpsdFdX): l = l+1
    if mpmath.almosteq(dfdxUp.b, dfdxLow.b, relEpsdFdX, absEpsdFdX): l = l+1
    return nonMonotoneZone, monotoneIncreasingZone, monotoneDecreasingZone, l


def addIntervaltoMonotoneZone(newInterval, monotoneZone, dict_options):
    """ adds one or two monotone intervals newInterval to list of other monotone 
    intervals. Function is related to function testIntervalOnMonotony, since if  the
    lower and upper part of an interval are identified as monoton towards the same direction
    they are joined and both parts are added to monotoneZone. If monotoneZone contains
    an interval that has the shares a bound with newInterval they are joined. Intersections
    should not occur.
    
    Args:
        newInterval:         list with interval(s) in mpmath.mpi logic
        monotoneZone:        list with intervals from mpmath.mpi logic
        dict_options:        dictionary with variable interval specified tolerances
                            absolute = absTolX, relative = relTolX
    
    Return:
        monotoneZone:        monotoneZone including newInterval
        
    """
    
    absEpsX = dict_options["absTolX"]
    relEpsX = dict_options["relTolX"] 
        
    if newInterval != []:
        
        if len(newInterval) > 1:
            newInterval = [mpmath.mpi(newInterval[0].a, newInterval[1].b)]
            
        if monotoneZone == [] :
            monotoneZone.append(newInterval)
            return removeListInList(monotoneZone)

        else:
            for i in range(0, len(newInterval)):
                monotoneZone.append(newInterval[i])
            
            return joinIntervalSet(monotoneZone, relEpsX, absEpsX)
    
    return monotoneZone


def joinIntervalSet(ivSet, relEpsX, absEpsX):
    """joins all intervals in an interval set ivSet that intersec or share the
    same bound
    
    Args:
        ivSet:              set of intervals in mpmath.mpi logic
        relEps:             relative tolerance of variable intervals
        absEps:             absolute tolerance of variable intervals
    
    Returns:
        newIvSet:           new set of joint intervals
        
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
    """add copy of newInterval(s) to already stored ones in list curIntervals
    
    """
    if newIntervals != []: curIntervals.append(copy.deepcopy(newIntervals))  
    

def checkTolerance(interval, relEpsX):
    """
    checks if width of intervals is smaller than a given relative tolerance relEpsX
    
    Args:
        interval:           set of intervals in mpmath.mpi-logic
        relEpsX:             relative x tolerance
    
    Returns:
        reducedInterval:    set of intervals with a higher width than absEps
        
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


def reduceNonMonotoneIntervals(nonMonotoneZone, reducedIntervals, fx, xSymbolic, i, xBounds, bi, 
                               dict_options):
    """ Reduces non monotone intervals by simply calculating function values for
    interval segments of a discretized variable interval and keeps those segments
    that intersect with bi. The discretization resolution is defined in dict_options.
    
    Args: 
        nonMonotoneZone:    list with non monotone variable intervals
        reducedIntervals:   lits with reduced non monotone variable intervals
        fx:                 variable-dependent function in mpmath.mpi logic 
        xSymbolic:          list with symbolic variables in sympy logic
        i:                  integer with current iteration variable index
        xBounds:            numpy array with set of variable bounds
        bi:                 current function residual bounds
        dict_options:       for function and variable interval tolerances in the used
                            algorithms and resolution of the discretization

    Returns:                reduced x-Interval(s) and list of monotone x-intervals
        
 """   
    
    relEpsX = dict_options["relTolX"]
    precision = dict_options["precision"]
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
        x:          numpy list with segments for iteration variable xi
        f:          x-dependent function in mpmath.mpi logic
        xBounds:    numpy array with variable bounds in mpmath.mpi.logic
        i:          current iteration variable index
    
    Returns:        list with lower function value bounds within x and upper 
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
    """reduces two interval sets to one and keeps only the resulting interval
    when elements of both intervals intersect. Each element of the longer interval 
    set is compared to the list of the shorter interval set. 
    
    Args:
        ivSet1:          list 1 with intervals in mpmath.mpi logic
        ivSet2:          list 2 with intervals in mpmath.mpi logic
        
    Return: list with reduced set of intervals
    
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
    """checks if there is an intersection betweeen interval iv and a list of 
    intervals ivSet. If there is one the intersection is returned. 
    
    Args:
        iv:         interval in mpmath.mpi logic
        ivSet:      list with intervals in mpmath.mpi logic
    
    Returns:        intersection or empty list if there is no intersection
    
    """
    
    for i in range(0, len(ivSet)):
        newIV = ivIntersection(iv, ivSet[i])
        if newIV != []: return newIV
    return []


def checkWidths(X, Y, relEps, absEps):
    """ returns the maximum interval width of a set of intervals X
     
        Args: 
            X:            list with set of intervals    

        Returns:
            mpmath.mpi    interval of maximum width
                        
    """
    almostEqual = False * numpy.ones(len(X), dtype = bool)
    for i in range(0,len(X)):
         if(mpmath.almosteq(X[i].delta, Y[i].delta, relEps, absEps)):
             almostEqual[i] = True
             
    return almostEqual.all()
