"""
***************************************************
Import packages
***************************************************
"""
import copy
import numpy
import mpmath
import iNes_procedure
import itertools
from multiprocessing import Manager, Process #cpu_count
from FailedSystem import FailedSystem

__all__ = ['reduceMultipleXBounds', 'reduceXBounds']

"""
***************************************************
Algorithm for parallelization in iNes procedure
***************************************************
"""

def reduceMultipleXBounds(xBounds, model, blocks, dimVar,
                                        xSymbolic, parameter, dict_options):
    """ reduction of multiple solution interval sets
    
    Args:
        :xBounds:               list with list of interval sets in mpmath.mpi logic
        :model:                 object of type model
        :blocks:                list with block indices of equation system
        :dimVar:                integer that equals iteration variable dimension
        :xSymbolic:             list with symbolic iteration variables in sympy logic
        :parameter:             list with parameter values of equation system
        :dict_options:          dictionary with user specified algorithm settings
    
    Return:
        :output:                dictionary with newXBounds and a list of booleans
                                that marks all non reducable variable bounds with True.
                                If solver terminates because of a NoSolution case the
                                critical equation is also stored in results for the error
                                analysis.
            
    """
    
    output = {}
    CPU_count = dict_options["CPU count"]
    jobs = []
    manager = Manager()
    results = manager.dict()
    done = numpy.zeros(len(xBounds))
    started = numpy.zeros(len(xBounds))
    actNum = 0
    #output["xAlmostEqual"] = False * numpy.ones(len(xBounds), dtype=bool)   

    for k in range(0,len(xBounds)):  
        p = Process(target=reduceMultipleXBounds_Worker, args=(xBounds, k,
                                                               model,
                                                               blocks, dimVar,
                                                               xSymbolic, 
                                                               parameter,
                                                               dict_options,
                                                               results))
        jobs.append(p)
        
    startAndDeleteJobs(actNum, jobs, started, done, len(xBounds), CPU_count)
          
    output["newXBounds"], output["xAlmostEqual"] = getReducedXBoundsResults(results, len(xBounds))
     
    
    if output["newXBounds"] == []:
        output["noSolution"] = results["noSolution"]
        
    return output


def reduceMultipleXBounds_Worker(xBounds, k, model, blocks, dimVar, xSymbolic, parameter, dict_options, results):
    """ contains work that can be done in parallel during the reduction of multiple 
    solution interval sets stored in xBounds. The package multiprocessing is used 
    for parallelization.
    
    Args:
        :xBounds:               list with list of interval sets in mpmath.mpi logic
        :k:                     index of current job
        :model:                 object of type model
        :blocks:                list with block indices of equation system
        :dimVar:                integer that equals iteration variable dimension
        :xSymbolic:             list with symbolic iteration variables in sympy logic
        :parameter:             list with parameter values of equation system
        :dict_options:          dictionary with user specified algorithm settings
        :results:               dictionary from multiprocessing where results are stored
                                after a job is done    
    Return:                     True if method finishes ordinary
                             
    """
    
    newXBounds = []
    FsymPerm, xSymbolicPerm, xBoundsPerm = model.getBoundsOfPermutedModel(xBounds[k], 
                                                                              xSymbolic, 
                                                                              parameter)
    dict_options["precision"] = iNes_procedure.getPrecision(xBoundsPerm)

    if dict_options["Parallel Variables"]:
        output = reduceXBounds(xBoundsPerm, xSymbolicPerm, FsymPerm,
                                                         blocks, dict_options)
    
    else:
        output = iNes_procedure.reduceXBounds(xBoundsPerm, xSymbolicPerm, FsymPerm,
                                                         blocks, dict_options)   
    intervalsPerm = output["intervalsPerm"]
        
    if output.has_key("noSolution"):
        results["noSolution"] = output["noSolution"]
    
    for m in range(0, len(intervalsPerm)):
        
        x = numpy.empty(dimVar, dtype=object)     
        x[model.colPerm]  = numpy.array(intervalsPerm[m])           
        newXBounds.append(convertMpiToList(x))
        
#        if iNes_procedure.checkWidths(xBounds[k], x, dict_options["relTolX"], 
#                       dict_options["absTolX"]): 

#            results['%d' %k] = ([convertMpiToList(x)], True, boundsAlmostEqual)
#            return True
    
    results['%d' %k] = (newXBounds, output["xAlmostEqual"])
    return True


def startAndDeleteJobs(actNum, jobs, started, done, jobNo, CPU_count):
    """ starts jobs using multiprocessing and deletes finished ones
    
    Args:
        :actNum:        current number of active jobs
        :jobs:          list with all jobs from process
        :started:       jobNo dimensional numpy array. The entries are binaries 
                        (1 = job started, 0 = job not started)
        :done:          jobNo dimensional numpy array. The entries are binaries 
                        (1 = job done, 0 = job not done)
        :jobNo:         integer with total number of jobs
        :CPU_count:     number of cores
                           
    """  
    
    while numpy.sum(done) < jobNo:
        for jobId in range(0, jobNo):
            actNum = addNotStartedJobs(actNum, jobs, started, done, jobId, CPU_count)
            actNum = deleteFinishedJobs(actNum, jobs, started, done, jobId)


def getReducedXBoundsResults(results, noOfxBounds):
    """ extracts quantities from multiprocessing results
    
    Args:
        :results:           dictionary from multiprocessing, where results are stored
                            after a job is done
        :noOfxBounds:       number of solution interval sets in xBounds


    Return:
        :newXBounds:        list with reduced solution interval sets
        :xAlmostEqual:      numpy array with boolean entries for each interval vector
                            is true for interval vectors that have not changed in the
                            last reduction step anymore
                                 
    """
    xAlmostEqual = False * numpy.ones(noOfxBounds, dtype=bool) 
    newXBounds = []
    #boundsAlmostEqual = False
    
    for k in range(0, noOfxBounds):
        
        if results['%d' %k][0] != []: 
            curNewXBounds = results['%d' %k][0] # [[[a1], [b1], [c1]], [[a2], [b2], [c2]]]
            for curNewXBound in curNewXBounds:
                newXBounds.append(convertListToMpi(curNewXBound))
                
        xAlmostEqual[k] = (results['%d' %k][1])   
    
        #boundsAlmostEqual = results['%d' %k][2]
        
    return newXBounds, xAlmostEqual
    

def reduceXBounds(xBounds, xSymbolic, f, blocks, dict_options): #boundsAlmostEqual):
    """ Solves an equation system blockwise. For block dimensions > 1 each 
    iteration variable interval of the block is reduced sequentially by all 
    equations of the block. The narrowest bounds from this procedure are taken
    over.
     
        Args: 
            :xBounds:            One set of variable interavls as numpy array
            :xSymbolic:          list with symbolic variables in sympy logic
            :f:                  list with symbolic equation system in sympy logic
            :blocks:             List with blocklists, whereas the blocklists contain
                                 the block elements with index after permutation
            :dict_options:       dictionary with solving settings
        Returns:
            :output:            dictionary with new interval sets(s) in a list and
                                eventually an instance of class failedSystem if
                                the procedure failed.
                        
    """    
    
    output = {}
    xNewBounds = copy.deepcopy(xBounds)
    CPU_count = dict_options["CPU count"] 
    xUnchanged = True
    # Block loop
    for b in range(0, len(blocks)):
        blockDim = len(blocks[b])
        # Variables Loop
        jobs = []
        manager = Manager()
        results = manager.dict()
        done = numpy.zeros (blockDim)
        started = numpy.zeros (blockDim)
        actNum = 0
        for n in range(0, blockDim):
            p = Process(target=reduceXBounds_Worker, args=(xBounds, xNewBounds, xSymbolic, f, blocks, dict_options, n, b, blockDim, results))
            jobs.append(p)        
        
        startAndDeleteJobs(actNum, jobs, started, done, blockDim, CPU_count)
        
        for n in range(0, blockDim):
            if results['%d' % n][0] != []:
                xNewBounds[n] = convertListToMpi(results['%d' % n][0])
                
                if xNewBounds[n] != [xBounds[n]] and xUnchanged:
                    xUnchanged = False
                #boundsAlmostEqual[n] = results['%d' % n][1]
            else: 
                output["intervalsPerm"]  = [] 
                output["noSolution"] = results['%d' % n][1] 
                output["xAlmostEqual"] = False
                return output
    output["xAlmostEqual"] = xUnchanged
    output["intervalsPerm"] = list(itertools.product(*xNewBounds))
    return output


def reduceXBounds_Worker(xBounds, xNewBounds, xSymbolic, f, blocks, dict_options, n, b, blockDim, results):
    """ contains work that can be done in parallel during the reduction of one 
    solution interval sets stored in xBounds. The package multiprocessing is used 
    for parallelization.
    
    Args:
        :xBounds:               list with iteration variable intervals in mpmath.mpi logic
        :xNewBounds:            list with current lists of reduced intervals in mpmath.mpi
                                logic
        :xSymbolic:             list with symbolic iteration variables in sympy logic
        :f:                     list with symbolic equation system in sympy logic
        :blocks:                list with block indices of equation system  
        :dict_options:          dictionary with user specified algorithm settings                              
        :n:                     index of current job (equals global iteration variable index)
        :b:                     index of current block
        :blockDim:              integer with dimension of current block
        :results:               dictionary from multiprocessing where results are stored
                                after a job is done    
                                
    Return:
        :True:                  If method finishes regulary
                                
    """
    y = [] 
    j = blocks[b][n]
    absEpsX = dict_options["absTolX"]
    relEpsX = dict_options["relTolX"]    
    #if boundsAlmostEqual[j]: 
    #    xNewBounds[j] = [xNewBounds[j]]
    #    results['%d' % n] = (convertMpiToList(xNewBounds[j]), True)
    #    return True
    if checkVariableBound(xBounds[j], relEpsX, absEpsX):
            xNewBounds[j] = [xBounds[j]]

    else:
    
        if dict_options["Debug-Modus"]: print j
    
    # Equations Loop
        for m in range(0, blockDim): #TODO: possilby Parallelizing
            i = blocks[b][m]
            if xSymbolic[j] in f[i].free_symbols:
            
                if y == []: y = iNes_procedure.reduceXIntervalByFunction(xBounds, xSymbolic, 
                        f[i], j, dict_options)
                else: 
                    y = iNes_procedure.reduceTwoIVSets(y, iNes_procedure.reduceXIntervalByFunction(xBounds, 
                                                                            xSymbolic, f[i], j, dict_options))
                if y==[] or y==[[]]:
                    results['%d' % n] = ([], FailedSystem(f[i], xSymbolic[j]))
                    return True                
                
        xNewBounds[j] = y

    #if len(xNewBounds[j]) == 1: 
    #    relEpsX = dict_options["relTolX"]
    #    absEpsX = dict_options["absTolX"]
    #    boundsAlmostEqual[j] = checkVariableBound(xNewBounds[j][0], relEpsX, absEpsX)

    results['%d' % n] = (convertMpiToList(xNewBounds[j]),[])#, boundsAlmostEqual[j])
    return True

    
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
    
    
def addNotStartedJobs(actNum, jobs, started, done, jobId, CPU_count):
    """ starts jobs using multiprocessing
    
    Args:
        :actNum:        current number of active jobs
        :jobs:          list with all jobs from process
        :started:       jobNo dimensional numpy array. The entries are binaries 
                        (1 = job started, 0 = job not started)
        :done:          jobNo dimensional numpy array. The entries are binaries 
                        (1 = job done, 0 = job not done)
        :jobId:         integer with index of current job
        :CPU_count:     number of processors
                                   
    """   
    
    if jobs[jobId].is_alive() == False and (started[jobId] == 0) and (actNum < CPU_count):
        jobs[jobId].start()
        started[jobId] = 1
        actNum += 1
    return actNum     

                               
def deleteFinishedJobs(actNum, jobs, started, done, jobId):
    """ deletes finished jobs using multiprocessing
    
    Args:
        :actNum:        current number of active jobs
        :jobs:          list with all jobs from process
        :started:       jobNo dimensional numpy array. The entries are binaries 
                        (1 = job started, 0 = job not started)
        :done:          jobNo dimensional numpy array. The entries are binaries 
                        (1 = job done, 0 = job not done)
        :jobId:         integer with index of current job
                                   
    """   
    
    if ((started[jobId] == 1) & (done[jobId] == 0)):
        jobs[jobId].join(1e-4)

        if (jobs[jobId].is_alive() == False):
            done[jobId] = 1
            actNum -= 1
    return actNum


def convertMpiToList(listWitMpiIntervals):
    """ converts list with intervals in mpmath.mpi formate to list in list with
    interval bounds.
    
    Args:
        :listWitMpiIntervals:       list with mpmath.mpi interval(s)
        
    Returns:
        :listWithIntervalBounds:    list with list(s) [a, b] whereas a is the lower
                                    bound and b the upper bound of the interval
                                    
    """
    
    listWithIntervalBounds = []
    for curInterval in listWitMpiIntervals:
        listWithIntervalBounds.append([float(mpmath.mpf(curInterval.a)), float(mpmath.mpf(curInterval.b))])
    return listWithIntervalBounds
    

def convertListToMpi(listWithIntervalBounds):
    """ converts list with intervals lists into lists with intervals in the
        mpmath.mpi formate.
    
    Args:
        :listWithIntervalBounds:    list with list(s) [a, b] whereas a is the lower
                                    bound and b the upper bound of the interval 
        
    Returns:
        :listWitMpiIntervals:       list with mpmath.mpi interval(s)
        
    """
    
    listWithMpiIntervals = []
    
    for curInterval in listWithIntervalBounds:
        listWithMpiIntervals.append(mpmath.mpi(curInterval[0], curInterval[1]))       
    return listWithMpiIntervals