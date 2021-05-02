"""
***************************************************
Import packages
***************************************************
"""
import copy
import numpy
import mpmath
from modOpt.constraints import iNes_procedure
import itertools
from multiprocessing import Manager, Process #cpu_count
from modOpt.constraints.FailedSystem import FailedSystem

__all__ = ['reduceBoxes', 'reduceBox', 'get_tight_bBounds',
           'reduceXBounds_byFunction', 'convertMpiToList']

"""
***************************************************
Algorithm for parallelization in iNes procedure
***************************************************
"""
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

    CPU_count = dict_options["CPU count Variables"]
    jobs = []
    manager = Manager()
    results = manager.dict()
    done = numpy.zeros(len(f.glb_ID))
    started = numpy.zeros(len(f.glb_ID))
    
    for x_id in range(0, len(f.glb_ID)): # get g(x) and b(x),y 
        p = Process(target=reduceXBounds_byFunction_Worker, args=(f, x_id,
                                                               xBounds,
                                                               dict_options,
                                                               results))
        jobs.append(p)

    startAndDeleteJobs(jobs, started, done, len(f.glb_ID), CPU_count)
      
    for x_id in list(results.keys()):
        iNes_procedure.store_reduced_xBounds(f, int(x_id), convertListToMpi(results[x_id]), varBounds)
   
     
def reduceXBounds_byFunction_Worker(f, x_id, xBounds, dict_options, results):
    """ contains work that can be done in parallel during the reduction of the 
    each variable in a function
    
    Args:
        :f:                 instance of class function
        :x_id:              index (integer) of current variable bound that is 
                            reduced
        :xBounds:           list with variable bonunds in mpmath.mpi formate
        :dict_options:      dictionary with tolerance options
        :results:           dictionary that contains list with reduced variable bounds
        
    """     
    if mpmath.almosteq(xBounds[x_id].a, xBounds[x_id].b, 
                       dict_options["absTol"],
                       dict_options["relTol"]):  
        results['%d' %x_id] = convertMpiToList([xBounds[x_id]])
        return True
             
    if dict_options["Parallel b's"]:
        b = get_tight_bBounds(f, x_id, xBounds, dict_options)
        
    else:    
        b = iNes_procedure.get_tight_bBounds(f, x_id, xBounds, dict_options)
        
    if b == []: 
        results['%d' %x_id] = convertMpiToList([xBounds[x_id]])
        return True
    
    else:
        intervals = iNes_procedure.reduce_x_by_gb(f.g_sym[x_id], f.dgdx_sym[x_id],
                                                 b, f.x_sym, x_id,
                                                 copy.deepcopy(xBounds),
                                                 dict_options)            
        results['%d' %x_id] = convertMpiToList(intervals)
        
    return True                                            
                                 
    
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
    b = iNes_procedure.getBoundsOfFunctionExpression(f.b_sym[x_id], f.x_sym, xBounds, dict_options)  
    if mpmath.almosteq(b.a, b.b, dict_options["relTol"], dict_options["absTol"]):
        return b
    
    if len(f.glb_ID)==1: # this is import if b is interval but there is only one variable in f (for design var intervals in future)
        return b
        
    CPU_count = dict_options["CPU count b's"]
    jobs = []
    manager = Manager()
    b_min = manager.dict()
    b_max = manager.dict()
    done = numpy.zeros(len(f.glb_ID))
    started = numpy.zeros(len(f.glb_ID))

    for y_id in range(0, len(f.glb_ID)): # get b(y)
        p = Process(target=get_tight_bBounds_Worker, args=(f, y_id, x_id,
                                                               xBounds,
                                                               dict_options,
                                                               b_min,
                                                               b_max))
        jobs.append(p)

    startAndDeleteJobs(jobs, started, done, len(f.glb_ID), CPU_count)
    
    try: 
        return mpmath.mpi(max(max([x for x in b_min.values() if x!=[]])), min(min([x for x in b_max.values() if x!=[]])))
         
    except:
        return []


def get_tight_bBounds_Worker(f, y_id, x_id, xBounds, dict_options, b_min, b_max):
    """ contains work that can be done in parallel during the reduction of the 
    bounds of b
    
    Args:
        :f:                 instance of class function
        :x_id:              index (integer) of current variable bound that is 
                            reduced
        :y_id:              index (integer) of current variable bound that is not
                            constant and b is evaluated at.
        :xBounds:           list with variable bonunds in mpmath.mpi formate
        :dict_options:      dictionary with tolerance options
        :b_min:             dictionary for lower bounds of b(y)
        :b_max:             dictionary forupper bounds bof b(y)
        
    """ 
    
    cur_b_min = []
    cur_b_max = []
    
    iNes_procedure.get_tight_bBounds_y(f, x_id, y_id, xBounds, cur_b_min, cur_b_max, dict_options)
        
    b_min['%d' %y_id] = cur_b_min
    b_max['%d' %y_id] = cur_b_max
    
    return True


def reduceBoxes(model, functions, dict_varId_fIds, dict_options, sampling_options=None, solv_options=None):
    """ reduction of multiple boxes
    
    Args:
        :model:                 object of type model
        :functions:             list with instances of class Function        
        :dict_varId_fIds:       dictionary with variable's glb id (key) and list 
                                 with function's glb id they appear in   
        :dict_options:          dictionary with user specified algorithm settings
        :sampling_options:      dictionary with sampling settings
        :solv_options:          dicionary with settings for numerical solver
    
    Return:
        :output:                dictionary with newXBounds and a list of booleans
                                that marks all non reducable variable bounds with True.
                                If solver terminates because of a NoSolution case the
                                critical equation is also stored in results for the error
                                analysis.
            
    """
    output = {}
    CPU_count = dict_options["CPU count Branches"]
    jobs = []
    manager = Manager()
    results = manager.dict()
    done = numpy.zeros(len(model.xBounds))
    started = numpy.zeros(len(model.xBounds))
    output["xSolved"]  = False * numpy.ones(len(model.xBounds), dtype=bool)
    output["xAlmostEqual"] = False * numpy.ones(len(model.xBounds), dtype=bool)   

    for k in range(len(model.xBounds)):  
        p = Process(target=reduceBoxes_Worker, args=(k, model,
                                                               functions, dict_varId_fIds,
                                                               dict_options,
                                                               results,
                                                               sampling_options,
                                                               solv_options))
        jobs.append(p)
    # TODO: Check current boxNo = len(newXBounds) + (nl - (k+1))
    
    startAndDeleteJobs(jobs, started, done, len(model.xBounds), CPU_count)
    #if results.__contains__("tear_id"): dict_options["tear_id"] = results["tear_id"]
    
    output["newXBounds"], output["xAlmostEqual"], output["xSolved"], tearVarIds, output["num_solved"], output["disconti"]  = getReducedXBoundsResults(results, model, dict_options["maxBoxNo"])
    
    if tearVarIds != []:
        for curId in tearVarIds:
            if curId != dict_options["tear_id"]:
                dict_options["tear_id"] = curId
                break

    if output["newXBounds"] == [] and output["xSolved"].all():
        print("All solutions have been found. The last boxes before proof by Interval-Newton are returned.")
        output["xSolved"] = numpy.array([True])    

    elif output["newXBounds"] == []:
        output["noSolution"] = results["0"][1]
        output["newXBounds"] = model.xBounds
    else:
        model.xBounds = output["newXBounds"] 
    return output


def reduceBoxes_Worker(k, model, functions, dict_varId_fIds, dict_options, results, sampling_options=None, solv_options=None):
    """ contains work that can be done in parallel during the reduction of multiple 
    solution interval sets stored in xBounds. The package multiprocessing is used 
    for parallelization.
    
    Args:
        :k:                     index of current job
        :model:                 object of type model
        :functions:             list with instances of class Function
        :dict_varId_fIds:       dictionary with variable's glb id (key) and list 
                                with function's glb id they appear in           
        :dict_options:          dictionary with user specified algorithm settings
        :results:               dictionary from multiprocessing where results are stored
                                after a job is done   
        :sampling_options:      dictionary with sampling settings
        :solv_options:          dicionary with settings for numerical solver
                                
    Return:                     True if method finishes ordinary
                             
    """
    
    disconti = dict_options["disconti"][k]
   
    allBoxes = []
    newtonMethods = {'newton', 'detNewton', '3PNewton'}
    boxNo = len(model.xBounds)
    xBounds = model.xBounds[k]
    num_solved = False
    xAlmostEqual = dict_options["xAlmostEqual"][k]
    xSolved = False

    if dict_options['newton_method'] in newtonMethods:
        newtonSystemDic = iNes_procedure.getNewtonIntervalSystem(xBounds, model, dict_options) 
    else:
        newtonSystemDic = {}

    if xAlmostEqual and not disconti:
        output = {"xNewBounds": [xBounds],
                  "xAlmostEqual": xAlmostEqual,
                  "xSolved": xSolved,                      
            }
      
        output = iNes_procedure.reduceConsistentBox(output, model, functions, dict_options, 
                                     k, dict_varId_fIds, newtonSystemDic, boxNo,
                                     newtonMethods)
    else:  
        output = iNes_procedure.contractBox(xBounds, model, functions, dict_varId_fIds, 
                                        boxNo, dict_options, newtonSystemDic)
            
        if numpy.array(output["xAlmostEqual"]).all() and not output["xSolved"]: 
            if not sampling_options ==None and not solv_options == None:
                num_solved = iNes_procedure.lookForSolutionInBox(model, k, dict_options, 
                                                  sampling_options, solv_options)
         
    for box in output["xNewBounds"]:
        allBoxes.append(convertMpiToList(numpy.array(box, dtype=object)))
        
    if output.__contains__("noSolution"):
        results['%d' %k] = ([], output["noSolution"],[],[],[],[])
        
    elif output.__contains__("uniqueSolutionInBox"):
        results['%d' %k] = ([], output["xAlmostEqual"], 
                            True,
                            dict_options["tear_id"], 
                            num_solved,  len(output["xNewBounds"])*[output["disconti"]])        
        
    else:        
        results['%d' %k] = (allBoxes, output["xAlmostEqual"], 
                            output["xSolved"],
                            dict_options["tear_id"], 
                            num_solved, len(output["xNewBounds"])*[output["disconti"]])

    return True


def startAndDeleteJobs(jobs, started, done, jobNo, CPU_count):
    """ starts jobs using multiprocessing and deletes finished ones
    
    Args:
        :jobs:          list with all jobs from process
        :started:       jobNo dimensional numpy array. The entries are binaries 
                        (1 = job started, 0 = job not started)
        :done:          jobNo dimensional numpy array. The entries are binaries 
                        (1 = job done, 0 = job not done)
        :jobNo:         integer with total number of jobs
        :CPU_count:     number of cores
                           
    """  
    
    actNum = 0
    
    while numpy.sum(done) < jobNo:
        for jobId in range(0, jobNo):
            actNum = addNotStartedJobs(actNum, jobs, started, done, jobId, CPU_count)
            actNum = deleteFinishedJobs(actNum, jobs, started, done, jobId)


def getReducedXBoundsResults(results, model, maxBoxNo):
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
    noOfxBounds = len(model.xBounds)
    xAlmostEqual = [False] * noOfxBounds 
    xSolved = False * numpy.ones(noOfxBounds, dtype=bool) 
    newXBounds = []
    tearVarIds = []
    num_solved = False
    disconti = [False] * noOfxBounds 
    
    for k in range(0, noOfxBounds):
        if noOfxBounds < 1: return [], [], [], []             
        if results['%d' %k][0] != []: 
            curNewXBounds = results['%d' %k][0] # [[[a1], [b1], [c1]], [[a2], [b2], [c2]]]
            boxNo = len(newXBounds) + len(curNewXBounds) + (noOfxBounds - (k+1))
            if boxNo <= maxBoxNo:
                for curNewXBound in curNewXBounds:
                    newXBounds.append(numpy.array(convertListToMpi(curNewXBound), dtype=object))
                xAlmostEqual[k] = (results['%d' %k][1])
                xSolved[k] = (results['%d' %k][2])   
                tearVarIds.append(results['%d' %k][3])
                disconti[k] = results['%d' %k][5]
                if results['%d' %k][4]==True: num_solved=True
            else:
                newXBounds.append(model.xBounds[k])
                xAlmostEqual[k] = (True)
                xSolved[k] = (False)
                disconti[k] = results['%d' %k][5]
        
        elif xSolved[k] ==True:
             noOfxBounds -= 1
             xAlmostEqual[k] = (False)
             xSolved[k] = (True) 
             disconti[k] = [] 
             if results['%d' %k][4]==True: num_solved=True
        else:
            noOfxBounds -= 1
            xAlmostEqual[k] = (False)
            xSolved[k] = (False) 
            disconti[k] = []
    disconti_cleaned = []
    
    for cb in disconti:
            if isinstance(cb, list): # if interval splits
                disconti_cleaned += cb
            elif cb == []:
                continue
            else:
                disconti_cleaned.append(cb)
                    
    return newXBounds, xAlmostEqual, xSolved, tearVarIds, num_solved, disconti_cleaned
    

def reduceBox(xBounds, model, functions, dict_varId_fIds, boxNo, dict_options, newtonSystemDic): #boundsAlmostEqual):
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
        :output:            dictionary with new interval sets(s) in a list and
                            eventually an instance of class failedSystem if
                            the procedure failed.
                        
    """        
    output = {}
    dim = len(model.xSymbolic)
    xNewBounds = copy.deepcopy(xBounds)
    CPU_count = dict_options["CPU count Variables"] 
    xUnchanged = []
    xSolved = []
    
    if dict_options['hc_method']=='HC4':
        output, empty = iNes_procedure.doHC4(model, functions, xBounds, xNewBounds, output)
        if empty: return output
          
    jobs = []
    manager = Manager()
    results = manager.dict()
    done = numpy.zeros (dim)
    started = numpy.zeros (dim)
    
    for i in range(0, dim):
        p = Process(target=reduceBox_Worker, args=(xBounds, xNewBounds, functions, i, dict_varId_fIds, dict_options, newtonSystemDic, results))
        jobs.append(p)        
        
    startAndDeleteJobs(jobs, started, done, dim, CPU_count)
        
    for i in range(0, dim):
        
        if results['%d' % i][0] != []:
            xNewBounds[i] = convertListToMpi(results['%d' % i][0])
            xUnchanged.append(results['%d' % i][2])
            xSolved.append(results['%d' % i][3])
            #xUnchanged = iNes_procedure.checkXforEquality(xBounds[i], 
            #                                              xNewBounds[i], 
            #                                              xUnchanged, dict_options)
            
            
        else: 
            output["xNewBounds"]  = [] 
            output["noSolution"] = results['%d' % i][1] 
            output["xSolved"] = False
            output["xAlmostEqual"] = False
            return output
        #results['%d' % i][1]: (xNewBounds[i] ==xBounds[i])
        
    output["xSolved"] = all(xSolved)    
    output["xAlmostEqual"] = all(xUnchanged)
    output["xNewBounds"] = list(itertools.product(*xNewBounds))
    
    return output


def reduceBox_Worker(xBounds, xNewBounds, functions, i, dict_varId_fIds, dict_options, newtonSystemDic, results):
    """ contains work that can be done in parallel during the reduction of one 
    solution interval sets stored in xBounds. The package multiprocessing is used 
    for parallelization.
    
    Args:
        :xBounds:               list with iteration variable intervals in mpmath.mpi logic
        :xNewBounds:            list with current lists of reduced intervals in mpmath.mpi
                                logic
        :functions:             list with instances of class Function
        :i:                     index of current job (equals global iteration variable index)
        :dict_varId_fIds:       dictionary with variable's glb id (key) and list 
        :dict_options:          dictionary with user specified algorithm settings                              
        :results:               dictionary from multiprocessing where results are stored
                                after a job is done    
                                
    Return:
        :True:                  If method finishes regulary
                                
    """

    xUnchanged = True
    xSolved = True
    y = [xNewBounds[i]]   
    dict_options_temp = copy.deepcopy(dict_options)
    newtonMethods = {'newton', 'detNewton', '3PNewton'}
    
    if dict_options["Debug-Modus"]: print(i)
    iNes_procedure.checkIntervalAccuracy(xNewBounds, i, dict_options_temp) 

    if not iNes_procedure.variableSolved(y, dict_options_temp) and dict_options['newton_method'] in newtonMethods: 
        y = iNes_procedure.doIntervalNewton(newtonSystemDic, y, xBounds, i, dict_options_temp)
                    
        if y == [] or y ==[[]]: 
            j=dict_varId_fIds[i][0]
            results['%d' % i] = ([], FailedSystem(functions[j].f_sym, functions[j].x_sym[functions[j].glb_ID.index(i)]), 
                                 False, False)  
            return True  
           
    if not iNes_procedure.variableSolved(y, dict_options_temp) and dict_options['bc_method']=='b_normal':
        for j in dict_varId_fIds[i]:
            f = functions[j]
            y = iNes_procedure.doBoxReduction(f, xBounds, y, i, dict_options_temp)
                    
            if y == [] or y ==[[]]: 
                results['%d' % i] = ([], FailedSystem(f.f_sym, f.x_sym[f.glb_ID.index(i)]), 
                                     False, False)
                return True      
    
    # Update quantities        
    if not iNes_procedure.variableSolved(y, dict_options): xSolved = False 
    xNewBounds[i] = y
    xUnchanged = iNes_procedure.checkXforEquality(xBounds[i], xNewBounds[i], xUnchanged, 
                                       dict_options_temp)    
    results['%d' % i] = (convertMpiToList(xNewBounds[i]),[], xUnchanged, xSolved)
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
    
    if jobs[jobId].is_alive() == False and (started[jobId] == 0) and (actNum <= CPU_count):
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