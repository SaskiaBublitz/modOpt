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
numpy.seterr(all="ignore")

__all__ = ['reduceBoxes', 'reduceBox', 'convertMpiToList']

"""
***************************************************
Algorithm for parallelization in iNes procedure
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
        :output:                dictionary with newXBounds and a list of booleans
                                that marks all non reducable variable bounds with 
                                True. If solver terminates because of a NoSolution 
                                case the critical equation is also stored in 
                                results for the error analysis.
            
    """
    output = {}
    
    CPU_count = min(len(model.xBounds), bxrd_options["cpuCountBoxes"])
    #CPU_count = bxrd_options["cpuCountBoxes"]
    jobs = []
    manager = Manager()
    results = manager.dict()
    if bxrd_options["cutBox"] in {"all", "tear", True}: model.cut = True
    bxrd_options["ready_for_reduction"] = iNes_procedure.get_index_of_boxes_for_reduction(bxrd_options["xSolved"],
                                                                          bxrd_options["cut"], 
                                                                           bxrd_options["maxBoxNo"])    
    for k,x in enumerate(model.xBounds):  
        p = Process(target=reduce_boxes_worker, args=(k, model, bxrd_options,
                                                               results,
                                                               sampling_options,
                                                               solv_options))
        jobs.append(p)
    
    startAndDeleteJobs(jobs, CPU_count)
    
    output, tearVarIds = getReducedXBoundsResults(results, model, 
                                                  bxrd_options["maxBoxNo"],
                                                  bxrd_options)
                                                                                                       
    if tearVarIds != []: 
        for curId in tearVarIds:
            if curId != bxrd_options["tear_id"]:
                bxrd_options["tear_id"] = curId
                break
    
    if output["newXBounds"] == [] and not any(output["xSolved"]):
        output["noSolution"] = results["0"][1]
        output["newXBounds"] = model.xBounds   

    elif output["newXBounds"] == [] and all(output["xSolved"]):
        print("All solutions have been found. The last boxes before proof"+ 
              "by Interval-Newton are returned.")
        output["xSolved"] = numpy.array([True])    

    elif output["newXBounds"] == []:
        output["noSolution"] = results["0"][1]
        output["newXBounds"] = model.xBounds
    else:
        model.xBounds = output["newXBounds"]
    
    return output

def reduce_boxes_worker(k, model, bxrd_options, results_tot, sampling_options=None, 
                       solv_options=None):
    """ wrapping function for reducing a box in parallel
    
    Args:
        :k:                     index of currently reduced box
        :model:                 object of type model 
        :bxrd_options:          dictionary with user specified algorithm settings
        :results_tot:           dictionary from multiprocessing for output from reduce_box
        :sampling_options:      dictionary with sampling settings
        :solv_options:          dicionary with settings for numerical solver
    
    Return:
        :output:                dictionary with newXBounds and a list of booleans
                                that marks all non reducable variable bounds with 
                                True. If solver terminates because of a NoSolution 
                                case the critical equation is also stored in 
                                results for the error analysis.
            
    """    
    
    newSol = []
    output = {"num_solved": False, "disconti": [], "complete_parent_boxes": [],
        "xSolved": [], "xAlmostEqual": [], "cut": [],
        }
    allBoxes = []
    emptyBoxes = []
    bxrd_options["boxNo"] = len(model.xBounds)
    if "FoundSolutions" in bxrd_options.keys(): 
        solNo = len(bxrd_options["FoundSolutions"])
    else: solNo = 0
    
    emptyboxes = iNes_procedure.reduce_box(model, allBoxes, emptyBoxes, 
                                       k, output, bxrd_options,
                                       sampling_options, solv_options)
    
    if "FoundSolutions" in bxrd_options.keys():   
        if len(bxrd_options["FoundSolutions"]) > solNo:
            newSol = [bxrd_options["FoundSolutions"][i] for i in 
                      range(solNo, len(bxrd_options["FoundSolutions"]))]     

    if emptyboxes:
        results_tot['%d' %k] = ([], emptyboxes,[],[],[],[],[], False, [],[])
    else:
        list_box = [convertMpiToList(numpy.array(box, dtype=object)) for box in 
                                 allBoxes]
        results_tot['%d' %k] = (list_box, output["xAlmostEqual"], 
                                output["xSolved"],
                                bxrd_options["tear_id"], 
                                output["num_solved"], 
                                output["disconti"],
                                output["complete_parent_boxes"], 
                                output["cut"], newSol)
    return True


def startAndDeleteJobs(jobs, CPU_count):
    """ starts jobs using multiprocessing and deletes finished ones
    
    Args:
        :jobs:          list with all jobs from process
        :CPU_count:     number of cores
                           
    """    
    actNum = 0
    started = [False] * len(jobs)
    done = [False] * len(jobs)
    
    while numpy.sum(done) < len(jobs):
        for jobId in range(len(jobs)):
            actNum = addNotStartedJobs(actNum, jobs, started, done, jobId, CPU_count)
            actNum = deleteFinishedJobs(actNum, jobs, started, done, jobId)


def getReducedXBoundsResults(results, model, maxBoxNo, bxrd_options):
    """ extracts quantities from multiprocessing results
    
    Args:
        :results:           dictionary from multiprocessing, where results are stored
                            after a job is done
        :model:             instance of type model                    
        :maxBoxNo:          maximum number of boxes
        :bxrd_options:      dictionary with results from former reduction step


    Return:
        :newXBounds:        list with reduced solution interval sets
        :xAlmostEqual:      numpy array with boolean entries for each interval vector
                            is true for interval vectors that have not changed in the
                            last reduction step anymore
                                 
    """
    noOfxBounds = len(model.xBounds)
    output = {"num_solved": False, "disconti": [], "complete_parent_boxes": [],
              "xSolved": [], "xAlmostEqual": [], "cut": [],"newXBounds":[],
              }
    
    tearVarIds = []
    cut = []

    for k in range(noOfxBounds):
        if results['%d' %k][8] != []:
            if not bxrd_options.__contains__("FoundSolutions"):
                bxrd_options["FoundSolutions"] = results['%d' %k][8]            
            else:
                for new_sol in results['%d' %k][8]:
                    for old_sol in bxrd_options["FoundSolutions"]:
                        if (new_sol == old_sol).all(): break
                    else:
                        bxrd_options["FoundSolutions"].append(new_sol)  
                       
        if noOfxBounds < 1: return [], [], [], []             
        if results['%d' %k][0] != []: 
            curNewXBounds = results['%d' %k][0] # [[[a1], [b1], [c1]], [[a2], [b2], [c2]]]
            boxNo = (len(output["newXBounds"]) + len(curNewXBounds) + 
                     (noOfxBounds - (k+1)))
            if boxNo <= maxBoxNo:
                for curNewXBound in curNewXBounds:
                    output["newXBounds"].append(numpy.array(convertListToMpi(
                        curNewXBound), dtype=object))
                output["xAlmostEqual"] += results['%d' %k][1]
                output["xSolved"] += results['%d' %k][2]   
                tearVarIds.append(results['%d' %k][3])
                output["disconti"] += results['%d' %k][5]

                output["cut"] += results['%d' %k][7]
                output["complete_parent_boxes"] += results['%d' %k][6]
                if results['%d' %k][4]==True: output["num_solved"] = True
                if results['%d' %k][7] != []: cut += results['%d' %k][7]
            else:
                output["newXBounds"].append(model.xBounds[k])
                output["xAlmostEqual"] += [True]
                output["xSolved"] += [False]
                output["cut"] += [False]
                output["disconti"] += [bxrd_options["disconti"][k]]#results['%d' %k][5]
                output["complete_parent_boxes"] += [model.complete_parent_boxes[k]]
                if results['%d' %k][7] != []: cut += results['%d' %k][7]
             
        else:
            noOfxBounds -= 1
    print(output["xSolved"])
    if output["cut"]: 
        model.cut = any(output["cut"])

    return output, tearVarIds 


def reduceBox(xBounds, model, boxNo, bxrd_options): #boundsAlmostEqual):
    """ reduce box spanned by current intervals of xBounds.
     
    Args: 
        :xBounds:            numpy array with box
        :model:              instance of class Model
        :boxNo:              number of boxes as integer  
        :bxrd_options:       dictionary with user specified algorithm settings
        
    Return:
        :output:            dictionary with new interval sets(s) in a list and
                            eventually an instance of class failedSystem if
                            the procedure failed.
                        
    """        
    output = {}
    dim = len(model.xSymbolic)
    xNewBounds = copy.deepcopy(xBounds)
    CPU_count = bxrd_options["cpuCountVariables"] 
    xUnchanged = []
    xSolved = []
    
    if bxrd_options['hcMethod']=='HC4':
        output, empty = iNes_procedure.doHC4(model, xBounds, xNewBounds, output)
        if empty: return output
          
    jobs = []
    manager = Manager()
    results = manager.dict()
    done = numpy.zeros (dim)
    started = numpy.zeros (dim)
    
    for i in range(0, dim):
        p = Process(target=reduceBox_Worker, args=(model, xBounds, xNewBounds, 
                                                   i, bxrd_options, results))
        jobs.append(p)        
        
    startAndDeleteJobs(jobs, started, done, dim, CPU_count)
        
    for i in range(0, dim):
        
        if results['%d' % i][0] != []:
            xNewBounds[i] = convertListToMpi(results['%d' % i][0])
            xUnchanged.append(results['%d' % i][2])
            xSolved.append(results['%d' % i][3])
        else: 
            output["xNewBounds"]  = [] 
            output["noSolution"] = results['%d' % i][1] 
            output["xSolved"] = False
            output["xAlmostEqual"] = False
            return output
        
    output["xSolved"] = all(xSolved)    
    output["xAlmostEqual"] = all(xUnchanged)
    output["xNewBounds"] = list(itertools.product(*xNewBounds))
    
    return output


def reduceBox_Worker(model, xBounds, xNewBounds, i, bxrd_options, results):
    """ contains work that can be done in parallel during the reduction of one 
    solution interval sets stored in xBounds. The package multiprocessing is 
    used for parallelization.
    
    Args:
        :model:                 instance of type model
        :xBounds:               list with iteration variable intervals in 
                                mpmath.mpi logic
        :xNewBounds:            list with current lists of reduced intervals in 
                                mpmath.mpi logic
        :i:                     index of current job (equals global iteration 
                                                      variable index)
        :bxrd_options:          dictionary with user specified algorithm settings                              
        :results:               dictionary from multiprocessing where results 
                                are stored after a job is done    
                                
    Return:
        :True:                  If method finishes regulary
                                
    """
    xUnchanged = True
    xSolved = True
    y = [xNewBounds[i]]   
    bxrd_options_temp = copy.deepcopy(bxrd_options)
    newtonMethods = {'newton', 'detNewton', '3PNewton'}
    
    if bxrd_options["Debug-Modus"]: print(i)
    iNes_procedure.checkIntervalAccuracy(xNewBounds, i, bxrd_options_temp) 

    if (not iNes_procedure.variableSolved(y, bxrd_options_temp) 
        and bxrd_options['newtonMethod'] in newtonMethods): 
        y = iNes_procedure.doIntervalNewton(y, xBounds, i, bxrd_options_temp)
                    
        if y == [] or y ==[[]]: 
            j=model.dict_varId_fIds[i][0]
            results['%d' % i] = ([], 
                                 FailedSystem(model.functions[j].f_sym, 
                                 model.functions[j].x_sym[model.functions[j].glb_ID.index(i)]), 
                                 False, False)  
            return True  
           
    if (not iNes_procedure.variableSolved(y, bxrd_options_temp) 
        and bxrd_options['bcMethod']=='b_normal'):
        for j in model.dict_varId_fIds[i]:
            f = model.functions[j]
            y = iNes_procedure.doBoxReduction(f, xBounds, y, i, 
                                              bxrd_options_temp)                    
            if y == [] or y ==[[]]: 
                results['%d' % i] = ([], 
                                     FailedSystem(f.f_sym, 
                                     f.x_sym[f.glb_ID.index(i)]), 
                                     False, False)
                return True      
    
    # Update quantities        
    if not iNes_procedure.variableSolved(y, bxrd_options): xSolved = False 
    xNewBounds[i] = y
    xUnchanged = iNes_procedure.checkXforEquality(xBounds[i], xNewBounds[i], 
                                                  xUnchanged, bxrd_options_temp)    
    results['%d' % i] = (convertMpiToList(xNewBounds[i]),[], xUnchanged, xSolved)
    
    return True

    
def checkVariableBound(newXInterval, relEpsX, absEpsX):
    """ if lower and upper bound of a variable are almost equal the boolean 
    boundsAlmostEqual is set to true.
    
    Args:
        newXInterval:       variable interval in mpmath.mpi logic
        relEpsX:            relative variable interval tolerance
        absEpsX:            absolute variable interval tolerance
        
    Return:                True, if lower and upper variable bound are almost
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
    if (jobs[jobId].is_alive() == False 
        and (started[jobId] == 0) and (actNum <= CPU_count)):
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
        
    Returns:                        list with list(s) [a, b] whereas a is the lower
                                    bound and b the upper bound of the interval
                                    
    """
    return [[float(mpmath.mpf(iv.a)), 
             float(mpmath.mpf(iv.b))] for iv in listWitMpiIntervals]


def get_index_of_boxes_for_reduction(xSolved, cut, maxBoxNo):
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


def convertListToMpi(listWithIntervalBounds):
    """ converts list with intervals lists into lists with intervals in the
        mpmath.mpi formate.
    
    Args:
        :listWithIntervalBounds:    list with list(s) [a, b] whereas a is the 
                                    lower bound and b the upper bound of the 
                                    interval 
        
    Return:                        list with mpmath.mpi interval(s)
        
    """    
    return [mpmath.mpi(iv[0], iv[1]) for iv in listWithIntervalBounds]