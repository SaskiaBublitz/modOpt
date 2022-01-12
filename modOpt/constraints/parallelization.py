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
                                          
def reduceBoxes(model, dict_options, sampling_options=None, solv_options=None):
    """ reduction of multiple boxes
    
    Args:
        :model:                 object of type model 
        :dict_options:          dictionary with user specified algorithm settings
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
    foundSolutions = []
    CPU_count = dict_options["CPU count Branches"]
    jobs = []
    manager = Manager()
    results = manager.dict()
    done = numpy.zeros(len(model.xBounds))
    started = numpy.zeros(len(model.xBounds))
    output["xSolved"]  = dict_options["xSolved"]
    output["xAlmostEqual"] = dict_options["xAlmostEqual"]  
    output["disconti"] = dict_options["disconti"]
    if dict_options["cut_Box"] in {"all", "tear", True}: model.cut = True
    dict_options["ready_for_reduction"] = get_index_of_boxes_for_reduction(dict_options["xSolved"],
                                                                           dict_options["xAlmostEqual"], 
                                                                           dict_options["maxBoxNo"])    
    for k,x in enumerate(model.xBounds):  
        p = Process(target=reduceBoxes_Worker, args=(k, model, dict_options,
                                                               results,
                                                               sampling_options,
                                                               solv_options))
        jobs.append(p)
    
    startAndDeleteJobs(jobs, started, done, len(model.xBounds), CPU_count)
    
    (output["newXBounds"], 
     output["xAlmostEqual"], 
     output["xSolved"], tearVarIds, 
     output["num_solved"], 
     output["disconti"], 
     output["complete_parent_boxes"], 
     output["complete_boxes"]) = getReducedXBoundsResults(results, model, 
                                                          dict_options["maxBoxNo"],
                                                          foundSolutions)
                                                          
    if foundSolutions != []:  dict_options["FoundSolutions"] = foundSolutions        
                                             
    if tearVarIds != []: 
        for curId in tearVarIds:
            if curId != dict_options["tear_id"]:
                dict_options["tear_id"] = curId
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


def reduceBoxes_Worker(k, model, dict_options, results, sampling_options=None, 
                       solv_options=None):
    """ contains work that can be done in parallel during the reduction of multiple 
    solution interval sets stored in xBounds. The package multiprocessing is used 
    for parallelization.
    
    Args:
        :k:                     index of current job
        :model:                 object of type model
                                with function's glb id they appear in           
        :dict_options:          dictionary with user specified algorithm settings
        :results:               dictionary from multiprocessing where results are 
                                stored after a job is done   
        :sampling_options:      dictionary with sampling settings
        :solv_options:          dicionary with settings for numerical solver
                                
    Return:                     True if method finishes ordinary
                             
    """

    if "FoundSolutions" in dict_options.keys(): 
        solNo = len(dict_options["FoundSolutions"])
    else: solNo = 0
    newSol = []
    model.interval_jac = None  
    model.jac_center = None
    disconti = dict_options["disconti"][k]
    if dict_options["cut_Box"] in {"all", "tear", True}: model.cut = True
    
    allBoxes = []
    newtonMethods = {'newton', 'detNewton', '3PNewton'}
    boxNo = len(model.xBounds)
    xBounds = model.xBounds[k]
    num_solved = False
    xAlmostEqual = dict_options["xAlmostEqual"][k]
    xSolved = dict_options["xSolved"][k]
    
    if xSolved: 
        output = {"xNewBounds": [xBounds],
                  "xAlmostEqual": [xAlmostEqual],
                  "xSolved": [xSolved],
                  "disconti": disconti,
                  "complete_parent_boxes":[model.complete_parent_boxes[k]],
                  "uniqueSolutionInBox": True
            }
        model.cut = False
    elif xAlmostEqual and not disconti:    
        output = {"xNewBounds": [xBounds],
                  "xAlmostEqual": [xAlmostEqual],
                  "xSolved": [xSolved]                      
            }
        output = iNes_procedure.reduceConsistentBox(output, model, dict_options, 
                                     k, boxNo,
                                     newtonMethods)
        model.cut = output["cut"]
        #print(model.cut)
        if model.teared: 
            output["complete_parent_boxes"] = (len(output["xNewBounds"]) * 
                                               [[dict_options["iterNo"]-1, k]])
            model.teared = False
        else: 
            output["complete_parent_boxes"] = (len(output["xNewBounds"]) * 
                                           [ model.complete_parent_boxes[k]])               
    elif ((xAlmostEqual and disconti and boxNo >= dict_options["maxBoxNo"]) or 
          (not dict_options["ready_for_reduction"][k])):
        output = {"xNewBounds": [xBounds],
                  "xAlmostEqual": [xAlmostEqual],
                  "xSolved": [xSolved],
                  "disconti": disconti,
                  "complete_parent_boxes": [model.complete_parent_boxes[k]],
            }
        model.cut = False
    else:
        if dict_options["Debug-Modus"]: print(f'Box {k+1}')
        if not xAlmostEqual and disconti and dict_options["consider_disconti"]: 
            store_boxNo = dict_options["boxNo"]
            dict_options["boxNo"] = dict_options["maxBoxNo"]
            output = iNes_procedure.contractBox(xBounds, model, 
                                                dict_options["boxNo"] , 
                                                dict_options)
            dict_options["boxNo"]  = store_boxNo   
        else:
            output = iNes_procedure.contractBox(xBounds, model, 
                                        boxNo, dict_options)
        
            output["complete_parent_boxes"] = (len(output["xNewBounds"]) * 
                                           [ model.complete_parent_boxes[k]])   
        
        # if (True in output["xSolved"] and len(output["xNewBounds"]) == 1 
        #     and dict_options["hybrid_approach"] and output.__contains__("box_has_unique_solution")):
        #     #if not "FoundSolutions" in dict_options.keys():  
        #     model.xBounds[k] = output["xNewBounds"][0]
        #     num_solved = iNes_procedure.lookForSolutionInBox(model, k, 
        #                                                       dict_options, 
        #                                                       sampling_options, 
        #                                                       solv_options)        
        
        #     if num_solved:    
        #          if not iNes_procedure.test_for_root_inclusion(output["xNewBounds"][0], 
        #                                  dict_options["FoundSolutions"], 
        #                                  dict_options["absTol"]):
        #              output["uniqueSolutionInBox"]= True
        
        # if (True in output["xSolved"] and len(output["xNewBounds"]) == 1 
        #     and dict_options["hybrid_approach"]):
        #     model.xBounds[k] = output["xNewBounds"][0]

        #     num_solved = iNes_procedure.lookForSolutionInBox(model, k, 
        #                                                      dict_options, 
        #                                                      sampling_options, 
        #                                                      solv_options) 
        # elif (True in output["xSolved"] and len(output["xNewBounds"]) == 1 and 
        #         not output.__contains__("box_has_unique_solution")):
        #         output["xSolved"][0]=False
            # if num_solved:    
            #     if not iNes_procedure.test_for_root_inclusion(output["xNewBounds"][0], 
            #                             dict_options["FoundSolutions"], 
            #                             dict_options["absTol"]):
            #         output["xSolved"][0] = False
            #if not num_solved: 
            #    output["xSolved"][0] = False
                #model.cut = True
        if (all(output["xAlmostEqual"]) and not all(output["xSolved"]) and 
            dict_options["hybrid_approach"]): 

            if not sampling_options ==None and not solv_options == None:
                num_solved = iNes_procedure.lookForSolutionInBox(model, k, 
                                                                 dict_options, 
                                                                 sampling_options, 
                                                                 solv_options) 
                if "FoundSolutions" in dict_options.keys():
                    if len(dict_options["FoundSolutions"]) > solNo:
                        newSol = [dict_options["FoundSolutions"][i] 
                                  for i in range(solNo, 
                                                 len(dict_options["FoundSolutions"]))] 

                    
    for box in output["xNewBounds"]:
        allBoxes.append(convertMpiToList(numpy.array(box, dtype=object)))
    if output.__contains__("noSolution"):
        results['%d' %k] = ([], output["noSolution"],[],[],[],[],[], False, [])
        
    elif output.__contains__("uniqueSolutionInBox"):
        results['%d' %k] = (allBoxes, [True], 
                            [True],
                            dict_options["tear_id"], 
                            num_solved,  
                            len(output["xNewBounds"])*[output["disconti"]],
                            output["complete_parent_boxes"], model.cut, newSol)        
        
    # elif (not output.__contains__("box_has_unique_solution") and True in output["xSolved"] ):
    #     results['%d' %k] = (allBoxes, output["xAlmostEqual"], 
    #                         [False],
    #                         dict_options["tear_id"], 
    #                         num_solved, 
    #                         len(output["xNewBounds"])*[output["disconti"]],
    #                         output["complete_parent_boxes"], model.cut, newSol)
    else:
        results['%d' %k] = (allBoxes, output["xAlmostEqual"], 
                            output["xSolved"],
                            dict_options["tear_id"], 
                            num_solved, 
                            len(output["xNewBounds"])*[output["disconti"]],
                            output["complete_parent_boxes"], model.cut, newSol)
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


def getReducedXBoundsResults(results, model, maxBoxNo, foundSolutions=None):
    """ extracts quantities from multiprocessing results
    
    Args:
        :results:           dictionary from multiprocessing, where results are stored
                            after a job is done
        :model:             instance of type model                    
        :noOfxBounds:       number of solution interval sets in xBounds


    Return:
        :newXBounds:        list with reduced solution interval sets
        :xAlmostEqual:      numpy array with boolean entries for each interval vector
                            is true for interval vectors that have not changed in the
                            last reduction step anymore
                                 
    """
    noOfxBounds = len(model.xBounds)
    xAlmostEqual = [[]] * noOfxBounds 
    xSolved = [[]] * noOfxBounds  
    newXBounds = []
    tearVarIds = []
    num_solved = False
    disconti = [False] * noOfxBounds 
    complete_parent_boxes = []
    cut = []

    for k in range(noOfxBounds):
        if results['%d' %k][8] != []:
            if not foundSolutions:
                foundSolutions = results['%d' %k][8]            
            else:
                for new_sol in results['%d' %k][8]:
                    for old_sol in foundSolutions:
                        if (new_sol == old_sol).all(): break
                    else:
                        foundSolutions.append(new_sol)  
                       
        if noOfxBounds < 1: return [], [], [], []             
        if results['%d' %k][0] != []: 
            curNewXBounds = results['%d' %k][0] # [[[a1], [b1], [c1]], [[a2], [b2], [c2]]]
            boxNo = len(newXBounds) + len(curNewXBounds) + (noOfxBounds - (k+1))
            if boxNo <= maxBoxNo:
                for curNewXBound in curNewXBounds:
                    newXBounds.append(numpy.array(convertListToMpi(curNewXBound), 
                                                  dtype=object))
                xAlmostEqual[k] = (results['%d' %k][1])
                xSolved[k] = (results['%d' %k][2])   
                tearVarIds.append(results['%d' %k][3])
                disconti[k] = results['%d' %k][5]
                complete_parent_boxes += results['%d' %k][6]
                if results['%d' %k][4]==True: num_solved=True
                cut += [results['%d' %k][7]]
            else:
                newXBounds.append(model.xBounds[k])
                xAlmostEqual[k] = [True]
                xSolved[k] = [False]
                disconti[k] = [True]#results['%d' %k][5]
                complete_parent_boxes += [model.complete_parent_boxes[k]]
                cut += [results['%d' %k][7]]

        elif xSolved[k] ==True:
             noOfxBounds -= 1
             
             xAlmostEqual[k] = (False)
             xSolved[k] = (True) 
             disconti[k] = [] 
             complete_parent_boxes +=results['%d' %k][6]
             if results['%d' %k][4]==True: num_solved=True
             cut += [results['%d' %k][7]]
             
        else:
            noOfxBounds -= 1
            xAlmostEqual[k] = []
            xSolved[k] = [] 
            disconti[k] = []
            complete_parent_boxes += []
            
    disconti_cleaned = []
    if not cut == []: model.cut = all(cut)
    for cb in disconti:
            if isinstance(cb, list): # if interval splits
                disconti_cleaned += cb
            elif cb == []:
                continue
            else:
                disconti_cleaned.append(cb)
    
    return (newXBounds, xAlmostEqual, xSolved, tearVarIds, num_solved, 
            disconti_cleaned, complete_parent_boxes, cut)
    

def reduceBox(xBounds, model, boxNo, dict_options): #boundsAlmostEqual):
    """ reduce box spanned by current intervals of xBounds.
     
    Args: 
        :xBounds:            numpy array with box
        :model:              instance of class Model
        :boxNo:              number of boxes as integer  
        :dict_options:       dictionary with user specified algorithm settings
        
    Return:
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
        output, empty = iNes_procedure.doHC4(model, xBounds, xNewBounds, output)
        if empty: return output
          
    jobs = []
    manager = Manager()
    results = manager.dict()
    done = numpy.zeros (dim)
    started = numpy.zeros (dim)
    
    for i in range(0, dim):
        p = Process(target=reduceBox_Worker, args=(model, xBounds, xNewBounds, 
                                                   i, dict_options, results))
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


def reduceBox_Worker(model, xBounds, xNewBounds, i, dict_options, results):
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
        :dict_options:          dictionary with user specified algorithm settings                              
        :results:               dictionary from multiprocessing where results 
                                are stored after a job is done    
                                
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

    if (not iNes_procedure.variableSolved(y, dict_options_temp) 
        and dict_options['newton_method'] in newtonMethods): 
        y = iNes_procedure.doIntervalNewton(y, xBounds, i, dict_options_temp)
                    
        if y == [] or y ==[[]]: 
            j=model.dict_varId_fIds[i][0]
            results['%d' % i] = ([], 
                                 FailedSystem(model.functions[j].f_sym, 
                                 model.functions[j].x_sym[model.functions[j].glb_ID.index(i)]), 
                                 False, False)  
            return True  
           
    if (not iNes_procedure.variableSolved(y, dict_options_temp) 
        and dict_options['bc_method']=='b_normal'):
        for j in model.dict_varId_fIds[i]:
            f = model.functions[j]
            y = iNes_procedure.doBoxReduction(f, xBounds, y, i, 
                                              dict_options_temp)                    
            if y == [] or y ==[[]]: 
                results['%d' % i] = ([], 
                                     FailedSystem(f.f_sym, 
                                     f.x_sym[f.glb_ID.index(i)]), 
                                     False, False)
                return True      
    
    # Update quantities        
    if not iNes_procedure.variableSolved(y, dict_options): xSolved = False 
    xNewBounds[i] = y
    xUnchanged = iNes_procedure.checkXforEquality(xBounds[i], xNewBounds[i], 
                                                  xUnchanged, dict_options_temp)    
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