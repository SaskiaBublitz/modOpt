"""
***************************************************
Import packages
***************************************************
"""
import copy
import numpy
from modOpt.solver import main, results
from multiprocessing import Manager, Process #cpu_count

__all__ = ['solveMultipleSamples']

"""
***************************************************
Algorithm for parallelization in iNes procedure
***************************************************
"""

def solveMultipleSamples(model, sampleData, dict_equations, dict_variables, dict_options, solv_options, sampling_option):
    """ solve samples from array sampleData in parallel and returns number of converged samples. The converged
    samples and their solutions are written into text files.

    Args:
        :model:             object of class model in modOpt.model that contains all
                            information of the NLE-evaluation and decomposition
        :sampleData:        array with samples
        :dict_equations:    dictionary with information about equations
        :dict_variables:    dictionary with information about iteration variables                            
        :dict_options:      dictionary with user specified settings
        :solv_options:      dictionary with user defined solver settings
        :sampling_options:  dictionary with sampling options

    Return:
        :converged:         integer with number of converged runs
       
    """
   
    CPU_count = solv_options["CPU count"]
    nSample = len(sampleData)
    jobs = []
    manager = Manager()
    res = manager.dict()
    done = numpy.zeros(len(sampleData))
    started = numpy.zeros(len(sampleData))
    actNum = 0  
    
    for k in range(0,nSample):  
        p = Process(target=solveMultipleSamples_Worker, args=(sampleData, k,
                                                               model,
                                                               dict_equations,
                                                               dict_variables, 
                                                               dict_options,
                                                               solv_options,
                                                               sampling_option,
                                                               res))
        jobs.append(p)
        
    startAndDeleteJobs(actNum, jobs, started, done, nSample, CPU_count)

    converged = getConvergenceResults(res, nSample)
             
    return converged


def solveMultipleSamples_Worker(sampleData, k, model, dict_equations, dict_variables, dict_options, solv_options, sampling_options, res):
    """ contains work that can be done in parallel during solving multiple samples. The package multiprocessing is used 
    for parallelization.
    
    Args:
        :model:             object of class model in modOpt.model that contains all
                            information of the NLE-evaluation and decomposition
        :sampleData:        array with samples
        :dict_equations:    dictionary with information about equations
        :dict_variables:    dictionary with information about iteration variables                            
        :dict_options:      dictionary with user specified settings
        :solv_options:      dictionary with user defined solver settings
        :sampling_options:  dictionary with sampling options
        :res:               dictionary from multiprocessing where convergence results are stored 
                            after a job is done    
    Return:                 True if method finishes ordinary
                             
    """
    
    model.stateVarValues = [sampleData[k]]
    initial_sample = copy.deepcopy(model)
    #cpModel = copy.deepcopy(model)
    
    res_solver = main.solveSystem_NLE(model, dict_equations, dict_variables, solv_options, dict_options)

    # Results:  
    if not model.failed:
        res['%d' %k] = 1
        results.writeConvergedSample(initial_sample, k, dict_options, res_solver, sampling_options)
    
    else: res['%d' %k] = 0
    
    return True

def getConvergenceResults(results, nSample):
    """ counts converged examples
    
    Args:
        :results:           dictionary from multiprocessing, where convergence results are stored
                            after a job is done
        :nSample:           integer with number of samples


    Return:
        :converged:         integer with number of converged samples
                                 
    """

    converged = 0
    for k in range(0, nSample):
        converged = converged + results['%d' %k]

    return converged


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