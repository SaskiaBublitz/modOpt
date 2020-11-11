"""
***************************************************
Import packages
***************************************************
"""
import copy
import numpy
from modOpt.initialization import main
from multiprocessing import Manager, Process #cpu_count

__all__ = ['sample_box']

"""
***************************************************
Algorithm for parallelization in sampling
***************************************************
"""

def sample_box(model, sampling_options, dict_options):
    """ samples all boxes in parallel by selected method from sampling_options 
    and stores the samples with the minimum functional residual.
    
    Args:
        :model:             instance of class model
        :sampling_options:  dictionary with number of samples to generate and 
                            sampleNo_min_residual which is the number of candidates
                            with lowest residuals that are stored
        :dict_options:      dictionary with user-settings regarding parallelization
                            and box reduction steps applied before

    Returns:                dictionary for storage of samples with minimum residual
                            for all boxes
    """
    
    CPU_count = dict_options["CPU count"]
    jobs = []
    manager = Manager()
    res = manager.dict()
    done = numpy.zeros(len(model.xBounds))
    started = numpy.zeros(len(model.xBounds))
    actNum = 0 
    
    for boxID in range(0,len(model.xBounds)):  
        p = Process(target=main.sample_box, args=(model, boxID, 
                                                  sampling_options, dict_options, res))
        jobs.append(p)
        
    startAndDeleteJobs(actNum, jobs, started, done, len(model.xBounds), CPU_count)
    return res


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