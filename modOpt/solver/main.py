""" Solver """

"""
***************************************************
Import packages
***************************************************
"""
import numpy
from modOpt.solver import (newton, results, scipyMinimization, 
parallelization, block, matlabSolver)
import modOpt.storage as mostg
import modOpt.scaling as mos
import modOpt.initialization as moi
import copy
import time

"""
****************************************************
Main that starts solving procedure in NLE's
****************************************************
"""
__all__ = ['solveSamples', 'solveSystem_NLE', 'solveBoxes', 'solveBlocksSequence']


def solveBoxes(model, dict_variables, dict_equations, dict_options, solv_options,sampling_options=None):
    """ solves multiple samples in boxes related to an equation system
    
    Args:
        :model:             instance of class model
        :dict_variables:    dictionary with variable related quantities
        :dict_equations:    dictionary with function related quantities
        :dict_options:      dictionary with user-settings for decompositon 
                            and scaling
        :solv_options:      dictionary with user-settings for solver                    

    Returns:                None
    
    """
    dict_options["sampling"] == True
    boxes = mostg.get_entry_from_npz_dict(dict_options["BoxReduction_fileName"], dict_options["redStep"]) 
    
    mainfilename = dict_options["fileName"] 
    npzName = dict_options["fileName"]
    for cur_solver in solv_options["solver"]:
        npzName+=cur_solver
    npzName+=".npz"
    dict_options["mainfileName"] = mainfilename  
    dict_options["npzName"] = npzName  
        
    if solv_options["parallel_boxes"]:  #TODO: needs to be adjusted
        res = parallelization.solveBoxesParallel(model, boxes, dict_variables, dict_equations, 
                    dict_options, solv_options) 
    
    else:
        res = {}
        if "boxes" in dict_options.keys():
            for l in dict_options["boxes"]:
                res[l]  = solveOneBox(model, boxes, l, dict_variables, dict_equations, 
                        dict_options, solv_options, sampling_options)                
        else:
            for l in range(0, len(boxes)):
                res[l]  = solveOneBox(model, boxes, l, dict_variables, dict_equations, 
                        dict_options, solv_options, sampling_options)
    results.write_successful_results(res, dict_options, sampling_options, solv_options)           
    #store_results(res, boxes, dict_options)


def store_results(res, boxes, dict_options):
    if "sampling" in dict_options.keys():
        mostg.store_list_in_npz_dict(dict_options["npzName"], results, 0, allow_pickle=True)
        return
    if "boxes" in dict_options.keys():#isinstance(dict_options["boxes"],list):
        boxNo = len(dict_options["boxes"])
    else:
        boxNo = len(boxes)
    for l in range(0, boxNo):
        if "boxes" in dict_options.keys(): k = dict_options["boxes"][l]
        else: k = l
        mostg.store_list_in_npz_dict(dict_options["npzName"], results['%d'%k], l, allow_pickle=True)


def solveOneBox(model, boxes, l, dict_variables, dict_equations, dict_options, solv_options, sampling_options=None):
    model.xBounds = [boxes[l]]
    dict_options["box_ID"] = l
 
    #else:
    #    initValues = mostg.get_entry_from_npz_dict(dict_options["Sampling_fileName"], l)
    #    results['%d'%l]  = solveSamples(model, initValues, 
    #                               dict_equations, 
    #                               dict_variables, solv_options, dict_options)

    return solveBlocksSequence(model, solv_options, dict_options, sampling_options)


def sample_multiple_solutions(model, b, rBlocks, cBlocks, xInF, sampling_options, dict_options, solv_options, res_blocks):
    """ samples and solves a model that contains multiple solutions in the already solved blocks.
    Hence, each solution needs to be sampled and solved for the current block separately.
    
    Args:
        :model:             instance of class model
        :b:                 integer with block index
        :rBlocks:           list with permuation index for rows 
        :cBlocks:           list with permuation index for columns 
        :xInf:              nested list with functions in global order that contains 
                            global ID's of all variables that occur in this function
        :sampling_options:  user-specified dictionary with sampling settings
        :dict_options:      user-specified dictionary with absolute tolerance value
        :solv_options:      user-specified dictionary with solver settings
        :res_blocks:        dictionary for all block iteration results
    
    Return:
        :res_blocks:    updated dictionary by current iteration results
        :solved:        boolean variable, which is true if solution was found

    """

    newSubSolutions = []
    
    subSolutions = list(model.stateVarValues)
    for x in subSolutions:
        model.stateVarValues= [x]
        solved, res_blocks = sample_and_solve_one_block(model, b, rBlocks, cBlocks, xInF, 
                                            sampling_options, dict_options, solv_options,
                                            res_blocks)
        if solved: 
            newSubSolutions.append(*model.stateVarValues)
        
    if not newSubSolutions == []: model.stateVarValues = newSubSolutions
    else: model.failed == True
    return res_blocks


def sample_and_solve_one_block(model, b, rBlocks, cBlocks, xInF, sampling_options, dict_options, solv_options, res_blocks):
    """ samples and solves one block for at a certain point of already solved variables.
    Note: There can be multiple solutions in a former block, so that this functions needs to
    be called for each of them. There is only one box used for sampling therefore the boxID
    is manually set to 0 in sample_box_in_block.
    
    Args:
        :model:             instance of class model
        :b:                 integer with block index
        :rBlocks:           list with permuation index for rows 
        :cBlocks:           list with permuation index for columns 
        :xInf:              nested list with functions in global order that contains 
                            global ID's of all variables that occur in this function
        :sampling_options:  user-specified dictionary with sampling settings
        :dict_options:      user-specified dictionary with absolute tolerance value
        :solv_options:      user-specified dictionary with solver settings
        :res_blocks:        dictionary for all block iteration results
    
    Return:
        :res_blocks:    updated dictionary by current iteration results
        :solved:        boolean variable, which is true if solution was found

    """

    samples = {}    
    cur_block = block.Block(rBlocks[b], cBlocks[b], xInF, model.jacobian, 
               model.fSymCasadi, model.stateVarValues[0], 
               model.xBounds[0], model.parameter,
               model.constraints, model.xSymbolic, model.jacobianSympy
               )
    
    moi.sample_box_in_block(cur_block, 0, sampling_options, dict_options, samples)
    
    return solve_samples_block(model, cur_block, b, samples[0], solv_options, 
                                                               dict_options,
                                                               res_blocks)    


def solve_samples_block(model, block, b, samples, solv_options, dict_options, res_blocks):
    """ iterates generated samples in a block 
    
    Args:
        :model:         instance of class model
        :block:         instance of class block
        :b:             integer with block index
        :samples:       list with samples of current block
        :dict_options:  user-specified dictionary with absolute tolerance value
        :res_blocks:    dictionary for all block iteration results
    
    Return:
        :res_blocks:    updated dictionary by current iteration results
        :solved:        boolean variable, which is true if solution was found

    """
    solved = False

    for sample in samples:
        block.x_tot[block.colPerm] = sample
        
        if dict_options["scaling"] != 'None': getInitialScaling(dict_options, 
                                model, block)
        
        res_solver = startSolver(block, b, solv_options, dict_options)  
        sort_results_from_solver_to_block(block, b, res_solver, res_blocks, dict_options)
        
    if not block.FoundSolutions == []: 
        res_blocks[b]["colPerm"] = block.colPerm
        res_blocks[b]["rowPerm"] = block.rowPerm
        update_model_to_block_results(block, model)
        solved = True
    else:
        #model.stateVarValues[0] = block.x_tot 
        #res_solver[b] = res_solver # stores failed system, it has no key "FoundSolutions"
        solved = False
        
    return res_blocks, solved


def update_model_to_block_results(block, model):
    """ updates the model to all found solutions in the current block.
    
    Args:
        :block:         instance of class block
        :model:         instance of class model

    """
    model.stateVarValues = [model.stateVarValues[0]] * len(block.FoundSolutions)
    for s in range(0, len(block.FoundSolutions)):
        model.stateVarValues[s][block.colPerm] = block.FoundSolutions[s]    

        
def sort_results_from_solver_to_block(block, b, res_solver, res_blocks, dict_options):
    """ sorts results from iteration of one sample in one block to dictionary 
    for all blocks results
    
    Args:
        :block:         instance of class block
        :b:             integer with block index
        :res_solver:    dictionary with solver output from current sample
        :res_blocks:    dictionary for all block iteration results
        :dict_options:  user-specified dictionary with absolute tolerance value

    """
    
    if res_solver["Exitflag"]==1 and solutionInBounds(block):
        solution_id = update_block_solutions(block, dict_options)
        
        if len(block.FoundSolutions) == solution_id + 1: # condition that new solution for block was found
            if not b in res_blocks.keys(): # first solution for block
                res_blocks[b] = {}
                for key in res_solver.keys():
                    res_blocks[b][key]=[res_solver[key]]
                
            else: # >1 solution for block 
                for key in res_solver.keys():
                    res_blocks[b][key].append(res_solver[key])
            res_blocks[b]["FoundSolutions"] = block.FoundSolutions
            
        else: # already found solution, if less iterations were applied this time it is updated in res_blocks
           res_blocks[b]["IterNo"][solution_id] = min(res_blocks[b]["IterNo"][solution_id],
                                                      res_solver["IterNo"]) 

def solutionInBounds(block):
    """ checks if all variable solutions are within their bounds
    
    Args:
        :block:     instance of class block
        
    Return:         Boolean = True if in bounds, = False if out of bounds

    """
    
    for i in range(0, len(block.colPerm)):
        bounds = block.xBounds_tot[block.colPerm][i]
        x = block.x_tot[block.colPerm][i]
        if isinstance(bounds, list) or isinstance(bounds, numpy.ndarray):
            if x< bounds[0] or x>bounds[1] :
                return False
        else:  
            if not block.x_tot[block.colPerm][i] in block.xBounds_tot[block.colPerm][i]:
                return False
    return True


def update_block_solutions(block, dict_options):
    """ updates found solutions of block, only stores new solutions and returns
    index of found block solution.
    
    Args:
        :block:         instance of class block
        :dict_options:  user-specified dictionary with absolute tolerance value
        
    Return: id of the currently found solution in block
    
    """  
    i = 0
    if not block.FoundSolutions == []:
        for solution in block.FoundSolutions:
            if numpy.allclose(block.x_tot[block.colPerm], solution, atol=dict_options["absTol"]):
                return i 
            i+=1
    block.FoundSolutions.append(block.x_tot[block.colPerm])
    
    return len(block.FoundSolutions) - 1 


def solveSamples(model, initValues, dict_equations, dict_variables, solv_options, dict_options):
    """ starts iteration from samples in a certain box with index l

    Args:
        :model:             object of class model in modOpt.model that contains all
                            information of the NLE-evaluation and decomposition
        :initValues:        numpy array with sample values
        :mainfilename:      string with general file name
        :l:                 integer with index of current box
        :dict_equations:    dictionary with information about equations
        :dict_variables:    dictionary with information about iteration variables                            
        :solv_options:      dictionary with user defined solver settings
        :dict_options:      dictionary with user specified settings

    Returns:                 None.
       
    """
    res_multistart = {}
                
    for k in range(0, len(initValues)): # TODO: Parallelization
        
        dict_options["sample_ID"] = k
        model.stateVarValues = [initValues[k]]    

        res_multistart['%d' %k] = solveSystem_NLE(model, dict_equations, dict_variables, solv_options, dict_options)

    return res_multistart

      
def solveSystem_NLE(model, dict_equations, dict_variables, solv_options, dict_options):
    """ solve nonlinear algebraic equation system (NLE)
    
    Args:
        :model:                 object of class model in modOpt.model that contains all
                                information of the NLE-evaluation and decomposition
        :dict_equations:        dictionary with information about equations
        :dict_variables:        dichtionary with information about variables                                
        :solv_options:          dictionary with user defined solver settings
        :dict_options:          dictionary with user specified settings
     
    Return:     
        :res_solver:            dictionary with solver results
                
    """
    
    if dict_options["scaling"] != 'None' and dict_options["scaling procedure"] == 'tot_init':
        mos.scaleSystem(model, dict_equations, dict_variables, dict_options) 

            
    if dict_options["decomp"] == 'DM' or dict_options["decomp"] == 'None': 
        res_solver = solveBlocksSequence(model, solv_options, dict_options)
        updateToSolver(res_solver, dict_equations, dict_variables)
        return res_solver
    
    if dict_options["decomp"] == 'BBTF':
        i=0
        res_solver=[]

        while not numpy.linalg.norm(model.getFunctionValues()) <= solv_options["FTOL"] and i <= solv_options["iterMax_tear"]:
            res_solver = solveBlocksSequence(model, solv_options, dict_options)
            updateToSolver(res_solver, dict_equations, dict_variables)
            model = res_solver["Model"] 
            print (numpy.linalg.norm(model.getFunctionValues())) 
            i+=1
        return res_solver
    
    
def updateToSolver(res_solver, dict_equations, dict_variables):
    """ updates dictionaries to solver results
    
    Args:
        :res_solver:            dictionary with solver results  
        :dict_equations:        dictionary with information about equations
        :dict_variables:        dichtionary with information about variables          
    
    """

    model = res_solver["Model"]     
    fVal = model.getFunctionValues()
    
    
    for i in range(0,len(model.xSymbolic)):
        dict_equations[model.fSymbolic[i]][0] = fVal[i]
        dict_equations[model.fSymbolic[i]][3] = model.rowSca[i]
        
        dict_variables[model.xSymbolic[i]][0] = model.stateVarValues[0][i]
        dict_variables[model.xSymbolic[i]][3] = model.colSca[i]
    
        
def solveBlocksSequence(model, solv_options, dict_options,
                        sampling_options=None):
    """ solve block decomposed system in sequence. This method can also be used
    if no decomposition is done. In this case the system contains one block that 
    equals the entire system.

    Args:
        :model:             object of class model in modOpt.model that contains all
                            information of the NLE-evaluation and decomposition
        :solv_options:      dictionary with user defined solver settings
        :dict_options:      dictionary with user specified settings
 
    Return:
        :res_solver:    dictionary with solver results
       
    """
    # Initialization of known system quantities

    rBlocks, cBlocks, xInF = getBlockInformation(model)

    #res_solver = createSolverResultDictionary(len(rBlocks))
    res_blocks = {}
    # Block iteration:    
    for b in range(len(rBlocks)): 
        if "sampling" in dict_options.keys():
            if len(model.stateVarValues) > 1: 
                res_blocks = sample_multiple_solutions(model, b, rBlocks, cBlocks, 
                                                       xInF, sampling_options, 
                                                       dict_options, solv_options, 
                                                       res_blocks)
            else:      
                res_blocks, solved = sample_and_solve_one_block(model, b, rBlocks, 
                                                                cBlocks, xInF, 
                                                                sampling_options,
                                                                dict_options,
                                                                solv_options, 
                                                                res_blocks)
                
                if not solved: 
                    model.failed = True
                    res_blocks["Model"] = model  
                    return res_blocks
                                
        else:
            curBlock = block.Block(rBlocks[b], cBlocks[b], xInF, model.jacobian, 
                    model.fSymCasadi, model.stateVarValues[0], 
                    model.xBounds[0], model.parameter,
                    model.constraints, model.xSymbolic, model.jacobianSympy
                               )            
            res_blocks[b] = startSolver(curBlock, b, solv_options, dict_options)
        #if isinstance(solv_options["solver"], list):
        #   TODO alternating solvers
        # TODO: Add other solvers, e.g. ipopt
        
            if res_blocks[b]["Exitflag"] < 1: 
                model.failed = True
            
            # Update model    
            model.stateVarValues[0] = curBlock.x_tot
    
    # Write Results:
    res_blocks["Model"] = model        
    #putResultsInDict(model, res_solver)

    return res_blocks 


def startSolver(curBlock, b, solv_options, dict_options):
    res_solver = {}
    if isinstance(solv_options["solver"],list):
        if len(solv_options["solver"])==1: solv_options["solver"] = solv_options["solver"][0]
        # else: TODO alternating algorithm for solver list
        
    if solv_options["solver"] == 'newton': 
        doNewton(curBlock, b, solv_options, dict_options, res_solver)
    
    if solv_options["solver"] == "fsolve":
        doScipyOptimize(curBlock, b, solv_options, dict_options, res_solver)
    
    if solv_options["solver"] in ['SLSQP', 'trust-constr', 'TNC']:
        doScipyOptiMinimize(curBlock, b, solv_options, dict_options, res_solver)
        
    if solv_options["solver"] == 'ipopt':
        doipoptMinimize(curBlock, b, solv_options, dict_options, res_solver)

    if solv_options["solver"] == 'matlab-fsolve':
        doMatlabSolver(curBlock, b, solv_options, dict_options, res_solver)

    if solv_options["solver"] == 'matlab-fsolve-mscript':
        doMatlabSolver_mscript(curBlock, b, solv_options, dict_options, res_solver) 
        
    return res_solver           
        

def getBlockInformation(model):
    """ collects information available from block decomposition
    
    Args:
        :model:         object of class model in modOpt.model that contains all

    Return:
        :rBlocks:      Nested list with blocks that contain global ID's of the 
                       functions
        :rBlocks:      Nested list with blocks that contain global ID's of the 
                       iteration variables
        :xInf:         Nested list with functions in global order that contains 
                       global ID's of all variables that occur in this function
                        
    """
    
    rBlocks, cBlocks = getListsWithBlockMembersByGlbID(model)
    xInF = getListWithFunctionMembersByGlbID(model) 
    return rBlocks, cBlocks, xInF 
    

def createSolverResultDictionary(blockNo):
    """ sets up dictionary for solver results
    
    Args:
        :blockNo:   Number of blocks
        
    Return:         initial dictionary
        
    """

    res_solver = {}
    res_solver["IterNo"] = numpy.zeros(blockNo)
    res_solver["Exitflag"] = numpy.zeros(blockNo)
    res_solver["CondNo"] = numpy.zeros(blockNo)
    
    return res_solver

  
def getInitialScaling(dict_options, model, curBlock):
    """solve block decomposed system in sequence. This method can also be used
    if no decomposition is done. In this case the system contains one block that 
    equals the entire system.

    Args:
        :dict_options:      dictionary with user specified settings        
        :model:             object of class model in modOpt.model that contains all
                            information of the NLE-evaluation and decomposition
        :curBlock:          object of type Block
        
    """
    
    if dict_options["scaling procedure"] =='tot_init': # scaling of complete decomposed system
            curBlock.setScaling(model)
        
    elif dict_options["scaling procedure"] =='block_init' or dict_options["scaling procedure"] =='block_iter': # blockwise scaling
            mos.scaleSystem(curBlock, dict_options)   


def doNewton(curBlock, b, solv_options, dict_options, res_solver):
    """ starts Newton-Raphson Method
    
    Args:
        :model:             object of type Model
        :curBlock:          object of type Block
        :b:                 current block index
        :solv_options:      dictionary with user specified solver settings
        :dict_options:      dictionary with user specified structure settings
        :res_solver:        dictionary with results from solver
       
    """
    
    try: 
        exitflag, iterNo = newton.doNewton(curBlock, solv_options, dict_options)
        res_solver["IterNo"] = iterNo
        res_solver["Exitflag"] = exitflag
        res_solver["CondNo"] = numpy.linalg.cond(curBlock.getScaledJacobian())
        res_solver["FRES"] = curBlock.getFunctionValues()    
        
    except: 
        print ("Error in Block ", b)
        res_solver["IterNo"] = 0
        res_solver["Exitflag"] = -1 
        res_solver["CondNo"] = 'nan'
        res_solver["FRES"] = 'nan'   
        

def doScipyOptimize(curBlock, b, solv_options, dict_options, res_solver):
    """ starts Newton-Raphson Method
    
    Args:
        :model:             object of type Model
        :curBlock:          object of type Block
        :b:                 current block index
        :solv_options:      dictionary with user specified solver settings
        :dict_options:      dictionary with user specified structure settings
        :res_solver:        dictionary with results from solver
        
    """
    
    try: 
        exitflag, iterNo = scipyMinimization.fsolve(curBlock, solv_options, 
                                                    dict_options)
        res_solver["IterNo"] = iterNo
        res_solver["Exitflag"] = exitflag
        res_solver["CondNo"] = numpy.linalg.cond(curBlock.getScaledJacobian())
        res_solver["FRES"] = curBlock.getFunctionValues()
        
    except: 
        print ("Error in Block ", b)
        res_solver["IterNo"] = 0
        res_solver["Exitflag"] = -1 
        res_solver["CondNo"] = 'nan'
        res_solver["FRES"] = 'nan'    
    
        
def doScipyOptiMinimize(curBlock, b, solv_options, dict_options, res_solver):
    """ starts minimization procedure from scipy
    
    Args:
        :curBlock:          object of type Block
        :b:                 current block index
        :solv_options:      dictionary with user specified solver settings
        :dict_options:      dictionary with user specified structure settings
        :res_solver:        dictionary with results from solver
        
    """
    
    try: 
        exitflag, iterNo = scipyMinimization.minimize(curBlock, solv_options, dict_options)
        res_solver["IterNo"] = iterNo
        res_solver["Exitflag"] = exitflag
        res_solver["CondNo"] = numpy.linalg.cond(curBlock.getScaledJacobian())
        res_solver["FRES"] = curBlock.getFunctionValues()
        
    except: 
        print ("Error in Block ", b)
        res_solver["IterNo"] = 0
        res_solver["Exitflag"] = -1 
        res_solver["CondNo"] = 'nan'
        res_solver["FRES"] = 'nan'    
       
    
def doipoptMinimize(curBlock, b, solv_options, dict_options, res_solver):
    """ starts minimization procedure from scipy
    
    Args:
        :curBlock:          object of type Block
        :b:                 current block index
        :solv_options:      dictionary with user specified solver settings
        :dict_options:      dictionary with user specified structure settings
        :res_solver:        dictionary with results from solver
        
    """
    
    try:
        from modOpt.solver import ipoptMinimization
        exitflag, iterNo = ipoptMinimization.minimize(curBlock, solv_options, dict_options)
        res_solver["IterNo"] = iterNo-1
        res_solver["Exitflag"] = exitflag
        res_solver["CondNo"] = numpy.linalg.cond(curBlock.getScaledJacobian())
        res_solver["FRES"] = curBlock.getFunctionValues()
    except: 
        print ("Error in Block ", b)
        res_solver["IterNo"] = 0
        res_solver["Exitflag"] = -1
        res_solver["CondNo"] = 'nan'
        res_solver["FRES"] = 'nan'

    


def doMatlabSolver_mscript(curBlock, b, solv_options, dict_options, res_solver):
    """ starts matlab runner for externam matlab file with fsolve
    CAUTION: Additional matlab file is required
    
    Args:
        :curBlock:          object of type Block
        :b:                 current block index
        :solv_options:      dictionary with user specified solver settings
        :dict_options:      dictionary with user specified structure settings
        :res_solver:        dictionary with results from solver
        
    """

    try:
        exitflag, iterNo = matlabSolver.fsolve_mscript(curBlock, solv_options, dict_options)
        res_solver["IterNo"] = iterNo-1
        res_solver["Exitflag"] = exitflag
        if exitflag == 1:
            res_solver["CondNo"] = numpy.linalg.cond(curBlock.getScaledJacobian())
            res_solver["FRES"] = curBlock.getFunctionValues()
        else: 
            res_solver["CondNo"] = 'nan'
            res_solver["FRES"] = 'nan'        
    except: 
        print ("Error in Block ", b)
        res_solver["IterNo"] = 0
        res_solver["Exitflag"] = -1
        res_solver["CondNo"] = 'nan'
        res_solver["FRES"] = 'nan'
    

def doMatlabSolver(curBlock, b, solv_options, dict_options, res_solver):
    """ starts matlab runner for externam matlab file with fsolve
    CAUTION: Additional matlab file is required
    
    Args:
        :curBlock:          object of type Block
        :b:                 current block index
        :solv_options:      dictionary with user specified solver settings
        :dict_options:      dictionary with user specified structure settings
        :res_solver:        dictionary with results from solver
        
    """

    try:
        exitflag, iterNo = matlabSolver.fsolve(curBlock, solv_options, dict_options)
        res_solver["IterNo"] = iterNo-1
        res_solver["Exitflag"] = exitflag
        res_solver["CondNo"] = numpy.linalg.cond(curBlock.getScaledJacobian())
        res_solver["FRES"] = curBlock.getFunctionValues()
    except: 
        print ("Error in Block ", b)
        res_solver["IterNo"] = 0
        res_solver["Exitflag"] = -1
        res_solver["CondNo"] = 'nan'
        res_solver["FRES"] = 'nan'
    
    
def putResultsInDict(model, res_solver):
    """ updates model and results dictionary
    Args:
        :model:           object of type Model
        :res_solver:      dictionary with results from solver    
        
    """
    res_solver["Model"] = model
    res_solver["Residual"] = model.getFunctionValuesResidual()
    res_solver["IterNo_tot"] = sum(res_solver["IterNo"])


def getListsWithBlockMembersByGlbID(model):
    """ get indices of equations and variables of all blocks by their global ID
    
    Args:
        :model:         object of class model in modOpt.model that contains all
                        information of the NLE-evaluation and decomposition
     
    Return:
        :rBlock:        list with blocks containing global equation ID's, formate
                        [[[a,d]], [e,b,c], ...]
        :cBlock:        list with blocks containing global variable ID's, formate
                        [[[a,d]], [e,b,c], ...]         
                        
    """      
    
    rowPerm = model.rowPerm
    colPerm = model.colPerm
    blocksPerm = model.blocks
    
    rBlocks = []
    cBlocks =[]
    
    for curBlock in blocksPerm:
        oneRblock = []
        oneCblock = []
        for i in curBlock:
            
            oneRblock.append(rowPerm[i])
            oneCblock.append(colPerm[i])
    
        rBlocks.append(oneRblock)
        cBlocks.append(oneCblock)
    
    return rBlocks, cBlocks
        
    
def getListWithFunctionMembersByGlbID(model):
    """ creates list that contains all state variables of one block (also tearing
    variables)
    
    Args:
        :model:         object of class model in modOpt.model that contains all
                        information of the NLE-evaluation and decomposition
    
    Return:             nested list with all state variables of all blocks
    
    """
    
    nz = numpy.nonzero(model.getJacobian())
    xInF = []

     
    for i in range(0, len(model.fSymbolic)):
         xInF.append([])
         
    for i in range(0, len(nz[0])):
        
        row = nz[0][i]
        col = nz[1][i]
        
        xInF[row].append(col)
    
    return xInF
     
     