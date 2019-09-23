""" Solver """

"""
***************************************************
Import packages
***************************************************
"""
import newton
import numpy
import block
import modOpt.scaling as mos
"""
****************************************************
Main that starts solving procedure in decomposed NLE
****************************************************
"""
__all__ = ['solveSystem_NLE']


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
        res_solver = solveBlocksSequence(model, solv_options, dict_options, dict_equations, dict_variables)
        updateToSolver(res_solver, dict_equations, dict_variables)
        return res_solver
    
    if dict_options["decomp"] == 'BBTF':
        #TODO: Add Nested procuedure
        return []
    
    
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
    
    
def solveBlocksSequence(model, solv_options, dict_options, dict_equations, dict_variables):
    """ solve block decomposed system in sequence. This method can also be used
    if no decomposition is done. In this case the system contains one block that 
    equals the entire system.

    Args:
        :model:             object of class model in modOpt.model that contains all
                            information of the NLE-evaluation and decomposition
        :solv_options:      dictionary with user defined solver settings
        :dict_options:      dictionary with user specified settings
        :dict_equations:    dictionary with information about equations
        :dict_variables:    dictionary with information about iteration variables
 
    Return:
        :res_solver:    dictionary with solver results
       
    """
    # Initialization of known system quantities
    rBlocks, cBlocks, xInF = getBlockInformation(model)
    res_solver = createSolverResultDictionary(len(rBlocks))
    
    # Block iteration:    
    for b in range(len(rBlocks)): 

        curBlock = block.Block(rBlocks[b], cBlocks[b], xInF, model.jacobian, 
                               model.fSymCasadi, model.stateVarValues[0])
        
        if dict_options["scaling"] != 'None': getInitialScaling(dict_options, 
                       model, curBlock, dict_equations, dict_variables)
            
        if solv_options["solver"] == 'newton': 
            doNewton(curBlock, b, solv_options, dict_options, res_solver, 
                     dict_equations, dict_variables)
             
        # TODO: Add other solvers, e.g. ipopt

        if res_solver["Exitflag"][b] < 1: 
            model.failed = True
    # Write Results:         
    putResultsInDict(curBlock.x_tot, model, res_solver)

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
    
    return res_solver

  
def getInitialScaling(dict_options, model, curBlock, dict_equations, dict_variables):
    """solve block decomposed system in sequence. This method can also be used
    if no decomposition is done. In this case the system contains one block that 
    equals the entire system.

    Args:
        :dict_options:      dictionary with user specified settings        
        :model:             object of class model in modOpt.model that contains all
                            information of the NLE-evaluation and decomposition
        :curBlock:          object of type Block
        :dict_equations:    dictionary with information about equations
        :dict_variables:    dictionary with information about iteration variables
        
    """
    
    if dict_options["scaling procedure"] =='tot_init': # scaling of complete decomposed system
            curBlock.setScaling(model)
        
    elif dict_options["scaling procedure"] =='block_init': # blockwise scaling
            mos.scaleSystem(curBlock, dict_equations, dict_variables, dict_options)   


def doNewton(curBlock, b, solv_options, dict_options, res_solver, dict_equations, dict_variables):
    """ starts Newton-Raphson Method
    
    Args:
        :model:             object of type Model
        :curBlock:          object of type Block
        :b:                 current block index
        :solv_options:      dictionary with user specified solver settings
        :dict_options:      dictionary with user specified structure settings
        :res_solver:        dictionary with results from solver
        :dict_equations:    dictionary with information about equations
        :dict_variables:    dictionary with information about iteration variables
        
    """
    
    try: 
        exitflag, iterNo = newton.doNewton(curBlock, solv_options, dict_options, dict_equations, dict_variables)
        res_solver["IterNo"][b] = iterNo
        res_solver["Exitflag"][b] = exitflag
        
    except: 
        print "Error in Block ", b
        res_solver["IterNo"][b] = 0
        res_solver["Exitflag"][b] = -1    
    

def putResultsInDict(x, model, res_solver):
    """ updates model and results dictionary
    Args:
        :x:               numpy array with final iteration variable values
        :model:           object of type Model
        :res_solver:      dictionary with results from solver    
        
    """
    
    model.stateVarValues[0] = x
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
     
     