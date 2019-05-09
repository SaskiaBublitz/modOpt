""" Solver """

"""
***************************************************
Import packages
***************************************************
"""
import newton
import numpy
import block
"""
****************************************************
Main that starts solving procedure in decomposed NLE
****************************************************
"""
__all__ = ['solveSystem_NLE']


def solveSystem_NLE(model, solv_options, dict_options):
    """ solve nonlinear algebraic equation system (NLE)
    
    Args:
        :model:         object of class model in modOpt.model that contains all
                        information of the NLE-evaluation and decomposition
        :solv_options:  dictionary with user defined solver settings
        :dict_options:  dictionary with user specified settings
     
    Return:     updated model and equation system residual measured by the
                Euclydian norm of all function residuals after the solver terminated
                
    """
            
    if dict_options["decomp"] == 'DM' or dict_options["decomp"] == 'None': 
        res_solver = solveBlocksSequence(model, solv_options, dict_options)
        return res_solver
    
    if dict_options["decomp"] == 'BBTF':
        #TODO: Add Nested procuedure
        return []
    
   

def solveBlocksSequence(model, solv_options, dict_options):
    """ solve block decomposed system in sequence. This method can also be used
    if no decomposition is done. In this case the system contains one block that 
    equals the entire system.

    Args:
        :model:         object of class model in modOpt.model that contains all
                        information of the NLE-evaluation and decomposition
        :solv_options:  dictionary with user defined solver settings
        :dict_options:  dictionary with user specified settings
 
    Return:
        :res_solver:    dictionary with solver results
       
    """
    
    res_solver = {}
    rBlocks, cBlocks = getListsWithBlockMembersByGlbID(model)
    xInF = getListWithFunctionMembersByGlbID(model)   
    
    x = model.stateVarValues[0]

    for b in range(len(rBlocks)):

        curBlock = block.Block(rBlocks[b], cBlocks[b], xInF, model.jacobian, model.fSymCasadi)
        
        if solv_options["solver"] == 'newton':
            newton.doNewton(curBlock, x, solv_options)
        # TODO: Add other solvers, e.g. ipopt
            
    model.stateVarValues[0] = x
    
    res_solver["Model"] = model
    res_solver["Residual"] = model.getFunctionValuesResidual()

    return res_solver 


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
    blocksPerm = model.blocks #for permuted matrix
    
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
     
     