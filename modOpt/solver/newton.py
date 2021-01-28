"""
***************************************************
Import packages
***************************************************
"""
import numpy
import modOpt.scaling as mos

"""
****************************************************
Newton Solver Procedure
****************************************************
"""
__all__ = ['doNewton']

def doNewton(curBlock, solv_options, dict_options):
    """  solves nonlinear algebraic equation system (NLE) by Newton
    Raphson procedure
    
    Args:
        :curBlock:      object of class Block with block information
        :solv_options:  dictionary with solver settings
          
    """
    
    iterNo = 0
    FTOL = solv_options["FTOL"]
    iterMax = solv_options["iterMax"]
    tol = numpy.linalg.norm(curBlock.getScaledFunctionValues())
    if numpy.isnan(tol): return -1, iterNo # nan

    while not tol <= FTOL and iterNo < iterMax:
        J, x, F = getLinearSystem(dict_options, curBlock)
        dx = - numpy.dot(numpy.linalg.inv(J), F)
        x = x + dx
        
        updateIterVars(dict_options, curBlock, x)
        scaleBlockInIteration(dict_options, curBlock)
        
        iterNo = iterNo + 1
        tol = numpy.linalg.norm(curBlock.getScaledFunctionValues())
        if numpy.isnan(tol): return -1, iterNo
        
    if iterNo == iterMax and tol > FTOL: return 0, iterNo

    else: return 1, iterNo

 
def getLinearSystem(dict_options, curBlock):
    
    if dict_options["scaling"] != 'None':
        J = curBlock.getScaledJacobian()
        F = curBlock.getScaledFunctionValues()    
        x = curBlock.getScaledIterVarValues()

    else:
        J = curBlock.getPermutedJacobian()
        F = curBlock.getPermutedFunctionValues()    
        x = curBlock.getIterVarValues() 
    
    return J, x, F
 
    
def updateIterVars(dict_options, curBlock, x): 
    """ update iteration variables in newton procedure
    
    Args:
        :dict_options:          dictionary with user specified settings
        :curBlock:              instance of class Block
        :x:                     iteration variable values after Newton step
        
    """
    
    if dict_options["scaling"] != 'None': 
        curBlock.x_tot[curBlock.colPerm] = x*curBlock.colSca
    else:
        curBlock.x_tot[curBlock.colPerm] = x

    
def scaleBlockInIteration(dict_options, curBlock):
    """ if chosen, scales block during iteration
    
    Args:
        :dict_options:          dictionary with user specified settings
        :curBlock:              instance of class Block          
    """    
    
    if dict_options["scaling"] != 'None' and dict_options["scaling procedure"] == 'block_iter':
                mos.scaleSystem(curBlock, dict_options) 
