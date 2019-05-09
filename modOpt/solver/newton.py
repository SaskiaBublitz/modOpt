"""
***************************************************
Import packages
***************************************************
"""
import numpy

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
        :curBlock:     object of class Block with block information
        :solv_options: dictionary with solver settings
             
    """
    
    iterNo = 0
    FTOL = solv_options["FTOL"]
    iterMax = solv_options["iterMax"]
    tol = 1e6
    if dict_options["scaling"] != 'None':
        J = curBlock.getScaledJacobian()
        F = curBlock.getScaledFunctionValues()    
        x = curBlock.getScaledIterVarValues()

    else:
        J = curBlock.getPermutedJacobian()
        F = curBlock.getPermutedFunctionValues()    
        x = curBlock.getIterVarValues()

    while tol > FTOL and iterNo < iterMax:
        dx = - numpy.dot(numpy.linalg.inv(J), F)
        x = x + dx
        if dict_options["scaling"] != 'None': 
            curBlock.x_tot[curBlock.colPerm] = x*curBlock.colSca
            J = curBlock.getScaledJacobian()
            F = curBlock.getScaledFunctionValues()             
        else:    
            curBlock.x_tot[curBlock.colPerm] = x
            J = curBlock.getPermutedJacobian()
            F = curBlock.getPermutedFunctionValues()  
            
        iterNo = iterNo + 1
        tol = numpy.linalg.norm(F)
    
    return tol, iterNo