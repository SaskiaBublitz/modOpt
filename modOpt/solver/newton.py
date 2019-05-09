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


def doNewton(curBlock, x_tot, solv_options):
    """  solves nonlinear algebraic equation system (NLE) by Newton
    Raphson procedure
    
    Args:
        :curBlock:     object of class Block with block information
        :x_tot:        current state variable array of total system
        :FTOL:         Function tolerance for solver termination
        
    """
    FTOL = solv_options["FTOL"]
    tol = 1e6
    J = curBlock.getJacobian(x_tot)
    F = curBlock.getFunctionValues(x_tot)    
    x = curBlock.getIterVarValues(x_tot)
    
    while tol > FTOL:
        dx = - numpy.dot(numpy.linalg.inv(J), F)
        x = x + dx
        x_tot[curBlock.xb_ID] = x
        J = curBlock.getJacobian(x_tot)
        F = curBlock.getFunctionValues(x_tot)  
        
        tol = numpy.linalg.norm(F)

    return True