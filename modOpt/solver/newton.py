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


def doNewton(curBlock, solv_options, dict_options, dict_eq, dict_var):
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
        if dict_options["scaling procedure"] == 'block_iter' or dict_options["scaling procedure"] == 'block_init':
                mos.scaleSystem(curBlock, dict_eq, dict_var, dict_options) 
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
            if dict_options["scaling procedure"] == 'block_iter':
                mos.scaleSystem(curBlock, dict_eq, dict_var, dict_options) 
                
            J = curBlock.getScaledJacobian()
            F = curBlock.getScaledFunctionValues() 
                
        else:    
            curBlock.x_tot[curBlock.colPerm] = x
            J = curBlock.getPermutedJacobian()
            F = curBlock.getPermutedFunctionValues()  
            
        iterNo = iterNo + 1
        tol = numpy.linalg.norm(curBlock.getPermutedFunctionValues())
        
    if iterNo == iterMax and tol > FTOL: return 0, iterNo
    else: return 1, iterNo