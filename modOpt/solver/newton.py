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

def doNewton(curBlock, solv_options, num_options):
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
    print("Squared function residuals of Newton:")
    while not tol <= FTOL and iterNo < iterMax:
        J, x, F = getLinearSystem(num_options, curBlock)
        #dx = - numpy.dot(numpy.linalg.inv(J), F)
        try:
            dx = - numpy.linalg.solve(J,F)
        except: 
            print("Could not compute Newton step")
            return -1, iterNo
        x = x + dx
        
        updateIterVars(num_options, curBlock, x)
        scaleBlockInIteration(num_options, curBlock)
        
        iterNo = iterNo + 1
        tol = numpy.linalg.norm(curBlock.getScaledFunctionValues())
        print(tol)
        if numpy.isnan(tol): return -1, iterNo
        
    if iterNo == iterMax and tol > FTOL: 
        print("Maximum number of iteration steps reached.")
        return 0, iterNo

    else: 
        print("Required tolerance is satisfied.")
        return 1, iterNo

 
def getLinearSystem(num_options, curBlock):
    
    if num_options["scaling"] != 'None':
        J = curBlock.getScaledJacobian()
        F = curBlock.getScaledFunctionValues()    
        x = curBlock.getScaledIterVarValues()

    else:
        J = curBlock.getPermutedJacobian()
        F = curBlock.getPermutedFunctionValues()    
        x = curBlock.getIterVarValues() 
    
    return J, x, F
 
    
def updateIterVars(num_options, curBlock, x): 
    """ update iteration variables in newton procedure
    
    Args:
        :num_options:          dictionary with user specified settings
        :curBlock:              instance of class Block
        :x:                     iteration variable values after Newton step
        
    """
    
    if num_options["scaling"] != 'None': 
        curBlock.x_tot[curBlock.colPerm] = x*curBlock.colSca
    else:
        curBlock.x_tot[curBlock.colPerm] = x

    
def scaleBlockInIteration(num_options, curBlock):
    """ if chosen, scales block during iteration
    
    Args:
        :num_options:          dictionary with user specified settings
        :curBlock:              instance of class Block          
    """    
    
    if num_options["scaling"] != 'None' and num_options["scalingProcedure"] == 'block_iter':
                mos.scaleSystem(curBlock, num_options) 
