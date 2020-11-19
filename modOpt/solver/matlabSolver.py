"""
***************************************************
Import packages
***************************************************
"""
import numpy
import matlab.engine

"""
****************************************************
Newton Solver Procedure
****************************************************
"""

def fsolve(curBlock, solv_options, dict_options, dict_equations, dict_variables):
    """  solves nonlinear algebraic equation system (NLE) by starting seperate matlab 
    function containing fsolve
    CAUTION: Additional matlab file is required
    
    Args:
        :curBlock:      object of class Block with block information
        :solv_options:  dictionary with solver settings
        :dict_equations:       dictionary with information about equations
        :dict_variables:      dictionary with information about iteration variables   
          
    """

    FTOL = solv_options["FTOL"]
    iterMax = solv_options["iterMax"]
    x0 = curBlock.getIterVarValues()
    sortedVariables = list(map(str, dict_variables.keys()))
    
    eng = matlab.engine.start_matlab()
    RES, fval, exitflag, output = eval('eng.'+dict_options["fileName"]+'(matlab.double(x0.tolist()), FTOL, iterMax, sortedVariables, nargout=4)')
    x = numpy.array(RES[0])

    if any(numpy.iscomplex(x)): x = x0
    curBlock.x_tot[curBlock.colPerm] = x

    if exitflag >= 1: return 1, output['iterations']
    else: return 0, output['iterations']
