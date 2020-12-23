"""
***************************************************
Import packages
***************************************************
"""
import pathlib
import numpy
import matlab.engine
from modOpt.solver import block
import sympy
from sympy.parsing.sympy_parser import parse_expr
"""
****************************************************
Newton Solver Procedure
****************************************************
"""

def fsolve_old(curBlock, solv_options, dict_options, dict_equations, dict_variables):
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
    results = eval('eng.'+dict_options["fileName"]+'(matlab.double('+str(x0.tolist())+'),'+ str(FTOL)+','+ str(iterMax)+','+str(sortedVariables)+', nargout=4)')
    x = numpy.array(results[0][0])
    exitflag = results[2]
    output = results[3]

    if any(numpy.iscomplex(x)): x = x0
    
    curBlock.x_tot[curBlock.colPerm] = x

    if exitflag >= 1: return 1, output['iterations']
    else: return 0, output['iterations']


def get_sym_iter_functions_and_vars(curBlock):
    x_sym = []
    y_sym = []
    x = []
    y = []
    for glbID in curBlock.yb_ID:
        if glbID in curBlock.colPerm:
            x_sym.append(curBlock.x_sym_tot[glbID])
            x.append(curBlock.x_tot[glbID])
        else:
            y_sym.append(curBlock.x_sym_tot[glbID])
            y.append(curBlock.x_tot[glbID])  
            
    fSymbolic = numpy.array(curBlock.allConstraints(curBlock.x_sym_tot, curBlock.parameter))[curBlock.rowPerm].tolist()

    if y != []: 
        for i in range(0, len(fSymbolic)):
            fy = sympy.lambdify(y_sym, fSymbolic[i])
            fSymbolic[i] = fy(*y)
    
    return x_sym, x, fSymbolic
            
    
def fsolve(curBlock, solv_options, dict_options, dict_equations, dict_variables):
    FTOL = solv_options["FTOL"]
    iterMax = solv_options["iterMax"]

    xSymbolic, x0, fSymbolic = get_sym_iter_functions_and_vars(curBlock)


    list_of_functions = [str(f) for f in fSymbolic]
    list_of_xSymbols = [str(s) for s in xSymbolic]
    xSymbolic_string = " ".join(list_of_xSymbols)
    
    eng = matlab.engine.start_matlab()
    eng.addpath(str(pathlib.Path(__file__).parent.absolute()))
    results = eng.matlabScript(list_of_functions, xSymbolic_string, matlab.double(x0)[0], FTOL, iterMax, nargout=4)
    if isinstance(results[0], float): x = numpy.array([results[0]])
    else: x = numpy.array(results[0][0])
    exitflag = results[2]
    output = results[3]

    #if any(numpy.iscomplex(x)): x = x0
    for i in range(0, len(xSymbolic)):
       glb_ID = curBlock.x_sym_tot.tolist().index(xSymbolic[i])
       curBlock.x_tot[glb_ID] = x[i]

    if exitflag >= 1: return 1, output['iterations']
    else: return 0, output['iterations']
    

def systemToSolve(x, fSymbolic, xSymbolic):
    """ iterated function in matlab
    
    Args:
        :x:         current iteration point in matlab.double formate
        :fSymbolic: list with symbolic sympy functions
        :xSymbolic: string with symbolic variable names
    
    Returns:
        :fun_all:   list with current function values

    """
    if isinstance(x, float): x =[x]
    fun_all = []
    xSymbolic = sympy.symbols(xSymbolic)

    for fun in fSymbolic:
        fun = sympy.lambdify(xSymbolic, fun)
        #if numpy.isnan(fun(*x)): # TODO react on nan values
        #    fun_all.append(1.0)
        #else: 
        fun_all.append(fun(*x))
    return fun_all


