"""
***************************************************
Import packages
***************************************************
"""
import pathlib
import numpy
from modOpt.solver import block
import sympy
from sympy.parsing.sympy_parser import parse_expr
"""
****************************************************
Newton Solver Procedure
****************************************************
"""
def fsolve_mscript(curBlock, solv_options, dict_options):
    """ matlab is invoked on model-specific matlab script that needs to be created by UDLS 
    from MOSAICmodeling. The name of the matlab file needs to be identical to the output 
    file name set by the user in dict_options.
    
    Args:
        :curBlock:          instance of class Block
        :solv_options:      dictionary with settings for the solver
        :dict_options:      dictionary containing the output file's name
        
    Return:
        :exitflag:          1 = solved, 0 is not solved
        :iterNo:            number of iterations as integer
    
    """
    import matlab.engine
    FTOL = solv_options["FTOL"]
    iterMax = solv_options["iterMax"]
    x_init = curBlock.x_tot
    rowPerm = curBlock.rowPerm + numpy.ones(len(curBlock.rowPerm)) # col index in matlab starts with 1
    colPerm = curBlock.colPerm + numpy.ones(len(curBlock.colPerm)) # row index in matlab starts with 1
    varNames = list(map(str, curBlock.x_sym_tot))

    try:
        eng = matlab.engine.start_matlab()
        results = eval('eng.'+dict_options["fileName"]+'(matlab.double(x_init.tolist()), matlab.double(colPerm.tolist()), matlab.double(rowPerm.tolist()), varNames, FTOL, iterMax, nargout=4)') #file and function need to have the same name, as System
        if isinstance(results[0], float): x = numpy.array([results[0]])
        else: x = numpy.array(results[0][0])
        exitflag = results[2]
        output = results[3]

        if any(numpy.iscomplex(x)): 
            print("Warning: Complex number(s) is/are casted to real")
            x = x.real
              
        curBlock.x_tot[curBlock.colPerm] = x

        if exitflag >= 1: return 1, output['iterations']
        else: return 0, output['iterations']
        
    except:
        print("Error: The system could not be parsed to Matlab.")
        return 0, -1
    
def fsolve(curBlock, solv_options, dict_options):
    """ matlab is invoked using a general matlab script included in modOpt. 
    This general script invokes the python function systemToSolve to iterate 
    the python model. This option takes in general longer than fsolve_mscript 
    due to the extra model transformations between matlab and python during the 
    iteration. It should only be used if the UDLS for the fsolve_mscript is not 
    available since the latter is able to iterate complex variables contrary to 
    
    Args:
        :curBlock:          instance of class Block
        :solv_options:      dictionary with settings for the solver
        :dict_options:      dictionary containing the output file's name
    
    Return:
        :exitflag:          1 = solved, 0 is not solved
        :iterNo:            number of iterations as integer
    
    """
    import matlab.engine    
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


