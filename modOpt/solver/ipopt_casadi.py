"""
***************************************************
Import packages
***************************************************
"""

import casadi
import numpy
import sympy
from modOpt.solver import newton

"""
****************************************************
Minimization Procedures from scipy.optimization
****************************************************
"""

__all__ = ['minimize']

def objective(functions):
    """
    Args:
        :functions:     list with functions
    Return
        :obj            symbolic objective function

    """
    
    obj = 0
    for f in functions: obj += f**2
    return obj


def convert_sympy_vars_to_casadi(x_symbolic):
    """ converting sympy symbols to variables in casaid.SX.sym formate. The
    variables are returned in a list.
    
    Args:
        :x_symbolic:     list wit symbolic state variables in sypmy logic

    """  
    x_casadi = []
    
    for xs in x_symbolic:
        exec("%s = casadi.SX.sym('%s')" % (repr(xs), repr(xs)))
        exec("x_casadi.append(%s)" % (repr(xs)))
    return x_casadi


def lambdifyToCasadi(x, f):
    """Converting operations of symoblic equation system f (simpy) to 
    arithmetic interval functions (mpmath.iv)
    
    """
    
    toCasadi = {"exp" : casadi.SX.exp,
            "sin" : casadi.SX.sin,
            "cos" : casadi.SX.cos,
            "log" : casadi.SX.log,
            "abs":  casadi.SX.fabs,
            "sqrt": casadi.SX.sqrt,}   
    return sympy.lambdify(x,f, toCasadi) 


def minimize(curBlock, solv_options, dict_options):
    """ This function calls the ipopt solver.
    
    Args:
        :curBlock:      object of class Block with block information
        :solv_options:  dictionary with solver settings
        :dict_options:  dictionary with user-specified settings  
        
    """    
    x_Bounds = curBlock.getIterVarBoundValues()    
    functions = [f.f_sym for f in curBlock.functions_block]
    glb_ID = []
    
    for f in curBlock.functions_block: glb_ID+=f.glb_ID
    glb_ID = list(set(glb_ID))
    glb_ID_non_iter = [cur_id for cur_id in glb_ID if not cur_id in curBlock.colPerm]    

    x_sympy = curBlock.x_sym_tot[glb_ID] 
    x_casadi = convert_sympy_vars_to_casadi(list(x_sympy))    
    x_0 = curBlock.x_tot[glb_ID]
    x_Bounds = curBlock.xBounds_tot[glb_ID]
    
    x_L = numpy.array(x_Bounds[:,0], dtype=float)
    x_U = numpy.array(x_Bounds[:,1], dtype=float)
    
    for i,ID in enumerate(glb_ID):
        if ID in glb_ID_non_iter:
            x_L[i] = x_0[i]
            x_U[i] = x_0[i]
            
    obj_casadi= lambdifyToCasadi(x_sympy, objective(functions))
    nlp = {'x':casadi.vertcat(*x_casadi), 'f':obj_casadi(*x_casadi)}
    #options={"max_iter": 50000};
    S = casadi.nlpsol('S', 'ipopt', nlp, {"ipopt":{'max_iter':solv_options["iterMax"]}})


    r = S(x0=x_0, lbx=x_L, ubx=x_U, lbg=0, ubg=0)
    curBlock.x_tot[glb_ID] = r['x'].T
    
    if solv_options["FTOL"] < numpy.linalg.norm(curBlock.getScaledFunctionValues()):
        return newton.doNewton(curBlock, solv_options, dict_options)
    else: return 1, solv_options["iterMax"]
    
    
    
    