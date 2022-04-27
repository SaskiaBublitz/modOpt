"""
***************************************************
Import packages
***************************************************
"""

import casadi
import numpy
import sympy
from modOpt.solver import newton
import modOpt.solver.scipyMinimization
"""
****************************************************
Minimization Procedures from scipy.optimization
****************************************************
"""

__all__ = ['minimize']

def objective(functions,a):
    """
    Args:
        :functions:     list with functions
    Return
        :obj            symbolic objective function

    """
    
    obj = 0
    for f in functions: obj += f**2
    return a/2.0*obj


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
    #if "max_cpu_time" in solv_options.keys(): cpu_max = solv_options["max_cpu_time"]
    #else: cpu_max = 5.0
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
            
    obj_casadi= lambdifyToCasadi(x_sympy, objective(functions,1.0/solv_options["FTOL"]))#1.0/solv_options["FTOL"]))
    nlp = {'x':casadi.vertcat(*x_casadi), 'f':obj_casadi(*x_casadi)}
    #options={"max_iter": 50000};
    S = casadi.nlpsol('S', 'ipopt', nlp, {"ipopt":{'max_iter':solv_options["iterMax"],
                                                   'linear_solver': 'ma27',
                                                   'tol': solv_options["FTOL"],
                                                   'linear_system_scaling': 'mc19',
                                                   'warm_start_init_point':'yes',
                                                   #'mu_strategy': 'adaptive',
                                                   'mu_max': 1e-10,
                                                   'mu_min': 1e-30,
                                                   'mu_init': 1e-1,
                                                   #'ma57_automatic_scaling': 'yes',
                                                   'mu_oracle': 'loqo',
                                                   #'max_cpu_time':100,
                                                   #'ma57_pivot_order': 4,
                                                   }})


    r = S(x0=x_0, lbx=x_L, ubx=x_U, lbg=0, ubg=0)
    curBlock.x_tot[glb_ID] = r['x'].T
    fresidual = numpy.linalg.norm(curBlock.getFunctionValues())
    if solv_options["FTOL"] < fresidual:        
        #return modOpt.solver.scipyMinimization.fsolve(curBlock, solv_options, dict_options)
        return newton.doNewton(curBlock, solv_options, dict_options)
    elif numpy.isnan(fresidual): return -1, solv_options["iterMax"]
    
    else: return 1, solv_options["iterMax"]
    
    
    
    