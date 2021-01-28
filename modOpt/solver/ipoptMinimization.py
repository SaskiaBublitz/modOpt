"""
***************************************************
Import packages
***************************************************
"""
#import pyipopt
import ipopt 
import numpy
from modOpt.solver import ipopt_problem

"""
****************************************************
Minimization Procedures from scipy.optimization
****************************************************
"""

__all__ = ['minimize']

def minimize(curBlock, solv_options, dict_options):
    """ This function calls the ipopt solver.
    
    Args:
        :curBlock:      object of class Block with block information
        :solv_options:  dictionary with solver settings
        :dict_options:  dictionary with user-specified settings  
        
    """
    
    x0 = numpy.array(curBlock.getIterVarValues(), dtype=float)
    xBounds = curBlock.getIterVarBoundValues()    
    x_L = numpy.array(xBounds[:,0], dtype=float)
    x_U = numpy.array(xBounds[:,1], dtype=float)   
    #args = (curBlock)
    nvar = len(x0)
    if solv_options["mode"] == 1:
        ncon = 0
        g_L = numpy.array([], dtype=float) 
        g_U = numpy.array([], dtype=float)  
        problem_obj = ipopt_problem.Ipopt_problem(curBlock)
        nlp = ipopt.problem(nvar, ncon, problem_obj, x_L, x_U)
        nlp.addOption('print_user_options', 'no')
        nlp.addOption('print_level', 5)    
        nlp.addOption('warm_start_init_point','yes') # try: yes
        #nlp.addOption('linear_solver', 'ma97')     # ma57 oder ma77, ma86, ma97, mumps
        #nlp.addOption('ma57_pivot_order', 4)
        #nlp.addOption('ma57_automatic_scaling', 'yes')
        nlp.addOption('mu_init', 1e-10)    # 1e-10
        nlp.addOption('mu_strategy', 'adaptive')   # adaptive, monotone
        nlp.addOption('mu_min', 1e-30)     # 1e-20 / 1e-30
        nlp.addOption('mu_max', 1e-1)      # 1e+3  / 1e-1
        nlp.addOption('mu_oracle', 'loqo')
        nlp.addOption('warm_start_mult_bound_push', 1e-10)
        nlp.addOption('warm_start_bound_push', 1e-10)
        nlp.addOption('max_iter', solv_options["iterMax"]) 
        nlp.addOption('acceptable_constr_viol_tol', 1e-5)
        nlp.addOption('tol', solv_options["FTOL"])  
    
        x, info = nlp.solve(x0)
        curBlock.x_tot[curBlock.colPerm] = x
        if not info['status']==0 and not info['status']==1: return 0, solv_options["iterMax"]
    
        else: return 1, solv_options["iterMax"]  


def minimize27(curBlock, solv_options, dict_options):
    """  solves nonlinear algebraic equation system (NLE) with ipopt in python27 
    based on Erik's interface
    
    Args:
        :curBlock:      object of class Block with block information
        :solv_options:  dictionary with solver settings          
    """
    
    # TODO: Add scaling
    #if dict_options["scaling"] != 'None': 
    #    x0 = curBlock.getScaledIterVarValues()
    #    xBounds = curBlock.getScaledIterVarBoundValues()
    #else:     
    x0 = numpy.array(curBlock.getIterVarValues(), dtype=float)
    xBounds = curBlock.getIterVarBoundValues()
    args = (curBlock,0)
    
    x_L = numpy.array(xBounds[:,0], dtype=float)
    x_U = numpy.array(xBounds[:,1], dtype=float)     
    
    nvar = len(x0)
    
    if solv_options["mode"] == 1:
        ncon = 0
        nnzj = 0 
        nnzh = 0
        g_L = numpy.array([], dtype=float) 
        g_U = numpy.array([], dtype=float)
        #nlp = pyipopt.create(nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, 
        #                     eval_m, eval_grad_m, 
        #                     eval_cons_empty, eval_grad_cons_empty)
        
    if solv_options["mode"] == 2:
        ncon = nvar
        nnzj = len(numpy.nonzero(curBlock.getPermutedJacobian())[0])
        nnzh = 0#(nvar*(nvar+1))/2 
       # print eval_cons(x0, args)  
        #print eval_grad_obj_empty(x0, args)   
        #print eval_obj_empty(x0, args) 
        g_L = numpy.zeros(nvar, dtype=float) 
        g_U = numpy.zeros(nvar, dtype=float) 
        
        #nlp = pyipopt.create(nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, 
        #                     eval_obj_empty, eval_grad_obj_empty, 
        #                     eval_cons, eval_grad_cons)        
    #pyipopt.set_loglevel(0)
    
    #nlp.str_option('print_user_options', 'no')
    #nlp.int_option('print_level', 5)    
    #nlp.str_option('warm_start_init_point','yes') # try: yes
    #nlp.str_option('linear_solver', 'ma57')     # ma57 oder ma77, ma86, ma97, mumps
    #nlp.int_option('ma57_pivot_order', 4)
    #nlp.str_option('ma57_automatic_scaling', 'yes')
    #nlp.num_option('mu_init', 1e-10)    # 1e-10
    #nlp.str_option('mu_strategy', 'adaptive')   # adaptive, monotone
    #nlp.num_option('mu_min', 1e-40)     # 1e-20 / 1e-30
    #nlp.num_option('mu_max', 1e-3)      # 1e+3  / 1e-1
    #nlp.str_option('mu_oracle', 'loqo')
    #nlp.num_option('warm_start_mult_bound_push', 1e-10)
    #nlp.num_option('warm_start_bound_push', 1e-10)
    #nlp.int_option('max_iter', solv_options["iterMax"]) 
    #nlp.num_option('acceptable_constr_viol_tol', 1e-5)
    #nlp.num_option('tol', solv_options["FTOL"])  
    

    #res  = nlp.solve(x0, args) 
    #nlp.close()

    #curBlock.x_tot[curBlock.colPerm] = res[0]

    #if not res[5]==0 and not res[5]==1: return 0, solv_options["iterMax"]
    
    #else: return 1, solv_options["iterMax"]  


def eval_obj_empty(x, args=None):
    return (0.0, 1.0)

    
def eval_grad_obj_empty(x, args=None):
    return numpy.zeros(len(x+1), dtype=float)


def eval_cons(x, args=None):
    curBlock = args[0]
    curBlock.x_tot[curBlock.colPerm] = x
    f = curBlock.getFunctionValues()
    return numpy.array(numpy.insert(f,0,0), dtype=float)


def eval_grad_cons(x, flag, args=None):
    curBlock = args[0]
    jac = numpy.matrix(curBlock.getPermutedJacobian(), dtype=float)
    if flag:
        rows = numpy.nonzero(jac)[0]
        cols = numpy.nonzero(jac)[1]
        return (rows, cols)
    else:      
        nz = numpy.array(jac[numpy.nonzero(jac)])[0]
        return numpy.array(numpy.insert(nz, 0, 0),dtype=float)


def eval_m(x, args=None): 
    curBlock = args[0]
    curBlock.x_tot[curBlock.colPerm] = x
    f = curBlock.getFunctionValues()
    m = 0

    for fi in f: m += fi**2.0 
             
    return (0.0, m)


def eval_grad_m(x, args=None): 
    curBlock = args[0]
    #curBlock.x_tot[curBlock.colPerm] = x
    grad_m = numpy.zeros(len(x)+1)
    
    f = curBlock.getFunctionValues()
    jac = curBlock.getPermutedJacobian()
    
    for j in range(0, len(x)):
        for i in range(0, len(f)):
            grad_m[j+1] += 2 * f[i] * jac[i, j]
            
    return numpy.array(grad_m, dtype=float)


def eval_cons_empty(x, args=None): 
    return numpy.array([]) 


def eval_grad_cons_empty(x, flag, args=None): 
    if flag:
        rows = numpy.array([], dtype=int)
        cols = numpy.array([], dtype=int)
        return (rows, cols)
    else: 
        return numpy.array([], dtype=float)
