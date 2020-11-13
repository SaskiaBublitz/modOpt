"""
***************************************************
Import packages
***************************************************
"""
import scipy.optimize
import numpy

"""
****************************************************
Minimization Procedures from scipy.optimization
****************************************************
"""

__all__ = ['minimize']

def fsolve(curBlock, solv_options, dict_options, dict_equations, dict_variables):
    FTOL = solv_options["FTOL"]
    iterMax = solv_options["iterMax"]
    x0 = curBlock.getIterVarValues()
    diag = [1/y for y in curBlock.colSca]
            
    args = (curBlock, dict_options)
    x, infodict, ier, mesg = scipy.optimize.fsolve(iter_functions, x0, args, 
                                                   xtol=FTOL, maxfev=iterMax,
                                                   full_output=True, diag=diag)
    curBlock.x_tot[curBlock.colPerm] = x
    
    if ier == 1: return 1, infodict['nfev']
    elif ier==2: return 0, infodict['nfev']
    else: return -1, infodict['nfev']
    
    
    

def iter_functions(x, curBlock, dict_options):
    #block_funs = []
    curBlock.x_tot[curBlock.colPerm] = x
    #allFun = curBlock.allConstraints(curBlock.x_tot, curBlock.parameter) 
    #for glbID in curBlock.rowPerm:
    if dict_options["scaling"] != "None": return curBlock.getScaledFunctionValues()    
    return curBlock.getPermutedFunctionValues()     


def minimize(curBlock, solv_options, dict_options, dict_eq, dict_var):
    """  solves nonlinear algebraic equation system (NLE) by minimization method
    from scipy.optimize package
    
    Args:
        :curBlock:      object of class Block with block information
        :solv_options:  dictionary with solver settings
        :dict_eq:       dictionary with information about equations
        :dict_var:      dictionary with information about iteration variables   
          
    """
    # TODO: Add scaling
    #if dict_options["scaling"] != 'None': 
    #    x0 = curBlock.getScaledIterVarValues()
    #    xBounds = curBlock.getScaledIterVarBoundValues()
    #else: 
    x0 = curBlock.getIterVarValues()
    xBounds = curBlock.getIterVarBoundValues()
    cons = []
    
    for glbID in curBlock.rowPerm:
        cons.append({'type': 'eq', 'fun': getEqualityConstraint(curBlock, glbID)})
        
    FTOL = solv_options["FTOL"]
    iterMax = solv_options["iterMax"]
    minMethod = solv_options["solver"]
    
    if solv_options["mode"]==1: 
        args = (curBlock, 0)
        res = scipy.optimize.minimize(eval_m, 
                                      x0,
                                      args= args,
                                      jac = eval_grad_m,
                                      method= minMethod, #'trust-constr',
                                      options = {'maxiter': iterMax, 'ftol': FTOL, 'disp': True},
                                      tol = None,
                                      bounds = xBounds,
                                      constraints = ())    
    
    if solv_options["mode"]==2:
    
        res = scipy.optimize.minimize(objective, 
                                      x0, 
                                      method= minMethod, #'trust-constr',
                                      ooptions = {'maxiter': iterMax, 'ftol': FTOL},
                                      bounds = xBounds,
                                      jac=None, 
                                      constraints = cons)
    #print res.x
    for i in range(0, len(res.x)):
        curBlock.x_tot[curBlock.colPerm[i]]=res.x[i]

    if not res.success: return 0, res.nit
    
    else: return 1, res.nit   


def eval_m(x, *args): 
    curBlock = args[0]
    curBlock.x_tot[curBlock.colPerm] = x
    f = curBlock.getFunctionValues()
    m = 0

    for fi in f: m += fi**2.0 
             
    return m 


def eval_grad_m(x, *args): 
    curBlock = args[0]
    #curBlock.x_tot[curBlock.colPerm] = x
    grad_m = numpy.zeros(len(x))
    
    f = curBlock.getFunctionValues()
    jac = curBlock.getPermutedJacobian()
    
    for j in range(0, len(x)):
        for i in range(0, len(f)):
            grad_m[j] += 2 * f[i] * jac[i, j]
            
    return grad_m


def getEqualityConstraint(curBlock, i):
    """ returns the ith equation of the NLE as equality constraint of an 
    optimization problem.
      
        Args:
        :curBlock:        instance of class block
        :i:               integer with global index of current equation
        
        Return:     lambda function with ith equality constraint
        
    """
    return lambda x: getSymbolicFunctions(x, curBlock, i)
    
    
def objective(x):
    """ Pseudo objective to solve NLE
    Args:
        :x:     list with iteration variable values
    Return:     constant value (here 0)    
    
    """
    
    return 0.0    

   
def subscribeXwithIterVars(x, curBlock):
    """ writes lambda block iteration variables x to a list of all iteration
    variables. This needs to be done to create the related equality constraints.
    
    Args:
        :curBlock:              instance of class block
        :x:                     vector with iteration variables from lambda function
    
    Return:
        :xWithIterVars:         numpy array with symbolic block iteration variables and
                                float values of all other iteration varibales
                                
    """
    
    xWithIterVars = numpy.array(curBlock.x_tot, dtype = object)
    xWithIterVars[curBlock.colPerm] = x
    return xWithIterVars    
 
    
def getSymbolicFunctions(x, curBlock, i):
    """ initializes block attribute allConstraints, which is from type function, 
    with symbolic block iteration variables and float values for all other 
    iteration variables and parameter.
                 
    Args:
        :curBlock:       instance of class block
        :x:              vector with iteration variables from lambda function
        :i:              global id of current equation
    
    Return:              ith equation for lambda function
                                
    """

    xWithIterVars = subscribeXwithIterVars(x, curBlock)
    
    return curBlock.allConstraints(xWithIterVars, curBlock.parameter)[i]  