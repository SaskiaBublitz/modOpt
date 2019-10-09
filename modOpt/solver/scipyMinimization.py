"""
***************************************************
Import packages
***************************************************
"""
import scipy.optimize
import numpy
import sympy
"""
****************************************************
Minimization Procedures from scipy.optimization
****************************************************
"""

__all__ = ['minimize']


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
    #    x = curBlock.getScaledIterVarValues()
    #    xBounds = curBlock.getScaledIterVarBoundValues()
    #else: 
    x0 = curBlock.getIterVarValues()
    xSym = curBlock.getSymbolicVariablesOfBlock()
    xBounds = curBlock.getIterVarBoundValues()
    cons = []
    
    for glbID in curBlock.rowPerm:
        cons.append(getConstraintDictionary(glbID, curBlock, getSymbolicFunctions, xSym))
        
    FTOL = solv_options["FTOL"]
    iterMax = solv_options["iterMax"]
    minMethod = solv_options["solver"]
    
    res = scipy.optimize.minimize(objective, 
                     x0, 
                     method= minMethod, #'trust-constr',
                     options = {'maxiter': iterMax},
                     tol =FTOL,
                     bounds = xBounds,
                     jac=None, 
                     constraints = cons)

    for i in range(0, len(res.x)):
        curBlock.x_tot[curBlock.colPerm[i]]=res.x[i]

    if not res.success: return 0, res.nit
    
    else: return 1, res.nit   

    
def objective(x):
    """ Pseudo objective to solve NLE
    Args:
        :x:     list with iteration variable values
    Return:     constant value (here 0)    
    
    """
    
    return 0.0    


def getEqualityConstraint(i, curBlock, getSymbolicFunctions, *args):
    # TODO: Docu
    c = getSymbolicFunctions(curBlock, *args)
    return c[i]


def getConstraintDictionary(i, curBlock, getSymbolicFunctions, *args):
    # TODO: Docu
    return {'type': 'eq', 'fun': lambda x: getEqualityConstraint(i, curBlock,
                                                                 getSymbolicFunctions, 
                                                                 x)}

def subscribeXwithIterVars(curBlock, x):
    xWithIterVars = numpy.array(curBlock.x_tot, dtype = object)
    xWithIterVars[curBlock.colPerm] = x
    return xWithIterVars    
 
    
def getSymbolicFunctions(curBlock, x):
    xWithIterVars = subscribeXwithIterVars(curBlock, x)
    return curBlock.allConstraints(xWithIterVars, curBlock.parameter)    