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
    """ returns the ith equation of the NLE as equality constraint of an 
    optimization problem.
    
    Args:
        :i:                     integer with global index of equation
        :curBlock:              instance of class block
        :getSymbolicFunctions:  function defined by the UDLS with equation system
        :*args:                 flexible arguments that equals here the symbolic
                                iteration variables
    Return:
        :c[i]:                  ith equality constraint
        
    """
    
    c = getSymbolicFunctions(curBlock, *args)
    return c[i]


def getConstraintDictionary(i, curBlock, getSymbolicFunctions, *args):
    """ returns equality constraint as dictionary. This formate used in the 
    scipy.optimize.minimize function.
    
    Args:
        :i:                     integer with global index of equation
        :curBlock:              instance of class block
        :getSymbolicFunctions:  function defined by the UDLS with equation system
        :*args:                 flexible arguments that equals here the symbolic
                                iteration variables
                                
    Return:                     dictionary with equality constraint
    
    """
    return {'type': 'eq', 'fun': lambda x: getEqualityConstraint(i, curBlock,
                                                                 getSymbolicFunctions, 
                                                                 x)}

    
def subscribeXwithIterVars(curBlock, x):
    """ writes symbolic block iteration variables x to a list of all iteration
    variables. This needs to be done to create the related equality constraints.
    
    Args:
        :curBlock:              instance of class block
        :x:                     array in sympy logic with symbolic iteration variables
    
    Return:
        :xWithIterVars:         numpy array with symbolic block iteration variables and
                                float values of all other iteration varibales
                                
    """
    
    xWithIterVars = numpy.array(curBlock.x_tot, dtype = object)
    xWithIterVars[curBlock.colPerm] = x
    return xWithIterVars    
 
    
def getSymbolicFunctions(curBlock, x):
    """ initializes block attribute allConstraints, which is from type function, 
    with symbolic block iteration variables and float values for all other 
    iteration variables and parameter.
                 
    Args:
        :curBlock:       instance of class block
        :x:              array in sympy logic with symbolic iteration variables
    
    Return:              symbolic equation system in sympy logic
                                
    """
    
    xWithIterVars = subscribeXwithIterVars(curBlock, x)
    return curBlock.allConstraints(xWithIterVars, curBlock.parameter)    