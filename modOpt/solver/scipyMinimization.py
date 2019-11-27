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
    
    res = scipy.optimize.minimize(objective, 
                     x0, 
                     method= minMethod, #'trust-constr',
                     options = {'maxiter': iterMax},
                     tol =FTOL,
                     bounds = xBounds,
                     jac=None, 
                     constraints = cons)
    print res.x
    for i in range(0, len(res.x)):
        curBlock.x_tot[curBlock.colPerm[i]]=res.x[i]

    if not res.success: return 0, res.nit
    
    else: return 1, res.nit   


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