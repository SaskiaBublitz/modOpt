"""
***************************************************
Imported Packages
***************************************************
"""
import casadi
import sympy
"""
***************************************************
Dulmage-Mendelsohn-Decomposition based on casadi package (python)
***************************************************
"""

__all__ =['getCasadiJandF','doDulmageMendelsohn']

def doDulmageMendelsohn(A):
    """invokes Dulmage-Mendelsohn-Decomposition from Casadi's Package
    Args:
        A (m,n):          mxn matrix
    
    Returns:     
        output (dict):    Results from Dulmage-Mendelsohn Decomposition 
                          with items as seen from the Code below.
                        
    """  
    
    output = createDict(A.sparsity().btf(), [
            'Block Borders',
            'Row Permutation',
            'Column Permutation',
            'Number of Row Blocks',
            'Number of Column Blocks', 
            'Number Coarse Row Blocks', 
            'Number Coarse Column Blocks',
            ])
    
    output["Matrix"] = A[output['Row Permutation'], 
           output['Column Permutation']]
    
    return output


# Methods to order and transfer results:

def createDict(X, LABEL):
    X_DICT = {}
      
    for i in range(0,len(X)):
        X_DICT[LABEL[i]] = X[i]
        
    return X_DICT


def writeResults(fileName, dict_variables):
    """ Creates File(s) with  final start values and lower and upper bounds. 
    For the ith set of start values, lower and upper bounds a new text file is 
    generated using the label "_i". 
    
    Args:
        fileName                String with name of text file(s)
        dict_variables          Dictionary with sets of state variable values,
                                lower and upper bounds
        
    """
    if dict_variables[dict_variables.keys()[0]][0] != []:
        initKey = dict_variables.keys()[0]
    
    
        for i in range(0, len(dict_variables[initKey][0])):
            res_file = open(''.join([fileName,"_", str(i+1),".txt"]), "w") 
            
            res_file.write("***** %s th Set of Initial Values and Bounds*****\n\n"%(i+1))
    
            for var in dict_variables.keys():
                res_file.write("%s %s %s %s\n"%(var,
                                                dict_variables[var][0][i], 
                                                dict_variables[var][1][i],
                                                dict_variables[var][2][i]))
            res_file.write("\n")
            res_file.close()


def getCasadiJandF(xSymbolic, fSymbolic):
    """Symbolic Jacoobian and Functions are prepared from sympy variables
    and function which are converted to casadi objects.

    Args:
        xSymbolic:        n variables in sympy formate
        fSymbolic:        m functions in sympy formate
    Returns:
            casadi.SX.array(m,n):     symbolic jacobian matrix
            casadi.SX.array(m,1):     symbolic equation system
            
    """
  
    # Symbolic Variables:
    x = convertSympyVariablesToCasadi(xSymbolic)
    fcasadi = lambdifyToCasadi(xSymbolic, fSymbolic)


    # Symbolic Equation System:
    f = casadi.Function('F', x, fcasadi(*x))

    return f.jacobian(), f


def convertSympyVariablesToCasadi(xSymbolic):
    """Converting sympy symbols to variables in casaid.SX.sym formate. The
    variables are returned in a list.
    
    """
    xSymbolicInCasadi = []
    for x in xSymbolic:
        exec("%s = casadi.SX.sym('%s')" % (repr(x), repr(x)))
        exec("xSymbolicInCasadi.append(%s)" % (repr(x)))
    return xSymbolicInCasadi


def lambdifyToCasadi(x, f):
    """Converting operations of symoblic equation system f (simpy) to 
    arithmetic interval functions (mpmath.iv)
    
    """
    
    toCasadi = {"exp" : casadi.SX.exp,
            "sin" : casadi.SX.sin,
            "cos" : casadi.SX.cos,
            "log" : casadi.SX.log}
    
    return sympy.lambdify(x,f, toCasadi) 

