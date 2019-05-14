"""
***************************************************
Imported Packages
***************************************************
"""
import MC29AD
import numpy
import casadi

"""
***************************************************
Scaling method MC29AD from HSL library
***************************************************
"""

def doMC29(A, F): 
    """ invokes FORTRAN Routine MC29 that was transformed to executable Python code via f2py
    Args:
        :A:          mxn matrix
        :F:          m Function value array
    
    Return:     
        :output:    dictionary with scaling output
    
    """ 
    
    # Initialization  
    output = {}     
    m, n = A.size()
    a = numpy.array(A.nonzeros())
    nzi = len(a)
    irn = numpy.array(A.row()) + 1
    jcn = numpy.array(A.sparsity().get_col()) + 1  
    
    r, c,ifail = MC29AD.mc29ad(m,n,nzi,a,irn,jcn,0) #TODO: if python module exists it will be invoked here, in/out variables need to be checked
     
    A_sca = casadi.mtimes(casadi.mtimes(casadi.diag(casadi.DM(numpy.exp(r))),A),casadi.diag(casadi.DM(numpy.exp(c))))
    row_sca = 1/numpy.exp(r)
    col_sca = numpy.exp(c)
    F_sca = F * numpy.exp(r)
        
    output["Matrix"] = A_sca
    output["FunctionVector"] = F_sca
    output["Equations"] = row_sca
    output["Variables"] = col_sca
    
    return output

