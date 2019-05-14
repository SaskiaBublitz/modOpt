"""
***************************************************
Imported Packages
***************************************************
"""
import MC77D
import numpy
import casadi

"""
***************************************************
Scaling method MC77D from HSL library
***************************************************
"""

def doMC77(A, F, job = None):
    """ invokes FORTRAN Routine MC77 that was transformed to executable Python code via f2py
    
    Args:
        :A:          mxn matrix
        :F:          m Function value array
        :job:        additional information about norm, default = infinity-norm (0)
    
    Return:     
        :output:    dictionary with scaling output    
        
    """
    # Initialization:
    col_sca = numpy.ones(28)    
    if job == None: job = 0
    m, n = A.size()
    a = numpy.array(A.nonzeros())
    nnz = len(a)
    irn = numpy.array(A.row()) + 1
    jcn = numpy.array(A.sparsity().get_col()) + 1 
    output = {}
    
    # Calculate control variables for scaling routine:
    icntl, cntl = MC77D.mc77id() 
    
    # Calculate workspace dimensions based on control variable results:
    if icntl[5] == 0: liw = m + n
    else: liw = m
    
    if icntl[4] == 0 and icntl[5] == 0: ldw = nnz + 2 * (m + n)
    elif icntl[4] == 1 and icntl[5] == 0: ldw = 2 * (m + n)
    elif icntl[4] == 0 and icntl[5] != 0: ldw = nnz + 2 * m
    elif icntl[4] == 1 and icntl[5] != 0: ldw = 2 * m
    
    # Invoke scaling HSL method MC77BD:
    dw, info, rinfo = MC77D.mc77bd(job, m, n, nnz, irn, jcn, a, liw, ldw, icntl, cntl)
    # Postprocessing of scaling results:
    if info[0] >= 0: # No error
        if info[0]>0: print("Warning: Something might be wrong in scaling factor calculation by MC77. Try to use alternativ routine.")
        
        row_sca = dw[0:m]
        F_sca =  F / row_sca
        
        if icntl[5] == 0: # successful column scaling
            col_sca = 1 / dw[m:m+n]
        
        A_sca = casadi.mtimes(casadi.mtimes(casadi.diag(casadi.DM(1/row_sca)),A),casadi.diag(casadi.DM(col_sca)))
    
    output["Matrix"] = A_sca
    output["FunctionVector"] = F_sca
    output["Equations"] = row_sca
    output["Variables"] = col_sca
    
    return output

