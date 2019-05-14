"""
***************************************************
Imported Packages
***************************************************
"""
import numpy
import MC33AD

"""
***************************************************
Embeded MC33 FORTRAN Routine
***************************************************
"""

__all__ =['doMC33']

def doMC33(A, itype=None):
    """ invokes FORTRAN Routine MC33 that was transformed to executable Python code via f2py
    Args:
        :A:          mxn matrix
    
    Return:     
        :output:    dictionary with results from MC33
        
    """ 
    
    # Local variables:
    if itype == None: itype = 5
    
    m, n = A.size()
    a = numpy.array(A.nonzeros())
    nzi = len(a)
    irn = numpy.array(A.row()) + 1
    jcn = numpy.array(A.sparsity().get_col()) + 1
    
    # Invoke MC33:   
    nd, rowPermFortran, colPermFortran, blocks, [nzRows,nzCols,borderWidth], ierr = MC33AD.mc33ad(m,n,nzi,a,irn,jcn,itype)
    
    # Create Dictionary 
    output = createDict([blocks, rowPermFortran - 1, colPermFortran - 1,
                       borderWidth, A[rowPermFortran - 1,  colPermFortran - 1],
                       nd, nzRows, nzCols, itype, ierr], [
            'Block Borders',
            'Row Permutation',
            'Column Permutation',
            'Border Width',
            'Matrix',
            'Number of Duplicate Entries',
            'Number of Nonzero Rows',
            'Number of Nonzero Columns',
            'Decomposition Method', # itype == 3 P3-Algorithm; itype!=3 and itype>0 P5-Algorithm =DEFAULT
            'Error message' #ierr < 0 error ocurred in MC33AD, ierr == 0 MC33AD sucessfully finished
            ])
    
    return output


def createDict(X, LABEL):
    """ creates dictionary with values X, and keys LABEL """
    
    X_DICT = {}
      
    for i in range(0,len(X)):
        X_DICT[LABEL[i]] = X[i]
        
    return X_DICT