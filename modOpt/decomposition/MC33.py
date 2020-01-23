"""
***************************************************
Imported Packages
***************************************************
"""
import numpy
from modOpt.decomposition import MC33

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
    blocks = list(blocks)
    blocks = getBlockBorders(m, blocks, borderWidth)

    # Create Dictionary 
    output = createDict([blocks, list(rowPermFortran - 1), list(colPermFortran - 1),
                       borderWidth, A[rowPermFortran - 1,  colPermFortran - 1],
                       nd, nzRows, nzCols, itype, ierr], [
            'Number of Row Blocks',
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

def getBlockBorders(dim, blocks, borderWidth):
    
    blockborders = list(set(blocks[0:len(blocks)-borderWidth]))
    blockborders.insert(0,0)

    blockborders.append(max(blockborders)+borderWidth)

    return blockborders
        


def createDict(X, LABEL):
    """ creates dictionary with values X, and keys LABEL """
    
    X_DICT = {}
      
    for i in range(0,len(X)):
        X_DICT[LABEL[i]] = X[i]
        
    return X_DICT