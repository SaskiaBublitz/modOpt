""" Scaling Methods """

"""
***************************************************
Import packages
***************************************************
"""

import MC29
import MC77
import numpy
import casadi

"""
***************************************************
Main that invokes scaling methods
***************************************************
"""

__all__ = ['scaleSystem']


def scaleSystem(model, dict_eq, dict_var, dict_options):
    """ equation system is scaled by user-defined input
    
    Args:
        :model:         object of class model in modOpt.model that contains all
                        information of the NLE-evaluation from MOSAICm. 
        :dict_eq:       dictionary with equation data
        :dict_var:      dictionary with variable data                        
        :dict_options:  dictionary with user-specified information
        
    """
    
    xValues = model.stateVarValues[0]
    jacobian = model.getJacobian(xValues)
    jacobian = jacobian[model.rowPerm, model.colPerm]
    F = model.fValues[model.rowPerm]
    

    if dict_options["scaling"] == 'MC77':
        res_scaling = MC77.doMC77(jacobian, F)
  
    if dict_options["scaling"] == 'MC29':
        res_scaling = MC29.doMC29(jacobian, F)
    
    if dict_options["scaling"] == 'Inf RowSca and Mean ColSca':
        res_scaling = {}
        colScalingMean(jacobian, res_scaling)
        jacobian = res_scaling["Matrix"]
        rowScaling(jacobian, F, res_scaling)
 
    if dict_options["scaling"] == 'Inf RowSca and gMean ColSca':
        res_scaling = {}
        colScalingGmean(jacobian, res_scaling)
        jacobian = res_scaling["Matrix"]
        rowScaling(jacobian, F, res_scaling)
    
    model.updateToScaling(res_scaling)
    
    updateDictionaries(dict_eq, dict_var, res_scaling, model)

    return True


def updateDictionaries(dict_eq, dict_var, res_scaling, model):
    """ updates all dictionaries after scaling

    Args:
        :dict_eq:       dictionary with information about all equations
        :dict_var:      dictionary with information about all variables
        :res_scaling:   dictionary with information about scaling results
        :model:         object of class Model
    
    """
  
    rowPermId = getPermutationIndex(model.rowPerm)
    colPermId = getPermutationIndex(model.colPerm)

    if res_scaling.has_key("Equations"):
        EqPermSca = numpy.array(res_scaling["Equations"])
        dict_eq = setValuesByGlobalId(dict_eq, 3, EqPermSca[rowPermId])
        
    if res_scaling.has_key("FunctionVector"):    
        FPermSca = numpy.array(res_scaling ["FunctionVector"])
        dict_eq = setValuesByGlobalId(dict_eq, 0, FPermSca[rowPermId]) 
    
    if res_scaling.has_key("Variables"):
        XPermSca = numpy.array(res_scaling["Variables"])
        dict_var = setValuesByGlobalId(dict_var, 0, model.stateVarValues[0]) 
        dict_var = setValuesByGlobalId(dict_var, 3 , XPermSca[colPermId])  


def setValuesByGlobalId(dictionary, column, newValues):
    """ writes newValues in a column of dictionary

    Args:
        :dictionary:   dictionary where newValues are stored
        :column:       column index in dictionary where newValues shall be stored
        :newValues:    new values that shall be written to dictionary
    
    Return:            dictionary

    """  
     
    for i in range(0,len(dictionary)):
        glbID = dictionary.values()[i][1]
        dictionary.values()[i][column] = newValues[glbID]
    return dictionary


def getPermutationIndex(permOrder):
    """ gets index after permutation based on permOrder

    Args:
        :permOrder:   list with permuted global indices
    
    Return:
        :permID:      list with permuted permutation indices

    """     
    
    permID = numpy.zeros(len(permOrder), int)
    
    for glbID in range(0,len(permOrder)):
        permID[glbID] = list(permOrder).index(glbID)
    return permID


def colScalingMean(A, res_scaling):
    """ column Scaling based on mean value of column sum
    
    Args:
        :A:              mxn matrix    
        :res_scaling:    dictonary with results of column scaling
        
    """

    colSum = numpy.sum(abs(numpy.array(A)),axis=0)
    col_nz = numpy.count_nonzero(A, axis=0)
     
    X_CONV = col_nz / colSum
    
    res_scaling["Matrix"] = casadi.mtimes(A,casadi.diag(X_CONV))
    res_scaling["Variables"] = X_CONV
    

def colScalingGmean(A, res_scaling):
    """ column Scaling based on geometric mean value of column sum
    
    Args:
        :A:              mxn matrix
        :res_scaling:    dictonary with results of column scaling

        
    """
    
    col_nz = numpy.count_nonzero(A, axis=0)
    X_CONV = numpy.empty(A.size()[1])   
    
    for i in range(0, A.size()[1]):
        A_col = A[numpy.nonzero(A[:,i])[0,:], i]
        colProd = numpy.prod(A_col)
        X_CONV[i] = 1/(abs(colProd))**(1/float(col_nz[i]))
        
    res_scaling["Matrix"] = casadi.mtimes(A,casadi.diag(X_CONV))
    res_scaling["Variables"] = X_CONV
    
    return res_scaling


def rowScaling(A, F, res_scaling):
    """ row Scaling based on row sums of a matrix
    
    Args:
        :A:              mxn matrix
        :F:              n dimensional function value array
        :res_scaling:    dictonary with results of row scaling
        
    """   
   
    rowSum = numpy.sum(abs(numpy.array(A)),axis=1)
    row_sca = numpy.max([rowSum,abs(numpy.array(F))],axis=0)
    di = 1 / row_sca
    
    F_sca = di*F
    A_sca = casadi.mtimes(casadi.diag(casadi.DM(di)),A)
 
    res_scaling["Matrix"] = A_sca
    res_scaling["FunctionVector"] = F_sca
    res_scaling["Equations"] = row_sca  


