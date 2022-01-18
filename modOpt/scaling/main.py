""" Scaling Methods """

"""
***************************************************
Import packages
***************************************************
"""
from modOpt.scaling import MC29, MC77
import numpy
import casadi

"""
***************************************************
Main that invokes scaling methods
***************************************************
"""

__all__ = ['scaleSystem']


def scaleSystem(model, dict_options):
    """ equation system is scaled by user-defined input
    
    Args:
        :model:         object of class model in modOpt.model that contains all
                        information of the NLE-evaluation from MOSAICm.                     
        :dict_options:  dictionary with user-specified information
        
    """
    
    jacobian = model.getPermutedJacobian()
    #F = [model.getFunctionValues()[i] for i in model.rowPerm]
    F = model.getPermutedFunctionValues()

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
    # TODO: Scaling with Bounds
    model.updateToScaling(res_scaling)
    
    #updateDictionaries(dict_eq, dict_var, res_scaling, model)

    return True


def updateDictionaries(dict_eq, dict_var,res_scaling, model):
    """ updates all dictionaries after scaling

    Args:
        :dict_eq:       dictionary with information about all equations
        :dict_var:      dictionary with information about all variables
        :res_scaling:   dictionary with information about scaling results
        :model:         object of class Model
    
    """
  

    if res_scaling.__contains__("Equations"):
        dict_eq = setDictionary(dict_eq, model.rowPerm, 3, model.rowSca)
        
    
    if res_scaling.__contains__("Variables"):
        dict_var = setDictionary(dict_var, model.colPerm, 3 , model.colSca)  


def setDictionary(dictionary, glbID, column, newValues):
    """ writes newValues in a column of dictionary referring to list with
    glbIDs

    Args:
        :dictionary:   dictionary where newValues are stored
        :glbID:        list with glbID's referring to list newValues
        :column:       column index in dictionary where newValues shall be stored
        :newValues:    new values that shall be written to dictionary
    
    Return:            dictionary

    """  
     
    for i in glbID:
        for entry in dictionary:
            if dictionary[entry][1] == i: 
                dictionary[entry][column] = newValues[list(glbID).index(i)]
                break
        
    return dictionary


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