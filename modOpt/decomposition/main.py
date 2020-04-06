""" Decompositon Methods """

"""
***************************************************
Import packages
***************************************************
"""
from modOpt.decomposition import dM
from modOpt.decomposition import MC33
import numpy

"""
***************************************************
Main that invokes decomposition methods
***************************************************
"""
__all__ = ['decomposeSystem']


def decomposeSystem(model, dict_eq, dict_var, dict_options):
    """ equation system is decomposed by user-defined input
    
    Args:
        :model:         object of class model in modOpt.model that contains all
                        information of the NLE-evaluation from MOSAICm. 
        :dict_eq:       dictionary with equation data
        :dict_var:      dictionary with variable data                        
        :dict_options:  dictionary with user-specified information
        
    """
    
    jacobian = model.getCasadiJacobian()
    
    if dict_options["decomp"] == 'DM':
        res_permutation = dM.doDulmageMendelsohn(jacobian)
        blocks = res_permutation["Number of Row Blocks"]
    
    if dict_options["decomp"] == 'BBTF':
        res_permutation = MC33.doMC33(jacobian)   
        blocks = res_permutation["Number of Row Blocks"]
        
    model.updateToPermutation(res_permutation["Row Permutation"], # TODO: input only res_permutation and hasKey()
                             res_permutation["Column Permutation"],
                             blocks)
    
    updateDictionariesToPermutation(model, dict_eq, dict_var)

    return True


def updateDictionariesToPermutation(model, dict_eq, dict_var):
    """ updates dictionaries of variables and equations with data from model
    
    Args:
         :model:      object of class model in modOpt.model that contains all
                      information of the NLE-evaluation from MOSAICm. 
        :dict_eq:     dictionary with equation data
        :dict_var:    dictionary with variable data
        
    """
    
    rowPermId = getPermutationIndex(model.rowPerm)
    colPermId = getPermutationIndex(model.colPerm)   
    setValuesByGlobalId(dict_eq, 2, rowPermId)                                       
    setValuesByGlobalId(dict_var, 2, colPermId)
    

def getPermutationIndex(permOrder):
    """ get new Index after permutation
    
    Args:
        :permOrder:       list with global indices order after permutation
    
    """
    
    permID = numpy.zeros(len(permOrder), int)
    
    for glbID in range(0,len(permOrder)):
        permID[glbID] = list(permOrder).index(glbID)
    return permID


def setValuesByGlobalId(dictionary, column, newValues):
    """ set certain values of a quantity in a dictionary column by its global index 
    
    Args:
        :dictionary:       dictionary
        :column:           column index in dictionary
        :newValues:        list with new values in permuted order
    
    """
    
    for i in range(0,len(dictionary)):
        glbID = list(dictionary.values())[i][1]
        list(dictionary.values())[i][column] = newValues[glbID]