"""
***************************************************
Import packages
***************************************************
"""
import mpmath
from modOpt.constraints import parallelization
import numpy
"""
***************************************************
Update directories
***************************************************
"""

__all__ = ['updateDictToModel', 'storeNewBoxesInNPZ', 'get_entry_from_npz_dict']



def storeNewBoxesInNPZ(dict_name, model, iterNo):
    newBoxes = model.xBounds
    listNewBoxes = []
    for box in newBoxes:
        listNewBoxes.append(numpy.array(parallelization.convertMpiToList(box)))
    if iterNo ==1: numpy.savez_compressed(dict_name, numpy.array(listNewBoxes))
    else: append_npz_dict(dict_name, numpy.array(listNewBoxes))
    
    
def append_npz_dict(dict_name, array):
    arrays = load_npz_dict(dict_name)
    arrays.append(array)
    numpy.savez_compressed(dict_name, *arrays)

    
def load_npz_dict(dict_name):
    arrays = []
    dict_data = numpy.load(dict_name)
    for key in dict_data.keys():
       arrays.append(dict_data[key])
    return arrays


def get_entry_from_npz_dict(dict_name, position):
    return load_npz_dict(dict_name)[position]


def updateDictToModel(dict_variables, res_solver):
    """ updates variable dictionary to current model values
    
    Args:
        :dict_variables:      dictionary with set of state variable values, lower 
                              and upper bounds
        :res_solver:          dictionary with resulting model after variable bounds 
                              reduction
           
    """
    
    model = res_solver["Model"]
    
    #if not model.failed :
    xValues = model.stateVarValues
    xBounds = model.xBounds

    
    for i in range(0, len(dict_variables)):
        glbIdx = dict_variables[list(dict_variables)[i]][3]
        curXValues = []
        curXLowerBounds = []
        curXUpperBounds = []
    
        for j in range(0, len(xValues)):
            curXValues.append(float(xValues[j][glbIdx]))
            curXLowerBounds.append(float(mpmath.mpf(xBounds[j][glbIdx].a)))
            curXUpperBounds.append(float(mpmath.mpf(xBounds[j][glbIdx].b)))
        
        dict_variables[list(dict_variables)[i]][0] = curXValues
        dict_variables[list(dict_variables)[i]][1] = curXLowerBounds
        dict_variables[list(dict_variables)[i]][2] = curXUpperBounds
