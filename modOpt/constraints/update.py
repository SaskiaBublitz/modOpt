"""
***************************************************
Import packages
***************************************************
"""
import mpmath

"""
***************************************************
Update directories
***************************************************
"""

__all__ = ['updateDictToModel']

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
