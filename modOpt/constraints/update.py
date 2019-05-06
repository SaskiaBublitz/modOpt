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

def updateDictToModel(dict_variables, model):
    """ updates variable dictionary to current model values
    
    Args:
        :dict_variables:      dictionary with set of state variable values, lower 
                              and upper bounds
        :model:               object of class model
    
    Return:
        :dict_variables:      updated state variable dictionary
           
    """
    
    xValues = model.getXValues()
    xBounds = model.getXBounds()
    
    for i in range(0, len(dict_variables)):
        glbIdx = dict_variables[dict_variables.keys()[i]][3]
        curXValues = []
        curXLowerBounds = []
        curXUpperBounds = []
        
        for j in range(0, len(xValues)):
            curXValues.append(float(xValues[j][glbIdx]))
            curXLowerBounds.append(float(mpmath.mpf(xBounds[j][glbIdx].a)))
            curXUpperBounds.append(float(mpmath.mpf(xBounds[j][glbIdx].b)))
            
        dict_variables[dict_variables.keys()[i]][0] = curXValues
        dict_variables[dict_variables.keys()[i]][1] = curXLowerBounds
        dict_variables[dict_variables.keys()[i]][2] = curXUpperBounds
    
    return dict_variables