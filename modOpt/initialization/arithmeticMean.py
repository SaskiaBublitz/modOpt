"""
***************************************************
Import packages
***************************************************
"""
import numpy
import mpmath

"""
***************************************************
sets initials to arithemtic mean value of an interval
***************************************************
"""

__all__ = ['setStateVarValuesToMidPointOfIntervals']

def setStateVarValuesToMidPointOfIntervals(res_solver, dict_options):
    """ Set all state variable values of a model to the midpoint of their related
    interval
    
    Args:
        :res_solver:     dictionary with resulting model after variable bounds 
                        reduction
        :dict_options:   dictionary with user settings used because of absEps. 
                        if mean value is 0, the inital value is set to the absEps
                        value
        
    """
    
    model = res_solver["Model"]
    
    if model.xBounds !=[]:
        absEps = dict_options["absTolX"]
        model.stateVarValues = numpy.empty((len(model.xBounds), len(model.xBounds[0])))
        for i in range(0, len(model.xBounds)):
            for j in range(0, len(model.xBounds[i])):

                midPoint = float(mpmath.mpf(model.xBounds[i][j].mid))
                if midPoint != 0: 
                    model.stateVarValues[i][j]= midPoint
                else :
                    model.stateVarValues[i][j] = absEps