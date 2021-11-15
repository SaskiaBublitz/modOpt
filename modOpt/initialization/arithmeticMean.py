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
    if "Model" in res_solver.keys():
        model = res_solver["Model"]
    
        if model.xBounds !=[]:
            absEps = dict_options["absTol"]
            model.stateVarValues = numpy.empty((len(model.xBounds), len(model.xBounds[0])))
            for i in range(0, len(model.xBounds)):
                for j in range(0, len(model.xBounds[i])):
                    if isinstance(model.xBounds[i][j], numpy.ndarray):
                        midPoint = (model.xBounds[i][j][0] + model.xBounds[i][j][1]) / 2.0
                    else:
                        midPoint = float(mpmath.mpf(model.xBounds[i][j].mid))
                        if midPoint != 0: 
                            model.stateVarValues[i][j]= midPoint
                        else :
                            model.stateVarValues[i][j] = absEps

    if "Block" in res_solver.keys():
        block = res_solver["Block"]
    
        if block.xBounds_tot !=[]:
            absEps = dict_options["absTol"]
            for i in block.colPerm:

                if isinstance(block.xBounds_tot[i], numpy.ndarray):
                    midPoint = (block.xBounds_tot[i][0] + block.xBounds_tot[i][1]) / 2.0
                else:
                    midPoint = float(mpmath.mpf(block.xBounds_tot[i].mid))
                if midPoint != 0: 
                    block.x_tot[i]= midPoint
                else :
                    block.x_tot[i] = absEps