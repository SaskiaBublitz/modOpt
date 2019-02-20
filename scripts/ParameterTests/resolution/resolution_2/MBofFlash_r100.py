""" ModOpt for NLE Evaluations from MOSAICmodeling
 
Author: Saskia Bublitz (saskia.bublitz@tu-berlin.de)
Date: 12.02.2019
"""

"""
***************************************************
Import packages
***************************************************
"""

import mpmath
import numpy
import sympy
import modOpt
import modOpt.constraints as moc
import modOpt.initialization as moi
import modOpt.decomposition as mod

"""
***************************************************
User specifications
***************************************************
"""

def main():
    
# Solver settings:
    dict_options = {"fileName": "MBofFlash_r100",
                    "iterMaxNewton": 15,
                    "machEpsRelNewton": 2.22e-14,
                    "machEpsAbsNewton": 2.22e-14,
                    "absTolX": 2.22e-14, #numpy.finfo(numpy.float).eps
                    "relTolX": 2.22e-14,
                    "absTolF": 2.22e-14,
                    "relTolF": 2.22e-2,
                    "resolution": 100,
                    "Debug-Modus": False,
                    "NoOfNonChangingValues": 3,
                    'timer': True,
                    'method': 'complete',#'complete', 'partial'
                    'analysis': True}

# Model initialization:
    initialModel, dict_variables = getEquationsVariablesAndParameters(dict_options)


# Bound reduction:   
    modelWithReducedBounds, iterNo, t = moc.reduceVariableBounds(initialModel, 
                                                             dict_options)
  
# Start value generation:    
    moi.arithmeticMean.setStateVarValuesToMidPointOfIntervals(modelWithReducedBounds,
                                                              dict_options["absTolX"])
    
    dict_variables = moc.updateDictToModel(dict_variables, 
                                                  modelWithReducedBounds)

 # Result export:    
    moc.writeResults(dict_options["fileName"], dict_variables, t, iterNo)
    
    if dict_options['analysis'] == True:
        moc.analyseResults(dict_options["fileName"], initialModel.xSymbolic, 
                       initialModel.xBounds[0], modelWithReducedBounds.xBounds)


"""
***************************************************
Methods
***************************************************
"""

def getSymbolicFunctions(x, p):
    """ User specified Equation system 
    
    Args:
        x:          sympy array with symbolic state variable bounds
        p:          numpy array with parameter values
    
    Returns:
        f:          sympy array with symbolic residual bounds
        
    """
# Getting variable values:
    e0_x_F_i2 = x[0]
    e0_x_i2 = x[1]
    e0_y_i2 = x[2]
    e0_L = x[3]
    e0_V = x[4]



# Getting parameter values:
    e0_x_F_i1 = p[0]
    e0_x_i1 = p[1]
    e0_y_i1 = p[2]
    e0_F = p[3]



   # Getting function values:



# Solve equation system for given x:
    f= [
    (e0_x_F_i1+e0_x_F_i2)-(1.0) ,
    (e0_x_i1+e0_x_i2)-(1.0) ,
    (e0_y_i1+e0_y_i2)-(1.0) ,
    0.0-((e0_F) *(e0_x_F_i1)-(e0_V) *(e0_y_i1)-(e0_L) *(e0_x_i1)) ,
    0.0-((e0_F) *(e0_x_F_i2)-(e0_V) *(e0_y_i2)-(e0_L) *(e0_x_i2)) 

]
    return f

def getEquationsVariablesAndParameters(dict_options):
    """
    Initialize model of the mathematical problem and orders variable information 
    into a dictionary        
        
    Return:     
        dict_variables (dict):    uses variables as keys and has following 
                                  values in order: variable value, globalId,
                                  permId, scaling factor                                 
        initial_model (obj):      instance of type model with initialized values 
                                  from the user

    """

    x = numpy.empty(5)
    xInitial = numpy.empty((5), dtype = object) 
    parameter = numpy.empty(4)
    dict_variables = {}

    # Iteration variable initializatio
    x[0] = 0.5 	# e0_x_F_i2
    x[1] = 0.5 	# e0_x_i2
    x[2] = 0.5 	# e0_y_i2
    x[3] = 75.0 	# e0_L
    x[4] = 75.0 	# e0_V


    # Constant parameter setting:
    parameter[0] = 0.4 	# e0_x_F_i1
    parameter[1] = 0.1 	# e0_x_i1
    parameter[2] = 0.9 	# e0_y_i1
    parameter[3] = 150.0 	# e0_F


    # Get symbolic quantities:
    xSymbolic = sympy.symbols('e0_x_F_i2 e0_x_i2 e0_y_i2 e0_L e0_V ')
    fSymbolic = getSymbolicFunctions(xSymbolic, parameter)

    # Intial variable bounds:
    xInitial[0] = mpmath.mpi(0.0, 1.0)  	# e0_x_F_i2
    xInitial[1] = mpmath.mpi(0.0, 1.0)  	# e0_x_i2
    xInitial[2] = mpmath.mpi(0.0, 1.0)  	# e0_y_i2
    xInitial[3] = mpmath.mpi(0.0, 200.0)  	# e0_L
    xInitial[4] = mpmath.mpi(0.0, 200.0)  	# e0_V


    for i in range(0,len(x)):
        dict_variables[xSymbolic[i]] = [[float(x[i])], 
                      [float(mpmath.mpf(xInitial[i].a))], 
                      [float(mpmath.mpf(xInitial[i].b))],
                      i]

    if dict_options["method"] =='complete':
        return modOpt.Model(x, xInitial, xSymbolic, fSymbolic, parameter,
                            []), dict_variables

    if dict_options["method"] =='partial':
        jCasadi, fCasadi = mod.dM.getCasadiJandF(xSymbolic, fSymbolic)
        return modOpt.Model(x, xInitial, xSymbolic, fSymbolic, parameter, 
                            jCasadi), dict_variables



"""
***************************************************
Call of main method
***************************************************
"""
# Invoke main method:
if __name__ == "__main__": main()   
