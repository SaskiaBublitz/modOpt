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
    dict_options = {"fileName": "VanDerWaals_r50",
                    "iterMaxNewton": 15,
                    "machEpsRelNewton": 2.22e-16,
                    "machEpsAbsNewton": 2.22e-16,
                    "absTolX": 2.22e-16, #numpy.finfo(numpy.float).eps
                    "relTolX": 2.22e-16,
                    "absTolF": 2.22e-16,
                    "relTolF": 2.22e-2,
                    "resolution": 50,
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
    e0_v = x[0]
    e0_dPdv_crit = x[1]
    e0_v_ph = x[2]



# Getting parameter values:
    e0_P = p[0]
    e0_R = p[1]
    e0_T = p[2]
    e0_a = p[3]
    e0_b = p[4]
    e0_P_crit = p[5]
    e0_v_crit = p[6]



   # Getting function values:



# Solve equation system for given x:
    f= [
    (e0_P) *(((e0_v-e0_b)) *(((e0_v))**(2.0)))-((((e0_R) *(e0_T))) *(((e0_v))**(2.0))-(e0_a) *((e0_v-e0_b))) ,
    (e0_dPdv_crit) *((((e0_v-e0_b))**(2.0)) *(((e0_v))**(3.0)))-(-(((e0_R) *(e0_T))/(e0_P_crit)) *(((e0_v))**(3.0))+(((2.0) *(e0_a))/(e0_P_crit)) *(((e0_v-e0_b))**(2.0))) ,
    e0_v_ph-(e0_v-e0_v_crit) 

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

    x = numpy.empty(3)
    xInitial = numpy.empty((3), dtype = object) 
    parameter = numpy.empty(7)
    dict_variables = {}

    # Iteration variable initializatio
    x[0] = 10.0 	# e0_v
    x[1] = 0.0 	# e0_dPdv_crit
    x[2] = 0.0 	# e0_v_ph


    # Constant parameter setting:
    parameter[0] = 215000.0 	# e0_P
    parameter[1] = 8.314 	# e0_R
    parameter[2] = 427.85 	# e0_T
    parameter[3] = 3.789 	# e0_a
    parameter[4] = 2.37E-4 	# e0_b
    parameter[5] = 1.9987E7 	# e0_P_crit
    parameter[6] = 7.11E-4 	# e0_v_crit


    # Get symbolic quantities:
    xSymbolic = sympy.symbols('e0_v e0_dPdv_crit e0_v_ph ')
    fSymbolic = getSymbolicFunctions(xSymbolic, parameter)

    # Intial variable bounds:
    xInitial[0] = mpmath.mpi(2.37E-4, 1.0E9)  	# e0_v
    xInitial[1] = mpmath.mpi(-1.0E9, 0.0)  	# e0_dPdv_crit
    xInitial[2] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_v_ph


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
