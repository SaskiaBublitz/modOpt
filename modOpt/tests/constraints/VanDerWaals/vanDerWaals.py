""" ModOpt for NLE Evaluations from MOSAICmodeling
 
Author: Saskia Bublitz (saskia.bublitz@tu-berlin.de)
Date: 12.02.2019
"""

"""
***************************************************
Import packages
***************************************************
"""
import sys
import mpmath
import numpy
import sympy
from modOpt.model import Model
import modOpt.constraints as moc
import modOpt.initialization as moi

"""
***************************************************
User specifications
***************************************************
"""

def main():
    
# Solver settings:
    dict_options = {"fileName": str(sys.argv[1]),
                    "redStepMax": 10,
                    "maxBoxNo": 32,
                    "absTol": 2.22e-7, #numpy.finfo(numpy.float).eps
                    "relTol": 2.22e-7,
                    "resolution": 50,
                    "Parallel Branches": bool(sys.argv[5]),
                    "Parallel Variables": bool(sys.argv[4]),
                    "Parallel b's": False,
                    "newton_method": str(sys.argv[3]),
                    "bc_method": str(sys.argv[2]),#'b_normal', 'b_tight', 'b_normal_newton', 'b_normal_detNewton', 'b_normal_3PNewton'
                    "Debug-Modus": False,
                    "timer": True,
                    "analysis": True,
                    "CPU count Branches": 4,
                    "CPU count Variables": 2,
                    "CPU count b's":2
}
# Model initialization:
    initialModel, dict_variables = getEquationsVariablesAndParameters(dict_options)

# Bound reduction:   
    res_solver = moc.reduceVariableBounds(initialModel, dict_options)
  
# Start value generation:    
    moi.arithmeticMean.setStateVarValuesToMidPointOfIntervals(res_solver,
                                                              dict_options)
    
    moc.updateDictToModel(dict_variables, res_solver)

 # Result export:         
    moc.trackErrors(initialModel, res_solver, dict_options)
    moc.writeResults(dict_options, dict_variables, res_solver)
            
    if dict_options['analysis'] == True:
        moc.analyseResults(dict_options, initialModel, res_solver)


"""
***************************************************
Methods
***************************************************
"""

def getSymbolicFunctions(x, p):
    """ User specified Equation system 
    
    Args:
        :x:          sympy array with symbolic state variable bounds
        :p:          numpy array with parameter values
    
    Return:
        :f:          sympy array with symbolic residual bounds
        
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
    e0_P-(((e0_R) *(e0_T))/(e0_v-e0_b)-(e0_a)/(((e0_v))**(2.0))) ,
    (e0_dPdv_crit) *(((e0_v-e0_b))**(2.0))-(-((e0_R) *(e0_T))/(e0_P_crit)+(((2.0) *(e0_a))/(e0_P_crit)) *((((e0_v-e0_b))**(2.0))/(((e0_v))**(3.0)))) ,
    e0_v_ph-(e0_v-e0_v_crit) 

]
    return f

def getEquationsVariablesAndParameters(dict_options):
    """ initialize linear model of the mathematical problem and orders equations
    and variable information into dictionaries        

    Args:
        dict_options:      dictionary with user settings

    Return:
        dict_equations:    dictionary uses equations as keys and has following 
                           values in order: function value, globalId, permId, 
                           scaling factor       
        dict_variables:    dictionary uses variables as keys and has following 
                           values in order: variable value, globalId, permId, 
                           scaling factor                                 
        dict_parameters:   dictionary uses parameters as keys and has following 
                           values in order: variable value, globalId, parameter value
        initial_model:     instance of type model at initial point
        model:             instance of type model that is updated to
                           the user-specified restructuring settings
                                  
    """

    x = numpy.empty(3)
    xInitial = numpy.empty((3), dtype = object) 
    parameter = numpy.empty(7)
    dict_variables = {}

    # Iteration variable initializatio
    x[0] = 0.5 	# e0_v
    x[1] = 0.5 	# e0_dPdv_crit
    x[2] = 0.5 	# e0_v_ph


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
    xInitial[0] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_v
    xInitial[1] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_dPdv_crit
    xInitial[2] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_v_ph


    for i in range(0,len(x)):
        dict_variables[xSymbolic[i]] = [[float(x[i])], 
                      [float(mpmath.mpf(xInitial[i].a))], 
                      [float(mpmath.mpf(xInitial[i].b))],
                      i]

    for i in range(0,len(x)):
        dict_variables[xSymbolic[i]] = [[float(x[i])], 
                      [float(mpmath.mpf(xInitial[i].a))], 
                      [float(mpmath.mpf(xInitial[i].b))],
                      i]

    model = Model(x, xInitial, xSymbolic, fSymbolic, parameter, [])
    moc.nestBlocks(model)
    return model, dict_variables


"""
***************************************************
Call of main method
***************************************************
"""
# Invoke main method:
if __name__ == "__main__": main()   

