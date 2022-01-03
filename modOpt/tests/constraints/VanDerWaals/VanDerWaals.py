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
from modOpt.model import Model
from modOpt.constraints.function import Function
import modOpt.initialization as moi
import modOpt.decomposition as mod
import modOpt.constraints as moc
import sys

"""
***************************************************
User specifications
***************************************************
"""

def main():

# Solver settings:
    dict_options = {"fileName": str(sys.argv[1]),
                    "save_path": './results',
                    "redStepMax": int(sys.argv[2]),
                    "maxBoxNo": 1,
                    "absTol": 1.0e-8, #numpy.finfo(numpy.float).eps
                    "relTol": 1.0e-3,
                    "resolution": int(sys.argv[3]),
                    "Parallel Branches": bool(int(sys.argv[4])),
                    "Parallel Variables": False,
                    "Parallel b's": False,
                    "bc_method": str(sys.argv[5]),#'bnormal',
                    "Affine_arithmetic": bool(int(sys.argv[6])),
		            "tight_bounds": bool(int(sys.argv[7])),
                    "hc_method": str(sys.argv[8]), # 'HC4', 'None'
                    "newton_method": str(sys.argv[9]), # 'newton', 'detNewton', 'newton3P'
                    "newton_point": str(sys.argv[10]),  
                    "preconditioning": str(sys.argv[11]),
                    "InverseOrHybrid": 'Hybrid', # 'Hybrid', 'both', 'None'
                    "combined_algorithm": False, #'None', 'HC4_bnormal', 'detNewton_HC4', 'HC4_3PNewton_bnormal'
                    "split_Box": str(sys.argv[12]), # 'TearVar', 'LargestDer', 'forecastSplit', 'LeastChanged'
                    "consider_disconti": bool(int(sys.argv[13])),
                    "cut_Box": str(sys.argv[14]),
                    "Debug-Modus": False,
                    "timer": True,
                    "analysis": True,
                    "CPU count Branches": int(sys.argv[15]),
                    "CPU count Variables": 2,
                    "CPU count b's":2,
                    "decomp": str(sys.argv[16]), # DM, None
                    "hybrid_approach": True,
}

    sampling_options = {"number of samples": 0,
                    "sampleNo_min_resiudal": 1,
                    "sampling method": 'sobol', #sobol, hammersley, latin_hypercube, ax_optimization
                    "max_iter": 10,
}

    solv_options = {"solver": 'newton', # 'newton', 'SLSQP', 'trust-constr', 'ipopt, fsolve, TNC', 'matlab-fsolve', 'matlab-fsolve-mscript'
                "mode": 1, # relevant for ipopt 1 = minimization of function residuals, 2 = equality constraints, constant objective
                "FTOL": 1e-8,
                "iterMax": 100,
                "iterMax_tear": 10,
                "parallel_boxes": False,
                "CPU count": 2}   


# Hybrid approach or box reduction only:
    if not dict_options["hybrid_approach"]:
        sampling_options = None
        solv_options = None

# Model initialization:
    initialModel, dict_variables, dict_equations = getEquationsVariablesAndParameters(dict_options)

# Decomposition:
    if dict_options["decomp"] != 'None':     
        mod.decomposeSystem(initialModel, dict_equations, dict_variables, dict_options)
    else: initialModel.updateToPermutation(rowPerm = range(len(initialModel.xSymbolic)), 
                                           colPerm = range(len(initialModel.xSymbolic)), 
                                           blocks = [range(len(initialModel.xSymbolic))])

# Bound reduction:   
    res_solver = moc.reduceVariableBounds(initialModel, dict_options, 
                                          sampling_options, solv_options)
  
# Start value generation:    
    moi.setStateVarValuesToMidPointOfIntervals(res_solver,
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
    dict_equations = {}

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
    xInitial[1] = mpmath.mpi(-1.0E9, 0.0)  	# e0_dPdv_crit
    xInitial[2] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_v_ph


    Jcasadi, fcasadi = mod.getCasadiJandF(xSymbolic, fSymbolic)
      
    model = Model(x, xInitial, xSymbolic, fSymbolic, parameter, Jcasadi, fcasadi, 
                  getSymbolicFunctions)
    initial_f = model.getFunctionValues()
    for i in range(0,len(x)):
        dict_variables[xSymbolic[i]] = [[[float(x[i])], 
                      [float(mpmath.mpf(xInitial[i].a))], 
                      [float(mpmath.mpf(xInitial[i].b))]],
                      i, i]
        dict_equations[fSymbolic[i]] = [initial_f[i], i, i, 1]
        

    for i, f in enumerate(model.fSymbolic):
        model.functions.append(Function(f, model.xSymbolic, dict_options["Affine_arithmetic"], True))
        moc.sort_fId_to_varIds(i, model.functions[i].glb_ID, model.dict_varId_fIds)
        
    return model, dict_variables, dict_equations

"""
***************************************************
Call of main method
***************************************************
"""
# Invoke main method:
if __name__ == "__main__": main()   

