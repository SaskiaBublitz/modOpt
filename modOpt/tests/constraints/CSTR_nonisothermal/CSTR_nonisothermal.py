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
    bxrd_options = {"fileName": str(sys.argv[1]),
                    "savePath": './results',
                    "redStepMax": int(sys.argv[2]),
                    "maxBoxNo": 1,
                    "absTol": 1.0e-8, #numpy.finfo(numpy.float).eps
                    "relTol": 1.0e-3,
                    "resolution": int(sys.argv[3]),
                    "parallelBoxes": bool(int(sys.argv[4])),
                    "parallelVariables": False,
                    "bcMethod": str(sys.argv[5]),#'bnormal',
                    "affineArithmetic": bool(int(sys.argv[6])),
		            "tightBounds": bool(int(sys.argv[7])),
                    "hcMethod": str(sys.argv[8]), # 'HC4', 'None'
                    "newtonMethod": str(sys.argv[9]), # 'newton', 'detNewton', 'newton3P'
                    "newtonPoint": str(sys.argv[10]),  
                    "preconditioning": str(sys.argv[11]),
                    "splitBox": str(sys.argv[12]), # 'tearVar', 'largestDer', 'forecastSplit', 'leastChanged'
                    "considerDisconti": bool(int(sys.argv[13])),
                    "cutBox": str(sys.argv[14]),
                    "debugMode": False,
                    "timer": True,
                    "analysis": True,
                    "cpuCountBoxes": int(sys.argv[15]),
                    "cpuCountVariables": 2,
                    "hybridApproach": True,
}

    sampling_options = {"smplNo": 0,
                    "smplBest": 1,
                    "smplMethod method": 'sobol', #sobol, hammersley, latin_hypercube, optuna
}

    solv_options = {"solver": 'newton', # 'newton', 'SLSQP', 'trust-constr', 'ipopt, fsolve, TNC', 'matlab-fsolve', 'matlab-fsolve-mscript'
                "mode": 1, # relevant for ipopt 1 = minimization of function residuals, 2 = equality constraints, constant objective
                "FTOL": 1e-8,
                "iterMax": 100,
                "scaling": "None",
                "scalingProcedure": "block_iter", #"tot_init", "block_init", "tot_iter", "block_iter",
                "termination": "all_solutions", # "one_solution"
                }


# Hybrid approach or box reduction only:
    if not bxrd_options["hybridApproach"]:
        sampling_options = None
        solv_options = None

# Model initialization:
    initialModel, dict_variables, dict_equations = getEquationsVariablesAndParameters(bxrd_options)

# Decomposition:
    bxrd_options["decomp"] = 'DM'     
    mod.decomposeSystem(initialModel, dict_equations, dict_variables, bxrd_options)
    #else: initialModel.updateToPermutation(rowPerm = range(len(initialModel.xSymbolic)), 
    #                                       colPerm = range(len(initialModel.xSymbolic)), 
    #                                       blocks = [range(len(initialModel.xSymbolic))])

# Bound reduction:   
    res_solver = moc.reduceVariableBounds(initialModel, bxrd_options, 
                                          sampling_options, solv_options)
  
# Start value generation:    
    moi.setStateVarValuesToMidPointOfIntervals(res_solver,
                                                              bxrd_options)
    
    moc.updateDictToModel(dict_variables, res_solver)

 # Result export:         
    moc.trackErrors(res_solver, bxrd_options)
    moc.writeResults(bxrd_options, dict_variables, res_solver)
            
    if bxrd_options['analysis'] == True:
        moc.analyseResults(bxrd_options, res_solver)



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
    e0_c_c1 = x[0]
    e0_T = x[1]
    e0_X_c1 = x[2]
    e0_cFeed_c1 = x[3]
    e0_r = x[4]



# Getting parameter values:
    e0_greek_DeltaHR = p[0]
    e0_greek_rho = p[1]
    e0_M_c1 = p[2]
    e0_xFeed_c1 = p[3]
    e0_E = p[4]
    e0_R = p[5]
    e0_TFeed = p[6]
    e0_cp = p[7]
    e0_QFeed = p[8]
    e0_mCat = p[9]



   # Getting function values:



# Solve equation system for given x:
    f= [
    (e0_T-e0_TFeed)/(e0_X_c1)-((-(e0_greek_DeltaHR) *(e0_cFeed_c1))/((e0_greek_rho) *(e0_cp))) ,
    (e0_QFeed) *(e0_X_c1)-(((1.0-e0_X_c1)) *((e0_mCat) *(e0_r))) ,
    e0_cFeed_c1-((e0_xFeed_c1) *((e0_greek_rho)/(e0_M_c1))) ,
    e0_X_c1-((e0_cFeed_c1-e0_c_c1)/(e0_cFeed_c1)) ,
    #e0_r-((1.3) *((((10.0))**(11.0)) *(sympy.exp((-e0_E)/((e0_R) *(e0_T)))))) 
    e0_r-(1.3) *(numpy.log(1.0e11) + sympy.exp((-e0_E)/((e0_R) *(e0_T)))) 

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

    x = numpy.empty(5)
    xInitial = numpy.empty((5), dtype = object) 
    parameter = numpy.empty(10)
    dict_variables = {}
    dict_equations = {}

    # Iteration variable initializatio
    x[0] = 0.0 	# e0_c_c1
    x[1] = 0.0 	# e0_T
    x[2] = 0.0 	# e0_X_c1
    x[3] = 0.0 	# e0_cFeed_c1
    x[4] = 0.0 	# e0_r


    # Constant parameter setting:
    parameter[0] = -200000.0 	# e0_greek_DeltaHR
    parameter[1] = 0.85 	# e0_greek_rho
    parameter[2] = 210.0 	# e0_M_c1
    parameter[3] = 0.3 	# e0_xFeed_c1
    parameter[4] = 135518.2 	# e0_E
    parameter[5] = 8.314 	# e0_R
    parameter[6] = 582.0 	# e0_TFeed
    parameter[7] = 1.75 	# e0_cp
    parameter[8] = 30.0 	# e0_QFeed
    parameter[9] = 20.0 	# e0_mCat


    # Get symbolic quantities:
    xSymbolic = sympy.symbols('e0_c_c1 e0_T e0_X_c1 e0_cFeed_c1 e0_r ')
    fSymbolic = getSymbolicFunctions(xSymbolic, parameter)

    # Intial variable bounds:
    xInitial[0] = mpmath.mpi(0.0, 1.0E9)  	# e0_c_c1
    xInitial[1] = mpmath.mpi(0.0, 1.0E9)  	# e0_T
    xInitial[2] = mpmath.mpi(0.0, 1.0E9)  	# e0_X_c1
    xInitial[3] = mpmath.mpi(0.0, 1.0E9)  	# e0_cFeed_c1
    xInitial[4] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_r


    Jcasadi, fcasadi = mod.getCasadiJandF(xSymbolic, fSymbolic)
    model = Model(x, xInitial, xSymbolic, fSymbolic, parameter, Jcasadi, fcasadi, getSymbolicFunctions)
    initial_f = model.getFunctionValues()
    for i in range(0,len(x)):
        dict_variables[xSymbolic[i]] = [[[float(x[i])], 
                      [float(mpmath.mpf(xInitial[i].a))], 
                      [float(mpmath.mpf(xInitial[i].b))]],
                      i, i]
        dict_equations[fSymbolic[i]] = [initial_f[i], i, i, 1]

    for i, f in enumerate(model.fSymbolic):
        model.functions.append(Function(f, model.xSymbolic, dict_options["affineArithmetic"], True))
        moc.sort_fId_to_varIds(i, model.functions[i].glb_ID, model.dict_varId_fIds)
        
    return model, dict_variables, dict_equations


"""
***************************************************
Call of main method
***************************************************
"""
# Invoke main method:
if __name__ == "__main__": main()   

