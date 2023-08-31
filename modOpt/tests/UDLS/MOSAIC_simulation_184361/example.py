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

"""
***************************************************
User specifications
***************************************************
"""

def main():

# Solver settings:
    bxrd_options = {"fileName": "example",
                    "savePath": './results',
                    "redStepMax": 1,
                    "maxBoxNo": 1,
                    "absTol": 1.0e-8,  #numpy.finfo(numpy.float).eps
                    "relTol": 1.0e-3,
                    "resolution": 8,
                    "parallelBoxes": False,
                    "parallelVariables": False,
                    "cpuCountBoxes": 2,
                    "cpuCountVariables": 2,
                    "affineArithmetic": False,
                    "hcMethod": 'HC4',  # 'HC4', 'None'
                    "newtonMethod": 'newton',  # 'newton', 'None'
                    "newtonPoint": 'center',  # 'center', '3P', 'condJ'
                    "preconditioning": 'pivotAll',  # inverseCentered, inversePoint, diagInverseCentered, pivotAll, 'None' 
                    "bcMethod": 'bnormal',  # 'bnormal', 'None'
                    "tightBounds": True,
                    "splitBox": 'tearVar',  # 'tearVar', 'forecastTear', 'forecastSplit', 'leastChanged', 'None'
                    "cutBox": 'tear',  # 'tear', 'all', 'None'
                    "considerDisconti": False,
                    "hybridApproach": True,
                    "debugMode": True,
                    "timer": True,
                    "analysis": True,
                    "decomp": 'DM',  # 'DM', 'None'
}

    smpl_options = {"smplNo": 0,
                    "smplNoBest": 1,  # only considered if smplNo > 1
                    "smplMethod": 'sobol' #sobol, hammersley, latin_hypercube, optuna
}

    num_options = {"solver": 'newton',  # 'newton', 'SLSQP', 'ipopt', 'fsolve', 'TNC', 'matlab-fsolve', 'matlab-fsolve-mscript','casadi-ipopt'
                "mode": 1, #  relevant for ipopt and SLSQP 1 = minimization of function residuals, 2 = equality constraints, constant objective
                "FTOL": 1.0e-6,
                "termination": 'all_solutions',  # 'all_solutions', 'one_solution'
                "scaling": 'None',  # 'MC29', 'MC77', 'None', only considered in newton solver
                "scalingProcedure": 'None',  # 'block_init', 'block_iter', 'tot_init', 'tot_iter'
                "iterMax": 1000,
}


# Hybrid approach or box reduction only:
    if not bxrd_options["hybridApproach"]:
        smpl_options = None
        num_options = None

# Model initialization:
    initialModel, dict_variables, dict_equations = getEquationsVariablesAndParameters(bxrd_options)
# Decomposition:
    if bxrd_options["decomp"] != 'None':     
        mod.decomposeSystem(initialModel, dict_equations, dict_variables, bxrd_options)
    else: initialModel.updateToPermutation(rowPerm = range(len(initialModel.fSymbolic)),
                                           colPerm = range(len(initialModel.xSymbolic)), 
                                           blocks = [range(len(initialModel.fSymbolic))])
# Bound reduction:   
    res_solver = moc.reduceVariableBounds(initialModel, bxrd_options, 
                                          smpl_options, num_options)

# Start value generation:    
    moi.setStateVarValuesToMidPointOfIntervals(res_solver, bxrd_options)
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



# Getting parameter values:



   # Getting function values:



# Solve equation system for given x:
    f= [
    ((e0_x))**(3.0)-(10.0) *(e0_y)+e0_z-(0.0) ,
    sympy.exp((e0_x) *(e0_y))-(2.0) *(e0_z)-1.0-(0.0) ,
    e0_x+((e0_y))**(2.0)+(3.0) *(((e0_z))**(3.0))-10.0-(0.0) 

]
    return f

def getEquationsVariablesAndParameters(bxrd_options):
    """ initialize linear model of the mathematical problem and orders equations
    and variable information into dictionaries        

    Args:
        bxrd_options:      dictionary with user settings

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

    x = numpy.empty(0)
    xInitial = numpy.empty((0), dtype = object) 
    parameter = numpy.empty(0)
    dict_variables = {}
    dict_equations = {}

    # Iteration variable initializatio


    # Constant parameter setting:


    # Get symbolic quantities:
    xSymbolic = sympy.symbols('')
    fSymbolic = getSymbolicFunctions(xSymbolic, parameter)

    # Intial variable bounds:


    Jcasadi, fcasadi = mod.getCasadiJandF(xSymbolic, fSymbolic)
    model = Model(x, xInitial, xSymbolic, fSymbolic, parameter, Jcasadi, fcasadi, getSymbolicFunctions)
    initial_f = model.getFunctionValues()
    for i in range(len(x)):
        dict_variables[xSymbolic[i]] = [[[float(x[i])], 
                      [float(mpmath.mpf(xInitial[i].a))], 
                      [float(mpmath.mpf(xInitial[i].b))]],
                      i, i]

    for i, f in enumerate(model.fSymbolic):
        dict_equations[fSymbolic[i]] = [initial_f[i], i, i, 1]
        model.functions.append(Function(f, model.xSymbolic, bxrd_options["affineArithmetic"], True))
        moc.sort_fId_to_varIds(i, model.functions[i].glb_ID, model.dict_varId_fIds)
        
    return model, dict_variables, dict_equations

"""
***************************************************
Call of main method
***************************************************
"""
# Invoke main method:
if __name__ == "__main__": main()   
