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
                    "sampling method": 'sobol' #sobol, hammersley, latin_hypercube
}

    solv_options = {"solver": 'newton', # 'newton', 'SLSQP', 'trust-constr', 'ipopt, fsolve, TNC', 'matlab-fsolve', 'matlab-fsolve-mscript'
                "mode": 1, # relevant for ipopt 1 = minimization of function residuals, 2 = equality constraints, constant objective
                "FTOL": 1e-6,
                "scaling": "MC77",
                "scaling procedure": "block_iter", #"tot_init", "block_init", "tot_iter", "block_iter",
                "iterMax": 1000,
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
    e0_h_V_i4 = x[0]
    e0_T = x[1]
    e0_p_LV_i1 = x[2]
    e0_p_LV_i2 = x[3]
    e0_p_LV_i3 = x[4]
    e0_p_LV_i4 = x[5]
    e0_L_i1_j1 = x[6]
    e0_L_i1_j2 = x[7]
    e0_L_i1_j3 = x[8]
    e0_L_i1_j4 = x[9]
    e0_L_i2_j1 = x[10]
    e0_L_i2_j2 = x[11]
    e0_L_i2_j3 = x[12]
    e0_L_i2_j4 = x[13]
    e0_L_i3_j1 = x[14]
    e0_L_i3_j2 = x[15]
    e0_L_i3_j3 = x[16]
    e0_L_i3_j4 = x[17]
    e0_L_i4_j1 = x[18]
    e0_L_i4_j2 = x[19]
    e0_L_i4_j3 = x[20]
    e0_L_i4_j4 = x[21]
    e0_kf = x[22]
    e0_Keq = x[23]
    e0_greek_gamma_i1 = x[24]
    e0_greek_gamma_i2 = x[25]
    e0_greek_gamma_i3 = x[26]
    e0_greek_gamma_i4 = x[27]
    e0_r = x[28]
    e0_x_i1 = x[29]
    e0_x_i2 = x[30]
    e0_x_i3 = x[31]
    e0_x_i4 = x[32]
    e0_N_L = x[33]
    e0_N_V = x[34]
    e0_y_i1 = x[35]
    e0_y_i2 = x[36]
    e0_y_i3 = x[37]
    e0_y_i4 = x[38]
    e0_h_F = x[39]
    e0_h_L = x[40]
    e0_h_V = x[41]
    e0_h_L_i1 = x[42]
    e0_h_L_i2 = x[43]
    e0_h_L_i3 = x[44]
    e0_h_L_i4 = x[45]
    e0_h_F_i1 = x[46]
    e0_h_F_i2 = x[47]
    e0_h_F_i3 = x[48]
    e0_h_F_i4 = x[49]
    e0_h_V_i1 = x[50]
    e0_h_V_i2 = x[51]
    e0_h_V_i3 = x[52]



# Getting parameter values:
    e0_A_Antoine_i1 = p[0]
    e0_A_Antoine_i2 = p[1]
    e0_C_Antoine_i3 = p[2]
    e0_T_F = p[3]
    e0_T_ref_i1 = p[4]
    e0_T_ref_i2 = p[5]
    e0_T_ref_i3 = p[6]
    e0_T_ref_i4 = p[7]
    e0_h_ref_i1 = p[8]
    e0_h_ref_i2 = p[9]
    e0_h_ref_i3 = p[10]
    e0_h_ref_i4 = p[11]
    e0_C_Antoine_i4 = p[12]
    e0_T_LV_i1 = p[13]
    e0_T_LV_i2 = p[14]
    e0_T_LV_i3 = p[15]
    e0_T_LV_i4 = p[16]
    e0_c_Vp_i1 = p[17]
    e0_c_Vp_i2 = p[18]
    e0_c_Vp_i3 = p[19]
    e0_c_Vp_i4 = p[20]
    e0_h_LV_i1 = p[21]
    e0_h_LV_i2 = p[22]
    e0_h_LV_i3 = p[23]
    e0_h_LV_i4 = p[24]
    e0_A_Wilson_i1_j1 = p[25]
    e0_A_Wilson_i1_j2 = p[26]
    e0_A_Wilson_i1_j3 = p[27]
    e0_A_Antoine_i3 = p[28]
    e0_A_Wilson_i1_j4 = p[29]
    e0_A_Wilson_i2_j1 = p[30]
    e0_A_Wilson_i2_j2 = p[31]
    e0_A_Wilson_i2_j3 = p[32]
    e0_A_Wilson_i2_j4 = p[33]
    e0_A_Wilson_i3_j1 = p[34]
    e0_A_Wilson_i3_j2 = p[35]
    e0_A_Wilson_i3_j3 = p[36]
    e0_A_Wilson_i3_j4 = p[37]
    e0_A_Wilson_i4_j1 = p[38]
    e0_A_Antoine_i4 = p[39]
    e0_A_Wilson_i4_j2 = p[40]
    e0_A_Wilson_i4_j3 = p[41]
    e0_A_Wilson_i4_j4 = p[42]
    e0_B_Antoine_i1 = p[43]
    e0_R = p[44]
    e0_B_Antoine_i2 = p[45]
    e0_v_i1 = p[46]
    e0_v_i2 = p[47]
    e0_v_i3 = p[48]
    e0_v_i4 = p[49]
    e0_B_Antoine_i3 = p[50]
    e0_greek_nu_i1 = p[51]
    e0_greek_nu_i2 = p[52]
    e0_greek_nu_i3 = p[53]
    e0_greek_nu_i4 = p[54]
    e0_HU = p[55]
    e0_B_Antoine_i4 = p[56]
    e0_N_F = p[57]
    e0_x_F_i1 = p[58]
    e0_x_F_i2 = p[59]
    e0_x_F_i3 = p[60]
    e0_x_F_i4 = p[61]
    e0_C_Antoine_i1 = p[62]
    e0_c_Lp_i1 = p[63]
    e0_c_Lp_i2 = p[64]
    e0_c_Lp_i3 = p[65]
    e0_c_Lp_i4 = p[66]
    e0_p = p[67]
    e0_C_Antoine_i2 = p[68]



   # Getting function values:



# Solve equation system for given x:
    f= [
    sympy.log(e0_p_LV_i1)-(e0_A_Antoine_i1+(e0_B_Antoine_i1)/(e0_T+e0_C_Antoine_i1)) ,
    sympy.log(e0_p_LV_i2)-(e0_A_Antoine_i2+(e0_B_Antoine_i2)/(e0_T+e0_C_Antoine_i2)) ,
    sympy.log(e0_p_LV_i3)-(e0_A_Antoine_i3+(e0_B_Antoine_i3)/(e0_T+e0_C_Antoine_i3)) ,
    sympy.log(e0_p_LV_i4)-(e0_A_Antoine_i4+(e0_B_Antoine_i4)/(e0_T+e0_C_Antoine_i4)) ,
    e0_L_i1_j1-(((e0_v_i1)/(e0_v_i1)) *(sympy.exp((-(1.0) *(e0_A_Wilson_i1_j1))/((e0_R) *(e0_T))))) ,
    e0_L_i1_j2-(((e0_v_i2)/(e0_v_i1)) *(sympy.exp((-(1.0) *(e0_A_Wilson_i1_j2))/((e0_R) *(e0_T))))) ,
    e0_L_i1_j3-(((e0_v_i3)/(e0_v_i1)) *(sympy.exp((-(1.0) *(e0_A_Wilson_i1_j3))/((e0_R) *(e0_T))))) ,
    e0_L_i1_j4-(((e0_v_i4)/(e0_v_i1)) *(sympy.exp((-(1.0) *(e0_A_Wilson_i1_j4))/((e0_R) *(e0_T))))) ,
    e0_L_i2_j1-(((e0_v_i1)/(e0_v_i2)) *(sympy.exp((-(1.0) *(e0_A_Wilson_i2_j1))/((e0_R) *(e0_T))))) ,
    e0_L_i2_j2-(((e0_v_i2)/(e0_v_i2)) *(sympy.exp((-(1.0) *(e0_A_Wilson_i2_j2))/((e0_R) *(e0_T))))) ,
    e0_L_i2_j3-(((e0_v_i3)/(e0_v_i2)) *(sympy.exp((-(1.0) *(e0_A_Wilson_i2_j3))/((e0_R) *(e0_T))))) ,
    e0_L_i2_j4-(((e0_v_i4)/(e0_v_i2)) *(sympy.exp((-(1.0) *(e0_A_Wilson_i2_j4))/((e0_R) *(e0_T))))) ,
    e0_L_i3_j1-(((e0_v_i1)/(e0_v_i3)) *(sympy.exp((-(1.0) *(e0_A_Wilson_i3_j1))/((e0_R) *(e0_T))))) ,
    e0_L_i3_j2-(((e0_v_i2)/(e0_v_i3)) *(sympy.exp((-(1.0) *(e0_A_Wilson_i3_j2))/((e0_R) *(e0_T))))) ,
    e0_L_i3_j3-(((e0_v_i3)/(e0_v_i3)) *(sympy.exp((-(1.0) *(e0_A_Wilson_i3_j3))/((e0_R) *(e0_T))))) ,
    e0_L_i3_j4-(((e0_v_i4)/(e0_v_i3)) *(sympy.exp((-(1.0) *(e0_A_Wilson_i3_j4))/((e0_R) *(e0_T))))) ,
    e0_L_i4_j1-(((e0_v_i1)/(e0_v_i4)) *(sympy.exp((-(1.0) *(e0_A_Wilson_i4_j1))/((e0_R) *(e0_T))))) ,
    e0_L_i4_j2-(((e0_v_i2)/(e0_v_i4)) *(sympy.exp((-(1.0) *(e0_A_Wilson_i4_j2))/((e0_R) *(e0_T))))) ,
    e0_L_i4_j3-(((e0_v_i3)/(e0_v_i4)) *(sympy.exp((-(1.0) *(e0_A_Wilson_i4_j3))/((e0_R) *(e0_T))))) ,
    e0_L_i4_j4-(((e0_v_i4)/(e0_v_i4)) *(sympy.exp((-(1.0) *(e0_A_Wilson_i4_j4))/((e0_R) *(e0_T))))) ,
    e0_kf-((((9.732) *(((10.0))**(8.0)))/(3600.0)) *(sympy.exp((-6287.7)/(e0_T)))) ,
    e0_Keq-((2.32) *(sympy.exp((782.98)/(e0_T)))) ,
    e0_r-((e0_kf) *(((e0_greek_gamma_i1) *((e0_x_i1) *((e0_greek_gamma_i2) *(e0_x_i2)))-((e0_greek_gamma_i3) *((e0_x_i3) *((e0_greek_gamma_i4) *(e0_x_i4))))/(e0_Keq)))) ,
    sympy.log(e0_greek_gamma_i1)-(1.0-sympy.log(((e0_x_i1) *(e0_L_i1_j1)+(e0_x_i2) *(e0_L_i1_j2)+(e0_x_i3) *(e0_L_i1_j3)+(e0_x_i4) *(e0_L_i1_j4)))-(((e0_x_i1) *(e0_L_i1_j1))/(((e0_x_i1) *(e0_L_i1_j1)+(e0_x_i2) *(e0_L_i1_j2)+(e0_x_i3) *(e0_L_i1_j3)+(e0_x_i4) *(e0_L_i1_j4)))+((e0_x_i2) *(e0_L_i2_j1))/(((e0_x_i1) *(e0_L_i2_j1)+(e0_x_i2) *(e0_L_i2_j2)+(e0_x_i3) *(e0_L_i2_j3)+(e0_x_i4) *(e0_L_i2_j4)))+((e0_x_i3) *(e0_L_i3_j1))/(((e0_x_i1) *(e0_L_i3_j1)+(e0_x_i2) *(e0_L_i3_j2)+(e0_x_i3) *(e0_L_i3_j3)+(e0_x_i4) *(e0_L_i3_j4)))+((e0_x_i4) *(e0_L_i4_j1))/(((e0_x_i1) *(e0_L_i4_j1)+(e0_x_i2) *(e0_L_i4_j2)+(e0_x_i3) *(e0_L_i4_j3)+(e0_x_i4) *(e0_L_i4_j4))))) ,
    sympy.log(e0_greek_gamma_i2)-(1.0-sympy.log(((e0_x_i1) *(e0_L_i2_j1)+(e0_x_i2) *(e0_L_i2_j2)+(e0_x_i3) *(e0_L_i2_j3)+(e0_x_i4) *(e0_L_i2_j4)))-(((e0_x_i1) *(e0_L_i1_j2))/(((e0_x_i1) *(e0_L_i1_j1)+(e0_x_i2) *(e0_L_i1_j2)+(e0_x_i3) *(e0_L_i1_j3)+(e0_x_i4) *(e0_L_i1_j4)))+((e0_x_i2) *(e0_L_i2_j2))/(((e0_x_i1) *(e0_L_i2_j1)+(e0_x_i2) *(e0_L_i2_j2)+(e0_x_i3) *(e0_L_i2_j3)+(e0_x_i4) *(e0_L_i2_j4)))+((e0_x_i3) *(e0_L_i3_j2))/(((e0_x_i1) *(e0_L_i3_j1)+(e0_x_i2) *(e0_L_i3_j2)+(e0_x_i3) *(e0_L_i3_j3)+(e0_x_i4) *(e0_L_i3_j4)))+((e0_x_i4) *(e0_L_i4_j2))/(((e0_x_i1) *(e0_L_i4_j1)+(e0_x_i2) *(e0_L_i4_j2)+(e0_x_i3) *(e0_L_i4_j3)+(e0_x_i4) *(e0_L_i4_j4))))) ,
    sympy.log(e0_greek_gamma_i3)-(1.0-sympy.log(((e0_x_i1) *(e0_L_i3_j1)+(e0_x_i2) *(e0_L_i3_j2)+(e0_x_i3) *(e0_L_i3_j3)+(e0_x_i4) *(e0_L_i3_j4)))-(((e0_x_i1) *(e0_L_i1_j3))/(((e0_x_i1) *(e0_L_i1_j1)+(e0_x_i2) *(e0_L_i1_j2)+(e0_x_i3) *(e0_L_i1_j3)+(e0_x_i4) *(e0_L_i1_j4)))+((e0_x_i2) *(e0_L_i2_j3))/(((e0_x_i1) *(e0_L_i2_j1)+(e0_x_i2) *(e0_L_i2_j2)+(e0_x_i3) *(e0_L_i2_j3)+(e0_x_i4) *(e0_L_i2_j4)))+((e0_x_i3) *(e0_L_i3_j3))/(((e0_x_i1) *(e0_L_i3_j1)+(e0_x_i2) *(e0_L_i3_j2)+(e0_x_i3) *(e0_L_i3_j3)+(e0_x_i4) *(e0_L_i3_j4)))+((e0_x_i4) *(e0_L_i4_j3))/(((e0_x_i1) *(e0_L_i4_j1)+(e0_x_i2) *(e0_L_i4_j2)+(e0_x_i3) *(e0_L_i4_j3)+(e0_x_i4) *(e0_L_i4_j4))))) ,
    sympy.log(e0_greek_gamma_i4)-(1.0-sympy.log(((e0_x_i1) *(e0_L_i4_j1)+(e0_x_i2) *(e0_L_i4_j2)+(e0_x_i3) *(e0_L_i4_j3)+(e0_x_i4) *(e0_L_i4_j4)))-(((e0_x_i1) *(e0_L_i1_j4))/(((e0_x_i1) *(e0_L_i1_j1)+(e0_x_i2) *(e0_L_i1_j2)+(e0_x_i3) *(e0_L_i1_j3)+(e0_x_i4) *(e0_L_i1_j4)))+((e0_x_i2) *(e0_L_i2_j4))/(((e0_x_i1) *(e0_L_i2_j1)+(e0_x_i2) *(e0_L_i2_j2)+(e0_x_i3) *(e0_L_i2_j3)+(e0_x_i4) *(e0_L_i2_j4)))+((e0_x_i3) *(e0_L_i3_j4))/(((e0_x_i1) *(e0_L_i3_j1)+(e0_x_i2) *(e0_L_i3_j2)+(e0_x_i3) *(e0_L_i3_j3)+(e0_x_i4) *(e0_L_i3_j4)))+((e0_x_i4) *(e0_L_i4_j4))/(((e0_x_i1) *(e0_L_i4_j1)+(e0_x_i2) *(e0_L_i4_j2)+(e0_x_i3) *(e0_L_i4_j3)+(e0_x_i4) *(e0_L_i4_j4))))) ,
    0.0-((e0_N_F) *(e0_x_F_i1)-(e0_N_L) *(e0_x_i1)-(e0_N_V) *(e0_y_i1)+(e0_greek_nu_i1) *(e0_r)) ,
    0.0-((e0_N_F) *(e0_x_F_i2)-(e0_N_L) *(e0_x_i2)-(e0_N_V) *(e0_y_i2)+(e0_greek_nu_i2) *(e0_r)) ,
    0.0-((e0_N_F) *(e0_x_F_i3)-(e0_N_L) *(e0_x_i3)-(e0_N_V) *(e0_y_i3)+(e0_greek_nu_i3) *(e0_r)) ,
    0.0-((e0_N_F) *(e0_x_F_i4)-(e0_N_L) *(e0_x_i4)-(e0_N_V) *(e0_y_i4)+(e0_greek_nu_i4) *(e0_r)) ,
    0.0-((e0_N_F) *(e0_h_F)-(e0_N_L) *(e0_h_L)-(e0_N_V) *(e0_h_V)) ,
    (e0_y_i1) *(e0_p)-((e0_greek_gamma_i1) *((e0_x_i1) *(e0_p_LV_i1))) ,
    (e0_y_i2) *(e0_p)-((e0_greek_gamma_i2) *((e0_x_i2) *(e0_p_LV_i2))) ,
    (e0_y_i3) *(e0_p)-((e0_greek_gamma_i3) *((e0_x_i3) *(e0_p_LV_i3))) ,
    (e0_y_i4) *(e0_p)-((e0_greek_gamma_i4) *((e0_x_i4) *(e0_p_LV_i4))) ,
    e0_h_L-(((e0_x_i1) *(e0_h_L_i1)+(e0_x_i2) *(e0_h_L_i2)+(e0_x_i3) *(e0_h_L_i3)+(e0_x_i4) *(e0_h_L_i4))) ,
    e0_h_F-(((e0_x_F_i1) *(e0_h_F_i1)+(e0_x_F_i2) *(e0_h_F_i2)+(e0_x_F_i3) *(e0_h_F_i3)+(e0_x_F_i4) *(e0_h_F_i4))) ,
    e0_h_V-(((e0_y_i1) *(e0_h_V_i1)+(e0_y_i2) *(e0_h_V_i2)+(e0_y_i3) *(e0_h_V_i3)+(e0_y_i4) *(e0_h_V_i4))) ,
    e0_h_F_i1-((e0_c_Lp_i1) *((e0_T_F-e0_T_ref_i1))+e0_h_ref_i1) ,
    e0_h_F_i2-((e0_c_Lp_i2) *((e0_T_F-e0_T_ref_i2))+e0_h_ref_i2) ,
    e0_h_F_i3-((e0_c_Lp_i3) *((e0_T_F-e0_T_ref_i3))+e0_h_ref_i3) ,
    e0_h_F_i4-((e0_c_Lp_i4) *((e0_T_F-e0_T_ref_i4))+e0_h_ref_i4) ,
    e0_h_L_i1-((e0_c_Lp_i1) *((e0_T-e0_T_ref_i1))+e0_h_ref_i1) ,
    e0_h_L_i2-((e0_c_Lp_i2) *((e0_T-e0_T_ref_i2))+e0_h_ref_i2) ,
    e0_h_L_i3-((e0_c_Lp_i3) *((e0_T-e0_T_ref_i3))+e0_h_ref_i3) ,
    e0_h_L_i4-((e0_c_Lp_i4) *((e0_T-e0_T_ref_i4))+e0_h_ref_i4) ,
    e0_h_V_i1-((e0_c_Vp_i1) *((e0_T-e0_T_LV_i1))+e0_h_LV_i1+(e0_c_Lp_i1) *((e0_T_LV_i1-e0_T_ref_i1))+e0_h_ref_i1) ,
    e0_h_V_i2-((e0_c_Vp_i2) *((e0_T-e0_T_LV_i2))+e0_h_LV_i2+(e0_c_Lp_i2) *((e0_T_LV_i2-e0_T_ref_i2))+e0_h_ref_i2) ,
    e0_h_V_i3-((e0_c_Vp_i3) *((e0_T-e0_T_LV_i3))+e0_h_LV_i3+(e0_c_Lp_i3) *((e0_T_LV_i3-e0_T_ref_i3))+e0_h_ref_i3) ,
    e0_h_V_i4-((e0_c_Vp_i4) *((e0_T-e0_T_LV_i4))+e0_h_LV_i4+(e0_c_Lp_i4) *((e0_T_LV_i4-e0_T_ref_i4))+e0_h_ref_i4) ,
    (e0_x_i1+e0_x_i2+e0_x_i3+e0_x_i4)-(1.0) ,
    (e0_y_i1+e0_y_i2+e0_y_i3+e0_y_i4)-(1.0) 

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

    x = numpy.empty(53)
    xInitial = numpy.empty((53), dtype = object) 
    parameter = numpy.empty(69)
    dict_variables = {}
    dict_equations = {}

    # Iteration variable initializatio
    x[0] = 1.0 	# e0_h_V_i4
    x[1] = 323.15 	# e0_T
    x[2] = 1.0 	# e0_p_LV_i1
    x[3] = 1.0 	# e0_p_LV_i2
    x[4] = 1.0 	# e0_p_LV_i3
    x[5] = 1.0 	# e0_p_LV_i4
    x[6] = 1.0 	# e0_L_i1_j1
    x[7] = 1.0 	# e0_L_i1_j2
    x[8] = 1.0 	# e0_L_i1_j3
    x[9] = 1.0 	# e0_L_i1_j4
    x[10] = 1.0 	# e0_L_i2_j1
    x[11] = 1.0 	# e0_L_i2_j2
    x[12] = 1.0 	# e0_L_i2_j3
    x[13] = 1.0 	# e0_L_i2_j4
    x[14] = 1.0 	# e0_L_i3_j1
    x[15] = 1.0 	# e0_L_i3_j2
    x[16] = 1.0 	# e0_L_i3_j3
    x[17] = 1.0 	# e0_L_i3_j4
    x[18] = 1.0 	# e0_L_i4_j1
    x[19] = 1.0 	# e0_L_i4_j2
    x[20] = 1.0 	# e0_L_i4_j3
    x[21] = 1.0 	# e0_L_i4_j4
    x[22] = 1.0 	# e0_kf
    x[23] = 1.0 	# e0_Keq
    x[24] = 1.0 	# e0_greek_gamma_i1
    x[25] = 1.0 	# e0_greek_gamma_i2
    x[26] = 1.0 	# e0_greek_gamma_i3
    x[27] = 1.0 	# e0_greek_gamma_i4
    x[28] = 1.0 	# e0_r
    x[29] = 0.25 	# e0_x_i1
    x[30] = 0.25 	# e0_x_i2
    x[31] = 0.25 	# e0_x_i3
    x[32] = 0.25 	# e0_x_i4
    x[33] = 0.5 	# e0_N_L
    x[34] = 0.4 	# e0_N_V
    x[35] = 0.25 	# e0_y_i1
    x[36] = 0.25 	# e0_y_i2
    x[37] = 0.25 	# e0_y_i3
    x[38] = 0.25 	# e0_y_i4
    x[39] = 1.0 	# e0_h_F
    x[40] = 1.0 	# e0_h_L
    x[41] = 1.0 	# e0_h_V
    x[42] = 1.0 	# e0_h_L_i1
    x[43] = 1.0 	# e0_h_L_i2
    x[44] = 1.0 	# e0_h_L_i3
    x[45] = 1.0 	# e0_h_L_i4
    x[46] = 1.0 	# e0_h_F_i1
    x[47] = 1.0 	# e0_h_F_i2
    x[48] = 1.0 	# e0_h_F_i3
    x[49] = 1.0 	# e0_h_F_i4
    x[50] = 1.0 	# e0_h_V_i1
    x[51] = 1.0 	# e0_h_V_i2
    x[52] = 1.0 	# e0_h_V_i3


    # Constant parameter setting:
    parameter[0] = 22.1001 	# e0_A_Antoine_i1
    parameter[1] = 23.4999 	# e0_A_Antoine_i2
    parameter[2] = -53.46 	# e0_C_Antoine_i3
    parameter[3] = 373.15 	# e0_T_F
    parameter[4] = 298.15 	# e0_T_ref_i1
    parameter[5] = 298.15 	# e0_T_ref_i2
    parameter[6] = 298.15 	# e0_T_ref_i3
    parameter[7] = 298.15 	# e0_T_ref_i4
    parameter[8] = 483520.0 	# e0_h_ref_i1
    parameter[9] = 238400.0 	# e0_h_ref_i2
    parameter[10] = 445890.0 	# e0_h_ref_i3
    parameter[11] = 285830.0 	# e0_h_ref_i4
    parameter[12] = -45.343 	# e0_C_Antoine_i4
    parameter[13] = 391.2 	# e0_T_LV_i1
    parameter[14] = 337.8 	# e0_T_LV_i2
    parameter[15] = 330.0 	# e0_T_LV_i3
    parameter[16] = 373.15 	# e0_T_LV_i4
    parameter[17] = 50.0 	# e0_c_Vp_i1
    parameter[18] = 45.0 	# e0_c_Vp_i2
    parameter[19] = 86.0 	# e0_c_Vp_i3
    parameter[20] = 35.0 	# e0_c_Vp_i4
    parameter[21] = 875160.0 	# e0_h_LV_i1
    parameter[22] = 725700.0 	# e0_h_LV_i2
    parameter[23] = 1583000.0 	# e0_h_LV_i3
    parameter[24] = 40650.0 	# e0_h_LV_i4
    parameter[25] = 0.0 	# e0_A_Wilson_i1_j1
    parameter[26] = 2535.2019 	# e0_A_Wilson_i1_j2
    parameter[27] = 1123.1444 	# e0_A_Wilson_i1_j3
    parameter[28] = 21.152 	# e0_A_Antoine_i3
    parameter[29] = 237.5248 	# e0_A_Wilson_i1_j4
    parameter[30] = -547.5248 	# e0_A_Wilson_i2_j1
    parameter[31] = 0.0 	# e0_A_Wilson_i2_j2
    parameter[32] = 813.1843 	# e0_A_Wilson_i2_j3
    parameter[33] = 107.3832 	# e0_A_Wilson_i2_j4
    parameter[34] = -696.5031 	# e0_A_Wilson_i3_j1
    parameter[35] = -31.1932 	# e0_A_Wilson_i3_j2
    parameter[36] = 0.0 	# e0_A_Wilson_i3_j3
    parameter[37] = 645.7225 	# e0_A_Wilson_i3_j4
    parameter[38] = 658.0266 	# e0_A_Wilson_i4_j1
    parameter[39] = 23.2256 	# e0_A_Antoine_i4
    parameter[40] = 469.5509 	# e0_A_Wilson_i4_j2
    parameter[41] = 1918.232 	# e0_A_Wilson_i4_j3
    parameter[42] = 0.0 	# e0_A_Wilson_i4_j4
    parameter[43] = -3654.62 	# e0_B_Antoine_i1
    parameter[44] = 8.314 	# e0_R
    parameter[45] = -3643.3136 	# e0_B_Antoine_i2
    parameter[46] = 57.54 	# e0_v_i1
    parameter[47] = 44.44 	# e0_v_i2
    parameter[48] = 79.84 	# e0_v_i3
    parameter[49] = 18.07 	# e0_v_i4
    parameter[50] = -2662.78 	# e0_B_Antoine_i3
    parameter[51] = -1.0 	# e0_greek_nu_i1
    parameter[52] = -1.0 	# e0_greek_nu_i2
    parameter[53] = 1.0 	# e0_greek_nu_i3
    parameter[54] = 1.0 	# e0_greek_nu_i4
    parameter[55] = 1.0 	# e0_HU
    parameter[56] = -3835.18 	# e0_B_Antoine_i4
    parameter[57] = 1.0 	# e0_N_F
    parameter[58] = 0.55 	# e0_x_F_i1
    parameter[59] = 0.25 	# e0_x_F_i2
    parameter[60] = 0.0 	# e0_x_F_i3
    parameter[61] = 0.2 	# e0_x_F_i4
    parameter[62] = -45.392 	# e0_C_Antoine_i1
    parameter[63] = 123.0 	# e0_c_Lp_i1
    parameter[64] = 80.0 	# e0_c_Lp_i2
    parameter[65] = 140.0 	# e0_c_Lp_i3
    parameter[66] = 75.6 	# e0_c_Lp_i4
    parameter[67] = 100000.0 	# e0_p
    parameter[68] = -33.434 	# e0_C_Antoine_i2


    # Get symbolic quantities:
    xSymbolic = sympy.symbols('e0_h_V_i4 e0_T e0_p_LV_i1 e0_p_LV_i2 e0_p_LV_i3 e0_p_LV_i4 e0_L_i1_j1 e0_L_i1_j2 e0_L_i1_j3 e0_L_i1_j4 e0_L_i2_j1 e0_L_i2_j2 e0_L_i2_j3 e0_L_i2_j4 e0_L_i3_j1 e0_L_i3_j2 e0_L_i3_j3 e0_L_i3_j4 e0_L_i4_j1 e0_L_i4_j2 e0_L_i4_j3 e0_L_i4_j4 e0_kf e0_Keq e0_greek_gamma_i1 e0_greek_gamma_i2 e0_greek_gamma_i3 e0_greek_gamma_i4 e0_r e0_x_i1 e0_x_i2 e0_x_i3 e0_x_i4 e0_N_L e0_N_V e0_y_i1 e0_y_i2 e0_y_i3 e0_y_i4 e0_h_F e0_h_L e0_h_V e0_h_L_i1 e0_h_L_i2 e0_h_L_i3 e0_h_L_i4 e0_h_F_i1 e0_h_F_i2 e0_h_F_i3 e0_h_F_i4 e0_h_V_i1 e0_h_V_i2 e0_h_V_i3 ')
    fSymbolic = getSymbolicFunctions(xSymbolic, parameter)

    # Intial variable bounds:
    xInitial[0] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_h_V_i4
    xInitial[1] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_T
    xInitial[2] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_p_LV_i1
    xInitial[3] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_p_LV_i2
    xInitial[4] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_p_LV_i3
    xInitial[5] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_p_LV_i4
    xInitial[6] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_L_i1_j1
    xInitial[7] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_L_i1_j2
    xInitial[8] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_L_i1_j3
    xInitial[9] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_L_i1_j4
    xInitial[10] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_L_i2_j1
    xInitial[11] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_L_i2_j2
    xInitial[12] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_L_i2_j3
    xInitial[13] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_L_i2_j4
    xInitial[14] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_L_i3_j1
    xInitial[15] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_L_i3_j2
    xInitial[16] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_L_i3_j3
    xInitial[17] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_L_i3_j4
    xInitial[18] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_L_i4_j1
    xInitial[19] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_L_i4_j2
    xInitial[20] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_L_i4_j3
    xInitial[21] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_L_i4_j4
    xInitial[22] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_kf
    xInitial[23] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_Keq
    xInitial[24] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_greek_gamma_i1
    xInitial[25] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_greek_gamma_i2
    xInitial[26] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_greek_gamma_i3
    xInitial[27] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_greek_gamma_i4
    xInitial[28] = mpmath.mpi(-1.0e9, 1.0E9)  	# e0_r
    xInitial[29] = mpmath.mpi(0.0, 1.0E9)  	# e0_x_i1
    xInitial[30] = mpmath.mpi(0.0, 1.0E9)   	# e0_x_i2
    xInitial[31] = mpmath.mpi(0.0, 1.0E9)    	# e0_x_i3
    xInitial[32] = mpmath.mpi(0.0, 1.0E9)   	# e0_x_i4
    xInitial[33] = mpmath.mpi(0.0, 1.0E9)  	# e0_N_L
    xInitial[34] = mpmath.mpi(0.0, 1.0E9)  	# e0_N_V
    xInitial[35] = mpmath.mpi(0.0, 1.0E9)   	# e0_y_i1
    xInitial[36] = mpmath.mpi(0.0, 1.0E9)   	# e0_y_i2
    xInitial[37] = mpmath.mpi(0.0, 1.0E9)   	# e0_y_i3
    xInitial[38] = mpmath.mpi(0.0, 1.0E9)   	# e0_y_i4
    xInitial[39] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_h_F
    xInitial[40] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_h_L
    xInitial[41] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_h_V
    xInitial[42] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_h_L_i1
    xInitial[43] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_h_L_i2
    xInitial[44] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_h_L_i3
    xInitial[45] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_h_L_i4
    xInitial[46] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_h_F_i1
    xInitial[47] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_h_F_i2
    xInitial[48] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_h_F_i3
    xInitial[49] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_h_F_i4
    xInitial[50] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_h_V_i1
    xInitial[51] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_h_V_i2
    xInitial[52] = mpmath.mpi(-1.0E9, 1.0E9)  	# e0_h_V_i3


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

