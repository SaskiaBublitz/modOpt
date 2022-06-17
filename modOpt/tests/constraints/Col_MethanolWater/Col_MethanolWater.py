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
def fun_87125(std_X,std_A_Lhnd100,std_B_Lhnd100,std_C_Lhnd100,std_D_Lhnd100,std_E_Lhnd100,std_T_ho,std_h_Lno): # fun_87125__Enthalpy_DIPPR100
	return (std_A_Lhnd100) *((std_X-std_T_ho))+((std_B_Lhnd100)/(2.0)) *((((std_X))**(2.0)-((std_T_ho))**(2.0)))+((std_C_Lhnd100)/(3.0)) *((((std_X))**(3.0)-((std_T_ho))**(3.0)))+((std_D_Lhnd100)/(4.0)) *((((std_X))**(4.0)-((std_T_ho))**(4.0)))+((std_E_Lhnd100)/(5.0)) *((((std_X))**(5.0)-((std_T_ho))**(5.0)))+std_h_Lno
def fun_87127(std_X,std_A_LVhnd106,std_B_LVhnd106,std_C_LVhnd106,std_D_LVhnd106,std_E_LVhnd106,std_T_crit): # fun_87127__DIPPR106___Evaporation_molar
	return (std_A_LVhnd106) *(((1.0-(std_X)/(std_T_crit)))**(std_B_LVhnd106+(std_C_LVhnd106) *((std_X)/(std_T_crit))+(std_D_LVhnd106) *((((std_X)/(std_T_crit)))**(2.0))+(std_E_LVhnd106) *((((std_X)/(std_T_crit)))**(3.0))))
def fun_87129(std_X,std_A_pLVd101,std_B_pLVd101,std_C_pLVd101,std_D_pLVd101,std_E_pLVd101): # fun_87129__DIPPR101
	return sympy.exp(std_A_pLVd101+(std_B_pLVd101)/(std_X)+(std_C_pLVd101) *(sympy.log(std_X))+(std_D_pLVd101) *(((std_X))**(std_E_pLVd101)))

def getSymbolicFunctions(x, p):
    """ User specified Equation system 
    
    Args:
        :x:          sympy array with symbolic state variable bounds
        :p:          numpy array with parameter values
    
    Return:
        :f:          sympy array with symbolic residual bounds
        
    """

# Getting variable values:
    e0_F_Ln_tr2 = x[0]
    e0_F_Ln_tr3 = x[1]
    e0_F_Ln_tr4 = x[2]
    e0_F_Ln_tr5 = x[3]
    e0_F_Ln_tr6 = x[4]
    e0_F_Ln_tr7 = x[5]
    e0_F_Ln_tr8 = x[6]
    e0_F_Ln_tr9 = x[7]
    e0_F_Ln_tr10 = x[8]
    e0_F_Ln_tr11 = x[9]
    e0_F_Vn_tr1 = x[10]
    e0_F_Vn_tr2 = x[11]
    e0_F_Vn_tr3 = x[12]
    e0_F_Vn_tr4 = x[13]
    e0_F_Vn_tr5 = x[14]
    e0_F_Vn_tr6 = x[15]
    e0_F_Vn_tr7 = x[16]
    e0_F_Vn_tr8 = x[17]
    e0_h_Vn_tr0 = x[18]
    e0_F_Vn_tr9 = x[19]
    e0_F_Vn_tr10 = x[20]
    e0_T_tr1 = x[21]
    e0_T_tr2 = x[22]
    e0_T_tr3 = x[23]
    e0_T_tr4 = x[24]
    e0_T_tr5 = x[25]
    e0_T_tr6 = x[26]
    e0_T_tr7 = x[27]
    e0_T_tr8 = x[28]
    e0_p_tr0 = x[29]
    e0_T_tr9 = x[30]
    e0_T_tr10 = x[31]
    e0_h_Lfn = x[32]
    e0_h_Ln_tr2 = x[33]
    e0_h_Ln_tr3 = x[34]
    e0_h_Ln_tr4 = x[35]
    e0_h_Ln_tr5 = x[36]
    e0_xR_i1 = x[37]
    e0_h_Ln_tr6 = x[38]
    e0_h_Ln_tr7 = x[39]
    e0_h_Ln_tr8 = x[40]
    e0_h_Ln_tr9 = x[41]
    e0_h_Ln_tr10 = x[42]
    e0_h_Ln_tr11 = x[43]
    e0_xR_i2 = x[44]
    e0_x_i1_tr1 = x[45]
    e0_h_Vn_tr1 = x[46]
    e0_h_Vn_tr2 = x[47]
    e0_h_Vn_tr3 = x[48]
    e0_h_Vn_tr4 = x[49]
    e0_F_Ln_tr1 = x[50]
    e0_x_i2_tr1 = x[51]
    e0_h_Vn_tr5 = x[52]
    e0_h_Vn_tr6 = x[53]
    e0_h_Vn_tr7 = x[54]
    e0_h_Vn_tr8 = x[55]
    e0_h_Vn_tr9 = x[56]
    e0_h_Vn_tr10 = x[57]
    e0_p_tr1 = x[58]
    e0_p_tr2 = x[59]
    e0_p_tr3 = x[60]
    e0_p_tr4 = x[61]
    e0_y_i1_tr0 = x[62]
    e0_p_tr5 = x[63]
    e0_p_tr6 = x[64]
    e0_p_tr7 = x[65]
    e0_p_tr8 = x[66]
    e0_p_tr9 = x[67]
    e0_y_i2_tr0 = x[68]
    e0_x_tr2_i1 = x[69]
    e0_x_tr3_i1 = x[70]
    e0_x_tr4_i1 = x[71]
    e0_x_tr5_i1 = x[72]
    e0_x_tr6_i1 = x[73]
    e0_x_tr7_i1 = x[74]
    e0_x_tr8_i1 = x[75]
    e0_x_tr9_i1 = x[76]
    e0_x_tr10_i1 = x[77]
    e0_x_tr11_i1 = x[78]
    e0_x_tr2_i2 = x[79]
    e0_x_tr3_i2 = x[80]
    e0_x_tr4_i2 = x[81]
    e0_x_tr5_i2 = x[82]
    e0_x_tr6_i2 = x[83]
    e0_x_tr7_i2 = x[84]
    e0_x_tr8_i2 = x[85]
    e0_x_tr9_i2 = x[86]
    e0_x_tr10_i2 = x[87]
    e0_x_tr11_i2 = x[88]
    e0_y_tr1_i1 = x[89]
    e0_y_tr2_i1 = x[90]
    e0_y_tr3_i1 = x[91]
    e0_y_tr4_i1 = x[92]
    e0_y_tr5_i1 = x[93]
    e0_y_tr6_i1 = x[94]
    e0_y_tr7_i1 = x[95]
    e0_y_tr8_i1 = x[96]
    e0_y_tr9_i1 = x[97]
    e0_y_tr10_i1 = x[98]
    e0_y_tr1_i2 = x[99]
    e0_y_tr2_i2 = x[100]
    e0_y_tr3_i2 = x[101]
    e0_y_tr4_i2 = x[102]
    e0_y_tr5_i2 = x[103]
    e0_y_tr6_i2 = x[104]
    e0_y_tr7_i2 = x[105]
    e0_y_tr8_i2 = x[106]
    e0_y_tr9_i2 = x[107]
    e0_y_tr10_i2 = x[108]
    e0_F_LnC = x[109]
    e0_QC = x[110]
    e0_TC = x[111]
    e0_yC_i1 = x[112]
    e0_yC_i2 = x[113]
    e0_F_LnR = x[114]
    e0_F_Vn_tr0 = x[115]
    e0_QR = x[116]
    e0_h_Ln_tr1 = x[117]
    e0_h_LnR = x[118]



# Getting parameter values:
    e0_greek_gammaR_i1 = p[0]
    e0_greek_gammaR_i2 = p[1]
    e0_greek_phi_V_tr9_i2 = p[2]
    e0_greek_phi_V_tr10_i2 = p[3]
    e0_F_Lfn_tr1 = p[4]
    e0_F_Lfn_tr2 = p[5]
    e0_F_Lfn_tr3 = p[6]
    e0_F_Lfn_tr4 = p[7]
    e0_F_Lfn_tr5 = p[8]
    e0_F_Lfn_tr6 = p[9]
    e0_F_Lfn_tr7 = p[10]
    e0_F_Lfn_tr8 = p[11]
    e0_F_Lfn_tr9 = p[12]
    e0_F_Lfn_tr10 = p[13]
    e0_T_f = p[14]
    e0_p_tr10 = p[15]
    e0_x_f_i1 = p[16]
    e0_x_f_i2 = p[17]
    e0_greek_gammaC_i1 = p[18]
    e0_greek_gammaC_i2 = p[19]
    e0_R = p[20]
    e0_TR = p[21]
    e0_greek_Deltap = p[22]
    e0_greek_gamma_tr1_i1 = p[23]
    e0_greek_gamma_tr2_i1 = p[24]
    e0_greek_gamma_tr3_i1 = p[25]
    e0_greek_gamma_tr4_i1 = p[26]
    e0_greek_gamma_tr5_i1 = p[27]
    e0_greek_gamma_tr6_i1 = p[28]
    e0_greek_gamma_tr7_i1 = p[29]
    e0_greek_gamma_tr8_i1 = p[30]
    e0_greek_gamma_tr9_i1 = p[31]
    e0_greek_gamma_tr10_i1 = p[32]
    e0_greek_gamma_tr1_i2 = p[33]
    e0_greek_gamma_tr2_i2 = p[34]
    e0_greek_gamma_tr3_i2 = p[35]
    e0_greek_gamma_tr4_i2 = p[36]
    e0_greek_gamma_tr5_i2 = p[37]
    e0_greek_gamma_tr6_i2 = p[38]
    e0_greek_gamma_tr7_i2 = p[39]
    e0_greek_gamma_tr8_i2 = p[40]
    e0_greek_gamma_tr9_i2 = p[41]
    e0_greek_gamma_tr10_i2 = p[42]
    e0_greek_phi_V_tr1_i1 = p[43]
    e0_greek_phi_V_tr2_i1 = p[44]
    e0_greek_phi_V_tr3_i1 = p[45]
    e0_greek_phi_V_tr4_i1 = p[46]
    e0_greek_phi_V_tr5_i1 = p[47]
    e0_greek_phi_V_tr6_i1 = p[48]
    e0_greek_phi_V_tr7_i1 = p[49]
    e0_greek_phi_V_tr8_i1 = p[50]
    e0_greek_phi_V_tr9_i1 = p[51]
    e0_greek_phi_V_tr10_i1 = p[52]
    e0_greek_phi_V_tr1_i2 = p[53]
    e0_greek_phi_V_tr2_i2 = p[54]
    e0_greek_phi_V_tr3_i2 = p[55]
    e0_greek_phi_V_tr4_i2 = p[56]
    e0_greek_phi_V_tr5_i2 = p[57]
    e0_greek_phi_V_tr6_i2 = p[58]
    e0_greek_phi_V_tr7_i2 = p[59]
    e0_greek_phi_V_tr8_i2 = p[60]
    e0_A_Lhnd100_i1 = p[61]
    e0_A_Lhnd100_i2 = p[62]
    e0_A_pLVd101_i1 = p[63]
    e0_A_pLVd101_i2 = p[64]
    e0_A_LVhnd106_i1 = p[65]
    e0_A_LVhnd106_i2 = p[66]
    e0_B_Lhnd100_i1 = p[67]
    e0_B_Lhnd100_i2 = p[68]
    e0_B_pLVd101_i1 = p[69]
    e0_B_pLVd101_i2 = p[70]
    e0_B_LVhnd106_i1 = p[71]
    e0_B_LVhnd106_i2 = p[72]
    e0_C_Lhnd100_i1 = p[73]
    e0_C_Lhnd100_i2 = p[74]
    e0_C_pLVd101_i1 = p[75]
    e0_C_pLVd101_i2 = p[76]
    e0_C_LVhnd106_i1 = p[77]
    e0_C_LVhnd106_i2 = p[78]
    e0_D_Lhnd100_i1 = p[79]
    e0_D_Lhnd100_i2 = p[80]
    e0_D_pLVd101_i1 = p[81]
    e0_D_pLVd101_i2 = p[82]
    e0_D_LVhnd106_i1 = p[83]
    e0_D_LVhnd106_i2 = p[84]
    e0_E_Lhnd100_i1 = p[85]
    e0_E_Lhnd100_i2 = p[86]
    e0_E_pLVd101_i1 = p[87]
    e0_E_pLVd101_i2 = p[88]
    e0_E_LVhnd106_i1 = p[89]
    e0_E_LVhnd106_i2 = p[90]
    e0_T_crit_i1 = p[91]
    e0_T_ho_i1 = p[92]
    e0_T_crit_i2 = p[93]
    e0_T_ho_i2 = p[94]
    e0_h_sca = p[95]
    e0_h_Lno_i1 = p[96]
    e0_h_Lno_i2 = p[97]
    e0_p_sca = p[98]



   # Getting function values:
    e0_h_LVn_tr1_i1 = fun_87127(e0_T_tr1,e0_A_LVhnd106_i1,e0_B_LVhnd106_i1,e0_C_LVhnd106_i1,e0_D_LVhnd106_i1,e0_E_LVhnd106_i1,e0_T_crit_i1)
    e0_p_LV_tr6_i1 = fun_87129(e0_T_tr6,e0_A_pLVd101_i1,e0_B_pLVd101_i1,e0_C_pLVd101_i1,e0_D_pLVd101_i1,e0_E_pLVd101_i1)
    e0_h_LVn_tr8_i2 = fun_87127(e0_T_tr8,e0_A_LVhnd106_i2,e0_B_LVhnd106_i2,e0_C_LVhnd106_i2,e0_D_LVhnd106_i2,e0_E_LVhnd106_i2,e0_T_crit_i2)
    e0_p_LV_tr9_i2 = fun_87129(e0_T_tr9,e0_A_pLVd101_i2,e0_B_pLVd101_i2,e0_C_pLVd101_i2,e0_D_pLVd101_i2,e0_E_pLVd101_i2)
    e0_h_LVn_tr6_i1 = fun_87127(e0_T_tr6,e0_A_LVhnd106_i1,e0_B_LVhnd106_i1,e0_C_LVhnd106_i1,e0_D_LVhnd106_i1,e0_E_LVhnd106_i1,e0_T_crit_i1)
    e0_h_LVn_tr7_i2 = fun_87127(e0_T_tr7,e0_A_LVhnd106_i2,e0_B_LVhnd106_i2,e0_C_LVhnd106_i2,e0_D_LVhnd106_i2,e0_E_LVhnd106_i2,e0_T_crit_i2)
    e0_h_LVn_tr5_i1 = fun_87127(e0_T_tr5,e0_A_LVhnd106_i1,e0_B_LVhnd106_i1,e0_C_LVhnd106_i1,e0_D_LVhnd106_i1,e0_E_LVhnd106_i1,e0_T_crit_i1)
    e0_h_LnC_i1 = fun_87125(e0_TC,e0_A_Lhnd100_i1,e0_B_Lhnd100_i1,e0_C_Lhnd100_i1,e0_D_Lhnd100_i1,e0_E_Lhnd100_i1,e0_T_ho_i1,e0_h_Lno_i1)
    e0_h_LnC_i2 = fun_87125(e0_TC,e0_A_Lhnd100_i2,e0_B_Lhnd100_i2,e0_C_Lhnd100_i2,e0_D_Lhnd100_i2,e0_E_Lhnd100_i2,e0_T_ho_i2,e0_h_Lno_i2)
    e0_p_LV_tr5_i1 = fun_87129(e0_T_tr5,e0_A_pLVd101_i1,e0_B_pLVd101_i1,e0_C_pLVd101_i1,e0_D_pLVd101_i1,e0_E_pLVd101_i1)
    e0_p_LV_tr1_i2 = fun_87129(e0_T_tr1,e0_A_pLVd101_i2,e0_B_pLVd101_i2,e0_C_pLVd101_i2,e0_D_pLVd101_i2,e0_E_pLVd101_i2)
    e0_h_LVn_tr4_i2 = fun_87127(e0_T_tr4,e0_A_LVhnd106_i2,e0_B_LVhnd106_i2,e0_C_LVhnd106_i2,e0_D_LVhnd106_i2,e0_E_LVhnd106_i2,e0_T_crit_i2)
    e0_h_Ln_tr2_i1 = fun_87125(e0_T_tr2,e0_A_Lhnd100_i1,e0_B_Lhnd100_i1,e0_C_Lhnd100_i1,e0_D_Lhnd100_i1,e0_E_Lhnd100_i1,e0_T_ho_i1,e0_h_Lno_i1)
    e0_h_Ln_tr10_i1 = fun_87125(e0_T_tr10,e0_A_Lhnd100_i1,e0_B_Lhnd100_i1,e0_C_Lhnd100_i1,e0_D_Lhnd100_i1,e0_E_Lhnd100_i1,e0_T_ho_i1,e0_h_Lno_i1)
    e0_h_Ln_tr8_i2 = fun_87125(e0_T_tr8,e0_A_Lhnd100_i2,e0_B_Lhnd100_i2,e0_C_Lhnd100_i2,e0_D_Lhnd100_i2,e0_E_Lhnd100_i2,e0_T_ho_i2,e0_h_Lno_i2)
    e0_h_LVn_tr2_i1 = fun_87127(e0_T_tr2,e0_A_LVhnd106_i1,e0_B_LVhnd106_i1,e0_C_LVhnd106_i1,e0_D_LVhnd106_i1,e0_E_LVhnd106_i1,e0_T_crit_i1)
    e0_h_LVn_tr5_i2 = fun_87127(e0_T_tr5,e0_A_LVhnd106_i2,e0_B_LVhnd106_i2,e0_C_LVhnd106_i2,e0_D_LVhnd106_i2,e0_E_LVhnd106_i2,e0_T_crit_i2)
    e0_h_LnR_i1 = fun_87125(e0_TR,e0_A_Lhnd100_i1,e0_B_Lhnd100_i1,e0_C_Lhnd100_i1,e0_D_Lhnd100_i1,e0_E_Lhnd100_i1,e0_T_ho_i1,e0_h_Lno_i1)
    e0_h_Ln_tr9_i2 = fun_87125(e0_T_tr9,e0_A_Lhnd100_i2,e0_B_Lhnd100_i2,e0_C_Lhnd100_i2,e0_D_Lhnd100_i2,e0_E_Lhnd100_i2,e0_T_ho_i2,e0_h_Lno_i2)
    e0_p_LV_tr2_i2 = fun_87129(e0_T_tr2,e0_A_pLVd101_i2,e0_B_pLVd101_i2,e0_C_pLVd101_i2,e0_D_pLVd101_i2,e0_E_pLVd101_i2)
    e0_h_Ln_tr9_i1 = fun_87125(e0_T_tr9,e0_A_Lhnd100_i1,e0_B_Lhnd100_i1,e0_C_Lhnd100_i1,e0_D_Lhnd100_i1,e0_E_Lhnd100_i1,e0_T_ho_i1,e0_h_Lno_i1)
    e0_h_Ln_tr7_i1 = fun_87125(e0_T_tr7,e0_A_Lhnd100_i1,e0_B_Lhnd100_i1,e0_C_Lhnd100_i1,e0_D_Lhnd100_i1,e0_E_Lhnd100_i1,e0_T_ho_i1,e0_h_Lno_i1)
    e0_h_Ln_tr10_i2 = fun_87125(e0_T_tr10,e0_A_Lhnd100_i2,e0_B_Lhnd100_i2,e0_C_Lhnd100_i2,e0_D_Lhnd100_i2,e0_E_Lhnd100_i2,e0_T_ho_i2,e0_h_Lno_i2)
    e0_h_LVn_tr3_i1 = fun_87127(e0_T_tr3,e0_A_LVhnd106_i1,e0_B_LVhnd106_i1,e0_C_LVhnd106_i1,e0_D_LVhnd106_i1,e0_E_LVhnd106_i1,e0_T_crit_i1)
    e0_h_LVn_tr4_i1 = fun_87127(e0_T_tr4,e0_A_LVhnd106_i1,e0_B_LVhnd106_i1,e0_C_LVhnd106_i1,e0_D_LVhnd106_i1,e0_E_LVhnd106_i1,e0_T_crit_i1)
    e0_h_LVnR_i1 = fun_87127(e0_TR,e0_A_LVhnd106_i1,e0_B_LVhnd106_i1,e0_C_LVhnd106_i1,e0_D_LVhnd106_i1,e0_E_LVhnd106_i1,e0_T_crit_i1)
    e0_h_Ln_tr4_i1 = fun_87125(e0_T_tr4,e0_A_Lhnd100_i1,e0_B_Lhnd100_i1,e0_C_Lhnd100_i1,e0_D_Lhnd100_i1,e0_E_Lhnd100_i1,e0_T_ho_i1,e0_h_Lno_i1)
    e0_p_LV_tr9_i1 = fun_87129(e0_T_tr9,e0_A_pLVd101_i1,e0_B_pLVd101_i1,e0_C_pLVd101_i1,e0_D_pLVd101_i1,e0_E_pLVd101_i1)
    e0_h_LVn_tr9_i1 = fun_87127(e0_T_tr9,e0_A_LVhnd106_i1,e0_B_LVhnd106_i1,e0_C_LVhnd106_i1,e0_D_LVhnd106_i1,e0_E_LVhnd106_i1,e0_T_crit_i1)
    e0_p_LV_tr3_i1 = fun_87129(e0_T_tr3,e0_A_pLVd101_i1,e0_B_pLVd101_i1,e0_C_pLVd101_i1,e0_D_pLVd101_i1,e0_E_pLVd101_i1)
    e0_h_Ln_tr3_i2 = fun_87125(e0_T_tr3,e0_A_Lhnd100_i2,e0_B_Lhnd100_i2,e0_C_Lhnd100_i2,e0_D_Lhnd100_i2,e0_E_Lhnd100_i2,e0_T_ho_i2,e0_h_Lno_i2)
    e0_p_LV_tr8_i1 = fun_87129(e0_T_tr8,e0_A_pLVd101_i1,e0_B_pLVd101_i1,e0_C_pLVd101_i1,e0_D_pLVd101_i1,e0_E_pLVd101_i1)
    e0_p_LV_tr10_i2 = fun_87129(e0_T_tr10,e0_A_pLVd101_i2,e0_B_pLVd101_i2,e0_C_pLVd101_i2,e0_D_pLVd101_i2,e0_E_pLVd101_i2)
    e0_p_LV_tr7_i1 = fun_87129(e0_T_tr7,e0_A_pLVd101_i1,e0_B_pLVd101_i1,e0_C_pLVd101_i1,e0_D_pLVd101_i1,e0_E_pLVd101_i1)
    e0_p_LV_tr1_i1 = fun_87129(e0_T_tr1,e0_A_pLVd101_i1,e0_B_pLVd101_i1,e0_C_pLVd101_i1,e0_D_pLVd101_i1,e0_E_pLVd101_i1)
    e0_h_LVn_tr6_i2 = fun_87127(e0_T_tr6,e0_A_LVhnd106_i2,e0_B_LVhnd106_i2,e0_C_LVhnd106_i2,e0_D_LVhnd106_i2,e0_E_LVhnd106_i2,e0_T_crit_i2)
    e0_h_LVn_tr2_i2 = fun_87127(e0_T_tr2,e0_A_LVhnd106_i2,e0_B_LVhnd106_i2,e0_C_LVhnd106_i2,e0_D_LVhnd106_i2,e0_E_LVhnd106_i2,e0_T_crit_i2)
    e0_h_LVn_tr10_i1 = fun_87127(e0_T_tr10,e0_A_LVhnd106_i1,e0_B_LVhnd106_i1,e0_C_LVhnd106_i1,e0_D_LVhnd106_i1,e0_E_LVhnd106_i1,e0_T_crit_i1)
    e0_p_LV_tr6_i2 = fun_87129(e0_T_tr6,e0_A_pLVd101_i2,e0_B_pLVd101_i2,e0_C_pLVd101_i2,e0_D_pLVd101_i2,e0_E_pLVd101_i2)
    e0_h_LnR_i2 = fun_87125(e0_TR,e0_A_Lhnd100_i2,e0_B_Lhnd100_i2,e0_C_Lhnd100_i2,e0_D_Lhnd100_i2,e0_E_Lhnd100_i2,e0_T_ho_i2,e0_h_Lno_i2)
    e0_h_Ln_tr2_i2 = fun_87125(e0_T_tr2,e0_A_Lhnd100_i2,e0_B_Lhnd100_i2,e0_C_Lhnd100_i2,e0_D_Lhnd100_i2,e0_E_Lhnd100_i2,e0_T_ho_i2,e0_h_Lno_i2)
    e0_h_LVn_tr3_i2 = fun_87127(e0_T_tr3,e0_A_LVhnd106_i2,e0_B_LVhnd106_i2,e0_C_LVhnd106_i2,e0_D_LVhnd106_i2,e0_E_LVhnd106_i2,e0_T_crit_i2)
    e0_h_Ln_tr5_i1 = fun_87125(e0_T_tr5,e0_A_Lhnd100_i1,e0_B_Lhnd100_i1,e0_C_Lhnd100_i1,e0_D_Lhnd100_i1,e0_E_Lhnd100_i1,e0_T_ho_i1,e0_h_Lno_i1)
    e0_h_Ln_tr5_i2 = fun_87125(e0_T_tr5,e0_A_Lhnd100_i2,e0_B_Lhnd100_i2,e0_C_Lhnd100_i2,e0_D_Lhnd100_i2,e0_E_Lhnd100_i2,e0_T_ho_i2,e0_h_Lno_i2)
    e0_p_LVR_i1 = fun_87129(e0_TR,e0_A_pLVd101_i1,e0_B_pLVd101_i1,e0_C_pLVd101_i1,e0_D_pLVd101_i1,e0_E_pLVd101_i1)
    e0_h_Ln_tr1_i1 = fun_87125(e0_T_tr1,e0_A_Lhnd100_i1,e0_B_Lhnd100_i1,e0_C_Lhnd100_i1,e0_D_Lhnd100_i1,e0_E_Lhnd100_i1,e0_T_ho_i1,e0_h_Lno_i1)
    e0_h_Lfn_i2 = fun_87125(e0_T_f,e0_A_Lhnd100_i2,e0_B_Lhnd100_i2,e0_C_Lhnd100_i2,e0_D_Lhnd100_i2,e0_E_Lhnd100_i2,e0_T_ho_i2,e0_h_Lno_i2)
    e0_p_LVC_i1 = fun_87129(e0_TC,e0_A_pLVd101_i1,e0_B_pLVd101_i1,e0_C_pLVd101_i1,e0_D_pLVd101_i1,e0_E_pLVd101_i1)
    e0_h_Ln_tr4_i2 = fun_87125(e0_T_tr4,e0_A_Lhnd100_i2,e0_B_Lhnd100_i2,e0_C_Lhnd100_i2,e0_D_Lhnd100_i2,e0_E_Lhnd100_i2,e0_T_ho_i2,e0_h_Lno_i2)
    e0_h_LVnR_i2 = fun_87127(e0_TR,e0_A_LVhnd106_i2,e0_B_LVhnd106_i2,e0_C_LVhnd106_i2,e0_D_LVhnd106_i2,e0_E_LVhnd106_i2,e0_T_crit_i2)
    e0_p_LV_tr4_i2 = fun_87129(e0_T_tr4,e0_A_pLVd101_i2,e0_B_pLVd101_i2,e0_C_pLVd101_i2,e0_D_pLVd101_i2,e0_E_pLVd101_i2)
    e0_p_LVC_i2 = fun_87129(e0_TC,e0_A_pLVd101_i2,e0_B_pLVd101_i2,e0_C_pLVd101_i2,e0_D_pLVd101_i2,e0_E_pLVd101_i2)
    e0_p_LV_tr2_i1 = fun_87129(e0_T_tr2,e0_A_pLVd101_i1,e0_B_pLVd101_i1,e0_C_pLVd101_i1,e0_D_pLVd101_i1,e0_E_pLVd101_i1)
    e0_h_Ln_tr8_i1 = fun_87125(e0_T_tr8,e0_A_Lhnd100_i1,e0_B_Lhnd100_i1,e0_C_Lhnd100_i1,e0_D_Lhnd100_i1,e0_E_Lhnd100_i1,e0_T_ho_i1,e0_h_Lno_i1)
    e0_p_LV_tr7_i2 = fun_87129(e0_T_tr7,e0_A_pLVd101_i2,e0_B_pLVd101_i2,e0_C_pLVd101_i2,e0_D_pLVd101_i2,e0_E_pLVd101_i2)
    e0_p_LV_tr10_i1 = fun_87129(e0_T_tr10,e0_A_pLVd101_i1,e0_B_pLVd101_i1,e0_C_pLVd101_i1,e0_D_pLVd101_i1,e0_E_pLVd101_i1)
    e0_h_Lfn_i1 = fun_87125(e0_T_f,e0_A_Lhnd100_i1,e0_B_Lhnd100_i1,e0_C_Lhnd100_i1,e0_D_Lhnd100_i1,e0_E_Lhnd100_i1,e0_T_ho_i1,e0_h_Lno_i1)
    e0_h_Ln_tr7_i2 = fun_87125(e0_T_tr7,e0_A_Lhnd100_i2,e0_B_Lhnd100_i2,e0_C_Lhnd100_i2,e0_D_Lhnd100_i2,e0_E_Lhnd100_i2,e0_T_ho_i2,e0_h_Lno_i2)
    e0_h_Ln_tr6_i2 = fun_87125(e0_T_tr6,e0_A_Lhnd100_i2,e0_B_Lhnd100_i2,e0_C_Lhnd100_i2,e0_D_Lhnd100_i2,e0_E_Lhnd100_i2,e0_T_ho_i2,e0_h_Lno_i2)
    e0_p_LVR_i2 = fun_87129(e0_TR,e0_A_pLVd101_i2,e0_B_pLVd101_i2,e0_C_pLVd101_i2,e0_D_pLVd101_i2,e0_E_pLVd101_i2)
    e0_p_LV_tr8_i2 = fun_87129(e0_T_tr8,e0_A_pLVd101_i2,e0_B_pLVd101_i2,e0_C_pLVd101_i2,e0_D_pLVd101_i2,e0_E_pLVd101_i2)
    e0_h_LVn_tr10_i2 = fun_87127(e0_T_tr10,e0_A_LVhnd106_i2,e0_B_LVhnd106_i2,e0_C_LVhnd106_i2,e0_D_LVhnd106_i2,e0_E_LVhnd106_i2,e0_T_crit_i2)
    e0_h_LVn_tr8_i1 = fun_87127(e0_T_tr8,e0_A_LVhnd106_i1,e0_B_LVhnd106_i1,e0_C_LVhnd106_i1,e0_D_LVhnd106_i1,e0_E_LVhnd106_i1,e0_T_crit_i1)
    e0_h_LVn_tr9_i2 = fun_87127(e0_T_tr9,e0_A_LVhnd106_i2,e0_B_LVhnd106_i2,e0_C_LVhnd106_i2,e0_D_LVhnd106_i2,e0_E_LVhnd106_i2,e0_T_crit_i2)
    e0_p_LV_tr3_i2 = fun_87129(e0_T_tr3,e0_A_pLVd101_i2,e0_B_pLVd101_i2,e0_C_pLVd101_i2,e0_D_pLVd101_i2,e0_E_pLVd101_i2)
    e0_p_LV_tr5_i2 = fun_87129(e0_T_tr5,e0_A_pLVd101_i2,e0_B_pLVd101_i2,e0_C_pLVd101_i2,e0_D_pLVd101_i2,e0_E_pLVd101_i2)
    e0_h_LVn_tr7_i1 = fun_87127(e0_T_tr7,e0_A_LVhnd106_i1,e0_B_LVhnd106_i1,e0_C_LVhnd106_i1,e0_D_LVhnd106_i1,e0_E_LVhnd106_i1,e0_T_crit_i1)
    e0_p_LV_tr4_i1 = fun_87129(e0_T_tr4,e0_A_pLVd101_i1,e0_B_pLVd101_i1,e0_C_pLVd101_i1,e0_D_pLVd101_i1,e0_E_pLVd101_i1)
    e0_h_LVn_tr1_i2 = fun_87127(e0_T_tr1,e0_A_LVhnd106_i2,e0_B_LVhnd106_i2,e0_C_LVhnd106_i2,e0_D_LVhnd106_i2,e0_E_LVhnd106_i2,e0_T_crit_i2)
    e0_h_Ln_tr3_i1 = fun_87125(e0_T_tr3,e0_A_Lhnd100_i1,e0_B_Lhnd100_i1,e0_C_Lhnd100_i1,e0_D_Lhnd100_i1,e0_E_Lhnd100_i1,e0_T_ho_i1,e0_h_Lno_i1)
    e0_h_Ln_tr6_i1 = fun_87125(e0_T_tr6,e0_A_Lhnd100_i1,e0_B_Lhnd100_i1,e0_C_Lhnd100_i1,e0_D_Lhnd100_i1,e0_E_Lhnd100_i1,e0_T_ho_i1,e0_h_Lno_i1)
    e0_h_Ln_tr1_i2 = fun_87125(e0_T_tr1,e0_A_Lhnd100_i2,e0_B_Lhnd100_i2,e0_C_Lhnd100_i2,e0_D_Lhnd100_i2,e0_E_Lhnd100_i2,e0_T_ho_i2,e0_h_Lno_i2)



# Solve equation system for given x:
    f= [
    0.0-((e0_F_Ln_tr1) *(e0_x_i1_tr1)-(e0_F_LnR) *(e0_xR_i1)-(e0_F_Vn_tr0) *(e0_y_i1_tr0)) ,
    0.0-((e0_F_Ln_tr1) *(e0_x_i2_tr1)-(e0_F_LnR) *(e0_xR_i2)-(e0_F_Vn_tr0) *(e0_y_i2_tr0)) ,
    0.0-(((e0_F_Ln_tr1) *(e0_h_Ln_tr1)-(e0_F_LnR) *(e0_h_LnR)-(e0_F_Vn_tr0) *(e0_h_Vn_tr0)+e0_QR)/(e0_h_sca)) ,
    ((e0_y_i1_tr0) *(e0_p_tr0))/(e0_p_sca)-(((e0_xR_i1) *((e0_p_LVR_i1) *(e0_greek_gammaR_i1)))/(e0_p_sca)) ,
    ((e0_y_i2_tr0) *(e0_p_tr0))/(e0_p_sca)-(((e0_xR_i2) *((e0_p_LVR_i2) *(e0_greek_gammaR_i2)))/(e0_p_sca)) ,
    1.0-((e0_xR_i1+e0_xR_i2)) ,
    1.0-((e0_y_i1_tr0+e0_y_i2_tr0)) ,
    (e0_h_LnR)/(e0_h_sca)-((((e0_xR_i1) *(e0_h_LnR_i1)+(e0_xR_i2) *(e0_h_LnR_i2)))/(e0_h_sca)) ,
    (e0_h_Vn_tr0)/(e0_h_sca)-((((e0_y_i1_tr0) *((e0_h_LnR_i1+e0_h_LVnR_i1))+(e0_y_i2_tr0) *((e0_h_LnR_i2+e0_h_LVnR_i2))))/(e0_h_sca)) ,
    0.0-((e0_F_Lfn_tr1) *(e0_x_f_i1)+(e0_F_Ln_tr2) *(e0_x_tr2_i1)+(e0_F_Vn_tr0) *(e0_y_i1_tr0)-(e0_F_Ln_tr1) *(e0_x_i1_tr1)-(e0_F_Vn_tr1) *(e0_y_tr1_i1)) ,
    0.0-((e0_F_Lfn_tr2) *(e0_x_f_i1)+(e0_F_Ln_tr3) *(e0_x_tr3_i1)+(e0_F_Vn_tr1) *(e0_y_tr1_i1)-(e0_F_Ln_tr2) *(e0_x_tr2_i1)-(e0_F_Vn_tr2) *(e0_y_tr2_i1)) ,
    0.0-((e0_F_Lfn_tr3) *(e0_x_f_i1)+(e0_F_Ln_tr4) *(e0_x_tr4_i1)+(e0_F_Vn_tr2) *(e0_y_tr2_i1)-(e0_F_Ln_tr3) *(e0_x_tr3_i1)-(e0_F_Vn_tr3) *(e0_y_tr3_i1)) ,
    0.0-((e0_F_Lfn_tr4) *(e0_x_f_i1)+(e0_F_Ln_tr5) *(e0_x_tr5_i1)+(e0_F_Vn_tr3) *(e0_y_tr3_i1)-(e0_F_Ln_tr4) *(e0_x_tr4_i1)-(e0_F_Vn_tr4) *(e0_y_tr4_i1)) ,
    0.0-((e0_F_Lfn_tr5) *(e0_x_f_i1)+(e0_F_Ln_tr6) *(e0_x_tr6_i1)+(e0_F_Vn_tr4) *(e0_y_tr4_i1)-(e0_F_Ln_tr5) *(e0_x_tr5_i1)-(e0_F_Vn_tr5) *(e0_y_tr5_i1)) ,
    0.0-((e0_F_Lfn_tr6) *(e0_x_f_i1)+(e0_F_Ln_tr7) *(e0_x_tr7_i1)+(e0_F_Vn_tr5) *(e0_y_tr5_i1)-(e0_F_Ln_tr6) *(e0_x_tr6_i1)-(e0_F_Vn_tr6) *(e0_y_tr6_i1)) ,
    0.0-((e0_F_Lfn_tr7) *(e0_x_f_i1)+(e0_F_Ln_tr8) *(e0_x_tr8_i1)+(e0_F_Vn_tr6) *(e0_y_tr6_i1)-(e0_F_Ln_tr7) *(e0_x_tr7_i1)-(e0_F_Vn_tr7) *(e0_y_tr7_i1)) ,
    0.0-((e0_F_Lfn_tr8) *(e0_x_f_i1)+(e0_F_Ln_tr9) *(e0_x_tr9_i1)+(e0_F_Vn_tr7) *(e0_y_tr7_i1)-(e0_F_Ln_tr8) *(e0_x_tr8_i1)-(e0_F_Vn_tr8) *(e0_y_tr8_i1)) ,
    0.0-((e0_F_Lfn_tr9) *(e0_x_f_i1)+(e0_F_Ln_tr10) *(e0_x_tr10_i1)+(e0_F_Vn_tr8) *(e0_y_tr8_i1)-(e0_F_Ln_tr9) *(e0_x_tr9_i1)-(e0_F_Vn_tr9) *(e0_y_tr9_i1)) ,
    0.0-((e0_F_Lfn_tr10) *(e0_x_f_i1)+(e0_F_Ln_tr11) *(e0_x_tr11_i1)+(e0_F_Vn_tr9) *(e0_y_tr9_i1)-(e0_F_Ln_tr10) *(e0_x_tr10_i1)-(e0_F_Vn_tr10) *(e0_y_tr10_i1)) ,
    0.0-((e0_F_Lfn_tr1) *(e0_x_f_i2)+(e0_F_Ln_tr2) *(e0_x_tr2_i2)+(e0_F_Vn_tr0) *(e0_y_i2_tr0)-(e0_F_Ln_tr1) *(e0_x_i2_tr1)-(e0_F_Vn_tr1) *(e0_y_tr1_i2)) ,
    0.0-((e0_F_Lfn_tr2) *(e0_x_f_i2)+(e0_F_Ln_tr3) *(e0_x_tr3_i2)+(e0_F_Vn_tr1) *(e0_y_tr1_i2)-(e0_F_Ln_tr2) *(e0_x_tr2_i2)-(e0_F_Vn_tr2) *(e0_y_tr2_i2)) ,
    0.0-((e0_F_Lfn_tr3) *(e0_x_f_i2)+(e0_F_Ln_tr4) *(e0_x_tr4_i2)+(e0_F_Vn_tr2) *(e0_y_tr2_i2)-(e0_F_Ln_tr3) *(e0_x_tr3_i2)-(e0_F_Vn_tr3) *(e0_y_tr3_i2)) ,
    0.0-((e0_F_Lfn_tr4) *(e0_x_f_i2)+(e0_F_Ln_tr5) *(e0_x_tr5_i2)+(e0_F_Vn_tr3) *(e0_y_tr3_i2)-(e0_F_Ln_tr4) *(e0_x_tr4_i2)-(e0_F_Vn_tr4) *(e0_y_tr4_i2)) ,
    0.0-((e0_F_Lfn_tr5) *(e0_x_f_i2)+(e0_F_Ln_tr6) *(e0_x_tr6_i2)+(e0_F_Vn_tr4) *(e0_y_tr4_i2)-(e0_F_Ln_tr5) *(e0_x_tr5_i2)-(e0_F_Vn_tr5) *(e0_y_tr5_i2)) ,
    0.0-((e0_F_Lfn_tr6) *(e0_x_f_i2)+(e0_F_Ln_tr7) *(e0_x_tr7_i2)+(e0_F_Vn_tr5) *(e0_y_tr5_i2)-(e0_F_Ln_tr6) *(e0_x_tr6_i2)-(e0_F_Vn_tr6) *(e0_y_tr6_i2)) ,
    0.0-((e0_F_Lfn_tr7) *(e0_x_f_i2)+(e0_F_Ln_tr8) *(e0_x_tr8_i2)+(e0_F_Vn_tr6) *(e0_y_tr6_i2)-(e0_F_Ln_tr7) *(e0_x_tr7_i2)-(e0_F_Vn_tr7) *(e0_y_tr7_i2)) ,
    0.0-((e0_F_Lfn_tr8) *(e0_x_f_i2)+(e0_F_Ln_tr9) *(e0_x_tr9_i2)+(e0_F_Vn_tr7) *(e0_y_tr7_i2)-(e0_F_Ln_tr8) *(e0_x_tr8_i2)-(e0_F_Vn_tr8) *(e0_y_tr8_i2)) ,
    0.0-((e0_F_Lfn_tr9) *(e0_x_f_i2)+(e0_F_Ln_tr10) *(e0_x_tr10_i2)+(e0_F_Vn_tr8) *(e0_y_tr8_i2)-(e0_F_Ln_tr9) *(e0_x_tr9_i2)-(e0_F_Vn_tr9) *(e0_y_tr9_i2)) ,
    0.0-((e0_F_Lfn_tr10) *(e0_x_f_i2)+(e0_F_Ln_tr11) *(e0_x_tr11_i2)+(e0_F_Vn_tr9) *(e0_y_tr9_i2)-(e0_F_Ln_tr10) *(e0_x_tr10_i2)-(e0_F_Vn_tr10) *(e0_y_tr10_i2)) ,
    0.0-(((e0_F_Lfn_tr1) *(e0_h_Lfn)+(e0_F_Ln_tr2) *(e0_h_Ln_tr2)+(e0_F_Vn_tr0) *(e0_h_Vn_tr0)-(e0_F_Ln_tr1) *(e0_h_Ln_tr1)-(e0_F_Vn_tr1) *(e0_h_Vn_tr1))/(e0_h_sca)) ,
    0.0-(((e0_F_Lfn_tr2) *(e0_h_Lfn)+(e0_F_Ln_tr3) *(e0_h_Ln_tr3)+(e0_F_Vn_tr1) *(e0_h_Vn_tr1)-(e0_F_Ln_tr2) *(e0_h_Ln_tr2)-(e0_F_Vn_tr2) *(e0_h_Vn_tr2))/(e0_h_sca)) ,
    0.0-(((e0_F_Lfn_tr3) *(e0_h_Lfn)+(e0_F_Ln_tr4) *(e0_h_Ln_tr4)+(e0_F_Vn_tr2) *(e0_h_Vn_tr2)-(e0_F_Ln_tr3) *(e0_h_Ln_tr3)-(e0_F_Vn_tr3) *(e0_h_Vn_tr3))/(e0_h_sca)) ,
    0.0-(((e0_F_Lfn_tr4) *(e0_h_Lfn)+(e0_F_Ln_tr5) *(e0_h_Ln_tr5)+(e0_F_Vn_tr3) *(e0_h_Vn_tr3)-(e0_F_Ln_tr4) *(e0_h_Ln_tr4)-(e0_F_Vn_tr4) *(e0_h_Vn_tr4))/(e0_h_sca)) ,
    0.0-(((e0_F_Lfn_tr5) *(e0_h_Lfn)+(e0_F_Ln_tr6) *(e0_h_Ln_tr6)+(e0_F_Vn_tr4) *(e0_h_Vn_tr4)-(e0_F_Ln_tr5) *(e0_h_Ln_tr5)-(e0_F_Vn_tr5) *(e0_h_Vn_tr5))/(e0_h_sca)) ,
    0.0-(((e0_F_Lfn_tr6) *(e0_h_Lfn)+(e0_F_Ln_tr7) *(e0_h_Ln_tr7)+(e0_F_Vn_tr5) *(e0_h_Vn_tr5)-(e0_F_Ln_tr6) *(e0_h_Ln_tr6)-(e0_F_Vn_tr6) *(e0_h_Vn_tr6))/(e0_h_sca)) ,
    0.0-(((e0_F_Lfn_tr7) *(e0_h_Lfn)+(e0_F_Ln_tr8) *(e0_h_Ln_tr8)+(e0_F_Vn_tr6) *(e0_h_Vn_tr6)-(e0_F_Ln_tr7) *(e0_h_Ln_tr7)-(e0_F_Vn_tr7) *(e0_h_Vn_tr7))/(e0_h_sca)) ,
    0.0-(((e0_F_Lfn_tr8) *(e0_h_Lfn)+(e0_F_Ln_tr9) *(e0_h_Ln_tr9)+(e0_F_Vn_tr7) *(e0_h_Vn_tr7)-(e0_F_Ln_tr8) *(e0_h_Ln_tr8)-(e0_F_Vn_tr8) *(e0_h_Vn_tr8))/(e0_h_sca)) ,
    0.0-(((e0_F_Lfn_tr9) *(e0_h_Lfn)+(e0_F_Ln_tr10) *(e0_h_Ln_tr10)+(e0_F_Vn_tr8) *(e0_h_Vn_tr8)-(e0_F_Ln_tr9) *(e0_h_Ln_tr9)-(e0_F_Vn_tr9) *(e0_h_Vn_tr9))/(e0_h_sca)) ,
    0.0-(((e0_F_Lfn_tr10) *(e0_h_Lfn)+(e0_F_Ln_tr11) *(e0_h_Ln_tr11)+(e0_F_Vn_tr9) *(e0_h_Vn_tr9)-(e0_F_Ln_tr10) *(e0_h_Ln_tr10)-(e0_F_Vn_tr10) *(e0_h_Vn_tr10))/(e0_h_sca)) ,
    ((e0_greek_phi_V_tr1_i1) *((e0_p_tr1) *(e0_y_tr1_i1)))/(e0_p_sca)-(((e0_greek_gamma_tr1_i1) *((e0_x_i1_tr1) *(e0_p_LV_tr1_i1)))/(e0_p_sca)) ,
    ((e0_greek_phi_V_tr2_i1) *((e0_p_tr2) *(e0_y_tr2_i1)))/(e0_p_sca)-(((e0_greek_gamma_tr2_i1) *((e0_x_tr2_i1) *(e0_p_LV_tr2_i1)))/(e0_p_sca)) ,
    ((e0_greek_phi_V_tr3_i1) *((e0_p_tr3) *(e0_y_tr3_i1)))/(e0_p_sca)-(((e0_greek_gamma_tr3_i1) *((e0_x_tr3_i1) *(e0_p_LV_tr3_i1)))/(e0_p_sca)) ,
    ((e0_greek_phi_V_tr4_i1) *((e0_p_tr4) *(e0_y_tr4_i1)))/(e0_p_sca)-(((e0_greek_gamma_tr4_i1) *((e0_x_tr4_i1) *(e0_p_LV_tr4_i1)))/(e0_p_sca)) ,
    ((e0_greek_phi_V_tr5_i1) *((e0_p_tr5) *(e0_y_tr5_i1)))/(e0_p_sca)-(((e0_greek_gamma_tr5_i1) *((e0_x_tr5_i1) *(e0_p_LV_tr5_i1)))/(e0_p_sca)) ,
    ((e0_greek_phi_V_tr6_i1) *((e0_p_tr6) *(e0_y_tr6_i1)))/(e0_p_sca)-(((e0_greek_gamma_tr6_i1) *((e0_x_tr6_i1) *(e0_p_LV_tr6_i1)))/(e0_p_sca)) ,
    ((e0_greek_phi_V_tr7_i1) *((e0_p_tr7) *(e0_y_tr7_i1)))/(e0_p_sca)-(((e0_greek_gamma_tr7_i1) *((e0_x_tr7_i1) *(e0_p_LV_tr7_i1)))/(e0_p_sca)) ,
    ((e0_greek_phi_V_tr8_i1) *((e0_p_tr8) *(e0_y_tr8_i1)))/(e0_p_sca)-(((e0_greek_gamma_tr8_i1) *((e0_x_tr8_i1) *(e0_p_LV_tr8_i1)))/(e0_p_sca)) ,
    ((e0_greek_phi_V_tr9_i1) *((e0_p_tr9) *(e0_y_tr9_i1)))/(e0_p_sca)-(((e0_greek_gamma_tr9_i1) *((e0_x_tr9_i1) *(e0_p_LV_tr9_i1)))/(e0_p_sca)) ,
    ((e0_greek_phi_V_tr10_i1) *((e0_p_tr10) *(e0_y_tr10_i1)))/(e0_p_sca)-(((e0_greek_gamma_tr10_i1) *((e0_x_tr10_i1) *(e0_p_LV_tr10_i1)))/(e0_p_sca)) ,
    ((e0_greek_phi_V_tr1_i2) *((e0_p_tr1) *(e0_y_tr1_i2)))/(e0_p_sca)-(((e0_greek_gamma_tr1_i2) *((e0_x_i2_tr1) *(e0_p_LV_tr1_i2)))/(e0_p_sca)) ,
    ((e0_greek_phi_V_tr2_i2) *((e0_p_tr2) *(e0_y_tr2_i2)))/(e0_p_sca)-(((e0_greek_gamma_tr2_i2) *((e0_x_tr2_i2) *(e0_p_LV_tr2_i2)))/(e0_p_sca)) ,
    ((e0_greek_phi_V_tr3_i2) *((e0_p_tr3) *(e0_y_tr3_i2)))/(e0_p_sca)-(((e0_greek_gamma_tr3_i2) *((e0_x_tr3_i2) *(e0_p_LV_tr3_i2)))/(e0_p_sca)) ,
    ((e0_greek_phi_V_tr4_i2) *((e0_p_tr4) *(e0_y_tr4_i2)))/(e0_p_sca)-(((e0_greek_gamma_tr4_i2) *((e0_x_tr4_i2) *(e0_p_LV_tr4_i2)))/(e0_p_sca)) ,
    ((e0_greek_phi_V_tr5_i2) *((e0_p_tr5) *(e0_y_tr5_i2)))/(e0_p_sca)-(((e0_greek_gamma_tr5_i2) *((e0_x_tr5_i2) *(e0_p_LV_tr5_i2)))/(e0_p_sca)) ,
    ((e0_greek_phi_V_tr6_i2) *((e0_p_tr6) *(e0_y_tr6_i2)))/(e0_p_sca)-(((e0_greek_gamma_tr6_i2) *((e0_x_tr6_i2) *(e0_p_LV_tr6_i2)))/(e0_p_sca)) ,
    ((e0_greek_phi_V_tr7_i2) *((e0_p_tr7) *(e0_y_tr7_i2)))/(e0_p_sca)-(((e0_greek_gamma_tr7_i2) *((e0_x_tr7_i2) *(e0_p_LV_tr7_i2)))/(e0_p_sca)) ,
    ((e0_greek_phi_V_tr8_i2) *((e0_p_tr8) *(e0_y_tr8_i2)))/(e0_p_sca)-(((e0_greek_gamma_tr8_i2) *((e0_x_tr8_i2) *(e0_p_LV_tr8_i2)))/(e0_p_sca)) ,
    ((e0_greek_phi_V_tr9_i2) *((e0_p_tr9) *(e0_y_tr9_i2)))/(e0_p_sca)-(((e0_greek_gamma_tr9_i2) *((e0_x_tr9_i2) *(e0_p_LV_tr9_i2)))/(e0_p_sca)) ,
    ((e0_greek_phi_V_tr10_i2) *((e0_p_tr10) *(e0_y_tr10_i2)))/(e0_p_sca)-(((e0_greek_gamma_tr10_i2) *((e0_x_tr10_i2) *(e0_p_LV_tr10_i2)))/(e0_p_sca)) ,
    1.0-((e0_x_i1_tr1+e0_x_i2_tr1)) ,
    1.0-((e0_x_tr2_i1+e0_x_tr2_i2)) ,
    1.0-((e0_x_tr3_i1+e0_x_tr3_i2)) ,
    1.0-((e0_x_tr4_i1+e0_x_tr4_i2)) ,
    1.0-((e0_x_tr5_i1+e0_x_tr5_i2)) ,
    1.0-((e0_x_tr6_i1+e0_x_tr6_i2)) ,
    1.0-((e0_x_tr7_i1+e0_x_tr7_i2)) ,
    1.0-((e0_x_tr8_i1+e0_x_tr8_i2)) ,
    1.0-((e0_x_tr9_i1+e0_x_tr9_i2)) ,
    1.0-((e0_x_tr10_i1+e0_x_tr10_i2)) ,
    1.0-((e0_y_tr1_i1+e0_y_tr1_i2)) ,
    1.0-((e0_y_tr2_i1+e0_y_tr2_i2)) ,
    1.0-((e0_y_tr3_i1+e0_y_tr3_i2)) ,
    1.0-((e0_y_tr4_i1+e0_y_tr4_i2)) ,
    1.0-((e0_y_tr5_i1+e0_y_tr5_i2)) ,
    1.0-((e0_y_tr6_i1+e0_y_tr6_i2)) ,
    1.0-((e0_y_tr7_i1+e0_y_tr7_i2)) ,
    1.0-((e0_y_tr8_i1+e0_y_tr8_i2)) ,
    1.0-((e0_y_tr9_i1+e0_y_tr9_i2)) ,
    1.0-((e0_y_tr10_i1+e0_y_tr10_i2)) ,
    e0_greek_Deltap-(e0_p_tr0-e0_p_tr1) ,
    e0_greek_Deltap-(e0_p_tr1-e0_p_tr2) ,
    e0_greek_Deltap-(e0_p_tr2-e0_p_tr3) ,
    e0_greek_Deltap-(e0_p_tr3-e0_p_tr4) ,
    e0_greek_Deltap-(e0_p_tr4-e0_p_tr5) ,
    e0_greek_Deltap-(e0_p_tr5-e0_p_tr6) ,
    e0_greek_Deltap-(e0_p_tr6-e0_p_tr7) ,
    e0_greek_Deltap-(e0_p_tr7-e0_p_tr8) ,
    e0_greek_Deltap-(e0_p_tr8-e0_p_tr9) ,
    e0_greek_Deltap-(e0_p_tr9-e0_p_tr10) ,
    (e0_h_Lfn)/(e0_h_sca)-((((e0_x_f_i1) *(e0_h_Lfn_i1)+(e0_x_f_i2) *(e0_h_Lfn_i2)))/(e0_h_sca)) ,
    (e0_h_Ln_tr1)/(e0_h_sca)-(((((e0_x_i1_tr1) *(e0_h_Ln_tr1_i1))+((e0_x_i2_tr1) *(e0_h_Ln_tr1_i2))))/(e0_h_sca)) ,
    (e0_h_Ln_tr2)/(e0_h_sca)-(((((e0_x_tr2_i1) *(e0_h_Ln_tr2_i1))+((e0_x_tr2_i2) *(e0_h_Ln_tr2_i2))))/(e0_h_sca)) ,
    (e0_h_Ln_tr3)/(e0_h_sca)-(((((e0_x_tr3_i1) *(e0_h_Ln_tr3_i1))+((e0_x_tr3_i2) *(e0_h_Ln_tr3_i2))))/(e0_h_sca)) ,
    (e0_h_Ln_tr4)/(e0_h_sca)-(((((e0_x_tr4_i1) *(e0_h_Ln_tr4_i1))+((e0_x_tr4_i2) *(e0_h_Ln_tr4_i2))))/(e0_h_sca)) ,
    (e0_h_Ln_tr5)/(e0_h_sca)-(((((e0_x_tr5_i1) *(e0_h_Ln_tr5_i1))+((e0_x_tr5_i2) *(e0_h_Ln_tr5_i2))))/(e0_h_sca)) ,
    (e0_h_Ln_tr6)/(e0_h_sca)-(((((e0_x_tr6_i1) *(e0_h_Ln_tr6_i1))+((e0_x_tr6_i2) *(e0_h_Ln_tr6_i2))))/(e0_h_sca)) ,
    (e0_h_Ln_tr7)/(e0_h_sca)-(((((e0_x_tr7_i1) *(e0_h_Ln_tr7_i1))+((e0_x_tr7_i2) *(e0_h_Ln_tr7_i2))))/(e0_h_sca)) ,
    (e0_h_Ln_tr8)/(e0_h_sca)-(((((e0_x_tr8_i1) *(e0_h_Ln_tr8_i1))+((e0_x_tr8_i2) *(e0_h_Ln_tr8_i2))))/(e0_h_sca)) ,
    (e0_h_Ln_tr9)/(e0_h_sca)-(((((e0_x_tr9_i1) *(e0_h_Ln_tr9_i1))+((e0_x_tr9_i2) *(e0_h_Ln_tr9_i2))))/(e0_h_sca)) ,
    (e0_h_Ln_tr10)/(e0_h_sca)-(((((e0_x_tr10_i1) *(e0_h_Ln_tr10_i1))+((e0_x_tr10_i2) *(e0_h_Ln_tr10_i2))))/(e0_h_sca)) ,
    (e0_h_Vn_tr1)/(e0_h_sca)-(((((e0_y_tr1_i1) *((e0_h_Ln_tr1_i1+e0_h_LVn_tr1_i1)))+((e0_y_tr1_i2) *((e0_h_Ln_tr1_i2+e0_h_LVn_tr1_i2)))))/(e0_h_sca)) ,
    (e0_h_Vn_tr2)/(e0_h_sca)-(((((e0_y_tr2_i1) *((e0_h_Ln_tr2_i1+e0_h_LVn_tr2_i1)))+((e0_y_tr2_i2) *((e0_h_Ln_tr2_i2+e0_h_LVn_tr2_i2)))))/(e0_h_sca)) ,
    (e0_h_Vn_tr3)/(e0_h_sca)-(((((e0_y_tr3_i1) *((e0_h_Ln_tr3_i1+e0_h_LVn_tr3_i1)))+((e0_y_tr3_i2) *((e0_h_Ln_tr3_i2+e0_h_LVn_tr3_i2)))))/(e0_h_sca)) ,
    (e0_h_Vn_tr4)/(e0_h_sca)-(((((e0_y_tr4_i1) *((e0_h_Ln_tr4_i1+e0_h_LVn_tr4_i1)))+((e0_y_tr4_i2) *((e0_h_Ln_tr4_i2+e0_h_LVn_tr4_i2)))))/(e0_h_sca)) ,
    (e0_h_Vn_tr5)/(e0_h_sca)-(((((e0_y_tr5_i1) *((e0_h_Ln_tr5_i1+e0_h_LVn_tr5_i1)))+((e0_y_tr5_i2) *((e0_h_Ln_tr5_i2+e0_h_LVn_tr5_i2)))))/(e0_h_sca)) ,
    (e0_h_Vn_tr6)/(e0_h_sca)-(((((e0_y_tr6_i1) *((e0_h_Ln_tr6_i1+e0_h_LVn_tr6_i1)))+((e0_y_tr6_i2) *((e0_h_Ln_tr6_i2+e0_h_LVn_tr6_i2)))))/(e0_h_sca)) ,
    (e0_h_Vn_tr7)/(e0_h_sca)-(((((e0_y_tr7_i1) *((e0_h_Ln_tr7_i1+e0_h_LVn_tr7_i1)))+((e0_y_tr7_i2) *((e0_h_Ln_tr7_i2+e0_h_LVn_tr7_i2)))))/(e0_h_sca)) ,
    (e0_h_Vn_tr8)/(e0_h_sca)-(((((e0_y_tr8_i1) *((e0_h_Ln_tr8_i1+e0_h_LVn_tr8_i1)))+((e0_y_tr8_i2) *((e0_h_Ln_tr8_i2+e0_h_LVn_tr8_i2)))))/(e0_h_sca)) ,
    (e0_h_Vn_tr9)/(e0_h_sca)-(((((e0_y_tr9_i1) *((e0_h_Ln_tr9_i1+e0_h_LVn_tr9_i1)))+((e0_y_tr9_i2) *((e0_h_Ln_tr9_i2+e0_h_LVn_tr9_i2)))))/(e0_h_sca)) ,
    (e0_h_Vn_tr10)/(e0_h_sca)-(((((e0_y_tr10_i1) *((e0_h_Ln_tr10_i1+e0_h_LVn_tr10_i1)))+((e0_y_tr10_i2) *((e0_h_Ln_tr10_i2+e0_h_LVn_tr10_i2)))))/(e0_h_sca)) ,
    0.0-((e0_F_Vn_tr10) *(e0_y_tr10_i1)-((e0_F_LnC+e0_F_Ln_tr11)) *(e0_x_tr11_i1)) ,
    0.0-((e0_F_Vn_tr10) *(e0_y_tr10_i2)-((e0_F_LnC+e0_F_Ln_tr11)) *(e0_x_tr11_i2)) ,
    ((e0_yC_i1) *(e0_p_tr10))/(e0_p_sca)-(((e0_x_tr11_i1) *((e0_p_LVC_i1) *(e0_greek_gammaC_i1)))/(e0_p_sca)) ,
    ((e0_yC_i2) *(e0_p_tr10))/(e0_p_sca)-(((e0_x_tr11_i2) *((e0_p_LVC_i2) *(e0_greek_gammaC_i2)))/(e0_p_sca)) ,
    (e0_x_tr11_i1+e0_x_tr11_i2)-(1.0) ,
    (e0_yC_i1+e0_yC_i2)-(1.0) ,
    0.0-(((e0_F_Vn_tr10) *(e0_h_Vn_tr10)-((e0_F_LnC+e0_F_Ln_tr11)) *(e0_h_Ln_tr11)+e0_QC)/(e0_h_sca)) ,
    e0_F_Ln_tr11-((e0_R) *(e0_F_LnC)) ,
    (e0_h_Ln_tr11)/(e0_h_sca)-((((e0_x_tr11_i1) *(e0_h_LnC_i1)+(e0_x_tr11_i2) *(e0_h_LnC_i2)))/(e0_h_sca)) 

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

    x = numpy.empty(119)
    xInitial = numpy.empty((119), dtype = object) 
    parameter = numpy.empty(99)
    dict_variables = {}
    dict_equations = {}

    # Iteration variable initializatio
    x[0] = 5.0 	# e0_F_Ln_tr2
    x[1] = 5.0 	# e0_F_Ln_tr3
    x[2] = 5.0 	# e0_F_Ln_tr4
    x[3] = 5.0 	# e0_F_Ln_tr5
    x[4] = 5.0 	# e0_F_Ln_tr6
    x[5] = 5.0 	# e0_F_Ln_tr7
    x[6] = 5.0 	# e0_F_Ln_tr8
    x[7] = 5.0 	# e0_F_Ln_tr9
    x[8] = 5.0 	# e0_F_Ln_tr10
    x[9] = 5.0 	# e0_F_Ln_tr11
    x[10] = 5.0 	# e0_F_Vn_tr1
    x[11] = 5.0 	# e0_F_Vn_tr2
    x[12] = 5.0 	# e0_F_Vn_tr3
    x[13] = 5.0 	# e0_F_Vn_tr4
    x[14] = 5.0 	# e0_F_Vn_tr5
    x[15] = 5.0 	# e0_F_Vn_tr6
    x[16] = 5.0 	# e0_F_Vn_tr7
    x[17] = 5.0 	# e0_F_Vn_tr8
    x[18] = -179356.652399 	# e0_h_Vn_tr0
    x[19] = 5.0 	# e0_F_Vn_tr9
    x[20] = 5.0 	# e0_F_Vn_tr10
    x[21] = 320.0 	# e0_T_tr1
    x[22] = 320.0 	# e0_T_tr2
    x[23] = 320.0 	# e0_T_tr3
    x[24] = 320.0 	# e0_T_tr4
    x[25] = 320.0 	# e0_T_tr5
    x[26] = 320.0 	# e0_T_tr6
    x[27] = 320.0 	# e0_T_tr7
    x[28] = 320.0 	# e0_T_tr8
    x[29] = 505005.0 	# e0_p_tr0
    x[30] = 320.0 	# e0_T_tr9
    x[31] = 320.0 	# e0_T_tr10
    x[32] = -219159.422927 	# e0_h_Lfn
    x[33] = -327927.609092 	# e0_h_Ln_tr2
    x[34] = -327927.609092 	# e0_h_Ln_tr3
    x[35] = -327927.609092 	# e0_h_Ln_tr4
    x[36] = -327927.609092 	# e0_h_Ln_tr5
    x[37] = 0.5 	# e0_xR_i1
    x[38] = -327927.609092 	# e0_h_Ln_tr6
    x[39] = -327927.609092 	# e0_h_Ln_tr7
    x[40] = -327927.609092 	# e0_h_Ln_tr8
    x[41] = -327927.609092 	# e0_h_Ln_tr9
    x[42] = -327927.609092 	# e0_h_Ln_tr10
    x[43] = -327927.609092 	# e0_h_Ln_tr11
    x[44] = 0.5 	# e0_xR_i2
    x[45] = 0.5 	# e0_x_i1_tr1
    x[46] = -293802.591258 	# e0_h_Vn_tr1
    x[47] = -293802.591258 	# e0_h_Vn_tr2
    x[48] = -293802.591258 	# e0_h_Vn_tr3
    x[49] = -293802.591258 	# e0_h_Vn_tr4
    x[50] = 5.0 	# e0_F_Ln_tr1
    x[51] = 0.5 	# e0_x_i2_tr1
    x[52] = -293802.591258 	# e0_h_Vn_tr5
    x[53] = -293802.591258 	# e0_h_Vn_tr6
    x[54] = -293802.591258 	# e0_h_Vn_tr7
    x[55] = -293802.591258 	# e0_h_Vn_tr8
    x[56] = -293802.591258 	# e0_h_Vn_tr9
    x[57] = -293802.591258 	# e0_h_Vn_tr10
    x[58] = 505000.0 	# e0_p_tr1
    x[59] = 505000.0 	# e0_p_tr2
    x[60] = 505000.0 	# e0_p_tr3
    x[61] = 505000.0 	# e0_p_tr4
    x[62] = 0.5 	# e0_y_i1_tr0
    x[63] = 505000.0 	# e0_p_tr5
    x[64] = 505000.0 	# e0_p_tr6
    x[65] = 505000.0 	# e0_p_tr7
    x[66] = 505000.0 	# e0_p_tr8
    x[67] = 100010.0 	# e0_p_tr9
    x[68] = 0.5 	# e0_y_i2_tr0
    x[69] = 0.5 	# e0_x_tr2_i1
    x[70] = 0.5 	# e0_x_tr3_i1
    x[71] = 0.5 	# e0_x_tr4_i1
    x[72] = 0.5 	# e0_x_tr5_i1
    x[73] = 0.5 	# e0_x_tr6_i1
    x[74] = 0.5 	# e0_x_tr7_i1
    x[75] = 0.5 	# e0_x_tr8_i1
    x[76] = 0.5 	# e0_x_tr9_i1
    x[77] = 0.5 	# e0_x_tr10_i1
    x[78] = 0.5 	# e0_x_tr11_i1
    x[79] = 0.5 	# e0_x_tr2_i2
    x[80] = 0.5 	# e0_x_tr3_i2
    x[81] = 0.5 	# e0_x_tr4_i2
    x[82] = 0.5 	# e0_x_tr5_i2
    x[83] = 0.5 	# e0_x_tr6_i2
    x[84] = 0.5 	# e0_x_tr7_i2
    x[85] = 0.5 	# e0_x_tr8_i2
    x[86] = 0.5 	# e0_x_tr9_i2
    x[87] = 0.5 	# e0_x_tr10_i2
    x[88] = 0.5 	# e0_x_tr11_i2
    x[89] = 0.5 	# e0_y_tr1_i1
    x[90] = 0.5 	# e0_y_tr2_i1
    x[91] = 0.5 	# e0_y_tr3_i1
    x[92] = 0.5 	# e0_y_tr4_i1
    x[93] = 0.5 	# e0_y_tr5_i1
    x[94] = 0.5 	# e0_y_tr6_i1
    x[95] = 0.5 	# e0_y_tr7_i1
    x[96] = 0.5 	# e0_y_tr8_i1
    x[97] = 0.5 	# e0_y_tr9_i1
    x[98] = 0.5 	# e0_y_tr10_i1
    x[99] = 0.5 	# e0_y_tr1_i2
    x[100] = 0.5 	# e0_y_tr2_i2
    x[101] = 0.5 	# e0_y_tr3_i2
    x[102] = 0.5 	# e0_y_tr4_i2
    x[103] = 0.5 	# e0_y_tr5_i2
    x[104] = 0.5 	# e0_y_tr6_i2
    x[105] = 0.5 	# e0_y_tr7_i2
    x[106] = 0.5 	# e0_y_tr8_i2
    x[107] = 0.5 	# e0_y_tr9_i2
    x[108] = 0.5 	# e0_y_tr10_i2
    x[109] = 2.5 	# e0_F_LnC
    x[110] = 500000.0 	# e0_QC
    x[111] = 320.0 	# e0_TC
    x[112] = 0.5 	# e0_yC_i1
    x[113] = 0.5 	# e0_yC_i2
    x[114] = 5.0 	# e0_F_LnR
    x[115] = 5.0 	# e0_F_Vn_tr0
    x[116] = 500000.0 	# e0_QR
    x[117] = -327927.609092 	# e0_h_Ln_tr1
    x[118] = -215879.723336 	# e0_h_LnR


    # Constant parameter setting:
    parameter[0] = 1.0 	# e0_greek_gammaR_i1
    parameter[1] = 1.0 	# e0_greek_gammaR_i2
    parameter[2] = 1.0 	# e0_greek_phi_V_tr9_i2
    parameter[3] = 1.0 	# e0_greek_phi_V_tr10_i2
    parameter[4] = 0.0 	# e0_F_Lfn_tr1
    parameter[5] = 0.0 	# e0_F_Lfn_tr2
    parameter[6] = 0.0 	# e0_F_Lfn_tr3
    parameter[7] = 0.0 	# e0_F_Lfn_tr4
    parameter[8] = 1.0 	# e0_F_Lfn_tr5
    parameter[9] = 0.0 	# e0_F_Lfn_tr6
    parameter[10] = 0.0 	# e0_F_Lfn_tr7
    parameter[11] = 0.0 	# e0_F_Lfn_tr8
    parameter[12] = 0.0 	# e0_F_Lfn_tr9
    parameter[13] = 0.0 	# e0_F_Lfn_tr10
    parameter[14] = 351.0 	# e0_T_f
    parameter[15] = 100000.0 	# e0_p_tr10
    parameter[16] = 0.55 	# e0_x_f_i1
    parameter[17] = 0.45 	# e0_x_f_i2
    parameter[18] = 1.0 	# e0_greek_gammaC_i1
    parameter[19] = 1.0 	# e0_greek_gammaC_i2
    parameter[20] = 2.0 	# e0_R
    parameter[21] = 365.0 	# e0_TR
    parameter[22] = 10.0 	# e0_greek_Deltap
    parameter[23] = 1.0 	# e0_greek_gamma_tr1_i1
    parameter[24] = 1.0 	# e0_greek_gamma_tr2_i1
    parameter[25] = 1.0 	# e0_greek_gamma_tr3_i1
    parameter[26] = 1.0 	# e0_greek_gamma_tr4_i1
    parameter[27] = 1.0 	# e0_greek_gamma_tr5_i1
    parameter[28] = 1.0 	# e0_greek_gamma_tr6_i1
    parameter[29] = 1.0 	# e0_greek_gamma_tr7_i1
    parameter[30] = 1.0 	# e0_greek_gamma_tr8_i1
    parameter[31] = 1.0 	# e0_greek_gamma_tr9_i1
    parameter[32] = 1.0 	# e0_greek_gamma_tr10_i1
    parameter[33] = 1.0 	# e0_greek_gamma_tr1_i2
    parameter[34] = 1.0 	# e0_greek_gamma_tr2_i2
    parameter[35] = 1.0 	# e0_greek_gamma_tr3_i2
    parameter[36] = 1.0 	# e0_greek_gamma_tr4_i2
    parameter[37] = 1.0 	# e0_greek_gamma_tr5_i2
    parameter[38] = 1.0 	# e0_greek_gamma_tr6_i2
    parameter[39] = 1.0 	# e0_greek_gamma_tr7_i2
    parameter[40] = 1.0 	# e0_greek_gamma_tr8_i2
    parameter[41] = 1.0 	# e0_greek_gamma_tr9_i2
    parameter[42] = 1.0 	# e0_greek_gamma_tr10_i2
    parameter[43] = 1.0 	# e0_greek_phi_V_tr1_i1
    parameter[44] = 1.0 	# e0_greek_phi_V_tr2_i1
    parameter[45] = 1.0 	# e0_greek_phi_V_tr3_i1
    parameter[46] = 1.0 	# e0_greek_phi_V_tr4_i1
    parameter[47] = 1.0 	# e0_greek_phi_V_tr5_i1
    parameter[48] = 1.0 	# e0_greek_phi_V_tr6_i1
    parameter[49] = 1.0 	# e0_greek_phi_V_tr7_i1
    parameter[50] = 1.0 	# e0_greek_phi_V_tr8_i1
    parameter[51] = 1.0 	# e0_greek_phi_V_tr9_i1
    parameter[52] = 1.0 	# e0_greek_phi_V_tr10_i1
    parameter[53] = 1.0 	# e0_greek_phi_V_tr1_i2
    parameter[54] = 1.0 	# e0_greek_phi_V_tr2_i2
    parameter[55] = 1.0 	# e0_greek_phi_V_tr3_i2
    parameter[56] = 1.0 	# e0_greek_phi_V_tr4_i2
    parameter[57] = 1.0 	# e0_greek_phi_V_tr5_i2
    parameter[58] = 1.0 	# e0_greek_phi_V_tr6_i2
    parameter[59] = 1.0 	# e0_greek_phi_V_tr7_i2
    parameter[60] = 1.0 	# e0_greek_phi_V_tr8_i2
    parameter[61] = 276.37 	# e0_A_Lhnd100_i1
    parameter[62] = 105.8 	# e0_A_Lhnd100_i2
    parameter[63] = 72.55 	# e0_A_pLVd101_i1
    parameter[64] = 81.768 	# e0_A_pLVd101_i2
    parameter[65] = 52053.0 	# e0_A_LVhnd106_i1
    parameter[66] = 52390.0 	# e0_A_LVhnd106_i2
    parameter[67] = -2.0901 	# e0_B_Lhnd100_i1
    parameter[68] = -0.36223 	# e0_B_Lhnd100_i2
    parameter[69] = -7206.7 	# e0_B_pLVd101_i1
    parameter[70] = -6876.0 	# e0_B_pLVd101_i2
    parameter[71] = 0.3199 	# e0_B_LVhnd106_i1
    parameter[72] = 0.3682 	# e0_B_LVhnd106_i2
    parameter[73] = 0.008125 	# e0_C_Lhnd100_i1
    parameter[74] = 9.379E-4 	# e0_C_Lhnd100_i2
    parameter[75] = -7.1385 	# e0_C_pLVd101_i1
    parameter[76] = -8.7078 	# e0_C_pLVd101_i2
    parameter[77] = 0.0 	# e0_C_LVhnd106_i1
    parameter[78] = 0.0 	# e0_C_LVhnd106_i2
    parameter[79] = -1.4116E-5 	# e0_D_Lhnd100_i1
    parameter[80] = 0.0 	# e0_D_Lhnd100_i2
    parameter[81] = 4.046E-6 	# e0_D_pLVd101_i1
    parameter[82] = 7.1926E-6 	# e0_D_pLVd101_i2
    parameter[83] = 0.0 	# e0_D_LVhnd106_i1
    parameter[84] = 0.0 	# e0_D_LVhnd106_i2
    parameter[85] = 9.3701E-9 	# e0_E_Lhnd100_i1
    parameter[86] = 0.0 	# e0_E_Lhnd100_i2
    parameter[87] = 2.0 	# e0_E_pLVd101_i1
    parameter[88] = 2.0 	# e0_E_pLVd101_i2
    parameter[89] = 0.0 	# e0_E_LVhnd106_i1
    parameter[90] = 0.0 	# e0_E_LVhnd106_i2
    parameter[91] = 647.35 	# e0_T_crit_i1
    parameter[92] = 298.15 	# e0_T_ho_i1
    parameter[93] = 512.64 	# e0_T_crit_i2
    parameter[94] = 298.15 	# e0_T_ho_i2
    parameter[95] = 100000.0 	# e0_h_sca
    parameter[96] = -241820.0 	# e0_h_Lno_i1
    parameter[97] = -200940.0 	# e0_h_Lno_i2
    parameter[98] = 100000.0 	# e0_p_sca


    # Get symbolic quantities:
    xSymbolic = sympy.symbols('e0_F_Ln_tr2 e0_F_Ln_tr3 e0_F_Ln_tr4 e0_F_Ln_tr5 e0_F_Ln_tr6 e0_F_Ln_tr7 e0_F_Ln_tr8 e0_F_Ln_tr9 e0_F_Ln_tr10 e0_F_Ln_tr11 e0_F_Vn_tr1 e0_F_Vn_tr2 e0_F_Vn_tr3 e0_F_Vn_tr4 e0_F_Vn_tr5 e0_F_Vn_tr6 e0_F_Vn_tr7 e0_F_Vn_tr8 e0_h_Vn_tr0 e0_F_Vn_tr9 e0_F_Vn_tr10 e0_T_tr1 e0_T_tr2 e0_T_tr3 e0_T_tr4 e0_T_tr5 e0_T_tr6 e0_T_tr7 e0_T_tr8 e0_p_tr0 e0_T_tr9 e0_T_tr10 e0_h_Lfn e0_h_Ln_tr2 e0_h_Ln_tr3 e0_h_Ln_tr4 e0_h_Ln_tr5 e0_xR_i1 e0_h_Ln_tr6 e0_h_Ln_tr7 e0_h_Ln_tr8 e0_h_Ln_tr9 e0_h_Ln_tr10 e0_h_Ln_tr11 e0_xR_i2 e0_x_i1_tr1 e0_h_Vn_tr1 e0_h_Vn_tr2 e0_h_Vn_tr3 e0_h_Vn_tr4 e0_F_Ln_tr1 e0_x_i2_tr1 e0_h_Vn_tr5 e0_h_Vn_tr6 e0_h_Vn_tr7 e0_h_Vn_tr8 e0_h_Vn_tr9 e0_h_Vn_tr10 e0_p_tr1 e0_p_tr2 e0_p_tr3 e0_p_tr4 e0_y_i1_tr0 e0_p_tr5 e0_p_tr6 e0_p_tr7 e0_p_tr8 e0_p_tr9 e0_y_i2_tr0 e0_x_tr2_i1 e0_x_tr3_i1 e0_x_tr4_i1 e0_x_tr5_i1 e0_x_tr6_i1 e0_x_tr7_i1 e0_x_tr8_i1 e0_x_tr9_i1 e0_x_tr10_i1 e0_x_tr11_i1 e0_x_tr2_i2 e0_x_tr3_i2 e0_x_tr4_i2 e0_x_tr5_i2 e0_x_tr6_i2 e0_x_tr7_i2 e0_x_tr8_i2 e0_x_tr9_i2 e0_x_tr10_i2 e0_x_tr11_i2 e0_y_tr1_i1 e0_y_tr2_i1 e0_y_tr3_i1 e0_y_tr4_i1 e0_y_tr5_i1 e0_y_tr6_i1 e0_y_tr7_i1 e0_y_tr8_i1 e0_y_tr9_i1 e0_y_tr10_i1 e0_y_tr1_i2 e0_y_tr2_i2 e0_y_tr3_i2 e0_y_tr4_i2 e0_y_tr5_i2 e0_y_tr6_i2 e0_y_tr7_i2 e0_y_tr8_i2 e0_y_tr9_i2 e0_y_tr10_i2 e0_F_LnC e0_QC e0_TC e0_yC_i1 e0_yC_i2 e0_F_LnR e0_F_Vn_tr0 e0_QR e0_h_Ln_tr1 e0_h_LnR ')
    fSymbolic = getSymbolicFunctions(xSymbolic, parameter)

    # Intial variable bounds:
    xInitial[0] = mpmath.mpi(1e-9, 10.0)  	# e0_F_Ln_tr2
    xInitial[1] = mpmath.mpi(1e-9, 10.0)  	# e0_F_Ln_tr3
    xInitial[2] = mpmath.mpi(1e-9, 10.0)  	# e0_F_Ln_tr4
    xInitial[3] = mpmath.mpi(1e-9, 10.0)  	# e0_F_Ln_tr5
    xInitial[4] = mpmath.mpi(1e-9, 10.0)  	# e0_F_Ln_tr6
    xInitial[5] = mpmath.mpi(1e-9, 10.0)  	# e0_F_Ln_tr7
    xInitial[6] = mpmath.mpi(1e-9, 10.0)  	# e0_F_Ln_tr8
    xInitial[7] = mpmath.mpi(1e-9, 10.0)  	# e0_F_Ln_tr9
    xInitial[8] = mpmath.mpi(1e-9, 10.0)  	# e0_F_Ln_tr10
    xInitial[9] = mpmath.mpi(1e-9, 10.0)  	# e0_F_Ln_tr11
    xInitial[10] = mpmath.mpi(1e-9, 10.0)  	# e0_F_Vn_tr1
    xInitial[11] = mpmath.mpi(1e-9, 10.0)  	# e0_F_Vn_tr2
    xInitial[12] = mpmath.mpi(1e-9, 10.0)  	# e0_F_Vn_tr3
    xInitial[13] = mpmath.mpi(1e-9, 10.0)  	# e0_F_Vn_tr4
    xInitial[14] = mpmath.mpi(1e-9, 10.0)  	# e0_F_Vn_tr5
    xInitial[15] = mpmath.mpi(1e-9, 10.0)  	# e0_F_Vn_tr6
    xInitial[16] = mpmath.mpi(1e-9, 10.0)  	# e0_F_Vn_tr7
    xInitial[17] = mpmath.mpi(1e-9, 10.0)  	# e0_F_Vn_tr8
    xInitial[18] = mpmath.mpi(-1000000.0, -1e-9)  	# e0_h_Vn_tr0
    xInitial[19] = mpmath.mpi(1e-15, 10.0)  	# e0_F_Vn_tr9
    xInitial[20] = mpmath.mpi(1e-15, 10.0)  	# e0_F_Vn_tr10
    xInitial[21] = mpmath.mpi(240.0, 400.0)  	# e0_T_tr1
    xInitial[22] = mpmath.mpi(240.0, 400.0)  	# e0_T_tr2
    xInitial[23] = mpmath.mpi(240.0, 400.0)  	# e0_T_tr3
    xInitial[24] = mpmath.mpi(240.0, 400.0)  	# e0_T_tr4
    xInitial[25] = mpmath.mpi(240.0, 400.0)  	# e0_T_tr5
    xInitial[26] = mpmath.mpi(240.0, 400.0)  	# e0_T_tr6
    xInitial[27] = mpmath.mpi(240.0, 400.0)  	# e0_T_tr7
    xInitial[28] = mpmath.mpi(240.0, 400.0)  	# e0_T_tr8
    xInitial[29] = mpmath.mpi(1000.0, 1000000.0)  	# e0_p_tr0
    xInitial[30] = mpmath.mpi(240.0, 400.0)  	# e0_T_tr9
    xInitial[31] = mpmath.mpi(240.0, 400.0)  	# e0_T_tr10
    xInitial[32] = mpmath.mpi(-1000000.0, -1e-9)  	# e0_h_Lfn
    xInitial[33] = mpmath.mpi(-1000000.0, -1e-9)  	# e0_h_Ln_tr2
    xInitial[34] = mpmath.mpi(-1000000.0, -1e-9)  	# e0_h_Ln_tr3
    xInitial[35] = mpmath.mpi(-1000000.0, -1e-9)  	# e0_h_Ln_tr4
    xInitial[36] = mpmath.mpi(-1000000.0, -1e-9)  	# e0_h_Ln_tr5
    xInitial[37] = mpmath.mpi(1e-15, 1.0)  	# e0_xR_i1
    xInitial[38] = mpmath.mpi(-1000000.0, -1e-9)  	# e0_h_Ln_tr6
    xInitial[39] = mpmath.mpi(-1000000.0, -1e-9)  	# e0_h_Ln_tr7
    xInitial[40] = mpmath.mpi(-1000000.0, -1e-9)  	# e0_h_Ln_tr8
    xInitial[41] = mpmath.mpi(-1000000.0, -1e-9)  	# e0_h_Ln_tr9
    xInitial[42] = mpmath.mpi(-1000000.0, -1e-9)  	# e0_h_Ln_tr10
    xInitial[43] = mpmath.mpi(-1000000.0, -1e-9)  	# e0_h_Ln_tr11
    xInitial[44] = mpmath.mpi(1e-15, 1.0)  	# e0_xR_i2
    xInitial[45] = mpmath.mpi(1e-15, 1.0)  	# e0_x_i1_tr1
    xInitial[46] = mpmath.mpi(-1000000.0, -1e-9)  	# e0_h_Vn_tr1
    xInitial[47] = mpmath.mpi(-1000000.0, -1e-9)  	# e0_h_Vn_tr2
    xInitial[48] = mpmath.mpi(-1000000.0, -1e-9)  	# e0_h_Vn_tr3
    xInitial[49] = mpmath.mpi(-1000000.0, -1e-9)  	# e0_h_Vn_tr4
    xInitial[50] = mpmath.mpi(1e-9, 10.0)  	# e0_F_Ln_tr1
    xInitial[51] = mpmath.mpi(1e-15, 1.0)  	# e0_x_i2_tr1
    xInitial[52] = mpmath.mpi(-1000000.0, 0.0)  	# e0_h_Vn_tr5
    xInitial[53] = mpmath.mpi(-1000000.0, 0.0)  	# e0_h_Vn_tr6
    xInitial[54] = mpmath.mpi(-1000000.0, 0.0)  	# e0_h_Vn_tr7
    xInitial[55] = mpmath.mpi(-1000000.0, 0.0)  	# e0_h_Vn_tr8
    xInitial[56] = mpmath.mpi(-1000000.0, 0.0)  	# e0_h_Vn_tr9
    xInitial[57] = mpmath.mpi(-1000000.0, 0.0)  	# e0_h_Vn_tr10
    xInitial[58] = mpmath.mpi(1000.0, 1000000.0)  	# e0_p_tr1
    xInitial[59] = mpmath.mpi(1000.0, 1000000.0)   	# e0_p_tr2
    xInitial[60] = mpmath.mpi(1000.0, 1000000.0)   	# e0_p_tr3
    xInitial[61] = mpmath.mpi(1000.0, 1000000.0)  	# e0_p_tr4
    xInitial[62] = mpmath.mpi(1e-15, 1.0)  	# e0_y_i1_tr0
    xInitial[63] = mpmath.mpi(1000.0, 1000000.0)  	# e0_p_tr5
    xInitial[64] = mpmath.mpi(1000.0, 1000000.0)   	# e0_p_tr6
    xInitial[65] = mpmath.mpi(1000.0, 1000000.0)   	# e0_p_tr7
    xInitial[66] = mpmath.mpi(1000.0, 1000000.0)   	# e0_p_tr8
    xInitial[67] = mpmath.mpi(1000.0, 1000000.0)  	# e0_p_tr9
    xInitial[68] = mpmath.mpi(1e-15, 1.0)  	# e0_y_i2_tr0
    xInitial[69] = mpmath.mpi(1e-15, 1.0)  	# e0_x_tr2_i1
    xInitial[70] = mpmath.mpi(1e-15, 1.0)  	# e0_x_tr3_i1
    xInitial[71] = mpmath.mpi(1e-15, 1.0)  	# e0_x_tr4_i1
    xInitial[72] = mpmath.mpi(1e-15, 1.0)  	# e0_x_tr5_i1
    xInitial[73] = mpmath.mpi(1e-15, 1.0)  	# e0_x_tr6_i1
    xInitial[74] = mpmath.mpi(1e-15, 1.0)  	# e0_x_tr7_i1
    xInitial[75] = mpmath.mpi(1e-15, 1.0)  	# e0_x_tr8_i1
    xInitial[76] = mpmath.mpi(1e-15, 1.0)  	# e0_x_tr9_i1
    xInitial[77] = mpmath.mpi(1e-15, 1.0)  	# e0_x_tr10_i1
    xInitial[78] = mpmath.mpi(1e-15, 1.0)  	# e0_x_tr11_i1
    xInitial[79] = mpmath.mpi(1e-15, 1.0)  	# e0_x_tr2_i2
    xInitial[80] = mpmath.mpi(1e-15, 1.0)  	# e0_x_tr3_i2
    xInitial[81] = mpmath.mpi(1e-15, 1.0)  	# e0_x_tr4_i2
    xInitial[82] = mpmath.mpi(1e-15, 1.0)  	# e0_x_tr5_i2
    xInitial[83] = mpmath.mpi(1e-15, 1.0)  	# e0_x_tr6_i2
    xInitial[84] = mpmath.mpi(1e-15, 1.0)  	# e0_x_tr7_i2
    xInitial[85] = mpmath.mpi(1e-15, 1.0)  	# e0_x_tr8_i2
    xInitial[86] = mpmath.mpi(1e-15, 1.0)  	# e0_x_tr9_i2
    xInitial[87] = mpmath.mpi(1e-15, 1.0)  	# e0_x_tr10_i2
    xInitial[88] = mpmath.mpi(1e-15, 1.0)  	# e0_x_tr11_i2
    xInitial[89] = mpmath.mpi(1e-15, 1.0)  	# e0_y_tr1_i1
    xInitial[90] = mpmath.mpi(1e-15, 1.0)  	# e0_y_tr2_i1
    xInitial[91] = mpmath.mpi(1e-15, 1.0)  	# e0_y_tr3_i1
    xInitial[92] = mpmath.mpi(1e-15, 1.0)  	# e0_y_tr4_i1
    xInitial[93] = mpmath.mpi(1e-15, 1.0)  	# e0_y_tr5_i1
    xInitial[94] = mpmath.mpi(1e-15, 1.0)  	# e0_y_tr6_i1
    xInitial[95] = mpmath.mpi(1e-15, 1.0)  	# e0_y_tr7_i1
    xInitial[96] = mpmath.mpi(1e-15, 1.0)  	# e0_y_tr8_i1
    xInitial[97] = mpmath.mpi(1e-15, 1.0)  	# e0_y_tr9_i1
    xInitial[98] = mpmath.mpi(1e-15, 1.0)  	# e0_y_tr10_i1
    xInitial[99] = mpmath.mpi(1e-15, 1.0)  	# e0_y_tr1_i2
    xInitial[100] = mpmath.mpi(1e-15, 1.0)  	# e0_y_tr2_i2
    xInitial[101] = mpmath.mpi(1e-15, 1.0)  	# e0_y_tr3_i2
    xInitial[102] = mpmath.mpi(1e-15, 1.0)  	# e0_y_tr4_i2
    xInitial[103] = mpmath.mpi(1e-15, 1.0)  	# e0_y_tr5_i2
    xInitial[104] = mpmath.mpi(1e-15, 1.0)  	# e0_y_tr6_i2
    xInitial[105] = mpmath.mpi(1e-15, 1.0)  	# e0_y_tr7_i2
    xInitial[106] = mpmath.mpi(1e-15, 1.0)  	# e0_y_tr8_i2
    xInitial[107] = mpmath.mpi(1e-15, 1.0)  	# e0_y_tr9_i2
    xInitial[108] = mpmath.mpi(1e-15, 1.0)  	# e0_y_tr10_i2
    xInitial[109] = mpmath.mpi(1e-15, 5.0)  	# e0_F_LnC
    xInitial[110] = mpmath.mpi(-1000000.0,-1e-15)  	# e0_QC
    xInitial[111] = mpmath.mpi(240.0, 400.0)  	# e0_TC
    xInitial[112] = mpmath.mpi(1e-15, 1.0)  	# e0_yC_i1
    xInitial[113] = mpmath.mpi(1e-15, 1.0)  	# e0_yC_i2
    xInitial[114] = mpmath.mpi(1e-15, 10.0)  	# e0_F_LnR
    xInitial[115] = mpmath.mpi(1e-15, 10.0)  	# e0_F_Vn_tr0
    xInitial[116] = mpmath.mpi(1e-15, 1000000.0)  	# e0_QR
    xInitial[117] = mpmath.mpi(-1e6, -1e-9)  	# e0_h_Ln_tr1
    xInitial[118] = mpmath.mpi(-1e6, -1e-9)  	# e0_h_LnR


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

