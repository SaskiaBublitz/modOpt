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
    dict_options = {"fileName": "HeatEx_r3",
                    "iterMaxNewton": 15,
                    "machEpsRelNewton": 2.22e-14,
                    "machEpsAbsNewton": 2.22e-14,
                    "absTolX": 2.22e-14, #numpy.finfo(numpy.float).eps
                    "relTolX": 2.22e-14,
                    "absTolF": 2.22e-14,
                    "relTolF": 2.22e-2,
                    "resolution": 3,
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
def fun_69026(std_greek_rho_Ln,std_h_LVn,std_h_Vn,std_p,std_p_LV): # fun_69026__enthalpy_molar_liquid_incompressible
	return std_h_Vn-std_h_LVn+((std_p-std_p_LV))/(std_greek_rho_Ln)
def fun_75394(std_X,std_A,std_B,std_C,std_D,std_E,std_T_ho,std_h_o): # fun_75394__enthalpy_IG_DIPPR100
	return std_h_o+(std_A) *((std_X-std_T_ho))+(std_B) *(((((std_X))**(2.0)-((std_T_ho))**(2.0)))/(2.0))+(std_C) *(((((std_X))**(3.0)-((std_T_ho))**(3.0)))/(3.0))+(std_D) *(((((std_X))**(4.0)-((std_T_ho))**(4.0)))/(4.0))+(std_E) *(((((std_X))**(5.0)-((std_T_ho))**(5.0)))/(5.0))
def fun_75386(std_X,std_A,std_B,std_C,std_D,std_E): # fun_75386__DIPPR101
	return sympy.exp(std_A+(std_B)/(std_X)+(std_C) *(sympy.log(std_X))+(std_D) *(((std_X))**(std_E)))
def fun_75389(std_X,std_A,std_B,std_C,std_D): # fun_75389__DIPPR105
	return (std_A)/(((std_B))**(1.0+((1.0-(std_X)/(std_C)))**(std_D)))
def fun_75388(std_X,std_A,std_B,std_C,std_D,std_E): # fun_75388__DIPPR100
	return std_A+(std_B) *(std_X)+(std_C) *(((std_X))**(2.0))+(std_D) *(((std_X))**(3.0))+(std_E) *(((std_X))**(4.0))
def fun_75390(std_X,std_A,std_B,std_C,std_D,std_E,std_T_crit): # fun_75390__DIPPR106
	return (std_A) *(((1.0-(std_X)/(std_T_crit)))**(std_B+(std_C) *((std_X)/(std_T_crit))+(std_D) *((((std_X)/(std_T_crit)))**(2.0))+(std_E) *((((std_X)/(std_T_crit)))**(3.0))))

def getSymbolicFunctions(x, p):
    """ User specified Equation system 
    
    Args:
        x:          sympy array with symbolic state variable bounds
        p:          numpy array with parameter values
    
    Returns:
        f:          sympy array with symbolic residual bounds
        
    """
# Getting variable values:
    e0_greek_DeltaT_logsh = x[0]
    e0_Q_sh = x[1]
    e0_HU_hnph2_i1 = x[2]
    e0_HU_hnph2_i2 = x[3]
    e0_K_hinph2_i1 = x[4]
    e0_K_hinph2_i2 = x[5]
    e0_K_houtph2_i1 = x[6]
    e0_K_houtph2_i2 = x[7]
    e0_Q_ph2 = x[8]
    e0_T_coutsc = x[9]
    e0_T_hinsc = x[10]
    e0_U_hph2 = x[11]
    e0_T_coutph2 = x[12]
    e0_V_Lhph2 = x[13]
    e0_V_Vhph2 = x[14]
    e0_h_Lhinnph2 = x[15]
    e0_h_hinnsc = x[16]
    e0_T_coutsh = x[17]
    e0_h_Vhnoutph2 = x[18]
    e0_h_cnoutsc = x[19]
    e0_T_hinph2 = x[20]
    e0_x_coutsc_k1 = x[21]
    e0_x_hinph2_i1 = x[22]
    e0_x_hinph2_i2 = x[23]
    e0_x_hinsc_i1 = x[24]
    e0_x_hinsc_i2 = x[25]
    e0_y_houtph2_i1 = x[26]
    e0_y_houtph2_i2 = x[27]
    e0_U_hsh = x[28]
    e0_V_hsh = x[29]
    e0_greek_DeltaT_logsc = x[30]
    e0_greek_rho_Lhnoutsc = x[31]
    e0_A_sc = x[32]
    e0_F_hnoutsc = x[33]
    e0_HU_hnsc = x[34]
    e0_HU_hnsc_i1 = x[35]
    e0_HU_hnsc_i2 = x[36]
    e0_Q_sc = x[37]
    e0_T_houtsc = x[38]
    e0_U_hsc = x[39]
    e0_V_hsc = x[40]
    e0_A_sh = x[41]
    e0_h_cnoutph2 = x[42]
    e0_h_cinnsc = x[43]
    e0_h_hnoutsc = x[44]
    e0_h_cnoutsh = x[45]
    e0_h_hinnsh = x[46]
    e0_x_houtsc_i1 = x[47]
    e0_x_houtsc_i2 = x[48]
    e0_A = x[49]
    e0_A_V = x[50]
    e0_A_aux = x[51]
    e0_A_auxsc = x[52]
    e0_A_auxsh = x[53]
    e0_H_sctotal = x[54]
    e0_H_shtotal = x[55]
    e0_N_Lpph2 = x[56]
    e0_N_psc = x[57]
    e0_N_psh = x[58]
    e0_V = x[59]
    e0_V_L = x[60]
    e0_V_Lpph2 = x[61]
    e0_V_Lph2total = x[62]
    e0_V_Ltotal = x[63]
    e0_V_Vh = x[64]
    e0_V_Vpph2 = x[65]
    e0_V_Vph2total = x[66]
    e0_V_Vtotal = x[67]
    e0_V_psc = x[68]
    e0_V_psh = x[69]
    e0_V_sctotal = x[70]
    e0_V_shtotal = x[71]
    e0_h_Vhinnph2 = x[72]
    e0_F_cnoutph2 = x[73]
    e0_x_coutph2_k1 = x[74]
    e0_x_coutsh_k1 = x[75]
    e0_y_hinph2_i1 = x[76]
    e0_y_hinph2_i2 = x[77]
    e0_F_cnoutsh = x[78]
    e0_F_Vhinnph2 = x[79]
    e0_HU_hnsh = x[80]
    e0_greek_DeltaT_logph2 = x[81]
    e0_greek_alpha_hinph2wilson_i1 = x[82]
    e0_greek_alpha_hinph2wilson_i2 = x[83]
    e0_greek_alpha_houtph2wilson_i1 = x[84]
    e0_HU_hnsh_i1 = x[85]
    e0_greek_alpha_houtph2wilson_i2 = x[86]
    e0_greek_gamma_hinph2_i1 = x[87]
    e0_greek_gamma_hinph2_i2 = x[88]
    e0_greek_gamma_houtph2_i1 = x[89]
    e0_greek_gamma_houtph2_i2 = x[90]
    e0_HU_hnsh_i2 = x[91]
    e0_greek_rho_Lhnoutph2 = x[92]
    e0_A_ph2 = x[93]
    e0_F_hinnsc = x[94]
    e0_F_cnoutsc = x[95]
    e0_HU_Lhnph2 = x[96]
    e0_HU_Vhnph2 = x[97]



# Getting parameter values:
    e0_T_hinsh = p[0]
    e0_k_ph2 = p[1]
    e0_p_coutph2 = p[2]
    e0_p_hinph2 = p[3]
    e0_p_houtph2 = p[4]
    e0_v_L_i1 = p[5]
    e0_v_L_i2 = p[6]
    e0_F_cinnsc = p[7]
    e0_T_cinsc = p[8]
    e0_k_sc = p[9]
    e0_p_cinsc = p[10]
    e0_p_coutsc = p[11]
    e0_p_houtsc = p[12]
    e0_x_cinsc_k1 = p[13]
    e0_H_Ltotal = p[14]
    e0_H_dim = p[15]
    e0_k_sh = p[16]
    e0_p_coutsh = p[17]
    e0_p_houtsh = p[18]
    e0_y_hinsh_i1 = p[19]
    e0_y_hinsh_i2 = p[20]
    e0_F_hinnsh = p[21]
    e0_greek_lambdawilson_i1 = p[22]
    e0_greek_lambdawilson_i2 = p[23]
    e0_F_Lhinnph2 = p[24]
    e0_F_Vhnoutph2 = p[25]
    e0_greek_pi = p[26]
    e0_AcNP = p[27]
    e0_A_cpd100_i1 = p[28]
    e0_A_Lnrhod100_i2 = p[29]
    e0_A_cpd100_i2 = p[30]
    e0_A_Lnrhod100_k1 = p[31]
    e0_A_cpd100_k1 = p[32]
    e0_A_LVpd101_i1 = p[33]
    e0_A_LVpd101_i2 = p[34]
    e0_A_LVpd101_k1 = p[35]
    e0_A_Lnrhod105_i1 = p[36]
    e0_A_LVhd106_i1 = p[37]
    e0_A_LVhd106_i2 = p[38]
    e0_A_LVhd106_k1 = p[39]
    e0_BcNP = p[40]
    e0_B_cpd100_i1 = p[41]
    e0_B_Lnrhod100_i2 = p[42]
    e0_B_cpd100_i2 = p[43]
    e0_B_Lnrhod100_k1 = p[44]
    e0_B_cpd100_k1 = p[45]
    e0_B_LVpd101_i1 = p[46]
    e0_B_LVpd101_i2 = p[47]
    e0_B_LVpd101_k1 = p[48]
    e0_B_Lnrhod105_i1 = p[49]
    e0_B_LVhd106_i1 = p[50]
    e0_B_LVhd106_i2 = p[51]
    e0_B_LVhd106_k1 = p[52]
    e0_CcNP = p[53]
    e0_C_cpd100_i1 = p[54]
    e0_C_Lnrhod100_i2 = p[55]
    e0_C_cpd100_i2 = p[56]
    e0_C_Lnrhod100_k1 = p[57]
    e0_C_cpd100_k1 = p[58]
    e0_C_LVpd101_i1 = p[59]
    e0_C_LVpd101_i2 = p[60]
    e0_C_LVpd101_k1 = p[61]
    e0_C_Lnrhod105_i1 = p[62]
    e0_C_LVhd106_i1 = p[63]
    e0_C_LVhd106_i2 = p[64]
    e0_C_LVhd106_k1 = p[65]
    e0_D_i = p[66]
    e0_D_cpd100_i1 = p[67]
    e0_D_Lnrhod100_i2 = p[68]
    e0_D_cpd100_i2 = p[69]
    e0_D_Lnrhod100_k1 = p[70]
    e0_D_cpd100_k1 = p[71]
    e0_D_LVpd101_i1 = p[72]
    e0_D_LVpd101_i2 = p[73]
    e0_D_LVpd101_k1 = p[74]
    e0_D_Lnrhod105_i1 = p[75]
    e0_D_LVhd106_i1 = p[76]
    e0_D_LVhd106_i2 = p[77]
    e0_D_LVhd106_k1 = p[78]
    e0_E_cpd100_i1 = p[79]
    e0_E_Lnrhod100_i2 = p[80]
    e0_E_cpd100_i2 = p[81]
    e0_E_Lnrhod100_k1 = p[82]
    e0_E_cpd100_k1 = p[83]
    e0_E_LVpd101_i1 = p[84]
    e0_E_LVpd101_i2 = p[85]
    e0_E_LVpd101_k1 = p[86]
    e0_E_LVhd106_i1 = p[87]
    e0_E_LVhd106_i2 = p[88]
    e0_E_LVhd106_k1 = p[89]
    e0_L = p[90]
    e0_N_ptotal = p[91]
    e0_R = p[92]
    e0_T_crit_i1 = p[93]
    e0_T_ho_i1 = p[94]
    e0_T_crit_i2 = p[95]
    e0_T_ho_i2 = p[96]
    e0_T_crit_k1 = p[97]
    e0_T_ho_k1 = p[98]
    e0_d_o = p[99]
    e0_h_o_i1 = p[100]
    e0_h_o_i2 = p[101]
    e0_h_o_k1 = p[102]



   # Getting function values:
    e0_h_LVcnoutsc_k1 = fun_75390(e0_T_coutsc,e0_A_LVhd106_k1,e0_B_LVhd106_k1,e0_C_LVhd106_k1,e0_D_LVhd106_k1,e0_E_LVhd106_k1,e0_T_crit_k1)
    e0_greek_rho_Lhnoutsc_i1 = fun_75389(e0_T_houtsc,e0_A_Lnrhod105_i1,e0_B_Lnrhod105_i1,e0_C_Lnrhod105_i1,e0_D_Lnrhod105_i1)
    e0_h_LVcinnsc_k1 = fun_75390(e0_T_cinsc,e0_A_LVhd106_k1,e0_B_LVhd106_k1,e0_C_LVhd106_k1,e0_D_LVhd106_k1,e0_E_LVhd106_k1,e0_T_crit_k1)
    e0_greek_rho_Lcnoutsh_k1 = fun_75388(e0_T_coutsh,e0_A_Lnrhod100_k1,e0_B_Lnrhod100_k1,e0_C_Lnrhod100_k1,e0_D_Lnrhod100_k1,e0_E_Lnrhod100_k1)
    e0_h_Vhnoutph2_i2 = fun_75394(e0_T_hinsc,e0_A_cpd100_i2,e0_B_cpd100_i2,e0_C_cpd100_i2,e0_D_cpd100_i2,e0_E_cpd100_i2,e0_T_ho_i2,e0_h_o_i2)
    e0_h_hinnsh_i2 = fun_75394(e0_T_hinsh,e0_A_cpd100_i2,e0_B_cpd100_i2,e0_C_cpd100_i2,e0_D_cpd100_i2,e0_E_cpd100_i2,e0_T_ho_i2,e0_h_o_i2)
    e0_h_LVhnoutsc_i1 = fun_75390(e0_T_houtsc,e0_A_LVhd106_i1,e0_B_LVhd106_i1,e0_C_LVhd106_i1,e0_D_LVhd106_i1,e0_E_LVhd106_i1,e0_T_crit_i1)
    e0_p_LVhoutsc_i1 = fun_75386(e0_T_houtsc,e0_A_LVpd101_i1,e0_B_LVpd101_i1,e0_C_LVpd101_i1,e0_D_LVpd101_i1,e0_E_LVpd101_i1)
    e0_p_LVhoutph2_i1 = fun_75386(e0_T_hinsc,e0_A_LVpd101_i1,e0_B_LVpd101_i1,e0_C_LVpd101_i1,e0_D_LVpd101_i1,e0_E_LVpd101_i1)
    e0_greek_rho_Lcnoutph2_k1 = fun_75388(e0_T_coutph2,e0_A_Lnrhod100_k1,e0_B_Lnrhod100_k1,e0_C_Lnrhod100_k1,e0_D_Lnrhod100_k1,e0_E_Lnrhod100_k1)
    e0_h_LVhnoutph2_i2 = fun_75390(e0_T_hinsc,e0_A_LVhd106_i2,e0_B_LVhd106_i2,e0_C_LVhd106_i2,e0_D_LVhd106_i2,e0_E_LVhd106_i2,e0_T_crit_i2)
    e0_greek_rho_Lcnoutsc_k1 = fun_75388(e0_T_coutsc,e0_A_Lnrhod100_k1,e0_B_Lnrhod100_k1,e0_C_Lnrhod100_k1,e0_D_Lnrhod100_k1,e0_E_Lnrhod100_k1)
    e0_h_hnoutsh_i2 = fun_75394(e0_T_hinph2,e0_A_cpd100_i2,e0_B_cpd100_i2,e0_C_cpd100_i2,e0_D_cpd100_i2,e0_E_cpd100_i2,e0_T_ho_i2,e0_h_o_i2)
    e0_h_LVcnoutph2_k1 = fun_75390(e0_T_coutph2,e0_A_LVhd106_k1,e0_B_LVhd106_k1,e0_C_LVhd106_k1,e0_D_LVhd106_k1,e0_E_LVhd106_k1,e0_T_crit_k1)
    e0_p_LVcoutsh_k1 = fun_75386(e0_T_coutsh,e0_A_LVpd101_k1,e0_B_LVpd101_k1,e0_C_LVpd101_k1,e0_D_LVpd101_k1,e0_E_LVpd101_k1)
    e0_greek_rho_Lhinnph2_i2 = fun_75388(e0_T_hinph2,e0_A_Lnrhod100_i2,e0_B_Lnrhod100_i2,e0_C_Lnrhod100_i2,e0_D_Lnrhod100_i2,e0_E_Lnrhod100_i2)
    e0_greek_rho_Lhnoutsc_i2 = fun_75388(e0_T_houtsc,e0_A_Lnrhod100_i2,e0_B_Lnrhod100_i2,e0_C_Lnrhod100_i2,e0_D_Lnrhod100_i2,e0_E_Lnrhod100_i2)
    e0_h_LVhinnph2_i2 = fun_75390(e0_T_hinph2,e0_A_LVhd106_i2,e0_B_LVhd106_i2,e0_C_LVhd106_i2,e0_D_LVhd106_i2,e0_E_LVhd106_i2,e0_T_crit_i2)
    e0_greek_rho_Lhinnph2_i1 = fun_75389(e0_T_hinph2,e0_A_Lnrhod105_i1,e0_B_Lnrhod105_i1,e0_C_Lnrhod105_i1,e0_D_Lnrhod105_i1)
    e0_h_LVhnoutsc_i2 = fun_75390(e0_T_houtsc,e0_A_LVhd106_i2,e0_B_LVhd106_i2,e0_C_LVhd106_i2,e0_D_LVhd106_i2,e0_E_LVhd106_i2,e0_T_crit_i2)
    e0_h_hnoutsh_i1 = fun_75394(e0_T_hinph2,e0_A_cpd100_i1,e0_B_cpd100_i1,e0_C_cpd100_i1,e0_D_cpd100_i1,e0_E_cpd100_i1,e0_T_ho_i1,e0_h_o_i1)
    e0_p_LVhinph2_i1 = fun_75386(e0_T_hinph2,e0_A_LVpd101_i1,e0_B_LVpd101_i1,e0_C_LVpd101_i1,e0_D_LVpd101_i1,e0_E_LVpd101_i1)
    e0_h_LVhnoutph2_i1 = fun_75390(e0_T_hinsc,e0_A_LVhd106_i1,e0_B_LVhd106_i1,e0_C_LVhd106_i1,e0_D_LVhd106_i1,e0_E_LVhd106_i1,e0_T_crit_i1)
    e0_greek_rho_Lcinnsc_k1 = fun_75388(e0_T_cinsc,e0_A_Lnrhod100_k1,e0_B_Lnrhod100_k1,e0_C_Lnrhod100_k1,e0_D_Lnrhod100_k1,e0_E_Lnrhod100_k1)
    e0_greek_rho_Lhnoutph2_i1 = fun_75389(e0_T_hinsc,e0_A_Lnrhod105_i1,e0_B_Lnrhod105_i1,e0_C_Lnrhod105_i1,e0_D_Lnrhod105_i1)
    e0_h_Vhinnph2_i1 = fun_75394(e0_T_hinph2,e0_A_cpd100_i1,e0_B_cpd100_i1,e0_C_cpd100_i1,e0_D_cpd100_i1,e0_E_cpd100_i1,e0_T_ho_i1,e0_h_o_i1)
    e0_h_LVcnoutsh_k1 = fun_75390(e0_T_coutsh,e0_A_LVhd106_k1,e0_B_LVhd106_k1,e0_C_LVhd106_k1,e0_D_LVhd106_k1,e0_E_LVhd106_k1,e0_T_crit_k1)
    e0_h_Vhnoutph2_i1 = fun_75394(e0_T_hinsc,e0_A_cpd100_i1,e0_B_cpd100_i1,e0_C_cpd100_i1,e0_D_cpd100_i1,e0_E_cpd100_i1,e0_T_ho_i1,e0_h_o_i1)
    e0_h_LVhinnph2_i1 = fun_75390(e0_T_hinph2,e0_A_LVhd106_i1,e0_B_LVhd106_i1,e0_C_LVhd106_i1,e0_D_LVhd106_i1,e0_E_LVhd106_i1,e0_T_crit_i1)
    e0_h_Vhinnph2_i2 = fun_75394(e0_T_hinph2,e0_A_cpd100_i2,e0_B_cpd100_i2,e0_C_cpd100_i2,e0_D_cpd100_i2,e0_E_cpd100_i2,e0_T_ho_i2,e0_h_o_i2)
    e0_p_LVhoutsc_i2 = fun_75386(e0_T_houtsc,e0_A_LVpd101_i2,e0_B_LVpd101_i2,e0_C_LVpd101_i2,e0_D_LVpd101_i2,e0_E_LVpd101_i2)
    e0_h_Vhnoutsc_i2 = fun_75394(e0_T_houtsc,e0_A_cpd100_i2,e0_B_cpd100_i2,e0_C_cpd100_i2,e0_D_cpd100_i2,e0_E_cpd100_i2,e0_T_ho_i2,e0_h_o_i2)
    e0_p_LVhoutph2_i2 = fun_75386(e0_T_hinsc,e0_A_LVpd101_i2,e0_B_LVpd101_i2,e0_C_LVpd101_i2,e0_D_LVpd101_i2,e0_E_LVpd101_i2)
    e0_p_LVcoutsc_k1 = fun_75386(e0_T_coutsc,e0_A_LVpd101_k1,e0_B_LVpd101_k1,e0_C_LVpd101_k1,e0_D_LVpd101_k1,e0_E_LVpd101_k1)
    e0_h_Vcnoutsh_k1 = fun_75394(e0_T_coutsh,e0_A_cpd100_k1,e0_B_cpd100_k1,e0_C_cpd100_k1,e0_D_cpd100_k1,e0_E_cpd100_k1,e0_T_ho_k1,e0_h_o_k1)
    e0_h_Vcinnsc_k1 = fun_75394(e0_T_cinsc,e0_A_cpd100_k1,e0_B_cpd100_k1,e0_C_cpd100_k1,e0_D_cpd100_k1,e0_E_cpd100_k1,e0_T_ho_k1,e0_h_o_k1)
    e0_h_Vhnoutsc_i1 = fun_75394(e0_T_houtsc,e0_A_cpd100_i1,e0_B_cpd100_i1,e0_C_cpd100_i1,e0_D_cpd100_i1,e0_E_cpd100_i1,e0_T_ho_i1,e0_h_o_i1)
    e0_h_hinnsh_i1 = fun_75394(e0_T_hinsh,e0_A_cpd100_i1,e0_B_cpd100_i1,e0_C_cpd100_i1,e0_D_cpd100_i1,e0_E_cpd100_i1,e0_T_ho_i1,e0_h_o_i1)
    e0_p_LVhinph2_i2 = fun_75386(e0_T_hinph2,e0_A_LVpd101_i2,e0_B_LVpd101_i2,e0_C_LVpd101_i2,e0_D_LVpd101_i2,e0_E_LVpd101_i2)
    e0_greek_rho_Lhnoutph2_i2 = fun_75388(e0_T_hinsc,e0_A_Lnrhod100_i2,e0_B_Lnrhod100_i2,e0_C_Lnrhod100_i2,e0_D_Lnrhod100_i2,e0_E_Lnrhod100_i2)
    e0_h_Vcnoutsc_k1 = fun_75394(e0_T_coutsc,e0_A_cpd100_k1,e0_B_cpd100_k1,e0_C_cpd100_k1,e0_D_cpd100_k1,e0_E_cpd100_k1,e0_T_ho_k1,e0_h_o_k1)
    e0_p_LVcoutph2_k1 = fun_75386(e0_T_coutph2,e0_A_LVpd101_k1,e0_B_LVpd101_k1,e0_C_LVpd101_k1,e0_D_LVpd101_k1,e0_E_LVpd101_k1)
    e0_h_Vcnoutph2_k1 = fun_75394(e0_T_coutph2,e0_A_cpd100_k1,e0_B_cpd100_k1,e0_C_cpd100_k1,e0_D_cpd100_k1,e0_E_cpd100_k1,e0_T_ho_k1,e0_h_o_k1)
    e0_p_LVcinsc_k1 = fun_75386(e0_T_cinsc,e0_A_LVpd101_k1,e0_B_LVpd101_k1,e0_C_LVpd101_k1,e0_D_LVpd101_k1,e0_E_LVpd101_k1)
    e0_h_Lhinnph2_i2 = fun_69026(e0_greek_rho_Lhinnph2_i2,e0_h_LVhinnph2_i2,e0_h_Vhinnph2_i2,e0_p_hinph2,e0_p_LVhinph2_i2)
    e0_h_Lhnoutph2_i2 = fun_69026(e0_greek_rho_Lhnoutph2_i2,e0_h_LVhnoutph2_i2,e0_h_Vhnoutph2_i2,e0_p_houtph2,e0_p_LVhoutph2_i2)
    e0_h_cnoutph2_k1 = fun_69026(e0_greek_rho_Lcnoutph2_k1,e0_h_LVcnoutph2_k1,e0_h_Vcnoutph2_k1,e0_p_coutph2,e0_p_LVcoutph2_k1)
    e0_h_cinnsc_k1 = fun_69026(e0_greek_rho_Lcinnsc_k1,e0_h_LVcinnsc_k1,e0_h_Vcinnsc_k1,e0_p_cinsc,e0_p_LVcinsc_k1)
    e0_h_Lcnoutsh_k1 = fun_69026(e0_greek_rho_Lcnoutsh_k1,e0_h_LVcnoutsh_k1,e0_h_Vcnoutsh_k1,e0_p_coutsh,e0_p_LVcoutsh_k1)
    e0_h_cnoutsc_k1 = fun_69026(e0_greek_rho_Lcnoutsc_k1,e0_h_LVcnoutsc_k1,e0_h_Vcnoutsc_k1,e0_p_coutsc,e0_p_LVcoutsc_k1)
    e0_h_hnoutsc_i1 = fun_69026(e0_greek_rho_Lhnoutsc_i1,e0_h_LVhnoutsc_i1,e0_h_Vhnoutsc_i1,e0_p_houtsc,e0_p_LVhoutsc_i1)
    e0_h_hnoutsc_i2 = fun_69026(e0_greek_rho_Lhnoutsc_i2,e0_h_LVhnoutsc_i2,e0_h_Vhnoutsc_i2,e0_p_houtsc,e0_p_LVhoutsc_i2)
    e0_h_Lhinnph2_i1 = fun_69026(e0_greek_rho_Lhinnph2_i1,e0_h_LVhinnph2_i1,e0_h_Vhinnph2_i1,e0_p_hinph2,e0_p_LVhinph2_i1)
    e0_h_Lhnoutph2_i1 = fun_69026(e0_greek_rho_Lhnoutph2_i1,e0_h_LVhnoutph2_i1,e0_h_Vhnoutph2_i1,e0_p_houtph2,e0_p_LVhoutph2_i1)



# Solve equation system for given x:
    f= [
    0.0-((e0_F_cnoutph2) *(e0_x_coutph2_k1)-(e0_F_cnoutsh) *(e0_x_coutsh_k1)) ,
    0.0-((e0_F_hinnsh) *(e0_y_hinsh_i1)-(e0_F_Vhinnph2) *(e0_y_hinph2_i1)) ,
    0.0-((e0_F_hinnsh) *(e0_y_hinsh_i2)-(e0_F_Vhinnph2) *(e0_y_hinph2_i2)) ,
    0.0-((e0_F_cnoutph2) *(e0_h_cnoutph2)-(e0_F_cnoutsh) *(e0_h_cnoutsh)+e0_Q_sh) ,
    0.0-((e0_F_hinnsh) *(e0_h_hinnsh)-(e0_F_Vhinnph2) *(e0_h_Vhinnph2)-e0_Q_sh) ,
    e0_Q_sh-((e0_k_sh) *((e0_A_sh) *(e0_greek_DeltaT_logsh))) ,
    e0_T_hinsh-e0_T_coutsh-(((e0_T_hinph2-e0_T_coutph2)) *(sympy.exp(((e0_T_hinsh-e0_T_coutsh)-(e0_T_hinph2-e0_T_coutph2))/(e0_greek_DeltaT_logsh)))) ,
    1.0-((e0_y_hinph2_i1+e0_y_hinph2_i2)) ,
    e0_h_hinnsh-(((e0_y_hinsh_i1) *(e0_h_hinnsh_i1)+(e0_y_hinsh_i2) *(e0_h_hinnsh_i2))) ,
    e0_h_Vhinnph2-(((e0_y_hinph2_i1) *(e0_h_hnoutsh_i1)+(e0_y_hinph2_i2) *(e0_h_hnoutsh_i2))) ,
    e0_h_cnoutsh-(((e0_x_coutsh_k1) *(e0_h_Lcnoutsh_k1))) ,
    1.0-((e0_x_coutsh_k1)) ,
    e0_U_hsh-((e0_HU_hnsh) *((e0_h_Vhinnph2+(e0_R) *(e0_T_hinph2)))) ,
    e0_V_hsh-(((e0_HU_hnsh) *((e0_R) *(e0_T_hinph2)))/(e0_p_houtsh)) ,
    e0_HU_hnsh_i1-((e0_y_hinph2_i1) *(e0_HU_hnsh)) ,
    e0_HU_hnsh_i2-((e0_y_hinph2_i2) *(e0_HU_hnsh)) ,
    e0_y_hinph2_i1-((e0_K_hinph2_i1) *(e0_x_hinph2_i1)) ,
    e0_y_hinph2_i2-((e0_K_hinph2_i2) *(e0_x_hinph2_i2)) ,
    e0_y_houtph2_i1-((e0_K_houtph2_i1) *(e0_x_hinsc_i1)) ,
    e0_y_houtph2_i2-((e0_K_houtph2_i2) *(e0_x_hinsc_i2)) ,
    1.0-((e0_y_houtph2_i1+e0_y_houtph2_i2)) ,
    1.0-((e0_x_coutph2_k1)) ,
    1.0-((e0_x_hinph2_i1+e0_x_hinph2_i2)) ,
    1.0-((e0_x_hinsc_i1+e0_x_hinsc_i2)) ,
    e0_greek_gamma_hinph2_i1-(((1.0)/(e0_x_hinph2_i1+(e0_greek_alpha_hinph2wilson_i1) *((1.0-e0_x_hinph2_i1)))) *(sympy.exp(((1.0-e0_x_hinph2_i1)) *(((e0_greek_alpha_hinph2wilson_i1)/(e0_x_hinph2_i1+(e0_greek_alpha_hinph2wilson_i1) *((1.0-e0_x_hinph2_i1)))-((e0_greek_alpha_hinph2wilson_i1+e0_greek_alpha_hinph2wilson_i2)-e0_greek_alpha_hinph2wilson_i1)/((((e0_greek_alpha_hinph2wilson_i1+e0_greek_alpha_hinph2wilson_i2)-e0_greek_alpha_hinph2wilson_i1)) *(e0_x_hinph2_i1)+(1.0-e0_x_hinph2_i1))))))) ,
    e0_greek_gamma_hinph2_i2-(((1.0)/(e0_x_hinph2_i2+(e0_greek_alpha_hinph2wilson_i2) *((1.0-e0_x_hinph2_i2)))) *(sympy.exp(((1.0-e0_x_hinph2_i2)) *(((e0_greek_alpha_hinph2wilson_i2)/(e0_x_hinph2_i2+(e0_greek_alpha_hinph2wilson_i2) *((1.0-e0_x_hinph2_i2)))-((e0_greek_alpha_hinph2wilson_i1+e0_greek_alpha_hinph2wilson_i2)-e0_greek_alpha_hinph2wilson_i2)/((((e0_greek_alpha_hinph2wilson_i1+e0_greek_alpha_hinph2wilson_i2)-e0_greek_alpha_hinph2wilson_i2)) *(e0_x_hinph2_i2)+(1.0-e0_x_hinph2_i2))))))) ,
    e0_greek_alpha_hinph2wilson_i1-((((e0_v_L_i1+e0_v_L_i2)-e0_v_L_i1)/(e0_v_L_i1)) *(sympy.exp((-e0_greek_lambdawilson_i1)/(e0_T_hinph2)))) ,
    e0_greek_alpha_hinph2wilson_i2-((((e0_v_L_i1+e0_v_L_i2)-e0_v_L_i2)/(e0_v_L_i2)) *(sympy.exp((-e0_greek_lambdawilson_i2)/(e0_T_hinph2)))) ,
    e0_K_hinph2_i1-(((e0_p_LVhinph2_i1)/(e0_p_hinph2)) *(e0_greek_gamma_hinph2_i1)) ,
    e0_K_hinph2_i2-(((e0_p_LVhinph2_i2)/(e0_p_hinph2)) *(e0_greek_gamma_hinph2_i2)) ,
    e0_greek_gamma_houtph2_i1-(((1.0)/(e0_x_hinsc_i1+(e0_greek_alpha_houtph2wilson_i1) *((1.0-e0_x_hinsc_i1)))) *(sympy.exp(((1.0-e0_x_hinsc_i1)) *(((e0_greek_alpha_houtph2wilson_i1)/(e0_x_hinsc_i1+(e0_greek_alpha_houtph2wilson_i1) *((1.0-e0_x_hinsc_i1)))-((e0_greek_alpha_houtph2wilson_i1+e0_greek_alpha_houtph2wilson_i2)-e0_greek_alpha_houtph2wilson_i1)/((((e0_greek_alpha_houtph2wilson_i1+e0_greek_alpha_houtph2wilson_i2)-e0_greek_alpha_houtph2wilson_i1)) *(e0_x_hinsc_i1)+(1.0-e0_x_hinsc_i1))))))) ,
    e0_greek_gamma_houtph2_i2-(((1.0)/(e0_x_hinsc_i2+(e0_greek_alpha_houtph2wilson_i2) *((1.0-e0_x_hinsc_i2)))) *(sympy.exp(((1.0-e0_x_hinsc_i2)) *(((e0_greek_alpha_houtph2wilson_i2)/(e0_x_hinsc_i2+(e0_greek_alpha_houtph2wilson_i2) *((1.0-e0_x_hinsc_i2)))-((e0_greek_alpha_houtph2wilson_i1+e0_greek_alpha_houtph2wilson_i2)-e0_greek_alpha_houtph2wilson_i2)/((((e0_greek_alpha_houtph2wilson_i1+e0_greek_alpha_houtph2wilson_i2)-e0_greek_alpha_houtph2wilson_i2)) *(e0_x_hinsc_i2)+(1.0-e0_x_hinsc_i2))))))) ,
    e0_greek_alpha_houtph2wilson_i1-((((e0_v_L_i1+e0_v_L_i2)-e0_v_L_i1)/(e0_v_L_i1)) *(sympy.exp((-e0_greek_lambdawilson_i1)/(e0_T_hinsc)))) ,
    e0_greek_alpha_houtph2wilson_i2-((((e0_v_L_i1+e0_v_L_i2)-e0_v_L_i2)/(e0_v_L_i2)) *(sympy.exp((-e0_greek_lambdawilson_i2)/(e0_T_hinsc)))) ,
    e0_K_houtph2_i1-(((e0_p_LVhoutph2_i1)/(e0_p_houtph2)) *(e0_greek_gamma_houtph2_i1)) ,
    e0_K_houtph2_i2-(((e0_p_LVhoutph2_i2)/(e0_p_houtph2)) *(e0_greek_gamma_houtph2_i2)) ,
    e0_h_Lhinnph2-(((e0_x_hinph2_i1) *(e0_h_Lhinnph2_i1)+(e0_x_hinph2_i2) *(e0_h_Lhinnph2_i2))) ,
    e0_h_hinnsc-(((e0_x_hinsc_i1) *(e0_h_Lhnoutph2_i1)+(e0_x_hinsc_i2) *(e0_h_Lhnoutph2_i2))) ,
    e0_h_Vhnoutph2-(((e0_y_houtph2_i1) *(e0_h_Vhnoutph2_i1)+(e0_y_houtph2_i2) *(e0_h_Vhnoutph2_i2))) ,
    e0_h_cnoutph2-(((e0_x_coutph2_k1) *(e0_h_cnoutph2_k1))) ,
    0.0-((e0_F_Vhinnph2) *(e0_y_hinph2_i1)+(e0_F_Lhinnph2) *(e0_x_hinph2_i1)-(e0_F_Vhnoutph2) *(e0_y_houtph2_i1)-(e0_F_hinnsc) *(e0_x_hinsc_i1)) ,
    0.0-((e0_F_Vhinnph2) *(e0_y_hinph2_i2)+(e0_F_Lhinnph2) *(e0_x_hinph2_i2)-(e0_F_Vhnoutph2) *(e0_y_houtph2_i2)-(e0_F_hinnsc) *(e0_x_hinsc_i2)) ,
    0.0-((e0_F_cnoutsc) *(e0_x_coutsc_k1)-(e0_F_cnoutph2) *(e0_x_coutph2_k1)) ,
    0.0-((e0_F_Lhinnph2) *(e0_h_Lhinnph2)+(e0_F_Vhinnph2) *(e0_h_Vhinnph2)-(e0_F_hinnsc) *(e0_h_hinnsc)-(e0_F_Vhnoutph2) *(e0_h_Vhnoutph2)-e0_Q_ph2) ,
    e0_Q_ph2-((e0_k_ph2) *((e0_A_ph2) *(e0_greek_DeltaT_logph2))) ,
    0.0-((e0_F_cnoutsc) *(e0_h_cnoutsc)-(e0_F_cnoutph2) *(e0_h_cnoutph2)+e0_Q_ph2) ,
    e0_U_hph2-((e0_HU_Lhnph2) *((e0_h_hinnsc+(e0_p_houtph2)/(e0_greek_rho_Lhnoutph2)))+(e0_HU_Vhnph2) *((e0_h_Vhnoutph2+(e0_R) *(e0_T_hinsc)))) ,
    e0_greek_rho_Lhnoutph2-(((e0_x_hinsc_i1) *(e0_greek_rho_Lhnoutph2_i1)+(e0_x_hinsc_i2) *(e0_greek_rho_Lhnoutph2_i2))) ,
    e0_V_Vhph2-(((e0_HU_Vhnph2) *((e0_R) *(e0_T_hinsc)))/(e0_p_houtph2)) ,
    (e0_greek_rho_Lhnoutph2) *(e0_V_Lhph2)-(e0_HU_Lhnph2) ,
    e0_T_hinph2-e0_T_coutph2-(((e0_T_hinsc-e0_T_coutsc)) *(sympy.exp(((e0_T_hinph2-e0_T_coutph2)-(e0_T_hinsc-e0_T_coutsc))/(e0_greek_DeltaT_logph2)))) ,
    e0_HU_hnph2_i1-((e0_x_hinsc_i1) *(e0_HU_Lhnph2)+(e0_y_houtph2_i1) *(e0_HU_Vhnph2)) ,
    e0_HU_hnph2_i2-((e0_x_hinsc_i2) *(e0_HU_Lhnph2)+(e0_y_houtph2_i2) *(e0_HU_Vhnph2)) ,
    1.0-((e0_x_houtsc_i1+e0_x_houtsc_i2)) ,
    0.0-((e0_F_hinnsc) *(e0_x_hinsc_i1)-(e0_F_hnoutsc) *(e0_x_houtsc_i1)) ,
    0.0-((e0_F_hinnsc) *(e0_x_hinsc_i2)-(e0_F_hnoutsc) *(e0_x_houtsc_i2)) ,
    e0_h_hnoutsc-(((e0_x_houtsc_i1) *(e0_h_hnoutsc_i1)+(e0_x_houtsc_i2) *(e0_h_hnoutsc_i2))) ,
    0.0-((e0_F_hinnsc) *(e0_h_hinnsc)-(e0_F_hnoutsc) *(e0_h_hnoutsc)-e0_Q_sc) ,
    e0_T_hinsc-e0_T_coutsc-(((e0_T_houtsc-e0_T_cinsc)) *(sympy.exp(((e0_T_hinsc-e0_T_coutsc)-(e0_T_houtsc-e0_T_cinsc))/(e0_greek_DeltaT_logsc)))) ,
    e0_Q_sc-((e0_k_sc) *((e0_A_sc) *(e0_greek_DeltaT_logsc))) ,
    e0_h_cinnsc-(((e0_x_cinsc_k1) *(e0_h_cinnsc_k1))) ,
    e0_h_cnoutsc-(((e0_x_coutsc_k1) *(e0_h_cnoutsc_k1))) ,
    1.0-((e0_x_coutsc_k1)) ,
    0.0-((e0_F_cinnsc) *(e0_x_cinsc_k1)-(e0_F_cnoutsc) *(e0_x_coutsc_k1)) ,
    0.0-((e0_F_cinnsc) *(e0_h_cinnsc)-(e0_F_cnoutsc) *(e0_h_cnoutsc)+e0_Q_sc) ,
    e0_greek_rho_Lhnoutsc-(((e0_x_houtsc_i1) *(e0_greek_rho_Lhnoutsc_i1)+(e0_x_houtsc_i2) *(e0_greek_rho_Lhnoutsc_i2))) ,
    (e0_greek_rho_Lhnoutsc) *(e0_V_hsc)-(e0_HU_hnsc) ,
    e0_U_hsc-((e0_HU_hnsc) *((e0_h_hnoutsc+(e0_p_houtsc)/(e0_greek_rho_Lhnoutsc)))) ,
    e0_HU_hnsc_i1-((e0_x_houtsc_i1) *(e0_HU_hnsc)) ,
    e0_HU_hnsc_i2-((e0_x_houtsc_i2) *(e0_HU_hnsc)) ,
    e0_A-(e0_A_V+e0_A_sc) ,
    e0_A_aux-(1.0-((2.0) *(e0_H_Ltotal))/(e0_D_i)) ,
    e0_V_Ltotal-(((((((e0_D_i)/(2.0)))**(2.0)) *(((e0_greek_pi)/(2.0)-e0_A_aux-(((e0_A_aux))**(3.0))/(6.0)-((3.0) *(((e0_A_aux))**(5.0)))/(40.0)))-(((e0_D_i)/(2.0)-e0_H_Ltotal)) *((((e0_D_i) *(e0_H_Ltotal)-((e0_H_Ltotal))**(2.0)))**(0.5)))) *(e0_L)) ,
    e0_A-((e0_N_ptotal) *((e0_greek_pi) *((e0_d_o) *(e0_L)))) ,
    e0_V-((((e0_greek_pi) *(((e0_D_i))**(2.0)))/(4.0)) *(e0_L)) ,
    e0_V-(e0_V_Ltotal+e0_V_Vtotal) ,
    e0_A_V-(e0_A_sh+e0_A_ph2) ,
    e0_V_Vh-(e0_V_hsh+e0_V_Vhph2) ,
    e0_V_shtotal-(((((((e0_D_i)/(2.0)))**(2.0)) *(((e0_greek_pi)/(2.0)-e0_A_auxsh-(((e0_A_auxsh))**(3.0))/(6.0)-((3.0) *(((e0_A_auxsh))**(5.0)))/(40.0)))-(((e0_D_i)/(2.0)-e0_H_shtotal)) *((((e0_D_i) *(e0_H_shtotal)-((e0_H_shtotal))**(2.0)))**(0.5)))) *(e0_L)) ,
    e0_A_auxsh-(1.0-((2.0) *(e0_H_shtotal))/(e0_D_i)) ,
    e0_N_psh-((e0_AcNP) *(sympy.sin((e0_BcNP) *(((e0_H_shtotal-(e0_D_i)/(2.0))/(e0_H_dim)))))+(e0_N_ptotal)/(2.0)+(e0_CcNP) *(sympy.exp(((-e0_H_shtotal)/(e0_H_dim)) *(1000.0)))) ,
    e0_A_sh-((e0_N_psh) *((e0_greek_pi) *((e0_d_o) *(e0_L)))) ,
    e0_V_psh-((e0_N_psh) *((((e0_greek_pi) *(((e0_d_o))**(2.0)))/(4.0)) *(e0_L))) ,
    e0_V_shtotal-(e0_V_hsh+e0_V_psh) ,
    e0_V_sctotal-(((((((e0_D_i)/(2.0)))**(2.0)) *(((e0_greek_pi)/(2.0)-e0_A_auxsc-(((e0_A_auxsc))**(3.0))/(6.0)-((3.0) *(((e0_A_auxsc))**(5.0)))/(40.0)))-(((e0_D_i)/(2.0)-e0_H_sctotal)) *((((e0_D_i) *(e0_H_sctotal)-((e0_H_sctotal))**(2.0)))**(0.5)))) *(e0_L)) ,
    e0_A_auxsc-(1.0-((2.0) *(e0_H_sctotal))/(e0_D_i)) ,
    e0_N_psc-((e0_AcNP) *(sympy.sin((e0_BcNP) *(((e0_H_sctotal-(e0_D_i)/(2.0))/(e0_H_dim)))))+(e0_N_ptotal)/(2.0)+(e0_CcNP) *(sympy.exp(((-e0_H_sctotal)/(e0_H_dim)) *(1000.0)))) ,
    e0_A_sc-((e0_N_psc) *((e0_greek_pi) *((e0_d_o) *(e0_L)))) ,
    e0_V_psc-((e0_N_psc) *((((e0_greek_pi) *(((e0_d_o))**(2.0)))/(4.0)) *(e0_L))) ,
    e0_V_sctotal-(e0_V_hsc+e0_V_psc) ,
    e0_V_Ltotal-(e0_V_sctotal+e0_V_Lph2total) ,
    e0_V_Lph2total-(e0_V_Lhph2+e0_V_Lpph2) ,
    e0_V_Vtotal-(e0_V_shtotal+e0_V_Vph2total) ,
    e0_N_Lpph2-((e0_AcNP) *(sympy.sin((e0_BcNP) *(((e0_H_Ltotal-(e0_D_i)/(2.0))/(e0_H_dim)))))+(e0_N_ptotal)/(2.0)+(e0_CcNP) *(sympy.exp(((-e0_H_Ltotal)/(e0_H_dim)) *(1000.0)))-e0_N_psc) ,
    e0_V_L-(e0_V_hsc+e0_V_Lhph2) ,
    e0_V_Lpph2-((e0_N_Lpph2) *((((e0_greek_pi) *(((e0_d_o))**(2.0)))/(4.0)) *(e0_L))) ,
    e0_V_Vpph2-(((e0_N_ptotal-e0_N_Lpph2-e0_N_psc-e0_N_psh)) *((((e0_greek_pi) *(((e0_d_o))**(2.0)))/(4.0)) *(e0_L))) ,
    e0_V_Vph2total-(e0_V_Vhph2+e0_V_Vpph2) 

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

    x = numpy.empty(98)
    xInitial = numpy.empty((98), dtype = object) 
    parameter = numpy.empty(103)
    dict_variables = {}

    # Iteration variable initializatio
    x[0] = 16.8322 	# e0_greek_DeltaT_logsh
    x[1] = 29.9984 	# e0_Q_sh
    x[2] = 70.0388 	# e0_HU_hnph2_i1
    x[3] = 99.1319 	# e0_HU_hnph2_i2
    x[4] = 4.7998 	# e0_K_hinph2_i1
    x[5] = 0.64321 	# e0_K_hinph2_i2
    x[6] = 1.5477 	# e0_K_houtph2_i1
    x[7] = 0.61622 	# e0_K_houtph2_i2
    x[8] = 14087.4115 	# e0_Q_ph2
    x[9] = 299.4305 	# e0_T_coutsc
    x[10] = 344.2077 	# e0_T_hinsc
    x[11] = -4.69319001878E7 	# e0_U_hph2
    x[12] = 335.3932 	# e0_T_coutph2
    x[13] = 0.00435 	# e0_V_Lhph2
    x[14] = 0.061682 	# e0_V_Vhph2
    x[15] = -280753.7226 	# e0_h_Lhinnph2
    x[16] = -277834.0621 	# e0_h_hinnsc
    x[17] = 335.4681 	# e0_T_coutsh
    x[18] = -234845.2032 	# e0_h_Vhnoutph2
    x[19] = -285703.2521 	# e0_h_cnoutsc
    x[20] = 351.4034 	# e0_T_hinph2
    x[21] = 1.0 	# e0_x_coutsc_k1
    x[22] = 0.085837 	# e0_x_hinph2_i1
    x[23] = 0.91416 	# e0_x_hinph2_i2
    x[24] = 0.412 	# e0_x_hinsc_i1
    x[25] = 0.588 	# e0_x_hinsc_i2
    x[26] = 0.63766 	# e0_y_houtph2_i1
    x[27] = 0.36234 	# e0_y_houtph2_i2
    x[28] = -27313.6894 	# e0_U_hsh
    x[29] = 0.0048826 	# e0_V_hsh
    x[30] = 38.0315 	# e0_greek_DeltaT_logsc
    x[31] = 38899.4223 	# e0_greek_rho_Lhnoutsc
    x[32] = 0.25859 	# e0_A_sc
    x[33] = 0.34 	# e0_F_hnoutsc
    x[34] = 147.1542 	# e0_HU_hnsc
    x[35] = 60.6275 	# e0_HU_hnsc_i1
    x[36] = 86.5267 	# e0_HU_hnsc_i2
    x[37] = 491.7208 	# e0_Q_sc
    x[38] = 330.15 	# e0_T_houtsc
    x[39] = -4.10970105544E7 	# e0_U_hsc
    x[40] = 0.0037829 	# e0_V_hsc
    x[41] = 0.035644 	# e0_A_sh
    x[42] = -282976.9482 	# e0_h_cnoutph2
    x[43] = -285798.4136 	# e0_h_cinnsc
    x[44] = -279280.2996 	# e0_h_hnoutsc
    x[45] = -282971.1427 	# e0_h_cnoutsh
    x[46] = -236312.2684 	# e0_h_hinnsh
    x[47] = 0.412 	# e0_x_houtsc_i1
    x[48] = 0.588 	# e0_x_houtsc_i2
    x[49] = 10.3673 	# e0_A
    x[50] = 10.1087 	# e0_A_V
    x[51] = 0.66443 	# e0_A_aux
    x[52] = 0.87594 	# e0_A_auxsc
    x[53] = 0.9365 	# e0_A_auxsh
    x[54] = 0.018485 	# e0_H_sctotal
    x[55] = 0.0094614 	# e0_H_shtotal
    x[56] = 5.9538 	# e0_N_Lpph2
    x[57] = 1.6462 	# e0_N_psc
    x[58] = 0.22692 	# e0_N_psh
    x[59] = 0.13949 	# e0_V
    x[60] = 0.0081329 	# e0_V_L
    x[61] = 0.0058451 	# e0_V_Lpph2
    x[62] = 0.010195 	# e0_V_Lph2total
    x[63] = 0.015594 	# e0_V_Ltotal
    x[64] = 0.066565 	# e0_V_Vh
    x[65] = 0.057111 	# e0_V_Vpph2
    x[66] = 0.11879 	# e0_V_Vph2total
    x[67] = 0.1239 	# e0_V_Vtotal
    x[68] = 0.0016162 	# e0_V_psc
    x[69] = 2.2278E-4 	# e0_V_psh
    x[70] = 0.0053991 	# e0_V_sctotal
    x[71] = 0.0051054 	# e0_V_shtotal
    x[72] = -236400.4989 	# e0_h_Vhinnph2
    x[73] = 5.2 	# e0_F_cnoutph2
    x[74] = 1.0 	# e0_x_coutph2_k1
    x[75] = 1.0 	# e0_x_coutsh_k1
    x[76] = 0.412 	# e0_y_hinph2_i1
    x[77] = 0.588 	# e0_y_hinph2_i2
    x[78] = 5.2 	# e0_F_cnoutsh
    x[79] = 0.34 	# e0_F_Vhinnph2
    x[80] = 0.11699 	# e0_HU_hnsh
    x[81] = 27.9706 	# e0_greek_DeltaT_logph2
    x[82] = 0.2345 	# e0_greek_alpha_hinph2wilson_i1
    x[83] = 0.76804 	# e0_greek_alpha_hinph2wilson_i2
    x[84] = 0.23317 	# e0_greek_alpha_houtph2wilson_i1
    x[85] = 0.048198 	# e0_HU_hnsh_i1
    x[86] = 0.74523 	# e0_greek_alpha_houtph2wilson_i2
    x[87] = 3.3234 	# e0_greek_gamma_hinph2_i1
    x[88] = 1.0205 	# e0_greek_gamma_hinph2_i2
    x[89] = 1.4327 	# e0_greek_gamma_houtph2_i1
    x[90] = 1.3218 	# e0_greek_gamma_houtph2_i2
    x[91] = 0.068788 	# e0_HU_hnsh_i2
    x[92] = 38542.9381 	# e0_greek_rho_Lhnoutph2
    x[93] = 10.073 	# e0_A_ph2
    x[94] = 0.34 	# e0_F_hinnsc
    x[95] = 5.2 	# e0_F_cnoutsc
    x[96] = 167.6619 	# e0_HU_Lhnph2
    x[97] = 1.5088 	# e0_HU_Vhnph2


    # Constant parameter setting:
    parameter[0] = 353.15 	# e0_T_hinsh
    parameter[1] = 50.0 	# e0_k_ph2
    parameter[2] = 100000.0 	# e0_p_coutph2
    parameter[3] = 70000.0 	# e0_p_hinph2
    parameter[4] = 70000.0 	# e0_p_houtph2
    parameter[5] = 5.869E-5 	# e0_v_L_i1
    parameter[6] = 1.807E-5 	# e0_v_L_i2
    parameter[7] = 5.2 	# e0_F_cinnsc
    parameter[8] = 298.15 	# e0_T_cinsc
    parameter[9] = 50.0 	# e0_k_sc
    parameter[10] = 100000.0 	# e0_p_cinsc
    parameter[11] = 100000.0 	# e0_p_coutsc
    parameter[12] = 70000.0 	# e0_p_houtsc
    parameter[13] = 1.0 	# e0_x_cinsc_k1
    parameter[14] = 0.05 	# e0_H_Ltotal
    parameter[15] = 1.0 	# e0_H_dim
    parameter[16] = 50.0 	# e0_k_sh
    parameter[17] = 100000.0 	# e0_p_coutsh
    parameter[18] = 70000.0 	# e0_p_houtsh
    parameter[19] = 0.412 	# e0_y_hinsh_i1
    parameter[20] = 0.588 	# e0_y_hinsh_i2
    parameter[21] = 0.34 	# e0_F_hinnsh
    parameter[22] = 95.68 	# e0_greek_lambdawilson_i1
    parameter[23] = 506.7 	# e0_greek_lambdawilson_i2
    parameter[24] = 0.0 	# e0_F_Lhinnph2
    parameter[25] = 0.0 	# e0_F_Vhnoutph2
    parameter[26] = 3.14159265359 	# e0_greek_pi
    parameter[27] = 38.7993 	# e0_AcNP
    parameter[28] = 9.008 	# e0_A_cpd100_i1
    parameter[29] = -13851.0 	# e0_A_Lnrhod100_i2
    parameter[30] = 32.22 	# e0_A_cpd100_i2
    parameter[31] = -13851.0 	# e0_A_Lnrhod100_k1
    parameter[32] = 32.22 	# e0_A_cpd100_k1
    parameter[33] = 73.304 	# e0_A_LVpd101_i1
    parameter[34] = 73.649 	# e0_A_LVpd101_i2
    parameter[35] = 73.649 	# e0_A_LVpd101_k1
    parameter[36] = 1628.8 	# e0_A_Lnrhod105_i1
    parameter[37] = 65831.0 	# e0_A_LVhd106_i1
    parameter[38] = 56600.0 	# e0_A_LVhd106_i2
    parameter[39] = 56600.0 	# e0_A_LVhd106_k1
    parameter[40] = 7.2093 	# e0_BcNP
    parameter[41] = 0.2139 	# e0_B_cpd100_i1
    parameter[42] = 640.38 	# e0_B_Lnrhod100_i2
    parameter[43] = 0.0019225 	# e0_B_cpd100_i2
    parameter[44] = 640.38 	# e0_B_Lnrhod100_k1
    parameter[45] = 0.0019225 	# e0_B_cpd100_k1
    parameter[46] = -7122.3 	# e0_B_LVpd101_i1
    parameter[47] = -7258.2 	# e0_B_LVpd101_i2
    parameter[48] = -7258.2 	# e0_B_LVpd101_k1
    parameter[49] = 0.27469 	# e0_B_Lnrhod105_i1
    parameter[50] = 1.1905 	# e0_B_LVhd106_i1
    parameter[51] = 0.61204 	# e0_B_LVhd106_i2
    parameter[52] = 0.61204 	# e0_B_LVhd106_k1
    parameter[53] = 1.1124 	# e0_CcNP
    parameter[54] = -8.3846E-5 	# e0_C_cpd100_i1
    parameter[55] = -1.9124 	# e0_C_Lnrhod100_i2
    parameter[56] = 1.0548E-5 	# e0_C_cpd100_i2
    parameter[57] = -1.9124 	# e0_C_Lnrhod100_k1
    parameter[58] = 1.0548E-5 	# e0_C_cpd100_k1
    parameter[59] = -7.1424 	# e0_C_LVpd101_i1
    parameter[60] = -7.3037 	# e0_C_LVpd101_i2
    parameter[61] = -7.3037 	# e0_C_LVpd101_k1
    parameter[62] = 514.0 	# e0_C_Lnrhod105_i1
    parameter[63] = -1.7666 	# e0_C_LVhd106_i1
    parameter[64] = -0.6257 	# e0_C_LVhd106_i2
    parameter[65] = -0.6257 	# e0_C_LVhd106_k1
    parameter[66] = 0.298 	# e0_D_i
    parameter[67] = 1.3723E-9 	# e0_D_cpd100_i1
    parameter[68] = 0.0018211 	# e0_D_Lnrhod100_i2
    parameter[69] = -3.594E-9 	# e0_D_cpd100_i2
    parameter[70] = 0.0018211 	# e0_D_Lnrhod100_k1
    parameter[71] = -3.594E-9 	# e0_D_cpd100_k1
    parameter[72] = 2.8853E-6 	# e0_D_LVpd101_i1
    parameter[73] = 4.1653E-6 	# e0_D_LVpd101_i2
    parameter[74] = 4.1653E-6 	# e0_D_LVpd101_k1
    parameter[75] = 0.23178 	# e0_D_Lnrhod105_i1
    parameter[76] = 1.0012 	# e0_D_LVhd106_i1
    parameter[77] = 0.3988 	# e0_D_LVhd106_i2
    parameter[78] = 0.3988 	# e0_D_LVhd106_k1
    parameter[79] = 0.0 	# e0_E_cpd100_i1
    parameter[80] = 0.0 	# e0_E_Lnrhod100_i2
    parameter[81] = 0.0 	# e0_E_cpd100_i2
    parameter[82] = 0.0 	# e0_E_Lnrhod100_k1
    parameter[83] = 0.0 	# e0_E_cpd100_k1
    parameter[84] = 2.0 	# e0_E_LVpd101_i1
    parameter[85] = 2.0 	# e0_E_LVpd101_i2
    parameter[86] = 2.0 	# e0_E_LVpd101_k1
    parameter[87] = 0.0 	# e0_E_LVhd106_i1
    parameter[88] = 0.0 	# e0_E_LVhd106_i2
    parameter[89] = 0.0 	# e0_E_LVhd106_k1
    parameter[90] = 2.0 	# e0_L
    parameter[91] = 66.0 	# e0_N_ptotal
    parameter[92] = 8.314 	# e0_R
    parameter[93] = 514.0 	# e0_T_crit_i1
    parameter[94] = 298.15 	# e0_T_ho_i1
    parameter[95] = 647.096 	# e0_T_crit_i2
    parameter[96] = 298.15 	# e0_T_ho_i2
    parameter[97] = 647.096 	# e0_T_crit_k1
    parameter[98] = 298.15 	# e0_T_ho_k1
    parameter[99] = 0.025 	# e0_d_o
    parameter[100] = -234950.0 	# e0_h_o_i1
    parameter[101] = -241818.0 	# e0_h_o_i2
    parameter[102] = -241818.0 	# e0_h_o_k1


    # Get symbolic quantities:
    xSymbolic = sympy.symbols('e0_greek_DeltaT_logsh e0_Q_sh e0_HU_hnph2_i1 e0_HU_hnph2_i2 e0_K_hinph2_i1 e0_K_hinph2_i2 e0_K_houtph2_i1 e0_K_houtph2_i2 e0_Q_ph2 e0_T_coutsc e0_T_hinsc e0_U_hph2 e0_T_coutph2 e0_V_Lhph2 e0_V_Vhph2 e0_h_Lhinnph2 e0_h_hinnsc e0_T_coutsh e0_h_Vhnoutph2 e0_h_cnoutsc e0_T_hinph2 e0_x_coutsc_k1 e0_x_hinph2_i1 e0_x_hinph2_i2 e0_x_hinsc_i1 e0_x_hinsc_i2 e0_y_houtph2_i1 e0_y_houtph2_i2 e0_U_hsh e0_V_hsh e0_greek_DeltaT_logsc e0_greek_rho_Lhnoutsc e0_A_sc e0_F_hnoutsc e0_HU_hnsc e0_HU_hnsc_i1 e0_HU_hnsc_i2 e0_Q_sc e0_T_houtsc e0_U_hsc e0_V_hsc e0_A_sh e0_h_cnoutph2 e0_h_cinnsc e0_h_hnoutsc e0_h_cnoutsh e0_h_hinnsh e0_x_houtsc_i1 e0_x_houtsc_i2 e0_A e0_A_V e0_A_aux e0_A_auxsc e0_A_auxsh e0_H_sctotal e0_H_shtotal e0_N_Lpph2 e0_N_psc e0_N_psh e0_V e0_V_L e0_V_Lpph2 e0_V_Lph2total e0_V_Ltotal e0_V_Vh e0_V_Vpph2 e0_V_Vph2total e0_V_Vtotal e0_V_psc e0_V_psh e0_V_sctotal e0_V_shtotal e0_h_Vhinnph2 e0_F_cnoutph2 e0_x_coutph2_k1 e0_x_coutsh_k1 e0_y_hinph2_i1 e0_y_hinph2_i2 e0_F_cnoutsh e0_F_Vhinnph2 e0_HU_hnsh e0_greek_DeltaT_logph2 e0_greek_alpha_hinph2wilson_i1 e0_greek_alpha_hinph2wilson_i2 e0_greek_alpha_houtph2wilson_i1 e0_HU_hnsh_i1 e0_greek_alpha_houtph2wilson_i2 e0_greek_gamma_hinph2_i1 e0_greek_gamma_hinph2_i2 e0_greek_gamma_houtph2_i1 e0_greek_gamma_houtph2_i2 e0_HU_hnsh_i2 e0_greek_rho_Lhnoutph2 e0_A_ph2 e0_F_hinnsc e0_F_cnoutsc e0_HU_Lhnph2 e0_HU_Vhnph2 ')
    fSymbolic = getSymbolicFunctions(xSymbolic, parameter)

    # Intial variable bounds:
    xInitial[0] = mpmath.mpi(0.0, 40.0)  	# e0_greek_DeltaT_logsh
    xInitial[1] = mpmath.mpi(0.0, 20000.0)  	# e0_Q_sh
    xInitial[2] = mpmath.mpi(0.0, 1000.0)  	# e0_HU_hnph2_i1
    xInitial[3] = mpmath.mpi(0.0, 1000.0)  	# e0_HU_hnph2_i2
    xInitial[4] = mpmath.mpi(0.0, 3.0)  	# e0_K_hinph2_i1
    xInitial[5] = mpmath.mpi(0.0, 3.0)  	# e0_K_hinph2_i2
    xInitial[6] = mpmath.mpi(0.0, 3.0)  	# e0_K_houtph2_i1
    xInitial[7] = mpmath.mpi(0.0, 3.0)  	# e0_K_houtph2_i2
    xInitial[8] = mpmath.mpi(0.0, 20000.0)  	# e0_Q_ph2
    xInitial[9] = mpmath.mpi(250.0, 380.0)  	# e0_T_coutsc
    xInitial[10] = mpmath.mpi(250.0, 380.0)  	# e0_T_hinsc
    xInitial[11] = mpmath.mpi(-5.0E7, -4.0E7)  	# e0_U_hph2
    xInitial[12] = mpmath.mpi(250.0, 380.0)  	# e0_T_coutph2
    xInitial[13] = mpmath.mpi(0.0, 0.5)  	# e0_V_Lhph2
    xInitial[14] = mpmath.mpi(0.0, 0.5)  	# e0_V_Vhph2
    xInitial[15] = mpmath.mpi(-300000.0, -200000.0)  	# e0_h_Lhinnph2
    xInitial[16] = mpmath.mpi(-300000.0, -200000.0)  	# e0_h_hinnsc
    xInitial[17] = mpmath.mpi(250.0, 380.0)  	# e0_T_coutsh
    xInitial[18] = mpmath.mpi(-300000.0, -200000.0)  	# e0_h_Vhnoutph2
    xInitial[19] = mpmath.mpi(-300000.0, -200000.0)  	# e0_h_cnoutsc
    xInitial[20] = mpmath.mpi(250.0, 380.0)  	# e0_T_hinph2
    xInitial[21] = mpmath.mpi(0.0, 1.0)  	# e0_x_coutsc_k1
    xInitial[22] = mpmath.mpi(0.0, 1.0)  	# e0_x_hinph2_i1
    xInitial[23] = mpmath.mpi(0.0, 1.0)  	# e0_x_hinph2_i2
    xInitial[24] = mpmath.mpi(0.0, 1.0)  	# e0_x_hinsc_i1
    xInitial[25] = mpmath.mpi(0.0, 1.0)  	# e0_x_hinsc_i2
    xInitial[26] = mpmath.mpi(0.0, 1.0)  	# e0_y_houtph2_i1
    xInitial[27] = mpmath.mpi(0.0, 1.0)  	# e0_y_houtph2_i2
    xInitial[28] = mpmath.mpi(-5.0E7, -4.0E7)  	# e0_U_hsh
    xInitial[29] = mpmath.mpi(0.0, 0.5)  	# e0_V_hsh
    xInitial[30] = mpmath.mpi(0.0, 40.0)  	# e0_greek_DeltaT_logsc
    xInitial[31] = mpmath.mpi(30000.0, 50000.0)  	# e0_greek_rho_Lhnoutsc
    xInitial[32] = mpmath.mpi(0.0, 100.0)  	# e0_A_sc
    xInitial[33] = mpmath.mpi(0.0, 100.0)  	# e0_F_hnoutsc
    xInitial[34] = mpmath.mpi(0.0, 1000.0)  	# e0_HU_hnsc
    xInitial[35] = mpmath.mpi(0.0, 1000.0)  	# e0_HU_hnsc_i1
    xInitial[36] = mpmath.mpi(0.0, 1000.0)  	# e0_HU_hnsc_i2
    xInitial[37] = mpmath.mpi(0.0, 20000.0)  	# e0_Q_sc
    xInitial[38] = mpmath.mpi(250.0, 380.0)  	# e0_T_houtsc
    xInitial[39] = mpmath.mpi(-5.0E7, -4.0E7)  	# e0_U_hsc
    xInitial[40] = mpmath.mpi(0.0, 0.5)  	# e0_V_hsc
    xInitial[41] = mpmath.mpi(0.0, 100.0)  	# e0_A_sh
    xInitial[42] = mpmath.mpi(-300000.0, -200000.0)  	# e0_h_cnoutph2
    xInitial[43] = mpmath.mpi(-300000.0, -200000.0)  	# e0_h_cinnsc
    xInitial[44] = mpmath.mpi(-300000.0, -200000.0)  	# e0_h_hnoutsc
    xInitial[45] = mpmath.mpi(-300000.0, -200000.0)  	# e0_h_cnoutsh
    xInitial[46] = mpmath.mpi(-300000.0, -200000.0)  	# e0_h_hinnsh
    xInitial[47] = mpmath.mpi(0.0, 1.0)  	# e0_x_houtsc_i1
    xInitial[48] = mpmath.mpi(0.0, 1.0)  	# e0_x_houtsc_i2
    xInitial[49] = mpmath.mpi(0.0, 100.0)  	# e0_A
    xInitial[50] = mpmath.mpi(0.0, 100.0)  	# e0_A_V
    xInitial[51] = mpmath.mpi(0.0, 100.0)  	# e0_A_aux
    xInitial[52] = mpmath.mpi(0.0, 100.0)  	# e0_A_auxsc
    xInitial[53] = mpmath.mpi(0.0, 100.0)  	# e0_A_auxsh
    xInitial[54] = mpmath.mpi(0.0, 0.1)  	# e0_H_sctotal
    xInitial[55] = mpmath.mpi(0.0, 0.1)  	# e0_H_shtotal
    xInitial[56] = mpmath.mpi(0.0, 10.0)  	# e0_N_Lpph2
    xInitial[57] = mpmath.mpi(0.0, 10.0)  	# e0_N_psc
    xInitial[58] = mpmath.mpi(0.0, 10.0)  	# e0_N_psh
    xInitial[59] = mpmath.mpi(0.0, 0.5)  	# e0_V
    xInitial[60] = mpmath.mpi(0.0, 0.5)  	# e0_V_L
    xInitial[61] = mpmath.mpi(0.0, 0.5)  	# e0_V_Lpph2
    xInitial[62] = mpmath.mpi(0.0, 0.5)  	# e0_V_Lph2total
    xInitial[63] = mpmath.mpi(0.0, 0.5)  	# e0_V_Ltotal
    xInitial[64] = mpmath.mpi(0.0, 0.5)  	# e0_V_Vh
    xInitial[65] = mpmath.mpi(0.0, 0.5)  	# e0_V_Vpph2
    xInitial[66] = mpmath.mpi(0.0, 0.5)  	# e0_V_Vph2total
    xInitial[67] = mpmath.mpi(0.0, 0.5)  	# e0_V_Vtotal
    xInitial[68] = mpmath.mpi(0.0, 0.5)  	# e0_V_psc
    xInitial[69] = mpmath.mpi(0.0, 0.5)  	# e0_V_psh
    xInitial[70] = mpmath.mpi(0.0, 0.5)  	# e0_V_sctotal
    xInitial[71] = mpmath.mpi(0.0, 0.5)  	# e0_V_shtotal
    xInitial[72] = mpmath.mpi(-300000.0, -200000.0)  	# e0_h_Vhinnph2
    xInitial[73] = mpmath.mpi(0.0, 100.0)  	# e0_F_cnoutph2
    xInitial[74] = mpmath.mpi(0.0, 1.0)  	# e0_x_coutph2_k1
    xInitial[75] = mpmath.mpi(0.0, 1.0)  	# e0_x_coutsh_k1
    xInitial[76] = mpmath.mpi(0.0, 1.0)  	# e0_y_hinph2_i1
    xInitial[77] = mpmath.mpi(0.0, 1.0)  	# e0_y_hinph2_i2
    xInitial[78] = mpmath.mpi(0.0, 100.0)  	# e0_F_cnoutsh
    xInitial[79] = mpmath.mpi(0.0, 100.0)  	# e0_F_Vhinnph2
    xInitial[80] = mpmath.mpi(0.0, 1000.0)  	# e0_HU_hnsh
    xInitial[81] = mpmath.mpi(0.0, 40.0)  	# e0_greek_DeltaT_logph2
    xInitial[82] = mpmath.mpi(0.0, 1.0)  	# e0_greek_alpha_hinph2wilson_i1
    xInitial[83] = mpmath.mpi(0.0, 1.0)  	# e0_greek_alpha_hinph2wilson_i2
    xInitial[84] = mpmath.mpi(0.0, 1.0)  	# e0_greek_alpha_houtph2wilson_i1
    xInitial[85] = mpmath.mpi(0.0, 1000.0)  	# e0_HU_hnsh_i1
    xInitial[86] = mpmath.mpi(0.0, 1.0)  	# e0_greek_alpha_houtph2wilson_i2
    xInitial[87] = mpmath.mpi(0.0, 5.0)  	# e0_greek_gamma_hinph2_i1
    xInitial[88] = mpmath.mpi(0.0, 5.0)  	# e0_greek_gamma_hinph2_i2
    xInitial[89] = mpmath.mpi(0.0, 5.0)  	# e0_greek_gamma_houtph2_i1
    xInitial[90] = mpmath.mpi(0.0, 5.0)  	# e0_greek_gamma_houtph2_i2
    xInitial[91] = mpmath.mpi(0.0, 1000.0)  	# e0_HU_hnsh_i2
    xInitial[92] = mpmath.mpi(30000.0, 50000.0)  	# e0_greek_rho_Lhnoutph2
    xInitial[93] = mpmath.mpi(0.0, 100.0)  	# e0_A_ph2
    xInitial[94] = mpmath.mpi(0.0, 100.0)  	# e0_F_hinnsc
    xInitial[95] = mpmath.mpi(0.0, 100.0)  	# e0_F_cnoutsc
    xInitial[96] = mpmath.mpi(0.0, 1000.0)  	# e0_HU_Lhnph2
    xInitial[97] = mpmath.mpi(0.0, 1000.0)  	# e0_HU_Vhnph2


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
