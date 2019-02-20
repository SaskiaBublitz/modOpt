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
    dict_options = {"fileName": "Deiso_r100",
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
def fun_75392(std_X,std_A,std_B,std_C,std_D): # fun_75392__DIPPR102
	return ((std_A) *(((std_X))**(std_B)))/(1.0+(std_C)/(std_X)+(std_D)/(((std_X))**(2.0)))
def fun_69026(std_greek_rho_Ln,std_h_LVn,std_h_Vn,std_p,std_p_LV): # fun_69026__enthalpy_molar_liquid_incompressible
	return std_h_Vn-std_h_LVn+((std_p-std_p_LV))/(std_greek_rho_Ln)
def fun_93415(std_T,std_A,std_B,std_C,std_D,std_E,std_F,std_G,std_T_ho,std_h_o): # fun_93415__DIPPR127_IG
	return std_h_o+(std_A) *((std_T-std_T_ho))+(std_B)/(sympy.exp((std_C)/(std_T))-1.0)-(std_B)/(sympy.exp((std_C)/(std_T_ho))-1.0)+(std_D)/(sympy.exp((std_E)/(std_T))-1.0)-(std_D)/(sympy.exp((std_E)/(std_T_ho))-1.0)+(std_F)/(sympy.exp((std_G)/(std_T))-1.0)-(std_F)/(sympy.exp((std_G)/(std_T_ho))-1.0)
def fun_75386(std_X,std_A,std_B,std_C,std_D,std_E): # fun_75386__DIPPR101
	return sympy.exp(std_A+(std_B)/(std_X)+(std_C) *(sympy.log(std_X))+(std_D) *(((std_X))**(std_E)))
def fun_75389(std_X,std_A,std_B,std_C,std_D): # fun_75389__DIPPR105
	return (std_A)/(((std_B))**(1.0+((1.0-(std_X)/(std_C)))**(std_D)))
def fun_94396(std_F_n,std_M): # fun_94396__molarToMassStream
	return (std_F_n) *(std_M)
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
    e0e1_x_tr1_i1 = x[0]
    e0e1_x_tr2_i1 = x[1]
    e0e1_x_tr3_i1 = x[2]
    e0e1_x_tr4_i1 = x[3]
    e0e1_x_tr5_i1 = x[4]
    e0e1_x_tr6_i1 = x[5]
    e0e1_x_tr7_i1 = x[6]
    e0e1_x_tr8_i1 = x[7]
    e0e1_x_tr9_i1 = x[8]
    e0e1_x_tr10_i1 = x[9]
    e0e1_x_tr11_i1 = x[10]
    e0e1_x_tr12_i1 = x[11]
    e0e1_x_tr13_i1 = x[12]
    e0e1_x_tr14_i1 = x[13]
    e0e1_x_tr15_i1 = x[14]
    e0e1_x_tr16_i1 = x[15]
    e0e1_x_tr17_i1 = x[16]
    e0e1_x_tr18_i1 = x[17]
    e0e1_x_tr19_i1 = x[18]
    e0_x_Reboilerin_i1 = x[19]
    e0_x_D_i2 = x[20]
    e0e1_x_tr1_i2 = x[21]
    e0e1_x_tr2_i2 = x[22]
    e0e1_x_tr3_i2 = x[23]
    e0e1_x_tr4_i2 = x[24]
    e0e1_x_tr5_i2 = x[25]
    e0e1_x_tr6_i2 = x[26]
    e0e1_x_tr7_i2 = x[27]
    e0e1_x_tr8_i2 = x[28]
    e0e1_x_tr9_i2 = x[29]
    e0e1_x_tr10_i2 = x[30]
    e0e1_x_tr11_i2 = x[31]
    e0e1_x_tr12_i2 = x[32]
    e0e1_x_tr13_i2 = x[33]
    e0e1_x_tr14_i2 = x[34]
    e0e1_x_tr15_i2 = x[35]
    e0e1_x_tr16_i2 = x[36]
    e0e1_x_tr17_i2 = x[37]
    e0e1_x_tr18_i2 = x[38]
    e0e1_x_tr19_i2 = x[39]
    e0_x_Reboilerin_i2 = x[40]
    e0e1_x_B_i2 = x[41]
    e0_x_Reboiler_i1 = x[42]
    e0_x_Reboiler_i2 = x[43]
    e0_y_D_i1 = x[44]
    e0e1_y_tr2_i1 = x[45]
    e0e1_y_tr3_i1 = x[46]
    e0e1_y_tr4_i1 = x[47]
    e0e1_y_tr5_i1 = x[48]
    e0e1_y_tr6_i1 = x[49]
    e0e1_y_tr7_i1 = x[50]
    e0e1_y_tr8_i1 = x[51]
    e0e1_y_tr9_i1 = x[52]
    e0e1_y_tr10_i1 = x[53]
    e0e1_y_tr11_i1 = x[54]
    e0e1_y_tr12_i1 = x[55]
    e0e1_y_tr13_i1 = x[56]
    e0e1_y_tr14_i1 = x[57]
    e0e1_y_tr15_i1 = x[58]
    e0e1_y_tr16_i1 = x[59]
    e0e1_y_tr17_i1 = x[60]
    e0e1_y_tr18_i1 = x[61]
    e0e1_y_tr19_i1 = x[62]
    e0e1_y_tr20_i1 = x[63]
    e0_y_Reboiler_i1 = x[64]
    e0_y_D_i2 = x[65]
    e0e1_y_tr2_i2 = x[66]
    e0e1_y_tr3_i2 = x[67]
    e0e1_y_tr4_i2 = x[68]
    e0e1_y_tr5_i2 = x[69]
    e0e1_y_tr6_i2 = x[70]
    e0e1_y_tr7_i2 = x[71]
    e0e1_y_tr8_i2 = x[72]
    e0e1_y_tr9_i2 = x[73]
    e0e1_y_tr10_i2 = x[74]
    e0e1_y_tr11_i2 = x[75]
    e0e1_y_tr12_i2 = x[76]
    e0e1_y_tr13_i2 = x[77]
    e0e1_y_tr14_i2 = x[78]
    e0e1_y_tr15_i2 = x[79]
    e0e1_y_tr16_i2 = x[80]
    e0e1_y_tr17_i2 = x[81]
    e0e1_y_tr18_i2 = x[82]
    e0e1_y_tr19_i2 = x[83]
    e0e1_y_tr20_i2 = x[84]
    e0_y_Reboiler_i2 = x[85]
    e0_F_DLn = x[86]
    e0_K_D_i1 = x[87]
    e0_K_D_i2 = x[88]
    e0_Q_D = x[89]
    e0_R_Reflux = x[90]
    e0_T_D = x[91]
    e0_p_DL = x[92]
    e0_y_DV_i1 = x[93]
    e0_y_DV_i2 = x[94]
    e0e1_greek_eta_L_tr1 = x[95]
    e0_K_Reboiler_i1 = x[96]
    e0_K_Reboiler_i2 = x[97]
    e0_Q_Reboiler = x[98]
    e0_T_Reboiler = x[99]
    e0e1_greek_eta_L_tr2 = x[100]
    e0_p_ReboilerV = x[101]
    e0e1_greek_eta_L_tr3 = x[102]
    e0e1_greek_eta_L_tr4 = x[103]
    e0e1_greek_eta_L_tr5 = x[104]
    e0e1_greek_eta_L_tr6 = x[105]
    e0e1_greek_eta_L_tr7 = x[106]
    e0e1_greek_eta_L_tr8 = x[107]
    e0e1_greek_eta_L_tr9 = x[108]
    e0e1_greek_eta_L_tr10 = x[109]
    e0e1_greek_eta_L_tr11 = x[110]
    e0e1_greek_eta_L_tr12 = x[111]
    e0e1_greek_eta_L_tr13 = x[112]
    e0e1_greek_eta_L_tr14 = x[113]
    e0e1_greek_eta_L_tr15 = x[114]
    e0e1_greek_eta_L_tr16 = x[115]
    e0e1_greek_eta_L_tr17 = x[116]
    e0e1_greek_eta_L_tr18 = x[117]
    e0e1_greek_eta_L_tr19 = x[118]
    e0e1_greek_eta_L_tr20 = x[119]
    e0_x_F_i2 = x[120]
    e0e1_greek_eta_V_tr1 = x[121]
    e0e1_greek_eta_V_tr2 = x[122]
    e0e1_greek_eta_V_tr3 = x[123]
    e0e1_greek_eta_V_tr4 = x[124]
    e0e1_greek_eta_V_tr5 = x[125]
    e0e1_greek_eta_V_tr6 = x[126]
    e0e1_greek_eta_V_tr7 = x[127]
    e0e1_greek_eta_V_tr8 = x[128]
    e0e1_greek_eta_V_tr9 = x[129]
    e0e1_greek_eta_V_tr10 = x[130]
    e0e1_greek_eta_V_tr11 = x[131]
    e0e1_greek_eta_V_tr12 = x[132]
    e0e1_greek_eta_V_tr13 = x[133]
    e0e1_greek_eta_V_tr14 = x[134]
    e0e1_greek_eta_V_tr15 = x[135]
    e0e1_greek_eta_V_tr16 = x[136]
    e0e1_greek_eta_V_tr17 = x[137]
    e0e1_greek_eta_V_tr18 = x[138]
    e0e1_greek_eta_V_tr19 = x[139]
    e0e1_greek_eta_V_tr20 = x[140]
    e0e1_greek_rho_Lm_tr1 = x[141]
    e0e1_greek_rho_Lm_tr2 = x[142]
    e0e1_greek_rho_Lm_tr3 = x[143]
    e0e1_greek_rho_Lm_tr4 = x[144]
    e0e1_greek_rho_Lm_tr5 = x[145]
    e0e1_greek_rho_Lm_tr6 = x[146]
    e0e1_greek_rho_Lm_tr7 = x[147]
    e0e1_greek_rho_Lm_tr8 = x[148]
    e0e1_greek_rho_Lm_tr9 = x[149]
    e0e1_greek_rho_Lm_tr10 = x[150]
    e0e1_greek_rho_Lm_tr11 = x[151]
    e0e1_greek_rho_Lm_tr12 = x[152]
    e0e1_greek_rho_Lm_tr13 = x[153]
    e0e1_greek_rho_Lm_tr14 = x[154]
    e0e1_greek_rho_Lm_tr15 = x[155]
    e0e1_greek_rho_Lm_tr16 = x[156]
    e0e1_greek_rho_Lm_tr17 = x[157]
    e0e1_greek_rho_Lm_tr18 = x[158]
    e0e1_greek_rho_Lm_tr19 = x[159]
    e0e1_greek_rho_Lm_tr20 = x[160]
    e0e1_greek_rho_Vm_tr1 = x[161]
    e0e1_greek_rho_Vm_tr2 = x[162]
    e0e1_greek_rho_Vm_tr3 = x[163]
    e0e1_greek_rho_Vm_tr4 = x[164]
    e0e1_greek_rho_Vm_tr5 = x[165]
    e0e1_greek_rho_Vm_tr6 = x[166]
    e0e1_greek_rho_Vm_tr7 = x[167]
    e0e1_greek_rho_Vm_tr8 = x[168]
    e0e1_greek_rho_Vm_tr9 = x[169]
    e0e1_greek_rho_Vm_tr10 = x[170]
    e0e1_greek_rho_Vm_tr11 = x[171]
    e0e1_greek_rho_Vm_tr12 = x[172]
    e0e1_greek_rho_Vm_tr13 = x[173]
    e0e1_greek_rho_Vm_tr14 = x[174]
    e0e1_greek_rho_Vm_tr15 = x[175]
    e0e1_greek_rho_Vm_tr16 = x[176]
    e0e1_greek_rho_Vm_tr17 = x[177]
    e0e1_greek_rho_Vm_tr18 = x[178]
    e0e1_greek_rho_Vm_tr19 = x[179]
    e0e1_greek_rho_Vm_tr20 = x[180]
    e0e1_greek_sigma_tr1 = x[181]
    e0e1_greek_sigma_tr2 = x[182]
    e0e1_greek_sigma_tr3 = x[183]
    e0e1_greek_sigma_tr4 = x[184]
    e0e1_greek_sigma_tr5 = x[185]
    e0e1_greek_sigma_tr6 = x[186]
    e0e1_greek_sigma_tr7 = x[187]
    e0e1_greek_sigma_tr8 = x[188]
    e0e1_greek_sigma_tr9 = x[189]
    e0e1_greek_sigma_tr10 = x[190]
    e0e1_greek_sigma_tr11 = x[191]
    e0e1_greek_sigma_tr12 = x[192]
    e0e1_greek_sigma_tr13 = x[193]
    e0e1_greek_sigma_tr14 = x[194]
    e0e1_greek_sigma_tr15 = x[195]
    e0e1_greek_sigma_tr16 = x[196]
    e0e1_greek_sigma_tr17 = x[197]
    e0e1_greek_sigma_tr18 = x[198]
    e0e1_greek_sigma_tr19 = x[199]
    e0e1_greek_sigma_tr20 = x[200]
    e0e1_greek_theta_i1_j1_tr1 = x[201]
    e0e1_greek_theta_i1_j1_tr2 = x[202]
    e0e1_greek_theta_i1_j1_tr3 = x[203]
    e0e1_greek_theta_i1_j1_tr4 = x[204]
    e0e1_greek_theta_i1_j1_tr5 = x[205]
    e0e1_greek_theta_i1_j1_tr6 = x[206]
    e0e1_greek_theta_i1_j1_tr7 = x[207]
    e0e1_greek_theta_i1_j1_tr8 = x[208]
    e0e1_greek_theta_i1_j1_tr9 = x[209]
    e0e1_greek_theta_i1_j1_tr10 = x[210]
    e0e1_greek_theta_i1_j1_tr11 = x[211]
    e0e1_greek_theta_i1_j1_tr12 = x[212]
    e0e1_greek_theta_i1_j1_tr13 = x[213]
    e0e1_greek_theta_i1_j1_tr14 = x[214]
    e0e1_greek_theta_i1_j1_tr15 = x[215]
    e0e1_greek_theta_i1_j1_tr16 = x[216]
    e0e1_greek_theta_i1_j1_tr17 = x[217]
    e0e1_greek_theta_i1_j1_tr18 = x[218]
    e0e1_greek_theta_i1_j1_tr19 = x[219]
    e0e1_greek_theta_i1_j1_tr20 = x[220]
    e0e1_greek_theta_i1_j2_tr1 = x[221]
    e0e1_greek_theta_i1_j2_tr2 = x[222]
    e0e1_greek_theta_i1_j2_tr3 = x[223]
    e0e1_greek_theta_i1_j2_tr4 = x[224]
    e0e1_greek_theta_i1_j2_tr5 = x[225]
    e0e1_greek_theta_i1_j2_tr6 = x[226]
    e0e1_greek_theta_i1_j2_tr7 = x[227]
    e0e1_greek_theta_i1_j2_tr8 = x[228]
    e0e1_greek_theta_i1_j2_tr9 = x[229]
    e0e1_greek_theta_i1_j2_tr10 = x[230]
    e0e1_greek_theta_i1_j2_tr11 = x[231]
    e0e1_greek_theta_i1_j2_tr12 = x[232]
    e0e1_greek_theta_i1_j2_tr13 = x[233]
    e0e1_greek_theta_i1_j2_tr14 = x[234]
    e0e1_greek_theta_i1_j2_tr15 = x[235]
    e0e1_greek_theta_i1_j2_tr16 = x[236]
    e0e1_greek_theta_i1_j2_tr17 = x[237]
    e0e1_greek_theta_i1_j2_tr18 = x[238]
    e0e1_greek_theta_i1_j2_tr19 = x[239]
    e0e1_greek_theta_i1_j2_tr20 = x[240]
    e0e1_greek_theta_i2_j1_tr1 = x[241]
    e0e1_greek_theta_i2_j1_tr2 = x[242]
    e0e1_greek_theta_i2_j1_tr3 = x[243]
    e0e1_greek_theta_i2_j1_tr4 = x[244]
    e0e1_greek_theta_i2_j1_tr5 = x[245]
    e0e1_greek_theta_i2_j1_tr6 = x[246]
    e0e1_greek_theta_i2_j1_tr7 = x[247]
    e0e1_greek_theta_i2_j1_tr8 = x[248]
    e0e1_greek_theta_i2_j1_tr9 = x[249]
    e0e1_greek_theta_i2_j1_tr10 = x[250]
    e0e1_greek_theta_i2_j1_tr11 = x[251]
    e0e1_greek_theta_i2_j1_tr12 = x[252]
    e0e1_greek_theta_i2_j1_tr13 = x[253]
    e0e1_greek_theta_i2_j1_tr14 = x[254]
    e0e1_greek_theta_i2_j1_tr15 = x[255]
    e0e1_greek_theta_i2_j1_tr16 = x[256]
    e0e1_greek_theta_i2_j1_tr17 = x[257]
    e0e1_greek_theta_i2_j1_tr18 = x[258]
    e0e1_greek_theta_i2_j1_tr19 = x[259]
    e0e1_greek_theta_i2_j1_tr20 = x[260]
    e0e1_greek_theta_i2_j2_tr1 = x[261]
    e0e1_greek_theta_i2_j2_tr2 = x[262]
    e0e1_greek_theta_i2_j2_tr3 = x[263]
    e0e1_greek_theta_i2_j2_tr4 = x[264]
    e0e1_greek_theta_i2_j2_tr5 = x[265]
    e0e1_greek_theta_i2_j2_tr6 = x[266]
    e0e1_greek_theta_i2_j2_tr7 = x[267]
    e0e1_greek_theta_i2_j2_tr8 = x[268]
    e0e1_greek_theta_i2_j2_tr9 = x[269]
    e0e1_greek_theta_i2_j2_tr10 = x[270]
    e0e1_greek_theta_i2_j2_tr11 = x[271]
    e0e1_greek_theta_i2_j2_tr12 = x[272]
    e0e1_greek_theta_i2_j2_tr13 = x[273]
    e0e1_greek_theta_i2_j2_tr14 = x[274]
    e0e1_greek_theta_i2_j2_tr15 = x[275]
    e0e1_greek_theta_i2_j2_tr16 = x[276]
    e0e1_greek_theta_i2_j2_tr17 = x[277]
    e0e1_greek_theta_i2_j2_tr18 = x[278]
    e0e1_greek_theta_i2_j2_tr19 = x[279]
    e0e1_greek_theta_i2_j2_tr20 = x[280]
    e0e1_A = x[281]
    e0e1_F_BLn = x[282]
    e0_F_LReboilerinn = x[283]
    e0_F_DLRefluxn = x[284]
    e0e1_F_Ln_tr1 = x[285]
    e0e1_F_Ln_tr2 = x[286]
    e0e1_F_Ln_tr3 = x[287]
    e0e1_F_Ln_tr4 = x[288]
    e0e1_F_Ln_tr5 = x[289]
    e0e1_F_Ln_tr6 = x[290]
    e0e1_F_Ln_tr7 = x[291]
    e0e1_F_Ln_tr8 = x[292]
    e0e1_F_Ln_tr9 = x[293]
    e0e1_F_Ln_tr10 = x[294]
    e0e1_F_Ln_tr11 = x[295]
    e0e1_F_Ln_tr12 = x[296]
    e0e1_F_Ln_tr13 = x[297]
    e0e1_F_Ln_tr14 = x[298]
    e0e1_F_Ln_tr15 = x[299]
    e0e1_F_Ln_tr16 = x[300]
    e0e1_F_Ln_tr17 = x[301]
    e0e1_F_Ln_tr18 = x[302]
    e0e1_F_Ln_tr19 = x[303]
    e0e1_F_Ln_tr20 = x[304]
    e0e1_F_Ln_tr21 = x[305]
    e0_F_DVn = x[306]
    e0e1_F_Vn_tr2 = x[307]
    e0e1_F_Vn_tr3 = x[308]
    e0e1_F_Vn_tr4 = x[309]
    e0e1_F_Vn_tr5 = x[310]
    e0e1_F_Vn_tr6 = x[311]
    e0e1_F_Vn_tr7 = x[312]
    e0e1_F_Vn_tr8 = x[313]
    e0e1_F_Vn_tr9 = x[314]
    e0e1_F_Vn_tr10 = x[315]
    e0e1_F_Vn_tr11 = x[316]
    e0e1_F_Vn_tr12 = x[317]
    e0e1_F_Vn_tr13 = x[318]
    e0e1_F_Vn_tr14 = x[319]
    e0e1_F_Vn_tr15 = x[320]
    e0e1_F_Vn_tr16 = x[321]
    e0e1_F_Vn_tr17 = x[322]
    e0e1_F_Vn_tr18 = x[323]
    e0e1_F_Vn_tr19 = x[324]
    e0e1_F_Vn_tr20 = x[325]
    e0_F_ReboilerVn = x[326]
    e0e1_K_tr1_i1 = x[327]
    e0e1_K_tr2_i1 = x[328]
    e0e1_K_tr3_i1 = x[329]
    e0e1_K_tr4_i1 = x[330]
    e0e1_K_tr5_i1 = x[331]
    e0e1_K_tr6_i1 = x[332]
    e0e1_K_tr7_i1 = x[333]
    e0e1_K_tr8_i1 = x[334]
    e0e1_K_tr9_i1 = x[335]
    e0e1_K_tr10_i1 = x[336]
    e0e1_K_tr11_i1 = x[337]
    e0e1_K_tr12_i1 = x[338]
    e0e1_K_tr13_i1 = x[339]
    e0e1_K_tr14_i1 = x[340]
    e0e1_K_tr15_i1 = x[341]
    e0e1_K_tr16_i1 = x[342]
    e0e1_K_tr17_i1 = x[343]
    e0e1_K_tr18_i1 = x[344]
    e0e1_K_tr19_i1 = x[345]
    e0e1_K_tr20_i1 = x[346]
    e0e1_K_tr1_i2 = x[347]
    e0e1_K_tr2_i2 = x[348]
    e0e1_K_tr3_i2 = x[349]
    e0e1_K_tr4_i2 = x[350]
    e0e1_K_tr5_i2 = x[351]
    e0e1_K_tr6_i2 = x[352]
    e0e1_K_tr7_i2 = x[353]
    e0e1_K_tr8_i2 = x[354]
    e0e1_K_tr9_i2 = x[355]
    e0e1_K_tr10_i2 = x[356]
    e0e1_K_tr11_i2 = x[357]
    e0e1_K_tr12_i2 = x[358]
    e0e1_K_tr13_i2 = x[359]
    e0e1_K_tr14_i2 = x[360]
    e0e1_K_tr15_i2 = x[361]
    e0e1_K_tr16_i2 = x[362]
    e0e1_K_tr17_i2 = x[363]
    e0e1_K_tr18_i2 = x[364]
    e0e1_K_tr19_i2 = x[365]
    e0e1_K_tr20_i2 = x[366]
    e0e1_M_L_tr1 = x[367]
    e0e1_M_L_tr2 = x[368]
    e0e1_M_L_tr3 = x[369]
    e0e1_M_L_tr4 = x[370]
    e0e1_M_L_tr5 = x[371]
    e0e1_M_L_tr6 = x[372]
    e0e1_M_L_tr7 = x[373]
    e0e1_M_L_tr8 = x[374]
    e0e1_M_L_tr9 = x[375]
    e0e1_M_L_tr10 = x[376]
    e0e1_M_L_tr11 = x[377]
    e0e1_M_L_tr12 = x[378]
    e0e1_M_L_tr13 = x[379]
    e0e1_M_L_tr14 = x[380]
    e0e1_M_L_tr15 = x[381]
    e0e1_M_L_tr16 = x[382]
    e0e1_M_L_tr17 = x[383]
    e0e1_M_L_tr18 = x[384]
    e0e1_M_L_tr19 = x[385]
    e0e1_M_L_tr20 = x[386]
    e0e1_M_V_tr1 = x[387]
    e0e1_M_V_tr2 = x[388]
    e0e1_M_V_tr3 = x[389]
    e0e1_M_V_tr4 = x[390]
    e0e1_M_V_tr5 = x[391]
    e0e1_M_V_tr6 = x[392]
    e0e1_M_V_tr7 = x[393]
    e0e1_M_V_tr8 = x[394]
    e0e1_M_V_tr9 = x[395]
    e0e1_M_V_tr10 = x[396]
    e0e1_M_V_tr11 = x[397]
    e0e1_M_V_tr12 = x[398]
    e0e1_M_V_tr13 = x[399]
    e0e1_M_V_tr14 = x[400]
    e0e1_M_V_tr15 = x[401]
    e0e1_M_V_tr16 = x[402]
    e0e1_M_V_tr17 = x[403]
    e0e1_M_V_tr18 = x[404]
    e0e1_M_V_tr19 = x[405]
    e0e1_M_V_tr20 = x[406]
    e0e1_P_i1_tr1 = x[407]
    e0e1_P_i1_tr2 = x[408]
    e0e1_P_i1_tr3 = x[409]
    e0e1_P_i1_tr4 = x[410]
    e0e1_P_i1_tr5 = x[411]
    e0e1_P_i1_tr6 = x[412]
    e0e1_P_i1_tr7 = x[413]
    e0e1_P_i1_tr8 = x[414]
    e0e1_P_i1_tr9 = x[415]
    e0e1_P_i1_tr10 = x[416]
    e0e1_P_i1_tr11 = x[417]
    e0e1_P_i1_tr12 = x[418]
    e0e1_P_i1_tr13 = x[419]
    e0e1_P_i1_tr14 = x[420]
    e0e1_P_i1_tr15 = x[421]
    e0e1_P_i1_tr16 = x[422]
    e0e1_P_i1_tr17 = x[423]
    e0e1_P_i1_tr18 = x[424]
    e0e1_P_i1_tr19 = x[425]
    e0e1_P_i1_tr20 = x[426]
    e0e1_P_i2_tr1 = x[427]
    e0e1_P_i2_tr2 = x[428]
    e0e1_P_i2_tr3 = x[429]
    e0e1_P_i2_tr4 = x[430]
    e0e1_P_i2_tr5 = x[431]
    e0e1_P_i2_tr6 = x[432]
    e0e1_P_i2_tr7 = x[433]
    e0e1_P_i2_tr8 = x[434]
    e0e1_P_i2_tr9 = x[435]
    e0e1_P_i2_tr10 = x[436]
    e0e1_P_i2_tr11 = x[437]
    e0e1_P_i2_tr12 = x[438]
    e0e1_P_i2_tr13 = x[439]
    e0e1_P_i2_tr14 = x[440]
    e0e1_P_i2_tr15 = x[441]
    e0e1_P_i2_tr16 = x[442]
    e0e1_P_i2_tr17 = x[443]
    e0e1_P_i2_tr18 = x[444]
    e0e1_P_i2_tr19 = x[445]
    e0e1_P_i2_tr20 = x[446]
    e0e1_P_tr1 = x[447]
    e0e1_P_tr2 = x[448]
    e0e1_P_tr3 = x[449]
    e0e1_P_tr4 = x[450]
    e0e1_P_tr5 = x[451]
    e0e1_P_tr6 = x[452]
    e0e1_P_tr7 = x[453]
    e0e1_P_tr8 = x[454]
    e0e1_P_tr9 = x[455]
    e0e1_P_tr10 = x[456]
    e0e1_P_tr11 = x[457]
    e0e1_P_tr12 = x[458]
    e0e1_P_tr13 = x[459]
    e0e1_P_tr14 = x[460]
    e0e1_P_tr15 = x[461]
    e0e1_P_tr16 = x[462]
    e0e1_P_tr17 = x[463]
    e0e1_P_tr18 = x[464]
    e0e1_P_tr19 = x[465]
    e0e1_P_tr20 = x[466]
    e0e1_T_tr1 = x[467]
    e0e1_T_tr2 = x[468]
    e0e1_T_tr3 = x[469]
    e0e1_T_tr4 = x[470]
    e0e1_T_tr5 = x[471]
    e0e1_T_tr6 = x[472]
    e0e1_T_tr7 = x[473]
    e0e1_T_tr8 = x[474]
    e0e1_T_tr9 = x[475]
    e0e1_T_tr10 = x[476]
    e0e1_T_tr11 = x[477]
    e0e1_T_tr12 = x[478]
    e0e1_T_tr13 = x[479]
    e0e1_T_tr14 = x[480]
    e0e1_T_tr15 = x[481]
    e0e1_T_tr16 = x[482]
    e0e1_T_tr17 = x[483]
    e0e1_T_tr18 = x[484]
    e0e1_T_tr19 = x[485]
    e0e1_T_tr20 = x[486]
    e0e1_T_B = x[487]
    e0e1_h_BLn = x[488]
    e0_h_LReboilern = x[489]
    e0_h_Fn = x[490]
    e0_h_DLn = x[491]
    e0e1_h_Ln_tr1 = x[492]
    e0e1_h_Ln_tr2 = x[493]
    e0e1_h_Ln_tr3 = x[494]
    e0e1_h_Ln_tr4 = x[495]
    e0e1_h_Ln_tr5 = x[496]
    e0e1_h_Ln_tr6 = x[497]
    e0e1_h_Ln_tr7 = x[498]
    e0e1_h_Ln_tr8 = x[499]
    e0e1_h_Ln_tr9 = x[500]
    e0e1_h_Ln_tr10 = x[501]
    e0e1_h_Ln_tr11 = x[502]
    e0e1_h_Ln_tr12 = x[503]
    e0e1_h_Ln_tr13 = x[504]
    e0e1_h_Ln_tr14 = x[505]
    e0e1_h_Ln_tr15 = x[506]
    e0e1_h_Ln_tr16 = x[507]
    e0e1_h_Ln_tr17 = x[508]
    e0e1_h_Ln_tr18 = x[509]
    e0e1_h_Ln_tr19 = x[510]
    e0_h_LReboilerinn = x[511]
    e0_h_DVn = x[512]
    e0e1_h_Vn_tr2 = x[513]
    e0e1_h_Vn_tr3 = x[514]
    e0e1_h_Vn_tr4 = x[515]
    e0e1_h_Vn_tr5 = x[516]
    e0e1_h_Vn_tr6 = x[517]
    e0e1_h_Vn_tr7 = x[518]
    e0e1_h_Vn_tr8 = x[519]
    e0e1_h_Vn_tr9 = x[520]
    e0e1_h_Vn_tr10 = x[521]
    e0e1_h_Vn_tr11 = x[522]
    e0e1_h_Vn_tr12 = x[523]
    e0e1_h_Vn_tr13 = x[524]
    e0e1_h_Vn_tr14 = x[525]
    e0e1_h_Vn_tr15 = x[526]
    e0e1_h_Vn_tr16 = x[527]
    e0e1_h_Vn_tr17 = x[528]
    e0e1_h_Vn_tr18 = x[529]
    e0e1_h_Vn_tr19 = x[530]
    e0e1_h_Vn_tr20 = x[531]
    e0_h_ReboilerVn = x[532]
    e0e1_p_tr1 = x[533]
    e0e1_p_tr2 = x[534]
    e0e1_p_tr3 = x[535]
    e0e1_p_tr4 = x[536]
    e0e1_p_tr5 = x[537]
    e0e1_p_tr6 = x[538]
    e0e1_p_tr7 = x[539]
    e0e1_p_tr8 = x[540]
    e0e1_p_tr9 = x[541]
    e0e1_p_tr10 = x[542]
    e0e1_p_tr11 = x[543]
    e0e1_p_tr12 = x[544]
    e0e1_p_tr13 = x[545]
    e0e1_p_tr14 = x[546]
    e0e1_p_tr15 = x[547]
    e0e1_p_tr16 = x[548]
    e0e1_p_tr17 = x[549]
    e0e1_p_tr18 = x[550]
    e0e1_p_tr19 = x[551]
    e0_p_LReboiler = x[552]
    e0e1_p_tr21 = x[553]



# Getting parameter values:
    e0e1_r_n_tr1_i1 = p[0]
    e0e1_r_n_tr2_i1 = p[1]
    e0e1_r_n_tr3_i1 = p[2]
    e0e1_r_n_tr4_i1 = p[3]
    e0e1_r_n_tr5_i1 = p[4]
    e0e1_r_n_tr6_i1 = p[5]
    e0e1_r_n_tr7_i1 = p[6]
    e0e1_r_n_tr8_i1 = p[7]
    e0e1_r_n_tr9_i1 = p[8]
    e0e1_r_n_tr10_i1 = p[9]
    e0e1_r_n_tr11_i1 = p[10]
    e0e1_r_n_tr12_i1 = p[11]
    e0e1_r_n_tr13_i1 = p[12]
    e0e1_r_n_tr14_i1 = p[13]
    e0e1_r_n_tr15_i1 = p[14]
    e0e1_r_n_tr16_i1 = p[15]
    e0e1_r_n_tr17_i1 = p[16]
    e0e1_r_n_tr18_i1 = p[17]
    e0e1_r_n_tr19_i1 = p[18]
    e0e1_r_n_tr20_i1 = p[19]
    e0e1_r_n_tr1_i2 = p[20]
    e0e1_r_n_tr2_i2 = p[21]
    e0e1_r_n_tr3_i2 = p[22]
    e0e1_r_n_tr4_i2 = p[23]
    e0e1_r_n_tr5_i2 = p[24]
    e0e1_r_n_tr6_i2 = p[25]
    e0e1_r_n_tr7_i2 = p[26]
    e0e1_r_n_tr8_i2 = p[27]
    e0e1_r_n_tr9_i2 = p[28]
    e0e1_r_n_tr10_i2 = p[29]
    e0e1_r_n_tr11_i2 = p[30]
    e0e1_r_n_tr12_i2 = p[31]
    e0e1_r_n_tr13_i2 = p[32]
    e0e1_r_n_tr14_i2 = p[33]
    e0e1_r_n_tr15_i2 = p[34]
    e0e1_r_n_tr16_i2 = p[35]
    e0e1_r_n_tr17_i2 = p[36]
    e0e1_r_n_tr18_i2 = p[37]
    e0e1_r_n_tr19_i2 = p[38]
    e0e1_r_n_tr20_i2 = p[39]
    e0_x_D_i1 = p[40]
    e0e1_x_B_i1 = p[41]
    e0_p_F = p[42]
    e0_greek_Deltap_D = p[43]
    e0_Q_DLoss = p[44]
    e0_h_DELn = p[45]
    e0_greek_Deltap_Reboiler = p[46]
    e0_Q_LossReboiler = p[47]
    e0_h_ELReboilern = p[48]
    e0_h_EReboilerVn = p[49]
    e0_x_F_i1 = p[50]
    e0_T_F = p[51]
    e0e1_greek_pi = p[52]
    e0_h_EFn = p[53]
    e0e1_F_Fn_tr1 = p[54]
    e0e1_F_Fn_tr2 = p[55]
    e0e1_F_Fn_tr3 = p[56]
    e0e1_F_Fn_tr4 = p[57]
    e0e1_F_Fn_tr5 = p[58]
    e0e1_F_Fn_tr6 = p[59]
    e0e1_F_Fn_tr7 = p[60]
    e0e1_F_Fn_tr8 = p[61]
    e0e1_F_Fn_tr9 = p[62]
    e0e1_F_Fn_tr10 = p[63]
    e0e1_F_Fn_tr11 = p[64]
    e0e1_F_Fn_tr12 = p[65]
    e0e1_F_Fn_tr13 = p[66]
    e0e1_F_Fn_tr14 = p[67]
    e0e1_F_Fn_tr15 = p[68]
    e0e1_F_Fn_tr16 = p[69]
    e0e1_F_Fn_tr17 = p[70]
    e0e1_F_Fn_tr18 = p[71]
    e0e1_F_Fn_tr19 = p[72]
    e0e1_F_Fn_tr20 = p[73]
    e0_F_LReboilern = p[74]
    e0e1_M_i1 = p[75]
    e0e1_M_i2 = p[76]
    e0e1_greek_Deltap_tr0 = p[77]
    e0e1_greek_Deltap_tr1 = p[78]
    e0e1_greek_Deltap_tr2 = p[79]
    e0e1_greek_Deltap_tr3 = p[80]
    e0e1_greek_Deltap_tr4 = p[81]
    e0e1_greek_Deltap_tr5 = p[82]
    e0e1_Q_Loss_tr1 = p[83]
    e0e1_Q_Loss_tr2 = p[84]
    e0e1_Q_Loss_tr3 = p[85]
    e0e1_greek_Deltap_tr6 = p[86]
    e0e1_Q_Loss_tr4 = p[87]
    e0e1_Q_Loss_tr5 = p[88]
    e0e1_Q_Loss_tr6 = p[89]
    e0e1_Q_Loss_tr7 = p[90]
    e0e1_Q_Loss_tr8 = p[91]
    e0e1_Q_Loss_tr9 = p[92]
    e0e1_Q_Loss_tr10 = p[93]
    e0e1_Q_Loss_tr11 = p[94]
    e0e1_Q_Loss_tr12 = p[95]
    e0e1_Q_Loss_tr13 = p[96]
    e0e1_greek_Deltap_tr7 = p[97]
    e0e1_Q_Loss_tr14 = p[98]
    e0e1_Q_Loss_tr15 = p[99]
    e0e1_Q_Loss_tr16 = p[100]
    e0e1_Q_Loss_tr17 = p[101]
    e0e1_Q_Loss_tr18 = p[102]
    e0e1_Q_Loss_tr19 = p[103]
    e0e1_Q_Loss_tr20 = p[104]
    e0e1_R = p[105]
    e0e1_greek_Deltap_tr8 = p[106]
    e0e1_greek_Deltap_tr9 = p[107]
    e0e1_d = p[108]
    e0e1_greek_Deltap_tr10 = p[109]
    e0e1_h_BELn = p[110]
    e0e1_h_ELn_tr1 = p[111]
    e0e1_h_ELn_tr2 = p[112]
    e0e1_greek_Deltap_tr11 = p[113]
    e0e1_h_ELn_tr3 = p[114]
    e0e1_h_ELn_tr4 = p[115]
    e0e1_h_ELn_tr5 = p[116]
    e0e1_h_ELn_tr6 = p[117]
    e0e1_h_ELn_tr7 = p[118]
    e0e1_h_ELn_tr8 = p[119]
    e0e1_h_ELn_tr9 = p[120]
    e0e1_h_ELn_tr10 = p[121]
    e0e1_h_ELn_tr11 = p[122]
    e0e1_h_ELn_tr12 = p[123]
    e0e1_greek_Deltap_tr12 = p[124]
    e0e1_h_ELn_tr13 = p[125]
    e0e1_h_ELn_tr14 = p[126]
    e0e1_h_ELn_tr15 = p[127]
    e0e1_h_ELn_tr16 = p[128]
    e0e1_h_ELn_tr17 = p[129]
    e0e1_h_ELn_tr18 = p[130]
    e0e1_h_ELn_tr19 = p[131]
    e0e1_h_ELn_tr20 = p[132]
    e0e1_h_EVn_tr1 = p[133]
    e0e1_h_EVn_tr2 = p[134]
    e0e1_greek_Deltap_tr13 = p[135]
    e0e1_h_EVn_tr3 = p[136]
    e0e1_h_EVn_tr4 = p[137]
    e0e1_h_EVn_tr5 = p[138]
    e0e1_h_EVn_tr6 = p[139]
    e0e1_h_EVn_tr7 = p[140]
    e0e1_h_EVn_tr8 = p[141]
    e0e1_h_EVn_tr9 = p[142]
    e0e1_h_EVn_tr10 = p[143]
    e0e1_h_EVn_tr11 = p[144]
    e0e1_h_EVn_tr12 = p[145]
    e0e1_greek_Deltap_tr14 = p[146]
    e0e1_h_EVn_tr13 = p[147]
    e0e1_h_EVn_tr14 = p[148]
    e0e1_h_EVn_tr15 = p[149]
    e0e1_h_EVn_tr16 = p[150]
    e0e1_h_EVn_tr17 = p[151]
    e0e1_h_EVn_tr18 = p[152]
    e0e1_h_EVn_tr19 = p[153]
    e0e1_h_EVn_tr20 = p[154]
    e0e1_greek_Deltap_tr15 = p[155]
    e0e1_greek_Deltap_tr16 = p[156]
    e0e1_greek_Deltap_tr17 = p[157]
    e0e1_greek_Deltap_tr18 = p[158]
    e0e1_greek_Deltap_tr19 = p[159]
    e0e1_greek_Deltap_tr20 = p[160]
    e0_p_DV = p[161]
    e0_A_LVpd101_i1 = p[162]
    e0_A_LVpd101_i2 = p[163]
    e0_A_Lnrhod105_i1 = p[164]
    e0_A_Lnrhod105_i2 = p[165]
    e0_A_LVhnd106_i1 = p[166]
    e0_A_LVhnd106_i2 = p[167]
    e0_A_Vhnd127_i1 = p[168]
    e0_A_Vhnd127_i2 = p[169]
    e0_B_LVpd101_i1 = p[170]
    e0_B_LVpd101_i2 = p[171]
    e0_B_Lnrhod105_i1 = p[172]
    e0_B_Lnrhod105_i2 = p[173]
    e0_B_LVhnd106_i1 = p[174]
    e0_B_LVhnd106_i2 = p[175]
    e0_B_Vhnd127_i1 = p[176]
    e0_B_Vhnd127_i2 = p[177]
    e0_C_LVpd101_i1 = p[178]
    e0_C_LVpd101_i2 = p[179]
    e0_C_Lnrhod105_i1 = p[180]
    e0_C_Lnrhod105_i2 = p[181]
    e0_C_LVhnd106_i1 = p[182]
    e0_C_LVhnd106_i2 = p[183]
    e0_C_Vhnd127_i1 = p[184]
    e0_C_Vhnd127_i2 = p[185]
    e0_D_LVpd101_i1 = p[186]
    e0_D_LVpd101_i2 = p[187]
    e0_D_Lnrhod105_i1 = p[188]
    e0_D_Lnrhod105_i2 = p[189]
    e0_D_LVhnd106_i1 = p[190]
    e0_D_LVhnd106_i2 = p[191]
    e0_D_Vhnd127_i1 = p[192]
    e0_D_Vhnd127_i2 = p[193]
    e0_E_LVpd101_i1 = p[194]
    e0_E_LVpd101_i2 = p[195]
    e0_E_LVhnd106_i1 = p[196]
    e0_E_LVhnd106_i2 = p[197]
    e0_E_Vhnd127_i1 = p[198]
    e0_E_Vhnd127_i2 = p[199]
    e0_F_Vhnd127_i1 = p[200]
    e0_F_Vhnd127_i2 = p[201]
    e0_G_Vhnd127_i1 = p[202]
    e0_G_Vhnd127_i2 = p[203]
    e0_T_crit_i1 = p[204]
    e0_T_ho_i1 = p[205]
    e0_T_crit_i2 = p[206]
    e0_T_ho_i2 = p[207]
    e0_h_o_i1 = p[208]
    e0_h_o_i2 = p[209]
    e0e1_A_Letad101_i1 = p[210]
    e0e1_A_LVpd101_i1 = p[211]
    e0e1_A_Letad101_i2 = p[212]
    e0e1_A_LVpd101_i2 = p[213]
    e0e1_A_Vetad102_i1 = p[214]
    e0e1_A_Vetad102_i2 = p[215]
    e0e1_A_Lnrhod105_i1 = p[216]
    e0e1_A_Lnrhod105_i2 = p[217]
    e0e1_A_LVhnd106_i1 = p[218]
    e0e1_A_sigd106_i1 = p[219]
    e0e1_A_LVhnd106_i2 = p[220]
    e0e1_A_sigd106_i2 = p[221]
    e0e1_A_Wsigd106 = p[222]
    e0e1_A_Vhnd127_i1 = p[223]
    e0e1_A_Vhnd127_i2 = p[224]
    e0e1_B_Letad101_i1 = p[225]
    e0e1_B_LVpd101_i1 = p[226]
    e0e1_B_Letad101_i2 = p[227]
    e0e1_B_LVpd101_i2 = p[228]
    e0e1_B_Vetad102_i1 = p[229]
    e0e1_B_Vetad102_i2 = p[230]
    e0e1_B_Lnrhod105_i1 = p[231]
    e0e1_B_Lnrhod105_i2 = p[232]
    e0e1_B_LVhnd106_i1 = p[233]
    e0e1_B_sigd106_i1 = p[234]
    e0e1_B_LVhnd106_i2 = p[235]
    e0e1_B_sigd106_i2 = p[236]
    e0e1_B_Wsigd106 = p[237]
    e0e1_B_Vhnd127_i1 = p[238]
    e0e1_B_Vhnd127_i2 = p[239]
    e0e1_C_Letad101_i1 = p[240]
    e0e1_C_LVpd101_i1 = p[241]
    e0e1_C_Letad101_i2 = p[242]
    e0e1_C_LVpd101_i2 = p[243]
    e0e1_C_Vetad102_i1 = p[244]
    e0e1_C_Vetad102_i2 = p[245]
    e0e1_C_Lnrhod105_i1 = p[246]
    e0e1_C_Lnrhod105_i2 = p[247]
    e0e1_C_LVhnd106_i1 = p[248]
    e0e1_C_sigd106_i1 = p[249]
    e0e1_C_LVhnd106_i2 = p[250]
    e0e1_C_sigd106_i2 = p[251]
    e0e1_C_Wsigd106 = p[252]
    e0e1_C_Vhnd127_i1 = p[253]
    e0e1_C_Vhnd127_i2 = p[254]
    e0e1_D_Letad101_i1 = p[255]
    e0e1_D_LVpd101_i1 = p[256]
    e0e1_D_Letad101_i2 = p[257]
    e0e1_D_LVpd101_i2 = p[258]
    e0e1_D_Vetad102_i1 = p[259]
    e0e1_D_Vetad102_i2 = p[260]
    e0e1_D_Lnrhod105_i1 = p[261]
    e0e1_D_Lnrhod105_i2 = p[262]
    e0e1_D_LVhnd106_i1 = p[263]
    e0e1_D_sigd106_i1 = p[264]
    e0e1_D_LVhnd106_i2 = p[265]
    e0e1_D_sigd106_i2 = p[266]
    e0e1_D_Wsigd106 = p[267]
    e0e1_D_Vhnd127_i1 = p[268]
    e0e1_D_Vhnd127_i2 = p[269]
    e0e1_E_Letad101_i1 = p[270]
    e0e1_E_LVpd101_i1 = p[271]
    e0e1_E_Letad101_i2 = p[272]
    e0e1_E_LVpd101_i2 = p[273]
    e0e1_E_LVhnd106_i1 = p[274]
    e0e1_E_sigd106_i1 = p[275]
    e0e1_E_LVhnd106_i2 = p[276]
    e0e1_E_sigd106_i2 = p[277]
    e0e1_E_Wsigd106 = p[278]
    e0e1_E_Vhnd127_i1 = p[279]
    e0e1_E_Vhnd127_i2 = p[280]
    e0e1_F_Vhnd127_i1 = p[281]
    e0e1_F_Vhnd127_i2 = p[282]
    e0e1_G_Vhnd127_i1 = p[283]
    e0e1_G_Vhnd127_i2 = p[284]
    e0e1_T_Wcrit = p[285]
    e0e1_T_crit_i1 = p[286]
    e0e1_T_ho_i1 = p[287]
    e0e1_T_crit_i2 = p[288]
    e0e1_T_ho_i2 = p[289]
    e0e1_h_o_i1 = p[290]
    e0e1_h_o_i2 = p[291]



   # Getting function values:
    e0e1_greek_rho_Ln_tr17_i1 = fun_75389(e0e1_T_tr17,e0e1_A_Lnrhod105_i1,e0e1_B_Lnrhod105_i1,e0e1_C_Lnrhod105_i1,e0e1_D_Lnrhod105_i1)
    e0e1_F_Vm_tr20 = fun_94396(e0e1_F_Vn_tr20,e0e1_M_V_tr20)
    e0e1_F_Lm_tr7 = fun_94396(e0e1_F_Ln_tr7,e0e1_M_L_tr7)
    e0e1_h_Vn_tr2_i1 = fun_93415(e0e1_T_tr2,e0e1_A_Vhnd127_i1,e0e1_B_Vhnd127_i1,e0e1_C_Vhnd127_i1,e0e1_D_Vhnd127_i1,e0e1_E_Vhnd127_i1,e0e1_F_Vhnd127_i1,e0e1_G_Vhnd127_i1,e0e1_T_ho_i1,e0e1_h_o_i1)
    e0e1_h_LVn_tr5_i1 = fun_75390(e0e1_T_tr5,e0e1_A_LVhnd106_i1,e0e1_B_LVhnd106_i1,e0e1_C_LVhnd106_i1,e0e1_D_LVhnd106_i1,e0e1_E_LVhnd106_i1,e0e1_T_crit_i1)
    e0e1_h_Vn_tr2_i2 = fun_93415(e0e1_T_tr2,e0e1_A_Vhnd127_i2,e0e1_B_Vhnd127_i2,e0e1_C_Vhnd127_i2,e0e1_D_Vhnd127_i2,e0e1_E_Vhnd127_i2,e0e1_F_Vhnd127_i2,e0e1_G_Vhnd127_i2,e0e1_T_ho_i2,e0e1_h_o_i2)
    e0e1_greek_rho_Ln_tr12_i1 = fun_75389(e0e1_T_tr12,e0e1_A_Lnrhod105_i1,e0e1_B_Lnrhod105_i1,e0e1_C_Lnrhod105_i1,e0e1_D_Lnrhod105_i1)
    e0e1_F_Lm_tr3 = fun_94396(e0e1_F_Ln_tr3,e0e1_M_L_tr3)
    e0e1_h_LVn_tr3_i2 = fun_75390(e0e1_T_tr3,e0e1_A_LVhnd106_i2,e0e1_B_LVhnd106_i2,e0e1_C_LVhnd106_i2,e0e1_D_LVhnd106_i2,e0e1_E_LVhnd106_i2,e0e1_T_crit_i2)
    e0e1_greek_eta_V_i1_tr6 = fun_75392(e0e1_T_tr6,e0e1_A_Vetad102_i1,e0e1_B_Vetad102_i1,e0e1_C_Vetad102_i1,e0e1_D_Vetad102_i1)
    e0e1_greek_sigma_W_tr3 = fun_75390(e0e1_T_tr3,e0e1_A_Wsigd106,e0e1_B_Wsigd106,e0e1_C_Wsigd106,e0e1_D_Wsigd106,e0e1_E_Wsigd106,e0e1_T_Wcrit)
    e0e1_greek_eta_L_tr6_i2 = fun_75386(e0e1_T_tr6,e0e1_A_Letad101_i2,e0e1_B_Letad101_i2,e0e1_C_Letad101_i2,e0e1_D_Letad101_i2,e0e1_E_Letad101_i2)
    e0_h_FLVn_i2 = fun_75390(e0_T_F,e0_A_LVhnd106_i2,e0_B_LVhnd106_i2,e0_C_LVhnd106_i2,e0_D_LVhnd106_i2,e0_E_LVhnd106_i2,e0_T_crit_i2)
    e0e1_greek_rho_Ln_tr5_i1 = fun_75389(e0e1_T_tr5,e0e1_A_Lnrhod105_i1,e0e1_B_Lnrhod105_i1,e0e1_C_Lnrhod105_i1,e0e1_D_Lnrhod105_i1)
    e0e1_h_LVn_tr17_i1 = fun_75390(e0e1_T_tr17,e0e1_A_LVhnd106_i1,e0e1_B_LVhnd106_i1,e0e1_C_LVhnd106_i1,e0e1_D_LVhnd106_i1,e0e1_E_LVhnd106_i1,e0e1_T_crit_i1)
    e0e1_h_LVn_tr20_i1 = fun_75390(e0e1_T_tr20,e0e1_A_LVhnd106_i1,e0e1_B_LVhnd106_i1,e0e1_C_LVhnd106_i1,e0e1_D_LVhnd106_i1,e0e1_E_LVhnd106_i1,e0e1_T_crit_i1)
    e0e1_greek_eta_L_tr4_i1 = fun_75386(e0e1_T_tr4,e0e1_A_Letad101_i1,e0e1_B_Letad101_i1,e0e1_C_Letad101_i1,e0e1_D_Letad101_i1,e0e1_E_Letad101_i1)
    e0e1_greek_eta_V_i2_tr12 = fun_75392(e0e1_T_tr12,e0e1_A_Vetad102_i2,e0e1_B_Vetad102_i2,e0e1_C_Vetad102_i2,e0e1_D_Vetad102_i2)
    e0e1_greek_eta_V_i2_tr14 = fun_75392(e0e1_T_tr14,e0e1_A_Vetad102_i2,e0e1_B_Vetad102_i2,e0e1_C_Vetad102_i2,e0e1_D_Vetad102_i2)
    e0e1_h_LVn_tr18_i1 = fun_75390(e0e1_T_tr18,e0e1_A_LVhnd106_i1,e0e1_B_LVhnd106_i1,e0e1_C_LVhnd106_i1,e0e1_D_LVhnd106_i1,e0e1_E_LVhnd106_i1,e0e1_T_crit_i1)
    e0e1_greek_rho_Ln_tr6_i1 = fun_75389(e0e1_T_tr6,e0e1_A_Lnrhod105_i1,e0e1_B_Lnrhod105_i1,e0e1_C_Lnrhod105_i1,e0e1_D_Lnrhod105_i1)
    e0e1_greek_sigma_i1_tr2 = fun_75390(e0e1_T_tr2,e0e1_A_sigd106_i1,e0e1_B_sigd106_i1,e0e1_C_sigd106_i1,e0e1_D_sigd106_i1,e0e1_E_sigd106_i1,e0e1_T_crit_i1)
    e0e1_greek_eta_L_tr18_i2 = fun_75386(e0e1_T_tr18,e0e1_A_Letad101_i2,e0e1_B_Letad101_i2,e0e1_C_Letad101_i2,e0e1_D_Letad101_i2,e0e1_E_Letad101_i2)
    e0e1_h_LVn_tr7_i2 = fun_75390(e0e1_T_tr7,e0e1_A_LVhnd106_i2,e0e1_B_LVhnd106_i2,e0e1_C_LVhnd106_i2,e0e1_D_LVhnd106_i2,e0e1_E_LVhnd106_i2,e0e1_T_crit_i2)
    e0e1_p_LV_tr4_i2 = fun_75386(e0e1_T_tr4,e0e1_A_LVpd101_i2,e0e1_B_LVpd101_i2,e0e1_C_LVpd101_i2,e0e1_D_LVpd101_i2,e0e1_E_LVpd101_i2)
    e0e1_greek_eta_L_tr5_i2 = fun_75386(e0e1_T_tr5,e0e1_A_Letad101_i2,e0e1_B_Letad101_i2,e0e1_C_Letad101_i2,e0e1_D_Letad101_i2,e0e1_E_Letad101_i2)
    e0_greek_rho_DLn_i1 = fun_75389(e0_T_D,e0_A_Lnrhod105_i1,e0_B_Lnrhod105_i1,e0_C_Lnrhod105_i1,e0_D_Lnrhod105_i1)
    e0e1_F_Vm_tr15 = fun_94396(e0e1_F_Vn_tr15,e0e1_M_V_tr15)
    e0e1_F_Lm_tr8 = fun_94396(e0e1_F_Ln_tr8,e0e1_M_L_tr8)
    e0e1_h_Vn_tr6_i2 = fun_93415(e0e1_T_tr6,e0e1_A_Vhnd127_i2,e0e1_B_Vhnd127_i2,e0e1_C_Vhnd127_i2,e0e1_D_Vhnd127_i2,e0e1_E_Vhnd127_i2,e0e1_F_Vhnd127_i2,e0e1_G_Vhnd127_i2,e0e1_T_ho_i2,e0e1_h_o_i2)
    e0e1_F_Vm_tr9 = fun_94396(e0e1_F_Vn_tr9,e0e1_M_V_tr9)
    e0e1_h_LVn_tr16_i1 = fun_75390(e0e1_T_tr16,e0e1_A_LVhnd106_i1,e0e1_B_LVhnd106_i1,e0e1_C_LVhnd106_i1,e0e1_D_LVhnd106_i1,e0e1_E_LVhnd106_i1,e0e1_T_crit_i1)
    e0e1_h_LVn_tr17_i2 = fun_75390(e0e1_T_tr17,e0e1_A_LVhnd106_i2,e0e1_B_LVhnd106_i2,e0e1_C_LVhnd106_i2,e0e1_D_LVhnd106_i2,e0e1_E_LVhnd106_i2,e0e1_T_crit_i2)
    e0e1_greek_eta_L_tr12_i1 = fun_75386(e0e1_T_tr12,e0e1_A_Letad101_i1,e0e1_B_Letad101_i1,e0e1_C_Letad101_i1,e0e1_D_Letad101_i1,e0e1_E_Letad101_i1)
    e0e1_greek_rho_Ln_tr18_i2 = fun_75389(e0e1_T_tr18,e0e1_A_Lnrhod105_i2,e0e1_B_Lnrhod105_i2,e0e1_C_Lnrhod105_i2,e0e1_D_Lnrhod105_i2)
    e0e1_h_Vn_tr13_i1 = fun_93415(e0e1_T_tr13,e0e1_A_Vhnd127_i1,e0e1_B_Vhnd127_i1,e0e1_C_Vhnd127_i1,e0e1_D_Vhnd127_i1,e0e1_E_Vhnd127_i1,e0e1_F_Vhnd127_i1,e0e1_G_Vhnd127_i1,e0e1_T_ho_i1,e0e1_h_o_i1)
    e0e1_p_LV_tr2_i2 = fun_75386(e0e1_T_tr2,e0e1_A_LVpd101_i2,e0e1_B_LVpd101_i2,e0e1_C_LVpd101_i2,e0e1_D_LVpd101_i2,e0e1_E_LVpd101_i2)
    e0e1_greek_eta_V_i1_tr2 = fun_75392(e0e1_T_tr2,e0e1_A_Vetad102_i1,e0e1_B_Vetad102_i1,e0e1_C_Vetad102_i1,e0e1_D_Vetad102_i1)
    e0e1_p_LV_tr7_i2 = fun_75386(e0e1_T_tr7,e0e1_A_LVpd101_i2,e0e1_B_LVpd101_i2,e0e1_C_LVpd101_i2,e0e1_D_LVpd101_i2,e0e1_E_LVpd101_i2)
    e0e1_greek_rho_Ln_tr8_i1 = fun_75389(e0e1_T_tr8,e0e1_A_Lnrhod105_i1,e0e1_B_Lnrhod105_i1,e0e1_C_Lnrhod105_i1,e0e1_D_Lnrhod105_i1)
    e0e1_greek_eta_V_i1_tr8 = fun_75392(e0e1_T_tr8,e0e1_A_Vetad102_i1,e0e1_B_Vetad102_i1,e0e1_C_Vetad102_i1,e0e1_D_Vetad102_i1)
    e0e1_greek_rho_Ln_tr16_i1 = fun_75389(e0e1_T_tr16,e0e1_A_Lnrhod105_i1,e0e1_B_Lnrhod105_i1,e0e1_C_Lnrhod105_i1,e0e1_D_Lnrhod105_i1)
    e0e1_greek_eta_V_i2_tr2 = fun_75392(e0e1_T_tr2,e0e1_A_Vetad102_i2,e0e1_B_Vetad102_i2,e0e1_C_Vetad102_i2,e0e1_D_Vetad102_i2)
    e0e1_greek_eta_V_i1_tr15 = fun_75392(e0e1_T_tr15,e0e1_A_Vetad102_i1,e0e1_B_Vetad102_i1,e0e1_C_Vetad102_i1,e0e1_D_Vetad102_i1)
    e0e1_greek_eta_L_tr1_i1 = fun_75386(e0e1_T_tr1,e0e1_A_Letad101_i1,e0e1_B_Letad101_i1,e0e1_C_Letad101_i1,e0e1_D_Letad101_i1,e0e1_E_Letad101_i1)
    e0e1_greek_rho_Ln_tr14_i1 = fun_75389(e0e1_T_tr14,e0e1_A_Lnrhod105_i1,e0e1_B_Lnrhod105_i1,e0e1_C_Lnrhod105_i1,e0e1_D_Lnrhod105_i1)
    e0e1_greek_eta_V_i2_tr20 = fun_75392(e0e1_T_tr20,e0e1_A_Vetad102_i2,e0e1_B_Vetad102_i2,e0e1_C_Vetad102_i2,e0e1_D_Vetad102_i2)
    e0e1_greek_rho_Ln_tr8_i2 = fun_75389(e0e1_T_tr8,e0e1_A_Lnrhod105_i2,e0e1_B_Lnrhod105_i2,e0e1_C_Lnrhod105_i2,e0e1_D_Lnrhod105_i2)
    e0_h_LVReboilern_i1 = fun_75390(e0_T_Reboiler,e0_A_LVhnd106_i1,e0_B_LVhnd106_i1,e0_C_LVhnd106_i1,e0_D_LVhnd106_i1,e0_E_LVhnd106_i1,e0_T_crit_i1)
    e0e1_greek_sigma_i1_tr1 = fun_75390(e0e1_T_tr1,e0e1_A_sigd106_i1,e0e1_B_sigd106_i1,e0e1_C_sigd106_i1,e0e1_D_sigd106_i1,e0e1_E_sigd106_i1,e0_T_crit_i1)
    e0e1_greek_eta_V_i2_tr15 = fun_75392(e0e1_T_tr15,e0e1_A_Vetad102_i2,e0e1_B_Vetad102_i2,e0e1_C_Vetad102_i2,e0e1_D_Vetad102_i2)
    e0e1_h_Vn_tr15_i1 = fun_93415(e0e1_T_tr15,e0e1_A_Vhnd127_i1,e0e1_B_Vhnd127_i1,e0e1_C_Vhnd127_i1,e0e1_D_Vhnd127_i1,e0e1_E_Vhnd127_i1,e0e1_F_Vhnd127_i1,e0e1_G_Vhnd127_i1,e0e1_T_ho_i1,e0e1_h_o_i1)
    e0e1_p_LV_tr15_i1 = fun_75386(e0e1_T_tr15,e0e1_A_LVpd101_i1,e0e1_B_LVpd101_i1,e0e1_C_LVpd101_i1,e0e1_D_LVpd101_i1,e0e1_E_LVpd101_i1)
    e0e1_greek_eta_L_tr18_i1 = fun_75386(e0e1_T_tr18,e0e1_A_Letad101_i1,e0e1_B_Letad101_i1,e0e1_C_Letad101_i1,e0e1_D_Letad101_i1,e0e1_E_Letad101_i1)
    e0_h_FVn_i1 = fun_93415(e0_T_F,e0_A_Vhnd127_i1,e0_B_Vhnd127_i1,e0_C_Vhnd127_i1,e0_D_Vhnd127_i1,e0_E_Vhnd127_i1,e0_F_Vhnd127_i1,e0_G_Vhnd127_i1,e0_T_ho_i1,e0_h_o_i1)
    e0e1_greek_eta_L_tr2_i2 = fun_75386(e0e1_T_tr2,e0e1_A_Letad101_i2,e0e1_B_Letad101_i2,e0e1_C_Letad101_i2,e0e1_D_Letad101_i2,e0e1_E_Letad101_i2)
    e0e1_greek_eta_L_tr10_i2 = fun_75386(e0e1_T_tr10,e0e1_A_Letad101_i2,e0e1_B_Letad101_i2,e0e1_C_Letad101_i2,e0e1_D_Letad101_i2,e0e1_E_Letad101_i2)
    e0e1_greek_eta_V_i1_tr14 = fun_75392(e0e1_T_tr14,e0e1_A_Vetad102_i1,e0e1_B_Vetad102_i1,e0e1_C_Vetad102_i1,e0e1_D_Vetad102_i1)
    e0e1_h_LVn_tr5_i2 = fun_75390(e0e1_T_tr5,e0e1_A_LVhnd106_i2,e0e1_B_LVhnd106_i2,e0e1_C_LVhnd106_i2,e0e1_D_LVhnd106_i2,e0e1_E_LVhnd106_i2,e0e1_T_crit_i2)
    e0e1_greek_sigma_W_tr16 = fun_75390(e0e1_T_tr16,e0e1_A_Wsigd106,e0e1_B_Wsigd106,e0e1_C_Wsigd106,e0e1_D_Wsigd106,e0e1_E_Wsigd106,e0e1_T_Wcrit)
    e0e1_h_Vn_tr14_i2 = fun_93415(e0e1_T_tr14,e0e1_A_Vhnd127_i2,e0e1_B_Vhnd127_i2,e0e1_C_Vhnd127_i2,e0e1_D_Vhnd127_i2,e0e1_E_Vhnd127_i2,e0e1_F_Vhnd127_i2,e0e1_G_Vhnd127_i2,e0e1_T_ho_i2,e0e1_h_o_i2)
    e0e1_greek_eta_V_i1_tr11 = fun_75392(e0e1_T_tr11,e0e1_A_Vetad102_i1,e0e1_B_Vetad102_i1,e0e1_C_Vetad102_i1,e0e1_D_Vetad102_i1)
    e0e1_F_Vm_tr6 = fun_94396(e0e1_F_Vn_tr6,e0e1_M_V_tr6)
    e0e1_greek_eta_V_i2_tr11 = fun_75392(e0e1_T_tr11,e0e1_A_Vetad102_i2,e0e1_B_Vetad102_i2,e0e1_C_Vetad102_i2,e0e1_D_Vetad102_i2)
    e0e1_greek_rho_BLn_i1 = fun_75389(e0e1_T_B,e0e1_A_Lnrhod105_i1,e0e1_B_Lnrhod105_i1,e0e1_C_Lnrhod105_i1,e0e1_D_Lnrhod105_i1)
    e0e1_greek_sigma_i1_tr11 = fun_75390(e0e1_T_tr11,e0e1_A_sigd106_i1,e0e1_B_sigd106_i1,e0e1_C_sigd106_i1,e0e1_D_sigd106_i1,e0e1_E_sigd106_i1,e0e1_T_crit_i1)
    e0e1_greek_eta_L_tr16_i2 = fun_75386(e0e1_T_tr16,e0e1_A_Letad101_i2,e0e1_B_Letad101_i2,e0e1_C_Letad101_i2,e0e1_D_Letad101_i2,e0e1_E_Letad101_i2)
    e0e1_greek_rho_Ln_tr3_i2 = fun_75389(e0e1_T_tr3,e0e1_A_Lnrhod105_i2,e0e1_B_Lnrhod105_i2,e0e1_C_Lnrhod105_i2,e0e1_D_Lnrhod105_i2)
    e0e1_h_Vn_tr8_i2 = fun_93415(e0e1_T_tr8,e0e1_A_Vhnd127_i2,e0e1_B_Vhnd127_i2,e0e1_C_Vhnd127_i2,e0e1_D_Vhnd127_i2,e0e1_E_Vhnd127_i2,e0e1_F_Vhnd127_i2,e0e1_G_Vhnd127_i2,e0e1_T_ho_i2,e0e1_h_o_i2)
    e0_p_DLV_i1 = fun_75386(e0_T_D,e0_A_LVpd101_i1,e0_B_LVpd101_i1,e0_C_LVpd101_i1,e0_D_LVpd101_i1,e0_E_LVpd101_i1)
    e0e1_h_LVn_tr1_i1 = fun_75390(e0e1_T_tr1,e0e1_A_LVhnd106_i1,e0e1_B_LVhnd106_i1,e0e1_C_LVhnd106_i1,e0e1_D_LVhnd106_i1,e0e1_E_LVhnd106_i1,e0e1_T_crit_i1)
    e0e1_h_LVn_tr9_i1 = fun_75390(e0e1_T_tr9,e0e1_A_LVhnd106_i1,e0e1_B_LVhnd106_i1,e0e1_C_LVhnd106_i1,e0e1_D_LVhnd106_i1,e0e1_E_LVhnd106_i1,e0e1_T_crit_i1)
    e0e1_greek_eta_V_i1_tr4 = fun_75392(e0e1_T_tr4,e0e1_A_Vetad102_i1,e0e1_B_Vetad102_i1,e0e1_C_Vetad102_i1,e0e1_D_Vetad102_i1)
    e0e1_greek_rho_Ln_tr11_i1 = fun_75389(e0e1_T_tr11,e0e1_A_Lnrhod105_i1,e0e1_B_Lnrhod105_i1,e0e1_C_Lnrhod105_i1,e0e1_D_Lnrhod105_i1)
    e0e1_greek_eta_V_i1_tr16 = fun_75392(e0e1_T_tr16,e0e1_A_Vetad102_i1,e0e1_B_Vetad102_i1,e0e1_C_Vetad102_i1,e0e1_D_Vetad102_i1)
    e0e1_greek_sigma_i2_tr10 = fun_75390(e0e1_T_tr10,e0e1_A_sigd106_i2,e0e1_B_sigd106_i2,e0e1_C_sigd106_i2,e0e1_D_sigd106_i2,e0e1_E_sigd106_i2,e0e1_T_crit_i2)
    e0e1_greek_eta_L_tr7_i2 = fun_75386(e0e1_T_tr7,e0e1_A_Letad101_i2,e0e1_B_Letad101_i2,e0e1_C_Letad101_i2,e0e1_D_Letad101_i2,e0e1_E_Letad101_i2)
    e0e1_h_LVn_tr16_i2 = fun_75390(e0e1_T_tr16,e0e1_A_LVhnd106_i2,e0e1_B_LVhnd106_i2,e0e1_C_LVhnd106_i2,e0e1_D_LVhnd106_i2,e0e1_E_LVhnd106_i2,e0e1_T_crit_i2)
    e0e1_greek_sigma_i1_tr19 = fun_75390(e0e1_T_tr19,e0e1_A_sigd106_i1,e0e1_B_sigd106_i1,e0e1_C_sigd106_i1,e0e1_D_sigd106_i1,e0e1_E_sigd106_i1,e0e1_T_crit_i1)
    e0e1_h_Vn_tr12_i2 = fun_93415(e0e1_T_tr12,e0e1_A_Vhnd127_i2,e0e1_B_Vhnd127_i2,e0e1_C_Vhnd127_i2,e0e1_D_Vhnd127_i2,e0e1_E_Vhnd127_i2,e0e1_F_Vhnd127_i2,e0e1_G_Vhnd127_i2,e0e1_T_ho_i2,e0e1_h_o_i2)
    e0e1_p_LV_tr18_i2 = fun_75386(e0e1_T_tr18,e0e1_A_LVpd101_i2,e0e1_B_LVpd101_i2,e0e1_C_LVpd101_i2,e0e1_D_LVpd101_i2,e0e1_E_LVpd101_i2)
    e0e1_greek_rho_Ln_tr2_i1 = fun_75389(e0e1_T_tr2,e0e1_A_Lnrhod105_i1,e0e1_B_Lnrhod105_i1,e0e1_C_Lnrhod105_i1,e0e1_D_Lnrhod105_i1)
    e0e1_h_LVn_tr1_i2 = fun_75390(e0e1_T_tr1,e0e1_A_LVhnd106_i2,e0e1_B_LVhnd106_i2,e0e1_C_LVhnd106_i2,e0e1_D_LVhnd106_i2,e0e1_E_LVhnd106_i2,e0e1_T_crit_i2)
    e0e1_p_LV_tr12_i1 = fun_75386(e0e1_T_tr12,e0e1_A_LVpd101_i1,e0e1_B_LVpd101_i1,e0e1_C_LVpd101_i1,e0e1_D_LVpd101_i1,e0e1_E_LVpd101_i1)
    e0e1_greek_eta_L_tr10_i1 = fun_75386(e0e1_T_tr10,e0e1_A_Letad101_i1,e0e1_B_Letad101_i1,e0e1_C_Letad101_i1,e0e1_D_Letad101_i1,e0e1_E_Letad101_i1)
    e0e1_greek_sigma_i1_tr17 = fun_75390(e0e1_T_tr17,e0e1_A_sigd106_i1,e0e1_B_sigd106_i1,e0e1_C_sigd106_i1,e0e1_D_sigd106_i1,e0e1_E_sigd106_i1,e0e1_T_crit_i1)
    e0e1_greek_eta_V_i1_tr9 = fun_75392(e0e1_T_tr9,e0e1_A_Vetad102_i1,e0e1_B_Vetad102_i1,e0e1_C_Vetad102_i1,e0e1_D_Vetad102_i1)
    e0e1_greek_eta_V_i2_tr8 = fun_75392(e0e1_T_tr8,e0e1_A_Vetad102_i2,e0e1_B_Vetad102_i2,e0e1_C_Vetad102_i2,e0e1_D_Vetad102_i2)
    e0e1_h_Vn_tr7_i1 = fun_93415(e0e1_T_tr7,e0e1_A_Vhnd127_i1,e0e1_B_Vhnd127_i1,e0e1_C_Vhnd127_i1,e0e1_D_Vhnd127_i1,e0e1_E_Vhnd127_i1,e0e1_F_Vhnd127_i1,e0e1_G_Vhnd127_i1,e0e1_T_ho_i1,e0e1_h_o_i1)
    e0e1_p_LV_tr6_i2 = fun_75386(e0e1_T_tr6,e0e1_A_LVpd101_i2,e0e1_B_LVpd101_i2,e0e1_C_LVpd101_i2,e0e1_D_LVpd101_i2,e0e1_E_LVpd101_i2)
    e0e1_greek_sigma_i1_tr5 = fun_75390(e0e1_T_tr5,e0e1_A_sigd106_i1,e0e1_B_sigd106_i1,e0e1_C_sigd106_i1,e0e1_D_sigd106_i1,e0e1_E_sigd106_i1,e0e1_T_crit_i1)
    e0_h_DVn_i1 = fun_93415(e0_T_D,e0_A_Vhnd127_i1,e0_B_Vhnd127_i1,e0_C_Vhnd127_i1,e0_D_Vhnd127_i1,e0_E_Vhnd127_i1,e0_F_Vhnd127_i1,e0_G_Vhnd127_i1,e0_T_ho_i1,e0_h_o_i1)
    e0e1_p_LV_tr9_i2 = fun_75386(e0e1_T_tr9,e0e1_A_LVpd101_i2,e0e1_B_LVpd101_i2,e0e1_C_LVpd101_i2,e0e1_D_LVpd101_i2,e0e1_E_LVpd101_i2)
    e0e1_greek_sigma_W_tr6 = fun_75390(e0e1_T_tr6,e0e1_A_Wsigd106,e0e1_B_Wsigd106,e0e1_C_Wsigd106,e0e1_D_Wsigd106,e0e1_E_Wsigd106,e0e1_T_Wcrit)
    e0e1_F_Vm_tr1 = fun_94396(e0_F_DVn,e0e1_M_V_tr1)
    e0e1_p_LV_tr17_i2 = fun_75386(e0e1_T_tr17,e0e1_A_LVpd101_i2,e0e1_B_LVpd101_i2,e0e1_C_LVpd101_i2,e0e1_D_LVpd101_i2,e0e1_E_LVpd101_i2)
    e0e1_greek_eta_L_tr16_i1 = fun_75386(e0e1_T_tr16,e0e1_A_Letad101_i1,e0e1_B_Letad101_i1,e0e1_C_Letad101_i1,e0e1_D_Letad101_i1,e0e1_E_Letad101_i1)
    e0_greek_rho_FLn_i2 = fun_75389(e0_T_F,e0_A_Lnrhod105_i2,e0_B_Lnrhod105_i2,e0_C_Lnrhod105_i2,e0_D_Lnrhod105_i2)
    e0e1_greek_sigma_i2_tr13 = fun_75390(e0e1_T_tr13,e0e1_A_sigd106_i2,e0e1_B_sigd106_i2,e0e1_C_sigd106_i2,e0e1_D_sigd106_i2,e0e1_E_sigd106_i2,e0e1_T_crit_i2)
    e0e1_greek_sigma_i2_tr18 = fun_75390(e0e1_T_tr18,e0e1_A_sigd106_i2,e0e1_B_sigd106_i2,e0e1_C_sigd106_i2,e0e1_D_sigd106_i2,e0e1_E_sigd106_i2,e0e1_T_crit_i2)
    e0e1_h_Vn_tr11_i2 = fun_93415(e0e1_T_tr11,e0e1_A_Vhnd127_i2,e0e1_B_Vhnd127_i2,e0e1_C_Vhnd127_i2,e0e1_D_Vhnd127_i2,e0e1_E_Vhnd127_i2,e0e1_F_Vhnd127_i2,e0e1_G_Vhnd127_i2,e0e1_T_ho_i2,e0e1_h_o_i2)
    e0e1_greek_sigma_i1_tr14 = fun_75390(e0e1_T_tr14,e0e1_A_sigd106_i1,e0e1_B_sigd106_i1,e0e1_C_sigd106_i1,e0e1_D_sigd106_i1,e0e1_E_sigd106_i1,e0e1_T_crit_i1)
    e0e1_greek_sigma_W_tr9 = fun_75390(e0e1_T_tr9,e0e1_A_Wsigd106,e0e1_B_Wsigd106,e0e1_C_Wsigd106,e0e1_D_Wsigd106,e0e1_E_Wsigd106,e0e1_T_Wcrit)
    e0e1_greek_sigma_i2_tr5 = fun_75390(e0e1_T_tr5,e0e1_A_sigd106_i2,e0e1_B_sigd106_i2,e0e1_C_sigd106_i2,e0e1_D_sigd106_i2,e0e1_E_sigd106_i2,e0e1_T_crit_i2)
    e0e1_p_LV_tr14_i1 = fun_75386(e0e1_T_tr14,e0e1_A_LVpd101_i1,e0e1_B_LVpd101_i1,e0e1_C_LVpd101_i1,e0e1_D_LVpd101_i1,e0e1_E_LVpd101_i1)
    e0_p_FLV_i2 = fun_75386(e0_T_F,e0_A_LVpd101_i2,e0_B_LVpd101_i2,e0_C_LVpd101_i2,e0_D_LVpd101_i2,e0_E_LVpd101_i2)
    e0e1_greek_rho_Ln_tr15_i2 = fun_75389(e0e1_T_tr15,e0e1_A_Lnrhod105_i2,e0e1_B_Lnrhod105_i2,e0e1_C_Lnrhod105_i2,e0e1_D_Lnrhod105_i2)
    e0e1_greek_rho_Ln_tr3_i1 = fun_75389(e0e1_T_tr3,e0e1_A_Lnrhod105_i1,e0e1_B_Lnrhod105_i1,e0e1_C_Lnrhod105_i1,e0e1_D_Lnrhod105_i1)
    e0_h_ReboilerVn_i1 = fun_93415(e0_T_Reboiler,e0_A_Vhnd127_i1,e0_B_Vhnd127_i1,e0_C_Vhnd127_i1,e0_D_Vhnd127_i1,e0_E_Vhnd127_i1,e0_F_Vhnd127_i1,e0_G_Vhnd127_i1,e0_T_ho_i1,e0_h_o_i1)
    e0e1_greek_eta_V_i1_tr18 = fun_75392(e0e1_T_tr18,e0e1_A_Vetad102_i1,e0e1_B_Vetad102_i1,e0e1_C_Vetad102_i1,e0e1_D_Vetad102_i1)
    e0e1_F_Lm_tr9 = fun_94396(e0e1_F_Ln_tr9,e0e1_M_L_tr9)
    e0_greek_rho_LReboilern_i2 = fun_75389(e0_T_Reboiler,e0_A_Lnrhod105_i2,e0_B_Lnrhod105_i2,e0_C_Lnrhod105_i2,e0_D_Lnrhod105_i2)
    e0e1_greek_rho_Ln_tr16_i2 = fun_75389(e0e1_T_tr16,e0e1_A_Lnrhod105_i2,e0e1_B_Lnrhod105_i2,e0e1_C_Lnrhod105_i2,e0e1_D_Lnrhod105_i2)
    e0e1_greek_sigma_W_tr10 = fun_75390(e0e1_T_tr10,e0e1_A_Wsigd106,e0e1_B_Wsigd106,e0e1_C_Wsigd106,e0e1_D_Wsigd106,e0e1_E_Wsigd106,e0e1_T_Wcrit)
    e0e1_greek_eta_V_i2_tr10 = fun_75392(e0e1_T_tr10,e0e1_A_Vetad102_i2,e0e1_B_Vetad102_i2,e0e1_C_Vetad102_i2,e0e1_D_Vetad102_i2)
    e0e1_greek_sigma_W_tr12 = fun_75390(e0e1_T_tr12,e0e1_A_Wsigd106,e0e1_B_Wsigd106,e0e1_C_Wsigd106,e0e1_D_Wsigd106,e0e1_E_Wsigd106,e0e1_T_Wcrit)
    e0e1_p_LV_tr20_i1 = fun_75386(e0e1_T_tr20,e0e1_A_LVpd101_i1,e0e1_B_LVpd101_i1,e0e1_C_LVpd101_i1,e0e1_D_LVpd101_i1,e0e1_E_LVpd101_i1)
    e0_p_LVReboiler_i2 = fun_75386(e0_T_Reboiler,e0_A_LVpd101_i2,e0_B_LVpd101_i2,e0_C_LVpd101_i2,e0_D_LVpd101_i2,e0_E_LVpd101_i2)
    e0e1_greek_eta_L_tr14_i1 = fun_75386(e0e1_T_tr14,e0e1_A_Letad101_i1,e0e1_B_Letad101_i1,e0e1_C_Letad101_i1,e0e1_D_Letad101_i1,e0e1_E_Letad101_i1)
    e0e1_p_LV_tr8_i1 = fun_75386(e0e1_T_tr8,e0e1_A_LVpd101_i1,e0e1_B_LVpd101_i1,e0e1_C_LVpd101_i1,e0e1_D_LVpd101_i1,e0e1_E_LVpd101_i1)
    e0e1_p_LV_tr16_i2 = fun_75386(e0e1_T_tr16,e0e1_A_LVpd101_i2,e0e1_B_LVpd101_i2,e0e1_C_LVpd101_i2,e0e1_D_LVpd101_i2,e0e1_E_LVpd101_i2)
    e0e1_greek_rho_Ln_tr15_i1 = fun_75389(e0e1_T_tr15,e0e1_A_Lnrhod105_i1,e0e1_B_Lnrhod105_i1,e0e1_C_Lnrhod105_i1,e0e1_D_Lnrhod105_i1)
    e0e1_greek_rho_Ln_tr17_i2 = fun_75389(e0e1_T_tr17,e0e1_A_Lnrhod105_i2,e0e1_B_Lnrhod105_i2,e0e1_C_Lnrhod105_i2,e0e1_D_Lnrhod105_i2)
    e0e1_p_BLV_i1 = fun_75386(e0e1_T_B,e0e1_A_LVpd101_i1,e0e1_B_LVpd101_i1,e0e1_C_LVpd101_i1,e0e1_D_LVpd101_i1,e0e1_E_LVpd101_i1)
    e0e1_greek_rho_Ln_tr1_i2 = fun_75389(e0e1_T_tr1,e0e1_A_Lnrhod105_i2,e0e1_B_Lnrhod105_i2,e0e1_C_Lnrhod105_i2,e0e1_D_Lnrhod105_i2)
    e0e1_greek_sigma_i2_tr15 = fun_75390(e0e1_T_tr15,e0e1_A_sigd106_i2,e0e1_B_sigd106_i2,e0e1_C_sigd106_i2,e0e1_D_sigd106_i2,e0e1_E_sigd106_i2,e0e1_T_crit_i2)
    e0e1_F_Vm_tr17 = fun_94396(e0e1_F_Vn_tr17,e0e1_M_V_tr17)
    e0e1_h_LVn_tr13_i2 = fun_75390(e0e1_T_tr13,e0e1_A_LVhnd106_i2,e0e1_B_LVhnd106_i2,e0e1_C_LVhnd106_i2,e0e1_D_LVhnd106_i2,e0e1_E_LVhnd106_i2,e0e1_T_crit_i2)
    e0e1_greek_sigma_i2_tr4 = fun_75390(e0e1_T_tr4,e0e1_A_sigd106_i2,e0e1_B_sigd106_i2,e0e1_C_sigd106_i2,e0e1_D_sigd106_i2,e0e1_E_sigd106_i2,e0e1_T_crit_i2)
    e0e1_h_LVn_tr4_i1 = fun_75390(e0e1_T_tr4,e0e1_A_LVhnd106_i1,e0e1_B_LVhnd106_i1,e0e1_C_LVhnd106_i1,e0e1_D_LVhnd106_i1,e0e1_E_LVhnd106_i1,e0e1_T_crit_i1)
    e0e1_h_BLVn_i1 = fun_75390(e0e1_T_B,e0e1_A_LVhnd106_i1,e0e1_B_LVhnd106_i1,e0e1_C_LVhnd106_i1,e0e1_D_LVhnd106_i1,e0e1_E_LVhnd106_i1,e0e1_T_crit_i1)
    e0e1_greek_eta_V_i2_tr18 = fun_75392(e0e1_T_tr18,e0e1_A_Vetad102_i2,e0e1_B_Vetad102_i2,e0e1_C_Vetad102_i2,e0e1_D_Vetad102_i2)
    e0e1_greek_rho_Ln_tr11_i2 = fun_75389(e0e1_T_tr11,e0e1_A_Lnrhod105_i2,e0e1_B_Lnrhod105_i2,e0e1_C_Lnrhod105_i2,e0e1_D_Lnrhod105_i2)
    e0e1_p_LV_tr9_i1 = fun_75386(e0e1_T_tr9,e0e1_A_LVpd101_i1,e0e1_B_LVpd101_i1,e0e1_C_LVpd101_i1,e0e1_D_LVpd101_i1,e0e1_E_LVpd101_i1)
    e0e1_greek_sigma_W_tr17 = fun_75390(e0e1_T_tr17,e0e1_A_Wsigd106,e0e1_B_Wsigd106,e0e1_C_Wsigd106,e0e1_D_Wsigd106,e0e1_E_Wsigd106,e0e1_T_Wcrit)
    e0e1_greek_eta_L_tr11_i1 = fun_75386(e0e1_T_tr11,e0e1_A_Letad101_i1,e0e1_B_Letad101_i1,e0e1_C_Letad101_i1,e0e1_D_Letad101_i1,e0e1_E_Letad101_i1)
    e0e1_h_LVn_tr8_i2 = fun_75390(e0e1_T_tr8,e0e1_A_LVhnd106_i2,e0e1_B_LVhnd106_i2,e0e1_C_LVhnd106_i2,e0e1_D_LVhnd106_i2,e0e1_E_LVhnd106_i2,e0e1_T_crit_i2)
    e0e1_p_LV_tr2_i1 = fun_75386(e0e1_T_tr2,e0e1_A_LVpd101_i1,e0e1_B_LVpd101_i1,e0e1_C_LVpd101_i1,e0e1_D_LVpd101_i1,e0e1_E_LVpd101_i1)
    e0e1_greek_rho_Ln_tr13_i1 = fun_75389(e0e1_T_tr13,e0e1_A_Lnrhod105_i1,e0e1_B_Lnrhod105_i1,e0e1_C_Lnrhod105_i1,e0e1_D_Lnrhod105_i1)
    e0e1_greek_eta_V_i2_tr19 = fun_75392(e0e1_T_tr19,e0e1_A_Vetad102_i2,e0e1_B_Vetad102_i2,e0e1_C_Vetad102_i2,e0e1_D_Vetad102_i2)
    e0e1_F_Lm_tr15 = fun_94396(e0e1_F_Ln_tr15,e0e1_M_L_tr15)
    e0e1_F_Lm_tr16 = fun_94396(e0e1_F_Ln_tr16,e0e1_M_L_tr16)
    e0e1_greek_sigma_i2_tr17 = fun_75390(e0e1_T_tr17,e0e1_A_sigd106_i2,e0e1_B_sigd106_i2,e0e1_C_sigd106_i2,e0e1_D_sigd106_i2,e0e1_E_sigd106_i2,e0e1_T_crit_i2)
    e0e1_greek_sigma_i1_tr20 = fun_75390(e0e1_T_tr20,e0e1_A_sigd106_i1,e0e1_B_sigd106_i1,e0e1_C_sigd106_i1,e0e1_D_sigd106_i1,e0e1_E_sigd106_i1,e0e1_T_crit_i1)
    e0e1_F_Vm_tr5 = fun_94396(e0e1_F_Vn_tr5,e0e1_M_V_tr5)
    e0_p_FLV_i1 = fun_75386(e0_T_F,e0_A_LVpd101_i1,e0_B_LVpd101_i1,e0_C_LVpd101_i1,e0_D_LVpd101_i1,e0_E_LVpd101_i1)
    e0e1_greek_rho_Ln_tr14_i2 = fun_75389(e0e1_T_tr14,e0e1_A_Lnrhod105_i2,e0e1_B_Lnrhod105_i2,e0e1_C_Lnrhod105_i2,e0e1_D_Lnrhod105_i2)
    e0e1_F_Lm_tr2 = fun_94396(e0e1_F_Ln_tr2,e0e1_M_L_tr2)
    e0e1_F_Lm_tr6 = fun_94396(e0e1_F_Ln_tr6,e0e1_M_L_tr6)
    e0e1_h_Vn_tr16_i2 = fun_93415(e0e1_T_tr16,e0e1_A_Vhnd127_i2,e0e1_B_Vhnd127_i2,e0e1_C_Vhnd127_i2,e0e1_D_Vhnd127_i2,e0e1_E_Vhnd127_i2,e0e1_F_Vhnd127_i2,e0e1_G_Vhnd127_i2,e0e1_T_ho_i2,e0e1_h_o_i2)
    e0e1_greek_sigma_W_tr1 = fun_75390(e0e1_T_tr1,e0e1_A_Wsigd106,e0e1_B_Wsigd106,e0e1_C_Wsigd106,e0e1_D_Wsigd106,e0e1_E_Wsigd106,e0e1_T_Wcrit)
    e0e1_p_LV_tr12_i2 = fun_75386(e0e1_T_tr12,e0e1_A_LVpd101_i2,e0e1_B_LVpd101_i2,e0e1_C_LVpd101_i2,e0e1_D_LVpd101_i2,e0e1_E_LVpd101_i2)
    e0e1_greek_rho_Ln_tr9_i1 = fun_75389(e0e1_T_tr9,e0e1_A_Lnrhod105_i1,e0e1_B_Lnrhod105_i1,e0e1_C_Lnrhod105_i1,e0e1_D_Lnrhod105_i1)
    e0e1_h_Vn_tr20_i2 = fun_93415(e0e1_T_tr20,e0e1_A_Vhnd127_i2,e0e1_B_Vhnd127_i2,e0e1_C_Vhnd127_i2,e0e1_D_Vhnd127_i2,e0e1_E_Vhnd127_i2,e0e1_F_Vhnd127_i2,e0e1_G_Vhnd127_i2,e0e1_T_ho_i2,e0e1_h_o_i2)
    e0e1_greek_sigma_i2_tr8 = fun_75390(e0e1_T_tr8,e0e1_A_sigd106_i2,e0e1_B_sigd106_i2,e0e1_C_sigd106_i2,e0e1_D_sigd106_i2,e0e1_E_sigd106_i2,e0e1_T_crit_i2)
    e0e1_F_Vm_tr7 = fun_94396(e0e1_F_Vn_tr7,e0e1_M_V_tr7)
    e0e1_F_Lm_tr20 = fun_94396(e0e1_F_Ln_tr20,e0e1_M_L_tr20)
    e0e1_h_LVn_tr13_i1 = fun_75390(e0e1_T_tr13,e0e1_A_LVhnd106_i1,e0e1_B_LVhnd106_i1,e0e1_C_LVhnd106_i1,e0e1_D_LVhnd106_i1,e0e1_E_LVhnd106_i1,e0e1_T_crit_i1)
    e0e1_greek_rho_Ln_tr18_i1 = fun_75389(e0e1_T_tr18,e0e1_A_Lnrhod105_i1,e0e1_B_Lnrhod105_i1,e0e1_C_Lnrhod105_i1,e0e1_D_Lnrhod105_i1)
    e0e1_greek_eta_L_tr1_i2 = fun_75386(e0e1_T_tr1,e0e1_A_Letad101_i2,e0e1_B_Letad101_i2,e0e1_C_Letad101_i2,e0e1_D_Letad101_i2,e0e1_E_Letad101_i2)
    e0e1_greek_rho_Ln_tr12_i2 = fun_75389(e0e1_T_tr12,e0e1_A_Lnrhod105_i2,e0e1_B_Lnrhod105_i2,e0e1_C_Lnrhod105_i2,e0e1_D_Lnrhod105_i2)
    e0e1_F_Vm_tr3 = fun_94396(e0e1_F_Vn_tr3,e0e1_M_V_tr3)
    e0e1_greek_sigma_i2_tr7 = fun_75390(e0e1_T_tr7,e0e1_A_sigd106_i2,e0e1_B_sigd106_i2,e0e1_C_sigd106_i2,e0e1_D_sigd106_i2,e0e1_E_sigd106_i2,e0e1_T_crit_i2)
    e0e1_h_Vn_tr1_i2 = fun_93415(e0e1_T_tr1,e0e1_A_Vhnd127_i2,e0e1_B_Vhnd127_i2,e0e1_C_Vhnd127_i2,e0e1_D_Vhnd127_i2,e0e1_E_Vhnd127_i2,e0e1_F_Vhnd127_i2,e0e1_G_Vhnd127_i2,e0e1_T_ho_i2,e0_h_o_i2)
    e0e1_greek_eta_L_tr20_i1 = fun_75386(e0e1_T_tr20,e0e1_A_Letad101_i1,e0e1_B_Letad101_i1,e0e1_C_Letad101_i1,e0e1_D_Letad101_i1,e0e1_E_Letad101_i1)
    e0e1_greek_eta_L_tr15_i1 = fun_75386(e0e1_T_tr15,e0e1_A_Letad101_i1,e0e1_B_Letad101_i1,e0e1_C_Letad101_i1,e0e1_D_Letad101_i1,e0e1_E_Letad101_i1)
    e0e1_h_Vn_tr3_i2 = fun_93415(e0e1_T_tr3,e0e1_A_Vhnd127_i2,e0e1_B_Vhnd127_i2,e0e1_C_Vhnd127_i2,e0e1_D_Vhnd127_i2,e0e1_E_Vhnd127_i2,e0e1_F_Vhnd127_i2,e0e1_G_Vhnd127_i2,e0e1_T_ho_i2,e0e1_h_o_i2)
    e0e1_greek_eta_V_i2_tr13 = fun_75392(e0e1_T_tr13,e0e1_A_Vetad102_i2,e0e1_B_Vetad102_i2,e0e1_C_Vetad102_i2,e0e1_D_Vetad102_i2)
    e0e1_F_Vm_tr12 = fun_94396(e0e1_F_Vn_tr12,e0e1_M_V_tr12)
    e0e1_p_LV_tr10_i1 = fun_75386(e0e1_T_tr10,e0e1_A_LVpd101_i1,e0e1_B_LVpd101_i1,e0e1_C_LVpd101_i1,e0e1_D_LVpd101_i1,e0e1_E_LVpd101_i1)
    e0e1_greek_sigma_i1_tr6 = fun_75390(e0e1_T_tr6,e0e1_A_sigd106_i1,e0e1_B_sigd106_i1,e0e1_C_sigd106_i1,e0e1_D_sigd106_i1,e0e1_E_sigd106_i1,e0e1_T_crit_i1)
    e0e1_F_Vm_tr14 = fun_94396(e0e1_F_Vn_tr14,e0e1_M_V_tr14)
    e0e1_h_Vn_tr18_i2 = fun_93415(e0e1_T_tr18,e0e1_A_Vhnd127_i2,e0e1_B_Vhnd127_i2,e0e1_C_Vhnd127_i2,e0e1_D_Vhnd127_i2,e0e1_E_Vhnd127_i2,e0e1_F_Vhnd127_i2,e0e1_G_Vhnd127_i2,e0e1_T_ho_i2,e0e1_h_o_i2)
    e0e1_p_LV_tr13_i2 = fun_75386(e0e1_T_tr13,e0e1_A_LVpd101_i2,e0e1_B_LVpd101_i2,e0e1_C_LVpd101_i2,e0e1_D_LVpd101_i2,e0e1_E_LVpd101_i2)
    e0e1_greek_sigma_i1_tr3 = fun_75390(e0e1_T_tr3,e0e1_A_sigd106_i1,e0e1_B_sigd106_i1,e0e1_C_sigd106_i1,e0e1_D_sigd106_i1,e0e1_E_sigd106_i1,e0e1_T_crit_i1)
    e0e1_greek_sigma_W_tr11 = fun_75390(e0e1_T_tr11,e0e1_A_Wsigd106,e0e1_B_Wsigd106,e0e1_C_Wsigd106,e0e1_D_Wsigd106,e0e1_E_Wsigd106,e0e1_T_Wcrit)
    e0e1_h_Vn_tr17_i2 = fun_93415(e0e1_T_tr17,e0e1_A_Vhnd127_i2,e0e1_B_Vhnd127_i2,e0e1_C_Vhnd127_i2,e0e1_D_Vhnd127_i2,e0e1_E_Vhnd127_i2,e0e1_F_Vhnd127_i2,e0e1_G_Vhnd127_i2,e0e1_T_ho_i2,e0e1_h_o_i2)
    e0e1_p_LV_tr3_i1 = fun_75386(e0e1_T_tr3,e0e1_A_LVpd101_i1,e0e1_B_LVpd101_i1,e0e1_C_LVpd101_i1,e0e1_D_LVpd101_i1,e0e1_E_LVpd101_i1)
    e0e1_greek_eta_V_i1_tr20 = fun_75392(e0e1_T_tr20,e0e1_A_Vetad102_i1,e0e1_B_Vetad102_i1,e0e1_C_Vetad102_i1,e0e1_D_Vetad102_i1)
    e0e1_greek_sigma_W_tr8 = fun_75390(e0e1_T_tr8,e0e1_A_Wsigd106,e0e1_B_Wsigd106,e0e1_C_Wsigd106,e0e1_D_Wsigd106,e0e1_E_Wsigd106,e0e1_T_Wcrit)
    e0e1_greek_eta_L_tr8_i2 = fun_75386(e0e1_T_tr8,e0e1_A_Letad101_i2,e0e1_B_Letad101_i2,e0e1_C_Letad101_i2,e0e1_D_Letad101_i2,e0e1_E_Letad101_i2)
    e0e1_greek_eta_L_tr2_i1 = fun_75386(e0e1_T_tr2,e0e1_A_Letad101_i1,e0e1_B_Letad101_i1,e0e1_C_Letad101_i1,e0e1_D_Letad101_i1,e0e1_E_Letad101_i1)
    e0e1_h_LVn_tr8_i1 = fun_75390(e0e1_T_tr8,e0e1_A_LVhnd106_i1,e0e1_B_LVhnd106_i1,e0e1_C_LVhnd106_i1,e0e1_D_LVhnd106_i1,e0e1_E_LVhnd106_i1,e0e1_T_crit_i1)
    e0e1_p_LV_tr3_i2 = fun_75386(e0e1_T_tr3,e0e1_A_LVpd101_i2,e0e1_B_LVpd101_i2,e0e1_C_LVpd101_i2,e0e1_D_LVpd101_i2,e0e1_E_LVpd101_i2)
    e0_p_DLV_i2 = fun_75386(e0_T_D,e0_A_LVpd101_i2,e0_B_LVpd101_i2,e0_C_LVpd101_i2,e0_D_LVpd101_i2,e0_E_LVpd101_i2)
    e0e1_greek_eta_V_i2_tr5 = fun_75392(e0e1_T_tr5,e0e1_A_Vetad102_i2,e0e1_B_Vetad102_i2,e0e1_C_Vetad102_i2,e0e1_D_Vetad102_i2)
    e0e1_greek_rho_Ln_tr1_i1 = fun_75389(e0e1_T_tr1,e0e1_A_Lnrhod105_i1,e0e1_B_Lnrhod105_i1,e0e1_C_Lnrhod105_i1,e0e1_D_Lnrhod105_i1)
    e0e1_h_Vn_tr10_i1 = fun_93415(e0e1_T_tr10,e0e1_A_Vhnd127_i1,e0e1_B_Vhnd127_i1,e0e1_C_Vhnd127_i1,e0e1_D_Vhnd127_i1,e0e1_E_Vhnd127_i1,e0e1_F_Vhnd127_i1,e0e1_G_Vhnd127_i1,e0e1_T_ho_i1,e0e1_h_o_i1)
    e0e1_p_LV_tr8_i2 = fun_75386(e0e1_T_tr8,e0e1_A_LVpd101_i2,e0e1_B_LVpd101_i2,e0e1_C_LVpd101_i2,e0e1_D_LVpd101_i2,e0e1_E_LVpd101_i2)
    e0e1_greek_sigma_i1_tr13 = fun_75390(e0e1_T_tr13,e0e1_A_sigd106_i1,e0e1_B_sigd106_i1,e0e1_C_sigd106_i1,e0e1_D_sigd106_i1,e0e1_E_sigd106_i1,e0e1_T_crit_i1)
    e0e1_p_LV_tr19_i2 = fun_75386(e0e1_T_tr19,e0e1_A_LVpd101_i2,e0e1_B_LVpd101_i2,e0e1_C_LVpd101_i2,e0e1_D_LVpd101_i2,e0e1_E_LVpd101_i2)
    e0e1_greek_eta_V_i2_tr9 = fun_75392(e0e1_T_tr9,e0e1_A_Vetad102_i2,e0e1_B_Vetad102_i2,e0e1_C_Vetad102_i2,e0e1_D_Vetad102_i2)
    e0_greek_rho_FLn_i1 = fun_75389(e0_T_F,e0_A_Lnrhod105_i1,e0_B_Lnrhod105_i1,e0_C_Lnrhod105_i1,e0_D_Lnrhod105_i1)
    e0e1_greek_sigma_i2_tr16 = fun_75390(e0e1_T_tr16,e0e1_A_sigd106_i2,e0e1_B_sigd106_i2,e0e1_C_sigd106_i2,e0e1_D_sigd106_i2,e0e1_E_sigd106_i2,e0e1_T_crit_i2)
    e0e1_greek_eta_L_tr8_i1 = fun_75386(e0e1_T_tr8,e0e1_A_Letad101_i1,e0e1_B_Letad101_i1,e0e1_C_Letad101_i1,e0e1_D_Letad101_i1,e0e1_E_Letad101_i1)
    e0e1_greek_eta_V_i1_tr5 = fun_75392(e0e1_T_tr5,e0e1_A_Vetad102_i1,e0e1_B_Vetad102_i1,e0e1_C_Vetad102_i1,e0e1_D_Vetad102_i1)
    e0e1_h_Vn_tr10_i2 = fun_93415(e0e1_T_tr10,e0e1_A_Vhnd127_i2,e0e1_B_Vhnd127_i2,e0e1_C_Vhnd127_i2,e0e1_D_Vhnd127_i2,e0e1_E_Vhnd127_i2,e0e1_F_Vhnd127_i2,e0e1_G_Vhnd127_i2,e0e1_T_ho_i2,e0e1_h_o_i2)
    e0e1_greek_eta_L_tr17_i2 = fun_75386(e0e1_T_tr17,e0e1_A_Letad101_i2,e0e1_B_Letad101_i2,e0e1_C_Letad101_i2,e0e1_D_Letad101_i2,e0e1_E_Letad101_i2)
    e0_h_DLVn_i2 = fun_75390(e0_T_D,e0_A_LVhnd106_i2,e0_B_LVhnd106_i2,e0_C_LVhnd106_i2,e0_D_LVhnd106_i2,e0_E_LVhnd106_i2,e0_T_crit_i2)
    e0e1_greek_sigma_i2_tr1 = fun_75390(e0e1_T_tr1,e0e1_A_sigd106_i2,e0e1_B_sigd106_i2,e0e1_C_sigd106_i2,e0e1_D_sigd106_i2,e0e1_E_sigd106_i2,e0_T_crit_i2)
    e0e1_h_LVn_tr15_i2 = fun_75390(e0e1_T_tr15,e0e1_A_LVhnd106_i2,e0e1_B_LVhnd106_i2,e0e1_C_LVhnd106_i2,e0e1_D_LVhnd106_i2,e0e1_E_LVhnd106_i2,e0e1_T_crit_i2)
    e0e1_greek_sigma_i2_tr11 = fun_75390(e0e1_T_tr11,e0e1_A_sigd106_i2,e0e1_B_sigd106_i2,e0e1_C_sigd106_i2,e0e1_D_sigd106_i2,e0e1_E_sigd106_i2,e0e1_T_crit_i2)
    e0e1_p_BLV_i2 = fun_75386(e0e1_T_B,e0e1_A_LVpd101_i2,e0e1_B_LVpd101_i2,e0e1_C_LVpd101_i2,e0e1_D_LVpd101_i2,e0e1_E_LVpd101_i2)
    e0e1_greek_sigma_i2_tr20 = fun_75390(e0e1_T_tr20,e0e1_A_sigd106_i2,e0e1_B_sigd106_i2,e0e1_C_sigd106_i2,e0e1_D_sigd106_i2,e0e1_E_sigd106_i2,e0e1_T_crit_i2)
    e0e1_F_Vm_tr4 = fun_94396(e0e1_F_Vn_tr4,e0e1_M_V_tr4)
    e0e1_p_LV_tr20_i2 = fun_75386(e0e1_T_tr20,e0e1_A_LVpd101_i2,e0e1_B_LVpd101_i2,e0e1_C_LVpd101_i2,e0e1_D_LVpd101_i2,e0e1_E_LVpd101_i2)
    e0e1_greek_sigma_i1_tr9 = fun_75390(e0e1_T_tr9,e0e1_A_sigd106_i1,e0e1_B_sigd106_i1,e0e1_C_sigd106_i1,e0e1_D_sigd106_i1,e0e1_E_sigd106_i1,e0e1_T_crit_i1)
    e0e1_greek_eta_L_tr15_i2 = fun_75386(e0e1_T_tr15,e0e1_A_Letad101_i2,e0e1_B_Letad101_i2,e0e1_C_Letad101_i2,e0e1_D_Letad101_i2,e0e1_E_Letad101_i2)
    e0e1_h_LVn_tr15_i1 = fun_75390(e0e1_T_tr15,e0e1_A_LVhnd106_i1,e0e1_B_LVhnd106_i1,e0e1_C_LVhnd106_i1,e0e1_D_LVhnd106_i1,e0e1_E_LVhnd106_i1,e0e1_T_crit_i1)
    e0e1_greek_sigma_i2_tr9 = fun_75390(e0e1_T_tr9,e0e1_A_sigd106_i2,e0e1_B_sigd106_i2,e0e1_C_sigd106_i2,e0e1_D_sigd106_i2,e0e1_E_sigd106_i2,e0e1_T_crit_i2)
    e0e1_greek_rho_Ln_tr19_i2 = fun_75389(e0e1_T_tr19,e0e1_A_Lnrhod105_i2,e0e1_B_Lnrhod105_i2,e0e1_C_Lnrhod105_i2,e0e1_D_Lnrhod105_i2)
    e0e1_greek_sigma_W_tr5 = fun_75390(e0e1_T_tr5,e0e1_A_Wsigd106,e0e1_B_Wsigd106,e0e1_C_Wsigd106,e0e1_D_Wsigd106,e0e1_E_Wsigd106,e0e1_T_Wcrit)
    e0e1_p_LV_tr11_i1 = fun_75386(e0e1_T_tr11,e0e1_A_LVpd101_i1,e0e1_B_LVpd101_i1,e0e1_C_LVpd101_i1,e0e1_D_LVpd101_i1,e0e1_E_LVpd101_i1)
    e0e1_p_LV_tr15_i2 = fun_75386(e0e1_T_tr15,e0e1_A_LVpd101_i2,e0e1_B_LVpd101_i2,e0e1_C_LVpd101_i2,e0e1_D_LVpd101_i2,e0e1_E_LVpd101_i2)
    e0e1_greek_sigma_i2_tr14 = fun_75390(e0e1_T_tr14,e0e1_A_sigd106_i2,e0e1_B_sigd106_i2,e0e1_C_sigd106_i2,e0e1_D_sigd106_i2,e0e1_E_sigd106_i2,e0e1_T_crit_i2)
    e0e1_greek_eta_L_tr3_i1 = fun_75386(e0e1_T_tr3,e0e1_A_Letad101_i1,e0e1_B_Letad101_i1,e0e1_C_Letad101_i1,e0e1_D_Letad101_i1,e0e1_E_Letad101_i1)
    e0e1_greek_sigma_W_tr18 = fun_75390(e0e1_T_tr18,e0e1_A_Wsigd106,e0e1_B_Wsigd106,e0e1_C_Wsigd106,e0e1_D_Wsigd106,e0e1_E_Wsigd106,e0e1_T_Wcrit)
    e0e1_h_LVn_tr18_i2 = fun_75390(e0e1_T_tr18,e0e1_A_LVhnd106_i2,e0e1_B_LVhnd106_i2,e0e1_C_LVhnd106_i2,e0e1_D_LVhnd106_i2,e0e1_E_LVhnd106_i2,e0e1_T_crit_i2)
    e0e1_h_LVn_tr19_i2 = fun_75390(e0e1_T_tr19,e0e1_A_LVhnd106_i2,e0e1_B_LVhnd106_i2,e0e1_C_LVhnd106_i2,e0e1_D_LVhnd106_i2,e0e1_E_LVhnd106_i2,e0e1_T_crit_i2)
    e0_h_DVn_i2 = fun_93415(e0_T_D,e0_A_Vhnd127_i2,e0_B_Vhnd127_i2,e0_C_Vhnd127_i2,e0_D_Vhnd127_i2,e0_E_Vhnd127_i2,e0_F_Vhnd127_i2,e0_G_Vhnd127_i2,e0_T_ho_i2,e0_h_o_i2)
    e0e1_greek_rho_Ln_tr20_i2 = fun_75389(e0e1_T_tr20,e0e1_A_Lnrhod105_i2,e0e1_B_Lnrhod105_i2,e0e1_C_Lnrhod105_i2,e0e1_D_Lnrhod105_i2)
    e0e1_greek_sigma_i2_tr12 = fun_75390(e0e1_T_tr12,e0e1_A_sigd106_i2,e0e1_B_sigd106_i2,e0e1_C_sigd106_i2,e0e1_D_sigd106_i2,e0e1_E_sigd106_i2,e0e1_T_crit_i2)
    e0e1_h_LVn_tr14_i1 = fun_75390(e0e1_T_tr14,e0e1_A_LVhnd106_i1,e0e1_B_LVhnd106_i1,e0e1_C_LVhnd106_i1,e0e1_D_LVhnd106_i1,e0e1_E_LVhnd106_i1,e0e1_T_crit_i1)
    e0_h_FLVn_i1 = fun_75390(e0_T_F,e0_A_LVhnd106_i1,e0_B_LVhnd106_i1,e0_C_LVhnd106_i1,e0_D_LVhnd106_i1,e0_E_LVhnd106_i1,e0_T_crit_i1)
    e0e1_h_LVn_tr4_i2 = fun_75390(e0e1_T_tr4,e0e1_A_LVhnd106_i2,e0e1_B_LVhnd106_i2,e0e1_C_LVhnd106_i2,e0e1_D_LVhnd106_i2,e0e1_E_LVhnd106_i2,e0e1_T_crit_i2)
    e0e1_greek_rho_Ln_tr4_i2 = fun_75389(e0e1_T_tr4,e0e1_A_Lnrhod105_i2,e0e1_B_Lnrhod105_i2,e0e1_C_Lnrhod105_i2,e0e1_D_Lnrhod105_i2)
    e0e1_h_LVn_tr20_i2 = fun_75390(e0e1_T_tr20,e0e1_A_LVhnd106_i2,e0e1_B_LVhnd106_i2,e0e1_C_LVhnd106_i2,e0e1_D_LVhnd106_i2,e0e1_E_LVhnd106_i2,e0e1_T_crit_i2)
    e0e1_p_LV_tr4_i1 = fun_75386(e0e1_T_tr4,e0e1_A_LVpd101_i1,e0e1_B_LVpd101_i1,e0e1_C_LVpd101_i1,e0e1_D_LVpd101_i1,e0e1_E_LVpd101_i1)
    e0e1_h_LVn_tr19_i1 = fun_75390(e0e1_T_tr19,e0e1_A_LVhnd106_i1,e0e1_B_LVhnd106_i1,e0e1_C_LVhnd106_i1,e0e1_D_LVhnd106_i1,e0e1_E_LVhnd106_i1,e0e1_T_crit_i1)
    e0e1_p_LV_tr19_i1 = fun_75386(e0e1_T_tr19,e0e1_A_LVpd101_i1,e0e1_B_LVpd101_i1,e0e1_C_LVpd101_i1,e0e1_D_LVpd101_i1,e0e1_E_LVpd101_i1)
    e0e1_F_Lm_tr11 = fun_94396(e0e1_F_Ln_tr11,e0e1_M_L_tr11)
    e0e1_greek_eta_V_i2_tr1 = fun_75392(e0e1_T_tr1,e0e1_A_Vetad102_i2,e0e1_B_Vetad102_i2,e0e1_C_Vetad102_i2,e0e1_D_Vetad102_i2)
    e0e1_F_Lm_tr5 = fun_94396(e0e1_F_Ln_tr5,e0e1_M_L_tr5)
    e0e1_h_BVn_i2 = fun_93415(e0e1_T_B,e0e1_A_Vhnd127_i2,e0e1_B_Vhnd127_i2,e0e1_C_Vhnd127_i2,e0e1_D_Vhnd127_i2,e0e1_E_Vhnd127_i2,e0e1_F_Vhnd127_i2,e0e1_G_Vhnd127_i2,e0e1_T_ho_i2,e0e1_h_o_i2)
    e0e1_greek_eta_L_tr19_i2 = fun_75386(e0e1_T_tr19,e0e1_A_Letad101_i2,e0e1_B_Letad101_i2,e0e1_C_Letad101_i2,e0e1_D_Letad101_i2,e0e1_E_Letad101_i2)
    e0e1_greek_eta_V_i1_tr13 = fun_75392(e0e1_T_tr13,e0e1_A_Vetad102_i1,e0e1_B_Vetad102_i1,e0e1_C_Vetad102_i1,e0e1_D_Vetad102_i1)
    e0e1_F_Lm_tr13 = fun_94396(e0e1_F_Ln_tr13,e0e1_M_L_tr13)
    e0e1_greek_sigma_i2_tr19 = fun_75390(e0e1_T_tr19,e0e1_A_sigd106_i2,e0e1_B_sigd106_i2,e0e1_C_sigd106_i2,e0e1_D_sigd106_i2,e0e1_E_sigd106_i2,e0e1_T_crit_i2)
    e0e1_F_Lm_tr18 = fun_94396(e0e1_F_Ln_tr18,e0e1_M_L_tr18)
    e0e1_h_Vn_tr9_i2 = fun_93415(e0e1_T_tr9,e0e1_A_Vhnd127_i2,e0e1_B_Vhnd127_i2,e0e1_C_Vhnd127_i2,e0e1_D_Vhnd127_i2,e0e1_E_Vhnd127_i2,e0e1_F_Vhnd127_i2,e0e1_G_Vhnd127_i2,e0e1_T_ho_i2,e0e1_h_o_i2)
    e0e1_greek_rho_Ln_tr7_i1 = fun_75389(e0e1_T_tr7,e0e1_A_Lnrhod105_i1,e0e1_B_Lnrhod105_i1,e0e1_C_Lnrhod105_i1,e0e1_D_Lnrhod105_i1)
    e0e1_greek_eta_L_tr9_i2 = fun_75386(e0e1_T_tr9,e0e1_A_Letad101_i2,e0e1_B_Letad101_i2,e0e1_C_Letad101_i2,e0e1_D_Letad101_i2,e0e1_E_Letad101_i2)
    e0e1_greek_eta_L_tr20_i2 = fun_75386(e0e1_T_tr20,e0e1_A_Letad101_i2,e0e1_B_Letad101_i2,e0e1_C_Letad101_i2,e0e1_D_Letad101_i2,e0e1_E_Letad101_i2)
    e0e1_greek_sigma_i1_tr8 = fun_75390(e0e1_T_tr8,e0e1_A_sigd106_i1,e0e1_B_sigd106_i1,e0e1_C_sigd106_i1,e0e1_D_sigd106_i1,e0e1_E_sigd106_i1,e0e1_T_crit_i1)
    e0e1_greek_eta_V_i2_tr3 = fun_75392(e0e1_T_tr3,e0e1_A_Vetad102_i2,e0e1_B_Vetad102_i2,e0e1_C_Vetad102_i2,e0e1_D_Vetad102_i2)
    e0e1_h_Vn_tr12_i1 = fun_93415(e0e1_T_tr12,e0e1_A_Vhnd127_i1,e0e1_B_Vhnd127_i1,e0e1_C_Vhnd127_i1,e0e1_D_Vhnd127_i1,e0e1_E_Vhnd127_i1,e0e1_F_Vhnd127_i1,e0e1_G_Vhnd127_i1,e0e1_T_ho_i1,e0e1_h_o_i1)
    e0e1_h_Vn_tr4_i1 = fun_93415(e0e1_T_tr4,e0e1_A_Vhnd127_i1,e0e1_B_Vhnd127_i1,e0e1_C_Vhnd127_i1,e0e1_D_Vhnd127_i1,e0e1_E_Vhnd127_i1,e0e1_F_Vhnd127_i1,e0e1_G_Vhnd127_i1,e0e1_T_ho_i1,e0e1_h_o_i1)
    e0e1_F_Lm_tr1 = fun_94396(e0e1_F_Ln_tr1,e0e1_M_L_tr1)
    e0e1_greek_eta_L_tr3_i2 = fun_75386(e0e1_T_tr3,e0e1_A_Letad101_i2,e0e1_B_Letad101_i2,e0e1_C_Letad101_i2,e0e1_D_Letad101_i2,e0e1_E_Letad101_i2)
    e0e1_greek_rho_Ln_tr13_i2 = fun_75389(e0e1_T_tr13,e0e1_A_Lnrhod105_i2,e0e1_B_Lnrhod105_i2,e0e1_C_Lnrhod105_i2,e0e1_D_Lnrhod105_i2)
    e0e1_h_Vn_tr14_i1 = fun_93415(e0e1_T_tr14,e0e1_A_Vhnd127_i1,e0e1_B_Vhnd127_i1,e0e1_C_Vhnd127_i1,e0e1_D_Vhnd127_i1,e0e1_E_Vhnd127_i1,e0e1_F_Vhnd127_i1,e0e1_G_Vhnd127_i1,e0e1_T_ho_i1,e0e1_h_o_i1)
    e0e1_greek_eta_V_i2_tr7 = fun_75392(e0e1_T_tr7,e0e1_A_Vetad102_i2,e0e1_B_Vetad102_i2,e0e1_C_Vetad102_i2,e0e1_D_Vetad102_i2)
    e0e1_greek_rho_Ln_tr10_i1 = fun_75389(e0e1_T_tr10,e0e1_A_Lnrhod105_i1,e0e1_B_Lnrhod105_i1,e0e1_C_Lnrhod105_i1,e0e1_D_Lnrhod105_i1)
    e0e1_h_Vn_tr9_i1 = fun_93415(e0e1_T_tr9,e0e1_A_Vhnd127_i1,e0e1_B_Vhnd127_i1,e0e1_C_Vhnd127_i1,e0e1_D_Vhnd127_i1,e0e1_E_Vhnd127_i1,e0e1_F_Vhnd127_i1,e0e1_G_Vhnd127_i1,e0e1_T_ho_i1,e0e1_h_o_i1)
    e0e1_greek_eta_L_tr6_i1 = fun_75386(e0e1_T_tr6,e0e1_A_Letad101_i1,e0e1_B_Letad101_i1,e0e1_C_Letad101_i1,e0e1_D_Letad101_i1,e0e1_E_Letad101_i1)
    e0e1_greek_eta_V_i2_tr16 = fun_75392(e0e1_T_tr16,e0e1_A_Vetad102_i2,e0e1_B_Vetad102_i2,e0e1_C_Vetad102_i2,e0e1_D_Vetad102_i2)
    e0e1_F_Lm_tr10 = fun_94396(e0e1_F_Ln_tr10,e0e1_M_L_tr10)
    e0e1_h_Vn_tr18_i1 = fun_93415(e0e1_T_tr18,e0e1_A_Vhnd127_i1,e0e1_B_Vhnd127_i1,e0e1_C_Vhnd127_i1,e0e1_D_Vhnd127_i1,e0e1_E_Vhnd127_i1,e0e1_F_Vhnd127_i1,e0e1_G_Vhnd127_i1,e0e1_T_ho_i1,e0e1_h_o_i1)
    e0e1_h_Vn_tr11_i1 = fun_93415(e0e1_T_tr11,e0e1_A_Vhnd127_i1,e0e1_B_Vhnd127_i1,e0e1_C_Vhnd127_i1,e0e1_D_Vhnd127_i1,e0e1_E_Vhnd127_i1,e0e1_F_Vhnd127_i1,e0e1_G_Vhnd127_i1,e0e1_T_ho_i1,e0e1_h_o_i1)
    e0e1_F_Vm_tr10 = fun_94396(e0e1_F_Vn_tr10,e0e1_M_V_tr10)
    e0e1_greek_sigma_i1_tr10 = fun_75390(e0e1_T_tr10,e0e1_A_sigd106_i1,e0e1_B_sigd106_i1,e0e1_C_sigd106_i1,e0e1_D_sigd106_i1,e0e1_E_sigd106_i1,e0e1_T_crit_i1)
    e0e1_p_LV_tr7_i1 = fun_75386(e0e1_T_tr7,e0e1_A_LVpd101_i1,e0e1_B_LVpd101_i1,e0e1_C_LVpd101_i1,e0e1_D_LVpd101_i1,e0e1_E_LVpd101_i1)
    e0e1_greek_sigma_W_tr14 = fun_75390(e0e1_T_tr14,e0e1_A_Wsigd106,e0e1_B_Wsigd106,e0e1_C_Wsigd106,e0e1_D_Wsigd106,e0e1_E_Wsigd106,e0e1_T_Wcrit)
    e0e1_greek_sigma_i1_tr16 = fun_75390(e0e1_T_tr16,e0e1_A_sigd106_i1,e0e1_B_sigd106_i1,e0e1_C_sigd106_i1,e0e1_D_sigd106_i1,e0e1_E_sigd106_i1,e0e1_T_crit_i1)
    e0e1_h_Vn_tr19_i1 = fun_93415(e0e1_T_tr19,e0e1_A_Vhnd127_i1,e0e1_B_Vhnd127_i1,e0e1_C_Vhnd127_i1,e0e1_D_Vhnd127_i1,e0e1_E_Vhnd127_i1,e0e1_F_Vhnd127_i1,e0e1_G_Vhnd127_i1,e0e1_T_ho_i1,e0e1_h_o_i1)
    e0e1_F_Vm_tr13 = fun_94396(e0e1_F_Vn_tr13,e0e1_M_V_tr13)
    e0_h_LVReboilern_i2 = fun_75390(e0_T_Reboiler,e0_A_LVhnd106_i2,e0_B_LVhnd106_i2,e0_C_LVhnd106_i2,e0_D_LVhnd106_i2,e0_E_LVhnd106_i2,e0_T_crit_i2)
    e0e1_p_LV_tr16_i1 = fun_75386(e0e1_T_tr16,e0e1_A_LVpd101_i1,e0e1_B_LVpd101_i1,e0e1_C_LVpd101_i1,e0e1_D_LVpd101_i1,e0e1_E_LVpd101_i1)
    e0e1_greek_rho_Ln_tr20_i1 = fun_75389(e0e1_T_tr20,e0e1_A_Lnrhod105_i1,e0e1_B_Lnrhod105_i1,e0e1_C_Lnrhod105_i1,e0e1_D_Lnrhod105_i1)
    e0e1_p_LV_tr14_i2 = fun_75386(e0e1_T_tr14,e0e1_A_LVpd101_i2,e0e1_B_LVpd101_i2,e0e1_C_LVpd101_i2,e0e1_D_LVpd101_i2,e0e1_E_LVpd101_i2)
    e0e1_greek_rho_Ln_tr7_i2 = fun_75389(e0e1_T_tr7,e0e1_A_Lnrhod105_i2,e0e1_B_Lnrhod105_i2,e0e1_C_Lnrhod105_i2,e0e1_D_Lnrhod105_i2)
    e0e1_greek_sigma_i1_tr7 = fun_75390(e0e1_T_tr7,e0e1_A_sigd106_i1,e0e1_B_sigd106_i1,e0e1_C_sigd106_i1,e0e1_D_sigd106_i1,e0e1_E_sigd106_i1,e0e1_T_crit_i1)
    e0e1_greek_sigma_W_tr7 = fun_75390(e0e1_T_tr7,e0e1_A_Wsigd106,e0e1_B_Wsigd106,e0e1_C_Wsigd106,e0e1_D_Wsigd106,e0e1_E_Wsigd106,e0e1_T_Wcrit)
    e0e1_p_LV_tr13_i1 = fun_75386(e0e1_T_tr13,e0e1_A_LVpd101_i1,e0e1_B_LVpd101_i1,e0e1_C_LVpd101_i1,e0e1_D_LVpd101_i1,e0e1_E_LVpd101_i1)
    e0e1_greek_eta_L_tr17_i1 = fun_75386(e0e1_T_tr17,e0e1_A_Letad101_i1,e0e1_B_Letad101_i1,e0e1_C_Letad101_i1,e0e1_D_Letad101_i1,e0e1_E_Letad101_i1)
    e0e1_greek_eta_L_tr5_i1 = fun_75386(e0e1_T_tr5,e0e1_A_Letad101_i1,e0e1_B_Letad101_i1,e0e1_C_Letad101_i1,e0e1_D_Letad101_i1,e0e1_E_Letad101_i1)
    e0e1_p_LV_tr5_i1 = fun_75386(e0e1_T_tr5,e0e1_A_LVpd101_i1,e0e1_B_LVpd101_i1,e0e1_C_LVpd101_i1,e0e1_D_LVpd101_i1,e0e1_E_LVpd101_i1)
    e0e1_h_Vn_tr3_i1 = fun_93415(e0e1_T_tr3,e0e1_A_Vhnd127_i1,e0e1_B_Vhnd127_i1,e0e1_C_Vhnd127_i1,e0e1_D_Vhnd127_i1,e0e1_E_Vhnd127_i1,e0e1_F_Vhnd127_i1,e0e1_G_Vhnd127_i1,e0e1_T_ho_i1,e0e1_h_o_i1)
    e0e1_greek_rho_Ln_tr2_i2 = fun_75389(e0e1_T_tr2,e0e1_A_Lnrhod105_i2,e0e1_B_Lnrhod105_i2,e0e1_C_Lnrhod105_i2,e0e1_D_Lnrhod105_i2)
    e0e1_h_LVn_tr2_i1 = fun_75390(e0e1_T_tr2,e0e1_A_LVhnd106_i1,e0e1_B_LVhnd106_i1,e0e1_C_LVhnd106_i1,e0e1_D_LVhnd106_i1,e0e1_E_LVhnd106_i1,e0e1_T_crit_i1)
    e0e1_F_Vm_tr11 = fun_94396(e0e1_F_Vn_tr11,e0e1_M_V_tr11)
    e0e1_greek_sigma_i2_tr6 = fun_75390(e0e1_T_tr6,e0e1_A_sigd106_i2,e0e1_B_sigd106_i2,e0e1_C_sigd106_i2,e0e1_D_sigd106_i2,e0e1_E_sigd106_i2,e0e1_T_crit_i2)
    e0e1_greek_eta_V_i2_tr17 = fun_75392(e0e1_T_tr17,e0e1_A_Vetad102_i2,e0e1_B_Vetad102_i2,e0e1_C_Vetad102_i2,e0e1_D_Vetad102_i2)
    e0e1_greek_eta_L_tr9_i1 = fun_75386(e0e1_T_tr9,e0e1_A_Letad101_i1,e0e1_B_Letad101_i1,e0e1_C_Letad101_i1,e0e1_D_Letad101_i1,e0e1_E_Letad101_i1)
    e0e1_greek_eta_L_tr19_i1 = fun_75386(e0e1_T_tr19,e0e1_A_Letad101_i1,e0e1_B_Letad101_i1,e0e1_C_Letad101_i1,e0e1_D_Letad101_i1,e0e1_E_Letad101_i1)
    e0e1_greek_eta_V_i1_tr10 = fun_75392(e0e1_T_tr10,e0e1_A_Vetad102_i1,e0e1_B_Vetad102_i1,e0e1_C_Vetad102_i1,e0e1_D_Vetad102_i1)
    e0e1_h_LVn_tr9_i2 = fun_75390(e0e1_T_tr9,e0e1_A_LVhnd106_i2,e0e1_B_LVhnd106_i2,e0e1_C_LVhnd106_i2,e0e1_D_LVhnd106_i2,e0e1_E_LVhnd106_i2,e0e1_T_crit_i2)
    e0e1_greek_sigma_W_tr2 = fun_75390(e0e1_T_tr2,e0e1_A_Wsigd106,e0e1_B_Wsigd106,e0e1_C_Wsigd106,e0e1_D_Wsigd106,e0e1_E_Wsigd106,e0e1_T_Wcrit)
    e0e1_greek_sigma_i1_tr4 = fun_75390(e0e1_T_tr4,e0e1_A_sigd106_i1,e0e1_B_sigd106_i1,e0e1_C_sigd106_i1,e0e1_D_sigd106_i1,e0e1_E_sigd106_i1,e0e1_T_crit_i1)
    e0_h_FVn_i2 = fun_93415(e0_T_F,e0_A_Vhnd127_i2,e0_B_Vhnd127_i2,e0_C_Vhnd127_i2,e0_D_Vhnd127_i2,e0_E_Vhnd127_i2,e0_F_Vhnd127_i2,e0_G_Vhnd127_i2,e0_T_ho_i2,e0_h_o_i2)
    e0e1_greek_sigma_W_tr13 = fun_75390(e0e1_T_tr13,e0e1_A_Wsigd106,e0e1_B_Wsigd106,e0e1_C_Wsigd106,e0e1_D_Wsigd106,e0e1_E_Wsigd106,e0e1_T_Wcrit)
    e0e1_h_Vn_tr8_i1 = fun_93415(e0e1_T_tr8,e0e1_A_Vhnd127_i1,e0e1_B_Vhnd127_i1,e0e1_C_Vhnd127_i1,e0e1_D_Vhnd127_i1,e0e1_E_Vhnd127_i1,e0e1_F_Vhnd127_i1,e0e1_G_Vhnd127_i1,e0e1_T_ho_i1,e0e1_h_o_i1)
    e0e1_F_Lm_tr12 = fun_94396(e0e1_F_Ln_tr12,e0e1_M_L_tr12)
    e0e1_greek_eta_V_i1_tr7 = fun_75392(e0e1_T_tr7,e0e1_A_Vetad102_i1,e0e1_B_Vetad102_i1,e0e1_C_Vetad102_i1,e0e1_D_Vetad102_i1)
    e0e1_F_Vm_tr19 = fun_94396(e0e1_F_Vn_tr19,e0e1_M_V_tr19)
    e0e1_greek_rho_Ln_tr19_i1 = fun_75389(e0e1_T_tr19,e0e1_A_Lnrhod105_i1,e0e1_B_Lnrhod105_i1,e0e1_C_Lnrhod105_i1,e0e1_D_Lnrhod105_i1)
    e0e1_greek_rho_Ln_tr5_i2 = fun_75389(e0e1_T_tr5,e0e1_A_Lnrhod105_i2,e0e1_B_Lnrhod105_i2,e0e1_C_Lnrhod105_i2,e0e1_D_Lnrhod105_i2)
    e0e1_h_Vn_tr1_i1 = fun_93415(e0e1_T_tr1,e0e1_A_Vhnd127_i1,e0e1_B_Vhnd127_i1,e0e1_C_Vhnd127_i1,e0e1_D_Vhnd127_i1,e0e1_E_Vhnd127_i1,e0e1_F_Vhnd127_i1,e0e1_G_Vhnd127_i1,e0e1_T_ho_i1,e0_h_o_i1)
    e0e1_h_Vn_tr15_i2 = fun_93415(e0e1_T_tr15,e0e1_A_Vhnd127_i2,e0e1_B_Vhnd127_i2,e0e1_C_Vhnd127_i2,e0e1_D_Vhnd127_i2,e0e1_E_Vhnd127_i2,e0e1_F_Vhnd127_i2,e0e1_G_Vhnd127_i2,e0e1_T_ho_i2,e0e1_h_o_i2)
    e0e1_greek_eta_V_i1_tr19 = fun_75392(e0e1_T_tr19,e0e1_A_Vetad102_i1,e0e1_B_Vetad102_i1,e0e1_C_Vetad102_i1,e0e1_D_Vetad102_i1)
    e0_greek_rho_LReboilern_i1 = fun_75389(e0_T_Reboiler,e0_A_Lnrhod105_i1,e0_B_Lnrhod105_i1,e0_C_Lnrhod105_i1,e0_D_Lnrhod105_i1)
    e0e1_h_LVn_tr10_i1 = fun_75390(e0e1_T_tr10,e0e1_A_LVhnd106_i1,e0e1_B_LVhnd106_i1,e0e1_C_LVhnd106_i1,e0e1_D_LVhnd106_i1,e0e1_E_LVhnd106_i1,e0e1_T_crit_i1)
    e0e1_h_Vn_tr13_i2 = fun_93415(e0e1_T_tr13,e0e1_A_Vhnd127_i2,e0e1_B_Vhnd127_i2,e0e1_C_Vhnd127_i2,e0e1_D_Vhnd127_i2,e0e1_E_Vhnd127_i2,e0e1_F_Vhnd127_i2,e0e1_G_Vhnd127_i2,e0e1_T_ho_i2,e0e1_h_o_i2)
    e0e1_greek_eta_L_tr12_i2 = fun_75386(e0e1_T_tr12,e0e1_A_Letad101_i2,e0e1_B_Letad101_i2,e0e1_C_Letad101_i2,e0e1_D_Letad101_i2,e0e1_E_Letad101_i2)
    e0e1_h_LVn_tr12_i1 = fun_75390(e0e1_T_tr12,e0e1_A_LVhnd106_i1,e0e1_B_LVhnd106_i1,e0e1_C_LVhnd106_i1,e0e1_D_LVhnd106_i1,e0e1_E_LVhnd106_i1,e0e1_T_crit_i1)
    e0e1_greek_sigma_i2_tr3 = fun_75390(e0e1_T_tr3,e0e1_A_sigd106_i2,e0e1_B_sigd106_i2,e0e1_C_sigd106_i2,e0e1_D_sigd106_i2,e0e1_E_sigd106_i2,e0e1_T_crit_i2)
    e0e1_h_LVn_tr3_i1 = fun_75390(e0e1_T_tr3,e0e1_A_LVhnd106_i1,e0e1_B_LVhnd106_i1,e0e1_C_LVhnd106_i1,e0e1_D_LVhnd106_i1,e0e1_E_LVhnd106_i1,e0e1_T_crit_i1)
    e0e1_p_LV_tr1_i2 = fun_75386(e0e1_T_tr1,e0e1_A_LVpd101_i2,e0e1_B_LVpd101_i2,e0e1_C_LVpd101_i2,e0e1_D_LVpd101_i2,e0e1_E_LVpd101_i2)
    e0e1_greek_eta_V_i2_tr4 = fun_75392(e0e1_T_tr4,e0e1_A_Vetad102_i2,e0e1_B_Vetad102_i2,e0e1_C_Vetad102_i2,e0e1_D_Vetad102_i2)
    e0e1_h_Vn_tr17_i1 = fun_93415(e0e1_T_tr17,e0e1_A_Vhnd127_i1,e0e1_B_Vhnd127_i1,e0e1_C_Vhnd127_i1,e0e1_D_Vhnd127_i1,e0e1_E_Vhnd127_i1,e0e1_F_Vhnd127_i1,e0e1_G_Vhnd127_i1,e0e1_T_ho_i1,e0e1_h_o_i1)
    e0e1_greek_eta_L_tr13_i1 = fun_75386(e0e1_T_tr13,e0e1_A_Letad101_i1,e0e1_B_Letad101_i1,e0e1_C_Letad101_i1,e0e1_D_Letad101_i1,e0e1_E_Letad101_i1)
    e0e1_greek_sigma_i1_tr15 = fun_75390(e0e1_T_tr15,e0e1_A_sigd106_i1,e0e1_B_sigd106_i1,e0e1_C_sigd106_i1,e0e1_D_sigd106_i1,e0e1_E_sigd106_i1,e0e1_T_crit_i1)
    e0e1_greek_rho_BLn_i2 = fun_75389(e0e1_T_B,e0e1_A_Lnrhod105_i2,e0e1_B_Lnrhod105_i2,e0e1_C_Lnrhod105_i2,e0e1_D_Lnrhod105_i2)
    e0e1_greek_eta_L_tr13_i2 = fun_75386(e0e1_T_tr13,e0e1_A_Letad101_i2,e0e1_B_Letad101_i2,e0e1_C_Letad101_i2,e0e1_D_Letad101_i2,e0e1_E_Letad101_i2)
    e0e1_greek_eta_L_tr11_i2 = fun_75386(e0e1_T_tr11,e0e1_A_Letad101_i2,e0e1_B_Letad101_i2,e0e1_C_Letad101_i2,e0e1_D_Letad101_i2,e0e1_E_Letad101_i2)
    e0e1_F_Vm_tr2 = fun_94396(e0e1_F_Vn_tr2,e0e1_M_V_tr2)
    e0e1_p_LV_tr17_i1 = fun_75386(e0e1_T_tr17,e0e1_A_LVpd101_i1,e0e1_B_LVpd101_i1,e0e1_C_LVpd101_i1,e0e1_D_LVpd101_i1,e0e1_E_LVpd101_i1)
    e0e1_greek_sigma_i2_tr2 = fun_75390(e0e1_T_tr2,e0e1_A_sigd106_i2,e0e1_B_sigd106_i2,e0e1_C_sigd106_i2,e0e1_D_sigd106_i2,e0e1_E_sigd106_i2,e0e1_T_crit_i2)
    e0e1_F_Lm_tr14 = fun_94396(e0e1_F_Ln_tr14,e0e1_M_L_tr14)
    e0e1_F_Lm_tr19 = fun_94396(e0e1_F_Ln_tr19,e0e1_M_L_tr19)
    e0_p_LVReboiler_i1 = fun_75386(e0_T_Reboiler,e0_A_LVpd101_i1,e0_B_LVpd101_i1,e0_C_LVpd101_i1,e0_D_LVpd101_i1,e0_E_LVpd101_i1)
    e0e1_greek_sigma_i1_tr18 = fun_75390(e0e1_T_tr18,e0e1_A_sigd106_i1,e0e1_B_sigd106_i1,e0e1_C_sigd106_i1,e0e1_D_sigd106_i1,e0e1_E_sigd106_i1,e0e1_T_crit_i1)
    e0e1_greek_eta_L_tr14_i2 = fun_75386(e0e1_T_tr14,e0e1_A_Letad101_i2,e0e1_B_Letad101_i2,e0e1_C_Letad101_i2,e0e1_D_Letad101_i2,e0e1_E_Letad101_i2)
    e0e1_greek_eta_V_i1_tr1 = fun_75392(e0e1_T_tr1,e0e1_A_Vetad102_i1,e0e1_B_Vetad102_i1,e0e1_C_Vetad102_i1,e0e1_D_Vetad102_i1)
    e0e1_h_Vn_tr4_i2 = fun_93415(e0e1_T_tr4,e0e1_A_Vhnd127_i2,e0e1_B_Vhnd127_i2,e0e1_C_Vhnd127_i2,e0e1_D_Vhnd127_i2,e0e1_E_Vhnd127_i2,e0e1_F_Vhnd127_i2,e0e1_G_Vhnd127_i2,e0e1_T_ho_i2,e0e1_h_o_i2)
    e0e1_greek_eta_V_i1_tr3 = fun_75392(e0e1_T_tr3,e0e1_A_Vetad102_i1,e0e1_B_Vetad102_i1,e0e1_C_Vetad102_i1,e0e1_D_Vetad102_i1)
    e0e1_greek_eta_V_i2_tr6 = fun_75392(e0e1_T_tr6,e0e1_A_Vetad102_i2,e0e1_B_Vetad102_i2,e0e1_C_Vetad102_i2,e0e1_D_Vetad102_i2)
    e0e1_F_Lm_tr4 = fun_94396(e0e1_F_Ln_tr4,e0e1_M_L_tr4)
    e0e1_p_LV_tr11_i2 = fun_75386(e0e1_T_tr11,e0e1_A_LVpd101_i2,e0e1_B_LVpd101_i2,e0e1_C_LVpd101_i2,e0e1_D_LVpd101_i2,e0e1_E_LVpd101_i2)
    e0e1_h_LVn_tr2_i2 = fun_75390(e0e1_T_tr2,e0e1_A_LVhnd106_i2,e0e1_B_LVhnd106_i2,e0e1_C_LVhnd106_i2,e0e1_D_LVhnd106_i2,e0e1_E_LVhnd106_i2,e0e1_T_crit_i2)
    e0e1_p_LV_tr18_i1 = fun_75386(e0e1_T_tr18,e0e1_A_LVpd101_i1,e0e1_B_LVpd101_i1,e0e1_C_LVpd101_i1,e0e1_D_LVpd101_i1,e0e1_E_LVpd101_i1)
    e0_greek_rho_DLn_i2 = fun_75389(e0_T_D,e0_A_Lnrhod105_i2,e0_B_Lnrhod105_i2,e0_C_Lnrhod105_i2,e0_D_Lnrhod105_i2)
    e0e1_h_LVn_tr7_i1 = fun_75390(e0e1_T_tr7,e0e1_A_LVhnd106_i1,e0e1_B_LVhnd106_i1,e0e1_C_LVhnd106_i1,e0e1_D_LVhnd106_i1,e0e1_E_LVhnd106_i1,e0e1_T_crit_i1)
    e0e1_h_BLVn_i2 = fun_75390(e0e1_T_B,e0e1_A_LVhnd106_i2,e0e1_B_LVhnd106_i2,e0e1_C_LVhnd106_i2,e0e1_D_LVhnd106_i2,e0e1_E_LVhnd106_i2,e0e1_T_crit_i2)
    e0e1_greek_rho_Ln_tr4_i1 = fun_75389(e0e1_T_tr4,e0e1_A_Lnrhod105_i1,e0e1_B_Lnrhod105_i1,e0e1_C_Lnrhod105_i1,e0e1_D_Lnrhod105_i1)
    e0_h_ReboilerVn_i2 = fun_93415(e0_T_Reboiler,e0_A_Vhnd127_i2,e0_B_Vhnd127_i2,e0_C_Vhnd127_i2,e0_D_Vhnd127_i2,e0_E_Vhnd127_i2,e0_F_Vhnd127_i2,e0_G_Vhnd127_i2,e0_T_ho_i2,e0_h_o_i2)
    e0e1_F_Vm_tr16 = fun_94396(e0e1_F_Vn_tr16,e0e1_M_V_tr16)
    e0_h_DLVn_i1 = fun_75390(e0_T_D,e0_A_LVhnd106_i1,e0_B_LVhnd106_i1,e0_C_LVhnd106_i1,e0_D_LVhnd106_i1,e0_E_LVhnd106_i1,e0_T_crit_i1)
    e0e1_h_Vn_tr6_i1 = fun_93415(e0e1_T_tr6,e0e1_A_Vhnd127_i1,e0e1_B_Vhnd127_i1,e0e1_C_Vhnd127_i1,e0e1_D_Vhnd127_i1,e0e1_E_Vhnd127_i1,e0e1_F_Vhnd127_i1,e0e1_G_Vhnd127_i1,e0e1_T_ho_i1,e0e1_h_o_i1)
    e0e1_greek_sigma_i1_tr12 = fun_75390(e0e1_T_tr12,e0e1_A_sigd106_i1,e0e1_B_sigd106_i1,e0e1_C_sigd106_i1,e0e1_D_sigd106_i1,e0e1_E_sigd106_i1,e0e1_T_crit_i1)
    e0e1_h_BVn_i1 = fun_93415(e0e1_T_B,e0e1_A_Vhnd127_i1,e0e1_B_Vhnd127_i1,e0e1_C_Vhnd127_i1,e0e1_D_Vhnd127_i1,e0e1_E_Vhnd127_i1,e0e1_F_Vhnd127_i1,e0e1_G_Vhnd127_i1,e0e1_T_ho_i1,e0e1_h_o_i1)
    e0e1_greek_rho_Ln_tr10_i2 = fun_75389(e0e1_T_tr10,e0e1_A_Lnrhod105_i2,e0e1_B_Lnrhod105_i2,e0e1_C_Lnrhod105_i2,e0e1_D_Lnrhod105_i2)
    e0e1_greek_sigma_W_tr4 = fun_75390(e0e1_T_tr4,e0e1_A_Wsigd106,e0e1_B_Wsigd106,e0e1_C_Wsigd106,e0e1_D_Wsigd106,e0e1_E_Wsigd106,e0e1_T_Wcrit)
    e0e1_h_LVn_tr11_i2 = fun_75390(e0e1_T_tr11,e0e1_A_LVhnd106_i2,e0e1_B_LVhnd106_i2,e0e1_C_LVhnd106_i2,e0e1_D_LVhnd106_i2,e0e1_E_LVhnd106_i2,e0e1_T_crit_i2)
    e0e1_h_LVn_tr12_i2 = fun_75390(e0e1_T_tr12,e0e1_A_LVhnd106_i2,e0e1_B_LVhnd106_i2,e0e1_C_LVhnd106_i2,e0e1_D_LVhnd106_i2,e0e1_E_LVhnd106_i2,e0e1_T_crit_i2)
    e0e1_greek_rho_Ln_tr6_i2 = fun_75389(e0e1_T_tr6,e0e1_A_Lnrhod105_i2,e0e1_B_Lnrhod105_i2,e0e1_C_Lnrhod105_i2,e0e1_D_Lnrhod105_i2)
    e0e1_h_LVn_tr14_i2 = fun_75390(e0e1_T_tr14,e0e1_A_LVhnd106_i2,e0e1_B_LVhnd106_i2,e0e1_C_LVhnd106_i2,e0e1_D_LVhnd106_i2,e0e1_E_LVhnd106_i2,e0e1_T_crit_i2)
    e0e1_greek_sigma_W_tr15 = fun_75390(e0e1_T_tr15,e0e1_A_Wsigd106,e0e1_B_Wsigd106,e0e1_C_Wsigd106,e0e1_D_Wsigd106,e0e1_E_Wsigd106,e0e1_T_Wcrit)
    e0e1_h_Vn_tr5_i2 = fun_93415(e0e1_T_tr5,e0e1_A_Vhnd127_i2,e0e1_B_Vhnd127_i2,e0e1_C_Vhnd127_i2,e0e1_D_Vhnd127_i2,e0e1_E_Vhnd127_i2,e0e1_F_Vhnd127_i2,e0e1_G_Vhnd127_i2,e0e1_T_ho_i2,e0e1_h_o_i2)
    e0e1_h_Vn_tr16_i1 = fun_93415(e0e1_T_tr16,e0e1_A_Vhnd127_i1,e0e1_B_Vhnd127_i1,e0e1_C_Vhnd127_i1,e0e1_D_Vhnd127_i1,e0e1_E_Vhnd127_i1,e0e1_F_Vhnd127_i1,e0e1_G_Vhnd127_i1,e0e1_T_ho_i1,e0e1_h_o_i1)
    e0e1_p_LV_tr10_i2 = fun_75386(e0e1_T_tr10,e0e1_A_LVpd101_i2,e0e1_B_LVpd101_i2,e0e1_C_LVpd101_i2,e0e1_D_LVpd101_i2,e0e1_E_LVpd101_i2)
    e0e1_h_LVn_tr11_i1 = fun_75390(e0e1_T_tr11,e0e1_A_LVhnd106_i1,e0e1_B_LVhnd106_i1,e0e1_C_LVhnd106_i1,e0e1_D_LVhnd106_i1,e0e1_E_LVhnd106_i1,e0e1_T_crit_i1)
    e0e1_h_LVn_tr10_i2 = fun_75390(e0e1_T_tr10,e0e1_A_LVhnd106_i2,e0e1_B_LVhnd106_i2,e0e1_C_LVhnd106_i2,e0e1_D_LVhnd106_i2,e0e1_E_LVhnd106_i2,e0e1_T_crit_i2)
    e0e1_greek_sigma_W_tr20 = fun_75390(e0e1_T_tr20,e0e1_A_Wsigd106,e0e1_B_Wsigd106,e0e1_C_Wsigd106,e0e1_D_Wsigd106,e0e1_E_Wsigd106,e0e1_T_Wcrit)
    e0e1_F_Vm_tr8 = fun_94396(e0e1_F_Vn_tr8,e0e1_M_V_tr8)
    e0e1_greek_rho_Ln_tr9_i2 = fun_75389(e0e1_T_tr9,e0e1_A_Lnrhod105_i2,e0e1_B_Lnrhod105_i2,e0e1_C_Lnrhod105_i2,e0e1_D_Lnrhod105_i2)
    e0e1_h_Vn_tr7_i2 = fun_93415(e0e1_T_tr7,e0e1_A_Vhnd127_i2,e0e1_B_Vhnd127_i2,e0e1_C_Vhnd127_i2,e0e1_D_Vhnd127_i2,e0e1_E_Vhnd127_i2,e0e1_F_Vhnd127_i2,e0e1_G_Vhnd127_i2,e0e1_T_ho_i2,e0e1_h_o_i2)
    e0e1_p_LV_tr5_i2 = fun_75386(e0e1_T_tr5,e0e1_A_LVpd101_i2,e0e1_B_LVpd101_i2,e0e1_C_LVpd101_i2,e0e1_D_LVpd101_i2,e0e1_E_LVpd101_i2)
    e0e1_greek_eta_L_tr4_i2 = fun_75386(e0e1_T_tr4,e0e1_A_Letad101_i2,e0e1_B_Letad101_i2,e0e1_C_Letad101_i2,e0e1_D_Letad101_i2,e0e1_E_Letad101_i2)
    e0e1_h_Vn_tr19_i2 = fun_93415(e0e1_T_tr19,e0e1_A_Vhnd127_i2,e0e1_B_Vhnd127_i2,e0e1_C_Vhnd127_i2,e0e1_D_Vhnd127_i2,e0e1_E_Vhnd127_i2,e0e1_F_Vhnd127_i2,e0e1_G_Vhnd127_i2,e0e1_T_ho_i2,e0e1_h_o_i2)
    e0e1_greek_eta_L_tr7_i1 = fun_75386(e0e1_T_tr7,e0e1_A_Letad101_i1,e0e1_B_Letad101_i1,e0e1_C_Letad101_i1,e0e1_D_Letad101_i1,e0e1_E_Letad101_i1)
    e0e1_p_LV_tr6_i1 = fun_75386(e0e1_T_tr6,e0e1_A_LVpd101_i1,e0e1_B_LVpd101_i1,e0e1_C_LVpd101_i1,e0e1_D_LVpd101_i1,e0e1_E_LVpd101_i1)
    e0e1_greek_eta_V_i1_tr17 = fun_75392(e0e1_T_tr17,e0e1_A_Vetad102_i1,e0e1_B_Vetad102_i1,e0e1_C_Vetad102_i1,e0e1_D_Vetad102_i1)
    e0e1_F_Lm_tr17 = fun_94396(e0e1_F_Ln_tr17,e0e1_M_L_tr17)
    e0e1_greek_eta_V_i1_tr12 = fun_75392(e0e1_T_tr12,e0e1_A_Vetad102_i1,e0e1_B_Vetad102_i1,e0e1_C_Vetad102_i1,e0e1_D_Vetad102_i1)
    e0e1_h_Vn_tr5_i1 = fun_93415(e0e1_T_tr5,e0e1_A_Vhnd127_i1,e0e1_B_Vhnd127_i1,e0e1_C_Vhnd127_i1,e0e1_D_Vhnd127_i1,e0e1_E_Vhnd127_i1,e0e1_F_Vhnd127_i1,e0e1_G_Vhnd127_i1,e0e1_T_ho_i1,e0e1_h_o_i1)
    e0e1_h_Vn_tr20_i1 = fun_93415(e0e1_T_tr20,e0e1_A_Vhnd127_i1,e0e1_B_Vhnd127_i1,e0e1_C_Vhnd127_i1,e0e1_D_Vhnd127_i1,e0e1_E_Vhnd127_i1,e0e1_F_Vhnd127_i1,e0e1_G_Vhnd127_i1,e0e1_T_ho_i1,e0e1_h_o_i1)
    e0e1_h_LVn_tr6_i2 = fun_75390(e0e1_T_tr6,e0e1_A_LVhnd106_i2,e0e1_B_LVhnd106_i2,e0e1_C_LVhnd106_i2,e0e1_D_LVhnd106_i2,e0e1_E_LVhnd106_i2,e0e1_T_crit_i2)
    e0e1_greek_sigma_W_tr19 = fun_75390(e0e1_T_tr19,e0e1_A_Wsigd106,e0e1_B_Wsigd106,e0e1_C_Wsigd106,e0e1_D_Wsigd106,e0e1_E_Wsigd106,e0e1_T_Wcrit)
    e0e1_h_LVn_tr6_i1 = fun_75390(e0e1_T_tr6,e0e1_A_LVhnd106_i1,e0e1_B_LVhnd106_i1,e0e1_C_LVhnd106_i1,e0e1_D_LVhnd106_i1,e0e1_E_LVhnd106_i1,e0e1_T_crit_i1)
    e0e1_p_LV_tr1_i1 = fun_75386(e0e1_T_tr1,e0e1_A_LVpd101_i1,e0e1_B_LVpd101_i1,e0e1_C_LVpd101_i1,e0e1_D_LVpd101_i1,e0e1_E_LVpd101_i1)
    e0e1_F_Vm_tr18 = fun_94396(e0e1_F_Vn_tr18,e0e1_M_V_tr18)
    e0e1_h_BLn_i1 = fun_69026(e0e1_greek_rho_BLn_i1,e0e1_h_BLVn_i1,e0e1_h_BVn_i1,e0e1_p_tr21,e0e1_p_BLV_i1)
    e0e1_h_Ln_tr2_i2 = fun_69026(e0e1_greek_rho_Ln_tr2_i2,e0e1_h_LVn_tr2_i2,e0e1_h_Vn_tr2_i2,e0e1_p_tr2,e0e1_p_LV_tr2_i2)
    e0e1_h_Ln_tr2_i1 = fun_69026(e0e1_greek_rho_Ln_tr2_i1,e0e1_h_LVn_tr2_i1,e0e1_h_Vn_tr2_i1,e0e1_p_tr2,e0e1_p_LV_tr2_i1)
    e0e1_h_Ln_tr14_i1 = fun_69026(e0e1_greek_rho_Ln_tr14_i1,e0e1_h_LVn_tr14_i1,e0e1_h_Vn_tr14_i1,e0e1_p_tr14,e0e1_p_LV_tr14_i1)
    e0e1_h_Ln_tr8_i2 = fun_69026(e0e1_greek_rho_Ln_tr8_i2,e0e1_h_LVn_tr8_i2,e0e1_h_Vn_tr8_i2,e0e1_p_tr8,e0e1_p_LV_tr8_i2)
    e0_h_DLn_i2 = fun_69026(e0_greek_rho_DLn_i2,e0_h_DLVn_i2,e0_h_DVn_i2,e0_p_DL,e0_p_DLV_i2)
    e0_h_LReboilern_i2 = fun_69026(e0_greek_rho_LReboilern_i2,e0_h_LVReboilern_i2,e0_h_ReboilerVn_i2,e0_p_ReboilerV,e0_p_LVReboiler_i2)
    e0e1_h_Ln_tr7_i2 = fun_69026(e0e1_greek_rho_Ln_tr7_i2,e0e1_h_LVn_tr7_i2,e0e1_h_Vn_tr7_i2,e0e1_p_tr7,e0e1_p_LV_tr7_i2)
    e0e1_h_Ln_tr9_i1 = fun_69026(e0e1_greek_rho_Ln_tr9_i1,e0e1_h_LVn_tr9_i1,e0e1_h_Vn_tr9_i1,e0e1_p_tr9,e0e1_p_LV_tr9_i1)
    e0e1_h_Ln_tr13_i1 = fun_69026(e0e1_greek_rho_Ln_tr13_i1,e0e1_h_LVn_tr13_i1,e0e1_h_Vn_tr13_i1,e0e1_p_tr13,e0e1_p_LV_tr13_i1)
    e0e1_h_Ln_tr19_i1 = fun_69026(e0e1_greek_rho_Ln_tr19_i1,e0e1_h_LVn_tr19_i1,e0e1_h_Vn_tr19_i1,e0e1_p_tr19,e0e1_p_LV_tr19_i1)
    e0e1_h_Ln_tr15_i2 = fun_69026(e0e1_greek_rho_Ln_tr15_i2,e0e1_h_LVn_tr15_i2,e0e1_h_Vn_tr15_i2,e0e1_p_tr15,e0e1_p_LV_tr15_i2)
    e0e1_h_Ln_tr20_i2 = fun_69026(e0e1_greek_rho_Ln_tr20_i2,e0e1_h_LVn_tr20_i2,e0e1_h_Vn_tr20_i2,e0_p_LReboiler,e0e1_p_LV_tr20_i2)
    e0e1_h_Ln_tr7_i1 = fun_69026(e0e1_greek_rho_Ln_tr7_i1,e0e1_h_LVn_tr7_i1,e0e1_h_Vn_tr7_i1,e0e1_p_tr7,e0e1_p_LV_tr7_i1)
    e0e1_h_Ln_tr1_i2 = fun_69026(e0e1_greek_rho_Ln_tr1_i2,e0e1_h_LVn_tr1_i2,e0e1_h_Vn_tr1_i2,e0e1_p_tr1,e0e1_p_LV_tr1_i2)
    e0e1_h_Ln_tr3_i1 = fun_69026(e0e1_greek_rho_Ln_tr3_i1,e0e1_h_LVn_tr3_i1,e0e1_h_Vn_tr3_i1,e0e1_p_tr3,e0e1_p_LV_tr3_i1)
    e0e1_h_Ln_tr16_i2 = fun_69026(e0e1_greek_rho_Ln_tr16_i2,e0e1_h_LVn_tr16_i2,e0e1_h_Vn_tr16_i2,e0e1_p_tr16,e0e1_p_LV_tr16_i2)
    e0_h_Fn_i2 = fun_69026(e0_greek_rho_FLn_i2,e0_h_FLVn_i2,e0_h_FVn_i2,e0_p_F,e0_p_FLV_i2)
    e0e1_h_Ln_tr6_i1 = fun_69026(e0e1_greek_rho_Ln_tr6_i1,e0e1_h_LVn_tr6_i1,e0e1_h_Vn_tr6_i1,e0e1_p_tr6,e0e1_p_LV_tr6_i1)
    e0e1_h_Ln_tr13_i2 = fun_69026(e0e1_greek_rho_Ln_tr13_i2,e0e1_h_LVn_tr13_i2,e0e1_h_Vn_tr13_i2,e0e1_p_tr13,e0e1_p_LV_tr13_i2)
    e0e1_h_Ln_tr19_i2 = fun_69026(e0e1_greek_rho_Ln_tr19_i2,e0e1_h_LVn_tr19_i2,e0e1_h_Vn_tr19_i2,e0e1_p_tr19,e0e1_p_LV_tr19_i2)
    e0e1_h_Ln_tr1_i1 = fun_69026(e0e1_greek_rho_Ln_tr1_i1,e0e1_h_LVn_tr1_i1,e0e1_h_Vn_tr1_i1,e0e1_p_tr1,e0e1_p_LV_tr1_i1)
    e0e1_h_Ln_tr10_i1 = fun_69026(e0e1_greek_rho_Ln_tr10_i1,e0e1_h_LVn_tr10_i1,e0e1_h_Vn_tr10_i1,e0e1_p_tr10,e0e1_p_LV_tr10_i1)
    e0e1_h_Ln_tr11_i2 = fun_69026(e0e1_greek_rho_Ln_tr11_i2,e0e1_h_LVn_tr11_i2,e0e1_h_Vn_tr11_i2,e0e1_p_tr11,e0e1_p_LV_tr11_i2)
    e0e1_h_Ln_tr14_i2 = fun_69026(e0e1_greek_rho_Ln_tr14_i2,e0e1_h_LVn_tr14_i2,e0e1_h_Vn_tr14_i2,e0e1_p_tr14,e0e1_p_LV_tr14_i2)
    e0e1_h_Ln_tr5_i1 = fun_69026(e0e1_greek_rho_Ln_tr5_i1,e0e1_h_LVn_tr5_i1,e0e1_h_Vn_tr5_i1,e0e1_p_tr5,e0e1_p_LV_tr5_i1)
    e0e1_h_Ln_tr6_i2 = fun_69026(e0e1_greek_rho_Ln_tr6_i2,e0e1_h_LVn_tr6_i2,e0e1_h_Vn_tr6_i2,e0e1_p_tr6,e0e1_p_LV_tr6_i2)
    e0e1_h_Ln_tr9_i2 = fun_69026(e0e1_greek_rho_Ln_tr9_i2,e0e1_h_LVn_tr9_i2,e0e1_h_Vn_tr9_i2,e0e1_p_tr9,e0e1_p_LV_tr9_i2)
    e0e1_h_Ln_tr4_i1 = fun_69026(e0e1_greek_rho_Ln_tr4_i1,e0e1_h_LVn_tr4_i1,e0e1_h_Vn_tr4_i1,e0e1_p_tr4,e0e1_p_LV_tr4_i1)
    e0e1_h_Ln_tr4_i2 = fun_69026(e0e1_greek_rho_Ln_tr4_i2,e0e1_h_LVn_tr4_i2,e0e1_h_Vn_tr4_i2,e0e1_p_tr4,e0e1_p_LV_tr4_i2)
    e0e1_h_Ln_tr10_i2 = fun_69026(e0e1_greek_rho_Ln_tr10_i2,e0e1_h_LVn_tr10_i2,e0e1_h_Vn_tr10_i2,e0e1_p_tr10,e0e1_p_LV_tr10_i2)
    e0e1_h_Ln_tr11_i1 = fun_69026(e0e1_greek_rho_Ln_tr11_i1,e0e1_h_LVn_tr11_i1,e0e1_h_Vn_tr11_i1,e0e1_p_tr11,e0e1_p_LV_tr11_i1)
    e0_h_Fn_i1 = fun_69026(e0_greek_rho_FLn_i1,e0_h_FLVn_i1,e0_h_FVn_i1,e0_p_F,e0_p_FLV_i1)
    e0e1_h_Ln_tr17_i2 = fun_69026(e0e1_greek_rho_Ln_tr17_i2,e0e1_h_LVn_tr17_i2,e0e1_h_Vn_tr17_i2,e0e1_p_tr17,e0e1_p_LV_tr17_i2)
    e0e1_h_Ln_tr8_i1 = fun_69026(e0e1_greek_rho_Ln_tr8_i1,e0e1_h_LVn_tr8_i1,e0e1_h_Vn_tr8_i1,e0e1_p_tr8,e0e1_p_LV_tr8_i1)
    e0e1_h_Ln_tr5_i2 = fun_69026(e0e1_greek_rho_Ln_tr5_i2,e0e1_h_LVn_tr5_i2,e0e1_h_Vn_tr5_i2,e0e1_p_tr5,e0e1_p_LV_tr5_i2)
    e0e1_h_Ln_tr12_i1 = fun_69026(e0e1_greek_rho_Ln_tr12_i1,e0e1_h_LVn_tr12_i1,e0e1_h_Vn_tr12_i1,e0e1_p_tr12,e0e1_p_LV_tr12_i1)
    e0e1_h_Ln_tr3_i2 = fun_69026(e0e1_greek_rho_Ln_tr3_i2,e0e1_h_LVn_tr3_i2,e0e1_h_Vn_tr3_i2,e0e1_p_tr3,e0e1_p_LV_tr3_i2)
    e0_h_DLn_i1 = fun_69026(e0_greek_rho_DLn_i1,e0_h_DLVn_i1,e0_h_DVn_i1,e0_p_DL,e0_p_DLV_i1)
    e0e1_h_Ln_tr18_i1 = fun_69026(e0e1_greek_rho_Ln_tr18_i1,e0e1_h_LVn_tr18_i1,e0e1_h_Vn_tr18_i1,e0e1_p_tr18,e0e1_p_LV_tr18_i1)
    e0e1_h_BLn_i2 = fun_69026(e0e1_greek_rho_BLn_i2,e0e1_h_BLVn_i2,e0e1_h_BVn_i2,e0e1_p_tr21,e0e1_p_BLV_i2)
    e0e1_h_Ln_tr16_i1 = fun_69026(e0e1_greek_rho_Ln_tr16_i1,e0e1_h_LVn_tr16_i1,e0e1_h_Vn_tr16_i1,e0e1_p_tr16,e0e1_p_LV_tr16_i1)
    e0e1_h_Ln_tr15_i1 = fun_69026(e0e1_greek_rho_Ln_tr15_i1,e0e1_h_LVn_tr15_i1,e0e1_h_Vn_tr15_i1,e0e1_p_tr15,e0e1_p_LV_tr15_i1)
    e0e1_h_Ln_tr17_i1 = fun_69026(e0e1_greek_rho_Ln_tr17_i1,e0e1_h_LVn_tr17_i1,e0e1_h_Vn_tr17_i1,e0e1_p_tr17,e0e1_p_LV_tr17_i1)
    e0e1_h_Ln_tr20_i1 = fun_69026(e0e1_greek_rho_Ln_tr20_i1,e0e1_h_LVn_tr20_i1,e0e1_h_Vn_tr20_i1,e0_p_LReboiler,e0e1_p_LV_tr20_i1)
    e0e1_h_Ln_tr12_i2 = fun_69026(e0e1_greek_rho_Ln_tr12_i2,e0e1_h_LVn_tr12_i2,e0e1_h_Vn_tr12_i2,e0e1_p_tr12,e0e1_p_LV_tr12_i2)
    e0_h_LReboilern_i1 = fun_69026(e0_greek_rho_LReboilern_i1,e0_h_LVReboilern_i1,e0_h_ReboilerVn_i1,e0_p_ReboilerV,e0_p_LVReboiler_i1)
    e0e1_h_Ln_tr18_i2 = fun_69026(e0e1_greek_rho_Ln_tr18_i2,e0e1_h_LVn_tr18_i2,e0e1_h_Vn_tr18_i2,e0e1_p_tr18,e0e1_p_LV_tr18_i2)



# Solve equation system for given x:
    f= [
    e0_h_Fn-(((e0_x_F_i1) *(e0_h_Fn_i1)+(e0_x_F_i2) *(e0_h_Fn_i2))+e0_h_EFn) ,
    1.0-((e0_x_F_i1+e0_x_F_i2)) ,
    e0_h_DVn-(((e0_y_D_i1) *(e0e1_h_Vn_tr1_i1)+(e0_y_D_i2) *(e0e1_h_Vn_tr1_i2))+e0e1_h_EVn_tr1) ,
    e0e1_h_Vn_tr2-(((e0e1_y_tr2_i1) *(e0e1_h_Vn_tr2_i1)+(e0e1_y_tr2_i2) *(e0e1_h_Vn_tr2_i2))+e0e1_h_EVn_tr2) ,
    e0e1_h_Vn_tr3-(((e0e1_y_tr3_i1) *(e0e1_h_Vn_tr3_i1)+(e0e1_y_tr3_i2) *(e0e1_h_Vn_tr3_i2))+e0e1_h_EVn_tr3) ,
    e0e1_h_Vn_tr4-(((e0e1_y_tr4_i1) *(e0e1_h_Vn_tr4_i1)+(e0e1_y_tr4_i2) *(e0e1_h_Vn_tr4_i2))+e0e1_h_EVn_tr4) ,
    e0e1_h_Vn_tr5-(((e0e1_y_tr5_i1) *(e0e1_h_Vn_tr5_i1)+(e0e1_y_tr5_i2) *(e0e1_h_Vn_tr5_i2))+e0e1_h_EVn_tr5) ,
    e0e1_h_Vn_tr6-(((e0e1_y_tr6_i1) *(e0e1_h_Vn_tr6_i1)+(e0e1_y_tr6_i2) *(e0e1_h_Vn_tr6_i2))+e0e1_h_EVn_tr6) ,
    e0e1_h_Vn_tr7-(((e0e1_y_tr7_i1) *(e0e1_h_Vn_tr7_i1)+(e0e1_y_tr7_i2) *(e0e1_h_Vn_tr7_i2))+e0e1_h_EVn_tr7) ,
    e0e1_h_Vn_tr8-(((e0e1_y_tr8_i1) *(e0e1_h_Vn_tr8_i1)+(e0e1_y_tr8_i2) *(e0e1_h_Vn_tr8_i2))+e0e1_h_EVn_tr8) ,
    e0e1_h_Vn_tr9-(((e0e1_y_tr9_i1) *(e0e1_h_Vn_tr9_i1)+(e0e1_y_tr9_i2) *(e0e1_h_Vn_tr9_i2))+e0e1_h_EVn_tr9) ,
    e0e1_h_Vn_tr10-(((e0e1_y_tr10_i1) *(e0e1_h_Vn_tr10_i1)+(e0e1_y_tr10_i2) *(e0e1_h_Vn_tr10_i2))+e0e1_h_EVn_tr10) ,
    e0e1_h_Vn_tr11-(((e0e1_y_tr11_i1) *(e0e1_h_Vn_tr11_i1)+(e0e1_y_tr11_i2) *(e0e1_h_Vn_tr11_i2))+e0e1_h_EVn_tr11) ,
    e0e1_h_Vn_tr12-(((e0e1_y_tr12_i1) *(e0e1_h_Vn_tr12_i1)+(e0e1_y_tr12_i2) *(e0e1_h_Vn_tr12_i2))+e0e1_h_EVn_tr12) ,
    e0e1_h_Vn_tr13-(((e0e1_y_tr13_i1) *(e0e1_h_Vn_tr13_i1)+(e0e1_y_tr13_i2) *(e0e1_h_Vn_tr13_i2))+e0e1_h_EVn_tr13) ,
    e0e1_h_Vn_tr14-(((e0e1_y_tr14_i1) *(e0e1_h_Vn_tr14_i1)+(e0e1_y_tr14_i2) *(e0e1_h_Vn_tr14_i2))+e0e1_h_EVn_tr14) ,
    e0e1_h_Vn_tr15-(((e0e1_y_tr15_i1) *(e0e1_h_Vn_tr15_i1)+(e0e1_y_tr15_i2) *(e0e1_h_Vn_tr15_i2))+e0e1_h_EVn_tr15) ,
    e0e1_h_Vn_tr16-(((e0e1_y_tr16_i1) *(e0e1_h_Vn_tr16_i1)+(e0e1_y_tr16_i2) *(e0e1_h_Vn_tr16_i2))+e0e1_h_EVn_tr16) ,
    e0e1_h_Vn_tr17-(((e0e1_y_tr17_i1) *(e0e1_h_Vn_tr17_i1)+(e0e1_y_tr17_i2) *(e0e1_h_Vn_tr17_i2))+e0e1_h_EVn_tr17) ,
    e0e1_h_Vn_tr18-(((e0e1_y_tr18_i1) *(e0e1_h_Vn_tr18_i1)+(e0e1_y_tr18_i2) *(e0e1_h_Vn_tr18_i2))+e0e1_h_EVn_tr18) ,
    e0e1_h_Vn_tr19-(((e0e1_y_tr19_i1) *(e0e1_h_Vn_tr19_i1)+(e0e1_y_tr19_i2) *(e0e1_h_Vn_tr19_i2))+e0e1_h_EVn_tr19) ,
    e0e1_h_Vn_tr20-(((e0e1_y_tr20_i1) *(e0e1_h_Vn_tr20_i1)+(e0e1_y_tr20_i2) *(e0e1_h_Vn_tr20_i2))+e0e1_h_EVn_tr20) ,
    e0e1_h_Ln_tr1-(((e0e1_x_tr1_i1) *(e0e1_h_Ln_tr1_i1)+(e0e1_x_tr1_i2) *(e0e1_h_Ln_tr1_i2))+e0e1_h_ELn_tr1) ,
    e0e1_h_Ln_tr2-(((e0e1_x_tr2_i1) *(e0e1_h_Ln_tr2_i1)+(e0e1_x_tr2_i2) *(e0e1_h_Ln_tr2_i2))+e0e1_h_ELn_tr2) ,
    e0e1_h_Ln_tr3-(((e0e1_x_tr3_i1) *(e0e1_h_Ln_tr3_i1)+(e0e1_x_tr3_i2) *(e0e1_h_Ln_tr3_i2))+e0e1_h_ELn_tr3) ,
    e0e1_h_Ln_tr4-(((e0e1_x_tr4_i1) *(e0e1_h_Ln_tr4_i1)+(e0e1_x_tr4_i2) *(e0e1_h_Ln_tr4_i2))+e0e1_h_ELn_tr4) ,
    e0e1_h_Ln_tr5-(((e0e1_x_tr5_i1) *(e0e1_h_Ln_tr5_i1)+(e0e1_x_tr5_i2) *(e0e1_h_Ln_tr5_i2))+e0e1_h_ELn_tr5) ,
    e0e1_h_Ln_tr6-(((e0e1_x_tr6_i1) *(e0e1_h_Ln_tr6_i1)+(e0e1_x_tr6_i2) *(e0e1_h_Ln_tr6_i2))+e0e1_h_ELn_tr6) ,
    e0e1_h_Ln_tr7-(((e0e1_x_tr7_i1) *(e0e1_h_Ln_tr7_i1)+(e0e1_x_tr7_i2) *(e0e1_h_Ln_tr7_i2))+e0e1_h_ELn_tr7) ,
    e0e1_h_Ln_tr8-(((e0e1_x_tr8_i1) *(e0e1_h_Ln_tr8_i1)+(e0e1_x_tr8_i2) *(e0e1_h_Ln_tr8_i2))+e0e1_h_ELn_tr8) ,
    e0e1_h_Ln_tr9-(((e0e1_x_tr9_i1) *(e0e1_h_Ln_tr9_i1)+(e0e1_x_tr9_i2) *(e0e1_h_Ln_tr9_i2))+e0e1_h_ELn_tr9) ,
    e0e1_h_Ln_tr10-(((e0e1_x_tr10_i1) *(e0e1_h_Ln_tr10_i1)+(e0e1_x_tr10_i2) *(e0e1_h_Ln_tr10_i2))+e0e1_h_ELn_tr10) ,
    e0e1_h_Ln_tr11-(((e0e1_x_tr11_i1) *(e0e1_h_Ln_tr11_i1)+(e0e1_x_tr11_i2) *(e0e1_h_Ln_tr11_i2))+e0e1_h_ELn_tr11) ,
    e0e1_h_Ln_tr12-(((e0e1_x_tr12_i1) *(e0e1_h_Ln_tr12_i1)+(e0e1_x_tr12_i2) *(e0e1_h_Ln_tr12_i2))+e0e1_h_ELn_tr12) ,
    e0e1_h_Ln_tr13-(((e0e1_x_tr13_i1) *(e0e1_h_Ln_tr13_i1)+(e0e1_x_tr13_i2) *(e0e1_h_Ln_tr13_i2))+e0e1_h_ELn_tr13) ,
    e0e1_h_Ln_tr14-(((e0e1_x_tr14_i1) *(e0e1_h_Ln_tr14_i1)+(e0e1_x_tr14_i2) *(e0e1_h_Ln_tr14_i2))+e0e1_h_ELn_tr14) ,
    e0e1_h_Ln_tr15-(((e0e1_x_tr15_i1) *(e0e1_h_Ln_tr15_i1)+(e0e1_x_tr15_i2) *(e0e1_h_Ln_tr15_i2))+e0e1_h_ELn_tr15) ,
    e0e1_h_Ln_tr16-(((e0e1_x_tr16_i1) *(e0e1_h_Ln_tr16_i1)+(e0e1_x_tr16_i2) *(e0e1_h_Ln_tr16_i2))+e0e1_h_ELn_tr16) ,
    e0e1_h_Ln_tr17-(((e0e1_x_tr17_i1) *(e0e1_h_Ln_tr17_i1)+(e0e1_x_tr17_i2) *(e0e1_h_Ln_tr17_i2))+e0e1_h_ELn_tr17) ,
    e0e1_h_Ln_tr18-(((e0e1_x_tr18_i1) *(e0e1_h_Ln_tr18_i1)+(e0e1_x_tr18_i2) *(e0e1_h_Ln_tr18_i2))+e0e1_h_ELn_tr18) ,
    e0e1_h_Ln_tr19-(((e0e1_x_tr19_i1) *(e0e1_h_Ln_tr19_i1)+(e0e1_x_tr19_i2) *(e0e1_h_Ln_tr19_i2))+e0e1_h_ELn_tr19) ,
    e0_h_LReboilerinn-(((e0_x_Reboilerin_i1) *(e0e1_h_Ln_tr20_i1)+(e0_x_Reboilerin_i2) *(e0e1_h_Ln_tr20_i2))+e0e1_h_ELn_tr20) ,
    (e0e1_K_tr1_i1) *(e0e1_p_tr1)-(e0e1_p_LV_tr1_i1) ,
    (e0e1_K_tr2_i1) *(e0e1_p_tr2)-(e0e1_p_LV_tr2_i1) ,
    (e0e1_K_tr3_i1) *(e0e1_p_tr3)-(e0e1_p_LV_tr3_i1) ,
    (e0e1_K_tr4_i1) *(e0e1_p_tr4)-(e0e1_p_LV_tr4_i1) ,
    (e0e1_K_tr5_i1) *(e0e1_p_tr5)-(e0e1_p_LV_tr5_i1) ,
    (e0e1_K_tr6_i1) *(e0e1_p_tr6)-(e0e1_p_LV_tr6_i1) ,
    (e0e1_K_tr7_i1) *(e0e1_p_tr7)-(e0e1_p_LV_tr7_i1) ,
    (e0e1_K_tr8_i1) *(e0e1_p_tr8)-(e0e1_p_LV_tr8_i1) ,
    (e0e1_K_tr9_i1) *(e0e1_p_tr9)-(e0e1_p_LV_tr9_i1) ,
    (e0e1_K_tr10_i1) *(e0e1_p_tr10)-(e0e1_p_LV_tr10_i1) ,
    (e0e1_K_tr11_i1) *(e0e1_p_tr11)-(e0e1_p_LV_tr11_i1) ,
    (e0e1_K_tr12_i1) *(e0e1_p_tr12)-(e0e1_p_LV_tr12_i1) ,
    (e0e1_K_tr13_i1) *(e0e1_p_tr13)-(e0e1_p_LV_tr13_i1) ,
    (e0e1_K_tr14_i1) *(e0e1_p_tr14)-(e0e1_p_LV_tr14_i1) ,
    (e0e1_K_tr15_i1) *(e0e1_p_tr15)-(e0e1_p_LV_tr15_i1) ,
    (e0e1_K_tr16_i1) *(e0e1_p_tr16)-(e0e1_p_LV_tr16_i1) ,
    (e0e1_K_tr17_i1) *(e0e1_p_tr17)-(e0e1_p_LV_tr17_i1) ,
    (e0e1_K_tr18_i1) *(e0e1_p_tr18)-(e0e1_p_LV_tr18_i1) ,
    (e0e1_K_tr19_i1) *(e0e1_p_tr19)-(e0e1_p_LV_tr19_i1) ,
    (e0e1_K_tr20_i1) *(e0_p_LReboiler)-(e0e1_p_LV_tr20_i1) ,
    (e0e1_K_tr1_i2) *(e0e1_p_tr1)-(e0e1_p_LV_tr1_i2) ,
    (e0e1_K_tr2_i2) *(e0e1_p_tr2)-(e0e1_p_LV_tr2_i2) ,
    (e0e1_K_tr3_i2) *(e0e1_p_tr3)-(e0e1_p_LV_tr3_i2) ,
    (e0e1_K_tr4_i2) *(e0e1_p_tr4)-(e0e1_p_LV_tr4_i2) ,
    (e0e1_K_tr5_i2) *(e0e1_p_tr5)-(e0e1_p_LV_tr5_i2) ,
    (e0e1_K_tr6_i2) *(e0e1_p_tr6)-(e0e1_p_LV_tr6_i2) ,
    (e0e1_K_tr7_i2) *(e0e1_p_tr7)-(e0e1_p_LV_tr7_i2) ,
    (e0e1_K_tr8_i2) *(e0e1_p_tr8)-(e0e1_p_LV_tr8_i2) ,
    (e0e1_K_tr9_i2) *(e0e1_p_tr9)-(e0e1_p_LV_tr9_i2) ,
    (e0e1_K_tr10_i2) *(e0e1_p_tr10)-(e0e1_p_LV_tr10_i2) ,
    (e0e1_K_tr11_i2) *(e0e1_p_tr11)-(e0e1_p_LV_tr11_i2) ,
    (e0e1_K_tr12_i2) *(e0e1_p_tr12)-(e0e1_p_LV_tr12_i2) ,
    (e0e1_K_tr13_i2) *(e0e1_p_tr13)-(e0e1_p_LV_tr13_i2) ,
    (e0e1_K_tr14_i2) *(e0e1_p_tr14)-(e0e1_p_LV_tr14_i2) ,
    (e0e1_K_tr15_i2) *(e0e1_p_tr15)-(e0e1_p_LV_tr15_i2) ,
    (e0e1_K_tr16_i2) *(e0e1_p_tr16)-(e0e1_p_LV_tr16_i2) ,
    (e0e1_K_tr17_i2) *(e0e1_p_tr17)-(e0e1_p_LV_tr17_i2) ,
    (e0e1_K_tr18_i2) *(e0e1_p_tr18)-(e0e1_p_LV_tr18_i2) ,
    (e0e1_K_tr19_i2) *(e0e1_p_tr19)-(e0e1_p_LV_tr19_i2) ,
    (e0e1_K_tr20_i2) *(e0_p_LReboiler)-(e0e1_p_LV_tr20_i2) ,
    e0e1_M_L_tr1-(((e0e1_M_i1) *(e0e1_x_tr1_i1)+(e0e1_M_i2) *(e0e1_x_tr1_i2))) ,
    e0e1_M_L_tr2-(((e0e1_M_i1) *(e0e1_x_tr2_i1)+(e0e1_M_i2) *(e0e1_x_tr2_i2))) ,
    e0e1_M_L_tr3-(((e0e1_M_i1) *(e0e1_x_tr3_i1)+(e0e1_M_i2) *(e0e1_x_tr3_i2))) ,
    e0e1_M_L_tr4-(((e0e1_M_i1) *(e0e1_x_tr4_i1)+(e0e1_M_i2) *(e0e1_x_tr4_i2))) ,
    e0e1_M_L_tr5-(((e0e1_M_i1) *(e0e1_x_tr5_i1)+(e0e1_M_i2) *(e0e1_x_tr5_i2))) ,
    e0e1_M_L_tr6-(((e0e1_M_i1) *(e0e1_x_tr6_i1)+(e0e1_M_i2) *(e0e1_x_tr6_i2))) ,
    e0e1_M_L_tr7-(((e0e1_M_i1) *(e0e1_x_tr7_i1)+(e0e1_M_i2) *(e0e1_x_tr7_i2))) ,
    e0e1_M_L_tr8-(((e0e1_M_i1) *(e0e1_x_tr8_i1)+(e0e1_M_i2) *(e0e1_x_tr8_i2))) ,
    e0e1_M_L_tr9-(((e0e1_M_i1) *(e0e1_x_tr9_i1)+(e0e1_M_i2) *(e0e1_x_tr9_i2))) ,
    e0e1_M_L_tr10-(((e0e1_M_i1) *(e0e1_x_tr10_i1)+(e0e1_M_i2) *(e0e1_x_tr10_i2))) ,
    e0e1_M_L_tr11-(((e0e1_M_i1) *(e0e1_x_tr11_i1)+(e0e1_M_i2) *(e0e1_x_tr11_i2))) ,
    e0e1_M_L_tr12-(((e0e1_M_i1) *(e0e1_x_tr12_i1)+(e0e1_M_i2) *(e0e1_x_tr12_i2))) ,
    e0e1_M_L_tr13-(((e0e1_M_i1) *(e0e1_x_tr13_i1)+(e0e1_M_i2) *(e0e1_x_tr13_i2))) ,
    e0e1_M_L_tr14-(((e0e1_M_i1) *(e0e1_x_tr14_i1)+(e0e1_M_i2) *(e0e1_x_tr14_i2))) ,
    e0e1_M_L_tr15-(((e0e1_M_i1) *(e0e1_x_tr15_i1)+(e0e1_M_i2) *(e0e1_x_tr15_i2))) ,
    e0e1_M_L_tr16-(((e0e1_M_i1) *(e0e1_x_tr16_i1)+(e0e1_M_i2) *(e0e1_x_tr16_i2))) ,
    e0e1_M_L_tr17-(((e0e1_M_i1) *(e0e1_x_tr17_i1)+(e0e1_M_i2) *(e0e1_x_tr17_i2))) ,
    e0e1_M_L_tr18-(((e0e1_M_i1) *(e0e1_x_tr18_i1)+(e0e1_M_i2) *(e0e1_x_tr18_i2))) ,
    e0e1_M_L_tr19-(((e0e1_M_i1) *(e0e1_x_tr19_i1)+(e0e1_M_i2) *(e0e1_x_tr19_i2))) ,
    e0e1_M_L_tr20-(((e0e1_M_i1) *(e0_x_Reboilerin_i1)+(e0e1_M_i2) *(e0_x_Reboilerin_i2))) ,
    e0e1_M_V_tr1-(((e0e1_M_i1) *(e0_y_D_i1)+(e0e1_M_i2) *(e0_y_D_i2))) ,
    e0e1_M_V_tr2-(((e0e1_M_i1) *(e0e1_y_tr2_i1)+(e0e1_M_i2) *(e0e1_y_tr2_i2))) ,
    e0e1_M_V_tr3-(((e0e1_M_i1) *(e0e1_y_tr3_i1)+(e0e1_M_i2) *(e0e1_y_tr3_i2))) ,
    e0e1_M_V_tr4-(((e0e1_M_i1) *(e0e1_y_tr4_i1)+(e0e1_M_i2) *(e0e1_y_tr4_i2))) ,
    e0e1_M_V_tr5-(((e0e1_M_i1) *(e0e1_y_tr5_i1)+(e0e1_M_i2) *(e0e1_y_tr5_i2))) ,
    e0e1_M_V_tr6-(((e0e1_M_i1) *(e0e1_y_tr6_i1)+(e0e1_M_i2) *(e0e1_y_tr6_i2))) ,
    e0e1_M_V_tr7-(((e0e1_M_i1) *(e0e1_y_tr7_i1)+(e0e1_M_i2) *(e0e1_y_tr7_i2))) ,
    e0e1_M_V_tr8-(((e0e1_M_i1) *(e0e1_y_tr8_i1)+(e0e1_M_i2) *(e0e1_y_tr8_i2))) ,
    e0e1_M_V_tr9-(((e0e1_M_i1) *(e0e1_y_tr9_i1)+(e0e1_M_i2) *(e0e1_y_tr9_i2))) ,
    e0e1_M_V_tr10-(((e0e1_M_i1) *(e0e1_y_tr10_i1)+(e0e1_M_i2) *(e0e1_y_tr10_i2))) ,
    e0e1_M_V_tr11-(((e0e1_M_i1) *(e0e1_y_tr11_i1)+(e0e1_M_i2) *(e0e1_y_tr11_i2))) ,
    e0e1_M_V_tr12-(((e0e1_M_i1) *(e0e1_y_tr12_i1)+(e0e1_M_i2) *(e0e1_y_tr12_i2))) ,
    e0e1_M_V_tr13-(((e0e1_M_i1) *(e0e1_y_tr13_i1)+(e0e1_M_i2) *(e0e1_y_tr13_i2))) ,
    e0e1_M_V_tr14-(((e0e1_M_i1) *(e0e1_y_tr14_i1)+(e0e1_M_i2) *(e0e1_y_tr14_i2))) ,
    e0e1_M_V_tr15-(((e0e1_M_i1) *(e0e1_y_tr15_i1)+(e0e1_M_i2) *(e0e1_y_tr15_i2))) ,
    e0e1_M_V_tr16-(((e0e1_M_i1) *(e0e1_y_tr16_i1)+(e0e1_M_i2) *(e0e1_y_tr16_i2))) ,
    e0e1_M_V_tr17-(((e0e1_M_i1) *(e0e1_y_tr17_i1)+(e0e1_M_i2) *(e0e1_y_tr17_i2))) ,
    e0e1_M_V_tr18-(((e0e1_M_i1) *(e0e1_y_tr18_i1)+(e0e1_M_i2) *(e0e1_y_tr18_i2))) ,
    e0e1_M_V_tr19-(((e0e1_M_i1) *(e0e1_y_tr19_i1)+(e0e1_M_i2) *(e0e1_y_tr19_i2))) ,
    e0e1_M_V_tr20-(((e0e1_M_i1) *(e0e1_y_tr20_i1)+(e0e1_M_i2) *(e0e1_y_tr20_i2))) ,
    e0e1_greek_rho_Vm_tr1-(((e0e1_p_tr1) *(e0e1_M_V_tr1))/((e0e1_R) *((e0e1_T_tr1) *(1000.0)))) ,
    e0e1_greek_rho_Vm_tr2-(((e0e1_p_tr2) *(e0e1_M_V_tr2))/((e0e1_R) *((e0e1_T_tr2) *(1000.0)))) ,
    e0e1_greek_rho_Vm_tr3-(((e0e1_p_tr3) *(e0e1_M_V_tr3))/((e0e1_R) *((e0e1_T_tr3) *(1000.0)))) ,
    e0e1_greek_rho_Vm_tr4-(((e0e1_p_tr4) *(e0e1_M_V_tr4))/((e0e1_R) *((e0e1_T_tr4) *(1000.0)))) ,
    e0e1_greek_rho_Vm_tr5-(((e0e1_p_tr5) *(e0e1_M_V_tr5))/((e0e1_R) *((e0e1_T_tr5) *(1000.0)))) ,
    e0e1_greek_rho_Vm_tr6-(((e0e1_p_tr6) *(e0e1_M_V_tr6))/((e0e1_R) *((e0e1_T_tr6) *(1000.0)))) ,
    e0e1_greek_rho_Vm_tr7-(((e0e1_p_tr7) *(e0e1_M_V_tr7))/((e0e1_R) *((e0e1_T_tr7) *(1000.0)))) ,
    e0e1_greek_rho_Vm_tr8-(((e0e1_p_tr8) *(e0e1_M_V_tr8))/((e0e1_R) *((e0e1_T_tr8) *(1000.0)))) ,
    e0e1_greek_rho_Vm_tr9-(((e0e1_p_tr9) *(e0e1_M_V_tr9))/((e0e1_R) *((e0e1_T_tr9) *(1000.0)))) ,
    e0e1_greek_rho_Vm_tr10-(((e0e1_p_tr10) *(e0e1_M_V_tr10))/((e0e1_R) *((e0e1_T_tr10) *(1000.0)))) ,
    e0e1_greek_rho_Vm_tr11-(((e0e1_p_tr11) *(e0e1_M_V_tr11))/((e0e1_R) *((e0e1_T_tr11) *(1000.0)))) ,
    e0e1_greek_rho_Vm_tr12-(((e0e1_p_tr12) *(e0e1_M_V_tr12))/((e0e1_R) *((e0e1_T_tr12) *(1000.0)))) ,
    e0e1_greek_rho_Vm_tr13-(((e0e1_p_tr13) *(e0e1_M_V_tr13))/((e0e1_R) *((e0e1_T_tr13) *(1000.0)))) ,
    e0e1_greek_rho_Vm_tr14-(((e0e1_p_tr14) *(e0e1_M_V_tr14))/((e0e1_R) *((e0e1_T_tr14) *(1000.0)))) ,
    e0e1_greek_rho_Vm_tr15-(((e0e1_p_tr15) *(e0e1_M_V_tr15))/((e0e1_R) *((e0e1_T_tr15) *(1000.0)))) ,
    e0e1_greek_rho_Vm_tr16-(((e0e1_p_tr16) *(e0e1_M_V_tr16))/((e0e1_R) *((e0e1_T_tr16) *(1000.0)))) ,
    e0e1_greek_rho_Vm_tr17-(((e0e1_p_tr17) *(e0e1_M_V_tr17))/((e0e1_R) *((e0e1_T_tr17) *(1000.0)))) ,
    e0e1_greek_rho_Vm_tr18-(((e0e1_p_tr18) *(e0e1_M_V_tr18))/((e0e1_R) *((e0e1_T_tr18) *(1000.0)))) ,
    e0e1_greek_rho_Vm_tr19-(((e0e1_p_tr19) *(e0e1_M_V_tr19))/((e0e1_R) *((e0e1_T_tr19) *(1000.0)))) ,
    e0e1_greek_rho_Vm_tr20-(((e0_p_LReboiler) *(e0e1_M_V_tr20))/((e0e1_R) *((e0e1_T_tr20) *(1000.0)))) ,
    1.0-(((e0e1_greek_rho_Lm_tr1)/(e0e1_M_L_tr1)) *((((e0e1_x_tr1_i1)/(e0e1_greek_rho_Ln_tr1_i1)+(e0e1_x_tr1_i2)/(e0e1_greek_rho_Ln_tr1_i2))))) ,
    1.0-(((e0e1_greek_rho_Lm_tr2)/(e0e1_M_L_tr2)) *((((e0e1_x_tr2_i1)/(e0e1_greek_rho_Ln_tr2_i1)+(e0e1_x_tr2_i2)/(e0e1_greek_rho_Ln_tr2_i2))))) ,
    1.0-(((e0e1_greek_rho_Lm_tr3)/(e0e1_M_L_tr3)) *((((e0e1_x_tr3_i1)/(e0e1_greek_rho_Ln_tr3_i1)+(e0e1_x_tr3_i2)/(e0e1_greek_rho_Ln_tr3_i2))))) ,
    1.0-(((e0e1_greek_rho_Lm_tr4)/(e0e1_M_L_tr4)) *((((e0e1_x_tr4_i1)/(e0e1_greek_rho_Ln_tr4_i1)+(e0e1_x_tr4_i2)/(e0e1_greek_rho_Ln_tr4_i2))))) ,
    1.0-(((e0e1_greek_rho_Lm_tr5)/(e0e1_M_L_tr5)) *((((e0e1_x_tr5_i1)/(e0e1_greek_rho_Ln_tr5_i1)+(e0e1_x_tr5_i2)/(e0e1_greek_rho_Ln_tr5_i2))))) ,
    1.0-(((e0e1_greek_rho_Lm_tr6)/(e0e1_M_L_tr6)) *((((e0e1_x_tr6_i1)/(e0e1_greek_rho_Ln_tr6_i1)+(e0e1_x_tr6_i2)/(e0e1_greek_rho_Ln_tr6_i2))))) ,
    1.0-(((e0e1_greek_rho_Lm_tr7)/(e0e1_M_L_tr7)) *((((e0e1_x_tr7_i1)/(e0e1_greek_rho_Ln_tr7_i1)+(e0e1_x_tr7_i2)/(e0e1_greek_rho_Ln_tr7_i2))))) ,
    1.0-(((e0e1_greek_rho_Lm_tr8)/(e0e1_M_L_tr8)) *((((e0e1_x_tr8_i1)/(e0e1_greek_rho_Ln_tr8_i1)+(e0e1_x_tr8_i2)/(e0e1_greek_rho_Ln_tr8_i2))))) ,
    1.0-(((e0e1_greek_rho_Lm_tr9)/(e0e1_M_L_tr9)) *((((e0e1_x_tr9_i1)/(e0e1_greek_rho_Ln_tr9_i1)+(e0e1_x_tr9_i2)/(e0e1_greek_rho_Ln_tr9_i2))))) ,
    1.0-(((e0e1_greek_rho_Lm_tr10)/(e0e1_M_L_tr10)) *((((e0e1_x_tr10_i1)/(e0e1_greek_rho_Ln_tr10_i1)+(e0e1_x_tr10_i2)/(e0e1_greek_rho_Ln_tr10_i2))))) ,
    1.0-(((e0e1_greek_rho_Lm_tr11)/(e0e1_M_L_tr11)) *((((e0e1_x_tr11_i1)/(e0e1_greek_rho_Ln_tr11_i1)+(e0e1_x_tr11_i2)/(e0e1_greek_rho_Ln_tr11_i2))))) ,
    1.0-(((e0e1_greek_rho_Lm_tr12)/(e0e1_M_L_tr12)) *((((e0e1_x_tr12_i1)/(e0e1_greek_rho_Ln_tr12_i1)+(e0e1_x_tr12_i2)/(e0e1_greek_rho_Ln_tr12_i2))))) ,
    1.0-(((e0e1_greek_rho_Lm_tr13)/(e0e1_M_L_tr13)) *((((e0e1_x_tr13_i1)/(e0e1_greek_rho_Ln_tr13_i1)+(e0e1_x_tr13_i2)/(e0e1_greek_rho_Ln_tr13_i2))))) ,
    1.0-(((e0e1_greek_rho_Lm_tr14)/(e0e1_M_L_tr14)) *((((e0e1_x_tr14_i1)/(e0e1_greek_rho_Ln_tr14_i1)+(e0e1_x_tr14_i2)/(e0e1_greek_rho_Ln_tr14_i2))))) ,
    1.0-(((e0e1_greek_rho_Lm_tr15)/(e0e1_M_L_tr15)) *((((e0e1_x_tr15_i1)/(e0e1_greek_rho_Ln_tr15_i1)+(e0e1_x_tr15_i2)/(e0e1_greek_rho_Ln_tr15_i2))))) ,
    1.0-(((e0e1_greek_rho_Lm_tr16)/(e0e1_M_L_tr16)) *((((e0e1_x_tr16_i1)/(e0e1_greek_rho_Ln_tr16_i1)+(e0e1_x_tr16_i2)/(e0e1_greek_rho_Ln_tr16_i2))))) ,
    1.0-(((e0e1_greek_rho_Lm_tr17)/(e0e1_M_L_tr17)) *((((e0e1_x_tr17_i1)/(e0e1_greek_rho_Ln_tr17_i1)+(e0e1_x_tr17_i2)/(e0e1_greek_rho_Ln_tr17_i2))))) ,
    1.0-(((e0e1_greek_rho_Lm_tr18)/(e0e1_M_L_tr18)) *((((e0e1_x_tr18_i1)/(e0e1_greek_rho_Ln_tr18_i1)+(e0e1_x_tr18_i2)/(e0e1_greek_rho_Ln_tr18_i2))))) ,
    1.0-(((e0e1_greek_rho_Lm_tr19)/(e0e1_M_L_tr19)) *((((e0e1_x_tr19_i1)/(e0e1_greek_rho_Ln_tr19_i1)+(e0e1_x_tr19_i2)/(e0e1_greek_rho_Ln_tr19_i2))))) ,
    1.0-(((e0e1_greek_rho_Lm_tr20)/(e0e1_M_L_tr20)) *((((e0_x_Reboilerin_i1)/(e0e1_greek_rho_Ln_tr20_i1)+(e0_x_Reboilerin_i2)/(e0e1_greek_rho_Ln_tr20_i2))))) ,
    e0e1_greek_theta_i1_j1_tr1-((((1.0+((((e0e1_greek_eta_V_i1_tr1)/(e0e1_greek_eta_V_i1_tr1)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i1_j1_tr2-((((1.0+((((e0e1_greek_eta_V_i1_tr2)/(e0e1_greek_eta_V_i1_tr2)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i1_j1_tr3-((((1.0+((((e0e1_greek_eta_V_i1_tr3)/(e0e1_greek_eta_V_i1_tr3)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i1_j1_tr4-((((1.0+((((e0e1_greek_eta_V_i1_tr4)/(e0e1_greek_eta_V_i1_tr4)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i1_j1_tr5-((((1.0+((((e0e1_greek_eta_V_i1_tr5)/(e0e1_greek_eta_V_i1_tr5)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i1_j1_tr6-((((1.0+((((e0e1_greek_eta_V_i1_tr6)/(e0e1_greek_eta_V_i1_tr6)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i1_j1_tr7-((((1.0+((((e0e1_greek_eta_V_i1_tr7)/(e0e1_greek_eta_V_i1_tr7)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i1_j1_tr8-((((1.0+((((e0e1_greek_eta_V_i1_tr8)/(e0e1_greek_eta_V_i1_tr8)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i1_j1_tr9-((((1.0+((((e0e1_greek_eta_V_i1_tr9)/(e0e1_greek_eta_V_i1_tr9)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i1_j1_tr10-((((1.0+((((e0e1_greek_eta_V_i1_tr10)/(e0e1_greek_eta_V_i1_tr10)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i1_j1_tr11-((((1.0+((((e0e1_greek_eta_V_i1_tr11)/(e0e1_greek_eta_V_i1_tr11)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i1_j1_tr12-((((1.0+((((e0e1_greek_eta_V_i1_tr12)/(e0e1_greek_eta_V_i1_tr12)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i1_j1_tr13-((((1.0+((((e0e1_greek_eta_V_i1_tr13)/(e0e1_greek_eta_V_i1_tr13)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i1_j1_tr14-((((1.0+((((e0e1_greek_eta_V_i1_tr14)/(e0e1_greek_eta_V_i1_tr14)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i1_j1_tr15-((((1.0+((((e0e1_greek_eta_V_i1_tr15)/(e0e1_greek_eta_V_i1_tr15)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i1_j1_tr16-((((1.0+((((e0e1_greek_eta_V_i1_tr16)/(e0e1_greek_eta_V_i1_tr16)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i1_j1_tr17-((((1.0+((((e0e1_greek_eta_V_i1_tr17)/(e0e1_greek_eta_V_i1_tr17)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i1_j1_tr18-((((1.0+((((e0e1_greek_eta_V_i1_tr18)/(e0e1_greek_eta_V_i1_tr18)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i1_j1_tr19-((((1.0+((((e0e1_greek_eta_V_i1_tr19)/(e0e1_greek_eta_V_i1_tr19)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i1_j1_tr20-((((1.0+((((e0e1_greek_eta_V_i1_tr20)/(e0e1_greek_eta_V_i1_tr20)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i1_j2_tr1-((((1.0+((((e0e1_greek_eta_V_i1_tr1)/(e0e1_greek_eta_V_i2_tr1)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i1_j2_tr2-((((1.0+((((e0e1_greek_eta_V_i1_tr2)/(e0e1_greek_eta_V_i2_tr2)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i1_j2_tr3-((((1.0+((((e0e1_greek_eta_V_i1_tr3)/(e0e1_greek_eta_V_i2_tr3)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i1_j2_tr4-((((1.0+((((e0e1_greek_eta_V_i1_tr4)/(e0e1_greek_eta_V_i2_tr4)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i1_j2_tr5-((((1.0+((((e0e1_greek_eta_V_i1_tr5)/(e0e1_greek_eta_V_i2_tr5)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i1_j2_tr6-((((1.0+((((e0e1_greek_eta_V_i1_tr6)/(e0e1_greek_eta_V_i2_tr6)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i1_j2_tr7-((((1.0+((((e0e1_greek_eta_V_i1_tr7)/(e0e1_greek_eta_V_i2_tr7)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i1_j2_tr8-((((1.0+((((e0e1_greek_eta_V_i1_tr8)/(e0e1_greek_eta_V_i2_tr8)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i1_j2_tr9-((((1.0+((((e0e1_greek_eta_V_i1_tr9)/(e0e1_greek_eta_V_i2_tr9)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i1_j2_tr10-((((1.0+((((e0e1_greek_eta_V_i1_tr10)/(e0e1_greek_eta_V_i2_tr10)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i1_j2_tr11-((((1.0+((((e0e1_greek_eta_V_i1_tr11)/(e0e1_greek_eta_V_i2_tr11)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i1_j2_tr12-((((1.0+((((e0e1_greek_eta_V_i1_tr12)/(e0e1_greek_eta_V_i2_tr12)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i1_j2_tr13-((((1.0+((((e0e1_greek_eta_V_i1_tr13)/(e0e1_greek_eta_V_i2_tr13)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i1_j2_tr14-((((1.0+((((e0e1_greek_eta_V_i1_tr14)/(e0e1_greek_eta_V_i2_tr14)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i1_j2_tr15-((((1.0+((((e0e1_greek_eta_V_i1_tr15)/(e0e1_greek_eta_V_i2_tr15)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i1_j2_tr16-((((1.0+((((e0e1_greek_eta_V_i1_tr16)/(e0e1_greek_eta_V_i2_tr16)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i1_j2_tr17-((((1.0+((((e0e1_greek_eta_V_i1_tr17)/(e0e1_greek_eta_V_i2_tr17)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i1_j2_tr18-((((1.0+((((e0e1_greek_eta_V_i1_tr18)/(e0e1_greek_eta_V_i2_tr18)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i1_j2_tr19-((((1.0+((((e0e1_greek_eta_V_i1_tr19)/(e0e1_greek_eta_V_i2_tr19)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i1_j2_tr20-((((1.0+((((e0e1_greek_eta_V_i1_tr20)/(e0e1_greek_eta_V_i2_tr20)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i1)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i1)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i2_j1_tr1-((((1.0+((((e0e1_greek_eta_V_i2_tr1)/(e0e1_greek_eta_V_i1_tr1)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i2_j1_tr2-((((1.0+((((e0e1_greek_eta_V_i2_tr2)/(e0e1_greek_eta_V_i1_tr2)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i2_j1_tr3-((((1.0+((((e0e1_greek_eta_V_i2_tr3)/(e0e1_greek_eta_V_i1_tr3)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i2_j1_tr4-((((1.0+((((e0e1_greek_eta_V_i2_tr4)/(e0e1_greek_eta_V_i1_tr4)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i2_j1_tr5-((((1.0+((((e0e1_greek_eta_V_i2_tr5)/(e0e1_greek_eta_V_i1_tr5)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i2_j1_tr6-((((1.0+((((e0e1_greek_eta_V_i2_tr6)/(e0e1_greek_eta_V_i1_tr6)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i2_j1_tr7-((((1.0+((((e0e1_greek_eta_V_i2_tr7)/(e0e1_greek_eta_V_i1_tr7)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i2_j1_tr8-((((1.0+((((e0e1_greek_eta_V_i2_tr8)/(e0e1_greek_eta_V_i1_tr8)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i2_j1_tr9-((((1.0+((((e0e1_greek_eta_V_i2_tr9)/(e0e1_greek_eta_V_i1_tr9)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i2_j1_tr10-((((1.0+((((e0e1_greek_eta_V_i2_tr10)/(e0e1_greek_eta_V_i1_tr10)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i2_j1_tr11-((((1.0+((((e0e1_greek_eta_V_i2_tr11)/(e0e1_greek_eta_V_i1_tr11)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i2_j1_tr12-((((1.0+((((e0e1_greek_eta_V_i2_tr12)/(e0e1_greek_eta_V_i1_tr12)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i2_j1_tr13-((((1.0+((((e0e1_greek_eta_V_i2_tr13)/(e0e1_greek_eta_V_i1_tr13)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i2_j1_tr14-((((1.0+((((e0e1_greek_eta_V_i2_tr14)/(e0e1_greek_eta_V_i1_tr14)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i2_j1_tr15-((((1.0+((((e0e1_greek_eta_V_i2_tr15)/(e0e1_greek_eta_V_i1_tr15)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i2_j1_tr16-((((1.0+((((e0e1_greek_eta_V_i2_tr16)/(e0e1_greek_eta_V_i1_tr16)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i2_j1_tr17-((((1.0+((((e0e1_greek_eta_V_i2_tr17)/(e0e1_greek_eta_V_i1_tr17)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i2_j1_tr18-((((1.0+((((e0e1_greek_eta_V_i2_tr18)/(e0e1_greek_eta_V_i1_tr18)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i2_j1_tr19-((((1.0+((((e0e1_greek_eta_V_i2_tr19)/(e0e1_greek_eta_V_i1_tr19)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i2_j1_tr20-((((1.0+((((e0e1_greek_eta_V_i2_tr20)/(e0e1_greek_eta_V_i1_tr20)))**(0.5)) *((((e0e1_M_i1)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i1)))))**(0.5))) ,
    e0e1_greek_theta_i2_j2_tr1-((((1.0+((((e0e1_greek_eta_V_i2_tr1)/(e0e1_greek_eta_V_i2_tr1)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i2_j2_tr2-((((1.0+((((e0e1_greek_eta_V_i2_tr2)/(e0e1_greek_eta_V_i2_tr2)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i2_j2_tr3-((((1.0+((((e0e1_greek_eta_V_i2_tr3)/(e0e1_greek_eta_V_i2_tr3)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i2_j2_tr4-((((1.0+((((e0e1_greek_eta_V_i2_tr4)/(e0e1_greek_eta_V_i2_tr4)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i2_j2_tr5-((((1.0+((((e0e1_greek_eta_V_i2_tr5)/(e0e1_greek_eta_V_i2_tr5)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i2_j2_tr6-((((1.0+((((e0e1_greek_eta_V_i2_tr6)/(e0e1_greek_eta_V_i2_tr6)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i2_j2_tr7-((((1.0+((((e0e1_greek_eta_V_i2_tr7)/(e0e1_greek_eta_V_i2_tr7)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i2_j2_tr8-((((1.0+((((e0e1_greek_eta_V_i2_tr8)/(e0e1_greek_eta_V_i2_tr8)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i2_j2_tr9-((((1.0+((((e0e1_greek_eta_V_i2_tr9)/(e0e1_greek_eta_V_i2_tr9)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i2_j2_tr10-((((1.0+((((e0e1_greek_eta_V_i2_tr10)/(e0e1_greek_eta_V_i2_tr10)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i2_j2_tr11-((((1.0+((((e0e1_greek_eta_V_i2_tr11)/(e0e1_greek_eta_V_i2_tr11)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i2_j2_tr12-((((1.0+((((e0e1_greek_eta_V_i2_tr12)/(e0e1_greek_eta_V_i2_tr12)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i2_j2_tr13-((((1.0+((((e0e1_greek_eta_V_i2_tr13)/(e0e1_greek_eta_V_i2_tr13)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i2_j2_tr14-((((1.0+((((e0e1_greek_eta_V_i2_tr14)/(e0e1_greek_eta_V_i2_tr14)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i2_j2_tr15-((((1.0+((((e0e1_greek_eta_V_i2_tr15)/(e0e1_greek_eta_V_i2_tr15)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i2_j2_tr16-((((1.0+((((e0e1_greek_eta_V_i2_tr16)/(e0e1_greek_eta_V_i2_tr16)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i2_j2_tr17-((((1.0+((((e0e1_greek_eta_V_i2_tr17)/(e0e1_greek_eta_V_i2_tr17)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i2_j2_tr18-((((1.0+((((e0e1_greek_eta_V_i2_tr18)/(e0e1_greek_eta_V_i2_tr18)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i2_j2_tr19-((((1.0+((((e0e1_greek_eta_V_i2_tr19)/(e0e1_greek_eta_V_i2_tr19)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_theta_i2_j2_tr20-((((1.0+((((e0e1_greek_eta_V_i2_tr20)/(e0e1_greek_eta_V_i2_tr20)))**(0.5)) *((((e0e1_M_i2)/(e0e1_M_i2)))**(0.25))))**(2.0))/((((8.0) *((1.0+(e0e1_M_i2)/(e0e1_M_i2)))))**(0.5))) ,
    e0e1_greek_eta_V_tr1-((((e0_y_D_i1) *(e0e1_greek_eta_V_i1_tr1))/(((e0_y_D_i1) *(e0e1_greek_theta_i1_j1_tr1)+(e0_y_D_i2) *(e0e1_greek_theta_i1_j2_tr1)))+((e0_y_D_i2) *(e0e1_greek_eta_V_i2_tr1))/(((e0_y_D_i1) *(e0e1_greek_theta_i2_j1_tr1)+(e0_y_D_i2) *(e0e1_greek_theta_i2_j2_tr1))))) ,
    e0e1_greek_eta_V_tr2-((((e0e1_y_tr2_i1) *(e0e1_greek_eta_V_i1_tr2))/(((e0e1_y_tr2_i1) *(e0e1_greek_theta_i1_j1_tr2)+(e0e1_y_tr2_i2) *(e0e1_greek_theta_i1_j2_tr2)))+((e0e1_y_tr2_i2) *(e0e1_greek_eta_V_i2_tr2))/(((e0e1_y_tr2_i1) *(e0e1_greek_theta_i2_j1_tr2)+(e0e1_y_tr2_i2) *(e0e1_greek_theta_i2_j2_tr2))))) ,
    e0e1_greek_eta_V_tr3-((((e0e1_y_tr3_i1) *(e0e1_greek_eta_V_i1_tr3))/(((e0e1_y_tr3_i1) *(e0e1_greek_theta_i1_j1_tr3)+(e0e1_y_tr3_i2) *(e0e1_greek_theta_i1_j2_tr3)))+((e0e1_y_tr3_i2) *(e0e1_greek_eta_V_i2_tr3))/(((e0e1_y_tr3_i1) *(e0e1_greek_theta_i2_j1_tr3)+(e0e1_y_tr3_i2) *(e0e1_greek_theta_i2_j2_tr3))))) ,
    e0e1_greek_eta_V_tr4-((((e0e1_y_tr4_i1) *(e0e1_greek_eta_V_i1_tr4))/(((e0e1_y_tr4_i1) *(e0e1_greek_theta_i1_j1_tr4)+(e0e1_y_tr4_i2) *(e0e1_greek_theta_i1_j2_tr4)))+((e0e1_y_tr4_i2) *(e0e1_greek_eta_V_i2_tr4))/(((e0e1_y_tr4_i1) *(e0e1_greek_theta_i2_j1_tr4)+(e0e1_y_tr4_i2) *(e0e1_greek_theta_i2_j2_tr4))))) ,
    e0e1_greek_eta_V_tr5-((((e0e1_y_tr5_i1) *(e0e1_greek_eta_V_i1_tr5))/(((e0e1_y_tr5_i1) *(e0e1_greek_theta_i1_j1_tr5)+(e0e1_y_tr5_i2) *(e0e1_greek_theta_i1_j2_tr5)))+((e0e1_y_tr5_i2) *(e0e1_greek_eta_V_i2_tr5))/(((e0e1_y_tr5_i1) *(e0e1_greek_theta_i2_j1_tr5)+(e0e1_y_tr5_i2) *(e0e1_greek_theta_i2_j2_tr5))))) ,
    e0e1_greek_eta_V_tr6-((((e0e1_y_tr6_i1) *(e0e1_greek_eta_V_i1_tr6))/(((e0e1_y_tr6_i1) *(e0e1_greek_theta_i1_j1_tr6)+(e0e1_y_tr6_i2) *(e0e1_greek_theta_i1_j2_tr6)))+((e0e1_y_tr6_i2) *(e0e1_greek_eta_V_i2_tr6))/(((e0e1_y_tr6_i1) *(e0e1_greek_theta_i2_j1_tr6)+(e0e1_y_tr6_i2) *(e0e1_greek_theta_i2_j2_tr6))))) ,
    e0e1_greek_eta_V_tr7-((((e0e1_y_tr7_i1) *(e0e1_greek_eta_V_i1_tr7))/(((e0e1_y_tr7_i1) *(e0e1_greek_theta_i1_j1_tr7)+(e0e1_y_tr7_i2) *(e0e1_greek_theta_i1_j2_tr7)))+((e0e1_y_tr7_i2) *(e0e1_greek_eta_V_i2_tr7))/(((e0e1_y_tr7_i1) *(e0e1_greek_theta_i2_j1_tr7)+(e0e1_y_tr7_i2) *(e0e1_greek_theta_i2_j2_tr7))))) ,
    e0e1_greek_eta_V_tr8-((((e0e1_y_tr8_i1) *(e0e1_greek_eta_V_i1_tr8))/(((e0e1_y_tr8_i1) *(e0e1_greek_theta_i1_j1_tr8)+(e0e1_y_tr8_i2) *(e0e1_greek_theta_i1_j2_tr8)))+((e0e1_y_tr8_i2) *(e0e1_greek_eta_V_i2_tr8))/(((e0e1_y_tr8_i1) *(e0e1_greek_theta_i2_j1_tr8)+(e0e1_y_tr8_i2) *(e0e1_greek_theta_i2_j2_tr8))))) ,
    e0e1_greek_eta_V_tr9-((((e0e1_y_tr9_i1) *(e0e1_greek_eta_V_i1_tr9))/(((e0e1_y_tr9_i1) *(e0e1_greek_theta_i1_j1_tr9)+(e0e1_y_tr9_i2) *(e0e1_greek_theta_i1_j2_tr9)))+((e0e1_y_tr9_i2) *(e0e1_greek_eta_V_i2_tr9))/(((e0e1_y_tr9_i1) *(e0e1_greek_theta_i2_j1_tr9)+(e0e1_y_tr9_i2) *(e0e1_greek_theta_i2_j2_tr9))))) ,
    e0e1_greek_eta_V_tr10-((((e0e1_y_tr10_i1) *(e0e1_greek_eta_V_i1_tr10))/(((e0e1_y_tr10_i1) *(e0e1_greek_theta_i1_j1_tr10)+(e0e1_y_tr10_i2) *(e0e1_greek_theta_i1_j2_tr10)))+((e0e1_y_tr10_i2) *(e0e1_greek_eta_V_i2_tr10))/(((e0e1_y_tr10_i1) *(e0e1_greek_theta_i2_j1_tr10)+(e0e1_y_tr10_i2) *(e0e1_greek_theta_i2_j2_tr10))))) ,
    e0e1_greek_eta_V_tr11-((((e0e1_y_tr11_i1) *(e0e1_greek_eta_V_i1_tr11))/(((e0e1_y_tr11_i1) *(e0e1_greek_theta_i1_j1_tr11)+(e0e1_y_tr11_i2) *(e0e1_greek_theta_i1_j2_tr11)))+((e0e1_y_tr11_i2) *(e0e1_greek_eta_V_i2_tr11))/(((e0e1_y_tr11_i1) *(e0e1_greek_theta_i2_j1_tr11)+(e0e1_y_tr11_i2) *(e0e1_greek_theta_i2_j2_tr11))))) ,
    e0e1_greek_eta_V_tr12-((((e0e1_y_tr12_i1) *(e0e1_greek_eta_V_i1_tr12))/(((e0e1_y_tr12_i1) *(e0e1_greek_theta_i1_j1_tr12)+(e0e1_y_tr12_i2) *(e0e1_greek_theta_i1_j2_tr12)))+((e0e1_y_tr12_i2) *(e0e1_greek_eta_V_i2_tr12))/(((e0e1_y_tr12_i1) *(e0e1_greek_theta_i2_j1_tr12)+(e0e1_y_tr12_i2) *(e0e1_greek_theta_i2_j2_tr12))))) ,
    e0e1_greek_eta_V_tr13-((((e0e1_y_tr13_i1) *(e0e1_greek_eta_V_i1_tr13))/(((e0e1_y_tr13_i1) *(e0e1_greek_theta_i1_j1_tr13)+(e0e1_y_tr13_i2) *(e0e1_greek_theta_i1_j2_tr13)))+((e0e1_y_tr13_i2) *(e0e1_greek_eta_V_i2_tr13))/(((e0e1_y_tr13_i1) *(e0e1_greek_theta_i2_j1_tr13)+(e0e1_y_tr13_i2) *(e0e1_greek_theta_i2_j2_tr13))))) ,
    e0e1_greek_eta_V_tr14-((((e0e1_y_tr14_i1) *(e0e1_greek_eta_V_i1_tr14))/(((e0e1_y_tr14_i1) *(e0e1_greek_theta_i1_j1_tr14)+(e0e1_y_tr14_i2) *(e0e1_greek_theta_i1_j2_tr14)))+((e0e1_y_tr14_i2) *(e0e1_greek_eta_V_i2_tr14))/(((e0e1_y_tr14_i1) *(e0e1_greek_theta_i2_j1_tr14)+(e0e1_y_tr14_i2) *(e0e1_greek_theta_i2_j2_tr14))))) ,
    e0e1_greek_eta_V_tr15-((((e0e1_y_tr15_i1) *(e0e1_greek_eta_V_i1_tr15))/(((e0e1_y_tr15_i1) *(e0e1_greek_theta_i1_j1_tr15)+(e0e1_y_tr15_i2) *(e0e1_greek_theta_i1_j2_tr15)))+((e0e1_y_tr15_i2) *(e0e1_greek_eta_V_i2_tr15))/(((e0e1_y_tr15_i1) *(e0e1_greek_theta_i2_j1_tr15)+(e0e1_y_tr15_i2) *(e0e1_greek_theta_i2_j2_tr15))))) ,
    e0e1_greek_eta_V_tr16-((((e0e1_y_tr16_i1) *(e0e1_greek_eta_V_i1_tr16))/(((e0e1_y_tr16_i1) *(e0e1_greek_theta_i1_j1_tr16)+(e0e1_y_tr16_i2) *(e0e1_greek_theta_i1_j2_tr16)))+((e0e1_y_tr16_i2) *(e0e1_greek_eta_V_i2_tr16))/(((e0e1_y_tr16_i1) *(e0e1_greek_theta_i2_j1_tr16)+(e0e1_y_tr16_i2) *(e0e1_greek_theta_i2_j2_tr16))))) ,
    e0e1_greek_eta_V_tr17-((((e0e1_y_tr17_i1) *(e0e1_greek_eta_V_i1_tr17))/(((e0e1_y_tr17_i1) *(e0e1_greek_theta_i1_j1_tr17)+(e0e1_y_tr17_i2) *(e0e1_greek_theta_i1_j2_tr17)))+((e0e1_y_tr17_i2) *(e0e1_greek_eta_V_i2_tr17))/(((e0e1_y_tr17_i1) *(e0e1_greek_theta_i2_j1_tr17)+(e0e1_y_tr17_i2) *(e0e1_greek_theta_i2_j2_tr17))))) ,
    e0e1_greek_eta_V_tr18-((((e0e1_y_tr18_i1) *(e0e1_greek_eta_V_i1_tr18))/(((e0e1_y_tr18_i1) *(e0e1_greek_theta_i1_j1_tr18)+(e0e1_y_tr18_i2) *(e0e1_greek_theta_i1_j2_tr18)))+((e0e1_y_tr18_i2) *(e0e1_greek_eta_V_i2_tr18))/(((e0e1_y_tr18_i1) *(e0e1_greek_theta_i2_j1_tr18)+(e0e1_y_tr18_i2) *(e0e1_greek_theta_i2_j2_tr18))))) ,
    e0e1_greek_eta_V_tr19-((((e0e1_y_tr19_i1) *(e0e1_greek_eta_V_i1_tr19))/(((e0e1_y_tr19_i1) *(e0e1_greek_theta_i1_j1_tr19)+(e0e1_y_tr19_i2) *(e0e1_greek_theta_i1_j2_tr19)))+((e0e1_y_tr19_i2) *(e0e1_greek_eta_V_i2_tr19))/(((e0e1_y_tr19_i1) *(e0e1_greek_theta_i2_j1_tr19)+(e0e1_y_tr19_i2) *(e0e1_greek_theta_i2_j2_tr19))))) ,
    e0e1_greek_eta_V_tr20-((((e0e1_y_tr20_i1) *(e0e1_greek_eta_V_i1_tr20))/(((e0e1_y_tr20_i1) *(e0e1_greek_theta_i1_j1_tr20)+(e0e1_y_tr20_i2) *(e0e1_greek_theta_i1_j2_tr20)))+((e0e1_y_tr20_i2) *(e0e1_greek_eta_V_i2_tr20))/(((e0e1_y_tr20_i1) *(e0e1_greek_theta_i2_j1_tr20)+(e0e1_y_tr20_i2) *(e0e1_greek_theta_i2_j2_tr20))))) ,
    sympy.log(e0e1_greek_eta_L_tr1)-(((e0e1_x_tr1_i1) *(sympy.log(e0e1_greek_eta_L_tr1_i1))+(e0e1_x_tr1_i2) *(sympy.log(e0e1_greek_eta_L_tr1_i2)))) ,
    sympy.log(e0e1_greek_eta_L_tr2)-(((e0e1_x_tr2_i1) *(sympy.log(e0e1_greek_eta_L_tr2_i1))+(e0e1_x_tr2_i2) *(sympy.log(e0e1_greek_eta_L_tr2_i2)))) ,
    sympy.log(e0e1_greek_eta_L_tr3)-(((e0e1_x_tr3_i1) *(sympy.log(e0e1_greek_eta_L_tr3_i1))+(e0e1_x_tr3_i2) *(sympy.log(e0e1_greek_eta_L_tr3_i2)))) ,
    sympy.log(e0e1_greek_eta_L_tr4)-(((e0e1_x_tr4_i1) *(sympy.log(e0e1_greek_eta_L_tr4_i1))+(e0e1_x_tr4_i2) *(sympy.log(e0e1_greek_eta_L_tr4_i2)))) ,
    sympy.log(e0e1_greek_eta_L_tr5)-(((e0e1_x_tr5_i1) *(sympy.log(e0e1_greek_eta_L_tr5_i1))+(e0e1_x_tr5_i2) *(sympy.log(e0e1_greek_eta_L_tr5_i2)))) ,
    sympy.log(e0e1_greek_eta_L_tr6)-(((e0e1_x_tr6_i1) *(sympy.log(e0e1_greek_eta_L_tr6_i1))+(e0e1_x_tr6_i2) *(sympy.log(e0e1_greek_eta_L_tr6_i2)))) ,
    sympy.log(e0e1_greek_eta_L_tr7)-(((e0e1_x_tr7_i1) *(sympy.log(e0e1_greek_eta_L_tr7_i1))+(e0e1_x_tr7_i2) *(sympy.log(e0e1_greek_eta_L_tr7_i2)))) ,
    sympy.log(e0e1_greek_eta_L_tr8)-(((e0e1_x_tr8_i1) *(sympy.log(e0e1_greek_eta_L_tr8_i1))+(e0e1_x_tr8_i2) *(sympy.log(e0e1_greek_eta_L_tr8_i2)))) ,
    sympy.log(e0e1_greek_eta_L_tr9)-(((e0e1_x_tr9_i1) *(sympy.log(e0e1_greek_eta_L_tr9_i1))+(e0e1_x_tr9_i2) *(sympy.log(e0e1_greek_eta_L_tr9_i2)))) ,
    sympy.log(e0e1_greek_eta_L_tr10)-(((e0e1_x_tr10_i1) *(sympy.log(e0e1_greek_eta_L_tr10_i1))+(e0e1_x_tr10_i2) *(sympy.log(e0e1_greek_eta_L_tr10_i2)))) ,
    sympy.log(e0e1_greek_eta_L_tr11)-(((e0e1_x_tr11_i1) *(sympy.log(e0e1_greek_eta_L_tr11_i1))+(e0e1_x_tr11_i2) *(sympy.log(e0e1_greek_eta_L_tr11_i2)))) ,
    sympy.log(e0e1_greek_eta_L_tr12)-(((e0e1_x_tr12_i1) *(sympy.log(e0e1_greek_eta_L_tr12_i1))+(e0e1_x_tr12_i2) *(sympy.log(e0e1_greek_eta_L_tr12_i2)))) ,
    sympy.log(e0e1_greek_eta_L_tr13)-(((e0e1_x_tr13_i1) *(sympy.log(e0e1_greek_eta_L_tr13_i1))+(e0e1_x_tr13_i2) *(sympy.log(e0e1_greek_eta_L_tr13_i2)))) ,
    sympy.log(e0e1_greek_eta_L_tr14)-(((e0e1_x_tr14_i1) *(sympy.log(e0e1_greek_eta_L_tr14_i1))+(e0e1_x_tr14_i2) *(sympy.log(e0e1_greek_eta_L_tr14_i2)))) ,
    sympy.log(e0e1_greek_eta_L_tr15)-(((e0e1_x_tr15_i1) *(sympy.log(e0e1_greek_eta_L_tr15_i1))+(e0e1_x_tr15_i2) *(sympy.log(e0e1_greek_eta_L_tr15_i2)))) ,
    sympy.log(e0e1_greek_eta_L_tr16)-(((e0e1_x_tr16_i1) *(sympy.log(e0e1_greek_eta_L_tr16_i1))+(e0e1_x_tr16_i2) *(sympy.log(e0e1_greek_eta_L_tr16_i2)))) ,
    sympy.log(e0e1_greek_eta_L_tr17)-(((e0e1_x_tr17_i1) *(sympy.log(e0e1_greek_eta_L_tr17_i1))+(e0e1_x_tr17_i2) *(sympy.log(e0e1_greek_eta_L_tr17_i2)))) ,
    sympy.log(e0e1_greek_eta_L_tr18)-(((e0e1_x_tr18_i1) *(sympy.log(e0e1_greek_eta_L_tr18_i1))+(e0e1_x_tr18_i2) *(sympy.log(e0e1_greek_eta_L_tr18_i2)))) ,
    sympy.log(e0e1_greek_eta_L_tr19)-(((e0e1_x_tr19_i1) *(sympy.log(e0e1_greek_eta_L_tr19_i1))+(e0e1_x_tr19_i2) *(sympy.log(e0e1_greek_eta_L_tr19_i2)))) ,
    sympy.log(e0e1_greek_eta_L_tr20)-(((e0_x_Reboilerin_i1) *(sympy.log(e0e1_greek_eta_L_tr20_i1))+(e0_x_Reboilerin_i2) *(sympy.log(e0e1_greek_eta_L_tr20_i2)))) ,
    e0e1_greek_sigma_tr1-(((1.0)/(1000.0)) *(((((e0e1_P_tr1) *(e0e1_greek_rho_Lm_tr1))/((e0e1_M_L_tr1) *(((10.0))**(3.0)))))**(4.0))) ,
    e0e1_greek_sigma_tr2-(((1.0)/(1000.0)) *(((((e0e1_P_tr2) *(e0e1_greek_rho_Lm_tr2))/((e0e1_M_L_tr2) *(((10.0))**(3.0)))))**(4.0))) ,
    e0e1_greek_sigma_tr3-(((1.0)/(1000.0)) *(((((e0e1_P_tr3) *(e0e1_greek_rho_Lm_tr3))/((e0e1_M_L_tr3) *(((10.0))**(3.0)))))**(4.0))) ,
    e0e1_greek_sigma_tr4-(((1.0)/(1000.0)) *(((((e0e1_P_tr4) *(e0e1_greek_rho_Lm_tr4))/((e0e1_M_L_tr4) *(((10.0))**(3.0)))))**(4.0))) ,
    e0e1_greek_sigma_tr5-(((1.0)/(1000.0)) *(((((e0e1_P_tr5) *(e0e1_greek_rho_Lm_tr5))/((e0e1_M_L_tr5) *(((10.0))**(3.0)))))**(4.0))) ,
    e0e1_greek_sigma_tr6-(((1.0)/(1000.0)) *(((((e0e1_P_tr6) *(e0e1_greek_rho_Lm_tr6))/((e0e1_M_L_tr6) *(((10.0))**(3.0)))))**(4.0))) ,
    e0e1_greek_sigma_tr7-(((1.0)/(1000.0)) *(((((e0e1_P_tr7) *(e0e1_greek_rho_Lm_tr7))/((e0e1_M_L_tr7) *(((10.0))**(3.0)))))**(4.0))) ,
    e0e1_greek_sigma_tr8-(((1.0)/(1000.0)) *(((((e0e1_P_tr8) *(e0e1_greek_rho_Lm_tr8))/((e0e1_M_L_tr8) *(((10.0))**(3.0)))))**(4.0))) ,
    e0e1_greek_sigma_tr9-(((1.0)/(1000.0)) *(((((e0e1_P_tr9) *(e0e1_greek_rho_Lm_tr9))/((e0e1_M_L_tr9) *(((10.0))**(3.0)))))**(4.0))) ,
    e0e1_greek_sigma_tr10-(((1.0)/(1000.0)) *(((((e0e1_P_tr10) *(e0e1_greek_rho_Lm_tr10))/((e0e1_M_L_tr10) *(((10.0))**(3.0)))))**(4.0))) ,
    e0e1_greek_sigma_tr11-(((1.0)/(1000.0)) *(((((e0e1_P_tr11) *(e0e1_greek_rho_Lm_tr11))/((e0e1_M_L_tr11) *(((10.0))**(3.0)))))**(4.0))) ,
    e0e1_greek_sigma_tr12-(((1.0)/(1000.0)) *(((((e0e1_P_tr12) *(e0e1_greek_rho_Lm_tr12))/((e0e1_M_L_tr12) *(((10.0))**(3.0)))))**(4.0))) ,
    e0e1_greek_sigma_tr13-(((1.0)/(1000.0)) *(((((e0e1_P_tr13) *(e0e1_greek_rho_Lm_tr13))/((e0e1_M_L_tr13) *(((10.0))**(3.0)))))**(4.0))) ,
    e0e1_greek_sigma_tr14-(((1.0)/(1000.0)) *(((((e0e1_P_tr14) *(e0e1_greek_rho_Lm_tr14))/((e0e1_M_L_tr14) *(((10.0))**(3.0)))))**(4.0))) ,
    e0e1_greek_sigma_tr15-(((1.0)/(1000.0)) *(((((e0e1_P_tr15) *(e0e1_greek_rho_Lm_tr15))/((e0e1_M_L_tr15) *(((10.0))**(3.0)))))**(4.0))) ,
    e0e1_greek_sigma_tr16-(((1.0)/(1000.0)) *(((((e0e1_P_tr16) *(e0e1_greek_rho_Lm_tr16))/((e0e1_M_L_tr16) *(((10.0))**(3.0)))))**(4.0))) ,
    e0e1_greek_sigma_tr17-(((1.0)/(1000.0)) *(((((e0e1_P_tr17) *(e0e1_greek_rho_Lm_tr17))/((e0e1_M_L_tr17) *(((10.0))**(3.0)))))**(4.0))) ,
    e0e1_greek_sigma_tr18-(((1.0)/(1000.0)) *(((((e0e1_P_tr18) *(e0e1_greek_rho_Lm_tr18))/((e0e1_M_L_tr18) *(((10.0))**(3.0)))))**(4.0))) ,
    e0e1_greek_sigma_tr19-(((1.0)/(1000.0)) *(((((e0e1_P_tr19) *(e0e1_greek_rho_Lm_tr19))/((e0e1_M_L_tr19) *(((10.0))**(3.0)))))**(4.0))) ,
    e0e1_greek_sigma_tr20-(((1.0)/(1000.0)) *(((((e0e1_P_tr20) *(e0e1_greek_rho_Lm_tr20))/((e0e1_M_L_tr20) *(((10.0))**(3.0)))))**(4.0))) ,
    e0e1_P_tr1-((((e0e1_x_tr1_i1) *((e0e1_x_tr1_i1) *((e0e1_P_i1_tr1+e0e1_P_i1_tr1)/(2.0)))+(e0e1_x_tr1_i1) *((e0e1_x_tr1_i2) *((e0e1_P_i1_tr1+e0e1_P_i2_tr1)/(2.0))))+((e0e1_x_tr1_i2) *((e0e1_x_tr1_i1) *((e0e1_P_i2_tr1+e0e1_P_i1_tr1)/(2.0)))+(e0e1_x_tr1_i2) *((e0e1_x_tr1_i2) *((e0e1_P_i2_tr1+e0e1_P_i2_tr1)/(2.0)))))) ,
    e0e1_P_tr2-((((e0e1_x_tr2_i1) *((e0e1_x_tr2_i1) *((e0e1_P_i1_tr2+e0e1_P_i1_tr2)/(2.0)))+(e0e1_x_tr2_i1) *((e0e1_x_tr2_i2) *((e0e1_P_i1_tr2+e0e1_P_i2_tr2)/(2.0))))+((e0e1_x_tr2_i2) *((e0e1_x_tr2_i1) *((e0e1_P_i2_tr2+e0e1_P_i1_tr2)/(2.0)))+(e0e1_x_tr2_i2) *((e0e1_x_tr2_i2) *((e0e1_P_i2_tr2+e0e1_P_i2_tr2)/(2.0)))))) ,
    e0e1_P_tr3-((((e0e1_x_tr3_i1) *((e0e1_x_tr3_i1) *((e0e1_P_i1_tr3+e0e1_P_i1_tr3)/(2.0)))+(e0e1_x_tr3_i1) *((e0e1_x_tr3_i2) *((e0e1_P_i1_tr3+e0e1_P_i2_tr3)/(2.0))))+((e0e1_x_tr3_i2) *((e0e1_x_tr3_i1) *((e0e1_P_i2_tr3+e0e1_P_i1_tr3)/(2.0)))+(e0e1_x_tr3_i2) *((e0e1_x_tr3_i2) *((e0e1_P_i2_tr3+e0e1_P_i2_tr3)/(2.0)))))) ,
    e0e1_P_tr4-((((e0e1_x_tr4_i1) *((e0e1_x_tr4_i1) *((e0e1_P_i1_tr4+e0e1_P_i1_tr4)/(2.0)))+(e0e1_x_tr4_i1) *((e0e1_x_tr4_i2) *((e0e1_P_i1_tr4+e0e1_P_i2_tr4)/(2.0))))+((e0e1_x_tr4_i2) *((e0e1_x_tr4_i1) *((e0e1_P_i2_tr4+e0e1_P_i1_tr4)/(2.0)))+(e0e1_x_tr4_i2) *((e0e1_x_tr4_i2) *((e0e1_P_i2_tr4+e0e1_P_i2_tr4)/(2.0)))))) ,
    e0e1_P_tr5-((((e0e1_x_tr5_i1) *((e0e1_x_tr5_i1) *((e0e1_P_i1_tr5+e0e1_P_i1_tr5)/(2.0)))+(e0e1_x_tr5_i1) *((e0e1_x_tr5_i2) *((e0e1_P_i1_tr5+e0e1_P_i2_tr5)/(2.0))))+((e0e1_x_tr5_i2) *((e0e1_x_tr5_i1) *((e0e1_P_i2_tr5+e0e1_P_i1_tr5)/(2.0)))+(e0e1_x_tr5_i2) *((e0e1_x_tr5_i2) *((e0e1_P_i2_tr5+e0e1_P_i2_tr5)/(2.0)))))) ,
    e0e1_P_tr6-((((e0e1_x_tr6_i1) *((e0e1_x_tr6_i1) *((e0e1_P_i1_tr6+e0e1_P_i1_tr6)/(2.0)))+(e0e1_x_tr6_i1) *((e0e1_x_tr6_i2) *((e0e1_P_i1_tr6+e0e1_P_i2_tr6)/(2.0))))+((e0e1_x_tr6_i2) *((e0e1_x_tr6_i1) *((e0e1_P_i2_tr6+e0e1_P_i1_tr6)/(2.0)))+(e0e1_x_tr6_i2) *((e0e1_x_tr6_i2) *((e0e1_P_i2_tr6+e0e1_P_i2_tr6)/(2.0)))))) ,
    e0e1_P_tr7-((((e0e1_x_tr7_i1) *((e0e1_x_tr7_i1) *((e0e1_P_i1_tr7+e0e1_P_i1_tr7)/(2.0)))+(e0e1_x_tr7_i1) *((e0e1_x_tr7_i2) *((e0e1_P_i1_tr7+e0e1_P_i2_tr7)/(2.0))))+((e0e1_x_tr7_i2) *((e0e1_x_tr7_i1) *((e0e1_P_i2_tr7+e0e1_P_i1_tr7)/(2.0)))+(e0e1_x_tr7_i2) *((e0e1_x_tr7_i2) *((e0e1_P_i2_tr7+e0e1_P_i2_tr7)/(2.0)))))) ,
    e0e1_P_tr8-((((e0e1_x_tr8_i1) *((e0e1_x_tr8_i1) *((e0e1_P_i1_tr8+e0e1_P_i1_tr8)/(2.0)))+(e0e1_x_tr8_i1) *((e0e1_x_tr8_i2) *((e0e1_P_i1_tr8+e0e1_P_i2_tr8)/(2.0))))+((e0e1_x_tr8_i2) *((e0e1_x_tr8_i1) *((e0e1_P_i2_tr8+e0e1_P_i1_tr8)/(2.0)))+(e0e1_x_tr8_i2) *((e0e1_x_tr8_i2) *((e0e1_P_i2_tr8+e0e1_P_i2_tr8)/(2.0)))))) ,
    e0e1_P_tr9-((((e0e1_x_tr9_i1) *((e0e1_x_tr9_i1) *((e0e1_P_i1_tr9+e0e1_P_i1_tr9)/(2.0)))+(e0e1_x_tr9_i1) *((e0e1_x_tr9_i2) *((e0e1_P_i1_tr9+e0e1_P_i2_tr9)/(2.0))))+((e0e1_x_tr9_i2) *((e0e1_x_tr9_i1) *((e0e1_P_i2_tr9+e0e1_P_i1_tr9)/(2.0)))+(e0e1_x_tr9_i2) *((e0e1_x_tr9_i2) *((e0e1_P_i2_tr9+e0e1_P_i2_tr9)/(2.0)))))) ,
    e0e1_P_tr10-((((e0e1_x_tr10_i1) *((e0e1_x_tr10_i1) *((e0e1_P_i1_tr10+e0e1_P_i1_tr10)/(2.0)))+(e0e1_x_tr10_i1) *((e0e1_x_tr10_i2) *((e0e1_P_i1_tr10+e0e1_P_i2_tr10)/(2.0))))+((e0e1_x_tr10_i2) *((e0e1_x_tr10_i1) *((e0e1_P_i2_tr10+e0e1_P_i1_tr10)/(2.0)))+(e0e1_x_tr10_i2) *((e0e1_x_tr10_i2) *((e0e1_P_i2_tr10+e0e1_P_i2_tr10)/(2.0)))))) ,
    e0e1_P_tr11-((((e0e1_x_tr11_i1) *((e0e1_x_tr11_i1) *((e0e1_P_i1_tr11+e0e1_P_i1_tr11)/(2.0)))+(e0e1_x_tr11_i1) *((e0e1_x_tr11_i2) *((e0e1_P_i1_tr11+e0e1_P_i2_tr11)/(2.0))))+((e0e1_x_tr11_i2) *((e0e1_x_tr11_i1) *((e0e1_P_i2_tr11+e0e1_P_i1_tr11)/(2.0)))+(e0e1_x_tr11_i2) *((e0e1_x_tr11_i2) *((e0e1_P_i2_tr11+e0e1_P_i2_tr11)/(2.0)))))) ,
    e0e1_P_tr12-((((e0e1_x_tr12_i1) *((e0e1_x_tr12_i1) *((e0e1_P_i1_tr12+e0e1_P_i1_tr12)/(2.0)))+(e0e1_x_tr12_i1) *((e0e1_x_tr12_i2) *((e0e1_P_i1_tr12+e0e1_P_i2_tr12)/(2.0))))+((e0e1_x_tr12_i2) *((e0e1_x_tr12_i1) *((e0e1_P_i2_tr12+e0e1_P_i1_tr12)/(2.0)))+(e0e1_x_tr12_i2) *((e0e1_x_tr12_i2) *((e0e1_P_i2_tr12+e0e1_P_i2_tr12)/(2.0)))))) ,
    e0e1_P_tr13-((((e0e1_x_tr13_i1) *((e0e1_x_tr13_i1) *((e0e1_P_i1_tr13+e0e1_P_i1_tr13)/(2.0)))+(e0e1_x_tr13_i1) *((e0e1_x_tr13_i2) *((e0e1_P_i1_tr13+e0e1_P_i2_tr13)/(2.0))))+((e0e1_x_tr13_i2) *((e0e1_x_tr13_i1) *((e0e1_P_i2_tr13+e0e1_P_i1_tr13)/(2.0)))+(e0e1_x_tr13_i2) *((e0e1_x_tr13_i2) *((e0e1_P_i2_tr13+e0e1_P_i2_tr13)/(2.0)))))) ,
    e0e1_P_tr14-((((e0e1_x_tr14_i1) *((e0e1_x_tr14_i1) *((e0e1_P_i1_tr14+e0e1_P_i1_tr14)/(2.0)))+(e0e1_x_tr14_i1) *((e0e1_x_tr14_i2) *((e0e1_P_i1_tr14+e0e1_P_i2_tr14)/(2.0))))+((e0e1_x_tr14_i2) *((e0e1_x_tr14_i1) *((e0e1_P_i2_tr14+e0e1_P_i1_tr14)/(2.0)))+(e0e1_x_tr14_i2) *((e0e1_x_tr14_i2) *((e0e1_P_i2_tr14+e0e1_P_i2_tr14)/(2.0)))))) ,
    e0e1_P_tr15-((((e0e1_x_tr15_i1) *((e0e1_x_tr15_i1) *((e0e1_P_i1_tr15+e0e1_P_i1_tr15)/(2.0)))+(e0e1_x_tr15_i1) *((e0e1_x_tr15_i2) *((e0e1_P_i1_tr15+e0e1_P_i2_tr15)/(2.0))))+((e0e1_x_tr15_i2) *((e0e1_x_tr15_i1) *((e0e1_P_i2_tr15+e0e1_P_i1_tr15)/(2.0)))+(e0e1_x_tr15_i2) *((e0e1_x_tr15_i2) *((e0e1_P_i2_tr15+e0e1_P_i2_tr15)/(2.0)))))) ,
    e0e1_P_tr16-((((e0e1_x_tr16_i1) *((e0e1_x_tr16_i1) *((e0e1_P_i1_tr16+e0e1_P_i1_tr16)/(2.0)))+(e0e1_x_tr16_i1) *((e0e1_x_tr16_i2) *((e0e1_P_i1_tr16+e0e1_P_i2_tr16)/(2.0))))+((e0e1_x_tr16_i2) *((e0e1_x_tr16_i1) *((e0e1_P_i2_tr16+e0e1_P_i1_tr16)/(2.0)))+(e0e1_x_tr16_i2) *((e0e1_x_tr16_i2) *((e0e1_P_i2_tr16+e0e1_P_i2_tr16)/(2.0)))))) ,
    e0e1_P_tr17-((((e0e1_x_tr17_i1) *((e0e1_x_tr17_i1) *((e0e1_P_i1_tr17+e0e1_P_i1_tr17)/(2.0)))+(e0e1_x_tr17_i1) *((e0e1_x_tr17_i2) *((e0e1_P_i1_tr17+e0e1_P_i2_tr17)/(2.0))))+((e0e1_x_tr17_i2) *((e0e1_x_tr17_i1) *((e0e1_P_i2_tr17+e0e1_P_i1_tr17)/(2.0)))+(e0e1_x_tr17_i2) *((e0e1_x_tr17_i2) *((e0e1_P_i2_tr17+e0e1_P_i2_tr17)/(2.0)))))) ,
    e0e1_P_tr18-((((e0e1_x_tr18_i1) *((e0e1_x_tr18_i1) *((e0e1_P_i1_tr18+e0e1_P_i1_tr18)/(2.0)))+(e0e1_x_tr18_i1) *((e0e1_x_tr18_i2) *((e0e1_P_i1_tr18+e0e1_P_i2_tr18)/(2.0))))+((e0e1_x_tr18_i2) *((e0e1_x_tr18_i1) *((e0e1_P_i2_tr18+e0e1_P_i1_tr18)/(2.0)))+(e0e1_x_tr18_i2) *((e0e1_x_tr18_i2) *((e0e1_P_i2_tr18+e0e1_P_i2_tr18)/(2.0)))))) ,
    e0e1_P_tr19-((((e0e1_x_tr19_i1) *((e0e1_x_tr19_i1) *((e0e1_P_i1_tr19+e0e1_P_i1_tr19)/(2.0)))+(e0e1_x_tr19_i1) *((e0e1_x_tr19_i2) *((e0e1_P_i1_tr19+e0e1_P_i2_tr19)/(2.0))))+((e0e1_x_tr19_i2) *((e0e1_x_tr19_i1) *((e0e1_P_i2_tr19+e0e1_P_i1_tr19)/(2.0)))+(e0e1_x_tr19_i2) *((e0e1_x_tr19_i2) *((e0e1_P_i2_tr19+e0e1_P_i2_tr19)/(2.0)))))) ,
    e0e1_P_tr20-((((e0_x_Reboilerin_i1) *((e0_x_Reboilerin_i1) *((e0e1_P_i1_tr20+e0e1_P_i1_tr20)/(2.0)))+(e0_x_Reboilerin_i1) *((e0_x_Reboilerin_i2) *((e0e1_P_i1_tr20+e0e1_P_i2_tr20)/(2.0))))+((e0_x_Reboilerin_i2) *((e0_x_Reboilerin_i1) *((e0e1_P_i2_tr20+e0e1_P_i1_tr20)/(2.0)))+(e0_x_Reboilerin_i2) *((e0_x_Reboilerin_i2) *((e0e1_P_i2_tr20+e0e1_P_i2_tr20)/(2.0)))))) ,
    (((((e0e1_greek_rho_Ln_tr1_i1) *(e0e1_P_i1_tr1))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i1_tr1) ,
    (((((e0e1_greek_rho_Ln_tr2_i1) *(e0e1_P_i1_tr2))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i1_tr2) ,
    (((((e0e1_greek_rho_Ln_tr3_i1) *(e0e1_P_i1_tr3))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i1_tr3) ,
    (((((e0e1_greek_rho_Ln_tr4_i1) *(e0e1_P_i1_tr4))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i1_tr4) ,
    (((((e0e1_greek_rho_Ln_tr5_i1) *(e0e1_P_i1_tr5))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i1_tr5) ,
    (((((e0e1_greek_rho_Ln_tr6_i1) *(e0e1_P_i1_tr6))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i1_tr6) ,
    (((((e0e1_greek_rho_Ln_tr7_i1) *(e0e1_P_i1_tr7))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i1_tr7) ,
    (((((e0e1_greek_rho_Ln_tr8_i1) *(e0e1_P_i1_tr8))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i1_tr8) ,
    (((((e0e1_greek_rho_Ln_tr9_i1) *(e0e1_P_i1_tr9))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i1_tr9) ,
    (((((e0e1_greek_rho_Ln_tr10_i1) *(e0e1_P_i1_tr10))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i1_tr10) ,
    (((((e0e1_greek_rho_Ln_tr11_i1) *(e0e1_P_i1_tr11))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i1_tr11) ,
    (((((e0e1_greek_rho_Ln_tr12_i1) *(e0e1_P_i1_tr12))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i1_tr12) ,
    (((((e0e1_greek_rho_Ln_tr13_i1) *(e0e1_P_i1_tr13))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i1_tr13) ,
    (((((e0e1_greek_rho_Ln_tr14_i1) *(e0e1_P_i1_tr14))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i1_tr14) ,
    (((((e0e1_greek_rho_Ln_tr15_i1) *(e0e1_P_i1_tr15))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i1_tr15) ,
    (((((e0e1_greek_rho_Ln_tr16_i1) *(e0e1_P_i1_tr16))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i1_tr16) ,
    (((((e0e1_greek_rho_Ln_tr17_i1) *(e0e1_P_i1_tr17))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i1_tr17) ,
    (((((e0e1_greek_rho_Ln_tr18_i1) *(e0e1_P_i1_tr18))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i1_tr18) ,
    (((((e0e1_greek_rho_Ln_tr19_i1) *(e0e1_P_i1_tr19))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i1_tr19) ,
    (((((e0e1_greek_rho_Ln_tr20_i1) *(e0e1_P_i1_tr20))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i1_tr20) ,
    (((((e0e1_greek_rho_Ln_tr1_i2) *(e0e1_P_i2_tr1))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i2_tr1) ,
    (((((e0e1_greek_rho_Ln_tr2_i2) *(e0e1_P_i2_tr2))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i2_tr2) ,
    (((((e0e1_greek_rho_Ln_tr3_i2) *(e0e1_P_i2_tr3))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i2_tr3) ,
    (((((e0e1_greek_rho_Ln_tr4_i2) *(e0e1_P_i2_tr4))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i2_tr4) ,
    (((((e0e1_greek_rho_Ln_tr5_i2) *(e0e1_P_i2_tr5))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i2_tr5) ,
    (((((e0e1_greek_rho_Ln_tr6_i2) *(e0e1_P_i2_tr6))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i2_tr6) ,
    (((((e0e1_greek_rho_Ln_tr7_i2) *(e0e1_P_i2_tr7))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i2_tr7) ,
    (((((e0e1_greek_rho_Ln_tr8_i2) *(e0e1_P_i2_tr8))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i2_tr8) ,
    (((((e0e1_greek_rho_Ln_tr9_i2) *(e0e1_P_i2_tr9))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i2_tr9) ,
    (((((e0e1_greek_rho_Ln_tr10_i2) *(e0e1_P_i2_tr10))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i2_tr10) ,
    (((((e0e1_greek_rho_Ln_tr11_i2) *(e0e1_P_i2_tr11))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i2_tr11) ,
    (((((e0e1_greek_rho_Ln_tr12_i2) *(e0e1_P_i2_tr12))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i2_tr12) ,
    (((((e0e1_greek_rho_Ln_tr13_i2) *(e0e1_P_i2_tr13))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i2_tr13) ,
    (((((e0e1_greek_rho_Ln_tr14_i2) *(e0e1_P_i2_tr14))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i2_tr14) ,
    (((((e0e1_greek_rho_Ln_tr15_i2) *(e0e1_P_i2_tr15))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i2_tr15) ,
    (((((e0e1_greek_rho_Ln_tr16_i2) *(e0e1_P_i2_tr16))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i2_tr16) ,
    (((((e0e1_greek_rho_Ln_tr17_i2) *(e0e1_P_i2_tr17))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i2_tr17) ,
    (((((e0e1_greek_rho_Ln_tr18_i2) *(e0e1_P_i2_tr18))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i2_tr18) ,
    (((((e0e1_greek_rho_Ln_tr19_i2) *(e0e1_P_i2_tr19))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i2_tr19) ,
    (((((e0e1_greek_rho_Ln_tr20_i2) *(e0e1_P_i2_tr20))/(((10.0))**(3.0))))**(4.0))/(1000.0)-(e0e1_greek_sigma_i2_tr20) ,
    e0_y_D_i1-((e0e1_K_tr1_i1) *(e0e1_x_tr1_i1)) ,
    e0e1_y_tr2_i1-((e0e1_K_tr2_i1) *(e0e1_x_tr2_i1)) ,
    e0e1_y_tr3_i1-((e0e1_K_tr3_i1) *(e0e1_x_tr3_i1)) ,
    e0e1_y_tr4_i1-((e0e1_K_tr4_i1) *(e0e1_x_tr4_i1)) ,
    e0e1_y_tr5_i1-((e0e1_K_tr5_i1) *(e0e1_x_tr5_i1)) ,
    e0e1_y_tr6_i1-((e0e1_K_tr6_i1) *(e0e1_x_tr6_i1)) ,
    e0e1_y_tr7_i1-((e0e1_K_tr7_i1) *(e0e1_x_tr7_i1)) ,
    e0e1_y_tr8_i1-((e0e1_K_tr8_i1) *(e0e1_x_tr8_i1)) ,
    e0e1_y_tr9_i1-((e0e1_K_tr9_i1) *(e0e1_x_tr9_i1)) ,
    e0e1_y_tr10_i1-((e0e1_K_tr10_i1) *(e0e1_x_tr10_i1)) ,
    e0e1_y_tr11_i1-((e0e1_K_tr11_i1) *(e0e1_x_tr11_i1)) ,
    e0e1_y_tr12_i1-((e0e1_K_tr12_i1) *(e0e1_x_tr12_i1)) ,
    e0e1_y_tr13_i1-((e0e1_K_tr13_i1) *(e0e1_x_tr13_i1)) ,
    e0e1_y_tr14_i1-((e0e1_K_tr14_i1) *(e0e1_x_tr14_i1)) ,
    e0e1_y_tr15_i1-((e0e1_K_tr15_i1) *(e0e1_x_tr15_i1)) ,
    e0e1_y_tr16_i1-((e0e1_K_tr16_i1) *(e0e1_x_tr16_i1)) ,
    e0e1_y_tr17_i1-((e0e1_K_tr17_i1) *(e0e1_x_tr17_i1)) ,
    e0e1_y_tr18_i1-((e0e1_K_tr18_i1) *(e0e1_x_tr18_i1)) ,
    e0e1_y_tr19_i1-((e0e1_K_tr19_i1) *(e0e1_x_tr19_i1)) ,
    e0e1_y_tr20_i1-((e0e1_K_tr20_i1) *(e0_x_Reboilerin_i1)) ,
    e0_y_D_i2-((e0e1_K_tr1_i2) *(e0e1_x_tr1_i2)) ,
    e0e1_y_tr2_i2-((e0e1_K_tr2_i2) *(e0e1_x_tr2_i2)) ,
    e0e1_y_tr3_i2-((e0e1_K_tr3_i2) *(e0e1_x_tr3_i2)) ,
    e0e1_y_tr4_i2-((e0e1_K_tr4_i2) *(e0e1_x_tr4_i2)) ,
    e0e1_y_tr5_i2-((e0e1_K_tr5_i2) *(e0e1_x_tr5_i2)) ,
    e0e1_y_tr6_i2-((e0e1_K_tr6_i2) *(e0e1_x_tr6_i2)) ,
    e0e1_y_tr7_i2-((e0e1_K_tr7_i2) *(e0e1_x_tr7_i2)) ,
    e0e1_y_tr8_i2-((e0e1_K_tr8_i2) *(e0e1_x_tr8_i2)) ,
    e0e1_y_tr9_i2-((e0e1_K_tr9_i2) *(e0e1_x_tr9_i2)) ,
    e0e1_y_tr10_i2-((e0e1_K_tr10_i2) *(e0e1_x_tr10_i2)) ,
    e0e1_y_tr11_i2-((e0e1_K_tr11_i2) *(e0e1_x_tr11_i2)) ,
    e0e1_y_tr12_i2-((e0e1_K_tr12_i2) *(e0e1_x_tr12_i2)) ,
    e0e1_y_tr13_i2-((e0e1_K_tr13_i2) *(e0e1_x_tr13_i2)) ,
    e0e1_y_tr14_i2-((e0e1_K_tr14_i2) *(e0e1_x_tr14_i2)) ,
    e0e1_y_tr15_i2-((e0e1_K_tr15_i2) *(e0e1_x_tr15_i2)) ,
    e0e1_y_tr16_i2-((e0e1_K_tr16_i2) *(e0e1_x_tr16_i2)) ,
    e0e1_y_tr17_i2-((e0e1_K_tr17_i2) *(e0e1_x_tr17_i2)) ,
    e0e1_y_tr18_i2-((e0e1_K_tr18_i2) *(e0e1_x_tr18_i2)) ,
    e0e1_y_tr19_i2-((e0e1_K_tr19_i2) *(e0e1_x_tr19_i2)) ,
    e0e1_y_tr20_i2-((e0e1_K_tr20_i2) *(e0_x_Reboilerin_i2)) ,
    e0e1_greek_Deltap_tr1-(e0e1_p_tr2-e0e1_p_tr1) ,
    e0e1_greek_Deltap_tr2-(e0e1_p_tr3-e0e1_p_tr2) ,
    e0e1_greek_Deltap_tr3-(e0e1_p_tr4-e0e1_p_tr3) ,
    e0e1_greek_Deltap_tr4-(e0e1_p_tr5-e0e1_p_tr4) ,
    e0e1_greek_Deltap_tr5-(e0e1_p_tr6-e0e1_p_tr5) ,
    e0e1_greek_Deltap_tr6-(e0e1_p_tr7-e0e1_p_tr6) ,
    e0e1_greek_Deltap_tr7-(e0e1_p_tr8-e0e1_p_tr7) ,
    e0e1_greek_Deltap_tr8-(e0e1_p_tr9-e0e1_p_tr8) ,
    e0e1_greek_Deltap_tr9-(e0e1_p_tr10-e0e1_p_tr9) ,
    e0e1_greek_Deltap_tr10-(e0e1_p_tr11-e0e1_p_tr10) ,
    e0e1_greek_Deltap_tr11-(e0e1_p_tr12-e0e1_p_tr11) ,
    e0e1_greek_Deltap_tr12-(e0e1_p_tr13-e0e1_p_tr12) ,
    e0e1_greek_Deltap_tr13-(e0e1_p_tr14-e0e1_p_tr13) ,
    e0e1_greek_Deltap_tr14-(e0e1_p_tr15-e0e1_p_tr14) ,
    e0e1_greek_Deltap_tr15-(e0e1_p_tr16-e0e1_p_tr15) ,
    e0e1_greek_Deltap_tr16-(e0e1_p_tr17-e0e1_p_tr16) ,
    e0e1_greek_Deltap_tr17-(e0e1_p_tr18-e0e1_p_tr17) ,
    e0e1_greek_Deltap_tr18-(e0e1_p_tr19-e0e1_p_tr18) ,
    e0e1_greek_Deltap_tr19-(e0_p_LReboiler-e0e1_p_tr19) ,
    e0e1_greek_Deltap_tr20-(e0e1_p_tr21-e0_p_LReboiler) ,
    e0e1_greek_Deltap_tr0-(e0e1_p_tr1-e0_p_DV) ,
    1.0-((e0e1_x_tr1_i1+e0e1_x_tr1_i2)) ,
    1.0-((e0e1_x_tr2_i1+e0e1_x_tr2_i2)) ,
    1.0-((e0e1_x_tr3_i1+e0e1_x_tr3_i2)) ,
    1.0-((e0e1_x_tr4_i1+e0e1_x_tr4_i2)) ,
    1.0-((e0e1_x_tr5_i1+e0e1_x_tr5_i2)) ,
    1.0-((e0e1_x_tr6_i1+e0e1_x_tr6_i2)) ,
    1.0-((e0e1_x_tr7_i1+e0e1_x_tr7_i2)) ,
    1.0-((e0e1_x_tr8_i1+e0e1_x_tr8_i2)) ,
    1.0-((e0e1_x_tr9_i1+e0e1_x_tr9_i2)) ,
    1.0-((e0e1_x_tr10_i1+e0e1_x_tr10_i2)) ,
    1.0-((e0e1_x_tr11_i1+e0e1_x_tr11_i2)) ,
    1.0-((e0e1_x_tr12_i1+e0e1_x_tr12_i2)) ,
    1.0-((e0e1_x_tr13_i1+e0e1_x_tr13_i2)) ,
    1.0-((e0e1_x_tr14_i1+e0e1_x_tr14_i2)) ,
    1.0-((e0e1_x_tr15_i1+e0e1_x_tr15_i2)) ,
    1.0-((e0e1_x_tr16_i1+e0e1_x_tr16_i2)) ,
    1.0-((e0e1_x_tr17_i1+e0e1_x_tr17_i2)) ,
    1.0-((e0e1_x_tr18_i1+e0e1_x_tr18_i2)) ,
    1.0-((e0e1_x_tr19_i1+e0e1_x_tr19_i2)) ,
    1.0-((e0_x_Reboilerin_i1+e0_x_Reboilerin_i2)) ,
    1.0-((e0_y_D_i1+e0_y_D_i2)) ,
    1.0-((e0e1_y_tr2_i1+e0e1_y_tr2_i2)) ,
    1.0-((e0e1_y_tr3_i1+e0e1_y_tr3_i2)) ,
    1.0-((e0e1_y_tr4_i1+e0e1_y_tr4_i2)) ,
    1.0-((e0e1_y_tr5_i1+e0e1_y_tr5_i2)) ,
    1.0-((e0e1_y_tr6_i1+e0e1_y_tr6_i2)) ,
    1.0-((e0e1_y_tr7_i1+e0e1_y_tr7_i2)) ,
    1.0-((e0e1_y_tr8_i1+e0e1_y_tr8_i2)) ,
    1.0-((e0e1_y_tr9_i1+e0e1_y_tr9_i2)) ,
    1.0-((e0e1_y_tr10_i1+e0e1_y_tr10_i2)) ,
    1.0-((e0e1_y_tr11_i1+e0e1_y_tr11_i2)) ,
    1.0-((e0e1_y_tr12_i1+e0e1_y_tr12_i2)) ,
    1.0-((e0e1_y_tr13_i1+e0e1_y_tr13_i2)) ,
    1.0-((e0e1_y_tr14_i1+e0e1_y_tr14_i2)) ,
    1.0-((e0e1_y_tr15_i1+e0e1_y_tr15_i2)) ,
    1.0-((e0e1_y_tr16_i1+e0e1_y_tr16_i2)) ,
    1.0-((e0e1_y_tr17_i1+e0e1_y_tr17_i2)) ,
    1.0-((e0e1_y_tr18_i1+e0e1_y_tr18_i2)) ,
    1.0-((e0e1_y_tr19_i1+e0e1_y_tr19_i2)) ,
    1.0-((e0e1_y_tr20_i1+e0e1_y_tr20_i2)) ,
    0.0-(e0e1_F_Ln_tr20-e0e1_F_Ln_tr21-e0_F_LReboilerinn) ,
    0.0-((e0e1_F_Fn_tr1) *(e0_h_Fn)+(e0_F_DLRefluxn) *(e0_h_DLn)+(e0e1_F_Vn_tr2) *(e0e1_h_Vn_tr2)-(e0e1_F_Ln_tr1) *(e0e1_h_Ln_tr1)-(e0_F_DVn) *(e0_h_DVn)+e0e1_Q_Loss_tr1) ,
    0.0-((e0e1_F_Fn_tr2) *(e0_h_Fn)+(e0e1_F_Ln_tr1) *(e0e1_h_Ln_tr1)+(e0e1_F_Vn_tr3) *(e0e1_h_Vn_tr3)-(e0e1_F_Ln_tr2) *(e0e1_h_Ln_tr2)-(e0e1_F_Vn_tr2) *(e0e1_h_Vn_tr2)+e0e1_Q_Loss_tr2) ,
    0.0-((e0e1_F_Fn_tr3) *(e0_h_Fn)+(e0e1_F_Ln_tr2) *(e0e1_h_Ln_tr2)+(e0e1_F_Vn_tr4) *(e0e1_h_Vn_tr4)-(e0e1_F_Ln_tr3) *(e0e1_h_Ln_tr3)-(e0e1_F_Vn_tr3) *(e0e1_h_Vn_tr3)+e0e1_Q_Loss_tr3) ,
    0.0-((e0e1_F_Fn_tr4) *(e0_h_Fn)+(e0e1_F_Ln_tr3) *(e0e1_h_Ln_tr3)+(e0e1_F_Vn_tr5) *(e0e1_h_Vn_tr5)-(e0e1_F_Ln_tr4) *(e0e1_h_Ln_tr4)-(e0e1_F_Vn_tr4) *(e0e1_h_Vn_tr4)+e0e1_Q_Loss_tr4) ,
    0.0-((e0e1_F_Fn_tr5) *(e0_h_Fn)+(e0e1_F_Ln_tr4) *(e0e1_h_Ln_tr4)+(e0e1_F_Vn_tr6) *(e0e1_h_Vn_tr6)-(e0e1_F_Ln_tr5) *(e0e1_h_Ln_tr5)-(e0e1_F_Vn_tr5) *(e0e1_h_Vn_tr5)+e0e1_Q_Loss_tr5) ,
    0.0-((e0e1_F_Fn_tr6) *(e0_h_Fn)+(e0e1_F_Ln_tr5) *(e0e1_h_Ln_tr5)+(e0e1_F_Vn_tr7) *(e0e1_h_Vn_tr7)-(e0e1_F_Ln_tr6) *(e0e1_h_Ln_tr6)-(e0e1_F_Vn_tr6) *(e0e1_h_Vn_tr6)+e0e1_Q_Loss_tr6) ,
    0.0-((e0e1_F_Fn_tr7) *(e0_h_Fn)+(e0e1_F_Ln_tr6) *(e0e1_h_Ln_tr6)+(e0e1_F_Vn_tr8) *(e0e1_h_Vn_tr8)-(e0e1_F_Ln_tr7) *(e0e1_h_Ln_tr7)-(e0e1_F_Vn_tr7) *(e0e1_h_Vn_tr7)+e0e1_Q_Loss_tr7) ,
    0.0-((e0e1_F_Fn_tr8) *(e0_h_Fn)+(e0e1_F_Ln_tr7) *(e0e1_h_Ln_tr7)+(e0e1_F_Vn_tr9) *(e0e1_h_Vn_tr9)-(e0e1_F_Ln_tr8) *(e0e1_h_Ln_tr8)-(e0e1_F_Vn_tr8) *(e0e1_h_Vn_tr8)+e0e1_Q_Loss_tr8) ,
    0.0-((e0e1_F_Fn_tr9) *(e0_h_Fn)+(e0e1_F_Ln_tr8) *(e0e1_h_Ln_tr8)+(e0e1_F_Vn_tr10) *(e0e1_h_Vn_tr10)-(e0e1_F_Ln_tr9) *(e0e1_h_Ln_tr9)-(e0e1_F_Vn_tr9) *(e0e1_h_Vn_tr9)+e0e1_Q_Loss_tr9) ,
    0.0-((e0e1_F_Fn_tr10) *(e0_h_Fn)+(e0e1_F_Ln_tr9) *(e0e1_h_Ln_tr9)+(e0e1_F_Vn_tr11) *(e0e1_h_Vn_tr11)-(e0e1_F_Ln_tr10) *(e0e1_h_Ln_tr10)-(e0e1_F_Vn_tr10) *(e0e1_h_Vn_tr10)+e0e1_Q_Loss_tr10) ,
    0.0-((e0e1_F_Fn_tr11) *(e0_h_Fn)+(e0e1_F_Ln_tr10) *(e0e1_h_Ln_tr10)+(e0e1_F_Vn_tr12) *(e0e1_h_Vn_tr12)-(e0e1_F_Ln_tr11) *(e0e1_h_Ln_tr11)-(e0e1_F_Vn_tr11) *(e0e1_h_Vn_tr11)+e0e1_Q_Loss_tr11) ,
    0.0-((e0e1_F_Fn_tr12) *(e0_h_Fn)+(e0e1_F_Ln_tr11) *(e0e1_h_Ln_tr11)+(e0e1_F_Vn_tr13) *(e0e1_h_Vn_tr13)-(e0e1_F_Ln_tr12) *(e0e1_h_Ln_tr12)-(e0e1_F_Vn_tr12) *(e0e1_h_Vn_tr12)+e0e1_Q_Loss_tr12) ,
    0.0-((e0e1_F_Fn_tr13) *(e0_h_Fn)+(e0e1_F_Ln_tr12) *(e0e1_h_Ln_tr12)+(e0e1_F_Vn_tr14) *(e0e1_h_Vn_tr14)-(e0e1_F_Ln_tr13) *(e0e1_h_Ln_tr13)-(e0e1_F_Vn_tr13) *(e0e1_h_Vn_tr13)+e0e1_Q_Loss_tr13) ,
    0.0-((e0e1_F_Fn_tr14) *(e0_h_Fn)+(e0e1_F_Ln_tr13) *(e0e1_h_Ln_tr13)+(e0e1_F_Vn_tr15) *(e0e1_h_Vn_tr15)-(e0e1_F_Ln_tr14) *(e0e1_h_Ln_tr14)-(e0e1_F_Vn_tr14) *(e0e1_h_Vn_tr14)+e0e1_Q_Loss_tr14) ,
    0.0-((e0e1_F_Fn_tr15) *(e0_h_Fn)+(e0e1_F_Ln_tr14) *(e0e1_h_Ln_tr14)+(e0e1_F_Vn_tr16) *(e0e1_h_Vn_tr16)-(e0e1_F_Ln_tr15) *(e0e1_h_Ln_tr15)-(e0e1_F_Vn_tr15) *(e0e1_h_Vn_tr15)+e0e1_Q_Loss_tr15) ,
    0.0-((e0e1_F_Fn_tr16) *(e0_h_Fn)+(e0e1_F_Ln_tr15) *(e0e1_h_Ln_tr15)+(e0e1_F_Vn_tr17) *(e0e1_h_Vn_tr17)-(e0e1_F_Ln_tr16) *(e0e1_h_Ln_tr16)-(e0e1_F_Vn_tr16) *(e0e1_h_Vn_tr16)+e0e1_Q_Loss_tr16) ,
    0.0-((e0e1_F_Fn_tr17) *(e0_h_Fn)+(e0e1_F_Ln_tr16) *(e0e1_h_Ln_tr16)+(e0e1_F_Vn_tr18) *(e0e1_h_Vn_tr18)-(e0e1_F_Ln_tr17) *(e0e1_h_Ln_tr17)-(e0e1_F_Vn_tr17) *(e0e1_h_Vn_tr17)+e0e1_Q_Loss_tr17) ,
    0.0-((e0e1_F_Fn_tr18) *(e0_h_Fn)+(e0e1_F_Ln_tr17) *(e0e1_h_Ln_tr17)+(e0e1_F_Vn_tr19) *(e0e1_h_Vn_tr19)-(e0e1_F_Ln_tr18) *(e0e1_h_Ln_tr18)-(e0e1_F_Vn_tr18) *(e0e1_h_Vn_tr18)+e0e1_Q_Loss_tr18) ,
    0.0-((e0e1_F_Fn_tr19) *(e0_h_Fn)+(e0e1_F_Ln_tr18) *(e0e1_h_Ln_tr18)+(e0e1_F_Vn_tr20) *(e0e1_h_Vn_tr20)-(e0e1_F_Ln_tr19) *(e0e1_h_Ln_tr19)-(e0e1_F_Vn_tr19) *(e0e1_h_Vn_tr19)+e0e1_Q_Loss_tr19) ,
    0.0-((e0e1_F_Fn_tr20) *(e0_h_Fn)+(e0e1_F_Ln_tr19) *(e0e1_h_Ln_tr19)+(e0_F_ReboilerVn) *(e0_h_ReboilerVn)-(e0e1_F_Ln_tr20) *(e0_h_LReboilerinn)-(e0e1_F_Vn_tr20) *(e0e1_h_Vn_tr20)+e0e1_Q_Loss_tr20) ,
    0.0-((e0e1_F_Fn_tr1) *(e0_x_F_i1)+(e0_F_DLRefluxn) *(e0_x_D_i1)+(e0e1_F_Vn_tr2) *(e0e1_y_tr2_i1)-(e0e1_F_Ln_tr1) *(e0e1_x_tr1_i1)-(e0_F_DVn) *(e0_y_D_i1)+e0e1_r_n_tr1_i1) ,
    0.0-((e0e1_F_Fn_tr2) *(e0_x_F_i1)+(e0e1_F_Ln_tr1) *(e0e1_x_tr1_i1)+(e0e1_F_Vn_tr3) *(e0e1_y_tr3_i1)-(e0e1_F_Ln_tr2) *(e0e1_x_tr2_i1)-(e0e1_F_Vn_tr2) *(e0e1_y_tr2_i1)+e0e1_r_n_tr2_i1) ,
    0.0-((e0e1_F_Fn_tr3) *(e0_x_F_i1)+(e0e1_F_Ln_tr2) *(e0e1_x_tr2_i1)+(e0e1_F_Vn_tr4) *(e0e1_y_tr4_i1)-(e0e1_F_Ln_tr3) *(e0e1_x_tr3_i1)-(e0e1_F_Vn_tr3) *(e0e1_y_tr3_i1)+e0e1_r_n_tr3_i1) ,
    0.0-((e0e1_F_Fn_tr4) *(e0_x_F_i1)+(e0e1_F_Ln_tr3) *(e0e1_x_tr3_i1)+(e0e1_F_Vn_tr5) *(e0e1_y_tr5_i1)-(e0e1_F_Ln_tr4) *(e0e1_x_tr4_i1)-(e0e1_F_Vn_tr4) *(e0e1_y_tr4_i1)+e0e1_r_n_tr4_i1) ,
    0.0-((e0e1_F_Fn_tr5) *(e0_x_F_i1)+(e0e1_F_Ln_tr4) *(e0e1_x_tr4_i1)+(e0e1_F_Vn_tr6) *(e0e1_y_tr6_i1)-(e0e1_F_Ln_tr5) *(e0e1_x_tr5_i1)-(e0e1_F_Vn_tr5) *(e0e1_y_tr5_i1)+e0e1_r_n_tr5_i1) ,
    0.0-((e0e1_F_Fn_tr6) *(e0_x_F_i1)+(e0e1_F_Ln_tr5) *(e0e1_x_tr5_i1)+(e0e1_F_Vn_tr7) *(e0e1_y_tr7_i1)-(e0e1_F_Ln_tr6) *(e0e1_x_tr6_i1)-(e0e1_F_Vn_tr6) *(e0e1_y_tr6_i1)+e0e1_r_n_tr6_i1) ,
    0.0-((e0e1_F_Fn_tr7) *(e0_x_F_i1)+(e0e1_F_Ln_tr6) *(e0e1_x_tr6_i1)+(e0e1_F_Vn_tr8) *(e0e1_y_tr8_i1)-(e0e1_F_Ln_tr7) *(e0e1_x_tr7_i1)-(e0e1_F_Vn_tr7) *(e0e1_y_tr7_i1)+e0e1_r_n_tr7_i1) ,
    0.0-((e0e1_F_Fn_tr8) *(e0_x_F_i1)+(e0e1_F_Ln_tr7) *(e0e1_x_tr7_i1)+(e0e1_F_Vn_tr9) *(e0e1_y_tr9_i1)-(e0e1_F_Ln_tr8) *(e0e1_x_tr8_i1)-(e0e1_F_Vn_tr8) *(e0e1_y_tr8_i1)+e0e1_r_n_tr8_i1) ,
    0.0-((e0e1_F_Fn_tr9) *(e0_x_F_i1)+(e0e1_F_Ln_tr8) *(e0e1_x_tr8_i1)+(e0e1_F_Vn_tr10) *(e0e1_y_tr10_i1)-(e0e1_F_Ln_tr9) *(e0e1_x_tr9_i1)-(e0e1_F_Vn_tr9) *(e0e1_y_tr9_i1)+e0e1_r_n_tr9_i1) ,
    0.0-((e0e1_F_Fn_tr10) *(e0_x_F_i1)+(e0e1_F_Ln_tr9) *(e0e1_x_tr9_i1)+(e0e1_F_Vn_tr11) *(e0e1_y_tr11_i1)-(e0e1_F_Ln_tr10) *(e0e1_x_tr10_i1)-(e0e1_F_Vn_tr10) *(e0e1_y_tr10_i1)+e0e1_r_n_tr10_i1) ,
    0.0-((e0e1_F_Fn_tr11) *(e0_x_F_i1)+(e0e1_F_Ln_tr10) *(e0e1_x_tr10_i1)+(e0e1_F_Vn_tr12) *(e0e1_y_tr12_i1)-(e0e1_F_Ln_tr11) *(e0e1_x_tr11_i1)-(e0e1_F_Vn_tr11) *(e0e1_y_tr11_i1)+e0e1_r_n_tr11_i1) ,
    0.0-((e0e1_F_Fn_tr12) *(e0_x_F_i1)+(e0e1_F_Ln_tr11) *(e0e1_x_tr11_i1)+(e0e1_F_Vn_tr13) *(e0e1_y_tr13_i1)-(e0e1_F_Ln_tr12) *(e0e1_x_tr12_i1)-(e0e1_F_Vn_tr12) *(e0e1_y_tr12_i1)+e0e1_r_n_tr12_i1) ,
    0.0-((e0e1_F_Fn_tr13) *(e0_x_F_i1)+(e0e1_F_Ln_tr12) *(e0e1_x_tr12_i1)+(e0e1_F_Vn_tr14) *(e0e1_y_tr14_i1)-(e0e1_F_Ln_tr13) *(e0e1_x_tr13_i1)-(e0e1_F_Vn_tr13) *(e0e1_y_tr13_i1)+e0e1_r_n_tr13_i1) ,
    0.0-((e0e1_F_Fn_tr14) *(e0_x_F_i1)+(e0e1_F_Ln_tr13) *(e0e1_x_tr13_i1)+(e0e1_F_Vn_tr15) *(e0e1_y_tr15_i1)-(e0e1_F_Ln_tr14) *(e0e1_x_tr14_i1)-(e0e1_F_Vn_tr14) *(e0e1_y_tr14_i1)+e0e1_r_n_tr14_i1) ,
    0.0-((e0e1_F_Fn_tr15) *(e0_x_F_i1)+(e0e1_F_Ln_tr14) *(e0e1_x_tr14_i1)+(e0e1_F_Vn_tr16) *(e0e1_y_tr16_i1)-(e0e1_F_Ln_tr15) *(e0e1_x_tr15_i1)-(e0e1_F_Vn_tr15) *(e0e1_y_tr15_i1)+e0e1_r_n_tr15_i1) ,
    0.0-((e0e1_F_Fn_tr16) *(e0_x_F_i1)+(e0e1_F_Ln_tr15) *(e0e1_x_tr15_i1)+(e0e1_F_Vn_tr17) *(e0e1_y_tr17_i1)-(e0e1_F_Ln_tr16) *(e0e1_x_tr16_i1)-(e0e1_F_Vn_tr16) *(e0e1_y_tr16_i1)+e0e1_r_n_tr16_i1) ,
    0.0-((e0e1_F_Fn_tr17) *(e0_x_F_i1)+(e0e1_F_Ln_tr16) *(e0e1_x_tr16_i1)+(e0e1_F_Vn_tr18) *(e0e1_y_tr18_i1)-(e0e1_F_Ln_tr17) *(e0e1_x_tr17_i1)-(e0e1_F_Vn_tr17) *(e0e1_y_tr17_i1)+e0e1_r_n_tr17_i1) ,
    0.0-((e0e1_F_Fn_tr18) *(e0_x_F_i1)+(e0e1_F_Ln_tr17) *(e0e1_x_tr17_i1)+(e0e1_F_Vn_tr19) *(e0e1_y_tr19_i1)-(e0e1_F_Ln_tr18) *(e0e1_x_tr18_i1)-(e0e1_F_Vn_tr18) *(e0e1_y_tr18_i1)+e0e1_r_n_tr18_i1) ,
    0.0-((e0e1_F_Fn_tr19) *(e0_x_F_i1)+(e0e1_F_Ln_tr18) *(e0e1_x_tr18_i1)+(e0e1_F_Vn_tr20) *(e0e1_y_tr20_i1)-(e0e1_F_Ln_tr19) *(e0e1_x_tr19_i1)-(e0e1_F_Vn_tr19) *(e0e1_y_tr19_i1)+e0e1_r_n_tr19_i1) ,
    0.0-((e0e1_F_Fn_tr20) *(e0_x_F_i1)+(e0e1_F_Ln_tr19) *(e0e1_x_tr19_i1)+(e0_F_ReboilerVn) *(e0_y_Reboiler_i1)-(e0e1_F_Ln_tr20) *(e0_x_Reboilerin_i1)-(e0e1_F_Vn_tr20) *(e0e1_y_tr20_i1)+e0e1_r_n_tr20_i1) ,
    0.0-((e0e1_F_Fn_tr1) *(e0_x_F_i2)+(e0_F_DLRefluxn) *(e0_x_D_i2)+(e0e1_F_Vn_tr2) *(e0e1_y_tr2_i2)-(e0e1_F_Ln_tr1) *(e0e1_x_tr1_i2)-(e0_F_DVn) *(e0_y_D_i2)+e0e1_r_n_tr1_i2) ,
    0.0-((e0e1_F_Fn_tr2) *(e0_x_F_i2)+(e0e1_F_Ln_tr1) *(e0e1_x_tr1_i2)+(e0e1_F_Vn_tr3) *(e0e1_y_tr3_i2)-(e0e1_F_Ln_tr2) *(e0e1_x_tr2_i2)-(e0e1_F_Vn_tr2) *(e0e1_y_tr2_i2)+e0e1_r_n_tr2_i2) ,
    0.0-((e0e1_F_Fn_tr3) *(e0_x_F_i2)+(e0e1_F_Ln_tr2) *(e0e1_x_tr2_i2)+(e0e1_F_Vn_tr4) *(e0e1_y_tr4_i2)-(e0e1_F_Ln_tr3) *(e0e1_x_tr3_i2)-(e0e1_F_Vn_tr3) *(e0e1_y_tr3_i2)+e0e1_r_n_tr3_i2) ,
    0.0-((e0e1_F_Fn_tr4) *(e0_x_F_i2)+(e0e1_F_Ln_tr3) *(e0e1_x_tr3_i2)+(e0e1_F_Vn_tr5) *(e0e1_y_tr5_i2)-(e0e1_F_Ln_tr4) *(e0e1_x_tr4_i2)-(e0e1_F_Vn_tr4) *(e0e1_y_tr4_i2)+e0e1_r_n_tr4_i2) ,
    0.0-((e0e1_F_Fn_tr5) *(e0_x_F_i2)+(e0e1_F_Ln_tr4) *(e0e1_x_tr4_i2)+(e0e1_F_Vn_tr6) *(e0e1_y_tr6_i2)-(e0e1_F_Ln_tr5) *(e0e1_x_tr5_i2)-(e0e1_F_Vn_tr5) *(e0e1_y_tr5_i2)+e0e1_r_n_tr5_i2) ,
    0.0-((e0e1_F_Fn_tr6) *(e0_x_F_i2)+(e0e1_F_Ln_tr5) *(e0e1_x_tr5_i2)+(e0e1_F_Vn_tr7) *(e0e1_y_tr7_i2)-(e0e1_F_Ln_tr6) *(e0e1_x_tr6_i2)-(e0e1_F_Vn_tr6) *(e0e1_y_tr6_i2)+e0e1_r_n_tr6_i2) ,
    0.0-((e0e1_F_Fn_tr7) *(e0_x_F_i2)+(e0e1_F_Ln_tr6) *(e0e1_x_tr6_i2)+(e0e1_F_Vn_tr8) *(e0e1_y_tr8_i2)-(e0e1_F_Ln_tr7) *(e0e1_x_tr7_i2)-(e0e1_F_Vn_tr7) *(e0e1_y_tr7_i2)+e0e1_r_n_tr7_i2) ,
    0.0-((e0e1_F_Fn_tr8) *(e0_x_F_i2)+(e0e1_F_Ln_tr7) *(e0e1_x_tr7_i2)+(e0e1_F_Vn_tr9) *(e0e1_y_tr9_i2)-(e0e1_F_Ln_tr8) *(e0e1_x_tr8_i2)-(e0e1_F_Vn_tr8) *(e0e1_y_tr8_i2)+e0e1_r_n_tr8_i2) ,
    0.0-((e0e1_F_Fn_tr9) *(e0_x_F_i2)+(e0e1_F_Ln_tr8) *(e0e1_x_tr8_i2)+(e0e1_F_Vn_tr10) *(e0e1_y_tr10_i2)-(e0e1_F_Ln_tr9) *(e0e1_x_tr9_i2)-(e0e1_F_Vn_tr9) *(e0e1_y_tr9_i2)+e0e1_r_n_tr9_i2) ,
    0.0-((e0e1_F_Fn_tr10) *(e0_x_F_i2)+(e0e1_F_Ln_tr9) *(e0e1_x_tr9_i2)+(e0e1_F_Vn_tr11) *(e0e1_y_tr11_i2)-(e0e1_F_Ln_tr10) *(e0e1_x_tr10_i2)-(e0e1_F_Vn_tr10) *(e0e1_y_tr10_i2)+e0e1_r_n_tr10_i2) ,
    0.0-((e0e1_F_Fn_tr11) *(e0_x_F_i2)+(e0e1_F_Ln_tr10) *(e0e1_x_tr10_i2)+(e0e1_F_Vn_tr12) *(e0e1_y_tr12_i2)-(e0e1_F_Ln_tr11) *(e0e1_x_tr11_i2)-(e0e1_F_Vn_tr11) *(e0e1_y_tr11_i2)+e0e1_r_n_tr11_i2) ,
    0.0-((e0e1_F_Fn_tr12) *(e0_x_F_i2)+(e0e1_F_Ln_tr11) *(e0e1_x_tr11_i2)+(e0e1_F_Vn_tr13) *(e0e1_y_tr13_i2)-(e0e1_F_Ln_tr12) *(e0e1_x_tr12_i2)-(e0e1_F_Vn_tr12) *(e0e1_y_tr12_i2)+e0e1_r_n_tr12_i2) ,
    0.0-((e0e1_F_Fn_tr13) *(e0_x_F_i2)+(e0e1_F_Ln_tr12) *(e0e1_x_tr12_i2)+(e0e1_F_Vn_tr14) *(e0e1_y_tr14_i2)-(e0e1_F_Ln_tr13) *(e0e1_x_tr13_i2)-(e0e1_F_Vn_tr13) *(e0e1_y_tr13_i2)+e0e1_r_n_tr13_i2) ,
    0.0-((e0e1_F_Fn_tr14) *(e0_x_F_i2)+(e0e1_F_Ln_tr13) *(e0e1_x_tr13_i2)+(e0e1_F_Vn_tr15) *(e0e1_y_tr15_i2)-(e0e1_F_Ln_tr14) *(e0e1_x_tr14_i2)-(e0e1_F_Vn_tr14) *(e0e1_y_tr14_i2)+e0e1_r_n_tr14_i2) ,
    0.0-((e0e1_F_Fn_tr15) *(e0_x_F_i2)+(e0e1_F_Ln_tr14) *(e0e1_x_tr14_i2)+(e0e1_F_Vn_tr16) *(e0e1_y_tr16_i2)-(e0e1_F_Ln_tr15) *(e0e1_x_tr15_i2)-(e0e1_F_Vn_tr15) *(e0e1_y_tr15_i2)+e0e1_r_n_tr15_i2) ,
    0.0-((e0e1_F_Fn_tr16) *(e0_x_F_i2)+(e0e1_F_Ln_tr15) *(e0e1_x_tr15_i2)+(e0e1_F_Vn_tr17) *(e0e1_y_tr17_i2)-(e0e1_F_Ln_tr16) *(e0e1_x_tr16_i2)-(e0e1_F_Vn_tr16) *(e0e1_y_tr16_i2)+e0e1_r_n_tr16_i2) ,
    0.0-((e0e1_F_Fn_tr17) *(e0_x_F_i2)+(e0e1_F_Ln_tr16) *(e0e1_x_tr16_i2)+(e0e1_F_Vn_tr18) *(e0e1_y_tr18_i2)-(e0e1_F_Ln_tr17) *(e0e1_x_tr17_i2)-(e0e1_F_Vn_tr17) *(e0e1_y_tr17_i2)+e0e1_r_n_tr17_i2) ,
    0.0-((e0e1_F_Fn_tr18) *(e0_x_F_i2)+(e0e1_F_Ln_tr17) *(e0e1_x_tr17_i2)+(e0e1_F_Vn_tr19) *(e0e1_y_tr19_i2)-(e0e1_F_Ln_tr18) *(e0e1_x_tr18_i2)-(e0e1_F_Vn_tr18) *(e0e1_y_tr18_i2)+e0e1_r_n_tr18_i2) ,
    0.0-((e0e1_F_Fn_tr19) *(e0_x_F_i2)+(e0e1_F_Ln_tr18) *(e0e1_x_tr18_i2)+(e0e1_F_Vn_tr20) *(e0e1_y_tr20_i2)-(e0e1_F_Ln_tr19) *(e0e1_x_tr19_i2)-(e0e1_F_Vn_tr19) *(e0e1_y_tr19_i2)+e0e1_r_n_tr19_i2) ,
    0.0-((e0e1_F_Fn_tr20) *(e0_x_F_i2)+(e0e1_F_Ln_tr19) *(e0e1_x_tr19_i2)+(e0_F_ReboilerVn) *(e0_y_Reboiler_i2)-(e0e1_F_Ln_tr20) *(e0_x_Reboilerin_i2)-(e0e1_F_Vn_tr20) *(e0e1_y_tr20_i2)+e0e1_r_n_tr20_i2) ,
    e0e1_A-(((e0e1_greek_pi) *(((e0e1_d))**(2.0)))/(4.0)) ,
    1.0-((e0e1_x_B_i1+e0e1_x_B_i2)) ,
    e0e1_h_BLn-(((e0e1_x_B_i1) *(e0e1_h_BLn_i1)+e0e1_h_BELn+(e0e1_x_B_i2) *(e0e1_h_BLn_i2)+e0e1_h_BELn)) ,
    0.0-((e0_x_Reboiler_i1) *(e0_F_LReboilern)+(e0_x_Reboilerin_i1) *(e0e1_F_Ln_tr21)-(e0e1_x_B_i1) *(e0e1_F_BLn)) ,
    0.0-((e0_x_Reboiler_i2) *(e0_F_LReboilern)+(e0_x_Reboilerin_i2) *(e0e1_F_Ln_tr21)-(e0e1_x_B_i2) *(e0e1_F_BLn)) ,
    0.0-((e0_h_LReboilern) *(e0_F_LReboilern)+(e0_h_LReboilerinn) *(e0e1_F_Ln_tr21)-(e0e1_h_BLn) *(e0e1_F_BLn)) ,
    e0_greek_Deltap_D-(e0_p_DV-e0_p_DL) ,
    e0_y_DV_i1-((e0_K_D_i1) *(e0_x_D_i1)) ,
    e0_y_DV_i2-((e0_K_D_i2) *(e0_x_D_i2)) ,
    1.0-((e0_x_D_i1+e0_x_D_i2)) ,
    1.0-((e0_y_DV_i1+e0_y_DV_i2)) ,
    (e0_R_Reflux) *(e0_F_DLn)-(e0_F_DLRefluxn) ,
    e0_h_DLn-(((e0_x_D_i1) *(e0_h_DLn_i1)+(e0_x_D_i2) *(e0_h_DLn_i2))+e0_h_DELn) ,
    (e0_K_D_i1) *(e0_p_DL)-(e0_p_DLV_i1) ,
    (e0_K_D_i2) *(e0_p_DL)-(e0_p_DLV_i2) ,
    0.0-((e0_F_DVn) *(e0_y_D_i1)-((e0_F_DLRefluxn+e0_F_DLn)) *(e0_x_D_i1)) ,
    0.0-((e0_F_DVn) *(e0_y_D_i2)-((e0_F_DLRefluxn+e0_F_DLn)) *(e0_x_D_i2)) ,
    0.0-((e0_F_DVn) *(e0_h_DVn)-((e0_F_DLRefluxn+e0_F_DLn)) *(e0_h_DLn)-e0_Q_DLoss+e0_Q_D) ,
    e0_y_Reboiler_i1-((e0_K_Reboiler_i1) *(e0_x_Reboiler_i1)) ,
    e0_y_Reboiler_i2-((e0_K_Reboiler_i2) *(e0_x_Reboiler_i2)) ,
    e0_greek_Deltap_Reboiler-(e0_p_LReboiler-e0_p_ReboilerV) ,
    1.0-((e0_x_Reboiler_i1+e0_x_Reboiler_i2)) ,
    1.0-((e0_y_Reboiler_i1+e0_y_Reboiler_i2)) ,
    0.0-((e0_F_LReboilerinn) *(e0_h_LReboilerinn)-(e0_h_ReboilerVn) *(e0_F_ReboilerVn)+e0_Q_Reboiler+e0_Q_LossReboiler-(e0_F_LReboilern) *(e0_h_LReboilern)) ,
    e0_h_ReboilerVn-(((e0_y_Reboiler_i1) *(e0_h_ReboilerVn_i1)+(e0_y_Reboiler_i2) *(e0_h_ReboilerVn_i2))+e0_h_EReboilerVn) ,
    e0_h_LReboilern-(((e0_x_Reboiler_i1) *(e0_h_LReboilern_i1)+(e0_x_Reboiler_i2) *(e0_h_LReboilern_i2))+e0_h_ELReboilern) ,
    (e0_K_Reboiler_i1) *(e0_p_ReboilerV)-(e0_p_LVReboiler_i1) ,
    (e0_K_Reboiler_i2) *(e0_p_ReboilerV)-(e0_p_LVReboiler_i2) ,
    0.0-((e0_F_LReboilerinn) *(e0_x_Reboilerin_i1)-(e0_F_ReboilerVn) *(e0_y_Reboiler_i1)-(e0_F_LReboilern) *(e0_x_Reboiler_i1)) ,
    0.0-((e0_F_LReboilerinn) *(e0_x_Reboilerin_i2)-(e0_F_ReboilerVn) *(e0_y_Reboiler_i2)-(e0_F_LReboilern) *(e0_x_Reboiler_i2)) 

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

    x = numpy.empty(554)
    xInitial = numpy.empty((554), dtype = object) 
    parameter = numpy.empty(292)
    dict_variables = {}

    # Iteration variable initializatio
    x[0] = 0.93251 	# e0e1_x_tr1_i1
    x[1] = 0.91001 	# e0e1_x_tr2_i1
    x[2] = 0.88151 	# e0e1_x_tr3_i1
    x[3] = 0.84607 	# e0e1_x_tr4_i1
    x[4] = 0.80302 	# e0e1_x_tr5_i1
    x[5] = 0.75219 	# e0e1_x_tr6_i1
    x[6] = 0.6941 	# e0e1_x_tr7_i1
    x[7] = 0.63014 	# e0e1_x_tr8_i1
    x[8] = 0.5625 	# e0e1_x_tr9_i1
    x[9] = 0.4939 	# e0e1_x_tr10_i1
    x[10] = 0.42721 	# e0e1_x_tr11_i1
    x[11] = 0.36494 	# e0e1_x_tr12_i1
    x[12] = 0.30895 	# e0e1_x_tr13_i1
    x[13] = 0.26025 	# e0e1_x_tr14_i1
    x[14] = 0.2191 	# e0e1_x_tr15_i1
    x[15] = 0.18517 	# e0e1_x_tr16_i1
    x[16] = 0.15773 	# e0e1_x_tr17_i1
    x[17] = 0.1359 	# e0e1_x_tr18_i1
    x[18] = 0.10513 	# e0e1_x_tr19_i1
    x[19] = 0.080334 	# e0_x_Reboilerin_i1
    x[20] = 0.05 	# e0_x_D_i2
    x[21] = 0.067489 	# e0e1_x_tr1_i2
    x[22] = 0.089986 	# e0e1_x_tr2_i2
    x[23] = 0.11849 	# e0e1_x_tr3_i2
    x[24] = 0.15393 	# e0e1_x_tr4_i2
    x[25] = 0.19698 	# e0e1_x_tr5_i2
    x[26] = 0.24781 	# e0e1_x_tr6_i2
    x[27] = 0.3059 	# e0e1_x_tr7_i2
    x[28] = 0.36986 	# e0e1_x_tr8_i2
    x[29] = 0.4375 	# e0e1_x_tr9_i2
    x[30] = 0.5061 	# e0e1_x_tr10_i2
    x[31] = 0.57279 	# e0e1_x_tr11_i2
    x[32] = 0.63506 	# e0e1_x_tr12_i2
    x[33] = 0.69105 	# e0e1_x_tr13_i2
    x[34] = 0.73975 	# e0e1_x_tr14_i2
    x[35] = 0.7809 	# e0e1_x_tr15_i2
    x[36] = 0.81483 	# e0e1_x_tr16_i2
    x[37] = 0.84227 	# e0e1_x_tr17_i2
    x[38] = 0.8641 	# e0e1_x_tr18_i2
    x[39] = 0.89487 	# e0e1_x_tr19_i2
    x[40] = 0.91967 	# e0_x_Reboilerin_i2
    x[41] = 0.92 	# e0e1_x_B_i2
    x[42] = 0.060646 	# e0_x_Reboiler_i1
    x[43] = 0.93935 	# e0_x_Reboiler_i2
    x[44] = 0.95 	# e0_y_D_i1
    x[45] = 0.93289 	# e0e1_y_tr2_i1
    x[46] = 0.91088 	# e0e1_y_tr3_i1
    x[47] = 0.88299 	# e0e1_y_tr4_i1
    x[48] = 0.84832 	# e0e1_y_tr5_i1
    x[49] = 0.80622 	# e0e1_y_tr6_i1
    x[50] = 0.75651 	# e0e1_y_tr7_i1
    x[51] = 0.69972 	# e0e1_y_tr8_i1
    x[52] = 0.63719 	# e0e1_y_tr9_i1
    x[53] = 0.57107 	# e0e1_y_tr10_i1
    x[54] = 0.50404 	# e0e1_y_tr11_i1
    x[55] = 0.43887 	# e0e1_y_tr12_i1
    x[56] = 0.37805 	# e0e1_y_tr13_i1
    x[57] = 0.32336 	# e0e1_y_tr14_i1
    x[58] = 0.2758 	# e0e1_y_tr15_i1
    x[59] = 0.23561 	# e0e1_y_tr16_i1
    x[60] = 0.20247 	# e0e1_y_tr17_i1
    x[61] = 0.17568 	# e0e1_y_tr18_i1
    x[62] = 0.13726 	# e0e1_y_tr19_i1
    x[63] = 0.10574 	# e0e1_y_tr20_i1
    x[64] = 0.080342 	# e0_y_Reboiler_i1
    x[65] = 0.05 	# e0_y_D_i2
    x[66] = 0.067111 	# e0e1_y_tr2_i2
    x[67] = 0.089121 	# e0e1_y_tr3_i2
    x[68] = 0.11701 	# e0e1_y_tr4_i2
    x[69] = 0.15168 	# e0e1_y_tr5_i2
    x[70] = 0.19378 	# e0e1_y_tr6_i2
    x[71] = 0.24349 	# e0e1_y_tr7_i2
    x[72] = 0.30028 	# e0e1_y_tr8_i2
    x[73] = 0.36281 	# e0e1_y_tr9_i2
    x[74] = 0.42893 	# e0e1_y_tr10_i2
    x[75] = 0.49596 	# e0e1_y_tr11_i2
    x[76] = 0.56113 	# e0e1_y_tr12_i2
    x[77] = 0.62195 	# e0e1_y_tr13_i2
    x[78] = 0.67664 	# e0e1_y_tr14_i2
    x[79] = 0.7242 	# e0e1_y_tr15_i2
    x[80] = 0.76439 	# e0e1_y_tr16_i2
    x[81] = 0.79753 	# e0e1_y_tr17_i2
    x[82] = 0.82432 	# e0e1_y_tr18_i2
    x[83] = 0.86274 	# e0e1_y_tr19_i2
    x[84] = 0.89426 	# e0e1_y_tr20_i2
    x[85] = 0.91966 	# e0_y_Reboiler_i2
    x[86] = 0.027517 	# e0_F_DLn
    x[87] = 1.0138 	# e0_K_D_i1
    x[88] = 0.73706 	# e0_K_D_i2
    x[89] = -2.21493718104E7 	# e0_Q_D
    x[90] = 45.3946 	# e0_R_Reflux
    x[91] = 324.6319 	# e0_T_D
    x[92] = 699958.0 	# e0_p_DL
    x[93] = 0.96315 	# e0_y_DV_i1
    x[94] = 0.036853 	# e0_y_DV_i2
    x[95] = 1.4369E-4 	# e0e1_greek_eta_L_tr1
    x[96] = 1.3248 	# e0_K_Reboiler_i1
    x[97] = 0.97903 	# e0_K_Reboiler_i2
    x[98] = 2.21309864062E7 	# e0_Q_Reboiler
    x[99] = 336.0789 	# e0_T_Reboiler
    x[100] = 1.4298E-4 	# e0e1_greek_eta_L_tr2
    x[101] = 700798.0 	# e0_p_ReboilerV
    x[102] = 1.4208E-4 	# e0e1_greek_eta_L_tr3
    x[103] = 1.4095E-4 	# e0e1_greek_eta_L_tr4
    x[104] = 1.3957E-4 	# e0e1_greek_eta_L_tr5
    x[105] = 1.3793E-4 	# e0e1_greek_eta_L_tr6
    x[106] = 1.3606E-4 	# e0e1_greek_eta_L_tr7
    x[107] = 1.3397E-4 	# e0e1_greek_eta_L_tr8
    x[108] = 1.3175E-4 	# e0e1_greek_eta_L_tr9
    x[109] = 1.2948E-4 	# e0e1_greek_eta_L_tr10
    x[110] = 1.2725E-4 	# e0e1_greek_eta_L_tr11
    x[111] = 1.2515E-4 	# e0e1_greek_eta_L_tr12
    x[112] = 1.2325E-4 	# e0e1_greek_eta_L_tr13
    x[113] = 1.2159E-4 	# e0e1_greek_eta_L_tr14
    x[114] = 1.2017E-4 	# e0e1_greek_eta_L_tr15
    x[115] = 1.19E-4 	# e0e1_greek_eta_L_tr16
    x[116] = 1.1804E-4 	# e0e1_greek_eta_L_tr17
    x[117] = 1.1728E-4 	# e0e1_greek_eta_L_tr18
    x[118] = 1.162E-4 	# e0e1_greek_eta_L_tr19
    x[119] = 1.1533E-4 	# e0e1_greek_eta_L_tr20
    x[120] = 0.5 	# e0_x_F_i2
    x[121] = 8.1896E-6 	# e0e1_greek_eta_V_tr1
    x[122] = 8.1957E-6 	# e0e1_greek_eta_V_tr2
    x[123] = 8.2033E-6 	# e0e1_greek_eta_V_tr3
    x[124] = 8.2129E-6 	# e0e1_greek_eta_V_tr4
    x[125] = 8.2247E-6 	# e0e1_greek_eta_V_tr5
    x[126] = 8.2386E-6 	# e0e1_greek_eta_V_tr6
    x[127] = 8.2548E-6 	# e0e1_greek_eta_V_tr7
    x[128] = 8.2728E-6 	# e0e1_greek_eta_V_tr8
    x[129] = 8.2921E-6 	# e0e1_greek_eta_V_tr9
    x[130] = 8.3119E-6 	# e0e1_greek_eta_V_tr10
    x[131] = 8.3315E-6 	# e0e1_greek_eta_V_tr11
    x[132] = 8.3501E-6 	# e0e1_greek_eta_V_tr12
    x[133] = 8.367E-6 	# e0e1_greek_eta_V_tr13
    x[134] = 8.3819E-6 	# e0e1_greek_eta_V_tr14
    x[135] = 8.3946E-6 	# e0e1_greek_eta_V_tr15
    x[136] = 8.4052E-6 	# e0e1_greek_eta_V_tr16
    x[137] = 8.4138E-6 	# e0e1_greek_eta_V_tr17
    x[138] = 8.4208E-6 	# e0e1_greek_eta_V_tr18
    x[139] = 8.4305E-6 	# e0e1_greek_eta_V_tr19
    x[140] = 8.4385E-6 	# e0e1_greek_eta_V_tr20
    x[141] = 517.6996 	# e0e1_greek_rho_Lm_tr1
    x[142] = 517.8395 	# e0e1_greek_rho_Lm_tr2
    x[143] = 518.017 	# e0e1_greek_rho_Lm_tr3
    x[144] = 518.237 	# e0e1_greek_rho_Lm_tr4
    x[145] = 518.503 	# e0e1_greek_rho_Lm_tr5
    x[146] = 518.8148 	# e0e1_greek_rho_Lm_tr6
    x[147] = 519.1677 	# e0e1_greek_rho_Lm_tr7
    x[148] = 519.5519 	# e0e1_greek_rho_Lm_tr8
    x[149] = 519.953 	# e0e1_greek_rho_Lm_tr9
    x[150] = 520.3542 	# e0e1_greek_rho_Lm_tr10
    x[151] = 520.7389 	# e0e1_greek_rho_Lm_tr11
    x[152] = 521.0931 	# e0e1_greek_rho_Lm_tr12
    x[153] = 521.4076 	# e0e1_greek_rho_Lm_tr13
    x[154] = 521.678 	# e0e1_greek_rho_Lm_tr14
    x[155] = 521.904 	# e0e1_greek_rho_Lm_tr15
    x[156] = 522.0885 	# e0e1_greek_rho_Lm_tr16
    x[157] = 522.2361 	# e0e1_greek_rho_Lm_tr17
    x[158] = 522.3524 	# e0e1_greek_rho_Lm_tr18
    x[159] = 522.517 	# e0e1_greek_rho_Lm_tr19
    x[160] = 522.6484 	# e0e1_greek_rho_Lm_tr20
    x[161] = 15.0652 	# e0e1_greek_rho_Vm_tr1
    x[162] = 15.0541 	# e0e1_greek_rho_Vm_tr2
    x[163] = 15.0397 	# e0e1_greek_rho_Vm_tr3
    x[164] = 15.0215 	# e0e1_greek_rho_Vm_tr4
    x[165] = 14.999 	# e0e1_greek_rho_Vm_tr5
    x[166] = 14.9721 	# e0e1_greek_rho_Vm_tr6
    x[167] = 14.9409 	# e0e1_greek_rho_Vm_tr7
    x[168] = 14.9061 	# e0e1_greek_rho_Vm_tr8
    x[169] = 14.8687 	# e0e1_greek_rho_Vm_tr9
    x[170] = 14.8303 	# e0e1_greek_rho_Vm_tr10
    x[171] = 14.7924 	# e0e1_greek_rho_Vm_tr11
    x[172] = 14.7567 	# e0e1_greek_rho_Vm_tr12
    x[173] = 14.7242 	# e0e1_greek_rho_Vm_tr13
    x[174] = 14.6958 	# e0e1_greek_rho_Vm_tr14
    x[175] = 14.6717 	# e0e1_greek_rho_Vm_tr15
    x[176] = 14.6518 	# e0e1_greek_rho_Vm_tr16
    x[177] = 14.6357 	# e0e1_greek_rho_Vm_tr17
    x[178] = 14.623 	# e0e1_greek_rho_Vm_tr18
    x[179] = 14.6047 	# e0e1_greek_rho_Vm_tr19
    x[180] = 14.5901 	# e0e1_greek_rho_Vm_tr20
    x[181] = 0.0070979 	# e0e1_greek_sigma_tr1
    x[182] = 0.0071096 	# e0e1_greek_sigma_tr2
    x[183] = 0.0071246 	# e0e1_greek_sigma_tr3
    x[184] = 0.0071434 	# e0e1_greek_sigma_tr4
    x[185] = 0.0071664 	# e0e1_greek_sigma_tr5
    x[186] = 0.0071939 	# e0e1_greek_sigma_tr6
    x[187] = 0.0072257 	# e0e1_greek_sigma_tr7
    x[188] = 0.0072612 	# e0e1_greek_sigma_tr8
    x[189] = 0.0072993 	# e0e1_greek_sigma_tr9
    x[190] = 0.0073385 	# e0e1_greek_sigma_tr10
    x[191] = 0.0073773 	# e0e1_greek_sigma_tr11
    x[192] = 0.0074142 	# e0e1_greek_sigma_tr12
    x[193] = 0.0074479 	# e0e1_greek_sigma_tr13
    x[194] = 0.0074776 	# e0e1_greek_sigma_tr14
    x[195] = 0.007503 	# e0e1_greek_sigma_tr15
    x[196] = 0.0075242 	# e0e1_greek_sigma_tr16
    x[197] = 0.0075415 	# e0e1_greek_sigma_tr17
    x[198] = 0.0075553 	# e0e1_greek_sigma_tr18
    x[199] = 0.007575 	# e0e1_greek_sigma_tr19
    x[200] = 0.0075911 	# e0e1_greek_sigma_tr20
    x[201] = 1.0 	# e0e1_greek_theta_i1_j1_tr1
    x[202] = 1.0 	# e0e1_greek_theta_i1_j1_tr2
    x[203] = 1.0 	# e0e1_greek_theta_i1_j1_tr3
    x[204] = 1.0 	# e0e1_greek_theta_i1_j1_tr4
    x[205] = 1.0 	# e0e1_greek_theta_i1_j1_tr5
    x[206] = 1.0 	# e0e1_greek_theta_i1_j1_tr6
    x[207] = 1.0 	# e0e1_greek_theta_i1_j1_tr7
    x[208] = 1.0 	# e0e1_greek_theta_i1_j1_tr8
    x[209] = 1.0 	# e0e1_greek_theta_i1_j1_tr9
    x[210] = 1.0 	# e0e1_greek_theta_i1_j1_tr10
    x[211] = 1.0 	# e0e1_greek_theta_i1_j1_tr11
    x[212] = 1.0 	# e0e1_greek_theta_i1_j1_tr12
    x[213] = 1.0 	# e0e1_greek_theta_i1_j1_tr13
    x[214] = 1.0 	# e0e1_greek_theta_i1_j1_tr14
    x[215] = 1.0 	# e0e1_greek_theta_i1_j1_tr15
    x[216] = 1.0 	# e0e1_greek_theta_i1_j1_tr16
    x[217] = 1.0 	# e0e1_greek_theta_i1_j1_tr17
    x[218] = 1.0 	# e0e1_greek_theta_i1_j1_tr18
    x[219] = 1.0 	# e0e1_greek_theta_i1_j1_tr19
    x[220] = 1.0 	# e0e1_greek_theta_i1_j1_tr20
    x[221] = 1.0009 	# e0e1_greek_theta_i1_j2_tr1
    x[222] = 1.0009 	# e0e1_greek_theta_i1_j2_tr2
    x[223] = 1.0009 	# e0e1_greek_theta_i1_j2_tr3
    x[224] = 1.0009 	# e0e1_greek_theta_i1_j2_tr4
    x[225] = 1.0009 	# e0e1_greek_theta_i1_j2_tr5
    x[226] = 1.0009 	# e0e1_greek_theta_i1_j2_tr6
    x[227] = 1.0009 	# e0e1_greek_theta_i1_j2_tr7
    x[228] = 1.001 	# e0e1_greek_theta_i1_j2_tr8
    x[229] = 1.001 	# e0e1_greek_theta_i1_j2_tr9
    x[230] = 1.001 	# e0e1_greek_theta_i1_j2_tr10
    x[231] = 1.001 	# e0e1_greek_theta_i1_j2_tr11
    x[232] = 1.001 	# e0e1_greek_theta_i1_j2_tr12
    x[233] = 1.001 	# e0e1_greek_theta_i1_j2_tr13
    x[234] = 1.001 	# e0e1_greek_theta_i1_j2_tr14
    x[235] = 1.0011 	# e0e1_greek_theta_i1_j2_tr15
    x[236] = 1.0011 	# e0e1_greek_theta_i1_j2_tr16
    x[237] = 1.0011 	# e0e1_greek_theta_i1_j2_tr17
    x[238] = 1.0011 	# e0e1_greek_theta_i1_j2_tr18
    x[239] = 1.0011 	# e0e1_greek_theta_i1_j2_tr19
    x[240] = 1.0011 	# e0e1_greek_theta_i1_j2_tr20
    x[241] = 0.99911 	# e0e1_greek_theta_i2_j1_tr1
    x[242] = 0.9991 	# e0e1_greek_theta_i2_j1_tr2
    x[243] = 0.9991 	# e0e1_greek_theta_i2_j1_tr3
    x[244] = 0.99909 	# e0e1_greek_theta_i2_j1_tr4
    x[245] = 0.99908 	# e0e1_greek_theta_i2_j1_tr5
    x[246] = 0.99907 	# e0e1_greek_theta_i2_j1_tr6
    x[247] = 0.99905 	# e0e1_greek_theta_i2_j1_tr7
    x[248] = 0.99904 	# e0e1_greek_theta_i2_j1_tr8
    x[249] = 0.99902 	# e0e1_greek_theta_i2_j1_tr9
    x[250] = 0.99901 	# e0e1_greek_theta_i2_j1_tr10
    x[251] = 0.99899 	# e0e1_greek_theta_i2_j1_tr11
    x[252] = 0.99898 	# e0e1_greek_theta_i2_j1_tr12
    x[253] = 0.99897 	# e0e1_greek_theta_i2_j1_tr13
    x[254] = 0.99895 	# e0e1_greek_theta_i2_j1_tr14
    x[255] = 0.99895 	# e0e1_greek_theta_i2_j1_tr15
    x[256] = 0.99894 	# e0e1_greek_theta_i2_j1_tr16
    x[257] = 0.99893 	# e0e1_greek_theta_i2_j1_tr17
    x[258] = 0.99893 	# e0e1_greek_theta_i2_j1_tr18
    x[259] = 0.99892 	# e0e1_greek_theta_i2_j1_tr19
    x[260] = 0.99892 	# e0e1_greek_theta_i2_j1_tr20
    x[261] = 1.0 	# e0e1_greek_theta_i2_j2_tr1
    x[262] = 1.0 	# e0e1_greek_theta_i2_j2_tr2
    x[263] = 1.0 	# e0e1_greek_theta_i2_j2_tr3
    x[264] = 1.0 	# e0e1_greek_theta_i2_j2_tr4
    x[265] = 1.0 	# e0e1_greek_theta_i2_j2_tr5
    x[266] = 1.0 	# e0e1_greek_theta_i2_j2_tr6
    x[267] = 1.0 	# e0e1_greek_theta_i2_j2_tr7
    x[268] = 1.0 	# e0e1_greek_theta_i2_j2_tr8
    x[269] = 1.0 	# e0e1_greek_theta_i2_j2_tr9
    x[270] = 1.0 	# e0e1_greek_theta_i2_j2_tr10
    x[271] = 1.0 	# e0e1_greek_theta_i2_j2_tr11
    x[272] = 1.0 	# e0e1_greek_theta_i2_j2_tr12
    x[273] = 1.0 	# e0e1_greek_theta_i2_j2_tr13
    x[274] = 1.0 	# e0e1_greek_theta_i2_j2_tr14
    x[275] = 1.0 	# e0e1_greek_theta_i2_j2_tr15
    x[276] = 1.0 	# e0e1_greek_theta_i2_j2_tr16
    x[277] = 1.0 	# e0e1_greek_theta_i2_j2_tr17
    x[278] = 1.0 	# e0e1_greek_theta_i2_j2_tr18
    x[279] = 1.0 	# e0e1_greek_theta_i2_j2_tr19
    x[280] = 1.0 	# e0e1_greek_theta_i2_j2_tr20
    x[281] = 1.2868 	# e0e1_A
    x[282] = 0.029483 	# e0e1_F_BLn
    x[283] = 1.2076 	# e0_F_LReboilerinn
    x[284] = 1.2491 	# e0_F_DLRefluxn
    x[285] = 1.2475 	# e0e1_F_Ln_tr1
    x[286] = 1.2454 	# e0e1_F_Ln_tr2
    x[287] = 1.2428 	# e0e1_F_Ln_tr3
    x[288] = 1.2396 	# e0e1_F_Ln_tr4
    x[289] = 1.2358 	# e0e1_F_Ln_tr5
    x[290] = 1.2313 	# e0e1_F_Ln_tr6
    x[291] = 1.2264 	# e0e1_F_Ln_tr7
    x[292] = 1.2211 	# e0e1_F_Ln_tr8
    x[293] = 1.2157 	# e0e1_F_Ln_tr9
    x[294] = 1.2104 	# e0e1_F_Ln_tr10
    x[295] = 1.2054 	# e0e1_F_Ln_tr11
    x[296] = 1.2009 	# e0e1_F_Ln_tr12
    x[297] = 1.197 	# e0e1_F_Ln_tr13
    x[298] = 1.1936 	# e0e1_F_Ln_tr14
    x[299] = 1.1909 	# e0e1_F_Ln_tr15
    x[300] = 1.1886 	# e0e1_F_Ln_tr16
    x[301] = 1.1869 	# e0e1_F_Ln_tr17
    x[302] = 1.2402 	# e0e1_F_Ln_tr18
    x[303] = 1.2382 	# e0e1_F_Ln_tr19
    x[304] = 1.2366 	# e0e1_F_Ln_tr20
    x[305] = 0.028983 	# e0e1_F_Ln_tr21
    x[306] = 1.2767 	# e0_F_DVn
    x[307] = 1.275 	# e0e1_F_Vn_tr2
    x[308] = 1.2729 	# e0e1_F_Vn_tr3
    x[309] = 1.2703 	# e0e1_F_Vn_tr4
    x[310] = 1.2671 	# e0e1_F_Vn_tr5
    x[311] = 1.2633 	# e0e1_F_Vn_tr6
    x[312] = 1.2589 	# e0e1_F_Vn_tr7
    x[313] = 1.2539 	# e0e1_F_Vn_tr8
    x[314] = 1.2487 	# e0e1_F_Vn_tr9
    x[315] = 1.2432 	# e0e1_F_Vn_tr10
    x[316] = 1.2379 	# e0e1_F_Vn_tr11
    x[317] = 1.233 	# e0e1_F_Vn_tr12
    x[318] = 1.2284 	# e0e1_F_Vn_tr13
    x[319] = 1.2245 	# e0e1_F_Vn_tr14
    x[320] = 1.2211 	# e0e1_F_Vn_tr15
    x[321] = 1.2184 	# e0e1_F_Vn_tr16
    x[322] = 1.2162 	# e0e1_F_Vn_tr17
    x[323] = 1.2144 	# e0e1_F_Vn_tr18
    x[324] = 1.2107 	# e0e1_F_Vn_tr19
    x[325] = 1.2087 	# e0e1_F_Vn_tr20
    x[326] = 1.2071 	# e0_F_ReboilerVn
    x[327] = 1.0188 	# e0e1_K_tr1_i1
    x[328] = 1.0251 	# e0e1_K_tr2_i1
    x[329] = 1.0333 	# e0e1_K_tr3_i1
    x[330] = 1.0436 	# e0e1_K_tr4_i1
    x[331] = 1.0564 	# e0e1_K_tr5_i1
    x[332] = 1.0718 	# e0e1_K_tr6_i1
    x[333] = 1.0899 	# e0e1_K_tr7_i1
    x[334] = 1.1104 	# e0e1_K_tr8_i1
    x[335] = 1.1328 	# e0e1_K_tr9_i1
    x[336] = 1.1563 	# e0e1_K_tr10_i1
    x[337] = 1.1798 	# e0e1_K_tr11_i1
    x[338] = 1.2026 	# e0e1_K_tr12_i1
    x[339] = 1.2237 	# e0e1_K_tr13_i1
    x[340] = 1.2425 	# e0e1_K_tr14_i1
    x[341] = 1.2587 	# e0e1_K_tr15_i1
    x[342] = 1.2724 	# e0e1_K_tr16_i1
    x[343] = 1.2837 	# e0e1_K_tr17_i1
    x[344] = 1.2927 	# e0e1_K_tr18_i1
    x[345] = 1.3057 	# e0e1_K_tr19_i1
    x[346] = 1.3163 	# e0e1_K_tr20_i1
    x[347] = 0.74086 	# e0e1_K_tr1_i2
    x[348] = 0.7458 	# e0e1_K_tr2_i2
    x[349] = 0.75213 	# e0e1_K_tr3_i2
    x[350] = 0.76012 	# e0e1_K_tr4_i2
    x[351] = 0.77001 	# e0e1_K_tr5_i2
    x[352] = 0.78196 	# e0e1_K_tr6_i2
    x[353] = 0.79598 	# e0e1_K_tr7_i2
    x[354] = 0.81188 	# e0e1_K_tr8_i2
    x[355] = 0.82927 	# e0e1_K_tr9_i2
    x[356] = 0.84751 	# e0e1_K_tr10_i2
    x[357] = 0.86587 	# e0e1_K_tr11_i2
    x[358] = 0.88358 	# e0e1_K_tr12_i2
    x[359] = 0.90001 	# e0e1_K_tr13_i2
    x[360] = 0.9147 	# e0e1_K_tr14_i2
    x[361] = 0.9274 	# e0e1_K_tr15_i2
    x[362] = 0.93809 	# e0e1_K_tr16_i2
    x[363] = 0.94688 	# e0e1_K_tr17_i2
    x[364] = 0.95396 	# e0e1_K_tr18_i2
    x[365] = 0.96409 	# e0e1_K_tr19_i2
    x[366] = 0.97237 	# e0e1_K_tr20_i2
    x[367] = 58.12 	# e0e1_M_L_tr1
    x[368] = 58.12 	# e0e1_M_L_tr2
    x[369] = 58.12 	# e0e1_M_L_tr3
    x[370] = 58.12 	# e0e1_M_L_tr4
    x[371] = 58.12 	# e0e1_M_L_tr5
    x[372] = 58.12 	# e0e1_M_L_tr6
    x[373] = 58.12 	# e0e1_M_L_tr7
    x[374] = 58.12 	# e0e1_M_L_tr8
    x[375] = 58.12 	# e0e1_M_L_tr9
    x[376] = 58.12 	# e0e1_M_L_tr10
    x[377] = 58.12 	# e0e1_M_L_tr11
    x[378] = 58.12 	# e0e1_M_L_tr12
    x[379] = 58.12 	# e0e1_M_L_tr13
    x[380] = 58.12 	# e0e1_M_L_tr14
    x[381] = 58.12 	# e0e1_M_L_tr15
    x[382] = 58.12 	# e0e1_M_L_tr16
    x[383] = 58.12 	# e0e1_M_L_tr17
    x[384] = 58.12 	# e0e1_M_L_tr18
    x[385] = 58.12 	# e0e1_M_L_tr19
    x[386] = 58.12 	# e0e1_M_L_tr20
    x[387] = 58.12 	# e0e1_M_V_tr1
    x[388] = 58.12 	# e0e1_M_V_tr2
    x[389] = 58.12 	# e0e1_M_V_tr3
    x[390] = 58.12 	# e0e1_M_V_tr4
    x[391] = 58.12 	# e0e1_M_V_tr5
    x[392] = 58.12 	# e0e1_M_V_tr6
    x[393] = 58.12 	# e0e1_M_V_tr7
    x[394] = 58.12 	# e0e1_M_V_tr8
    x[395] = 58.12 	# e0e1_M_V_tr9
    x[396] = 58.12 	# e0e1_M_V_tr10
    x[397] = 58.12 	# e0e1_M_V_tr11
    x[398] = 58.12 	# e0e1_M_V_tr12
    x[399] = 58.12 	# e0e1_M_V_tr13
    x[400] = 58.12 	# e0e1_M_V_tr14
    x[401] = 58.12 	# e0e1_M_V_tr15
    x[402] = 58.12 	# e0e1_M_V_tr16
    x[403] = 58.12 	# e0e1_M_V_tr17
    x[404] = 58.12 	# e0e1_M_V_tr18
    x[405] = 58.12 	# e0e1_M_V_tr19
    x[406] = 58.12 	# e0e1_M_V_tr20
    x[407] = 183.0272 	# e0e1_P_i1_tr1
    x[408] = 182.979 	# e0e1_P_i1_tr2
    x[409] = 182.9173 	# e0e1_P_i1_tr3
    x[410] = 182.8394 	# e0e1_P_i1_tr4
    x[411] = 182.7431 	# e0e1_P_i1_tr5
    x[412] = 182.6268 	# e0e1_P_i1_tr6
    x[413] = 182.4903 	# e0e1_P_i1_tr7
    x[414] = 182.3351 	# e0e1_P_i1_tr8
    x[415] = 182.1654 	# e0e1_P_i1_tr9
    x[416] = 181.987 	# e0e1_P_i1_tr10
    x[417] = 181.8071 	# e0e1_P_i1_tr11
    x[418] = 181.633 	# e0e1_P_i1_tr12
    x[419] = 181.4712 	# e0e1_P_i1_tr13
    x[420] = 181.3262 	# e0e1_P_i1_tr14
    x[421] = 181.2004 	# e0e1_P_i1_tr15
    x[422] = 181.0943 	# e0e1_P_i1_tr16
    x[423] = 181.0068 	# e0e1_P_i1_tr17
    x[424] = 180.9361 	# e0e1_P_i1_tr18
    x[425] = 180.8351 	# e0e1_P_i1_tr19
    x[426] = 180.7522 	# e0e1_P_i1_tr20
    x[427] = 186.2431 	# e0e1_P_i2_tr1
    x[428] = 186.2152 	# e0e1_P_i2_tr2
    x[429] = 186.1794 	# e0e1_P_i2_tr3
    x[430] = 186.1343 	# e0e1_P_i2_tr4
    x[431] = 186.0785 	# e0e1_P_i2_tr5
    x[432] = 186.011 	# e0e1_P_i2_tr6
    x[433] = 185.9317 	# e0e1_P_i2_tr7
    x[434] = 185.8415 	# e0e1_P_i2_tr8
    x[435] = 185.7427 	# e0e1_P_i2_tr9
    x[436] = 185.6388 	# e0e1_P_i2_tr10
    x[437] = 185.5339 	# e0e1_P_i2_tr11
    x[438] = 185.4324 	# e0e1_P_i2_tr12
    x[439] = 185.338 	# e0e1_P_i2_tr13
    x[440] = 185.2533 	# e0e1_P_i2_tr14
    x[441] = 185.1798 	# e0e1_P_i2_tr15
    x[442] = 185.1177 	# e0e1_P_i2_tr16
    x[443] = 185.0666 	# e0e1_P_i2_tr17
    x[444] = 185.0253 	# e0e1_P_i2_tr18
    x[445] = 184.9662 	# e0e1_P_i2_tr19
    x[446] = 184.9177 	# e0e1_P_i2_tr20
    x[447] = 183.2442 	# e0e1_P_tr1
    x[448] = 183.2702 	# e0e1_P_tr2
    x[449] = 183.3038 	# e0e1_P_tr3
    x[450] = 183.3466 	# e0e1_P_tr4
    x[451] = 183.4001 	# e0e1_P_tr5
    x[452] = 183.4655 	# e0e1_P_tr6
    x[453] = 183.543 	# e0e1_P_tr7
    x[454] = 183.632 	# e0e1_P_tr8
    x[455] = 183.7305 	# e0e1_P_tr9
    x[456] = 183.8352 	# e0e1_P_tr10
    x[457] = 183.9418 	# e0e1_P_tr11
    x[458] = 184.0458 	# e0e1_P_tr12
    x[459] = 184.1433 	# e0e1_P_tr13
    x[460] = 184.2312 	# e0e1_P_tr14
    x[461] = 184.3079 	# e0e1_P_tr15
    x[462] = 184.3727 	# e0e1_P_tr16
    x[463] = 184.4263 	# e0e1_P_tr17
    x[464] = 184.4695 	# e0e1_P_tr18
    x[465] = 184.5319 	# e0e1_P_tr19
    x[466] = 184.5831 	# e0e1_P_tr20
    x[467] = 324.8359 	# e0e1_T_tr1
    x[468] = 325.0959 	# e0e1_T_tr2
    x[469] = 325.4267 	# e0e1_T_tr3
    x[470] = 325.8409 	# e0e1_T_tr4
    x[471] = 326.3484 	# e0e1_T_tr5
    x[472] = 326.9546 	# e0e1_T_tr6
    x[473] = 327.6569 	# e0e1_T_tr7
    x[474] = 328.4424 	# e0e1_T_tr8
    x[475] = 329.2877 	# e0e1_T_tr9
    x[476] = 330.1607 	# e0e1_T_tr10
    x[477] = 331.0251 	# e0e1_T_tr11
    x[478] = 331.8466 	# e0e1_T_tr12
    x[479] = 332.5978 	# e0e1_T_tr13
    x[480] = 333.261 	# e0e1_T_tr14
    x[481] = 333.8289 	# e0e1_T_tr15
    x[482] = 334.3026 	# e0e1_T_tr16
    x[483] = 334.6895 	# e0e1_T_tr17
    x[484] = 334.9999 	# e0e1_T_tr18
    x[485] = 335.4398 	# e0e1_T_tr19
    x[486] = 335.7975 	# e0e1_T_tr20
    x[487] = 335.8022 	# e0e1_T_B
    x[488] = -1.436032496517E8 	# e0e1_h_BLn
    x[489] = -1.434369314107E8 	# e0_h_LReboilern
    x[490] = -1.468488471178E8 	# e0_h_Fn
    x[491] = -1.509944142298E8 	# e0_h_DLn
    x[492] = -1.508472326026E8 	# e0e1_h_Ln_tr1
    x[493] = -1.50658230983E8 	# e0e1_h_Ln_tr2
    x[494] = -1.50418653232E8 	# e0e1_h_Ln_tr3
    x[495] = -1.501205943773E8 	# e0e1_h_Ln_tr4
    x[496] = -1.49758254217E8 	# e0e1_h_Ln_tr5
    x[497] = -1.49329869612E8 	# e0e1_h_Ln_tr6
    x[498] = -1.488396355607E8 	# e0e1_h_Ln_tr7
    x[499] = -1.482989081545E8 	# e0e1_h_Ln_tr8
    x[500] = -1.477259317082E8 	# e0e1_h_Ln_tr9
    x[501] = -1.471437256141E8 	# e0e1_h_Ln_tr10
    x[502] = -1.465765317174E8 	# e0e1_h_Ln_tr11
    x[503] = -1.460459266184E8 	# e0e1_h_Ln_tr12
    x[504] = -1.455678542455E8 	# e0e1_h_Ln_tr13
    x[505] = -1.451513342757E8 	# e0e1_h_Ln_tr14
    x[506] = -1.447988307008E8 	# e0e1_h_Ln_tr15
    x[507] = -1.445077064735E8 	# e0e1_h_Ln_tr16
    x[508] = -1.442720529022E8 	# e0e1_h_Ln_tr17
    x[509] = -1.440843599704E8 	# e0e1_h_Ln_tr18
    x[510] = -1.438196441435E8 	# e0e1_h_Ln_tr19
    x[511] = -1.436061189134E8 	# e0_h_LReboilerinn
    x[512] = -1.336448424678E8 	# e0_h_DVn
    x[513] = -1.334788140312E8 	# e0e1_h_Vn_tr2
    x[514] = -1.332653716633E8 	# e0e1_h_Vn_tr3
    x[515] = -1.329951071114E8 	# e0e1_h_Vn_tr4
    x[516] = -1.326593638477E8 	# e0e1_h_Vn_tr5
    x[517] = -1.322519724588E8 	# e0e1_h_Vn_tr6
    x[518] = -1.317714141569E8 	# e0e1_h_Vn_tr7
    x[519] = -1.312229219481E8 	# e0e1_h_Vn_tr8
    x[520] = -1.306197180437E8 	# e0e1_h_Vn_tr9
    x[521] = -1.299825608963E8 	# e0e1_h_Vn_tr10
    x[522] = -1.293372507535E8 	# e0e1_h_Vn_tr11
    x[523] = -1.287106077396E8 	# e0e1_h_Vn_tr12
    x[524] = -1.281261881192E8 	# e0e1_h_Vn_tr13
    x[525] = -1.276011137889E8 	# e0e1_h_Vn_tr14
    x[526] = -1.271447920824E8 	# e0e1_h_Vn_tr15
    x[527] = -1.26759446675E8 	# e0e1_h_Vn_tr16
    x[528] = -1.264417927793E8 	# e0e1_h_Vn_tr17
    x[529] = -1.261850720111E8 	# e0e1_h_Vn_tr18
    x[530] = -1.258170821967E8 	# e0e1_h_Vn_tr19
    x[531] = -1.255152736911E8 	# e0e1_h_Vn_tr20
    x[532] = -1.252723237137E8 	# e0_h_ReboilerVn
    x[533] = 700042.0 	# e0e1_p_tr1
    x[534] = 700084.0 	# e0e1_p_tr2
    x[535] = 700126.0 	# e0e1_p_tr3
    x[536] = 700168.0 	# e0e1_p_tr4
    x[537] = 700210.0 	# e0e1_p_tr5
    x[538] = 700252.0 	# e0e1_p_tr6
    x[539] = 700294.0 	# e0e1_p_tr7
    x[540] = 700336.0 	# e0e1_p_tr8
    x[541] = 700378.0 	# e0e1_p_tr9
    x[542] = 700420.0 	# e0e1_p_tr10
    x[543] = 700462.0 	# e0e1_p_tr11
    x[544] = 700504.0 	# e0e1_p_tr12
    x[545] = 700546.0 	# e0e1_p_tr13
    x[546] = 700588.0 	# e0e1_p_tr14
    x[547] = 700630.0 	# e0e1_p_tr15
    x[548] = 700672.0 	# e0e1_p_tr16
    x[549] = 700714.0 	# e0e1_p_tr17
    x[550] = 700756.0 	# e0e1_p_tr18
    x[551] = 700798.0 	# e0e1_p_tr19
    x[552] = 700840.0 	# e0_p_LReboiler
    x[553] = 700882.0 	# e0e1_p_tr21


    # Constant parameter setting:
    parameter[0] = 0.0 	# e0e1_r_n_tr1_i1
    parameter[1] = 0.0 	# e0e1_r_n_tr2_i1
    parameter[2] = 0.0 	# e0e1_r_n_tr3_i1
    parameter[3] = 0.0 	# e0e1_r_n_tr4_i1
    parameter[4] = 0.0 	# e0e1_r_n_tr5_i1
    parameter[5] = 0.0 	# e0e1_r_n_tr6_i1
    parameter[6] = 0.0 	# e0e1_r_n_tr7_i1
    parameter[7] = 0.0 	# e0e1_r_n_tr8_i1
    parameter[8] = 0.0 	# e0e1_r_n_tr9_i1
    parameter[9] = 0.0 	# e0e1_r_n_tr10_i1
    parameter[10] = 0.0 	# e0e1_r_n_tr11_i1
    parameter[11] = 0.0 	# e0e1_r_n_tr12_i1
    parameter[12] = 0.0 	# e0e1_r_n_tr13_i1
    parameter[13] = 0.0 	# e0e1_r_n_tr14_i1
    parameter[14] = 0.0 	# e0e1_r_n_tr15_i1
    parameter[15] = 0.0 	# e0e1_r_n_tr16_i1
    parameter[16] = 0.0 	# e0e1_r_n_tr17_i1
    parameter[17] = 0.0 	# e0e1_r_n_tr18_i1
    parameter[18] = 0.0 	# e0e1_r_n_tr19_i1
    parameter[19] = 0.0 	# e0e1_r_n_tr20_i1
    parameter[20] = 0.0 	# e0e1_r_n_tr1_i2
    parameter[21] = 0.0 	# e0e1_r_n_tr2_i2
    parameter[22] = 0.0 	# e0e1_r_n_tr3_i2
    parameter[23] = 0.0 	# e0e1_r_n_tr4_i2
    parameter[24] = 0.0 	# e0e1_r_n_tr5_i2
    parameter[25] = 0.0 	# e0e1_r_n_tr6_i2
    parameter[26] = 0.0 	# e0e1_r_n_tr7_i2
    parameter[27] = 0.0 	# e0e1_r_n_tr8_i2
    parameter[28] = 0.0 	# e0e1_r_n_tr9_i2
    parameter[29] = 0.0 	# e0e1_r_n_tr10_i2
    parameter[30] = 0.0 	# e0e1_r_n_tr11_i2
    parameter[31] = 0.0 	# e0e1_r_n_tr12_i2
    parameter[32] = 0.0 	# e0e1_r_n_tr13_i2
    parameter[33] = 0.0 	# e0e1_r_n_tr14_i2
    parameter[34] = 0.0 	# e0e1_r_n_tr15_i2
    parameter[35] = 0.0 	# e0e1_r_n_tr16_i2
    parameter[36] = 0.0 	# e0e1_r_n_tr17_i2
    parameter[37] = 0.0 	# e0e1_r_n_tr18_i2
    parameter[38] = 0.0 	# e0e1_r_n_tr19_i2
    parameter[39] = 0.0 	# e0e1_r_n_tr20_i2
    parameter[40] = 0.95 	# e0_x_D_i1
    parameter[41] = 0.08 	# e0e1_x_B_i1
    parameter[42] = 800000.0 	# e0_p_F
    parameter[43] = 42.0 	# e0_greek_Deltap_D
    parameter[44] = 0.0 	# e0_Q_DLoss
    parameter[45] = 0.0 	# e0_h_DELn
    parameter[46] = 42.0 	# e0_greek_Deltap_Reboiler
    parameter[47] = 0.0 	# e0_Q_LossReboiler
    parameter[48] = 0.0 	# e0_h_ELReboilern
    parameter[49] = 0.0 	# e0_h_EReboilerVn
    parameter[50] = 0.5 	# e0_x_F_i1
    parameter[51] = 333.15 	# e0_T_F
    parameter[52] = 3.14159265359 	# e0e1_greek_pi
    parameter[53] = 0.0 	# e0_h_EFn
    parameter[54] = 0.0 	# e0e1_F_Fn_tr1
    parameter[55] = 0.0 	# e0e1_F_Fn_tr2
    parameter[56] = 0.0 	# e0e1_F_Fn_tr3
    parameter[57] = 0.0 	# e0e1_F_Fn_tr4
    parameter[58] = 0.0 	# e0e1_F_Fn_tr5
    parameter[59] = 0.0 	# e0e1_F_Fn_tr6
    parameter[60] = 0.0 	# e0e1_F_Fn_tr7
    parameter[61] = 0.0 	# e0e1_F_Fn_tr8
    parameter[62] = 0.0 	# e0e1_F_Fn_tr9
    parameter[63] = 0.0 	# e0e1_F_Fn_tr10
    parameter[64] = 0.0 	# e0e1_F_Fn_tr11
    parameter[65] = 0.0 	# e0e1_F_Fn_tr12
    parameter[66] = 0.0 	# e0e1_F_Fn_tr13
    parameter[67] = 0.0 	# e0e1_F_Fn_tr14
    parameter[68] = 0.0 	# e0e1_F_Fn_tr15
    parameter[69] = 0.0 	# e0e1_F_Fn_tr16
    parameter[70] = 0.0 	# e0e1_F_Fn_tr17
    parameter[71] = 0.057 	# e0e1_F_Fn_tr18
    parameter[72] = 0.0 	# e0e1_F_Fn_tr19
    parameter[73] = 0.0 	# e0e1_F_Fn_tr20
    parameter[74] = 5.0E-4 	# e0_F_LReboilern
    parameter[75] = 58.12 	# e0e1_M_i1
    parameter[76] = 58.12 	# e0e1_M_i2
    parameter[77] = 42.0 	# e0e1_greek_Deltap_tr0
    parameter[78] = 42.0 	# e0e1_greek_Deltap_tr1
    parameter[79] = 42.0 	# e0e1_greek_Deltap_tr2
    parameter[80] = 42.0 	# e0e1_greek_Deltap_tr3
    parameter[81] = 42.0 	# e0e1_greek_Deltap_tr4
    parameter[82] = 42.0 	# e0e1_greek_Deltap_tr5
    parameter[83] = 0.0 	# e0e1_Q_Loss_tr1
    parameter[84] = 0.0 	# e0e1_Q_Loss_tr2
    parameter[85] = 0.0 	# e0e1_Q_Loss_tr3
    parameter[86] = 42.0 	# e0e1_greek_Deltap_tr6
    parameter[87] = 0.0 	# e0e1_Q_Loss_tr4
    parameter[88] = 0.0 	# e0e1_Q_Loss_tr5
    parameter[89] = 0.0 	# e0e1_Q_Loss_tr6
    parameter[90] = 0.0 	# e0e1_Q_Loss_tr7
    parameter[91] = 0.0 	# e0e1_Q_Loss_tr8
    parameter[92] = 0.0 	# e0e1_Q_Loss_tr9
    parameter[93] = 0.0 	# e0e1_Q_Loss_tr10
    parameter[94] = 0.0 	# e0e1_Q_Loss_tr11
    parameter[95] = 0.0 	# e0e1_Q_Loss_tr12
    parameter[96] = 0.0 	# e0e1_Q_Loss_tr13
    parameter[97] = 42.0 	# e0e1_greek_Deltap_tr7
    parameter[98] = 0.0 	# e0e1_Q_Loss_tr14
    parameter[99] = 0.0 	# e0e1_Q_Loss_tr15
    parameter[100] = 0.0 	# e0e1_Q_Loss_tr16
    parameter[101] = 0.0 	# e0e1_Q_Loss_tr17
    parameter[102] = 0.0 	# e0e1_Q_Loss_tr18
    parameter[103] = 0.0 	# e0e1_Q_Loss_tr19
    parameter[104] = 0.0 	# e0e1_Q_Loss_tr20
    parameter[105] = 8.314 	# e0e1_R
    parameter[106] = 42.0 	# e0e1_greek_Deltap_tr8
    parameter[107] = 42.0 	# e0e1_greek_Deltap_tr9
    parameter[108] = 1.28 	# e0e1_d
    parameter[109] = 42.0 	# e0e1_greek_Deltap_tr10
    parameter[110] = 0.0 	# e0e1_h_BELn
    parameter[111] = 0.0 	# e0e1_h_ELn_tr1
    parameter[112] = 0.0 	# e0e1_h_ELn_tr2
    parameter[113] = 42.0 	# e0e1_greek_Deltap_tr11
    parameter[114] = 0.0 	# e0e1_h_ELn_tr3
    parameter[115] = 0.0 	# e0e1_h_ELn_tr4
    parameter[116] = 0.0 	# e0e1_h_ELn_tr5
    parameter[117] = 0.0 	# e0e1_h_ELn_tr6
    parameter[118] = 0.0 	# e0e1_h_ELn_tr7
    parameter[119] = 0.0 	# e0e1_h_ELn_tr8
    parameter[120] = 0.0 	# e0e1_h_ELn_tr9
    parameter[121] = 0.0 	# e0e1_h_ELn_tr10
    parameter[122] = 0.0 	# e0e1_h_ELn_tr11
    parameter[123] = 0.0 	# e0e1_h_ELn_tr12
    parameter[124] = 42.0 	# e0e1_greek_Deltap_tr12
    parameter[125] = 0.0 	# e0e1_h_ELn_tr13
    parameter[126] = 0.0 	# e0e1_h_ELn_tr14
    parameter[127] = 0.0 	# e0e1_h_ELn_tr15
    parameter[128] = 0.0 	# e0e1_h_ELn_tr16
    parameter[129] = 0.0 	# e0e1_h_ELn_tr17
    parameter[130] = 0.0 	# e0e1_h_ELn_tr18
    parameter[131] = 0.0 	# e0e1_h_ELn_tr19
    parameter[132] = 0.0 	# e0e1_h_ELn_tr20
    parameter[133] = 0.0 	# e0e1_h_EVn_tr1
    parameter[134] = 0.0 	# e0e1_h_EVn_tr2
    parameter[135] = 42.0 	# e0e1_greek_Deltap_tr13
    parameter[136] = 0.0 	# e0e1_h_EVn_tr3
    parameter[137] = 0.0 	# e0e1_h_EVn_tr4
    parameter[138] = 0.0 	# e0e1_h_EVn_tr5
    parameter[139] = 0.0 	# e0e1_h_EVn_tr6
    parameter[140] = 0.0 	# e0e1_h_EVn_tr7
    parameter[141] = 0.0 	# e0e1_h_EVn_tr8
    parameter[142] = 0.0 	# e0e1_h_EVn_tr9
    parameter[143] = 0.0 	# e0e1_h_EVn_tr10
    parameter[144] = 0.0 	# e0e1_h_EVn_tr11
    parameter[145] = 0.0 	# e0e1_h_EVn_tr12
    parameter[146] = 42.0 	# e0e1_greek_Deltap_tr14
    parameter[147] = 0.0 	# e0e1_h_EVn_tr13
    parameter[148] = 0.0 	# e0e1_h_EVn_tr14
    parameter[149] = 0.0 	# e0e1_h_EVn_tr15
    parameter[150] = 0.0 	# e0e1_h_EVn_tr16
    parameter[151] = 0.0 	# e0e1_h_EVn_tr17
    parameter[152] = 0.0 	# e0e1_h_EVn_tr18
    parameter[153] = 0.0 	# e0e1_h_EVn_tr19
    parameter[154] = 0.0 	# e0e1_h_EVn_tr20
    parameter[155] = 42.0 	# e0e1_greek_Deltap_tr15
    parameter[156] = 42.0 	# e0e1_greek_Deltap_tr16
    parameter[157] = 42.0 	# e0e1_greek_Deltap_tr17
    parameter[158] = 42.0 	# e0e1_greek_Deltap_tr18
    parameter[159] = 42.0 	# e0e1_greek_Deltap_tr19
    parameter[160] = 42.0 	# e0e1_greek_Deltap_tr20
    parameter[161] = 700000.0 	# e0_p_DV
    parameter[162] = 108.43 	# e0_A_LVpd101_i1
    parameter[163] = 66.343 	# e0_A_LVpd101_i2
    parameter[164] = 1.0631 	# e0_A_Lnrhod105_i1
    parameter[165] = 1.0677 	# e0_A_Lnrhod105_i2
    parameter[166] = 3.9654E7 	# e0_A_LVhnd106_i1
    parameter[167] = 3.6238E7 	# e0_A_LVhnd106_i2
    parameter[168] = 33257.8886 	# e0_A_Vhnd127_i1
    parameter[169] = 33257.8886 	# e0_A_Vhnd127_i2
    parameter[170] = -5039.9 	# e0_B_LVpd101_i1
    parameter[171] = -4363.2 	# e0_B_LVpd101_i2
    parameter[172] = 0.27506 	# e0_B_Lnrhod105_i1
    parameter[173] = 0.27188 	# e0_B_Lnrhod105_i2
    parameter[174] = 1.274 	# e0_B_LVhnd106_i1
    parameter[175] = 0.8337 	# e0_B_LVhnd106_i2
    parameter[176] = 48701.33848 	# e0_B_Vhnd127_i1
    parameter[177] = 47100.60578 	# e0_B_Vhnd127_i2
    parameter[178] = -15.012 	# e0_C_LVpd101_i1
    parameter[179] = -7.046 	# e0_C_LVpd101_i2
    parameter[180] = 407.8 	# e0_C_Lnrhod105_i1
    parameter[181] = 425.12 	# e0_C_Lnrhod105_i2
    parameter[182] = -1.4255 	# e0_C_LVhnd106_i1
    parameter[183] = -0.82274 	# e0_C_LVhnd106_i2
    parameter[184] = -411.25771 	# e0_C_Vhnd127_i1
    parameter[185] = -306.72739 	# e0_C_Vhnd127_i2
    parameter[186] = 0.022725 	# e0_D_LVpd101_i1
    parameter[187] = 9.4509E-6 	# e0_D_LVpd101_i2
    parameter[188] = 0.2758 	# e0_D_Lnrhod105_i1
    parameter[189] = 0.28688 	# e0_D_Lnrhod105_i2
    parameter[190] = 0.60708 	# e0_D_LVhnd106_i1
    parameter[191] = 0.39613 	# e0_D_LVhnd106_i2
    parameter[192] = 141642.72064 	# e0_D_Vhnd127_i1
    parameter[193] = 98932.44264 	# e0_D_Vhnd127_i2
    parameter[194] = 1.0 	# e0_E_LVpd101_i1
    parameter[195] = 2.0 	# e0_E_LVpd101_i2
    parameter[196] = 0.0 	# e0_E_LVhnd106_i1
    parameter[197] = 0.0 	# e0_E_LVhnd106_i2
    parameter[198] = 1544.73381 	# e0_E_Vhnd127_i1
    parameter[199] = -3782.20749 	# e0_E_Vhnd127_i2
    parameter[200] = 93408.54987 	# e0_F_Vhnd127_i1
    parameter[201] = 137464.89025 	# e0_F_Vhnd127_i2
    parameter[202] = 3850.61681 	# e0_G_Vhnd127_i1
    parameter[203] = 1521.86945 	# e0_G_Vhnd127_i2
    parameter[204] = 407.8 	# e0_T_crit_i1
    parameter[205] = 298.15 	# e0_T_ho_i1
    parameter[206] = 425.12 	# e0_T_crit_i2
    parameter[207] = 298.15 	# e0_T_ho_i2
    parameter[208] = -1.3499E8 	# e0_h_o_i1
    parameter[209] = -1.2579E8 	# e0_h_o_i2
    parameter[210] = -13.912 	# e0e1_A_Letad101_i1
    parameter[211] = 108.43 	# e0e1_A_LVpd101_i1
    parameter[212] = -7.2471 	# e0e1_A_Letad101_i2
    parameter[213] = 66.343 	# e0e1_A_LVpd101_i2
    parameter[214] = 1.0871E-7 	# e0e1_A_Vetad102_i1
    parameter[215] = 3.4387E-8 	# e0e1_A_Vetad102_i2
    parameter[216] = 1.0631 	# e0e1_A_Lnrhod105_i1
    parameter[217] = 1.0677 	# e0e1_A_Lnrhod105_i2
    parameter[218] = 3.9654E7 	# e0e1_A_LVhnd106_i1
    parameter[219] = 0.051359 	# e0e1_A_sigd106_i1
    parameter[220] = 3.6238E7 	# e0e1_A_LVhnd106_i2
    parameter[221] = 0.05196 	# e0e1_A_sigd106_i2
    parameter[222] = 0.17766 	# e0e1_A_Wsigd106
    parameter[223] = 33257.8886 	# e0e1_A_Vhnd127_i1
    parameter[224] = 33257.8886 	# e0e1_A_Vhnd127_i2
    parameter[225] = 797.09 	# e0e1_B_Letad101_i1
    parameter[226] = -5039.9 	# e0e1_B_LVpd101_i1
    parameter[227] = 534.82 	# e0e1_B_Letad101_i2
    parameter[228] = -4363.2 	# e0e1_B_LVpd101_i2
    parameter[229] = 0.78135 	# e0e1_B_Vetad102_i1
    parameter[230] = 0.94604 	# e0e1_B_Vetad102_i2
    parameter[231] = 0.27506 	# e0e1_B_Lnrhod105_i1
    parameter[232] = 0.27188 	# e0e1_B_Lnrhod105_i2
    parameter[233] = 1.274 	# e0e1_B_LVhnd106_i1
    parameter[234] = 1.2532 	# e0e1_B_sigd106_i1
    parameter[235] = 0.8337 	# e0e1_B_LVhnd106_i2
    parameter[236] = 1.2181 	# e0e1_B_sigd106_i2
    parameter[237] = 2.567 	# e0e1_B_Wsigd106
    parameter[238] = 48701.33848 	# e0e1_B_Vhnd127_i1
    parameter[239] = 47100.60578 	# e0e1_B_Vhnd127_i2
    parameter[240] = 0.45308 	# e0e1_C_Letad101_i1
    parameter[241] = -15.012 	# e0e1_C_LVpd101_i1
    parameter[242] = -0.57469 	# e0e1_C_Letad101_i2
    parameter[243] = -7.046 	# e0e1_C_LVpd101_i2
    parameter[244] = 70.639 	# e0e1_C_Vetad102_i1
    parameter[245] = 0.0 	# e0e1_C_Vetad102_i2
    parameter[246] = 407.8 	# e0e1_C_Lnrhod105_i1
    parameter[247] = 425.12 	# e0e1_C_Lnrhod105_i2
    parameter[248] = -1.4255 	# e0e1_C_LVhnd106_i1
    parameter[249] = 0.0 	# e0e1_C_sigd106_i1
    parameter[250] = -0.82274 	# e0e1_C_LVhnd106_i2
    parameter[251] = 0.0 	# e0e1_C_sigd106_i2
    parameter[252] = -3.3377 	# e0e1_C_Wsigd106
    parameter[253] = -411.25771 	# e0e1_C_Vhnd127_i1
    parameter[254] = -306.72739 	# e0e1_C_Vhnd127_i2
    parameter[255] = 0.0 	# e0e1_D_Letad101_i1
    parameter[256] = 0.022725 	# e0e1_D_LVpd101_i1
    parameter[257] = -4.6625E-27 	# e0e1_D_Letad101_i2
    parameter[258] = 9.4509E-6 	# e0e1_D_LVpd101_i2
    parameter[259] = 0.0 	# e0e1_D_Vetad102_i1
    parameter[260] = 0.0 	# e0e1_D_Vetad102_i2
    parameter[261] = 0.2758 	# e0e1_D_Lnrhod105_i1
    parameter[262] = 0.28688 	# e0e1_D_Lnrhod105_i2
    parameter[263] = 0.60708 	# e0e1_D_LVhnd106_i1
    parameter[264] = 0.0 	# e0e1_D_sigd106_i1
    parameter[265] = 0.39613 	# e0e1_D_LVhnd106_i2
    parameter[266] = 0.0 	# e0e1_D_sigd106_i2
    parameter[267] = 1.9699 	# e0e1_D_Wsigd106
    parameter[268] = 141642.72064 	# e0e1_D_Vhnd127_i1
    parameter[269] = 98932.44264 	# e0e1_D_Vhnd127_i2
    parameter[270] = 0.0 	# e0e1_E_Letad101_i1
    parameter[271] = 1.0 	# e0e1_E_LVpd101_i1
    parameter[272] = 10.0 	# e0e1_E_Letad101_i2
    parameter[273] = 2.0 	# e0e1_E_LVpd101_i2
    parameter[274] = 0.0 	# e0e1_E_LVhnd106_i1
    parameter[275] = 0.0 	# e0e1_E_sigd106_i1
    parameter[276] = 0.0 	# e0e1_E_LVhnd106_i2
    parameter[277] = 0.0 	# e0e1_E_sigd106_i2
    parameter[278] = 0.0 	# e0e1_E_Wsigd106
    parameter[279] = 1544.73381 	# e0e1_E_Vhnd127_i1
    parameter[280] = -3782.20749 	# e0e1_E_Vhnd127_i2
    parameter[281] = 93408.54987 	# e0e1_F_Vhnd127_i1
    parameter[282] = 137464.89025 	# e0e1_F_Vhnd127_i2
    parameter[283] = 3850.61681 	# e0e1_G_Vhnd127_i1
    parameter[284] = 1521.86945 	# e0e1_G_Vhnd127_i2
    parameter[285] = 647.096 	# e0e1_T_Wcrit
    parameter[286] = 407.8 	# e0e1_T_crit_i1
    parameter[287] = 298.15 	# e0e1_T_ho_i1
    parameter[288] = 425.12 	# e0e1_T_crit_i2
    parameter[289] = 298.15 	# e0e1_T_ho_i2
    parameter[290] = -1.3499E8 	# e0e1_h_o_i1
    parameter[291] = -1.2579E8 	# e0e1_h_o_i2


    # Get symbolic quantities:
    xSymbolic = sympy.symbols('e0e1_x_tr1_i1 e0e1_x_tr2_i1 e0e1_x_tr3_i1 e0e1_x_tr4_i1 e0e1_x_tr5_i1 e0e1_x_tr6_i1 e0e1_x_tr7_i1 e0e1_x_tr8_i1 e0e1_x_tr9_i1 e0e1_x_tr10_i1 e0e1_x_tr11_i1 e0e1_x_tr12_i1 e0e1_x_tr13_i1 e0e1_x_tr14_i1 e0e1_x_tr15_i1 e0e1_x_tr16_i1 e0e1_x_tr17_i1 e0e1_x_tr18_i1 e0e1_x_tr19_i1 e0_x_Reboilerin_i1 e0_x_D_i2 e0e1_x_tr1_i2 e0e1_x_tr2_i2 e0e1_x_tr3_i2 e0e1_x_tr4_i2 e0e1_x_tr5_i2 e0e1_x_tr6_i2 e0e1_x_tr7_i2 e0e1_x_tr8_i2 e0e1_x_tr9_i2 e0e1_x_tr10_i2 e0e1_x_tr11_i2 e0e1_x_tr12_i2 e0e1_x_tr13_i2 e0e1_x_tr14_i2 e0e1_x_tr15_i2 e0e1_x_tr16_i2 e0e1_x_tr17_i2 e0e1_x_tr18_i2 e0e1_x_tr19_i2 e0_x_Reboilerin_i2 e0e1_x_B_i2 e0_x_Reboiler_i1 e0_x_Reboiler_i2 e0_y_D_i1 e0e1_y_tr2_i1 e0e1_y_tr3_i1 e0e1_y_tr4_i1 e0e1_y_tr5_i1 e0e1_y_tr6_i1 e0e1_y_tr7_i1 e0e1_y_tr8_i1 e0e1_y_tr9_i1 e0e1_y_tr10_i1 e0e1_y_tr11_i1 e0e1_y_tr12_i1 e0e1_y_tr13_i1 e0e1_y_tr14_i1 e0e1_y_tr15_i1 e0e1_y_tr16_i1 e0e1_y_tr17_i1 e0e1_y_tr18_i1 e0e1_y_tr19_i1 e0e1_y_tr20_i1 e0_y_Reboiler_i1 e0_y_D_i2 e0e1_y_tr2_i2 e0e1_y_tr3_i2 e0e1_y_tr4_i2 e0e1_y_tr5_i2 e0e1_y_tr6_i2 e0e1_y_tr7_i2 e0e1_y_tr8_i2 e0e1_y_tr9_i2 e0e1_y_tr10_i2 e0e1_y_tr11_i2 e0e1_y_tr12_i2 e0e1_y_tr13_i2 e0e1_y_tr14_i2 e0e1_y_tr15_i2 e0e1_y_tr16_i2 e0e1_y_tr17_i2 e0e1_y_tr18_i2 e0e1_y_tr19_i2 e0e1_y_tr20_i2 e0_y_Reboiler_i2 e0_F_DLn e0_K_D_i1 e0_K_D_i2 e0_Q_D e0_R_Reflux e0_T_D e0_p_DL e0_y_DV_i1 e0_y_DV_i2 e0e1_greek_eta_L_tr1 e0_K_Reboiler_i1 e0_K_Reboiler_i2 e0_Q_Reboiler e0_T_Reboiler e0e1_greek_eta_L_tr2 e0_p_ReboilerV e0e1_greek_eta_L_tr3 e0e1_greek_eta_L_tr4 e0e1_greek_eta_L_tr5 e0e1_greek_eta_L_tr6 e0e1_greek_eta_L_tr7 e0e1_greek_eta_L_tr8 e0e1_greek_eta_L_tr9 e0e1_greek_eta_L_tr10 e0e1_greek_eta_L_tr11 e0e1_greek_eta_L_tr12 e0e1_greek_eta_L_tr13 e0e1_greek_eta_L_tr14 e0e1_greek_eta_L_tr15 e0e1_greek_eta_L_tr16 e0e1_greek_eta_L_tr17 e0e1_greek_eta_L_tr18 e0e1_greek_eta_L_tr19 e0e1_greek_eta_L_tr20 e0_x_F_i2 e0e1_greek_eta_V_tr1 e0e1_greek_eta_V_tr2 e0e1_greek_eta_V_tr3 e0e1_greek_eta_V_tr4 e0e1_greek_eta_V_tr5 e0e1_greek_eta_V_tr6 e0e1_greek_eta_V_tr7 e0e1_greek_eta_V_tr8 e0e1_greek_eta_V_tr9 e0e1_greek_eta_V_tr10 e0e1_greek_eta_V_tr11 e0e1_greek_eta_V_tr12 e0e1_greek_eta_V_tr13 e0e1_greek_eta_V_tr14 e0e1_greek_eta_V_tr15 e0e1_greek_eta_V_tr16 e0e1_greek_eta_V_tr17 e0e1_greek_eta_V_tr18 e0e1_greek_eta_V_tr19 e0e1_greek_eta_V_tr20 e0e1_greek_rho_Lm_tr1 e0e1_greek_rho_Lm_tr2 e0e1_greek_rho_Lm_tr3 e0e1_greek_rho_Lm_tr4 e0e1_greek_rho_Lm_tr5 e0e1_greek_rho_Lm_tr6 e0e1_greek_rho_Lm_tr7 e0e1_greek_rho_Lm_tr8 e0e1_greek_rho_Lm_tr9 e0e1_greek_rho_Lm_tr10 e0e1_greek_rho_Lm_tr11 e0e1_greek_rho_Lm_tr12 e0e1_greek_rho_Lm_tr13 e0e1_greek_rho_Lm_tr14 e0e1_greek_rho_Lm_tr15 e0e1_greek_rho_Lm_tr16 e0e1_greek_rho_Lm_tr17 e0e1_greek_rho_Lm_tr18 e0e1_greek_rho_Lm_tr19 e0e1_greek_rho_Lm_tr20 e0e1_greek_rho_Vm_tr1 e0e1_greek_rho_Vm_tr2 e0e1_greek_rho_Vm_tr3 e0e1_greek_rho_Vm_tr4 e0e1_greek_rho_Vm_tr5 e0e1_greek_rho_Vm_tr6 e0e1_greek_rho_Vm_tr7 e0e1_greek_rho_Vm_tr8 e0e1_greek_rho_Vm_tr9 e0e1_greek_rho_Vm_tr10 e0e1_greek_rho_Vm_tr11 e0e1_greek_rho_Vm_tr12 e0e1_greek_rho_Vm_tr13 e0e1_greek_rho_Vm_tr14 e0e1_greek_rho_Vm_tr15 e0e1_greek_rho_Vm_tr16 e0e1_greek_rho_Vm_tr17 e0e1_greek_rho_Vm_tr18 e0e1_greek_rho_Vm_tr19 e0e1_greek_rho_Vm_tr20 e0e1_greek_sigma_tr1 e0e1_greek_sigma_tr2 e0e1_greek_sigma_tr3 e0e1_greek_sigma_tr4 e0e1_greek_sigma_tr5 e0e1_greek_sigma_tr6 e0e1_greek_sigma_tr7 e0e1_greek_sigma_tr8 e0e1_greek_sigma_tr9 e0e1_greek_sigma_tr10 e0e1_greek_sigma_tr11 e0e1_greek_sigma_tr12 e0e1_greek_sigma_tr13 e0e1_greek_sigma_tr14 e0e1_greek_sigma_tr15 e0e1_greek_sigma_tr16 e0e1_greek_sigma_tr17 e0e1_greek_sigma_tr18 e0e1_greek_sigma_tr19 e0e1_greek_sigma_tr20 e0e1_greek_theta_i1_j1_tr1 e0e1_greek_theta_i1_j1_tr2 e0e1_greek_theta_i1_j1_tr3 e0e1_greek_theta_i1_j1_tr4 e0e1_greek_theta_i1_j1_tr5 e0e1_greek_theta_i1_j1_tr6 e0e1_greek_theta_i1_j1_tr7 e0e1_greek_theta_i1_j1_tr8 e0e1_greek_theta_i1_j1_tr9 e0e1_greek_theta_i1_j1_tr10 e0e1_greek_theta_i1_j1_tr11 e0e1_greek_theta_i1_j1_tr12 e0e1_greek_theta_i1_j1_tr13 e0e1_greek_theta_i1_j1_tr14 e0e1_greek_theta_i1_j1_tr15 e0e1_greek_theta_i1_j1_tr16 e0e1_greek_theta_i1_j1_tr17 e0e1_greek_theta_i1_j1_tr18 e0e1_greek_theta_i1_j1_tr19 e0e1_greek_theta_i1_j1_tr20 e0e1_greek_theta_i1_j2_tr1 e0e1_greek_theta_i1_j2_tr2 e0e1_greek_theta_i1_j2_tr3 e0e1_greek_theta_i1_j2_tr4 e0e1_greek_theta_i1_j2_tr5 e0e1_greek_theta_i1_j2_tr6 e0e1_greek_theta_i1_j2_tr7 e0e1_greek_theta_i1_j2_tr8 e0e1_greek_theta_i1_j2_tr9 e0e1_greek_theta_i1_j2_tr10 e0e1_greek_theta_i1_j2_tr11 e0e1_greek_theta_i1_j2_tr12 e0e1_greek_theta_i1_j2_tr13 e0e1_greek_theta_i1_j2_tr14 e0e1_greek_theta_i1_j2_tr15 e0e1_greek_theta_i1_j2_tr16 e0e1_greek_theta_i1_j2_tr17 e0e1_greek_theta_i1_j2_tr18 e0e1_greek_theta_i1_j2_tr19 e0e1_greek_theta_i1_j2_tr20 e0e1_greek_theta_i2_j1_tr1 e0e1_greek_theta_i2_j1_tr2 e0e1_greek_theta_i2_j1_tr3 e0e1_greek_theta_i2_j1_tr4 e0e1_greek_theta_i2_j1_tr5 e0e1_greek_theta_i2_j1_tr6 e0e1_greek_theta_i2_j1_tr7 e0e1_greek_theta_i2_j1_tr8 e0e1_greek_theta_i2_j1_tr9 e0e1_greek_theta_i2_j1_tr10 e0e1_greek_theta_i2_j1_tr11 e0e1_greek_theta_i2_j1_tr12 e0e1_greek_theta_i2_j1_tr13 e0e1_greek_theta_i2_j1_tr14 e0e1_greek_theta_i2_j1_tr15 e0e1_greek_theta_i2_j1_tr16 e0e1_greek_theta_i2_j1_tr17 e0e1_greek_theta_i2_j1_tr18 e0e1_greek_theta_i2_j1_tr19 e0e1_greek_theta_i2_j1_tr20 e0e1_greek_theta_i2_j2_tr1 e0e1_greek_theta_i2_j2_tr2 e0e1_greek_theta_i2_j2_tr3 e0e1_greek_theta_i2_j2_tr4 e0e1_greek_theta_i2_j2_tr5 e0e1_greek_theta_i2_j2_tr6 e0e1_greek_theta_i2_j2_tr7 e0e1_greek_theta_i2_j2_tr8 e0e1_greek_theta_i2_j2_tr9 e0e1_greek_theta_i2_j2_tr10 e0e1_greek_theta_i2_j2_tr11 e0e1_greek_theta_i2_j2_tr12 e0e1_greek_theta_i2_j2_tr13 e0e1_greek_theta_i2_j2_tr14 e0e1_greek_theta_i2_j2_tr15 e0e1_greek_theta_i2_j2_tr16 e0e1_greek_theta_i2_j2_tr17 e0e1_greek_theta_i2_j2_tr18 e0e1_greek_theta_i2_j2_tr19 e0e1_greek_theta_i2_j2_tr20 e0e1_A e0e1_F_BLn e0_F_LReboilerinn e0_F_DLRefluxn e0e1_F_Ln_tr1 e0e1_F_Ln_tr2 e0e1_F_Ln_tr3 e0e1_F_Ln_tr4 e0e1_F_Ln_tr5 e0e1_F_Ln_tr6 e0e1_F_Ln_tr7 e0e1_F_Ln_tr8 e0e1_F_Ln_tr9 e0e1_F_Ln_tr10 e0e1_F_Ln_tr11 e0e1_F_Ln_tr12 e0e1_F_Ln_tr13 e0e1_F_Ln_tr14 e0e1_F_Ln_tr15 e0e1_F_Ln_tr16 e0e1_F_Ln_tr17 e0e1_F_Ln_tr18 e0e1_F_Ln_tr19 e0e1_F_Ln_tr20 e0e1_F_Ln_tr21 e0_F_DVn e0e1_F_Vn_tr2 e0e1_F_Vn_tr3 e0e1_F_Vn_tr4 e0e1_F_Vn_tr5 e0e1_F_Vn_tr6 e0e1_F_Vn_tr7 e0e1_F_Vn_tr8 e0e1_F_Vn_tr9 e0e1_F_Vn_tr10 e0e1_F_Vn_tr11 e0e1_F_Vn_tr12 e0e1_F_Vn_tr13 e0e1_F_Vn_tr14 e0e1_F_Vn_tr15 e0e1_F_Vn_tr16 e0e1_F_Vn_tr17 e0e1_F_Vn_tr18 e0e1_F_Vn_tr19 e0e1_F_Vn_tr20 e0_F_ReboilerVn e0e1_K_tr1_i1 e0e1_K_tr2_i1 e0e1_K_tr3_i1 e0e1_K_tr4_i1 e0e1_K_tr5_i1 e0e1_K_tr6_i1 e0e1_K_tr7_i1 e0e1_K_tr8_i1 e0e1_K_tr9_i1 e0e1_K_tr10_i1 e0e1_K_tr11_i1 e0e1_K_tr12_i1 e0e1_K_tr13_i1 e0e1_K_tr14_i1 e0e1_K_tr15_i1 e0e1_K_tr16_i1 e0e1_K_tr17_i1 e0e1_K_tr18_i1 e0e1_K_tr19_i1 e0e1_K_tr20_i1 e0e1_K_tr1_i2 e0e1_K_tr2_i2 e0e1_K_tr3_i2 e0e1_K_tr4_i2 e0e1_K_tr5_i2 e0e1_K_tr6_i2 e0e1_K_tr7_i2 e0e1_K_tr8_i2 e0e1_K_tr9_i2 e0e1_K_tr10_i2 e0e1_K_tr11_i2 e0e1_K_tr12_i2 e0e1_K_tr13_i2 e0e1_K_tr14_i2 e0e1_K_tr15_i2 e0e1_K_tr16_i2 e0e1_K_tr17_i2 e0e1_K_tr18_i2 e0e1_K_tr19_i2 e0e1_K_tr20_i2 e0e1_M_L_tr1 e0e1_M_L_tr2 e0e1_M_L_tr3 e0e1_M_L_tr4 e0e1_M_L_tr5 e0e1_M_L_tr6 e0e1_M_L_tr7 e0e1_M_L_tr8 e0e1_M_L_tr9 e0e1_M_L_tr10 e0e1_M_L_tr11 e0e1_M_L_tr12 e0e1_M_L_tr13 e0e1_M_L_tr14 e0e1_M_L_tr15 e0e1_M_L_tr16 e0e1_M_L_tr17 e0e1_M_L_tr18 e0e1_M_L_tr19 e0e1_M_L_tr20 e0e1_M_V_tr1 e0e1_M_V_tr2 e0e1_M_V_tr3 e0e1_M_V_tr4 e0e1_M_V_tr5 e0e1_M_V_tr6 e0e1_M_V_tr7 e0e1_M_V_tr8 e0e1_M_V_tr9 e0e1_M_V_tr10 e0e1_M_V_tr11 e0e1_M_V_tr12 e0e1_M_V_tr13 e0e1_M_V_tr14 e0e1_M_V_tr15 e0e1_M_V_tr16 e0e1_M_V_tr17 e0e1_M_V_tr18 e0e1_M_V_tr19 e0e1_M_V_tr20 e0e1_P_i1_tr1 e0e1_P_i1_tr2 e0e1_P_i1_tr3 e0e1_P_i1_tr4 e0e1_P_i1_tr5 e0e1_P_i1_tr6 e0e1_P_i1_tr7 e0e1_P_i1_tr8 e0e1_P_i1_tr9 e0e1_P_i1_tr10 e0e1_P_i1_tr11 e0e1_P_i1_tr12 e0e1_P_i1_tr13 e0e1_P_i1_tr14 e0e1_P_i1_tr15 e0e1_P_i1_tr16 e0e1_P_i1_tr17 e0e1_P_i1_tr18 e0e1_P_i1_tr19 e0e1_P_i1_tr20 e0e1_P_i2_tr1 e0e1_P_i2_tr2 e0e1_P_i2_tr3 e0e1_P_i2_tr4 e0e1_P_i2_tr5 e0e1_P_i2_tr6 e0e1_P_i2_tr7 e0e1_P_i2_tr8 e0e1_P_i2_tr9 e0e1_P_i2_tr10 e0e1_P_i2_tr11 e0e1_P_i2_tr12 e0e1_P_i2_tr13 e0e1_P_i2_tr14 e0e1_P_i2_tr15 e0e1_P_i2_tr16 e0e1_P_i2_tr17 e0e1_P_i2_tr18 e0e1_P_i2_tr19 e0e1_P_i2_tr20 e0e1_P_tr1 e0e1_P_tr2 e0e1_P_tr3 e0e1_P_tr4 e0e1_P_tr5 e0e1_P_tr6 e0e1_P_tr7 e0e1_P_tr8 e0e1_P_tr9 e0e1_P_tr10 e0e1_P_tr11 e0e1_P_tr12 e0e1_P_tr13 e0e1_P_tr14 e0e1_P_tr15 e0e1_P_tr16 e0e1_P_tr17 e0e1_P_tr18 e0e1_P_tr19 e0e1_P_tr20 e0e1_T_tr1 e0e1_T_tr2 e0e1_T_tr3 e0e1_T_tr4 e0e1_T_tr5 e0e1_T_tr6 e0e1_T_tr7 e0e1_T_tr8 e0e1_T_tr9 e0e1_T_tr10 e0e1_T_tr11 e0e1_T_tr12 e0e1_T_tr13 e0e1_T_tr14 e0e1_T_tr15 e0e1_T_tr16 e0e1_T_tr17 e0e1_T_tr18 e0e1_T_tr19 e0e1_T_tr20 e0e1_T_B e0e1_h_BLn e0_h_LReboilern e0_h_Fn e0_h_DLn e0e1_h_Ln_tr1 e0e1_h_Ln_tr2 e0e1_h_Ln_tr3 e0e1_h_Ln_tr4 e0e1_h_Ln_tr5 e0e1_h_Ln_tr6 e0e1_h_Ln_tr7 e0e1_h_Ln_tr8 e0e1_h_Ln_tr9 e0e1_h_Ln_tr10 e0e1_h_Ln_tr11 e0e1_h_Ln_tr12 e0e1_h_Ln_tr13 e0e1_h_Ln_tr14 e0e1_h_Ln_tr15 e0e1_h_Ln_tr16 e0e1_h_Ln_tr17 e0e1_h_Ln_tr18 e0e1_h_Ln_tr19 e0_h_LReboilerinn e0_h_DVn e0e1_h_Vn_tr2 e0e1_h_Vn_tr3 e0e1_h_Vn_tr4 e0e1_h_Vn_tr5 e0e1_h_Vn_tr6 e0e1_h_Vn_tr7 e0e1_h_Vn_tr8 e0e1_h_Vn_tr9 e0e1_h_Vn_tr10 e0e1_h_Vn_tr11 e0e1_h_Vn_tr12 e0e1_h_Vn_tr13 e0e1_h_Vn_tr14 e0e1_h_Vn_tr15 e0e1_h_Vn_tr16 e0e1_h_Vn_tr17 e0e1_h_Vn_tr18 e0e1_h_Vn_tr19 e0e1_h_Vn_tr20 e0_h_ReboilerVn e0e1_p_tr1 e0e1_p_tr2 e0e1_p_tr3 e0e1_p_tr4 e0e1_p_tr5 e0e1_p_tr6 e0e1_p_tr7 e0e1_p_tr8 e0e1_p_tr9 e0e1_p_tr10 e0e1_p_tr11 e0e1_p_tr12 e0e1_p_tr13 e0e1_p_tr14 e0e1_p_tr15 e0e1_p_tr16 e0e1_p_tr17 e0e1_p_tr18 e0e1_p_tr19 e0_p_LReboiler e0e1_p_tr21 ')
    fSymbolic = getSymbolicFunctions(xSymbolic, parameter)

    # Intial variable bounds:
    xInitial[0] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr1_i1
    xInitial[1] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr2_i1
    xInitial[2] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr3_i1
    xInitial[3] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr4_i1
    xInitial[4] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr5_i1
    xInitial[5] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr6_i1
    xInitial[6] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr7_i1
    xInitial[7] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr8_i1
    xInitial[8] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr9_i1
    xInitial[9] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr10_i1
    xInitial[10] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr11_i1
    xInitial[11] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr12_i1
    xInitial[12] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr13_i1
    xInitial[13] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr14_i1
    xInitial[14] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr15_i1
    xInitial[15] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr16_i1
    xInitial[16] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr17_i1
    xInitial[17] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr18_i1
    xInitial[18] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr19_i1
    xInitial[19] = mpmath.mpi(0.0, 1.0)  	# e0_x_Reboilerin_i1
    xInitial[20] = mpmath.mpi(0.0, 1.0)  	# e0_x_D_i2
    xInitial[21] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr1_i2
    xInitial[22] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr2_i2
    xInitial[23] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr3_i2
    xInitial[24] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr4_i2
    xInitial[25] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr5_i2
    xInitial[26] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr6_i2
    xInitial[27] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr7_i2
    xInitial[28] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr8_i2
    xInitial[29] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr9_i2
    xInitial[30] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr10_i2
    xInitial[31] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr11_i2
    xInitial[32] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr12_i2
    xInitial[33] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr13_i2
    xInitial[34] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr14_i2
    xInitial[35] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr15_i2
    xInitial[36] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr16_i2
    xInitial[37] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr17_i2
    xInitial[38] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr18_i2
    xInitial[39] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_tr19_i2
    xInitial[40] = mpmath.mpi(0.0, 1.0)  	# e0_x_Reboilerin_i2
    xInitial[41] = mpmath.mpi(0.0, 1.0)  	# e0e1_x_B_i2
    xInitial[42] = mpmath.mpi(0.0, 1.0)  	# e0_x_Reboiler_i1
    xInitial[43] = mpmath.mpi(0.0, 1.0)  	# e0_x_Reboiler_i2
    xInitial[44] = mpmath.mpi(0.0, 1.0)  	# e0_y_D_i1
    xInitial[45] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr2_i1
    xInitial[46] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr3_i1
    xInitial[47] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr4_i1
    xInitial[48] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr5_i1
    xInitial[49] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr6_i1
    xInitial[50] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr7_i1
    xInitial[51] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr8_i1
    xInitial[52] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr9_i1
    xInitial[53] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr10_i1
    xInitial[54] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr11_i1
    xInitial[55] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr12_i1
    xInitial[56] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr13_i1
    xInitial[57] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr14_i1
    xInitial[58] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr15_i1
    xInitial[59] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr16_i1
    xInitial[60] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr17_i1
    xInitial[61] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr18_i1
    xInitial[62] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr19_i1
    xInitial[63] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr20_i1
    xInitial[64] = mpmath.mpi(0.0, 1.0)  	# e0_y_Reboiler_i1
    xInitial[65] = mpmath.mpi(0.0, 1.0)  	# e0_y_D_i2
    xInitial[66] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr2_i2
    xInitial[67] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr3_i2
    xInitial[68] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr4_i2
    xInitial[69] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr5_i2
    xInitial[70] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr6_i2
    xInitial[71] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr7_i2
    xInitial[72] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr8_i2
    xInitial[73] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr9_i2
    xInitial[74] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr10_i2
    xInitial[75] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr11_i2
    xInitial[76] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr12_i2
    xInitial[77] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr13_i2
    xInitial[78] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr14_i2
    xInitial[79] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr15_i2
    xInitial[80] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr16_i2
    xInitial[81] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr17_i2
    xInitial[82] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr18_i2
    xInitial[83] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr19_i2
    xInitial[84] = mpmath.mpi(0.0, 1.0)  	# e0e1_y_tr20_i2
    xInitial[85] = mpmath.mpi(0.0, 1.0)  	# e0_y_Reboiler_i2
    xInitial[86] = mpmath.mpi(0.0, 10.0)  	# e0_F_DLn
    xInitial[87] = mpmath.mpi(0.0, 10.0)  	# e0_K_D_i1
    xInitial[88] = mpmath.mpi(0.0, 10.0)  	# e0_K_D_i2
    xInitial[89] = mpmath.mpi(-1.0E8, 0.0)  	# e0_Q_D
    xInitial[90] = mpmath.mpi(0.0, 100.0)  	# e0_R_Reflux
    xInitial[91] = mpmath.mpi(200.0, 500.0)  	# e0_T_D
    xInitial[92] = mpmath.mpi(0.0, 1.0E9)  	# e0_p_DL
    xInitial[93] = mpmath.mpi(0.0, 1.0)  	# e0_y_DV_i1
    xInitial[94] = mpmath.mpi(0.0, 1.0)  	# e0_y_DV_i2
    xInitial[95] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_L_tr1
    xInitial[96] = mpmath.mpi(0.0, 10.0)  	# e0_K_Reboiler_i1
    xInitial[97] = mpmath.mpi(0.0, 10.0)  	# e0_K_Reboiler_i2
    xInitial[98] = mpmath.mpi(0.0, 1.0E8)  	# e0_Q_Reboiler
    xInitial[99] = mpmath.mpi(200.0, 500.0)  	# e0_T_Reboiler
    xInitial[100] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_L_tr2
    xInitial[101] = mpmath.mpi(0.0, 1.0E9)  	# e0_p_ReboilerV
    xInitial[102] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_L_tr3
    xInitial[103] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_L_tr4
    xInitial[104] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_L_tr5
    xInitial[105] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_L_tr6
    xInitial[106] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_L_tr7
    xInitial[107] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_L_tr8
    xInitial[108] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_L_tr9
    xInitial[109] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_L_tr10
    xInitial[110] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_L_tr11
    xInitial[111] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_L_tr12
    xInitial[112] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_L_tr13
    xInitial[113] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_L_tr14
    xInitial[114] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_L_tr15
    xInitial[115] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_L_tr16
    xInitial[116] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_L_tr17
    xInitial[117] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_L_tr18
    xInitial[118] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_L_tr19
    xInitial[119] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_L_tr20
    xInitial[120] = mpmath.mpi(0.0, 1.0)  	# e0_x_F_i2
    xInitial[121] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_V_tr1
    xInitial[122] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_V_tr2
    xInitial[123] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_V_tr3
    xInitial[124] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_V_tr4
    xInitial[125] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_V_tr5
    xInitial[126] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_V_tr6
    xInitial[127] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_V_tr7
    xInitial[128] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_V_tr8
    xInitial[129] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_V_tr9
    xInitial[130] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_V_tr10
    xInitial[131] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_V_tr11
    xInitial[132] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_V_tr12
    xInitial[133] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_V_tr13
    xInitial[134] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_V_tr14
    xInitial[135] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_V_tr15
    xInitial[136] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_V_tr16
    xInitial[137] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_V_tr17
    xInitial[138] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_V_tr18
    xInitial[139] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_V_tr19
    xInitial[140] = mpmath.mpi(1.0E-6, 0.1)  	# e0e1_greek_eta_V_tr20
    xInitial[141] = mpmath.mpi(100.0, 700.0)  	# e0e1_greek_rho_Lm_tr1
    xInitial[142] = mpmath.mpi(100.0, 700.0)  	# e0e1_greek_rho_Lm_tr2
    xInitial[143] = mpmath.mpi(100.0, 700.0)  	# e0e1_greek_rho_Lm_tr3
    xInitial[144] = mpmath.mpi(100.0, 700.0)  	# e0e1_greek_rho_Lm_tr4
    xInitial[145] = mpmath.mpi(100.0, 700.0)  	# e0e1_greek_rho_Lm_tr5
    xInitial[146] = mpmath.mpi(100.0, 700.0)  	# e0e1_greek_rho_Lm_tr6
    xInitial[147] = mpmath.mpi(100.0, 700.0)  	# e0e1_greek_rho_Lm_tr7
    xInitial[148] = mpmath.mpi(100.0, 700.0)  	# e0e1_greek_rho_Lm_tr8
    xInitial[149] = mpmath.mpi(100.0, 700.0)  	# e0e1_greek_rho_Lm_tr9
    xInitial[150] = mpmath.mpi(100.0, 700.0)  	# e0e1_greek_rho_Lm_tr10
    xInitial[151] = mpmath.mpi(100.0, 700.0)  	# e0e1_greek_rho_Lm_tr11
    xInitial[152] = mpmath.mpi(100.0, 700.0)  	# e0e1_greek_rho_Lm_tr12
    xInitial[153] = mpmath.mpi(100.0, 700.0)  	# e0e1_greek_rho_Lm_tr13
    xInitial[154] = mpmath.mpi(100.0, 700.0)  	# e0e1_greek_rho_Lm_tr14
    xInitial[155] = mpmath.mpi(100.0, 700.0)  	# e0e1_greek_rho_Lm_tr15
    xInitial[156] = mpmath.mpi(100.0, 700.0)  	# e0e1_greek_rho_Lm_tr16
    xInitial[157] = mpmath.mpi(100.0, 700.0)  	# e0e1_greek_rho_Lm_tr17
    xInitial[158] = mpmath.mpi(100.0, 700.0)  	# e0e1_greek_rho_Lm_tr18
    xInitial[159] = mpmath.mpi(100.0, 700.0)  	# e0e1_greek_rho_Lm_tr19
    xInitial[160] = mpmath.mpi(100.0, 700.0)  	# e0e1_greek_rho_Lm_tr20
    xInitial[161] = mpmath.mpi(0.0, 100.0)  	# e0e1_greek_rho_Vm_tr1
    xInitial[162] = mpmath.mpi(0.0, 100.0)  	# e0e1_greek_rho_Vm_tr2
    xInitial[163] = mpmath.mpi(0.0, 100.0)  	# e0e1_greek_rho_Vm_tr3
    xInitial[164] = mpmath.mpi(0.0, 100.0)  	# e0e1_greek_rho_Vm_tr4
    xInitial[165] = mpmath.mpi(0.0, 100.0)  	# e0e1_greek_rho_Vm_tr5
    xInitial[166] = mpmath.mpi(0.0, 100.0)  	# e0e1_greek_rho_Vm_tr6
    xInitial[167] = mpmath.mpi(0.0, 100.0)  	# e0e1_greek_rho_Vm_tr7
    xInitial[168] = mpmath.mpi(0.0, 100.0)  	# e0e1_greek_rho_Vm_tr8
    xInitial[169] = mpmath.mpi(0.0, 100.0)  	# e0e1_greek_rho_Vm_tr9
    xInitial[170] = mpmath.mpi(0.0, 100.0)  	# e0e1_greek_rho_Vm_tr10
    xInitial[171] = mpmath.mpi(0.0, 100.0)  	# e0e1_greek_rho_Vm_tr11
    xInitial[172] = mpmath.mpi(0.0, 100.0)  	# e0e1_greek_rho_Vm_tr12
    xInitial[173] = mpmath.mpi(0.0, 100.0)  	# e0e1_greek_rho_Vm_tr13
    xInitial[174] = mpmath.mpi(0.0, 100.0)  	# e0e1_greek_rho_Vm_tr14
    xInitial[175] = mpmath.mpi(0.0, 100.0)  	# e0e1_greek_rho_Vm_tr15
    xInitial[176] = mpmath.mpi(0.0, 100.0)  	# e0e1_greek_rho_Vm_tr16
    xInitial[177] = mpmath.mpi(0.0, 100.0)  	# e0e1_greek_rho_Vm_tr17
    xInitial[178] = mpmath.mpi(0.0, 100.0)  	# e0e1_greek_rho_Vm_tr18
    xInitial[179] = mpmath.mpi(0.0, 100.0)  	# e0e1_greek_rho_Vm_tr19
    xInitial[180] = mpmath.mpi(0.0, 100.0)  	# e0e1_greek_rho_Vm_tr20
    xInitial[181] = mpmath.mpi(0.0, 0.1)  	# e0e1_greek_sigma_tr1
    xInitial[182] = mpmath.mpi(0.0, 0.1)  	# e0e1_greek_sigma_tr2
    xInitial[183] = mpmath.mpi(0.0, 0.1)  	# e0e1_greek_sigma_tr3
    xInitial[184] = mpmath.mpi(0.0, 0.1)  	# e0e1_greek_sigma_tr4
    xInitial[185] = mpmath.mpi(0.0, 0.1)  	# e0e1_greek_sigma_tr5
    xInitial[186] = mpmath.mpi(0.0, 0.1)  	# e0e1_greek_sigma_tr6
    xInitial[187] = mpmath.mpi(0.0, 0.1)  	# e0e1_greek_sigma_tr7
    xInitial[188] = mpmath.mpi(0.0, 0.1)  	# e0e1_greek_sigma_tr8
    xInitial[189] = mpmath.mpi(0.0, 0.1)  	# e0e1_greek_sigma_tr9
    xInitial[190] = mpmath.mpi(0.0, 0.1)  	# e0e1_greek_sigma_tr10
    xInitial[191] = mpmath.mpi(0.0, 0.1)  	# e0e1_greek_sigma_tr11
    xInitial[192] = mpmath.mpi(0.0, 0.1)  	# e0e1_greek_sigma_tr12
    xInitial[193] = mpmath.mpi(0.0, 0.1)  	# e0e1_greek_sigma_tr13
    xInitial[194] = mpmath.mpi(0.0, 0.1)  	# e0e1_greek_sigma_tr14
    xInitial[195] = mpmath.mpi(0.0, 0.1)  	# e0e1_greek_sigma_tr15
    xInitial[196] = mpmath.mpi(0.0, 0.1)  	# e0e1_greek_sigma_tr16
    xInitial[197] = mpmath.mpi(0.0, 0.1)  	# e0e1_greek_sigma_tr17
    xInitial[198] = mpmath.mpi(0.0, 0.1)  	# e0e1_greek_sigma_tr18
    xInitial[199] = mpmath.mpi(0.0, 0.1)  	# e0e1_greek_sigma_tr19
    xInitial[200] = mpmath.mpi(0.0, 0.1)  	# e0e1_greek_sigma_tr20
    xInitial[201] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j1_tr1
    xInitial[202] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j1_tr2
    xInitial[203] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j1_tr3
    xInitial[204] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j1_tr4
    xInitial[205] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j1_tr5
    xInitial[206] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j1_tr6
    xInitial[207] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j1_tr7
    xInitial[208] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j1_tr8
    xInitial[209] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j1_tr9
    xInitial[210] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j1_tr10
    xInitial[211] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j1_tr11
    xInitial[212] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j1_tr12
    xInitial[213] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j1_tr13
    xInitial[214] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j1_tr14
    xInitial[215] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j1_tr15
    xInitial[216] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j1_tr16
    xInitial[217] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j1_tr17
    xInitial[218] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j1_tr18
    xInitial[219] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j1_tr19
    xInitial[220] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j1_tr20
    xInitial[221] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j2_tr1
    xInitial[222] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j2_tr2
    xInitial[223] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j2_tr3
    xInitial[224] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j2_tr4
    xInitial[225] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j2_tr5
    xInitial[226] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j2_tr6
    xInitial[227] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j2_tr7
    xInitial[228] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j2_tr8
    xInitial[229] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j2_tr9
    xInitial[230] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j2_tr10
    xInitial[231] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j2_tr11
    xInitial[232] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j2_tr12
    xInitial[233] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j2_tr13
    xInitial[234] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j2_tr14
    xInitial[235] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j2_tr15
    xInitial[236] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j2_tr16
    xInitial[237] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j2_tr17
    xInitial[238] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j2_tr18
    xInitial[239] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j2_tr19
    xInitial[240] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i1_j2_tr20
    xInitial[241] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j1_tr1
    xInitial[242] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j1_tr2
    xInitial[243] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j1_tr3
    xInitial[244] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j1_tr4
    xInitial[245] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j1_tr5
    xInitial[246] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j1_tr6
    xInitial[247] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j1_tr7
    xInitial[248] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j1_tr8
    xInitial[249] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j1_tr9
    xInitial[250] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j1_tr10
    xInitial[251] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j1_tr11
    xInitial[252] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j1_tr12
    xInitial[253] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j1_tr13
    xInitial[254] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j1_tr14
    xInitial[255] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j1_tr15
    xInitial[256] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j1_tr16
    xInitial[257] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j1_tr17
    xInitial[258] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j1_tr18
    xInitial[259] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j1_tr19
    xInitial[260] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j1_tr20
    xInitial[261] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j2_tr1
    xInitial[262] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j2_tr2
    xInitial[263] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j2_tr3
    xInitial[264] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j2_tr4
    xInitial[265] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j2_tr5
    xInitial[266] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j2_tr6
    xInitial[267] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j2_tr7
    xInitial[268] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j2_tr8
    xInitial[269] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j2_tr9
    xInitial[270] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j2_tr10
    xInitial[271] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j2_tr11
    xInitial[272] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j2_tr12
    xInitial[273] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j2_tr13
    xInitial[274] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j2_tr14
    xInitial[275] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j2_tr15
    xInitial[276] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j2_tr16
    xInitial[277] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j2_tr17
    xInitial[278] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j2_tr18
    xInitial[279] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j2_tr19
    xInitial[280] = mpmath.mpi(0.0, 3.0)  	# e0e1_greek_theta_i2_j2_tr20
    xInitial[281] = mpmath.mpi(0.0, 3.0)  	# e0e1_A
    xInitial[282] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_BLn
    xInitial[283] = mpmath.mpi(0.0, 10.0)  	# e0_F_LReboilerinn
    xInitial[284] = mpmath.mpi(0.0, 10.0)  	# e0_F_DLRefluxn
    xInitial[285] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Ln_tr1
    xInitial[286] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Ln_tr2
    xInitial[287] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Ln_tr3
    xInitial[288] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Ln_tr4
    xInitial[289] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Ln_tr5
    xInitial[290] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Ln_tr6
    xInitial[291] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Ln_tr7
    xInitial[292] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Ln_tr8
    xInitial[293] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Ln_tr9
    xInitial[294] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Ln_tr10
    xInitial[295] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Ln_tr11
    xInitial[296] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Ln_tr12
    xInitial[297] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Ln_tr13
    xInitial[298] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Ln_tr14
    xInitial[299] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Ln_tr15
    xInitial[300] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Ln_tr16
    xInitial[301] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Ln_tr17
    xInitial[302] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Ln_tr18
    xInitial[303] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Ln_tr19
    xInitial[304] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Ln_tr20
    xInitial[305] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Ln_tr21
    xInitial[306] = mpmath.mpi(0.0, 10.0)  	# e0_F_DVn
    xInitial[307] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Vn_tr2
    xInitial[308] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Vn_tr3
    xInitial[309] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Vn_tr4
    xInitial[310] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Vn_tr5
    xInitial[311] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Vn_tr6
    xInitial[312] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Vn_tr7
    xInitial[313] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Vn_tr8
    xInitial[314] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Vn_tr9
    xInitial[315] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Vn_tr10
    xInitial[316] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Vn_tr11
    xInitial[317] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Vn_tr12
    xInitial[318] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Vn_tr13
    xInitial[319] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Vn_tr14
    xInitial[320] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Vn_tr15
    xInitial[321] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Vn_tr16
    xInitial[322] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Vn_tr17
    xInitial[323] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Vn_tr18
    xInitial[324] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Vn_tr19
    xInitial[325] = mpmath.mpi(0.0, 5.0)  	# e0e1_F_Vn_tr20
    xInitial[326] = mpmath.mpi(0.0, 10.0)  	# e0_F_ReboilerVn
    xInitial[327] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr1_i1
    xInitial[328] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr2_i1
    xInitial[329] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr3_i1
    xInitial[330] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr4_i1
    xInitial[331] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr5_i1
    xInitial[332] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr6_i1
    xInitial[333] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr7_i1
    xInitial[334] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr8_i1
    xInitial[335] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr9_i1
    xInitial[336] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr10_i1
    xInitial[337] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr11_i1
    xInitial[338] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr12_i1
    xInitial[339] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr13_i1
    xInitial[340] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr14_i1
    xInitial[341] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr15_i1
    xInitial[342] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr16_i1
    xInitial[343] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr17_i1
    xInitial[344] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr18_i1
    xInitial[345] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr19_i1
    xInitial[346] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr20_i1
    xInitial[347] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr1_i2
    xInitial[348] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr2_i2
    xInitial[349] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr3_i2
    xInitial[350] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr4_i2
    xInitial[351] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr5_i2
    xInitial[352] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr6_i2
    xInitial[353] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr7_i2
    xInitial[354] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr8_i2
    xInitial[355] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr9_i2
    xInitial[356] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr10_i2
    xInitial[357] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr11_i2
    xInitial[358] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr12_i2
    xInitial[359] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr13_i2
    xInitial[360] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr14_i2
    xInitial[361] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr15_i2
    xInitial[362] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr16_i2
    xInitial[363] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr17_i2
    xInitial[364] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr18_i2
    xInitial[365] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr19_i2
    xInitial[366] = mpmath.mpi(0.0, 3.0)  	# e0e1_K_tr20_i2
    xInitial[367] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_L_tr1
    xInitial[368] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_L_tr2
    xInitial[369] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_L_tr3
    xInitial[370] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_L_tr4
    xInitial[371] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_L_tr5
    xInitial[372] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_L_tr6
    xInitial[373] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_L_tr7
    xInitial[374] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_L_tr8
    xInitial[375] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_L_tr9
    xInitial[376] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_L_tr10
    xInitial[377] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_L_tr11
    xInitial[378] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_L_tr12
    xInitial[379] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_L_tr13
    xInitial[380] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_L_tr14
    xInitial[381] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_L_tr15
    xInitial[382] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_L_tr16
    xInitial[383] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_L_tr17
    xInitial[384] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_L_tr18
    xInitial[385] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_L_tr19
    xInitial[386] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_L_tr20
    xInitial[387] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_V_tr1
    xInitial[388] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_V_tr2
    xInitial[389] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_V_tr3
    xInitial[390] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_V_tr4
    xInitial[391] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_V_tr5
    xInitial[392] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_V_tr6
    xInitial[393] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_V_tr7
    xInitial[394] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_V_tr8
    xInitial[395] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_V_tr9
    xInitial[396] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_V_tr10
    xInitial[397] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_V_tr11
    xInitial[398] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_V_tr12
    xInitial[399] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_V_tr13
    xInitial[400] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_V_tr14
    xInitial[401] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_V_tr15
    xInitial[402] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_V_tr16
    xInitial[403] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_V_tr17
    xInitial[404] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_V_tr18
    xInitial[405] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_V_tr19
    xInitial[406] = mpmath.mpi(10.0, 70.0)  	# e0e1_M_V_tr20
    xInitial[407] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i1_tr1
    xInitial[408] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i1_tr2
    xInitial[409] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i1_tr3
    xInitial[410] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i1_tr4
    xInitial[411] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i1_tr5
    xInitial[412] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i1_tr6
    xInitial[413] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i1_tr7
    xInitial[414] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i1_tr8
    xInitial[415] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i1_tr9
    xInitial[416] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i1_tr10
    xInitial[417] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i1_tr11
    xInitial[418] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i1_tr12
    xInitial[419] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i1_tr13
    xInitial[420] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i1_tr14
    xInitial[421] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i1_tr15
    xInitial[422] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i1_tr16
    xInitial[423] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i1_tr17
    xInitial[424] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i1_tr18
    xInitial[425] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i1_tr19
    xInitial[426] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i1_tr20
    xInitial[427] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i2_tr1
    xInitial[428] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i2_tr2
    xInitial[429] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i2_tr3
    xInitial[430] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i2_tr4
    xInitial[431] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i2_tr5
    xInitial[432] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i2_tr6
    xInitial[433] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i2_tr7
    xInitial[434] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i2_tr8
    xInitial[435] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i2_tr9
    xInitial[436] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i2_tr10
    xInitial[437] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i2_tr11
    xInitial[438] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i2_tr12
    xInitial[439] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i2_tr13
    xInitial[440] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i2_tr14
    xInitial[441] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i2_tr15
    xInitial[442] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i2_tr16
    xInitial[443] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i2_tr17
    xInitial[444] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i2_tr18
    xInitial[445] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i2_tr19
    xInitial[446] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_i2_tr20
    xInitial[447] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_tr1
    xInitial[448] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_tr2
    xInitial[449] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_tr3
    xInitial[450] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_tr4
    xInitial[451] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_tr5
    xInitial[452] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_tr6
    xInitial[453] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_tr7
    xInitial[454] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_tr8
    xInitial[455] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_tr9
    xInitial[456] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_tr10
    xInitial[457] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_tr11
    xInitial[458] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_tr12
    xInitial[459] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_tr13
    xInitial[460] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_tr14
    xInitial[461] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_tr15
    xInitial[462] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_tr16
    xInitial[463] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_tr17
    xInitial[464] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_tr18
    xInitial[465] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_tr19
    xInitial[466] = mpmath.mpi(150.0, 200.0)  	# e0e1_P_tr20
    xInitial[467] = mpmath.mpi(0.0, 400.0)  	# e0e1_T_tr1
    xInitial[468] = mpmath.mpi(0.0, 400.0)  	# e0e1_T_tr2
    xInitial[469] = mpmath.mpi(0.0, 400.0)  	# e0e1_T_tr3
    xInitial[470] = mpmath.mpi(0.0, 400.0)  	# e0e1_T_tr4
    xInitial[471] = mpmath.mpi(0.0, 400.0)  	# e0e1_T_tr5
    xInitial[472] = mpmath.mpi(0.0, 400.0)  	# e0e1_T_tr6
    xInitial[473] = mpmath.mpi(0.0, 400.0)  	# e0e1_T_tr7
    xInitial[474] = mpmath.mpi(0.0, 400.0)  	# e0e1_T_tr8
    xInitial[475] = mpmath.mpi(0.0, 400.0)  	# e0e1_T_tr9
    xInitial[476] = mpmath.mpi(0.0, 400.0)  	# e0e1_T_tr10
    xInitial[477] = mpmath.mpi(0.0, 400.0)  	# e0e1_T_tr11
    xInitial[478] = mpmath.mpi(0.0, 400.0)  	# e0e1_T_tr12
    xInitial[479] = mpmath.mpi(0.0, 400.0)  	# e0e1_T_tr13
    xInitial[480] = mpmath.mpi(0.0, 400.0)  	# e0e1_T_tr14
    xInitial[481] = mpmath.mpi(0.0, 400.0)  	# e0e1_T_tr15
    xInitial[482] = mpmath.mpi(0.0, 400.0)  	# e0e1_T_tr16
    xInitial[483] = mpmath.mpi(0.0, 400.0)  	# e0e1_T_tr17
    xInitial[484] = mpmath.mpi(0.0, 400.0)  	# e0e1_T_tr18
    xInitial[485] = mpmath.mpi(0.0, 400.0)  	# e0e1_T_tr19
    xInitial[486] = mpmath.mpi(0.0, 400.0)  	# e0e1_T_tr20
    xInitial[487] = mpmath.mpi(0.0, 400.0)  	# e0e1_T_B
    xInitial[488] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_BLn
    xInitial[489] = mpmath.mpi(-1.0E9, 0.0)  	# e0_h_LReboilern
    xInitial[490] = mpmath.mpi(-1.0E9, 0.0)  	# e0_h_Fn
    xInitial[491] = mpmath.mpi(-1.0E9, 0.0)  	# e0_h_DLn
    xInitial[492] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Ln_tr1
    xInitial[493] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Ln_tr2
    xInitial[494] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Ln_tr3
    xInitial[495] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Ln_tr4
    xInitial[496] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Ln_tr5
    xInitial[497] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Ln_tr6
    xInitial[498] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Ln_tr7
    xInitial[499] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Ln_tr8
    xInitial[500] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Ln_tr9
    xInitial[501] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Ln_tr10
    xInitial[502] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Ln_tr11
    xInitial[503] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Ln_tr12
    xInitial[504] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Ln_tr13
    xInitial[505] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Ln_tr14
    xInitial[506] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Ln_tr15
    xInitial[507] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Ln_tr16
    xInitial[508] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Ln_tr17
    xInitial[509] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Ln_tr18
    xInitial[510] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Ln_tr19
    xInitial[511] = mpmath.mpi(-1.0E9, 0.0)  	# e0_h_LReboilerinn
    xInitial[512] = mpmath.mpi(-1.0E9, 0.0)  	# e0_h_DVn
    xInitial[513] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Vn_tr2
    xInitial[514] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Vn_tr3
    xInitial[515] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Vn_tr4
    xInitial[516] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Vn_tr5
    xInitial[517] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Vn_tr6
    xInitial[518] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Vn_tr7
    xInitial[519] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Vn_tr8
    xInitial[520] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Vn_tr9
    xInitial[521] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Vn_tr10
    xInitial[522] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Vn_tr11
    xInitial[523] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Vn_tr12
    xInitial[524] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Vn_tr13
    xInitial[525] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Vn_tr14
    xInitial[526] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Vn_tr15
    xInitial[527] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Vn_tr16
    xInitial[528] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Vn_tr17
    xInitial[529] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Vn_tr18
    xInitial[530] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Vn_tr19
    xInitial[531] = mpmath.mpi(-1.0E9, 0.0)  	# e0e1_h_Vn_tr20
    xInitial[532] = mpmath.mpi(-1.0E9, 0.0)  	# e0_h_ReboilerVn
    xInitial[533] = mpmath.mpi(600000.0, 800000.0)  	# e0e1_p_tr1
    xInitial[534] = mpmath.mpi(600000.0, 800000.0)  	# e0e1_p_tr2
    xInitial[535] = mpmath.mpi(600000.0, 800000.0)  	# e0e1_p_tr3
    xInitial[536] = mpmath.mpi(600000.0, 800000.0)  	# e0e1_p_tr4
    xInitial[537] = mpmath.mpi(600000.0, 800000.0)  	# e0e1_p_tr5
    xInitial[538] = mpmath.mpi(600000.0, 800000.0)  	# e0e1_p_tr6
    xInitial[539] = mpmath.mpi(600000.0, 800000.0)  	# e0e1_p_tr7
    xInitial[540] = mpmath.mpi(600000.0, 800000.0)  	# e0e1_p_tr8
    xInitial[541] = mpmath.mpi(600000.0, 800000.0)  	# e0e1_p_tr9
    xInitial[542] = mpmath.mpi(600000.0, 800000.0)  	# e0e1_p_tr10
    xInitial[543] = mpmath.mpi(600000.0, 800000.0)  	# e0e1_p_tr11
    xInitial[544] = mpmath.mpi(600000.0, 800000.0)  	# e0e1_p_tr12
    xInitial[545] = mpmath.mpi(600000.0, 800000.0)  	# e0e1_p_tr13
    xInitial[546] = mpmath.mpi(600000.0, 800000.0)  	# e0e1_p_tr14
    xInitial[547] = mpmath.mpi(600000.0, 800000.0)  	# e0e1_p_tr15
    xInitial[548] = mpmath.mpi(600000.0, 800000.0)  	# e0e1_p_tr16
    xInitial[549] = mpmath.mpi(600000.0, 800000.0)  	# e0e1_p_tr17
    xInitial[550] = mpmath.mpi(600000.0, 800000.0)  	# e0e1_p_tr18
    xInitial[551] = mpmath.mpi(600000.0, 800000.0)  	# e0e1_p_tr19
    xInitial[552] = mpmath.mpi(0.0, 1.0E9)  	# e0_p_LReboiler
    xInitial[553] = mpmath.mpi(600000.0, 800000.0)  	# e0e1_p_tr21


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
