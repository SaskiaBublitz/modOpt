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
    dict_options = {"fileName": "Reactor_r3",
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

def getSymbolicFunctions(x, p):
    """ User specified Equation system 
    
    Args:
        x:          sympy array with symbolic state variable bounds
        p:          numpy array with parameter values
    
    Returns:
        f:          sympy array with symbolic residual bounds
        
    """
# Getting variable values:
    e0_h_SecRfeed_c1 = x[0]
    e0_h_SecRfeed_c2 = x[1]
    e0_h_SecRfeed_c3 = x[2]
    e0_h_SecRfeed_c4 = x[3]
    e0_h_SecRfeed_c5 = x[4]
    e0_HU_LSecR = x[5]
    e0_v_LSecR = x[6]
    e0_greek_epsiv_LSecR = x[7]
    e0_V_SecRV = x[8]
    e0_HU_SecRV = x[9]
    e0_v_SecRV = x[10]
    e0_c_SecR_c1 = x[11]
    e0_c_SecR_c2 = x[12]
    e0_greek_DeltaP_SecR = x[13]
    e0_P_SecR = x[14]
    e0_greek_rho_LSecR = x[15]
    e0_M_LSecR = x[16]
    e0_greek_rho_SecRV = x[17]
    e0_M_SecRV = x[18]
    e0_greek_alpha_SecR_c1 = x[19]
    e0_greek_alpha_SecR_c2 = x[20]
    e0_greek_alpha_SecR_c3 = x[21]
    e0_greek_alpha_SecR_c4 = x[22]
    e0_greek_alpha_SecR_c5 = x[23]
    e0_a_c1 = x[24]
    e0_a_c2 = x[25]
    e0_a_c3 = x[26]
    e0_a_c4 = x[27]
    e0_a_c5 = x[28]
    e0_a_LSecR = x[29]
    e0_a_SecRV = x[30]
    e0_b_c1 = x[31]
    e0_b_c2 = x[32]
    e0_b_c3 = x[33]
    e0_b_c4 = x[34]
    e0_b_c5 = x[35]
    e0_b_LSecR = x[36]
    e0_b_SecRV = x[37]
    e0_m_c1 = x[38]
    e0_m_c2 = x[39]
    e0_m_c3 = x[40]
    e0_m_c4 = x[41]
    e0_m_c5 = x[42]
    e0_F_LSecR_c1 = x[43]
    e0_F_LSecR_c2 = x[44]
    e0_F_LSecR_c3 = x[45]
    e0_F_LSecR_c4 = x[46]
    e0_F_LSecR_c5 = x[47]
    e0_F_SecRV_c1 = x[48]
    e0_F_SecRV_c2 = x[49]
    e0_F_SecRV_c3 = x[50]
    e0_F_SecRV_c4 = x[51]
    e0_F_SecRV_c5 = x[52]
    e0_V_LSecR = x[53]
    e0_r_SecR_r1 = x[54]
    e0_r_SecR_r2 = x[55]
    e0_greek_phiv_LSecR_c1 = x[56]
    e0_greek_phiv_LSecR_c2 = x[57]
    e0_greek_phiv_LSecR_c3 = x[58]
    e0_greek_phiv_LSecR_c4 = x[59]
    e0_greek_phiv_LSecR_c5 = x[60]
    e0_greek_phiv_SecRV_c1 = x[61]
    e0_greek_phiv_SecRV_c2 = x[62]
    e0_greek_phiv_SecRV_c3 = x[63]
    e0_greek_phiv_SecRV_c4 = x[64]
    e0_greek_phiv_SecRV_c5 = x[65]
    e0_F_SecRV = x[66]
    e0_F_SecRfeed = x[67]
    e0_HU_LSecR_c1 = x[68]
    e0_HU_LSecR_c2 = x[69]
    e0_HU_LSecR_c3 = x[70]
    e0_HU_LSecR_c4 = x[71]
    e0_HU_LSecR_c5 = x[72]
    e0_HU_SecR_c1 = x[73]
    e0_HU_SecR_c2 = x[74]
    e0_HU_SecR_c3 = x[75]
    e0_HU_SecR_c4 = x[76]
    e0_HU_SecR_c5 = x[77]
    e0_HU_SecRV_c1 = x[78]
    e0_HU_SecRV_c2 = x[79]
    e0_HU_SecRV_c3 = x[80]
    e0_HU_SecRV_c4 = x[81]
    e0_HU_SecRV_c5 = x[82]
    e0_h_LSecR = x[83]
    e0_h_SecRV = x[84]
    e0_h_SecRfeed = x[85]
    e0_greek_Deltah_SecRVLdeparture = x[86]
    e0_h_SecRV_c1 = x[87]
    e0_h_SecRV_c2 = x[88]
    e0_h_SecRV_c3 = x[89]
    e0_h_SecRV_c4 = x[90]
    e0_h_SecRV_c5 = x[91]
    e0_T_SecR = x[92]



# Getting parameter values:
    e0_greek_nu_c1_r1 = p[0]
    e0_greek_nu_c1_r2 = p[1]
    e0_F_CONLSecR_c1 = p[2]
    e0_F_CONLSecR_c2 = p[3]
    e0_T_SecRfeed = p[4]
    e0_ASecR = p[5]
    e0_LSecR = p[6]
    e0_F_CONLSecR_c3 = p[7]
    e0_E_r1 = p[8]
    e0_k_r1 = p[9]
    e0_E_r2 = p[10]
    e0_k_r2 = p[11]
    e0_P_SecRtop = p[12]
    e0_F_CONLSecR_c4 = p[13]
    e0_F_CONLSecR_c5 = p[14]
    e0_greek_delta_c1_caux1 = p[15]
    e0_greek_delta_c1_caux2 = p[16]
    e0_greek_delta_c1_caux3 = p[17]
    e0_greek_delta_c1_caux4 = p[18]
    e0_greek_delta_c1_caux5 = p[19]
    e0_greek_delta_c2_caux1 = p[20]
    e0_greek_delta_c2_caux2 = p[21]
    e0_F_CoolerLSecR_c1 = p[22]
    e0_greek_delta_c2_caux3 = p[23]
    e0_greek_delta_c2_caux4 = p[24]
    e0_greek_delta_c2_caux5 = p[25]
    e0_greek_delta_c3_caux1 = p[26]
    e0_greek_delta_c3_caux2 = p[27]
    e0_greek_delta_c3_caux3 = p[28]
    e0_greek_delta_c3_caux4 = p[29]
    e0_greek_delta_c3_caux5 = p[30]
    e0_greek_delta_c4_caux1 = p[31]
    e0_greek_delta_c4_caux2 = p[32]
    e0_F_CoolerLSecR_c2 = p[33]
    e0_greek_delta_c4_caux3 = p[34]
    e0_greek_delta_c4_caux4 = p[35]
    e0_greek_delta_c4_caux5 = p[36]
    e0_greek_delta_c5_caux1 = p[37]
    e0_greek_delta_c5_caux2 = p[38]
    e0_greek_delta_c5_caux3 = p[39]
    e0_greek_delta_c5_caux4 = p[40]
    e0_greek_delta_c5_caux5 = p[41]
    e0_F_CoolerLSecR_c3 = p[42]
    e0_F_CoolerLSecR_c4 = p[43]
    e0_F_CoolerLSecR_c5 = p[44]
    e0_aEOS = p[45]
    e0_bEOS = p[46]
    e0_mEOS_m1 = p[47]
    e0_mEOS_m2 = p[48]
    e0_mEOS_m3 = p[49]
    e0_greek_nu_c2_r1 = p[50]
    e0_greek_nu_c2_r2 = p[51]
    e0_F_VCON_c1 = p[52]
    e0_F_VCON_c2 = p[53]
    e0_F_VCON_c3 = p[54]
    e0_F_VCON_c4 = p[55]
    e0_F_VCON_c5 = p[56]
    e0_PN = p[57]
    e0_TN = p[58]
    e0_V_SecRfeedN_c1 = p[59]
    e0_V_SecRfeedN_c2 = p[60]
    e0_greek_nu_c3_r1 = p[61]
    e0_V_SecRfeedN_c3 = p[62]
    e0_V_SecRfeedN_c4 = p[63]
    e0_V_SecRfeedN_c5 = p[64]
    e0_greek_nu_c3_r2 = p[65]
    e0_F_LSecR = p[66]
    e0_greek_nu_c4_r1 = p[67]
    e0_greek_nu_c4_r2 = p[68]
    e0_F_CONLSecR = p[69]
    e0_F_CoolerLSecR = p[70]
    e0_Q_SecR = p[71]
    e0_h_CONLSecR = p[72]
    e0_h_CoolerLSecR = p[73]
    e0_greek_nu_c5_r1 = p[74]
    e0_h_VCON = p[75]
    e0_greek_nu_c5_r2 = p[76]
    e0_greek_Deltahf_c1 = p[77]
    e0_greek_Deltahf_c2 = p[78]
    e0_greek_Deltahf_c3 = p[79]
    e0_greek_Deltahf_c4 = p[80]
    e0_greek_Deltahf_c5 = p[81]
    e0_greek_omega_c1 = p[82]
    e0_greek_omega_c2 = p[83]
    e0_greek_omega_c3 = p[84]
    e0_greek_omega_c4 = p[85]
    e0_greek_omega_c5 = p[86]
    e0_CPA_c1 = p[87]
    e0_CPA_c2 = p[88]
    e0_CPA_c3 = p[89]
    e0_CPA_c4 = p[90]
    e0_CPA_c5 = p[91]
    e0_CPB_c1 = p[92]
    e0_CPB_c2 = p[93]
    e0_CPB_c3 = p[94]
    e0_CPB_c4 = p[95]
    e0_CPB_c5 = p[96]
    e0_M_c1 = p[97]
    e0_M_c2 = p[98]
    e0_M_c3 = p[99]
    e0_M_c4 = p[100]
    e0_M_c5 = p[101]
    e0_PC_c1 = p[102]
    e0_PC_c2 = p[103]
    e0_PC_c3 = p[104]
    e0_PC_c4 = p[105]
    e0_PC_c5 = p[106]
    e0_R = p[107]
    e0_TC_c1 = p[108]
    e0_TC_c2 = p[109]
    e0_TC_c3 = p[110]
    e0_TC_c4 = p[111]
    e0_TC_c5 = p[112]
    e0_g = p[113]



   # Getting function values:



# Solve equation system for given x:
    f= [
    0.0-(((e0_PN) *(e0_V_SecRfeedN_c1))/((e0_R) *((e0_TN) *(3600.0)))+e0_F_VCON_c1+e0_F_CONLSecR_c1+e0_F_CoolerLSecR_c1-e0_F_LSecR_c1-e0_F_SecRV_c1+(e0_V_LSecR) *(((e0_greek_nu_c1_r1) *(e0_r_SecR_r1)+(e0_greek_nu_c1_r2) *(e0_r_SecR_r2)))) ,
    0.0-(((e0_PN) *(e0_V_SecRfeedN_c2))/((e0_R) *((e0_TN) *(3600.0)))+e0_F_VCON_c2+e0_F_CONLSecR_c2+e0_F_CoolerLSecR_c2-e0_F_LSecR_c2-e0_F_SecRV_c2+(e0_V_LSecR) *(((e0_greek_nu_c2_r1) *(e0_r_SecR_r1)+(e0_greek_nu_c2_r2) *(e0_r_SecR_r2)))) ,
    0.0-(((e0_PN) *(e0_V_SecRfeedN_c3))/((e0_R) *((e0_TN) *(3600.0)))+e0_F_VCON_c3+e0_F_CONLSecR_c3+e0_F_CoolerLSecR_c3-e0_F_LSecR_c3-e0_F_SecRV_c3+(e0_V_LSecR) *(((e0_greek_nu_c3_r1) *(e0_r_SecR_r1)+(e0_greek_nu_c3_r2) *(e0_r_SecR_r2)))) ,
    0.0-(((e0_PN) *(e0_V_SecRfeedN_c4))/((e0_R) *((e0_TN) *(3600.0)))+e0_F_VCON_c4+e0_F_CONLSecR_c4+e0_F_CoolerLSecR_c4-e0_F_LSecR_c4-e0_F_SecRV_c4+(e0_V_LSecR) *(((e0_greek_nu_c4_r1) *(e0_r_SecR_r1)+(e0_greek_nu_c4_r2) *(e0_r_SecR_r2)))) ,
    0.0-(((e0_PN) *(e0_V_SecRfeedN_c5))/((e0_R) *((e0_TN) *(3600.0)))+e0_F_VCON_c5+e0_F_CONLSecR_c5+e0_F_CoolerLSecR_c5-e0_F_LSecR_c5-e0_F_SecRV_c5+(e0_V_LSecR) *(((e0_greek_nu_c5_r1) *(e0_r_SecR_r1)+(e0_greek_nu_c5_r2) *(e0_r_SecR_r2)))) ,
    ((e0_F_SecRV_c1)/(e0_F_SecRV)) *(e0_greek_phiv_SecRV_c1)-(((e0_F_LSecR_c1)/(e0_F_LSecR)) *(e0_greek_phiv_LSecR_c1)) ,
    ((e0_F_SecRV_c2)/(e0_F_SecRV)) *(e0_greek_phiv_SecRV_c2)-(((e0_F_LSecR_c2)/(e0_F_LSecR)) *(e0_greek_phiv_LSecR_c2)) ,
    ((e0_F_SecRV_c3)/(e0_F_SecRV)) *(e0_greek_phiv_SecRV_c3)-(((e0_F_LSecR_c3)/(e0_F_LSecR)) *(e0_greek_phiv_LSecR_c3)) ,
    ((e0_F_SecRV_c4)/(e0_F_SecRV)) *(e0_greek_phiv_SecRV_c4)-(((e0_F_LSecR_c4)/(e0_F_LSecR)) *(e0_greek_phiv_LSecR_c4)) ,
    ((e0_F_SecRV_c5)/(e0_F_SecRV)) *(e0_greek_phiv_SecRV_c5)-(((e0_F_LSecR_c5)/(e0_F_LSecR)) *(e0_greek_phiv_LSecR_c5)) ,
    e0_F_SecRfeed-((((e0_PN) *(e0_V_SecRfeedN_c1))/((e0_R) *((e0_TN) *(3600.0)))+((e0_PN) *(e0_V_SecRfeedN_c2))/((e0_R) *((e0_TN) *(3600.0)))+((e0_PN) *(e0_V_SecRfeedN_c3))/((e0_R) *((e0_TN) *(3600.0)))+((e0_PN) *(e0_V_SecRfeedN_c4))/((e0_R) *((e0_TN) *(3600.0)))+((e0_PN) *(e0_V_SecRfeedN_c5))/((e0_R) *((e0_TN) *(3600.0))))) ,
    e0_F_SecRV-((e0_F_SecRV_c1+e0_F_SecRV_c2+e0_F_SecRV_c3+e0_F_SecRV_c4+e0_F_SecRV_c5)) ,
    e0_F_LSecR-((e0_F_LSecR_c1+e0_F_LSecR_c2+e0_F_LSecR_c3+e0_F_LSecR_c4+e0_F_LSecR_c5)) ,
    e0_HU_SecR_c1-(e0_HU_LSecR_c1+e0_HU_SecRV_c1) ,
    e0_HU_SecR_c2-(e0_HU_LSecR_c2+e0_HU_SecRV_c2) ,
    e0_HU_SecR_c3-(e0_HU_LSecR_c3+e0_HU_SecRV_c3) ,
    e0_HU_SecR_c4-(e0_HU_LSecR_c4+e0_HU_SecRV_c4) ,
    e0_HU_SecR_c5-(e0_HU_LSecR_c5+e0_HU_SecRV_c5) ,
    0.0-(e0_Q_SecR+(e0_F_SecRfeed) *(e0_h_SecRfeed)+((e0_F_VCON_c1+e0_F_VCON_c2+e0_F_VCON_c3+e0_F_VCON_c4+e0_F_VCON_c5)) *(e0_h_VCON)+(e0_F_CONLSecR) *(e0_h_CONLSecR)+(e0_F_CoolerLSecR) *(e0_h_CoolerLSecR)-(e0_F_SecRV) *(e0_h_SecRV)-(e0_F_LSecR) *(e0_h_LSecR)) ,
    (e0_F_LSecR) *(e0_h_LSecR)-(((e0_F_LSecR_c1) *(e0_h_SecRV_c1)+(e0_F_LSecR_c2) *(e0_h_SecRV_c2)+(e0_F_LSecR_c3) *(e0_h_SecRV_c3)+(e0_F_LSecR_c4) *(e0_h_SecRV_c4)+(e0_F_LSecR_c5) *(e0_h_SecRV_c5))-(e0_F_LSecR) *(e0_greek_Deltah_SecRVLdeparture)) ,
    (e0_F_SecRV) *(e0_h_SecRV)-(((e0_F_SecRV_c1) *(e0_h_SecRV_c1)+(e0_F_SecRV_c2) *(e0_h_SecRV_c2)+(e0_F_SecRV_c3) *(e0_h_SecRV_c3)+(e0_F_SecRV_c4) *(e0_h_SecRV_c4)+(e0_F_SecRV_c5) *(e0_h_SecRV_c5))) ,
    e0_h_SecRV_c1-(e0_greek_Deltahf_c1+(e0_CPA_c1) *(e0_T_SecR)+e0_CPB_c1) ,
    e0_h_SecRV_c2-(e0_greek_Deltahf_c2+(e0_CPA_c2) *(e0_T_SecR)+e0_CPB_c2) ,
    e0_h_SecRV_c3-(e0_greek_Deltahf_c3+(e0_CPA_c3) *(e0_T_SecR)+e0_CPB_c3) ,
    e0_h_SecRV_c4-(e0_greek_Deltahf_c4+(e0_CPA_c4) *(e0_T_SecR)+e0_CPB_c4) ,
    e0_h_SecRV_c5-(e0_greek_Deltahf_c5+(e0_CPA_c5) *(e0_T_SecR)+e0_CPB_c5) ,
    (e0_F_SecRfeed) *(e0_h_SecRfeed)-(((((e0_PN) *(e0_V_SecRfeedN_c1))/((e0_R) *((e0_TN) *(3600.0)))) *(e0_h_SecRfeed_c1)+(((e0_PN) *(e0_V_SecRfeedN_c2))/((e0_R) *((e0_TN) *(3600.0)))) *(e0_h_SecRfeed_c2)+(((e0_PN) *(e0_V_SecRfeedN_c3))/((e0_R) *((e0_TN) *(3600.0)))) *(e0_h_SecRfeed_c3)+(((e0_PN) *(e0_V_SecRfeedN_c4))/((e0_R) *((e0_TN) *(3600.0)))) *(e0_h_SecRfeed_c4)+(((e0_PN) *(e0_V_SecRfeedN_c5))/((e0_R) *((e0_TN) *(3600.0)))) *(e0_h_SecRfeed_c5))) ,
    e0_h_SecRfeed_c1-(e0_greek_Deltahf_c1+(e0_CPA_c1) *(e0_T_SecRfeed)+e0_CPB_c1) ,
    e0_h_SecRfeed_c2-(e0_greek_Deltahf_c2+(e0_CPA_c2) *(e0_T_SecRfeed)+e0_CPB_c2) ,
    e0_h_SecRfeed_c3-(e0_greek_Deltahf_c3+(e0_CPA_c3) *(e0_T_SecRfeed)+e0_CPB_c3) ,
    e0_h_SecRfeed_c4-(e0_greek_Deltahf_c4+(e0_CPA_c4) *(e0_T_SecRfeed)+e0_CPB_c4) ,
    e0_h_SecRfeed_c5-(e0_greek_Deltahf_c5+(e0_CPA_c5) *(e0_T_SecRfeed)+e0_CPB_c5) ,
    e0_V_LSecR-((e0_HU_LSecR) *(e0_v_LSecR)) ,
    e0_V_LSecR-((e0_greek_epsiv_LSecR) *((e0_ASecR) *(e0_LSecR))) ,
    e0_V_LSecR+e0_V_SecRV-((e0_ASecR) *(e0_LSecR)) ,
    e0_V_SecRV-((e0_HU_SecRV) *(e0_v_SecRV)) ,
    (e0_c_SecR_c1) *(e0_V_LSecR)-(e0_HU_LSecR_c1) ,
    (e0_c_SecR_c2) *(e0_V_LSecR)-(e0_HU_LSecR_c2) ,
    (e0_F_LSecR) *(e0_HU_LSecR_c1)-((e0_F_LSecR_c1) *(e0_HU_LSecR)) ,
    (e0_F_LSecR) *(e0_HU_LSecR_c2)-((e0_F_LSecR_c2) *(e0_HU_LSecR)) ,
    (e0_F_LSecR) *(e0_HU_LSecR_c3)-((e0_F_LSecR_c3) *(e0_HU_LSecR)) ,
    (e0_F_LSecR) *(e0_HU_LSecR_c4)-((e0_F_LSecR_c4) *(e0_HU_LSecR)) ,
    (e0_F_LSecR) *(e0_HU_LSecR_c5)-((e0_F_LSecR_c5) *(e0_HU_LSecR)) ,
    (e0_F_SecRV) *(e0_HU_SecRV_c1)-((e0_F_SecRV_c1) *(e0_HU_SecRV)) ,
    (e0_F_SecRV) *(e0_HU_SecRV_c2)-((e0_F_SecRV_c2) *(e0_HU_SecRV)) ,
    (e0_F_SecRV) *(e0_HU_SecRV_c3)-((e0_F_SecRV_c3) *(e0_HU_SecRV)) ,
    (e0_F_SecRV) *(e0_HU_SecRV_c4)-((e0_F_SecRV_c4) *(e0_HU_SecRV)) ,
    (e0_F_SecRV) *(e0_HU_SecRV_c5)-((e0_F_SecRV_c5) *(e0_HU_SecRV)) ,
    e0_r_SecR_r1-((e0_k_r1) *((sympy.exp(-(e0_E_r1)/(e0_T_SecR))) *((e0_c_SecR_c1) *(e0_c_SecR_c2)))) ,
    e0_r_SecR_r2-((e0_k_r2) *((sympy.exp(-(e0_E_r2)/(e0_T_SecR))) *((e0_c_SecR_c1) *(((e0_c_SecR_c2))**(2.0))))) ,
    e0_P_SecR-(e0_P_SecRtop+e0_greek_DeltaP_SecR) ,
    e0_greek_DeltaP_SecR-((e0_greek_rho_LSecR) *((e0_g) *((e0_greek_epsiv_LSecR) *((e0_LSecR) *(((10.0))**(-5.0)))))) ,
    (e0_greek_rho_LSecR) *(e0_v_LSecR)-(e0_M_LSecR) ,
    (e0_greek_rho_SecRV) *(e0_v_SecRV)-(e0_M_SecRV) ,
    (e0_F_LSecR) *(e0_M_LSecR)-(((e0_F_LSecR_c1) *(e0_M_c1)+(e0_F_LSecR_c2) *(e0_M_c2)+(e0_F_LSecR_c3) *(e0_M_c3)+(e0_F_LSecR_c4) *(e0_M_c4)+(e0_F_LSecR_c5) *(e0_M_c5))) ,
    (e0_F_SecRV) *(e0_M_SecRV)-(((e0_F_SecRV_c1) *(e0_M_c1)+(e0_F_SecRV_c2) *(e0_M_c2)+(e0_F_SecRV_c3) *(e0_M_c3)+(e0_F_SecRV_c4) *(e0_M_c4)+(e0_F_SecRV_c5) *(e0_M_c5))) ,
    e0_greek_phiv_LSecR_c1-((sympy.exp(sympy.log((e0_v_LSecR)/(e0_v_LSecR-e0_b_LSecR)))) *((sympy.exp((e0_b_c1)/(e0_v_LSecR-e0_b_LSecR))) *((sympy.exp(((1.0)/(((2.0)**(1/(2.0))) *((e0_b_LSecR) *((e0_R) *(e0_T_SecR))))) *((sympy.log((e0_v_LSecR+((1.0+(2.0)**(1/(2.0)))) *(e0_b_LSecR))/(e0_v_LSecR+((1.0-(2.0)**(1/(2.0)))) *(e0_b_LSecR)))) *((((e0_a_LSecR) *(e0_b_c1))/((2.0) *(e0_b_LSecR))-(((e0_F_LSecR_c1)/(e0_F_LSecR)) *(((2.0)**(1/((e0_greek_alpha_SecR_c1) *((e0_a_c1) *((e0_greek_alpha_SecR_c1) *(e0_a_c1)))))) *((1.0-e0_greek_delta_c1_caux1)))+((e0_F_LSecR_c2)/(e0_F_LSecR)) *(((2.0)**(1/((e0_greek_alpha_SecR_c1) *((e0_a_c1) *((e0_greek_alpha_SecR_c2) *(e0_a_c2)))))) *((1.0-e0_greek_delta_c1_caux2)))+((e0_F_LSecR_c3)/(e0_F_LSecR)) *(((2.0)**(1/((e0_greek_alpha_SecR_c1) *((e0_a_c1) *((e0_greek_alpha_SecR_c3) *(e0_a_c3)))))) *((1.0-e0_greek_delta_c1_caux3)))+((e0_F_LSecR_c4)/(e0_F_LSecR)) *(((2.0)**(1/((e0_greek_alpha_SecR_c1) *((e0_a_c1) *((e0_greek_alpha_SecR_c4) *(e0_a_c4)))))) *((1.0-e0_greek_delta_c1_caux4)))+((e0_F_LSecR_c5)/(e0_F_LSecR)) *(((2.0)**(1/((e0_greek_alpha_SecR_c1) *((e0_a_c1) *((e0_greek_alpha_SecR_c5) *(e0_a_c5)))))) *((1.0-e0_greek_delta_c1_caux5))))))))) *((sympy.exp(-((e0_a_LSecR)/((e0_b_LSecR) *((e0_R) *(e0_T_SecR)))) *(((e0_b_c1) *(e0_v_LSecR))/(((e0_v_LSecR))**(2.0)+(2.0) *((e0_b_LSecR) *(e0_v_LSecR))-((e0_b_LSecR))**(2.0))))) *(sympy.exp(-sympy.log(((e0_P_SecR) *((((10.0))**(5.0)) *(e0_v_LSecR)))/((e0_R) *(e0_T_SecR))))))))) ,
    e0_greek_phiv_LSecR_c2-((sympy.exp(sympy.log((e0_v_LSecR)/(e0_v_LSecR-e0_b_LSecR)))) *((sympy.exp((e0_b_c2)/(e0_v_LSecR-e0_b_LSecR))) *((sympy.exp(((1.0)/(((2.0)**(1/(2.0))) *((e0_b_LSecR) *((e0_R) *(e0_T_SecR))))) *((sympy.log((e0_v_LSecR+((1.0+(2.0)**(1/(2.0)))) *(e0_b_LSecR))/(e0_v_LSecR+((1.0-(2.0)**(1/(2.0)))) *(e0_b_LSecR)))) *((((e0_a_LSecR) *(e0_b_c2))/((2.0) *(e0_b_LSecR))-(((e0_F_LSecR_c1)/(e0_F_LSecR)) *(((2.0)**(1/((e0_greek_alpha_SecR_c2) *((e0_a_c2) *((e0_greek_alpha_SecR_c1) *(e0_a_c1)))))) *((1.0-e0_greek_delta_c2_caux1)))+((e0_F_LSecR_c2)/(e0_F_LSecR)) *(((2.0)**(1/((e0_greek_alpha_SecR_c2) *((e0_a_c2) *((e0_greek_alpha_SecR_c2) *(e0_a_c2)))))) *((1.0-e0_greek_delta_c2_caux2)))+((e0_F_LSecR_c3)/(e0_F_LSecR)) *(((2.0)**(1/((e0_greek_alpha_SecR_c2) *((e0_a_c2) *((e0_greek_alpha_SecR_c3) *(e0_a_c3)))))) *((1.0-e0_greek_delta_c2_caux3)))+((e0_F_LSecR_c4)/(e0_F_LSecR)) *(((2.0)**(1/((e0_greek_alpha_SecR_c2) *((e0_a_c2) *((e0_greek_alpha_SecR_c4) *(e0_a_c4)))))) *((1.0-e0_greek_delta_c2_caux4)))+((e0_F_LSecR_c5)/(e0_F_LSecR)) *(((2.0)**(1/((e0_greek_alpha_SecR_c2) *((e0_a_c2) *((e0_greek_alpha_SecR_c5) *(e0_a_c5)))))) *((1.0-e0_greek_delta_c2_caux5))))))))) *((sympy.exp(-((e0_a_LSecR)/((e0_b_LSecR) *((e0_R) *(e0_T_SecR)))) *(((e0_b_c2) *(e0_v_LSecR))/(((e0_v_LSecR))**(2.0)+(2.0) *((e0_b_LSecR) *(e0_v_LSecR))-((e0_b_LSecR))**(2.0))))) *(sympy.exp(-sympy.log(((e0_P_SecR) *((((10.0))**(5.0)) *(e0_v_LSecR)))/((e0_R) *(e0_T_SecR))))))))) ,
    e0_greek_phiv_LSecR_c3-((sympy.exp(sympy.log((e0_v_LSecR)/(e0_v_LSecR-e0_b_LSecR)))) *((sympy.exp((e0_b_c3)/(e0_v_LSecR-e0_b_LSecR))) *((sympy.exp(((1.0)/(((2.0)**(1/(2.0))) *((e0_b_LSecR) *((e0_R) *(e0_T_SecR))))) *((sympy.log((e0_v_LSecR+((1.0+(2.0)**(1/(2.0)))) *(e0_b_LSecR))/(e0_v_LSecR+((1.0-(2.0)**(1/(2.0)))) *(e0_b_LSecR)))) *((((e0_a_LSecR) *(e0_b_c3))/((2.0) *(e0_b_LSecR))-(((e0_F_LSecR_c1)/(e0_F_LSecR)) *(((2.0)**(1/((e0_greek_alpha_SecR_c3) *((e0_a_c3) *((e0_greek_alpha_SecR_c1) *(e0_a_c1)))))) *((1.0-e0_greek_delta_c3_caux1)))+((e0_F_LSecR_c2)/(e0_F_LSecR)) *(((2.0)**(1/((e0_greek_alpha_SecR_c3) *((e0_a_c3) *((e0_greek_alpha_SecR_c2) *(e0_a_c2)))))) *((1.0-e0_greek_delta_c3_caux2)))+((e0_F_LSecR_c3)/(e0_F_LSecR)) *(((2.0)**(1/((e0_greek_alpha_SecR_c3) *((e0_a_c3) *((e0_greek_alpha_SecR_c3) *(e0_a_c3)))))) *((1.0-e0_greek_delta_c3_caux3)))+((e0_F_LSecR_c4)/(e0_F_LSecR)) *(((2.0)**(1/((e0_greek_alpha_SecR_c3) *((e0_a_c3) *((e0_greek_alpha_SecR_c4) *(e0_a_c4)))))) *((1.0-e0_greek_delta_c3_caux4)))+((e0_F_LSecR_c5)/(e0_F_LSecR)) *(((2.0)**(1/((e0_greek_alpha_SecR_c3) *((e0_a_c3) *((e0_greek_alpha_SecR_c5) *(e0_a_c5)))))) *((1.0-e0_greek_delta_c3_caux5))))))))) *((sympy.exp(-((e0_a_LSecR)/((e0_b_LSecR) *((e0_R) *(e0_T_SecR)))) *(((e0_b_c3) *(e0_v_LSecR))/(((e0_v_LSecR))**(2.0)+(2.0) *((e0_b_LSecR) *(e0_v_LSecR))-((e0_b_LSecR))**(2.0))))) *(sympy.exp(-sympy.log(((e0_P_SecR) *((((10.0))**(5.0)) *(e0_v_LSecR)))/((e0_R) *(e0_T_SecR))))))))) ,
    e0_greek_phiv_LSecR_c4-((sympy.exp(sympy.log((e0_v_LSecR)/(e0_v_LSecR-e0_b_LSecR)))) *((sympy.exp((e0_b_c4)/(e0_v_LSecR-e0_b_LSecR))) *((sympy.exp(((1.0)/(((2.0)**(1/(2.0))) *((e0_b_LSecR) *((e0_R) *(e0_T_SecR))))) *((sympy.log((e0_v_LSecR+((1.0+(2.0)**(1/(2.0)))) *(e0_b_LSecR))/(e0_v_LSecR+((1.0-(2.0)**(1/(2.0)))) *(e0_b_LSecR)))) *((((e0_a_LSecR) *(e0_b_c4))/((2.0) *(e0_b_LSecR))-(((e0_F_LSecR_c1)/(e0_F_LSecR)) *(((2.0)**(1/((e0_greek_alpha_SecR_c4) *((e0_a_c4) *((e0_greek_alpha_SecR_c1) *(e0_a_c1)))))) *((1.0-e0_greek_delta_c4_caux1)))+((e0_F_LSecR_c2)/(e0_F_LSecR)) *(((2.0)**(1/((e0_greek_alpha_SecR_c4) *((e0_a_c4) *((e0_greek_alpha_SecR_c2) *(e0_a_c2)))))) *((1.0-e0_greek_delta_c4_caux2)))+((e0_F_LSecR_c3)/(e0_F_LSecR)) *(((2.0)**(1/((e0_greek_alpha_SecR_c4) *((e0_a_c4) *((e0_greek_alpha_SecR_c3) *(e0_a_c3)))))) *((1.0-e0_greek_delta_c4_caux3)))+((e0_F_LSecR_c4)/(e0_F_LSecR)) *(((2.0)**(1/((e0_greek_alpha_SecR_c4) *((e0_a_c4) *((e0_greek_alpha_SecR_c4) *(e0_a_c4)))))) *((1.0-e0_greek_delta_c4_caux4)))+((e0_F_LSecR_c5)/(e0_F_LSecR)) *(((2.0)**(1/((e0_greek_alpha_SecR_c4) *((e0_a_c4) *((e0_greek_alpha_SecR_c5) *(e0_a_c5)))))) *((1.0-e0_greek_delta_c4_caux5))))))))) *((sympy.exp(-((e0_a_LSecR)/((e0_b_LSecR) *((e0_R) *(e0_T_SecR)))) *(((e0_b_c4) *(e0_v_LSecR))/(((e0_v_LSecR))**(2.0)+(2.0) *((e0_b_LSecR) *(e0_v_LSecR))-((e0_b_LSecR))**(2.0))))) *(sympy.exp(-sympy.log(((e0_P_SecR) *((((10.0))**(5.0)) *(e0_v_LSecR)))/((e0_R) *(e0_T_SecR))))))))) ,
    e0_greek_phiv_LSecR_c5-((sympy.exp(sympy.log((e0_v_LSecR)/(e0_v_LSecR-e0_b_LSecR)))) *((sympy.exp((e0_b_c5)/(e0_v_LSecR-e0_b_LSecR))) *((sympy.exp(((1.0)/(((2.0)**(1/(2.0))) *((e0_b_LSecR) *((e0_R) *(e0_T_SecR))))) *((sympy.log((e0_v_LSecR+((1.0+(2.0)**(1/(2.0)))) *(e0_b_LSecR))/(e0_v_LSecR+((1.0-(2.0)**(1/(2.0)))) *(e0_b_LSecR)))) *((((e0_a_LSecR) *(e0_b_c5))/((2.0) *(e0_b_LSecR))-(((e0_F_LSecR_c1)/(e0_F_LSecR)) *(((2.0)**(1/((e0_greek_alpha_SecR_c5) *((e0_a_c5) *((e0_greek_alpha_SecR_c1) *(e0_a_c1)))))) *((1.0-e0_greek_delta_c5_caux1)))+((e0_F_LSecR_c2)/(e0_F_LSecR)) *(((2.0)**(1/((e0_greek_alpha_SecR_c5) *((e0_a_c5) *((e0_greek_alpha_SecR_c2) *(e0_a_c2)))))) *((1.0-e0_greek_delta_c5_caux2)))+((e0_F_LSecR_c3)/(e0_F_LSecR)) *(((2.0)**(1/((e0_greek_alpha_SecR_c5) *((e0_a_c5) *((e0_greek_alpha_SecR_c3) *(e0_a_c3)))))) *((1.0-e0_greek_delta_c5_caux3)))+((e0_F_LSecR_c4)/(e0_F_LSecR)) *(((2.0)**(1/((e0_greek_alpha_SecR_c5) *((e0_a_c5) *((e0_greek_alpha_SecR_c4) *(e0_a_c4)))))) *((1.0-e0_greek_delta_c5_caux4)))+((e0_F_LSecR_c5)/(e0_F_LSecR)) *(((2.0)**(1/((e0_greek_alpha_SecR_c5) *((e0_a_c5) *((e0_greek_alpha_SecR_c5) *(e0_a_c5)))))) *((1.0-e0_greek_delta_c5_caux5))))))))) *((sympy.exp(-((e0_a_LSecR)/((e0_b_LSecR) *((e0_R) *(e0_T_SecR)))) *(((e0_b_c5) *(e0_v_LSecR))/(((e0_v_LSecR))**(2.0)+(2.0) *((e0_b_LSecR) *(e0_v_LSecR))-((e0_b_LSecR))**(2.0))))) *(sympy.exp(-sympy.log(((e0_P_SecR) *((((10.0))**(5.0)) *(e0_v_LSecR)))/((e0_R) *(e0_T_SecR))))))))) ,
    e0_greek_phiv_SecRV_c1-((sympy.exp(sympy.log((e0_v_SecRV)/(e0_v_SecRV-e0_b_SecRV)))) *((sympy.exp((e0_b_c1)/(e0_v_SecRV-e0_b_SecRV))) *((sympy.exp(((1.0)/(((2.0)**(1/(2.0))) *((e0_b_SecRV) *((e0_R) *(e0_T_SecR))))) *((sympy.log((e0_v_SecRV+((1.0+(2.0)**(1/(2.0)))) *(e0_b_SecRV))/(e0_v_SecRV+((1.0-(2.0)**(1/(2.0)))) *(e0_b_SecRV)))) *((((e0_a_SecRV) *(e0_b_c1))/((2.0) *(e0_b_SecRV))-(((e0_F_SecRV_c1)/(e0_F_SecRV)) *(((2.0)**(1/((e0_greek_alpha_SecR_c1) *((e0_a_c1) *((e0_greek_alpha_SecR_c1) *(e0_a_c1)))))) *((1.0-e0_greek_delta_c1_caux1)))+((e0_F_SecRV_c2)/(e0_F_SecRV)) *(((2.0)**(1/((e0_greek_alpha_SecR_c1) *((e0_a_c1) *((e0_greek_alpha_SecR_c2) *(e0_a_c2)))))) *((1.0-e0_greek_delta_c1_caux2)))+((e0_F_SecRV_c3)/(e0_F_SecRV)) *(((2.0)**(1/((e0_greek_alpha_SecR_c1) *((e0_a_c1) *((e0_greek_alpha_SecR_c3) *(e0_a_c3)))))) *((1.0-e0_greek_delta_c1_caux3)))+((e0_F_SecRV_c4)/(e0_F_SecRV)) *(((2.0)**(1/((e0_greek_alpha_SecR_c1) *((e0_a_c1) *((e0_greek_alpha_SecR_c4) *(e0_a_c4)))))) *((1.0-e0_greek_delta_c1_caux4)))+((e0_F_SecRV_c5)/(e0_F_SecRV)) *(((2.0)**(1/((e0_greek_alpha_SecR_c1) *((e0_a_c1) *((e0_greek_alpha_SecR_c5) *(e0_a_c5)))))) *((1.0-e0_greek_delta_c1_caux5))))))))) *((sympy.exp(-((e0_a_SecRV)/((e0_b_SecRV) *((e0_R) *(e0_T_SecR)))) *(((e0_b_c1) *(e0_v_SecRV))/(((e0_v_SecRV))**(2.0)+(2.0) *((e0_b_SecRV) *(e0_v_SecRV))-((e0_b_SecRV))**(2.0))))) *(sympy.exp(-sympy.log(((e0_P_SecR) *((((10.0))**(5.0)) *(e0_v_SecRV)))/((e0_R) *(e0_T_SecR))))))))) ,
    e0_greek_phiv_SecRV_c2-((sympy.exp(sympy.log((e0_v_SecRV)/(e0_v_SecRV-e0_b_SecRV)))) *((sympy.exp((e0_b_c2)/(e0_v_SecRV-e0_b_SecRV))) *((sympy.exp(((1.0)/(((2.0)**(1/(2.0))) *((e0_b_SecRV) *((e0_R) *(e0_T_SecR))))) *((sympy.log((e0_v_SecRV+((1.0+(2.0)**(1/(2.0)))) *(e0_b_SecRV))/(e0_v_SecRV+((1.0-(2.0)**(1/(2.0)))) *(e0_b_SecRV)))) *((((e0_a_SecRV) *(e0_b_c2))/((2.0) *(e0_b_SecRV))-(((e0_F_SecRV_c1)/(e0_F_SecRV)) *(((2.0)**(1/((e0_greek_alpha_SecR_c2) *((e0_a_c2) *((e0_greek_alpha_SecR_c1) *(e0_a_c1)))))) *((1.0-e0_greek_delta_c2_caux1)))+((e0_F_SecRV_c2)/(e0_F_SecRV)) *(((2.0)**(1/((e0_greek_alpha_SecR_c2) *((e0_a_c2) *((e0_greek_alpha_SecR_c2) *(e0_a_c2)))))) *((1.0-e0_greek_delta_c2_caux2)))+((e0_F_SecRV_c3)/(e0_F_SecRV)) *(((2.0)**(1/((e0_greek_alpha_SecR_c2) *((e0_a_c2) *((e0_greek_alpha_SecR_c3) *(e0_a_c3)))))) *((1.0-e0_greek_delta_c2_caux3)))+((e0_F_SecRV_c4)/(e0_F_SecRV)) *(((2.0)**(1/((e0_greek_alpha_SecR_c2) *((e0_a_c2) *((e0_greek_alpha_SecR_c4) *(e0_a_c4)))))) *((1.0-e0_greek_delta_c2_caux4)))+((e0_F_SecRV_c5)/(e0_F_SecRV)) *(((2.0)**(1/((e0_greek_alpha_SecR_c2) *((e0_a_c2) *((e0_greek_alpha_SecR_c5) *(e0_a_c5)))))) *((1.0-e0_greek_delta_c2_caux5))))))))) *((sympy.exp(-((e0_a_SecRV)/((e0_b_SecRV) *((e0_R) *(e0_T_SecR)))) *(((e0_b_c2) *(e0_v_SecRV))/(((e0_v_SecRV))**(2.0)+(2.0) *((e0_b_SecRV) *(e0_v_SecRV))-((e0_b_SecRV))**(2.0))))) *(sympy.exp(-sympy.log(((e0_P_SecR) *((((10.0))**(5.0)) *(e0_v_SecRV)))/((e0_R) *(e0_T_SecR))))))))) ,
    e0_greek_phiv_SecRV_c3-((sympy.exp(sympy.log((e0_v_SecRV)/(e0_v_SecRV-e0_b_SecRV)))) *((sympy.exp((e0_b_c3)/(e0_v_SecRV-e0_b_SecRV))) *((sympy.exp(((1.0)/(((2.0)**(1/(2.0))) *((e0_b_SecRV) *((e0_R) *(e0_T_SecR))))) *((sympy.log((e0_v_SecRV+((1.0+(2.0)**(1/(2.0)))) *(e0_b_SecRV))/(e0_v_SecRV+((1.0-(2.0)**(1/(2.0)))) *(e0_b_SecRV)))) *((((e0_a_SecRV) *(e0_b_c3))/((2.0) *(e0_b_SecRV))-(((e0_F_SecRV_c1)/(e0_F_SecRV)) *(((2.0)**(1/((e0_greek_alpha_SecR_c3) *((e0_a_c3) *((e0_greek_alpha_SecR_c1) *(e0_a_c1)))))) *((1.0-e0_greek_delta_c3_caux1)))+((e0_F_SecRV_c2)/(e0_F_SecRV)) *(((2.0)**(1/((e0_greek_alpha_SecR_c3) *((e0_a_c3) *((e0_greek_alpha_SecR_c2) *(e0_a_c2)))))) *((1.0-e0_greek_delta_c3_caux2)))+((e0_F_SecRV_c3)/(e0_F_SecRV)) *(((2.0)**(1/((e0_greek_alpha_SecR_c3) *((e0_a_c3) *((e0_greek_alpha_SecR_c3) *(e0_a_c3)))))) *((1.0-e0_greek_delta_c3_caux3)))+((e0_F_SecRV_c4)/(e0_F_SecRV)) *(((2.0)**(1/((e0_greek_alpha_SecR_c3) *((e0_a_c3) *((e0_greek_alpha_SecR_c4) *(e0_a_c4)))))) *((1.0-e0_greek_delta_c3_caux4)))+((e0_F_SecRV_c5)/(e0_F_SecRV)) *(((2.0)**(1/((e0_greek_alpha_SecR_c3) *((e0_a_c3) *((e0_greek_alpha_SecR_c5) *(e0_a_c5)))))) *((1.0-e0_greek_delta_c3_caux5))))))))) *((sympy.exp(-((e0_a_SecRV)/((e0_b_SecRV) *((e0_R) *(e0_T_SecR)))) *(((e0_b_c3) *(e0_v_SecRV))/(((e0_v_SecRV))**(2.0)+(2.0) *((e0_b_SecRV) *(e0_v_SecRV))-((e0_b_SecRV))**(2.0))))) *(sympy.exp(-sympy.log(((e0_P_SecR) *((((10.0))**(5.0)) *(e0_v_SecRV)))/((e0_R) *(e0_T_SecR))))))))) ,
    e0_greek_phiv_SecRV_c4-((sympy.exp(sympy.log((e0_v_SecRV)/(e0_v_SecRV-e0_b_SecRV)))) *((sympy.exp((e0_b_c4)/(e0_v_SecRV-e0_b_SecRV))) *((sympy.exp(((1.0)/(((2.0)**(1/(2.0))) *((e0_b_SecRV) *((e0_R) *(e0_T_SecR))))) *((sympy.log((e0_v_SecRV+((1.0+(2.0)**(1/(2.0)))) *(e0_b_SecRV))/(e0_v_SecRV+((1.0-(2.0)**(1/(2.0)))) *(e0_b_SecRV)))) *((((e0_a_SecRV) *(e0_b_c4))/((2.0) *(e0_b_SecRV))-(((e0_F_SecRV_c1)/(e0_F_SecRV)) *(((2.0)**(1/((e0_greek_alpha_SecR_c4) *((e0_a_c4) *((e0_greek_alpha_SecR_c1) *(e0_a_c1)))))) *((1.0-e0_greek_delta_c4_caux1)))+((e0_F_SecRV_c2)/(e0_F_SecRV)) *(((2.0)**(1/((e0_greek_alpha_SecR_c4) *((e0_a_c4) *((e0_greek_alpha_SecR_c2) *(e0_a_c2)))))) *((1.0-e0_greek_delta_c4_caux2)))+((e0_F_SecRV_c3)/(e0_F_SecRV)) *(((2.0)**(1/((e0_greek_alpha_SecR_c4) *((e0_a_c4) *((e0_greek_alpha_SecR_c3) *(e0_a_c3)))))) *((1.0-e0_greek_delta_c4_caux3)))+((e0_F_SecRV_c4)/(e0_F_SecRV)) *(((2.0)**(1/((e0_greek_alpha_SecR_c4) *((e0_a_c4) *((e0_greek_alpha_SecR_c4) *(e0_a_c4)))))) *((1.0-e0_greek_delta_c4_caux4)))+((e0_F_SecRV_c5)/(e0_F_SecRV)) *(((2.0)**(1/((e0_greek_alpha_SecR_c4) *((e0_a_c4) *((e0_greek_alpha_SecR_c5) *(e0_a_c5)))))) *((1.0-e0_greek_delta_c4_caux5))))))))) *((sympy.exp(-((e0_a_SecRV)/((e0_b_SecRV) *((e0_R) *(e0_T_SecR)))) *(((e0_b_c4) *(e0_v_SecRV))/(((e0_v_SecRV))**(2.0)+(2.0) *((e0_b_SecRV) *(e0_v_SecRV))-((e0_b_SecRV))**(2.0))))) *(sympy.exp(-sympy.log(((e0_P_SecR) *((((10.0))**(5.0)) *(e0_v_SecRV)))/((e0_R) *(e0_T_SecR))))))))) ,
    e0_greek_phiv_SecRV_c5-((sympy.exp(sympy.log((e0_v_SecRV)/(e0_v_SecRV-e0_b_SecRV)))) *((sympy.exp((e0_b_c5)/(e0_v_SecRV-e0_b_SecRV))) *((sympy.exp(((1.0)/(((2.0)**(1/(2.0))) *((e0_b_SecRV) *((e0_R) *(e0_T_SecR))))) *((sympy.log((e0_v_SecRV+((1.0+(2.0)**(1/(2.0)))) *(e0_b_SecRV))/(e0_v_SecRV+((1.0-(2.0)**(1/(2.0)))) *(e0_b_SecRV)))) *((((e0_a_SecRV) *(e0_b_c5))/((2.0) *(e0_b_SecRV))-(((e0_F_SecRV_c1)/(e0_F_SecRV)) *(((2.0)**(1/((e0_greek_alpha_SecR_c5) *((e0_a_c5) *((e0_greek_alpha_SecR_c1) *(e0_a_c1)))))) *((1.0-e0_greek_delta_c5_caux1)))+((e0_F_SecRV_c2)/(e0_F_SecRV)) *(((2.0)**(1/((e0_greek_alpha_SecR_c5) *((e0_a_c5) *((e0_greek_alpha_SecR_c2) *(e0_a_c2)))))) *((1.0-e0_greek_delta_c5_caux2)))+((e0_F_SecRV_c3)/(e0_F_SecRV)) *(((2.0)**(1/((e0_greek_alpha_SecR_c5) *((e0_a_c5) *((e0_greek_alpha_SecR_c3) *(e0_a_c3)))))) *((1.0-e0_greek_delta_c5_caux3)))+((e0_F_SecRV_c4)/(e0_F_SecRV)) *(((2.0)**(1/((e0_greek_alpha_SecR_c5) *((e0_a_c5) *((e0_greek_alpha_SecR_c4) *(e0_a_c4)))))) *((1.0-e0_greek_delta_c5_caux4)))+((e0_F_SecRV_c5)/(e0_F_SecRV)) *(((2.0)**(1/((e0_greek_alpha_SecR_c5) *((e0_a_c5) *((e0_greek_alpha_SecR_c5) *(e0_a_c5)))))) *((1.0-e0_greek_delta_c5_caux5))))))))) *((sympy.exp(-((e0_a_SecRV)/((e0_b_SecRV) *((e0_R) *(e0_T_SecR)))) *(((e0_b_c5) *(e0_v_SecRV))/(((e0_v_SecRV))**(2.0)+(2.0) *((e0_b_SecRV) *(e0_v_SecRV))-((e0_b_SecRV))**(2.0))))) *(sympy.exp(-sympy.log(((e0_P_SecR) *((((10.0))**(5.0)) *(e0_v_SecRV)))/((e0_R) *(e0_T_SecR))))))))) ,
    0.0-((e0_P_SecR) *((((10.0))**(5.0)) *(((((e0_v_LSecR))**(2.0)+(2.0) *((e0_b_LSecR) *(e0_v_LSecR))-((e0_b_LSecR))**(2.0))) *((e0_v_LSecR-e0_b_LSecR))))-((e0_R) *((e0_T_SecR) *((((e0_v_LSecR))**(2.0)+(2.0) *((e0_b_LSecR) *(e0_v_LSecR))-((e0_b_LSecR))**(2.0))))-(e0_a_LSecR) *((e0_v_LSecR-e0_b_LSecR)))) ,
    0.0-((e0_P_SecR) *((((10.0))**(5.0)) *(((((e0_v_SecRV))**(2.0)+(2.0) *((e0_b_SecRV) *(e0_v_SecRV))-((e0_b_SecRV))**(2.0))) *((e0_v_SecRV-e0_b_SecRV))))-((e0_R) *((e0_T_SecR) *((((e0_v_SecRV))**(2.0)+(2.0) *((e0_b_SecRV) *(e0_v_SecRV))-((e0_b_SecRV))**(2.0))))-(e0_a_SecRV) *((e0_v_SecRV-e0_b_SecRV)))) ,
    e0_a_LSecR-(((1.0)/(((e0_HU_LSecR))**(2.0))) *((((e0_HU_LSecR_c1) *((e0_HU_LSecR_c1) *(((2.0)**(1/((e0_greek_alpha_SecR_c1) *((e0_a_c1) *((e0_greek_alpha_SecR_c1) *(e0_a_c1)))))) *((1.0-e0_greek_delta_c1_caux1))))+(e0_HU_LSecR_c1) *((e0_HU_LSecR_c2) *(((2.0)**(1/((e0_greek_alpha_SecR_c1) *((e0_a_c1) *((e0_greek_alpha_SecR_c2) *(e0_a_c2)))))) *((1.0-e0_greek_delta_c1_caux2))))+(e0_HU_LSecR_c1) *((e0_HU_LSecR_c3) *(((2.0)**(1/((e0_greek_alpha_SecR_c1) *((e0_a_c1) *((e0_greek_alpha_SecR_c3) *(e0_a_c3)))))) *((1.0-e0_greek_delta_c1_caux3))))+(e0_HU_LSecR_c1) *((e0_HU_LSecR_c4) *(((2.0)**(1/((e0_greek_alpha_SecR_c1) *((e0_a_c1) *((e0_greek_alpha_SecR_c4) *(e0_a_c4)))))) *((1.0-e0_greek_delta_c1_caux4))))+(e0_HU_LSecR_c1) *((e0_HU_LSecR_c5) *(((2.0)**(1/((e0_greek_alpha_SecR_c1) *((e0_a_c1) *((e0_greek_alpha_SecR_c5) *(e0_a_c5)))))) *((1.0-e0_greek_delta_c1_caux5)))))+((e0_HU_LSecR_c2) *((e0_HU_LSecR_c1) *(((2.0)**(1/((e0_greek_alpha_SecR_c2) *((e0_a_c2) *((e0_greek_alpha_SecR_c1) *(e0_a_c1)))))) *((1.0-e0_greek_delta_c2_caux1))))+(e0_HU_LSecR_c2) *((e0_HU_LSecR_c2) *(((2.0)**(1/((e0_greek_alpha_SecR_c2) *((e0_a_c2) *((e0_greek_alpha_SecR_c2) *(e0_a_c2)))))) *((1.0-e0_greek_delta_c2_caux2))))+(e0_HU_LSecR_c2) *((e0_HU_LSecR_c3) *(((2.0)**(1/((e0_greek_alpha_SecR_c2) *((e0_a_c2) *((e0_greek_alpha_SecR_c3) *(e0_a_c3)))))) *((1.0-e0_greek_delta_c2_caux3))))+(e0_HU_LSecR_c2) *((e0_HU_LSecR_c4) *(((2.0)**(1/((e0_greek_alpha_SecR_c2) *((e0_a_c2) *((e0_greek_alpha_SecR_c4) *(e0_a_c4)))))) *((1.0-e0_greek_delta_c2_caux4))))+(e0_HU_LSecR_c2) *((e0_HU_LSecR_c5) *(((2.0)**(1/((e0_greek_alpha_SecR_c2) *((e0_a_c2) *((e0_greek_alpha_SecR_c5) *(e0_a_c5)))))) *((1.0-e0_greek_delta_c2_caux5)))))+((e0_HU_LSecR_c3) *((e0_HU_LSecR_c1) *(((2.0)**(1/((e0_greek_alpha_SecR_c3) *((e0_a_c3) *((e0_greek_alpha_SecR_c1) *(e0_a_c1)))))) *((1.0-e0_greek_delta_c3_caux1))))+(e0_HU_LSecR_c3) *((e0_HU_LSecR_c2) *(((2.0)**(1/((e0_greek_alpha_SecR_c3) *((e0_a_c3) *((e0_greek_alpha_SecR_c2) *(e0_a_c2)))))) *((1.0-e0_greek_delta_c3_caux2))))+(e0_HU_LSecR_c3) *((e0_HU_LSecR_c3) *(((2.0)**(1/((e0_greek_alpha_SecR_c3) *((e0_a_c3) *((e0_greek_alpha_SecR_c3) *(e0_a_c3)))))) *((1.0-e0_greek_delta_c3_caux3))))+(e0_HU_LSecR_c3) *((e0_HU_LSecR_c4) *(((2.0)**(1/((e0_greek_alpha_SecR_c3) *((e0_a_c3) *((e0_greek_alpha_SecR_c4) *(e0_a_c4)))))) *((1.0-e0_greek_delta_c3_caux4))))+(e0_HU_LSecR_c3) *((e0_HU_LSecR_c5) *(((2.0)**(1/((e0_greek_alpha_SecR_c3) *((e0_a_c3) *((e0_greek_alpha_SecR_c5) *(e0_a_c5)))))) *((1.0-e0_greek_delta_c3_caux5)))))+((e0_HU_LSecR_c4) *((e0_HU_LSecR_c1) *(((2.0)**(1/((e0_greek_alpha_SecR_c4) *((e0_a_c4) *((e0_greek_alpha_SecR_c1) *(e0_a_c1)))))) *((1.0-e0_greek_delta_c4_caux1))))+(e0_HU_LSecR_c4) *((e0_HU_LSecR_c2) *(((2.0)**(1/((e0_greek_alpha_SecR_c4) *((e0_a_c4) *((e0_greek_alpha_SecR_c2) *(e0_a_c2)))))) *((1.0-e0_greek_delta_c4_caux2))))+(e0_HU_LSecR_c4) *((e0_HU_LSecR_c3) *(((2.0)**(1/((e0_greek_alpha_SecR_c4) *((e0_a_c4) *((e0_greek_alpha_SecR_c3) *(e0_a_c3)))))) *((1.0-e0_greek_delta_c4_caux3))))+(e0_HU_LSecR_c4) *((e0_HU_LSecR_c4) *(((2.0)**(1/((e0_greek_alpha_SecR_c4) *((e0_a_c4) *((e0_greek_alpha_SecR_c4) *(e0_a_c4)))))) *((1.0-e0_greek_delta_c4_caux4))))+(e0_HU_LSecR_c4) *((e0_HU_LSecR_c5) *(((2.0)**(1/((e0_greek_alpha_SecR_c4) *((e0_a_c4) *((e0_greek_alpha_SecR_c5) *(e0_a_c5)))))) *((1.0-e0_greek_delta_c4_caux5)))))+((e0_HU_LSecR_c5) *((e0_HU_LSecR_c1) *(((2.0)**(1/((e0_greek_alpha_SecR_c5) *((e0_a_c5) *((e0_greek_alpha_SecR_c1) *(e0_a_c1)))))) *((1.0-e0_greek_delta_c5_caux1))))+(e0_HU_LSecR_c5) *((e0_HU_LSecR_c2) *(((2.0)**(1/((e0_greek_alpha_SecR_c5) *((e0_a_c5) *((e0_greek_alpha_SecR_c2) *(e0_a_c2)))))) *((1.0-e0_greek_delta_c5_caux2))))+(e0_HU_LSecR_c5) *((e0_HU_LSecR_c3) *(((2.0)**(1/((e0_greek_alpha_SecR_c5) *((e0_a_c5) *((e0_greek_alpha_SecR_c3) *(e0_a_c3)))))) *((1.0-e0_greek_delta_c5_caux3))))+(e0_HU_LSecR_c5) *((e0_HU_LSecR_c4) *(((2.0)**(1/((e0_greek_alpha_SecR_c5) *((e0_a_c5) *((e0_greek_alpha_SecR_c4) *(e0_a_c4)))))) *((1.0-e0_greek_delta_c5_caux4))))+(e0_HU_LSecR_c5) *((e0_HU_LSecR_c5) *(((2.0)**(1/((e0_greek_alpha_SecR_c5) *((e0_a_c5) *((e0_greek_alpha_SecR_c5) *(e0_a_c5)))))) *((1.0-e0_greek_delta_c5_caux5)))))))) ,
    e0_a_SecRV-(((1.0)/(((e0_HU_SecRV))**(2.0))) *((((e0_HU_SecRV_c1) *((e0_HU_SecRV_c1) *(((2.0)**(1/((e0_greek_alpha_SecR_c1) *((e0_a_c1) *((e0_greek_alpha_SecR_c1) *(e0_a_c1)))))) *((1.0-e0_greek_delta_c1_caux1))))+(e0_HU_SecRV_c1) *((e0_HU_SecRV_c2) *(((2.0)**(1/((e0_greek_alpha_SecR_c1) *((e0_a_c1) *((e0_greek_alpha_SecR_c2) *(e0_a_c2)))))) *((1.0-e0_greek_delta_c1_caux2))))+(e0_HU_SecRV_c1) *((e0_HU_SecRV_c3) *(((2.0)**(1/((e0_greek_alpha_SecR_c1) *((e0_a_c1) *((e0_greek_alpha_SecR_c3) *(e0_a_c3)))))) *((1.0-e0_greek_delta_c1_caux3))))+(e0_HU_SecRV_c1) *((e0_HU_SecRV_c4) *(((2.0)**(1/((e0_greek_alpha_SecR_c1) *((e0_a_c1) *((e0_greek_alpha_SecR_c4) *(e0_a_c4)))))) *((1.0-e0_greek_delta_c1_caux4))))+(e0_HU_SecRV_c1) *((e0_HU_SecRV_c5) *(((2.0)**(1/((e0_greek_alpha_SecR_c1) *((e0_a_c1) *((e0_greek_alpha_SecR_c5) *(e0_a_c5)))))) *((1.0-e0_greek_delta_c1_caux5)))))+((e0_HU_SecRV_c2) *((e0_HU_SecRV_c1) *(((2.0)**(1/((e0_greek_alpha_SecR_c2) *((e0_a_c2) *((e0_greek_alpha_SecR_c1) *(e0_a_c1)))))) *((1.0-e0_greek_delta_c2_caux1))))+(e0_HU_SecRV_c2) *((e0_HU_SecRV_c2) *(((2.0)**(1/((e0_greek_alpha_SecR_c2) *((e0_a_c2) *((e0_greek_alpha_SecR_c2) *(e0_a_c2)))))) *((1.0-e0_greek_delta_c2_caux2))))+(e0_HU_SecRV_c2) *((e0_HU_SecRV_c3) *(((2.0)**(1/((e0_greek_alpha_SecR_c2) *((e0_a_c2) *((e0_greek_alpha_SecR_c3) *(e0_a_c3)))))) *((1.0-e0_greek_delta_c2_caux3))))+(e0_HU_SecRV_c2) *((e0_HU_SecRV_c4) *(((2.0)**(1/((e0_greek_alpha_SecR_c2) *((e0_a_c2) *((e0_greek_alpha_SecR_c4) *(e0_a_c4)))))) *((1.0-e0_greek_delta_c2_caux4))))+(e0_HU_SecRV_c2) *((e0_HU_SecRV_c5) *(((2.0)**(1/((e0_greek_alpha_SecR_c2) *((e0_a_c2) *((e0_greek_alpha_SecR_c5) *(e0_a_c5)))))) *((1.0-e0_greek_delta_c2_caux5)))))+((e0_HU_SecRV_c3) *((e0_HU_SecRV_c1) *(((2.0)**(1/((e0_greek_alpha_SecR_c3) *((e0_a_c3) *((e0_greek_alpha_SecR_c1) *(e0_a_c1)))))) *((1.0-e0_greek_delta_c3_caux1))))+(e0_HU_SecRV_c3) *((e0_HU_SecRV_c2) *(((2.0)**(1/((e0_greek_alpha_SecR_c3) *((e0_a_c3) *((e0_greek_alpha_SecR_c2) *(e0_a_c2)))))) *((1.0-e0_greek_delta_c3_caux2))))+(e0_HU_SecRV_c3) *((e0_HU_SecRV_c3) *(((2.0)**(1/((e0_greek_alpha_SecR_c3) *((e0_a_c3) *((e0_greek_alpha_SecR_c3) *(e0_a_c3)))))) *((1.0-e0_greek_delta_c3_caux3))))+(e0_HU_SecRV_c3) *((e0_HU_SecRV_c4) *(((2.0)**(1/((e0_greek_alpha_SecR_c3) *((e0_a_c3) *((e0_greek_alpha_SecR_c4) *(e0_a_c4)))))) *((1.0-e0_greek_delta_c3_caux4))))+(e0_HU_SecRV_c3) *((e0_HU_SecRV_c5) *(((2.0)**(1/((e0_greek_alpha_SecR_c3) *((e0_a_c3) *((e0_greek_alpha_SecR_c5) *(e0_a_c5)))))) *((1.0-e0_greek_delta_c3_caux5)))))+((e0_HU_SecRV_c4) *((e0_HU_SecRV_c1) *(((2.0)**(1/((e0_greek_alpha_SecR_c4) *((e0_a_c4) *((e0_greek_alpha_SecR_c1) *(e0_a_c1)))))) *((1.0-e0_greek_delta_c4_caux1))))+(e0_HU_SecRV_c4) *((e0_HU_SecRV_c2) *(((2.0)**(1/((e0_greek_alpha_SecR_c4) *((e0_a_c4) *((e0_greek_alpha_SecR_c2) *(e0_a_c2)))))) *((1.0-e0_greek_delta_c4_caux2))))+(e0_HU_SecRV_c4) *((e0_HU_SecRV_c3) *(((2.0)**(1/((e0_greek_alpha_SecR_c4) *((e0_a_c4) *((e0_greek_alpha_SecR_c3) *(e0_a_c3)))))) *((1.0-e0_greek_delta_c4_caux3))))+(e0_HU_SecRV_c4) *((e0_HU_SecRV_c4) *(((2.0)**(1/((e0_greek_alpha_SecR_c4) *((e0_a_c4) *((e0_greek_alpha_SecR_c4) *(e0_a_c4)))))) *((1.0-e0_greek_delta_c4_caux4))))+(e0_HU_SecRV_c4) *((e0_HU_SecRV_c5) *(((2.0)**(1/((e0_greek_alpha_SecR_c4) *((e0_a_c4) *((e0_greek_alpha_SecR_c5) *(e0_a_c5)))))) *((1.0-e0_greek_delta_c4_caux5)))))+((e0_HU_SecRV_c5) *((e0_HU_SecRV_c1) *(((2.0)**(1/((e0_greek_alpha_SecR_c5) *((e0_a_c5) *((e0_greek_alpha_SecR_c1) *(e0_a_c1)))))) *((1.0-e0_greek_delta_c5_caux1))))+(e0_HU_SecRV_c5) *((e0_HU_SecRV_c2) *(((2.0)**(1/((e0_greek_alpha_SecR_c5) *((e0_a_c5) *((e0_greek_alpha_SecR_c2) *(e0_a_c2)))))) *((1.0-e0_greek_delta_c5_caux2))))+(e0_HU_SecRV_c5) *((e0_HU_SecRV_c3) *(((2.0)**(1/((e0_greek_alpha_SecR_c5) *((e0_a_c5) *((e0_greek_alpha_SecR_c3) *(e0_a_c3)))))) *((1.0-e0_greek_delta_c5_caux3))))+(e0_HU_SecRV_c5) *((e0_HU_SecRV_c4) *(((2.0)**(1/((e0_greek_alpha_SecR_c5) *((e0_a_c5) *((e0_greek_alpha_SecR_c4) *(e0_a_c4)))))) *((1.0-e0_greek_delta_c5_caux4))))+(e0_HU_SecRV_c5) *((e0_HU_SecRV_c5) *(((2.0)**(1/((e0_greek_alpha_SecR_c5) *((e0_a_c5) *((e0_greek_alpha_SecR_c5) *(e0_a_c5)))))) *((1.0-e0_greek_delta_c5_caux5)))))))) ,
    e0_b_LSecR-(((1.0)/(e0_HU_LSecR)) *(((e0_HU_LSecR_c1) *(e0_b_c1)+(e0_HU_LSecR_c2) *(e0_b_c2)+(e0_HU_LSecR_c3) *(e0_b_c3)+(e0_HU_LSecR_c4) *(e0_b_c4)+(e0_HU_LSecR_c5) *(e0_b_c5)))) ,
    e0_b_SecRV-(((1.0)/(e0_HU_SecRV)) *(((e0_HU_SecRV_c1) *(e0_b_c1)+(e0_HU_SecRV_c2) *(e0_b_c2)+(e0_HU_SecRV_c3) *(e0_b_c3)+(e0_HU_SecRV_c4) *(e0_b_c4)+(e0_HU_SecRV_c5) *(e0_b_c5)))) ,
    e0_greek_alpha_SecR_c1-(((1.0+(e0_m_c1) *((1.0-(2.0)**(1/((e0_T_SecR)/(e0_TC_c1)))))))**(2.0)) ,
    e0_greek_alpha_SecR_c2-(((1.0+(e0_m_c2) *((1.0-(2.0)**(1/((e0_T_SecR)/(e0_TC_c2)))))))**(2.0)) ,
    e0_greek_alpha_SecR_c3-(((1.0+(e0_m_c3) *((1.0-(2.0)**(1/((e0_T_SecR)/(e0_TC_c3)))))))**(2.0)) ,
    e0_greek_alpha_SecR_c4-(((1.0+(e0_m_c4) *((1.0-(2.0)**(1/((e0_T_SecR)/(e0_TC_c4)))))))**(2.0)) ,
    e0_greek_alpha_SecR_c5-(((1.0+(e0_m_c5) *((1.0-(2.0)**(1/((e0_T_SecR)/(e0_TC_c5)))))))**(2.0)) ,
    sympy.log(e0_greek_Deltah_SecRVLdeparture)-(sympy.log(((((e0_a_LSecR-(e0_T_SecR) *((((1.0)/((2.0) *(((e0_F_LSecR))**(2.0)))) *((((e0_F_LSecR_c1) *((e0_F_LSecR_c1) *(((2.0)**(1/(((e0_a_c1) *(e0_a_c1))/((e0_greek_alpha_SecR_c1) *(e0_greek_alpha_SecR_c1))))) *(((((-e0_m_c1)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c1))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c1))) *(e0_greek_alpha_SecR_c1))+(e0_greek_alpha_SecR_c1) *(((-e0_m_c1)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c1))))) *((2.0)**(1/(e0_greek_alpha_SecR_c1)))))) *((1.0-e0_greek_delta_c1_caux1)))))+(e0_F_LSecR_c1) *((e0_F_LSecR_c2) *(((2.0)**(1/(((e0_a_c1) *(e0_a_c2))/((e0_greek_alpha_SecR_c1) *(e0_greek_alpha_SecR_c2))))) *(((((-e0_m_c1)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c1))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c1))) *(e0_greek_alpha_SecR_c2))+(e0_greek_alpha_SecR_c1) *(((-e0_m_c2)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c2))))) *((2.0)**(1/(e0_greek_alpha_SecR_c2)))))) *((1.0-e0_greek_delta_c1_caux2)))))+(e0_F_LSecR_c1) *((e0_F_LSecR_c3) *(((2.0)**(1/(((e0_a_c1) *(e0_a_c3))/((e0_greek_alpha_SecR_c1) *(e0_greek_alpha_SecR_c3))))) *(((((-e0_m_c1)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c1))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c1))) *(e0_greek_alpha_SecR_c3))+(e0_greek_alpha_SecR_c1) *(((-e0_m_c3)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c3))))) *((2.0)**(1/(e0_greek_alpha_SecR_c3)))))) *((1.0-e0_greek_delta_c1_caux3)))))+(e0_F_LSecR_c1) *((e0_F_LSecR_c4) *(((2.0)**(1/(((e0_a_c1) *(e0_a_c4))/((e0_greek_alpha_SecR_c1) *(e0_greek_alpha_SecR_c4))))) *(((((-e0_m_c1)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c1))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c1))) *(e0_greek_alpha_SecR_c4))+(e0_greek_alpha_SecR_c1) *(((-e0_m_c4)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c4))))) *((2.0)**(1/(e0_greek_alpha_SecR_c4)))))) *((1.0-e0_greek_delta_c1_caux4)))))+(e0_F_LSecR_c1) *((e0_F_LSecR_c5) *(((2.0)**(1/(((e0_a_c1) *(e0_a_c5))/((e0_greek_alpha_SecR_c1) *(e0_greek_alpha_SecR_c5))))) *(((((-e0_m_c1)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c1))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c1))) *(e0_greek_alpha_SecR_c5))+(e0_greek_alpha_SecR_c1) *(((-e0_m_c5)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c5))))) *((2.0)**(1/(e0_greek_alpha_SecR_c5)))))) *((1.0-e0_greek_delta_c1_caux5))))))+((e0_F_LSecR_c2) *((e0_F_LSecR_c1) *(((2.0)**(1/(((e0_a_c2) *(e0_a_c1))/((e0_greek_alpha_SecR_c2) *(e0_greek_alpha_SecR_c1))))) *(((((-e0_m_c2)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c2))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c2))) *(e0_greek_alpha_SecR_c1))+(e0_greek_alpha_SecR_c2) *(((-e0_m_c1)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c1))))) *((2.0)**(1/(e0_greek_alpha_SecR_c1)))))) *((1.0-e0_greek_delta_c2_caux1)))))+(e0_F_LSecR_c2) *((e0_F_LSecR_c2) *(((2.0)**(1/(((e0_a_c2) *(e0_a_c2))/((e0_greek_alpha_SecR_c2) *(e0_greek_alpha_SecR_c2))))) *(((((-e0_m_c2)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c2))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c2))) *(e0_greek_alpha_SecR_c2))+(e0_greek_alpha_SecR_c2) *(((-e0_m_c2)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c2))))) *((2.0)**(1/(e0_greek_alpha_SecR_c2)))))) *((1.0-e0_greek_delta_c2_caux2)))))+(e0_F_LSecR_c2) *((e0_F_LSecR_c3) *(((2.0)**(1/(((e0_a_c2) *(e0_a_c3))/((e0_greek_alpha_SecR_c2) *(e0_greek_alpha_SecR_c3))))) *(((((-e0_m_c2)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c2))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c2))) *(e0_greek_alpha_SecR_c3))+(e0_greek_alpha_SecR_c2) *(((-e0_m_c3)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c3))))) *((2.0)**(1/(e0_greek_alpha_SecR_c3)))))) *((1.0-e0_greek_delta_c2_caux3)))))+(e0_F_LSecR_c2) *((e0_F_LSecR_c4) *(((2.0)**(1/(((e0_a_c2) *(e0_a_c4))/((e0_greek_alpha_SecR_c2) *(e0_greek_alpha_SecR_c4))))) *(((((-e0_m_c2)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c2))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c2))) *(e0_greek_alpha_SecR_c4))+(e0_greek_alpha_SecR_c2) *(((-e0_m_c4)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c4))))) *((2.0)**(1/(e0_greek_alpha_SecR_c4)))))) *((1.0-e0_greek_delta_c2_caux4)))))+(e0_F_LSecR_c2) *((e0_F_LSecR_c5) *(((2.0)**(1/(((e0_a_c2) *(e0_a_c5))/((e0_greek_alpha_SecR_c2) *(e0_greek_alpha_SecR_c5))))) *(((((-e0_m_c2)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c2))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c2))) *(e0_greek_alpha_SecR_c5))+(e0_greek_alpha_SecR_c2) *(((-e0_m_c5)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c5))))) *((2.0)**(1/(e0_greek_alpha_SecR_c5)))))) *((1.0-e0_greek_delta_c2_caux5))))))+((e0_F_LSecR_c3) *((e0_F_LSecR_c1) *(((2.0)**(1/(((e0_a_c3) *(e0_a_c1))/((e0_greek_alpha_SecR_c3) *(e0_greek_alpha_SecR_c1))))) *(((((-e0_m_c3)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c3))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c3))) *(e0_greek_alpha_SecR_c1))+(e0_greek_alpha_SecR_c3) *(((-e0_m_c1)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c1))))) *((2.0)**(1/(e0_greek_alpha_SecR_c1)))))) *((1.0-e0_greek_delta_c3_caux1)))))+(e0_F_LSecR_c3) *((e0_F_LSecR_c2) *(((2.0)**(1/(((e0_a_c3) *(e0_a_c2))/((e0_greek_alpha_SecR_c3) *(e0_greek_alpha_SecR_c2))))) *(((((-e0_m_c3)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c3))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c3))) *(e0_greek_alpha_SecR_c2))+(e0_greek_alpha_SecR_c3) *(((-e0_m_c2)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c2))))) *((2.0)**(1/(e0_greek_alpha_SecR_c2)))))) *((1.0-e0_greek_delta_c3_caux2)))))+(e0_F_LSecR_c3) *((e0_F_LSecR_c3) *(((2.0)**(1/(((e0_a_c3) *(e0_a_c3))/((e0_greek_alpha_SecR_c3) *(e0_greek_alpha_SecR_c3))))) *(((((-e0_m_c3)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c3))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c3))) *(e0_greek_alpha_SecR_c3))+(e0_greek_alpha_SecR_c3) *(((-e0_m_c3)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c3))))) *((2.0)**(1/(e0_greek_alpha_SecR_c3)))))) *((1.0-e0_greek_delta_c3_caux3)))))+(e0_F_LSecR_c3) *((e0_F_LSecR_c4) *(((2.0)**(1/(((e0_a_c3) *(e0_a_c4))/((e0_greek_alpha_SecR_c3) *(e0_greek_alpha_SecR_c4))))) *(((((-e0_m_c3)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c3))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c3))) *(e0_greek_alpha_SecR_c4))+(e0_greek_alpha_SecR_c3) *(((-e0_m_c4)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c4))))) *((2.0)**(1/(e0_greek_alpha_SecR_c4)))))) *((1.0-e0_greek_delta_c3_caux4)))))+(e0_F_LSecR_c3) *((e0_F_LSecR_c5) *(((2.0)**(1/(((e0_a_c3) *(e0_a_c5))/((e0_greek_alpha_SecR_c3) *(e0_greek_alpha_SecR_c5))))) *(((((-e0_m_c3)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c3))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c3))) *(e0_greek_alpha_SecR_c5))+(e0_greek_alpha_SecR_c3) *(((-e0_m_c5)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c5))))) *((2.0)**(1/(e0_greek_alpha_SecR_c5)))))) *((1.0-e0_greek_delta_c3_caux5))))))+((e0_F_LSecR_c4) *((e0_F_LSecR_c1) *(((2.0)**(1/(((e0_a_c4) *(e0_a_c1))/((e0_greek_alpha_SecR_c4) *(e0_greek_alpha_SecR_c1))))) *(((((-e0_m_c4)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c4))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c4))) *(e0_greek_alpha_SecR_c1))+(e0_greek_alpha_SecR_c4) *(((-e0_m_c1)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c1))))) *((2.0)**(1/(e0_greek_alpha_SecR_c1)))))) *((1.0-e0_greek_delta_c4_caux1)))))+(e0_F_LSecR_c4) *((e0_F_LSecR_c2) *(((2.0)**(1/(((e0_a_c4) *(e0_a_c2))/((e0_greek_alpha_SecR_c4) *(e0_greek_alpha_SecR_c2))))) *(((((-e0_m_c4)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c4))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c4))) *(e0_greek_alpha_SecR_c2))+(e0_greek_alpha_SecR_c4) *(((-e0_m_c2)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c2))))) *((2.0)**(1/(e0_greek_alpha_SecR_c2)))))) *((1.0-e0_greek_delta_c4_caux2)))))+(e0_F_LSecR_c4) *((e0_F_LSecR_c3) *(((2.0)**(1/(((e0_a_c4) *(e0_a_c3))/((e0_greek_alpha_SecR_c4) *(e0_greek_alpha_SecR_c3))))) *(((((-e0_m_c4)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c4))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c4))) *(e0_greek_alpha_SecR_c3))+(e0_greek_alpha_SecR_c4) *(((-e0_m_c3)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c3))))) *((2.0)**(1/(e0_greek_alpha_SecR_c3)))))) *((1.0-e0_greek_delta_c4_caux3)))))+(e0_F_LSecR_c4) *((e0_F_LSecR_c4) *(((2.0)**(1/(((e0_a_c4) *(e0_a_c4))/((e0_greek_alpha_SecR_c4) *(e0_greek_alpha_SecR_c4))))) *(((((-e0_m_c4)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c4))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c4))) *(e0_greek_alpha_SecR_c4))+(e0_greek_alpha_SecR_c4) *(((-e0_m_c4)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c4))))) *((2.0)**(1/(e0_greek_alpha_SecR_c4)))))) *((1.0-e0_greek_delta_c4_caux4)))))+(e0_F_LSecR_c4) *((e0_F_LSecR_c5) *(((2.0)**(1/(((e0_a_c4) *(e0_a_c5))/((e0_greek_alpha_SecR_c4) *(e0_greek_alpha_SecR_c5))))) *(((((-e0_m_c4)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c4))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c4))) *(e0_greek_alpha_SecR_c5))+(e0_greek_alpha_SecR_c4) *(((-e0_m_c5)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c5))))) *((2.0)**(1/(e0_greek_alpha_SecR_c5)))))) *((1.0-e0_greek_delta_c4_caux5))))))+((e0_F_LSecR_c5) *((e0_F_LSecR_c1) *(((2.0)**(1/(((e0_a_c5) *(e0_a_c1))/((e0_greek_alpha_SecR_c5) *(e0_greek_alpha_SecR_c1))))) *(((((-e0_m_c5)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c5))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c5))) *(e0_greek_alpha_SecR_c1))+(e0_greek_alpha_SecR_c5) *(((-e0_m_c1)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c1))))) *((2.0)**(1/(e0_greek_alpha_SecR_c1)))))) *((1.0-e0_greek_delta_c5_caux1)))))+(e0_F_LSecR_c5) *((e0_F_LSecR_c2) *(((2.0)**(1/(((e0_a_c5) *(e0_a_c2))/((e0_greek_alpha_SecR_c5) *(e0_greek_alpha_SecR_c2))))) *(((((-e0_m_c5)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c5))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c5))) *(e0_greek_alpha_SecR_c2))+(e0_greek_alpha_SecR_c5) *(((-e0_m_c2)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c2))))) *((2.0)**(1/(e0_greek_alpha_SecR_c2)))))) *((1.0-e0_greek_delta_c5_caux2)))))+(e0_F_LSecR_c5) *((e0_F_LSecR_c3) *(((2.0)**(1/(((e0_a_c5) *(e0_a_c3))/((e0_greek_alpha_SecR_c5) *(e0_greek_alpha_SecR_c3))))) *(((((-e0_m_c5)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c5))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c5))) *(e0_greek_alpha_SecR_c3))+(e0_greek_alpha_SecR_c5) *(((-e0_m_c3)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c3))))) *((2.0)**(1/(e0_greek_alpha_SecR_c3)))))) *((1.0-e0_greek_delta_c5_caux3)))))+(e0_F_LSecR_c5) *((e0_F_LSecR_c4) *(((2.0)**(1/(((e0_a_c5) *(e0_a_c4))/((e0_greek_alpha_SecR_c5) *(e0_greek_alpha_SecR_c4))))) *(((((-e0_m_c5)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c5))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c5))) *(e0_greek_alpha_SecR_c4))+(e0_greek_alpha_SecR_c5) *(((-e0_m_c4)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c4))))) *((2.0)**(1/(e0_greek_alpha_SecR_c4)))))) *((1.0-e0_greek_delta_c5_caux4)))))+(e0_F_LSecR_c5) *((e0_F_LSecR_c5) *(((2.0)**(1/(((e0_a_c5) *(e0_a_c5))/((e0_greek_alpha_SecR_c5) *(e0_greek_alpha_SecR_c5))))) *(((((-e0_m_c5)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c5))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c5))) *(e0_greek_alpha_SecR_c5))+(e0_greek_alpha_SecR_c5) *(((-e0_m_c5)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c5))))) *((2.0)**(1/(e0_greek_alpha_SecR_c5)))))) *((1.0-e0_greek_delta_c5_caux5)))))))))))/((2.0) *(((2.0)**(1/(2.0))) *(e0_b_LSecR)))) *(sympy.log((e0_v_LSecR+((1.0+(2.0)**(1/(2.0)))) *(e0_b_LSecR))/(e0_v_LSecR+((1.0-(2.0)**(1/(2.0)))) *(e0_b_LSecR))))-((e0_a_SecRV-(e0_T_SecR) *((((1.0)/((2.0) *(((e0_F_SecRV))**(2.0)))) *((((e0_F_SecRV_c1) *((e0_F_SecRV_c1) *(((2.0)**(1/(((e0_a_c1) *(e0_a_c1))/((e0_greek_alpha_SecR_c1) *(e0_greek_alpha_SecR_c1))))) *(((((-e0_m_c1)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c1))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c1))) *(e0_greek_alpha_SecR_c1))+(e0_greek_alpha_SecR_c1) *(((-e0_m_c1)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c1))))) *((2.0)**(1/(e0_greek_alpha_SecR_c1)))))) *((1.0-e0_greek_delta_c1_caux1)))))+(e0_F_SecRV_c1) *((e0_F_SecRV_c2) *(((2.0)**(1/(((e0_a_c1) *(e0_a_c2))/((e0_greek_alpha_SecR_c1) *(e0_greek_alpha_SecR_c2))))) *(((((-e0_m_c1)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c1))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c1))) *(e0_greek_alpha_SecR_c2))+(e0_greek_alpha_SecR_c1) *(((-e0_m_c2)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c2))))) *((2.0)**(1/(e0_greek_alpha_SecR_c2)))))) *((1.0-e0_greek_delta_c1_caux2)))))+(e0_F_SecRV_c1) *((e0_F_SecRV_c3) *(((2.0)**(1/(((e0_a_c1) *(e0_a_c3))/((e0_greek_alpha_SecR_c1) *(e0_greek_alpha_SecR_c3))))) *(((((-e0_m_c1)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c1))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c1))) *(e0_greek_alpha_SecR_c3))+(e0_greek_alpha_SecR_c1) *(((-e0_m_c3)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c3))))) *((2.0)**(1/(e0_greek_alpha_SecR_c3)))))) *((1.0-e0_greek_delta_c1_caux3)))))+(e0_F_SecRV_c1) *((e0_F_SecRV_c4) *(((2.0)**(1/(((e0_a_c1) *(e0_a_c4))/((e0_greek_alpha_SecR_c1) *(e0_greek_alpha_SecR_c4))))) *(((((-e0_m_c1)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c1))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c1))) *(e0_greek_alpha_SecR_c4))+(e0_greek_alpha_SecR_c1) *(((-e0_m_c4)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c4))))) *((2.0)**(1/(e0_greek_alpha_SecR_c4)))))) *((1.0-e0_greek_delta_c1_caux4)))))+(e0_F_SecRV_c1) *((e0_F_SecRV_c5) *(((2.0)**(1/(((e0_a_c1) *(e0_a_c5))/((e0_greek_alpha_SecR_c1) *(e0_greek_alpha_SecR_c5))))) *(((((-e0_m_c1)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c1))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c1))) *(e0_greek_alpha_SecR_c5))+(e0_greek_alpha_SecR_c1) *(((-e0_m_c5)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c5))))) *((2.0)**(1/(e0_greek_alpha_SecR_c5)))))) *((1.0-e0_greek_delta_c1_caux5))))))+((e0_F_SecRV_c2) *((e0_F_SecRV_c1) *(((2.0)**(1/(((e0_a_c2) *(e0_a_c1))/((e0_greek_alpha_SecR_c2) *(e0_greek_alpha_SecR_c1))))) *(((((-e0_m_c2)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c2))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c2))) *(e0_greek_alpha_SecR_c1))+(e0_greek_alpha_SecR_c2) *(((-e0_m_c1)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c1))))) *((2.0)**(1/(e0_greek_alpha_SecR_c1)))))) *((1.0-e0_greek_delta_c2_caux1)))))+(e0_F_SecRV_c2) *((e0_F_SecRV_c2) *(((2.0)**(1/(((e0_a_c2) *(e0_a_c2))/((e0_greek_alpha_SecR_c2) *(e0_greek_alpha_SecR_c2))))) *(((((-e0_m_c2)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c2))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c2))) *(e0_greek_alpha_SecR_c2))+(e0_greek_alpha_SecR_c2) *(((-e0_m_c2)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c2))))) *((2.0)**(1/(e0_greek_alpha_SecR_c2)))))) *((1.0-e0_greek_delta_c2_caux2)))))+(e0_F_SecRV_c2) *((e0_F_SecRV_c3) *(((2.0)**(1/(((e0_a_c2) *(e0_a_c3))/((e0_greek_alpha_SecR_c2) *(e0_greek_alpha_SecR_c3))))) *(((((-e0_m_c2)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c2))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c2))) *(e0_greek_alpha_SecR_c3))+(e0_greek_alpha_SecR_c2) *(((-e0_m_c3)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c3))))) *((2.0)**(1/(e0_greek_alpha_SecR_c3)))))) *((1.0-e0_greek_delta_c2_caux3)))))+(e0_F_SecRV_c2) *((e0_F_SecRV_c4) *(((2.0)**(1/(((e0_a_c2) *(e0_a_c4))/((e0_greek_alpha_SecR_c2) *(e0_greek_alpha_SecR_c4))))) *(((((-e0_m_c2)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c2))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c2))) *(e0_greek_alpha_SecR_c4))+(e0_greek_alpha_SecR_c2) *(((-e0_m_c4)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c4))))) *((2.0)**(1/(e0_greek_alpha_SecR_c4)))))) *((1.0-e0_greek_delta_c2_caux4)))))+(e0_F_SecRV_c2) *((e0_F_SecRV_c5) *(((2.0)**(1/(((e0_a_c2) *(e0_a_c5))/((e0_greek_alpha_SecR_c2) *(e0_greek_alpha_SecR_c5))))) *(((((-e0_m_c2)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c2))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c2))) *(e0_greek_alpha_SecR_c5))+(e0_greek_alpha_SecR_c2) *(((-e0_m_c5)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c5))))) *((2.0)**(1/(e0_greek_alpha_SecR_c5)))))) *((1.0-e0_greek_delta_c2_caux5))))))+((e0_F_SecRV_c3) *((e0_F_SecRV_c1) *(((2.0)**(1/(((e0_a_c3) *(e0_a_c1))/((e0_greek_alpha_SecR_c3) *(e0_greek_alpha_SecR_c1))))) *(((((-e0_m_c3)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c3))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c3))) *(e0_greek_alpha_SecR_c1))+(e0_greek_alpha_SecR_c3) *(((-e0_m_c1)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c1))))) *((2.0)**(1/(e0_greek_alpha_SecR_c1)))))) *((1.0-e0_greek_delta_c3_caux1)))))+(e0_F_SecRV_c3) *((e0_F_SecRV_c2) *(((2.0)**(1/(((e0_a_c3) *(e0_a_c2))/((e0_greek_alpha_SecR_c3) *(e0_greek_alpha_SecR_c2))))) *(((((-e0_m_c3)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c3))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c3))) *(e0_greek_alpha_SecR_c2))+(e0_greek_alpha_SecR_c3) *(((-e0_m_c2)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c2))))) *((2.0)**(1/(e0_greek_alpha_SecR_c2)))))) *((1.0-e0_greek_delta_c3_caux2)))))+(e0_F_SecRV_c3) *((e0_F_SecRV_c3) *(((2.0)**(1/(((e0_a_c3) *(e0_a_c3))/((e0_greek_alpha_SecR_c3) *(e0_greek_alpha_SecR_c3))))) *(((((-e0_m_c3)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c3))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c3))) *(e0_greek_alpha_SecR_c3))+(e0_greek_alpha_SecR_c3) *(((-e0_m_c3)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c3))))) *((2.0)**(1/(e0_greek_alpha_SecR_c3)))))) *((1.0-e0_greek_delta_c3_caux3)))))+(e0_F_SecRV_c3) *((e0_F_SecRV_c4) *(((2.0)**(1/(((e0_a_c3) *(e0_a_c4))/((e0_greek_alpha_SecR_c3) *(e0_greek_alpha_SecR_c4))))) *(((((-e0_m_c3)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c3))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c3))) *(e0_greek_alpha_SecR_c4))+(e0_greek_alpha_SecR_c3) *(((-e0_m_c4)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c4))))) *((2.0)**(1/(e0_greek_alpha_SecR_c4)))))) *((1.0-e0_greek_delta_c3_caux4)))))+(e0_F_SecRV_c3) *((e0_F_SecRV_c5) *(((2.0)**(1/(((e0_a_c3) *(e0_a_c5))/((e0_greek_alpha_SecR_c3) *(e0_greek_alpha_SecR_c5))))) *(((((-e0_m_c3)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c3))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c3))) *(e0_greek_alpha_SecR_c5))+(e0_greek_alpha_SecR_c3) *(((-e0_m_c5)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c5))))) *((2.0)**(1/(e0_greek_alpha_SecR_c5)))))) *((1.0-e0_greek_delta_c3_caux5))))))+((e0_F_SecRV_c4) *((e0_F_SecRV_c1) *(((2.0)**(1/(((e0_a_c4) *(e0_a_c1))/((e0_greek_alpha_SecR_c4) *(e0_greek_alpha_SecR_c1))))) *(((((-e0_m_c4)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c4))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c4))) *(e0_greek_alpha_SecR_c1))+(e0_greek_alpha_SecR_c4) *(((-e0_m_c1)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c1))))) *((2.0)**(1/(e0_greek_alpha_SecR_c1)))))) *((1.0-e0_greek_delta_c4_caux1)))))+(e0_F_SecRV_c4) *((e0_F_SecRV_c2) *(((2.0)**(1/(((e0_a_c4) *(e0_a_c2))/((e0_greek_alpha_SecR_c4) *(e0_greek_alpha_SecR_c2))))) *(((((-e0_m_c4)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c4))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c4))) *(e0_greek_alpha_SecR_c2))+(e0_greek_alpha_SecR_c4) *(((-e0_m_c2)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c2))))) *((2.0)**(1/(e0_greek_alpha_SecR_c2)))))) *((1.0-e0_greek_delta_c4_caux2)))))+(e0_F_SecRV_c4) *((e0_F_SecRV_c3) *(((2.0)**(1/(((e0_a_c4) *(e0_a_c3))/((e0_greek_alpha_SecR_c4) *(e0_greek_alpha_SecR_c3))))) *(((((-e0_m_c4)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c4))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c4))) *(e0_greek_alpha_SecR_c3))+(e0_greek_alpha_SecR_c4) *(((-e0_m_c3)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c3))))) *((2.0)**(1/(e0_greek_alpha_SecR_c3)))))) *((1.0-e0_greek_delta_c4_caux3)))))+(e0_F_SecRV_c4) *((e0_F_SecRV_c4) *(((2.0)**(1/(((e0_a_c4) *(e0_a_c4))/((e0_greek_alpha_SecR_c4) *(e0_greek_alpha_SecR_c4))))) *(((((-e0_m_c4)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c4))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c4))) *(e0_greek_alpha_SecR_c4))+(e0_greek_alpha_SecR_c4) *(((-e0_m_c4)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c4))))) *((2.0)**(1/(e0_greek_alpha_SecR_c4)))))) *((1.0-e0_greek_delta_c4_caux4)))))+(e0_F_SecRV_c4) *((e0_F_SecRV_c5) *(((2.0)**(1/(((e0_a_c4) *(e0_a_c5))/((e0_greek_alpha_SecR_c4) *(e0_greek_alpha_SecR_c5))))) *(((((-e0_m_c4)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c4))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c4))) *(e0_greek_alpha_SecR_c5))+(e0_greek_alpha_SecR_c4) *(((-e0_m_c5)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c5))))) *((2.0)**(1/(e0_greek_alpha_SecR_c5)))))) *((1.0-e0_greek_delta_c4_caux5))))))+((e0_F_SecRV_c5) *((e0_F_SecRV_c1) *(((2.0)**(1/(((e0_a_c5) *(e0_a_c1))/((e0_greek_alpha_SecR_c5) *(e0_greek_alpha_SecR_c1))))) *(((((-e0_m_c5)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c5))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c5))) *(e0_greek_alpha_SecR_c1))+(e0_greek_alpha_SecR_c5) *(((-e0_m_c1)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c1))))) *((2.0)**(1/(e0_greek_alpha_SecR_c1)))))) *((1.0-e0_greek_delta_c5_caux1)))))+(e0_F_SecRV_c5) *((e0_F_SecRV_c2) *(((2.0)**(1/(((e0_a_c5) *(e0_a_c2))/((e0_greek_alpha_SecR_c5) *(e0_greek_alpha_SecR_c2))))) *(((((-e0_m_c5)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c5))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c5))) *(e0_greek_alpha_SecR_c2))+(e0_greek_alpha_SecR_c5) *(((-e0_m_c2)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c2))))) *((2.0)**(1/(e0_greek_alpha_SecR_c2)))))) *((1.0-e0_greek_delta_c5_caux2)))))+(e0_F_SecRV_c5) *((e0_F_SecRV_c3) *(((2.0)**(1/(((e0_a_c5) *(e0_a_c3))/((e0_greek_alpha_SecR_c5) *(e0_greek_alpha_SecR_c3))))) *(((((-e0_m_c5)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c5))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c5))) *(e0_greek_alpha_SecR_c3))+(e0_greek_alpha_SecR_c5) *(((-e0_m_c3)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c3))))) *((2.0)**(1/(e0_greek_alpha_SecR_c3)))))) *((1.0-e0_greek_delta_c5_caux3)))))+(e0_F_SecRV_c5) *((e0_F_SecRV_c4) *(((2.0)**(1/(((e0_a_c5) *(e0_a_c4))/((e0_greek_alpha_SecR_c5) *(e0_greek_alpha_SecR_c4))))) *(((((-e0_m_c5)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c5))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c5))) *(e0_greek_alpha_SecR_c4))+(e0_greek_alpha_SecR_c5) *(((-e0_m_c4)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c4))))) *((2.0)**(1/(e0_greek_alpha_SecR_c4)))))) *((1.0-e0_greek_delta_c5_caux4)))))+(e0_F_SecRV_c5) *((e0_F_SecRV_c5) *(((2.0)**(1/(((e0_a_c5) *(e0_a_c5))/((e0_greek_alpha_SecR_c5) *(e0_greek_alpha_SecR_c5))))) *(((((-e0_m_c5)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c5))))) *(((2.0)**(1/(e0_greek_alpha_SecR_c5))) *(e0_greek_alpha_SecR_c5))+(e0_greek_alpha_SecR_c5) *(((-e0_m_c5)/((2.0)**(1/((e0_T_SecR) *(e0_TC_c5))))) *((2.0)**(1/(e0_greek_alpha_SecR_c5)))))) *((1.0-e0_greek_delta_c5_caux5)))))))))))/((2.0) *(((2.0)**(1/(2.0))) *(e0_b_SecRV)))) *(sympy.log((e0_v_SecRV+((1.0+(2.0)**(1/(2.0)))) *(e0_b_SecRV))/(e0_v_SecRV+((1.0-(2.0)**(1/(2.0)))) *(e0_b_SecRV))))+(e0_P_SecR) *((((10.0))**(5.0)) *((e0_v_SecRV-e0_v_LSecR))))) *(((10.0))**(-3.0)))) ,
    e0_a_c1-((e0_aEOS) *(((((e0_R) *(e0_TC_c1)))**(2.0))/(e0_PC_c1))) ,
    e0_a_c2-((e0_aEOS) *(((((e0_R) *(e0_TC_c2)))**(2.0))/(e0_PC_c2))) ,
    e0_a_c3-((e0_aEOS) *(((((e0_R) *(e0_TC_c3)))**(2.0))/(e0_PC_c3))) ,
    e0_a_c4-((e0_aEOS) *(((((e0_R) *(e0_TC_c4)))**(2.0))/(e0_PC_c4))) ,
    e0_a_c5-((e0_aEOS) *(((((e0_R) *(e0_TC_c5)))**(2.0))/(e0_PC_c5))) ,
    e0_b_c1-((e0_bEOS) *(((e0_R) *(e0_TC_c1))/(e0_PC_c1))) ,
    e0_b_c2-((e0_bEOS) *(((e0_R) *(e0_TC_c2))/(e0_PC_c2))) ,
    e0_b_c3-((e0_bEOS) *(((e0_R) *(e0_TC_c3))/(e0_PC_c3))) ,
    e0_b_c4-((e0_bEOS) *(((e0_R) *(e0_TC_c4))/(e0_PC_c4))) ,
    e0_b_c5-((e0_bEOS) *(((e0_R) *(e0_TC_c5))/(e0_PC_c5))) ,
    e0_m_c1-(e0_mEOS_m1+(e0_mEOS_m2) *(e0_greek_omega_c1)+(e0_mEOS_m3) *(((e0_greek_omega_c1))**(2.0))) ,
    e0_m_c2-(e0_mEOS_m1+(e0_mEOS_m2) *(e0_greek_omega_c2)+(e0_mEOS_m3) *(((e0_greek_omega_c2))**(2.0))) ,
    e0_m_c3-(e0_mEOS_m1+(e0_mEOS_m2) *(e0_greek_omega_c3)+(e0_mEOS_m3) *(((e0_greek_omega_c3))**(2.0))) ,
    e0_m_c4-(e0_mEOS_m1+(e0_mEOS_m2) *(e0_greek_omega_c4)+(e0_mEOS_m3) *(((e0_greek_omega_c4))**(2.0))) ,
    e0_m_c5-(e0_mEOS_m1+(e0_mEOS_m2) *(e0_greek_omega_c5)+(e0_mEOS_m3) *(((e0_greek_omega_c5))**(2.0))) 

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

    x = numpy.empty(93)
    xInitial = numpy.empty((93), dtype = object) 
    parameter = numpy.empty(114)
    dict_variables = {}

    # Iteration variable initializatio
    x[0] = 52.543912518 	# e0_h_SecRfeed_c1
    x[1] = 0.040498577 	# e0_h_SecRfeed_c2
    x[2] = -129.71613166999998 	# e0_h_SecRfeed_c3
    x[3] = -141.91847508 	# e0_h_SecRfeed_c4
    x[4] = -92.28088485245 	# e0_h_SecRfeed_c5
    x[5] = 33973.6612952904 	# e0_HU_LSecR
    x[6] = 8.474729881161444E-5 	# e0_v_LSecR
    x[7] = 0.07745967246479979 	# e0_greek_epsiv_LSecR
    x[8] = 34.29082397448339 	# e0_V_SecRV
    x[9] = 1595.5012336010248 	# e0_HU_SecRV
    x[10] = 0.02149219521258517 	# e0_v_SecRV
    x[11] = 6.696187576591392 	# e0_c_SecR_c1
    x[12] = 19.607237038495796 	# e0_c_SecR_c2
    x[13] = 0.06210505248484072 	# e0_greek_DeltaP_SecR
    x[14] = 1.3621050524848406 	# e0_P_SecR
    x[15] = 1167.5735002378767 	# e0_greek_rho_LSecR
    x[16] = 0.09894870030918433 	# e0_M_LSecR
    x[17] = 4.268408896764255 	# e0_greek_rho_SecRV
    x[18] = 0.09173747725641823 	# e0_M_SecRV
    x[19] = 0.8692760087275578 	# e0_greek_alpha_SecR_c1
    x[20] = 1.0654978983192214 	# e0_greek_alpha_SecR_c2
    x[21] = 1.335833707787174 	# e0_greek_alpha_SecR_c3
    x[22] = 1.3665879216548895 	# e0_greek_alpha_SecR_c4
    x[23] = 0.9353832428023278 	# e0_greek_alpha_SecR_c5
    x[24] = 0.49979007858311525 	# e0_a_c1
    x[25] = 0.713328580688273 	# e0_a_c2
    x[26] = 1.856262622486394 	# e0_a_c3
    x[27] = 2.5566688247900164 	# e0_a_c4
    x[28] = 0.40085683942951444 	# e0_a_c5
    x[29] = 2.4759676618762305 	# e0_a_LSecR
    x[30] = 2.141906434001912 	# e0_a_SecRV
    x[31] = 3.622621837511605E-5 	# e0_b_c1
    x[32] = 3.4994932300856034E-5 	# e0_b_c2
    x[33] = 6.764257023284918E-5 	# e0_b_c3
    x[34] = 8.691320497500001E-5 	# e0_b_c4
    x[35] = 2.5268628546281584E-5 	# e0_b_c5
    x[36] = 6.761195791942727E-5 	# e0_b_LSecR
    x[37] = 6.348152544089798E-5 	# e0_b_SecRV
    x[38] = 0.5055771876352 	# e0_m_c1
    x[39] = 0.4794698378752 	# e0_m_c2
    x[40] = 0.7944806059648 	# e0_m_c3
    x[41] = 0.7561190779247999 	# e0_m_c4
    x[42] = 0.57277966588 	# e0_m_c5
    x[43] = 0.004256128570890748 	# e0_F_LSecR_c1
    x[44] = 0.012462452821290625 	# e0_F_LSecR_c2
    x[45] = 7.462729301548669 	# e0_F_LSecR_c3
    x[46] = 0.019172222566749886 	# e0_F_LSecR_c4
    x[47] = 0.0013798944923991774 	# e0_F_LSecR_c5
    x[48] = 8.060148956459981 	# e0_F_SecRV_c1
    x[49] = 4.096873696934329 	# e0_F_SecRV_c2
    x[50] = 92.52465063254081 	# e0_F_SecRV_c3
    x[51] = 0.10290858618580234 	# e0_F_SecRV_c4
    x[52] = 1.3352227087151913 	# e0_F_SecRV_c5
    x[53] = 2.8791760255166072 	# e0_V_LSecR
    x[54] = 3.9646939582829654 	# e0_r_SecR_r1
    x[55] = 0.042401218107044156 	# e0_r_SecR_r2
    x[56] = 134.6853556366935 	# e0_greek_phiv_LSecR_c1
    x[57] = 23.161118087331666 	# e0_greek_phiv_LSecR_c2
    x[58] = 0.8466897437382953 	# e0_greek_phiv_LSecR_c3
    x[59] = 0.3620103498144347 	# e0_greek_phiv_LSecR_c4
    x[60] = 68.86006962439318 	# e0_greek_phiv_LSecR_c5
    x[61] = 1.0062994294791057 	# e0_greek_phiv_SecRV_c1
    x[62] = 0.9968863536532157 	# e0_greek_phiv_SecRV_c2
    x[63] = 0.9662727049877151 	# e0_greek_phiv_SecRV_c3
    x[64] = 0.9542826229804309 	# e0_greek_phiv_SecRV_c4
    x[65] = 1.0069196285407644 	# e0_greek_phiv_SecRV_c5
    x[66] = 106.11980458083613 	# e0_F_SecRV
    x[67] = 6.8497433785230974 	# e0_F_SecRfeed
    x[68] = 19.279502732883596 	# e0_HU_LSecR_c1
    x[69] = 56.45268680785547 	# e0_HU_LSecR_c2
    x[70] = 33804.831683900506 	# e0_HU_LSecR_c3
    x[71] = 86.84674610142451 	# e0_HU_LSecR_c4
    x[72] = 6.250675747732658 	# e0_HU_LSecR_c5
    x[73] = 140.4630806129566 	# e0_HU_SecR_c1
    x[74] = 118.04879568861338 	# e0_HU_SecR_c2
    x[75] = 35195.93106262459 	# e0_HU_SecR_c3
    x[76] = 88.3939669699184 	# e0_HU_SecR_c4
    x[77] = 26.32562299534833 	# e0_HU_SecR_c5
    x[78] = 121.183577880073 	# e0_HU_SecRV_c1
    x[79] = 61.5961088807579 	# e0_HU_SecRV_c2
    x[80] = 1391.0993787240843 	# e0_HU_SecRV_c3
    x[81] = 1.5472208684938722 	# e0_HU_SecRV_c4
    x[82] = 20.074947247615672 	# e0_HU_SecRV_c5
    x[83] = -161.81290120072964 	# e0_h_LSecR
    x[84] = -110.39286643187798 	# e0_h_SecRV
    x[85] = 0.040498577 	# e0_h_SecRfeed
    x[86] = 32.40410779040382 	# e0_greek_Deltah_SecRVLdeparture
    x[87] = 52.54486297074739 	# e0_h_SecRV_c1
    x[88] = 0.049444400895101354 	# e0_h_SecRV_c2
    x[89] = -129.70351923580424 	# e0_h_SecRV_c3
    x[90] = -141.90389895249785 	# e0_h_SecRV_c4
    x[91] = -92.28083768513048 	# e0_h_SecRV_c5
    x[92] = 362.95388485657327 	# e0_T_SecR


    # Constant parameter setting:
    parameter[0] = -1.0 	# e0_greek_nu_c1_r1
    parameter[1] = -1.0 	# e0_greek_nu_c1_r2
    parameter[2] = 0.0 	# e0_F_CONLSecR_c1
    parameter[3] = 0.0 	# e0_F_CONLSecR_c2
    parameter[4] = 273.0 	# e0_T_SecRfeed
    parameter[5] = 5.31 	# e0_ASecR
    parameter[6] = 7.0 	# e0_LSecR
    parameter[7] = 80.0 	# e0_F_CONLSecR_c3
    parameter[8] = 2156.58 	# e0_E_r1
    parameter[9] = 11.493 	# e0_k_r1
    parameter[10] = 7282.21 	# e0_E_r2
    parameter[11] = 8517.0 	# e0_k_r2
    parameter[12] = 1.3 	# e0_P_SecRtop
    parameter[13] = 0.0 	# e0_F_CONLSecR_c4
    parameter[14] = 0.0 	# e0_F_CONLSecR_c5
    parameter[15] = 0.0 	# e0_greek_delta_c1_caux1
    parameter[16] = 0.0 	# e0_greek_delta_c1_caux2
    parameter[17] = 0.0347 	# e0_greek_delta_c1_caux3
    parameter[18] = 0.0 	# e0_greek_delta_c1_caux4
    parameter[19] = 0.0 	# e0_greek_delta_c1_caux5
    parameter[20] = 0.0 	# e0_greek_delta_c2_caux1
    parameter[21] = 0.0 	# e0_greek_delta_c2_caux2
    parameter[22] = 0.0 	# e0_F_CoolerLSecR_c1
    parameter[23] = 0.033 	# e0_greek_delta_c2_caux3
    parameter[24] = 0.0 	# e0_greek_delta_c2_caux4
    parameter[25] = 0.0 	# e0_greek_delta_c2_caux5
    parameter[26] = 0.0347 	# e0_greek_delta_c3_caux1
    parameter[27] = 0.033 	# e0_greek_delta_c3_caux2
    parameter[28] = 0.0 	# e0_greek_delta_c3_caux3
    parameter[29] = 0.0 	# e0_greek_delta_c3_caux4
    parameter[30] = 0.0 	# e0_greek_delta_c3_caux5
    parameter[31] = 0.0 	# e0_greek_delta_c4_caux1
    parameter[32] = 0.0 	# e0_greek_delta_c4_caux2
    parameter[33] = 0.0 	# e0_F_CoolerLSecR_c2
    parameter[34] = 0.0 	# e0_greek_delta_c4_caux3
    parameter[35] = 0.0 	# e0_greek_delta_c4_caux4
    parameter[36] = 0.0 	# e0_greek_delta_c4_caux5
    parameter[37] = 0.0 	# e0_greek_delta_c5_caux1
    parameter[38] = 0.0 	# e0_greek_delta_c5_caux2
    parameter[39] = 0.0 	# e0_greek_delta_c5_caux3
    parameter[40] = 0.0 	# e0_greek_delta_c5_caux4
    parameter[41] = 0.0 	# e0_greek_delta_c5_caux5
    parameter[42] = 0.52 	# e0_F_CoolerLSecR_c3
    parameter[43] = 0.0 	# e0_F_CoolerLSecR_c4
    parameter[44] = 0.0 	# e0_F_CoolerLSecR_c5
    parameter[45] = 0.457235 	# e0_aEOS
    parameter[46] = 0.077796 	# e0_bEOS
    parameter[47] = 0.37464 	# e0_mEOS_m1
    parameter[48] = 1.54226 	# e0_mEOS_m2
    parameter[49] = -0.26992 	# e0_mEOS_m3
    parameter[50] = -1.0 	# e0_greek_nu_c2_r1
    parameter[51] = -2.0 	# e0_greek_nu_c2_r2
    parameter[52] = 19.6015374488562 	# e0_F_VCON_c1
    parameter[53] = 8.91880570568436 	# e0_F_VCON_c2
    parameter[54] = 8.05232814089068 	# e0_F_VCON_c3
    parameter[55] = 2.38126041607688E-7 	# e0_F_VCON_c4
    parameter[56] = 1.21452203258108 	# e0_F_VCON_c5
    parameter[57] = 100000.0 	# e0_PN
    parameter[58] = 273.15 	# e0_TN
    parameter[59] = 0.0 	# e0_V_SecRfeedN_c1
    parameter[60] = 560.0 	# e0_V_SecRfeedN_c2
    parameter[61] = 1.0 	# e0_greek_nu_c3_r1
    parameter[62] = 0.0 	# e0_V_SecRfeedN_c3
    parameter[63] = 0.0 	# e0_V_SecRfeedN_c4
    parameter[64] = 0.0 	# e0_V_SecRfeedN_c5
    parameter[65] = 0.0 	# e0_greek_nu_c3_r2
    parameter[66] = 7.5 	# e0_F_LSecR
    parameter[67] = 0.0 	# e0_greek_nu_c4_r1
    parameter[68] = 1.0 	# e0_greek_nu_c4_r2
    parameter[69] = 80.0 	# e0_F_CONLSecR
    parameter[70] = 0.52 	# e0_F_CoolerLSecR
    parameter[71] = 0.0 	# e0_Q_SecR
    parameter[72] = -158.99223672755912 	# e0_h_CONLSecR
    parameter[73] = -160.0 	# e0_h_CoolerLSecR
    parameter[74] = 0.0 	# e0_greek_nu_c5_r1
    parameter[75] = -3.3388200240062313 	# e0_h_VCON
    parameter[76] = 1.0 	# e0_greek_nu_c5_r2
    parameter[77] = 52.51 	# e0_greek_Deltahf_c1
    parameter[78] = 0.0 	# e0_greek_Deltahf_c2
    parameter[79] = -129.79 	# e0_greek_Deltahf_c3
    parameter[80] = -142.0 	# e0_greek_Deltahf_c4
    parameter[81] = -92.31 	# e0_greek_Deltahf_c5
    parameter[82] = 0.0862 	# e0_greek_omega_c1
    parameter[83] = 0.0688 	# e0_greek_omega_c2
    parameter[84] = 0.2866 	# e0_greek_omega_c3
    parameter[85] = 0.2591 	# e0_greek_omega_c4
    parameter[86] = 0.1315 	# e0_greek_omega_c5
    parameter[87] = 9.9449E-5 	# e0_CPA_c1
    parameter[88] = 1.0566E-5 	# e0_CPA_c2
    parameter[89] = 1.4021E-4 	# e0_CPA_c3
    parameter[90] = 1.6204E-4 	# e0_CPA_c4
    parameter[91] = 5.2435E-7 	# e0_CPA_c5
    parameter[92] = 0.013349 	# e0_CPB_c1
    parameter[93] = 0.031028 	# e0_CPB_c2
    parameter[94] = 0.035591 	# e0_CPB_c3
    parameter[95] = 0.037288 	# e0_CPB_c4
    parameter[96] = 0.028972 	# e0_CPB_c5
    parameter[97] = 0.028053 	# e0_M_c1
    parameter[98] = 0.070906 	# e0_M_c2
    parameter[99] = 0.098959 	# e0_M_c3
    parameter[100] = 0.133404 	# e0_M_c4
    parameter[101] = 0.036461 	# e0_M_c5
    parameter[102] = 5041000.0 	# e0_PC_c1
    parameter[103] = 7710000.0 	# e0_PC_c2
    parameter[104] = 5370000.0 	# e0_PC_c3
    parameter[105] = 4480000.0 	# e0_PC_c4
    parameter[106] = 8310000.0 	# e0_PC_c5
    parameter[107] = 8.314 	# e0_R
    parameter[108] = 282.34 	# e0_TC_c1
    parameter[109] = 417.15 	# e0_TC_c2
    parameter[110] = 561.6 	# e0_TC_c3
    parameter[111] = 602.0 	# e0_TC_c4
    parameter[112] = 324.65 	# e0_TC_c5
    parameter[113] = 9.81 	# e0_g


    # Get symbolic quantities:
    xSymbolic = sympy.symbols('e0_h_SecRfeed_c1 e0_h_SecRfeed_c2 e0_h_SecRfeed_c3 e0_h_SecRfeed_c4 e0_h_SecRfeed_c5 e0_HU_LSecR e0_v_LSecR e0_greek_epsiv_LSecR e0_V_SecRV e0_HU_SecRV e0_v_SecRV e0_c_SecR_c1 e0_c_SecR_c2 e0_greek_DeltaP_SecR e0_P_SecR e0_greek_rho_LSecR e0_M_LSecR e0_greek_rho_SecRV e0_M_SecRV e0_greek_alpha_SecR_c1 e0_greek_alpha_SecR_c2 e0_greek_alpha_SecR_c3 e0_greek_alpha_SecR_c4 e0_greek_alpha_SecR_c5 e0_a_c1 e0_a_c2 e0_a_c3 e0_a_c4 e0_a_c5 e0_a_LSecR e0_a_SecRV e0_b_c1 e0_b_c2 e0_b_c3 e0_b_c4 e0_b_c5 e0_b_LSecR e0_b_SecRV e0_m_c1 e0_m_c2 e0_m_c3 e0_m_c4 e0_m_c5 e0_F_LSecR_c1 e0_F_LSecR_c2 e0_F_LSecR_c3 e0_F_LSecR_c4 e0_F_LSecR_c5 e0_F_SecRV_c1 e0_F_SecRV_c2 e0_F_SecRV_c3 e0_F_SecRV_c4 e0_F_SecRV_c5 e0_V_LSecR e0_r_SecR_r1 e0_r_SecR_r2 e0_greek_phiv_LSecR_c1 e0_greek_phiv_LSecR_c2 e0_greek_phiv_LSecR_c3 e0_greek_phiv_LSecR_c4 e0_greek_phiv_LSecR_c5 e0_greek_phiv_SecRV_c1 e0_greek_phiv_SecRV_c2 e0_greek_phiv_SecRV_c3 e0_greek_phiv_SecRV_c4 e0_greek_phiv_SecRV_c5 e0_F_SecRV e0_F_SecRfeed e0_HU_LSecR_c1 e0_HU_LSecR_c2 e0_HU_LSecR_c3 e0_HU_LSecR_c4 e0_HU_LSecR_c5 e0_HU_SecR_c1 e0_HU_SecR_c2 e0_HU_SecR_c3 e0_HU_SecR_c4 e0_HU_SecR_c5 e0_HU_SecRV_c1 e0_HU_SecRV_c2 e0_HU_SecRV_c3 e0_HU_SecRV_c4 e0_HU_SecRV_c5 e0_h_LSecR e0_h_SecRV e0_h_SecRfeed e0_greek_Deltah_SecRVLdeparture e0_h_SecRV_c1 e0_h_SecRV_c2 e0_h_SecRV_c3 e0_h_SecRV_c4 e0_h_SecRV_c5 e0_T_SecR ')
    fSymbolic = getSymbolicFunctions(xSymbolic, parameter)

    # Intial variable bounds:
    xInitial[0] = mpmath.mpi(-300.0, 100.0)  	# e0_h_SecRfeed_c1
    xInitial[1] = mpmath.mpi(-300.0, 100.0)  	# e0_h_SecRfeed_c2
    xInitial[2] = mpmath.mpi(-300.0, 100.0)  	# e0_h_SecRfeed_c3
    xInitial[3] = mpmath.mpi(-300.0, 100.0)  	# e0_h_SecRfeed_c4
    xInitial[4] = mpmath.mpi(-300.0, 100.0)  	# e0_h_SecRfeed_c5
    xInitial[5] = mpmath.mpi(0.0, 50000.0)  	# e0_HU_LSecR
    xInitial[6] = mpmath.mpi(0.0, 1.0)  	# e0_v_LSecR
    xInitial[7] = mpmath.mpi(0.0, 100.0)  	# e0_greek_epsiv_LSecR
    xInitial[8] = mpmath.mpi(0.0, 50.0)  	# e0_V_SecRV
    xInitial[9] = mpmath.mpi(0.0, 50000.0)  	# e0_HU_SecRV
    xInitial[10] = mpmath.mpi(0.0, 1.0)  	# e0_v_SecRV
    xInitial[11] = mpmath.mpi(0.0, 100.0)  	# e0_c_SecR_c1
    xInitial[12] = mpmath.mpi(0.0, 100.0)  	# e0_c_SecR_c2
    xInitial[13] = mpmath.mpi(0.0, 1.0)  	# e0_greek_DeltaP_SecR
    xInitial[14] = mpmath.mpi(0.5, 2.0)  	# e0_P_SecR
    xInitial[15] = mpmath.mpi(500.0, 2000.0)  	# e0_greek_rho_LSecR
    xInitial[16] = mpmath.mpi(0.0, 3.0)  	# e0_M_LSecR
    xInitial[17] = mpmath.mpi(0.0, 10.0)  	# e0_greek_rho_SecRV
    xInitial[18] = mpmath.mpi(0.0, 3.0)  	# e0_M_SecRV
    xInitial[19] = mpmath.mpi(0.0, 100.0)  	# e0_greek_alpha_SecR_c1
    xInitial[20] = mpmath.mpi(0.0, 100.0)  	# e0_greek_alpha_SecR_c2
    xInitial[21] = mpmath.mpi(0.0, 100.0)  	# e0_greek_alpha_SecR_c3
    xInitial[22] = mpmath.mpi(0.0, 100.0)  	# e0_greek_alpha_SecR_c4
    xInitial[23] = mpmath.mpi(0.0, 100.0)  	# e0_greek_alpha_SecR_c5
    xInitial[24] = mpmath.mpi(0.0, 10.0)  	# e0_a_c1
    xInitial[25] = mpmath.mpi(0.0, 10.0)  	# e0_a_c2
    xInitial[26] = mpmath.mpi(0.0, 10.0)  	# e0_a_c3
    xInitial[27] = mpmath.mpi(0.0, 10.0)  	# e0_a_c4
    xInitial[28] = mpmath.mpi(0.0, 10.0)  	# e0_a_c5
    xInitial[29] = mpmath.mpi(0.0, 10.0)  	# e0_a_LSecR
    xInitial[30] = mpmath.mpi(0.0, 10.0)  	# e0_a_SecRV
    xInitial[31] = mpmath.mpi(0.0, 0.001)  	# e0_b_c1
    xInitial[32] = mpmath.mpi(0.0, 0.001)  	# e0_b_c2
    xInitial[33] = mpmath.mpi(0.0, 0.001)  	# e0_b_c3
    xInitial[34] = mpmath.mpi(0.0, 0.001)  	# e0_b_c4
    xInitial[35] = mpmath.mpi(0.0, 0.001)  	# e0_b_c5
    xInitial[36] = mpmath.mpi(0.0, 0.001)  	# e0_b_LSecR
    xInitial[37] = mpmath.mpi(0.0, 0.001)  	# e0_b_SecRV
    xInitial[38] = mpmath.mpi(0.0, 1.0)  	# e0_m_c1
    xInitial[39] = mpmath.mpi(0.0, 1.0)  	# e0_m_c2
    xInitial[40] = mpmath.mpi(0.0, 1.0)  	# e0_m_c3
    xInitial[41] = mpmath.mpi(0.0, 1.0)  	# e0_m_c4
    xInitial[42] = mpmath.mpi(0.0, 1.0)  	# e0_m_c5
    xInitial[43] = mpmath.mpi(0.0, 50.0)  	# e0_F_LSecR_c1
    xInitial[44] = mpmath.mpi(0.0, 50.0)  	# e0_F_LSecR_c2
    xInitial[45] = mpmath.mpi(0.0, 50.0)  	# e0_F_LSecR_c3
    xInitial[46] = mpmath.mpi(0.0, 50.0)  	# e0_F_LSecR_c4
    xInitial[47] = mpmath.mpi(0.0, 50.0)  	# e0_F_LSecR_c5
    xInitial[48] = mpmath.mpi(0.0, 500.0)  	# e0_F_SecRV_c1
    xInitial[49] = mpmath.mpi(0.0, 500.0)  	# e0_F_SecRV_c2
    xInitial[50] = mpmath.mpi(0.0, 500.0)  	# e0_F_SecRV_c3
    xInitial[51] = mpmath.mpi(0.0, 500.0)  	# e0_F_SecRV_c4
    xInitial[52] = mpmath.mpi(0.0, 500.0)  	# e0_F_SecRV_c5
    xInitial[53] = mpmath.mpi(0.0, 50.0)  	# e0_V_LSecR
    xInitial[54] = mpmath.mpi(0.0, 10.0)  	# e0_r_SecR_r1
    xInitial[55] = mpmath.mpi(0.0, 10.0)  	# e0_r_SecR_r2
    xInitial[56] = mpmath.mpi(0.0, 150.0)  	# e0_greek_phiv_LSecR_c1
    xInitial[57] = mpmath.mpi(0.0, 100.0)  	# e0_greek_phiv_LSecR_c2
    xInitial[58] = mpmath.mpi(0.0, 100.0)  	# e0_greek_phiv_LSecR_c3
    xInitial[59] = mpmath.mpi(0.0, 100.0)  	# e0_greek_phiv_LSecR_c4
    xInitial[60] = mpmath.mpi(0.0, 100.0)  	# e0_greek_phiv_LSecR_c5
    xInitial[61] = mpmath.mpi(0.0, 100.0)  	# e0_greek_phiv_SecRV_c1
    xInitial[62] = mpmath.mpi(0.0, 100.0)  	# e0_greek_phiv_SecRV_c2
    xInitial[63] = mpmath.mpi(0.0, 100.0)  	# e0_greek_phiv_SecRV_c3
    xInitial[64] = mpmath.mpi(0.0, 100.0)  	# e0_greek_phiv_SecRV_c4
    xInitial[65] = mpmath.mpi(0.0, 100.0)  	# e0_greek_phiv_SecRV_c5
    xInitial[66] = mpmath.mpi(0.0, 500.0)  	# e0_F_SecRV
    xInitial[67] = mpmath.mpi(0.0, 500.0)  	# e0_F_SecRfeed
    xInitial[68] = mpmath.mpi(0.0, 50000.0)  	# e0_HU_LSecR_c1
    xInitial[69] = mpmath.mpi(0.0, 50000.0)  	# e0_HU_LSecR_c2
    xInitial[70] = mpmath.mpi(0.0, 50000.0)  	# e0_HU_LSecR_c3
    xInitial[71] = mpmath.mpi(0.0, 50000.0)  	# e0_HU_LSecR_c4
    xInitial[72] = mpmath.mpi(0.0, 50000.0)  	# e0_HU_LSecR_c5
    xInitial[73] = mpmath.mpi(0.0, 50000.0)  	# e0_HU_SecR_c1
    xInitial[74] = mpmath.mpi(0.0, 50000.0)  	# e0_HU_SecR_c2
    xInitial[75] = mpmath.mpi(0.0, 50000.0)  	# e0_HU_SecR_c3
    xInitial[76] = mpmath.mpi(0.0, 50000.0)  	# e0_HU_SecR_c4
    xInitial[77] = mpmath.mpi(0.0, 50000.0)  	# e0_HU_SecR_c5
    xInitial[78] = mpmath.mpi(0.0, 50000.0)  	# e0_HU_SecRV_c1
    xInitial[79] = mpmath.mpi(0.0, 50000.0)  	# e0_HU_SecRV_c2
    xInitial[80] = mpmath.mpi(0.0, 50000.0)  	# e0_HU_SecRV_c3
    xInitial[81] = mpmath.mpi(0.0, 50000.0)  	# e0_HU_SecRV_c4
    xInitial[82] = mpmath.mpi(0.0, 50000.0)  	# e0_HU_SecRV_c5
    xInitial[83] = mpmath.mpi(-300.0, 0.0)  	# e0_h_LSecR
    xInitial[84] = mpmath.mpi(-300.0, 0.0)  	# e0_h_SecRV
    xInitial[85] = mpmath.mpi(-300.0, 100.0)  	# e0_h_SecRfeed
    xInitial[86] = mpmath.mpi(0.0, 100.0)  	# e0_greek_Deltah_SecRVLdeparture
    xInitial[87] = mpmath.mpi(-300.0, 100.0)  	# e0_h_SecRV_c1
    xInitial[88] = mpmath.mpi(-300.0, 100.0)  	# e0_h_SecRV_c2
    xInitial[89] = mpmath.mpi(-300.0, 100.0)  	# e0_h_SecRV_c3
    xInitial[90] = mpmath.mpi(-300.0, 100.0)  	# e0_h_SecRV_c4
    xInitial[91] = mpmath.mpi(-300.0, 100.0)  	# e0_h_SecRV_c5
    xInitial[92] = mpmath.mpi(200.0, 500.0)  	# e0_T_SecR


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

