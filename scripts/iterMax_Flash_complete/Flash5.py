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
    dict_options = {"fileName": "Flash5",
                    "iterMaxNewton": 5,
                    "machEpsRelNewton": 2.22e-16,
                    "machEpsAbsNewton": 2.22e-16,
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
def fun_84405(std_T,std_A,std_B,std_C): # fun_84405__saturationPressurePureComponent
	return ((10.0))**(std_A-(std_B)/(std_C+std_T))
def fun_84404(std_T,std_greek_Deltah,std_A,std_B,std_C,std_D,std_E,std_F): # fun_84404__molarEnthalpyPureComponent
	return (std_A) *((std_T)/(1000.0))+((std_B)/(2.0)) *((((std_T)/(1000.0)))**(2.0))+((std_C)/(3.0)) *((((std_T)/(1000.0)))**(3.0))+((std_D)/(4.0)) *((((std_T)/(1000.0)))**(4.0))+(std_E) *((((std_T)/(1000.0)))**(-1.0))+std_F+std_greek_Deltah

def getSymbolicFunctions(x, p):
    """ User specified Equation system 
    
    Args:
        x:          sympy array with symbolic state variable bounds
        p:          numpy array with parameter values
    
    Returns:
        f:          sympy array with symbolic residual bounds
        
    """
# Getting variable values:
    e0_K_LV_i1 = x[0]
    e0_h_V = x[1]
    e0_p = x[2]
    e0_x_i1 = x[3]
    e0_x_i2 = x[4]
    e0_K_LV_i2 = x[5]
    e0_x_F_i2 = x[6]
    e0_y_i1 = x[7]
    e0_y_i2 = x[8]
    e0_greek_alpha_L_i1 = x[9]
    e0_greek_alpha_L_i2 = x[10]
    e0_greek_gamma_L_i1 = x[11]
    e0_greek_gamma_L_i2 = x[12]
    e0_h_F = x[13]
    e0_L = x[14]
    e0_Q = x[15]
    e0_V = x[16]
    e0_HU_i1 = x[17]
    e0_HU_i2 = x[18]
    e0_HU_L = x[19]
    e0_HU_V = x[20]
    e0_U = x[21]
    e0_V_L = x[22]
    e0_V_V = x[23]
    e0_V_total = x[24]
    e0_A = x[25]
    e0_r_DH = x[26]
    e0_h_L = x[27]



# Getting parameter values:
    e0_greek_Deltap = p[0]
    e0_p_F = p[1]
    e0_x_F_i1 = p[2]
    e0_T = p[3]
    e0_T_F = p[4]
    e0_h_EF = p[5]
    e0_h_EL = p[6]
    e0_h_EV = p[7]
    e0_F = p[8]
    e0_greek_pi = p[9]
    e0_D = p[10]
    e0_H_L = p[11]
    e0_v_EL = p[12]
    e0_z_V = p[13]
    e0_H = p[14]
    e0_greek_DeltahhL_i1 = p[15]
    e0_greek_DeltahhL_i2 = p[16]
    e0_greek_DeltahhV_i1 = p[17]
    e0_greek_DeltahhV_i2 = p[18]
    e0_greek_lambda_L_i1 = p[19]
    e0_greek_lambda_L_i2 = p[20]
    e0_AhL_i1 = p[21]
    e0_AhL_i2 = p[22]
    e0_AhV_i1 = p[23]
    e0_AhV_i2 = p[24]
    e0_ApLV_i1 = p[25]
    e0_ApLV_i2 = p[26]
    e0_BhL_i1 = p[27]
    e0_BhL_i2 = p[28]
    e0_BhV_i1 = p[29]
    e0_BhV_i2 = p[30]
    e0_BpLV_i1 = p[31]
    e0_BpLV_i2 = p[32]
    e0_ChL_i1 = p[33]
    e0_ChL_i2 = p[34]
    e0_ChV_i1 = p[35]
    e0_ChV_i2 = p[36]
    e0_CpLV_i1 = p[37]
    e0_CpLV_i2 = p[38]
    e0_DhL_i1 = p[39]
    e0_DhL_i2 = p[40]
    e0_DhV_i1 = p[41]
    e0_DhV_i2 = p[42]
    e0_EhL_i1 = p[43]
    e0_EhL_i2 = p[44]
    e0_EhV_i1 = p[45]
    e0_EhV_i2 = p[46]
    e0_FhL_i1 = p[47]
    e0_FhL_i2 = p[48]
    e0_FhV_i1 = p[49]
    e0_FhV_i2 = p[50]
    e0_R = p[51]
    e0_v_L_i1 = p[52]
    e0_v_L_i2 = p[53]



   # Getting function values:
    e0_p_LV_i2 = fun_84405(e0_T,e0_ApLV_i2,e0_BpLV_i2,e0_CpLV_i2)
    e0_p_LV_i1 = fun_84405(e0_T,e0_ApLV_i1,e0_BpLV_i1,e0_CpLV_i1)
    e0_h_L_i2 = fun_84404(e0_T,e0_greek_DeltahhL_i2,e0_AhL_i2,e0_BhL_i2,e0_ChL_i2,e0_DhL_i2,e0_EhL_i2,e0_FhL_i2)
    e0_h_V_i2 = fun_84404(e0_T,e0_greek_DeltahhV_i2,e0_AhV_i2,e0_BhV_i2,e0_ChV_i2,e0_DhV_i2,e0_EhV_i2,e0_FhV_i2)
    e0_h_F_i2 = fun_84404(e0_T_F,e0_greek_DeltahhL_i2,e0_AhL_i2,e0_BhL_i2,e0_ChL_i2,e0_DhL_i2,e0_EhL_i2,e0_FhL_i2)
    e0_h_L_i1 = fun_84404(e0_T,e0_greek_DeltahhL_i1,e0_AhL_i1,e0_BhL_i1,e0_ChL_i1,e0_DhL_i1,e0_EhL_i1,e0_FhL_i1)
    e0_h_F_i1 = fun_84404(e0_T_F,e0_greek_DeltahhL_i1,e0_AhL_i1,e0_BhL_i1,e0_ChL_i1,e0_DhL_i1,e0_EhL_i1,e0_FhL_i1)
    e0_h_V_i1 = fun_84404(e0_T,e0_greek_DeltahhV_i1,e0_AhV_i1,e0_BhV_i1,e0_ChV_i1,e0_DhV_i1,e0_EhV_i1,e0_FhV_i1)



# Solve equation system for given x:
    f= [
    e0_K_LV_i1-((e0_y_i1)/(e0_x_i1)) ,
    e0_K_LV_i2-((e0_y_i2)/(e0_x_i2)) ,
    (e0_y_i1+e0_y_i2)-(1.0) ,
    (e0_x_i1+e0_x_i2)-(1.0) ,
    (e0_x_F_i1+e0_x_F_i2)-(1.0) ,
    e0_h_F-(((e0_x_F_i1) *(e0_h_F_i1)+e0_h_EF+(e0_x_F_i2) *(e0_h_F_i2)+e0_h_EF)) ,
    e0_h_L-(((e0_x_i1) *(e0_h_L_i1)+e0_h_EL+(e0_x_i2) *(e0_h_L_i2)+e0_h_EL)) ,
    e0_h_V-(((e0_y_i1) *(e0_h_V_i1)+e0_h_EV+(e0_y_i2) *(e0_h_V_i2)+e0_h_EV)) ,
    e0_greek_Deltap-(e0_p_F-e0_p) ,
    e0_K_LV_i1-(((e0_greek_gamma_L_i1) *(e0_p_LV_i1))/(e0_p)) ,
    e0_K_LV_i2-(((e0_greek_gamma_L_i2) *(e0_p_LV_i2))/(e0_p)) ,
    e0_greek_gamma_L_i1-(((1.0)/(e0_x_i1+(e0_greek_alpha_L_i1) *((1.0-e0_x_i1)))) *(sympy.exp(((1.0-e0_x_i1)) *(((e0_greek_alpha_L_i1)/(e0_x_i1+(e0_greek_alpha_L_i1) *((1.0-e0_x_i1)))-((e0_greek_alpha_L_i1+e0_greek_alpha_L_i2)-e0_greek_alpha_L_i1)/((((e0_greek_alpha_L_i1+e0_greek_alpha_L_i2)-e0_greek_alpha_L_i1)) *(e0_x_i1)+(1.0-e0_x_i1))))))) ,
    e0_greek_gamma_L_i2-(((1.0)/(e0_x_i2+(e0_greek_alpha_L_i2) *((1.0-e0_x_i2)))) *(sympy.exp(((1.0-e0_x_i2)) *(((e0_greek_alpha_L_i2)/(e0_x_i2+(e0_greek_alpha_L_i2) *((1.0-e0_x_i2)))-((e0_greek_alpha_L_i1+e0_greek_alpha_L_i2)-e0_greek_alpha_L_i2)/((((e0_greek_alpha_L_i1+e0_greek_alpha_L_i2)-e0_greek_alpha_L_i2)) *(e0_x_i2)+(1.0-e0_x_i2))))))) ,
    e0_greek_alpha_L_i1-((((e0_v_L_i1+e0_v_L_i2)-e0_v_L_i1)/(e0_v_L_i1)) *(sympy.exp((-e0_greek_lambda_L_i1)/(e0_T)))) ,
    e0_greek_alpha_L_i2-((((e0_v_L_i1+e0_v_L_i2)-e0_v_L_i2)/(e0_v_L_i2)) *(sympy.exp((-e0_greek_lambda_L_i2)/(e0_T)))) ,
    0.0-((e0_F) *(e0_h_F)-(e0_V) *(e0_h_V)-(e0_L) *(e0_h_L)+e0_Q) ,
    0.0-((e0_F) *(e0_x_F_i1)-(e0_V) *(e0_y_i1)-(e0_L) *(e0_x_i1)) ,
    0.0-((e0_F) *(e0_x_F_i2)-(e0_V) *(e0_y_i2)-(e0_L) *(e0_x_i2)) ,
    e0_U-((e0_HU_L) *((e0_h_L-(e0_p) *((100000.0) *((((e0_x_i1) *(e0_v_L_i1)+(e0_x_i2) *(e0_v_L_i2))+e0_v_EL)))))+(e0_HU_V) *((e0_h_V-(e0_R) *((e0_T) *(e0_z_V))))) ,
    e0_V_total-(e0_V_L+e0_V_V) ,
    e0_HU_i1-((e0_x_i1) *(e0_HU_L)+(e0_y_i1) *(e0_HU_V)) ,
    e0_HU_i2-((e0_x_i2) *(e0_HU_L)+(e0_y_i2) *(e0_HU_V)) ,
    e0_V_L-(((((e0_v_L_i1) *(e0_x_i1)+(e0_v_L_i2) *(e0_x_i2))+e0_v_EL)) *(e0_HU_L)) ,
    e0_V_V-(((e0_HU_V) *((e0_R) *((e0_T) *(e0_z_V))))/((e0_p) *(100000.0))) ,
    e0_H_L-((4.0) *((e0_HU_L)/((e0_greek_pi) *(((e0_D))**(2.0))))) ,
    e0_A-(((e0_greek_pi)/(4.0)) *(((e0_D))**(2.0))) ,
    e0_V_total-((e0_A) *(e0_H)) ,
    e0_r_DH-((e0_D)/(e0_H)) 

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

    x = numpy.empty(28)
    xInitial = numpy.empty((28), dtype = object) 
    parameter = numpy.empty(54)
    dict_variables = {}

    # Iteration variable initializatio
    x[0] = 1.1 	# e0_K_LV_i1
    x[1] = 1.1 	# e0_h_V
    x[2] = 1.1 	# e0_p
    x[3] = 1.1 	# e0_x_i1
    x[4] = 1.1 	# e0_x_i2
    x[5] = 1.1 	# e0_K_LV_i2
    x[6] = 1.1 	# e0_x_F_i2
    x[7] = 1.1 	# e0_y_i1
    x[8] = 1.1 	# e0_y_i2
    x[9] = 1.1 	# e0_greek_alpha_L_i1
    x[10] = 1.1 	# e0_greek_alpha_L_i2
    x[11] = 1.1 	# e0_greek_gamma_L_i1
    x[12] = 1.1 	# e0_greek_gamma_L_i2
    x[13] = 1.1 	# e0_h_F
    x[14] = 1.1 	# e0_L
    x[15] = 1.1 	# e0_Q
    x[16] = 1.1 	# e0_V
    x[17] = 1.1 	# e0_HU_i1
    x[18] = 1.1 	# e0_HU_i2
    x[19] = 1.1 	# e0_HU_L
    x[20] = 1.1 	# e0_HU_V
    x[21] = 1.1 	# e0_U
    x[22] = 1.1 	# e0_V_L
    x[23] = 1.1 	# e0_V_V
    x[24] = 1.1 	# e0_V_total
    x[25] = 1.1 	# e0_A
    x[26] = 1.1 	# e0_r_DH
    x[27] = 1.1 	# e0_h_L


    # Constant parameter setting:
    parameter[0] = 0.25 	# e0_greek_Deltap
    parameter[1] = 1.0 	# e0_p_F
    parameter[2] = 0.15 	# e0_x_F_i1
    parameter[3] = 353.15 	# e0_T
    parameter[4] = 353.15 	# e0_T_F
    parameter[5] = 0.0 	# e0_h_EF
    parameter[6] = 0.0 	# e0_h_EL
    parameter[7] = 0.0 	# e0_h_EV
    parameter[8] = 80.0 	# e0_F
    parameter[9] = 3.14159265359 	# e0_greek_pi
    parameter[10] = 0.16 	# e0_D
    parameter[11] = 0.2 	# e0_H_L
    parameter[12] = 0.0 	# e0_v_EL
    parameter[13] = 1.0 	# e0_z_V
    parameter[14] = 0.5 	# e0_H
    parameter[15] = -276000.0 	# e0_greek_DeltahhL_i1
    parameter[16] = -285830.0 	# e0_greek_DeltahhL_i2
    parameter[17] = -234000.0 	# e0_greek_DeltahhV_i1
    parameter[18] = -276000.0 	# e0_greek_DeltahhV_i2
    parameter[19] = 95.68 	# e0_greek_lambda_L_i1
    parameter[20] = 506.7 	# e0_greek_lambda_L_i2
    parameter[21] = 102538.0 	# e0_AhL_i1
    parameter[22] = -203.606 	# e0_AhL_i2
    parameter[23] = 5385.58 	# e0_AhV_i1
    parameter[24] = 30.092 	# e0_AhV_i2
    parameter[25] = 5.24677 	# e0_ApLV_i1
    parameter[26] = 5.0768 	# e0_ApLV_i2
    parameter[27] = -138.44 	# e0_BhL_i1
    parameter[28] = 1523.29 	# e0_BhL_i2
    parameter[29] = 236.1088 	# e0_BhV_i1
    parameter[30] = 6.832514 	# e0_BhV_i2
    parameter[31] = 1598.673 	# e0_BpLV_i1
    parameter[32] = 1659.793 	# e0_BpLV_i2
    parameter[33] = -0.03469 	# e0_ChL_i1
    parameter[34] = -3196.413 	# e0_ChL_i2
    parameter[35] = 0.1237 	# e0_ChV_i1
    parameter[36] = 6.793435 	# e0_ChV_i2
    parameter[37] = -46.424 	# e0_CpLV_i1
    parameter[38] = -45.854 	# e0_CpLV_i2
    parameter[39] = 20.4367 	# e0_DhL_i1
    parameter[40] = 2474.455 	# e0_DhL_i2
    parameter[41] = 2.3E-5 	# e0_DhV_i1
    parameter[42] = -2.53448 	# e0_DhV_i2
    parameter[43] = 0.0 	# e0_EhL_i1
    parameter[44] = 3.855326 	# e0_EhL_i2
    parameter[45] = 3.7E-5 	# e0_EhV_i1
    parameter[46] = 0.082139 	# e0_EhV_i2
    parameter[47] = 0.0 	# e0_FhL_i1
    parameter[48] = -256.5478 	# e0_FhL_i2
    parameter[49] = 0.0 	# e0_FhV_i1
    parameter[50] = -250.881 	# e0_FhV_i2
    parameter[51] = 8.314 	# e0_R
    parameter[52] = 5.869E-5 	# e0_v_L_i1
    parameter[53] = 1.807E-5 	# e0_v_L_i2


    # Get symbolic quantities:
    xSymbolic = sympy.symbols('e0_K_LV_i1 e0_h_V e0_p e0_x_i1 e0_x_i2 e0_K_LV_i2 e0_x_F_i2 e0_y_i1 e0_y_i2 e0_greek_alpha_L_i1 e0_greek_alpha_L_i2 e0_greek_gamma_L_i1 e0_greek_gamma_L_i2 e0_h_F e0_L e0_Q e0_V e0_HU_i1 e0_HU_i2 e0_HU_L e0_HU_V e0_U e0_V_L e0_V_V e0_V_total e0_A e0_r_DH e0_h_L ')
    fSymbolic = getSymbolicFunctions(xSymbolic, parameter)

    # Intial variable bounds:
    xInitial[0] = mpmath.mpi(0.0, 5.0)  	# e0_K_LV_i1
    xInitial[1] = mpmath.mpi(-280000.0, -230000.0)  	# e0_h_V
    xInitial[2] = mpmath.mpi(0.0, 5.0)  	# e0_p
    xInitial[3] = mpmath.mpi(0.0, 1.0)  	# e0_x_i1
    xInitial[4] = mpmath.mpi(0.0, 1.0)  	# e0_x_i2
    xInitial[5] = mpmath.mpi(0.0, 5.0)  	# e0_K_LV_i2
    xInitial[6] = mpmath.mpi(0.0, 1.0)  	# e0_x_F_i2
    xInitial[7] = mpmath.mpi(0.0, 1.0)  	# e0_y_i1
    xInitial[8] = mpmath.mpi(0.0, 1.0)  	# e0_y_i2
    xInitial[9] = mpmath.mpi(0.0, 1.0)  	# e0_greek_alpha_L_i1
    xInitial[10] = mpmath.mpi(0.0, 1.0)  	# e0_greek_alpha_L_i2
    xInitial[11] = mpmath.mpi(0.0, 5.0)  	# e0_greek_gamma_L_i1
    xInitial[12] = mpmath.mpi(0.0, 5.0)  	# e0_greek_gamma_L_i2
    xInitial[13] = mpmath.mpi(-290000.0, -230000.0)  	# e0_h_F
    xInitial[14] = mpmath.mpi(0.0, 80.0)  	# e0_L
    xInitial[15] = mpmath.mpi(0.0, 200000.0)  	# e0_Q
    xInitial[16] = mpmath.mpi(0.0, 80.0)  	# e0_V
    xInitial[17] = mpmath.mpi(0.0, 0.2)  	# e0_HU_i1
    xInitial[18] = mpmath.mpi(0.0, 0.2)  	# e0_HU_i2
    xInitial[19] = mpmath.mpi(0.0, 0.2)  	# e0_HU_L
    xInitial[20] = mpmath.mpi(0.0, 0.3)  	# e0_HU_V
    xInitial[21] = mpmath.mpi(-1000000.0, 0.0)  	# e0_U
    xInitial[22] = mpmath.mpi(0.0, 0.1)  	# e0_V_L
    xInitial[23] = mpmath.mpi(0.0, 0.1)  	# e0_V_V
    xInitial[24] = mpmath.mpi(0.0, 0.1)  	# e0_V_total
    xInitial[25] = mpmath.mpi(0.0, 0.1)  	# e0_A
    xInitial[26] = mpmath.mpi(0.0, 1.0)  	# e0_r_DH
    xInitial[27] = mpmath.mpi(-290000.0, -230000.0)  	# e0_h_L


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

