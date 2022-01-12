"""
***************************************************
Output
***************************************************
"""

__all__ = ['writeResults']

def createDict(X, LABEL):
    X_DICT = {}
      
    for i in range(0,len(X)):
        X_DICT[LABEL[i]] = X[i]
        
    return X_DICT

#def writeBoxes2NPZ(dict_options, dict_variables, res_solver)


def get_file_name(dict_options, sampling_options=None, solv_options=None):
    
    
    filename = dict_options["fileName"]
    
    # contraction methods:
    if dict_options["bc_method"] == "bnormal": filename += "_bc"
    if dict_options["hc_method"] == "HC4": filename += "_hc4"        
    if dict_options["newton_method"] == "newton": 
        filename += "_n"
        # interval newton preconditioning:
        if dict_options["newton_point"] == "center": filename += "c"
        elif dict_options["newton_point"] == "3P": filename +="3P"
        elif dict_options["newton_point"] == "condJ": filename +="cJ"
        
        if dict_options["preconditioning"] == "all_functions": 
            filename += "af"
        elif dict_options["preconditioning"] == "inverse_centered": 
            filename +="ic"
        elif dict_options["preconditioning"] == "inverse_point": 
            filename +="ip"        
        elif dict_options["preconditioning"] == "diag_inverse_centered": 
            filename +="dic" 
                
    # tightening bounds:                     
    if dict_options["Affine_arithmetic"]: filename += "_aff"
    if dict_options["tight_bounds"]: filename += "_tb"
    
    # splitting methods:
    if dict_options["split_Box"] == "TearVar": filename += "_tv"
    if dict_options["split_Box"] == "LargestDer": filename += "_ld"
    if dict_options["split_Box"] == "forecastSplit": filename += "_fcs"
    if dict_options["split_Box"] == "LeastChanged": filename += "_lc"
    
    # cutting methods:
    if dict_options["cut_Box"]== "tear": filename += "_cbtv"
    if dict_options["cut_Box"]== "all": 
        filename += "_cba"
    
    # considering gaps in splitting:    
    if dict_options["consider_disconti"]: filename += "_gap"
    
    # decomposition for solving:
    if "decomp" in dict_options.keys():
        if dict_options["decomp"] =="DM": filename += "_dm"
        if dict_options["decomp"] =="BBTF": filename += "_bbtf"
    
    # hyper parameters for box reduction:
    filename += "_rs" + str(dict_options["resolution"])
    filename += "_r" + str(dict_options["redStepMax"])
    if dict_options["Parallel Branches"]:
        filename += "_pb" + str(dict_options["CPU count Branches"])
    #if dict_options["Parallel Variables"]:
    #    filename += "_pb" + str(dict_options["CPU count Variables"])
        
    # sampling methods and settings:    
    if sampling_options != None and sampling_options["number of samples"] != 0: 
        if sampling_options["sampling method"] == "sobol":
            filename += "_sb"
        elif sampling_options["sampling method"] == "hammersley":
            filename += "_hs"      
        elif sampling_options["sampling method"] == "latin_hypercube":
            filename += "_lhc"            
        filename += "_s" +str(sampling_options["number of samples"])
        filename += "_smin" +str(sampling_options["sampleNo_min_resiudal"])
        
    # numerical solvers:
    if not solv_options is None:
        if solv_options["solver"] == "newton": filename += "_nwt"
        elif solv_options["solver"] == "SLSQP": filename += "_slsqp"
        elif solv_options["solver"] == "TNC": filename += "_tnc"
        elif solv_options["solver"] == "fsolve": filename += "_fslv"
        elif solv_options["solver"] == "ipopt": filename += "_ipopt"
        elif solv_options["solver"] == "matlab-fsolve": filename += "_mfslv"
        elif solv_options["solver"] == "matlab-fsolve-mscript": filename += "_mfslvscrpt"
        elif solv_options["solver"] == "casadi-ipopt": filename += "_casipopt"
    
    return filename
    
    
def writeResults(dict_options, dict_variables, res_solver):
    """ creates File(s) with  final start values and lower and upper bounds. 
    For the ith set of start values, lower and upper bounds a new text file is 
    generated using the label "_i". 
    
    Args:
        :dict_options:            Dictionary with user settings
        :dict_variables:          Dictionary with sets of state variable values,
                                  lower and upper bounds
        :res_solver:              Dictionary with results from procedure              
        
    """
    t = res_solver["time"]
    iterNo = res_solver["iterNo"]
    fileName = dict_options["fileName"] 
    if "unified" in res_solver.keys(): 
        k=0
        res_unified = res_solver["unified"]
    if dict_variables[list(dict_variables)[0]][0] != []:
        initKey = list(dict_variables)[0]
    
    
        for i in range(0, len(dict_variables[initKey][0])):
            res_file = open(''.join([fileName,"_", str(i+1),".txt"]), "w") 
            
            res_file.write("***** %s th Set of Reduced Bounds*****\n\n"%(i+1))
            if "unified" in res_solver.keys():
                if res_unified["boxes_unified"][i]:
                    res_file.write("Boxes have been unified, as each subbox solved the system in the set tolerances:\n")                   
                    epsilon_uni = res_unified["epsilon_uni"][k]
                    variable = res_solver["Model"].xSymbolic[res_unified["var_id"][k]]
                    res_file.write("relTol = %s \nabsTol = %s \n"%(dict_options["relTol"], 
                                                                  dict_options["absTol"]))
                    res_file.write("But variable %s only fulfills:\ntol = %s \n"%(variable, epsilon_uni))
                    res_file.write("for tol = relTol = absTol = w(interval(%s)/(1+|interval(%s)|)\n\n"%(variable,
                                                                                                variable))
                    
                    k+=1
            if res_solver["Model"].failed:
                res_file.write("!!! Caution: Variable Reduction failed !!! \n Output equals the last reducable interval set.\n\n")
                
            for var in list(dict_variables):
                res_file.write("%s %s %s %s\n"%(var,
                                                dict_variables[var][0][i], 
                                                dict_variables[var][1][i],
                                                dict_variables[var][2][i]))     
            res_file.close()
    if t != []:        
        res_file = open(''.join([fileName,"_efficiency.txt"]), "w") 
        res_file.write("Time for iteration in s: %s \n"%(t))
        res_file.write("Number of outter iterations: %s\n"%(iterNo))