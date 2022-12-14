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
    if dict_options["redStepMax"] != 0:
        if "bcMethod" in dict_options.keys(): 
            if dict_options["bcMethod"] == "bnormal": filename += "_bc"
        if "hcMethod" in dict_options.keys(): 
            if dict_options["hcMethod"] == "HC4": filename += "_hc4" 
        if "newtonMethod" in dict_options.keys():       
            if dict_options["newtonMethod"] == "newton": 
                filename += "_n"
                # interval newton preconditioning:
                if dict_options["newtonPoint"] == "center": filename += "c"
                elif dict_options["newtonPoint"] == "3P": filename +="3P"
                elif dict_options["newtonPoint"] == "condJ": filename +="cJ"
                
                if dict_options["preconditioning"] == "pivotAll": 
                    filename += "af"
                elif dict_options["preconditioning"] == "inverseCentered": 
                    filename +="ic"
                elif dict_options["preconditioning"] == "inversePoint": 
                    filename +="ip"        
                elif dict_options["preconditioning"] == "diagInverseCentered": 
                    filename +="dic" 
                    
        # tightening bounds:                     
        if "affineArithmetic" in dict_options.keys(): 
            if dict_options["affineArithmetic"]: filename += "_aff"
        if "tightBounds" in dict_options.keys(): 
            if dict_options["tightBounds"]: filename += "_tb"
        
        # splitting methods:
        if "splitBox" in dict_options.keys(): 
            if dict_options["splitBox"] == "tearVar": filename += "_tv"
            if dict_options["splitBox"] == "largestDer": filename += "_ld"
            if dict_options["splitBox"] == "forecastSplit": filename += "_fcs"
            if dict_options["splitBox"] == "leastChanged": filename += "_lc"
            if dict_options["splitBox"] == "forecastTear": filename += "_ftv"
        
        # cutting methods:
        if "cutBox" in dict_options.keys(): 
            if dict_options["cutBox"]== "tear": filename += "_cbtv"
            if dict_options["cutBox"]== "all": 
                filename += "_cba"
        
        # considering gaps in splitting:    
        if "considerDisconti" in dict_options.keys():     
            if dict_options["considerDisconti"]: filename += "_gap"
    
    # decomposition for solving:
    if "decomp" in dict_options.keys():
        if dict_options["decomp"] =="DM": filename += "_dm"
        if dict_options["decomp"] =="BBTF": filename += "_bbtf"
    
    # hyper parameters for box reduction:
    if dict_options["redStepMax"] != 0:
        filename += "_rs" + str(dict_options["resolution"])
        filename += "_r" + str(dict_options["redStepMax"])
        if "parallelBoxes" in dict_options.keys():
            if dict_options["parallelBoxes"]:
                filename += "_pb" + str(dict_options["cpuCountBoxes"])

    # smplMethods and settings:    
    if sampling_options != None and sampling_options["smplNo"] > 0: 
        if sampling_options["smplMethod"] == "sobol":
            filename += "_sb"
        elif sampling_options["smplMethod"] == "hammersley":
            filename += "_hs"      
        elif sampling_options["smplMethod"] == "latin_hypercube":
            filename += "_lhc"            
        filename += "_s" +str(sampling_options["smplNo"])
        filename += "_smin" +str(sampling_options["smplNoBest"])
        
    # numerical solvers:
    if not solv_options is None:
        if solv_options["solver"] == "newton": 
            filename += "_nwt"
            if solv_options["scaling"] == "MC77": filename += "_mc77"
            elif solv_options["scaling"] == "MC29": filename += "_mc29"
        elif solv_options["solver"] == "SLSQP": filename += "_slsqp"
        elif solv_options["solver"] == "TNC": filename += "_tnc"
        elif solv_options["solver"] == "fsolve": filename += "_fslv"
        elif solv_options["solver"] == "ipopt": filename += "_ipopt"
        elif solv_options["solver"] == "matlab-fsolve": filename += "_mfslv"
        elif solv_options["solver"] == "matlab-fsolve-mscript": filename += "_mfslvscrpt"
        elif solv_options["solver"] == "casadi-ipopt": filename += "_casipopt"
        elif solv_options["solver"] == "hybr": filename +="_hybr"
        elif solv_options["solver"] == "lm": filename +="_lm"
        elif solv_options["solver"] == "broyden1": filename +="_bd1"
        elif solv_options["solver"] == "broyden2": filename +="_bd2"
        elif solv_options["solver"] == "krylov": filename +="_krlv"
        
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