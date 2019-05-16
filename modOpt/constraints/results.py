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
    
    if dict_variables[dict_variables.keys()[0]][0] != []:
        initKey = dict_variables.keys()[0]
    
    
        for i in range(0, len(dict_variables[initKey][0])):
            res_file = open(''.join([fileName,"_", str(i+1),".txt"]), "w") 
            
            res_file.write("***** %s th Set of Initial Values and Bounds*****\n\n"%(i+1))
    
            for var in dict_variables.keys():
                res_file.write("%s %s %s %s\n"%(var,
                                                dict_variables[var][0][i], 
                                                dict_variables[var][1][i],
                                                dict_variables[var][2][i]))     
            res_file.close()
    if t != []:        
        res_file = open(''.join([fileName,"_efficiency.txt"]), "w") 
        res_file.write("Time for iteration in s: %s \n"%(t))
        res_file.write("Number of outter iterations: %s\n"%(iterNo))