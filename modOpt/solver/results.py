"""
***************************************************
Output
***************************************************
"""

__all__ = ['writeInitialSettings', 'writeResults', 'writeResultsAnalytics']


def writeInitialSettings(dict_options, solv_options, model):
    """ creates File(s) with  final start values and lower and upper bounds. 
    For the ith set of start values, lower and upper bounds a new text file is 
    generated using the label "_i". 
    
    Args:
        :fileName:                String with name of text file(s)
        :dict_variables:          Dictionary with sets of state variable values,
                                 lower and upper bounds
        
    """
    res_file = open(''.join([dict_options["fileName"],"_initial.txt"]), "w") 
    writeRestructuringSettings(res_file, dict_options)
    writeSolverSettings(res_file, solv_options)
    writeIterVarValues(res_file, model)


def writeRestructuringSettings(res_file, dict_options):
    res_file.write("Restructuring Settings:\n")
    res_file.write("Decomposition Method: %s\n"%(dict_options["decomp"]))
    res_file.write("Scaling Method: %s\n\n"%(dict_options["scaling"]))


def writeSolverSettings(res_file, solv_options):
    res_file.write("Solver Settings:\n")
    res_file.write("Solver: %s\n"%(solv_options["solver"]))
    res_file.write("Function Residual: %s\n"%(solv_options["FTOL"]))
    res_file.write("Maximum iteration number per Block: %s\n\n"%(solv_options["iterMax"]))
    
    
def writeIterVarValues(res_file, model):
    res_file.write("Iteration Variable Values:\n")    
    for i in range(0, len(model.xSymbolic)):
        res_file.write("%s    %s\n"%(model.xSymbolic[i], model.stateVarValues[0][i]))
    res_file.write("\n")  

        
def writeResults(dict_options, res_solver):
    res_file = open(''.join([dict_options["fileName"],"_results.txt"]), "w")
    writeIterVarValues(res_file,  res_solver["Model"])
    writeFuncResiduals(res_file, res_solver["Model"])


def writeResultsAnalytics(dict_options, res_solver):
    res_file = open(''.join([dict_options["fileName"],"_analysis.txt"]), "w")
    writeSolverOutput(res_file, res_solver)
    writeFunctionLegend(res_file, res_solver["Model"])
    writeFunctionTable(res_file, res_solver)

def writeFunctionLegend(res_file, model):
    res_file.write("Legend of functions:\n\n") 
    res_file.write("Global ID:    Function Expression\n") 
    
    for i in range(0, len(model.fSymbolic)):
        res_file.write("%s:    %s\n"%(i, model.fSymbolic[i]))
    res_file.write("\n")  


def writeFunctionTable(res_file, res_solver):
    
    model = res_solver["Model"]
    blockID = model.getBlockID()    
    exitflag = getQuantityForFunction(res_solver["Exitflag"], blockID)
    iterNo = getQuantityForFunction(res_solver["IterNo"], blockID)
    funVal = model.getFunctionValues()
    
    res_file.write("Table with Function Results:\n\n") 
    res_file.write("GLbID  BlockID  Exitflag  IterNo  Residual\n") 
    
    for i in range(0, len(model.fSymbolic)):
        res_file.write("%s  %s  %s  %s  %s\n"%(i, 
                                               blockID[i], 
                                               exitflag[i], 
                                               iterNo[i], 
                                               funVal[i]))

def getQuantityForFunction(blockList, blockID):
    functionList = []
    for b in blockID:
        functionList.append(blockList[b])
        
    return functionList
    
def writeSolverOutput(res_file, res_solver):
    res_file.write("Solver Output:\n")
    res_file.write("Iteration number: %s\n"%(res_solver["IterNo_tot"]))
    res_file.write("Total Residual: %s\n\n"%(res_solver["Residual"])) 
    #TODO exitflag

def writeFuncResiduals(res_file, model):
    res_file.write("Function Residual Values:\n") 
    fValues = model.getFunctionValues()
    for i in range(0, len(model.fSymbolic)):
        res_file.write("%s  =   %s\n"%(fValues[i], model.fSymbolic[i]))
    res_file.write("\n")  
#def writeFunctionResiduals(res_file, dict_equations):   
     