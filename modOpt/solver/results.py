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
    
    fileName = getFileName(dict_options)
    res_file = open(''.join([fileName, "_initial.txt"]), "w") 
    writeRestructuringSettings(res_file, dict_options)
    writeSolverSettings(res_file, solv_options)
    writeIterVarValues(res_file, model)


def writeResults(dict_options, res_solver):
    """ writes iteration variable results to file res_file
    
    Args:
        :dict_options:          dictionary with user specified settings
        :res_solver:            dictionary with solver output
        
    """
    
    fileName = getFileName(dict_options)
    
    res_file = open(''.join([fileName, "_results.txt"]), "w")
    writeIterVarValues(res_file,  res_solver["Model"])


def writeResultsAnalytics(dict_options, res_solver):
    """ writes additional iteration  information to file res_file
    
    Args:
        :dict_options:          dictionary with user specified settings
        :res_solver:            dictionary with solver output
        
    """

    fileName = getFileName(dict_options)
    res_file = open(''.join([fileName,"_analysis.txt"]), "w")
    writeSolverOutput(res_file, res_solver)
    writeFunctionLegend(res_file, res_solver["Model"])
    writeFunctionTable(res_file, res_solver)
    

def writeRestructuringSettings(res_file, dict_options):
    """ writes Restructuring settings to file res_file
    
    Args:
        :res_file:          text document
        :dict_options:      dictionary with user specified settings
            
    """
    
    res_file.write(" ****************** Restructuring Settings ****************** \n\n")
    res_file.write("Decomposition Method: %s\n"%(dict_options["decomp"]))
    res_file.write("Scaling Method: %s\n\n"%(dict_options["scaling"]))


def writeSolverSettings(res_file, solv_options):
    res_file.write(" ****************** Solver Settings ****************** \n\n")
    res_file.write("Solver: %s\n"%(solv_options["solver"]))
    res_file.write("Function Residual: %s\n"%(solv_options["FTOL"]))
    res_file.write("Maximum iteration number per Block: %s\n\n"%(solv_options["iterMax"]))
    
    
def writeIterVarValues(res_file, model):
    """ writes final iteration variable values to file res_file
    
    Args:
        :res_file:          text document
        :model:             object of class Model
    
    """
    
    res_file.write(" ****************** Iteration Variable Values ****************** \n\n")    
    for i in range(0, len(model.xSymbolic)):
        res_file.write("%s    %s\n"%(model.xSymbolic[i], model.stateVarValues[0][i]))
    res_file.write("\n")  

        
def writeFunctionLegend(res_file, model):
    """ writes symbolic functions of model and their global indices to file res_file
    
    Args:
        :res_file:          text document
        :model:             object of class Model
    
    """
    
    res_file.write(" ****************** Legend of functions ****************** \n\n") 
    res_file.write("Global ID:    Function Expression\n") 
    
    for i in range(0, len(model.fSymbolic)):
        res_file.write("%s:    %s\n"%(i, model.fSymbolic[i]))
    res_file.write("\n")  


def writeFunctionTable(res_file, res_solver):
    """ creates a table in file res_file which contains function residuals, 
    their exit flag, block index and the final number of iteration steps until
    the solver terminated. The iteration steps refers to complete block that
    has been solved simultaneously.
    
    Args:
        :res_file:          text document
        :res_solver:        dictionary with solver output
    
    """
    
    model = res_solver["Model"]
    blockID = model.getBlockID()    
    exitflag = getQuantityForFunction(res_solver["Exitflag"], blockID)
    iterNo = getQuantityForFunction(res_solver["IterNo"], blockID)
    funVal = model.getFunctionValues()
    
    res_file.write(" ****************** Table with Function Results ****************** \n\n") 
    res_file.write("GLbID  BlockID  Exitflag  IterNo  Residual\n") 
    
    for i in range(0, len(model.fSymbolic)):
        res_file.write("%s  %s  %s  %s  %s\n"%(i, 
                                               blockID[i], 
                                               exitflag[i], 
                                               iterNo[i], 
                                               funVal[i]))


def getQuantityForFunction(blockList, blockID):
    """ get a quantities of the functions in global order referring to their block ID
    
    Args:
        :blocklist:         list with block quanities
        :blockID:           list with blockID of functions in global order
    
    Return:
        :functionList:      list with function quantities in global order
    
    """
    
    functionList = []
    for b in blockID:
        functionList.append(blockList[b])
        
    return functionList

    
def writeSolverOutput(res_file, res_solver):
    """ writes output data of the solver to file res_file
    
    Args:
        :res_file:          text document
        :res_solver:        dictionary with solver output
    
    """
    
    res_file.write(" ****************** Solver Output ****************** \n\n") 
    res_file.write("Iteration number: %s\n"%(res_solver["IterNo_tot"]))
    res_file.write("Total Residual: %s\n\n"%(res_solver["Residual"])) 


def getFileName(dict_options):
    """ returns file name as string due to the chosen restructuring methods
        by the user stored in dict_options.
    
    Args:
        :dict_options:      dictionary with user specified settings
     
    Return:                 string with text file name
    
    """
    
    name = dict_options["fileName"]
    
    if dict_options["decomp"] =='None': name = ''.join([name, '_org'])
    if dict_options["decomp"] =='DM': name = ''.join([name,'_DM'])
    if dict_options["decomp"] =='BBTF': name = ''.join([name,'_DM'])
    if dict_options["scaling"] =='MC29': name = ''.join([name,'_MC29'])
    if dict_options["scaling"] =='MC77': name = ''.join([name,'_MC77'])
    if dict_options["scaling"] =='Inf RowSca and Mean ColSca': name = ''.join([name,'_InfMean'])
    if dict_options["scaling"] =='Inf RowSca and gMean ColSca': name = ''.join([name,'_InfgMean'])
    if dict_options["scaling procedure"] =='tot_init': name = ''.join([name,'_totInit'])  
    if dict_options["scaling procedure"] =='block_init': name = ''.join([name,'_blcInit']) 
    if dict_options["scaling procedure"] =='block_iter': name = ''.join([name,'_blcIter']) 
    
    return name