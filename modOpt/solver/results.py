"""
***************************************************
Import packages
***************************************************
"""
import numpy
import mpmath

"""
***************************************************
Output
***************************************************
"""


__all__ = ['writeInitialSettings', 'writeResults', 'writeResultsAnalytics', 
           'write_successfulResults', 'write_results_with_bounds',
           'write_initial_values_with_bounds', 'write_analytics', 'write_successful_results']

def write_successful_results(res_boxes, dict_options, sampling_options, solv_options):
    """ writes out results from sampling and solving procedure blockwise. 
    
    Args:
        :res_boxes:         dictionary with solver results sorted by variable boxes
        :dict_options:      dictionary with user-specified box reduction, model structure settings
        :sampling_options:  dictionary with user-specified sampling settings
        :solv_options:      dictionary with user-specified solver settings
    
    """
    
    if res_boxes == {}: return True

    for box in res_boxes.keys():
        res_box = res_boxes[box]
        model = res_box["Model"]
        if model.failed: return True
        solutions = model.stateVarValues
        bounds = model.xBounds[box]
        xSymbolic = model.xSymbolic

        for solution in solutions:
            write_solution(xSymbolic, solution, bounds, dict_options, solv_options)
            write_block_analysis(res_box, solution, model, 
                                 dict_options, solv_options)
        return True


def write_block_analysis(res_box, solution, model, dict_options, solv_options):
    """ writes out analysis of the blocks from blockwise sampling and solving 
    procedure . 
    
    Args:
        :res_box:           dictionary with results from one box sorted by blocks
        :dict_options:      dictionary with user-specified box reduction, model structure settings
        :sampling_options:  dictionary with user-specified sampling settings
        :solv_options:      dictionary with user-specified solver settings
        
    """
    
    fileName = getFileName(dict_options, solv_options)
    res_file = open(''.join([fileName, "_analysis.txt"]), "w")
    res_file.write(" ****************** Analysis File ****************** \n\n") 
    writeSolverSettings(res_file, solv_options)
    writeRestructuringSettings(res_file, dict_options)
    writeFunctionLegend(res_file, model)
    
    res_file.write(" ****************** Table with Function Results ****************** \n\n") 
    res_file.write("BlockID  GLbID   CondNo  Exitflag  IterNo  Residual\n") 
    for block in res_box.keys():
        if block == "Model": continue
        res_block = res_box[block]
        colPerm = res_block["colPerm"]
        sol_id = get_id_block_solution(solution[colPerm], res_block["FoundSolutions"])
        wirte_block_analysis_table(res_file, res_block, block, sol_id)

def wirte_block_analysis_table(res_file, res_block, block, sol_id):
    """ writes out table with blockwise results for blockwise sampling and solving 
    procedure . 
    
    Args:
        :res_file:          textfile object
        :res_box:           dictionary with results from one box sorted by blocks
        :block:             object of type block
        :sol_id:            integer with index of current solution in block
        
    """    

    exitflag = res_block["Exitflag"][sol_id]
    iterNo = res_block["IterNo"][sol_id]
    condNo = res_block["CondNo"][sol_id]
    FRES = res_block["FRES"][sol_id]
    rowPerm = res_block["rowPerm"]



    for i in range(0,len(rowPerm)):
        res_file.write("%s  %s  %s  %s  %s  %s\n"%(block, 
                                                 rowPerm[i],
                                                 condNo,
                                                 exitflag,
                                                 iterNo,
                                                 FRES[i]))    
    
    
    


def write_solution(xSymbolic, solution, bounds, dict_options, solv_options):
    """ writes out file wit solution and current bounds with blockwise sampling and solving 
    procedure . 
    
    Args:
        :xSymbolic:         list with symbolic variables
        :solution:          numpy array with current variable bounds
        :dict_options:      dictionary with user-specified box reduction, model structure settings
        :solv_options:      dictionary with user-specified solver settings
        
    """
    if not "sol_count" in dict_options.keys(): dict_options["sol_count"] = 1
    else: dict_options["sol_count"] +=1
    
    fileName = getFileName(dict_options, solv_options)
    res_file = open(''.join([fileName, "_", str(dict_options["sol_count"]),"_results.txt"]), "w")
    
    res_file.write(" ****************** Iteration Variable Values ****************** \n\n") 
    if isinstance(bounds[0], list) or isinstance(bounds[0], numpy.ndarray):
        for i in range(0, len(xSymbolic)):
            res_file.write("%s    %s    %s    %s\n"%(xSymbolic[i], 
                                                 solution[i],
                                                 bounds[i][0],
                                                 bounds[i][1]))
    else:
        for i in range(0, len(xSymbolic)):
            res_file.write("%s    %s    %s    %s\n"%(xSymbolic[i], 
                                                 solution[i],
                                                 float(mpmath.mpf(bounds[i].a)),
                                                 float(mpmath.mpf(bounds[i].b))))
        

def get_id_block_solution(solution_block, foundSolutions):
    """ gets id of a total system's solution in block.
    
    Args:
        :solution_block:        numpy.array with block sub-vector of current solution
        :foundSolutions:        list with numpy.arrays containing all block solutions
    
    Return:
        :sol_id:                id of current solution in block

    """
    
    for sol_id in range(0, len(foundSolutions)):
        if(solution_block == foundSolutions[sol_id]).all():
            return sol_id
    
    

def write_successfulResults(res_solver, mainfilename, k, l, initial_model, solv_options, dict_options):
    """ method writes successful sample into a text file 
    
    Args:
    :res_solver:    dictionary with results from solver for current sample
    :mainfilname:   string with general filename for text output
    :k:             integer with id of current sample
    :l:             integer with id of current box
    :initial_model: instance of type model with initial samplepoint
    :solv_options:  dictionary with solver settings
    :dict_options:  dictionary with user-specified decomposition and scaling
                    settings
                    
    Return:         None.
    
    """
    if (res_solver["Exitflag"] == numpy.ones(len(res_solver["Exitflag"]))).all():
        dict_options["fileName"] += "_b"+str(l)+"_s"+ str(k)
        writeInitialSettings(dict_options, solv_options, initial_model)
        writeResults(dict_options, solv_options, res_solver)
        writeResultsAnalytics(dict_options, res_solver, solv_options)
        dict_options["fileName"] = mainfilename
               
               
def writeInitialSettings(dict_options, solv_options, model):
    """ creates File(s) with  final start values and lower and upper bounds. 
    For the ith set of start values, lower and upper bounds a new text file is 
    generated using the label "_i". 
    
    Args:
        :fileName:                String with name of text file(s)
        :dict_variables:          Dictionary with sets of state variable values,
                                 lower and upper bounds
                                 
    Return:         None.  
    
    """
    fileName = getFileName(dict_options, solv_options)
    res_file = open(''.join([fileName, "_initial.txt"]), "w") 
    writeRestructuringSettings(res_file, dict_options)
    writeSolverSettings(res_file, solv_options)
    writeIterVarValues(res_file, model)


def writeResults(dict_options, solv_options, res_solver):
    """ writes iteration variable results to file res_file
    
    Args:
        :dict_options:          dictionary with user specified settings
        :solv_options:          dictionary with user specified solver settings
        :res_solver:            dictionary with solver output
    
    Return:         None.   
    
    """
    if res_solver != []: 
        fileName = getFileName(dict_options, solv_options)
    
        res_file = open(''.join([fileName, "_results.txt"]), "w")
        writeIterVarValues(res_file,  res_solver["Model"])


def write_initial_values_with_bounds(res_solver, dict_options):
    """ writes out initial point with ID sample_ID and initial box with ID 
    box_ID into a text file    

    Args:
        :res_solver:        dictionary with instance of initial model
        :dict_options:      dictionary with sample_ID, box_ID and fileName

    """
    
    box_ID = dict_options["box_ID"]
    sample_ID = dict_options["sample_ID"]
    fileName = dict_options["fileName"] + "_" + str(box_ID) + "_" + str(sample_ID)
    fileName += "_initial.txt"
    
    res_file = open(fileName, "w")
    writeIterVarValues(res_file,  res_solver["initial_model"])



def write_results_with_bounds(res_solver, dict_options):
    box_ID = dict_options["box_ID"]
    sample_ID = dict_options["sample_ID"]
    fileName = dict_options["fileName"] + "_" + str(box_ID) + "_" + str(sample_ID)
    fileName += "_results.txt"
    
    res_file = open(fileName, "w")
    writeIterVarValues(res_file,  res_solver["Model"])




def writeSampleWIthMinResidual(sample, i, dict_options, sampling_options, solv_options):
    """ writes variable values and results of converged samples into files
    
    Args:
        :sample:            instance of class Model with sample iteration variable 
                            values
        :i:                 number of converged sample as integer
        :dict_options:      dictionary with user specified settings
        :res_solver:        dictionary with solver output
        :sampling_options:  dictionary with sampling options
        :solv_options:      dictionary with user specified solver settings
        
    """
    
    fileName = getFileName(dict_options, solv_options)
    sample_file = open(''.join([fileName, 
                                "_sample_minf_", str(i), "_",
                                sampling_options["sampling method"],
                                "._txt"]), "w")
    sample_file.write(" ****************** Sample ****************** \n\n")
    writeIterVarValues(sample_file,  sample)

    
def writeConvergedSample(sample, i, dict_options, res_solver, sampling_options, solv_options):
    """ writes variable values and results of converged samples into files
    
    Args:
        :sample:            instance of class Model with sample iteration variable 
                            values
        :i:                 number of converged sample as integer
        :dict_options:      dictionary with user specified settings
        :solv_options:      dictionary with user specified solver settings
        :res_solver:        dictionary with solver output
        :sampling_options:  dictionary with sampling options
        
    """
    
    fileName = getFileName(dict_options, solv_options)
    sample_file = open(''.join([fileName, 
                                "_sample_", str(i), "_",
                                sampling_options["sampling method"],
                                "._txt"]), "w")
    sample_file.write(" ****************** Sample ****************** \n\n")
    writeIterVarValues(sample_file,  sample)
    sample_file.write(" ****************** Result ****************** \n\n")
    writeIterVarValues(sample_file,  res_solver["Model"])

def write_analytics(res_solver, dict_options):
    """ writes additional iteration information to file res_file
    
    Args:
        :res_solver:            dictionary with solver output
        :dict_options:          dictionary with user specified settings
    """
    if res_solver != []:
        box_ID = dict_options["box_ID"]
        sample_ID = dict_options["sample_ID"]
        fileName = dict_options["fileName"] + "_" + str(box_ID) + "_" + str(sample_ID)
        fileName += "_analysis.txt"
    
        res_file = open(fileName, "w")
        #writeSolverOutput(res_file, res_solver)
        
        writeFunctionLegend(res_file, res_solver["Model"])
        res_solver["Exitflag"] *= numpy.ones(len(res_solver["Model"].blocks))
        writeFunctionTable(res_file, res_solver)

def writeResultsAnalytics(dict_options, res_solver, solv_options):
    """ writes additional iteration information to file res_file
    
    Args:
        :dict_options:          dictionary with user specified settings
        :solv_options:          dictionary with user specified solver settings
        :res_solver:            dictionary with solver output
        
    """
    if res_solver != []:
        fileName = getFileName(dict_options, solv_options)
        res_file = open(''.join([fileName,"_analysis.txt"]), "w")
        writeSolverOutput(res_file, res_solver)
        
        writeFunctionLegend(res_file, res_solver["Model"])
        writeFunctionTable(res_file, res_solver)
    
    
def writeRestructuringSettings(res_file, dict_options):
    """ writes Restructuring settings to file res_file
    
    Args:
        :res_file:          text document
        :dict_options:      dictionary with user specified settings
    
    Return:         None.        
    
    """
    res_file.write(" ****************** Restructuring Settings ****************** \n\n")
    res_file.write("Decomposition Method: %s\n"%(dict_options["decomp"]))
    res_file.write("Scaling Method: %s\n\n"%(dict_options["scaling"]))


def writeSolverSettings(res_file, solv_options):
    """ writes out solver settings
    
    Args:
        :res_file:          text document
        :solv_options:      dictionary with solver settings
    
    Return:         None.  
    """
    res_file.write(" ****************** Solver Settings ****************** \n\n")
    res_file.write("Solver: %s\n"%(solv_options["solver"]))
    res_file.write("Function Residual: %s\n"%(solv_options["FTOL"]))
    res_file.write("Maximum iteration number per Block: %s\n\n"%(solv_options["iterMax"]))
    
        
def writeIterVarValues(res_file, model):
    """ writes final iteration variable values to file res_file
    
    Args:
        :res_file:          text document
        :model:             object of class Model
    
    Return:         None.  
    """
    
    res_file.write(" ****************** Iteration Variable Values ****************** \n\n")    
    for i in range(0, len(model.xSymbolic)):
        res_file.write("%s    %s    %s    %s\n"%(model.xSymbolic[i], 
                                                 model.stateVarValues[0][i],
                                                 model.xBounds[0][i][0],
                                                 model.xBounds[0][i][1]))
    res_file.write("\n")  

        
def writeFunctionLegend(res_file, model):
    """ writes symbolic functions of model and their global indices to file res_file
    
    Args:
        :res_file:          text document
        :model:             object of class Model
    
    Return:         None.  
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
    
    Return:         None.  
    """
    
    model = res_solver["Model"]
    blockID = model.getBlockID()    
    exitflag = getQuantityForFunction(res_solver["Exitflag"], blockID)
    condNo = getQuantityForFunction(res_solver["CondNo"], blockID)
    iterNo = getQuantityForFunction(res_solver["IterNo"], blockID)
    funVal = model.getFunctionValues()
    
    res_file.write(" ****************** Table with Function Results ****************** \n\n") 
    res_file.write("GLbID  BlockID  Exitflag  CondNo  IterNo  Residual\n") 
    
    for i in range(0, len(model.fSymbolic)):
        res_file.write("%s  %s  %s  %s  %s  %s\n"%(i, 
                                               blockID[i], 
                                               exitflag[i],
                                               condNo[i],
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
    res_file.write("Total Residual: %s\n"%(res_solver["Residual"])) 
    #res_file.write("Condition Number: %s\n\n"%(res_solver["Model"].getPermutedAndScaledConditionNumber())) 


def getFileName(dict_options, solv_options):
    """ returns file name as string due to the chosen restructuring methods
        by the user stored in dict_options.
    
    Args:
        :dict_options:      dictionary with user specified settings
        :solv_options:      dictionary with user specified solver settings
    Return:                 string with text file name
    
    """
    
    name = dict_options["fileName"]
    
    if dict_options["decomp"] =='None': name = ''.join([name, '_org'])
    if dict_options["decomp"] =='DM': name = ''.join([name,'_DM'])
    if dict_options["decomp"] =='BBTF': name = ''.join([name,'_BBTF'])
    if dict_options["scaling"] =='MC29': name = ''.join([name,'_MC29'])
    if dict_options["scaling"] =='MC77': name = ''.join([name,'_MC77'])
    if dict_options["scaling"] =='Inf RowSca and Mean ColSca': name = ''.join([name,'_InfMean'])
    if dict_options["scaling"] =='Inf RowSca and gMean ColSca': name = ''.join([name,'_InfgMean'])
    if dict_options["scaling"] != 'None':
        if dict_options["scaling procedure"] =='tot_init': name = ''.join([name,'_totInit'])  
        if dict_options["scaling procedure"] =='block_init': name = ''.join([name,'_blcInit']) 
        if dict_options["scaling procedure"] =='block_iter': name = ''.join([name,'_blcIter']) 
    if solv_options["solver"]=='newton': name = ''.join([name,'_newton'])
    if solv_options["solver"]=='SLSQP': name = ''.join([name,'_SLSQP_', str(solv_options["mode"])])
    if solv_options["solver"]=='trust-constr': name = ''.join([name,'_trust-constr_', str(solv_options["mode"])])
    if solv_options["solver"]=='TNC': name = ''.join([name,'TNC_', str(solv_options["mode"])])
    if solv_options["solver"]=='ipopt': name = ''.join([name,'ipopt_', str(solv_options["mode"])])
    if solv_options["solver"]=='matlab-fsolve': name = ''.join([name,'matlabFsolve'])
    if solv_options["solver"]=='matlab-fsolve-mscript': name = ''.join([name,'matlabFsolvemscript'])
    if solv_options["solver"]=='fsolve': name = ''.join([name,'fsolve'])
    return name