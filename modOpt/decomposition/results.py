"""
***************************************************
Import packages
***************************************************
"""
#import platform
#if platform.system() != 'Windows': 
#    import matplotlib
#    matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import numpy

    
"""
***************************************************
Output
***************************************************
"""

__all__ = ['writeResults', 'plotIncidence', 'plotColoredIncidence']


def writeResults(dict_equations, dict_variables, dict_options, solv_options=None, res_conditionNumber = None):
    """  write results from dictionaries into text files
    
    Args:
        :dict_equations:       dictionary with information about all equations
        :dict_variables:       dictionary with information about all variables
        :dict_options:         dictionary with user specified settings
        :res_conditionNumber:  optional dictionary with calculated condtion 
                               numbers, default = None 
              
    """

    fileName = getFileName(dict_options, solv_options)
    res_file = open(''.join([fileName,".txt"]), "w") 
    
    writeUserSepcResults(res_file, dict_options, dict_equations, dict_variables, res_conditionNumber)
    
    res_file.close()
    

def writeUserSepcResults(res_file, dict_options, dict_eq, dict_var, res_condNo=None):
    """  write results from dictionaries into text files
    
    Args:
        :res_file:              opened text file
        :dict_options:          dictionary with user specified settings        
        :dict_eq:               dictionary with information about all equations        
        :dict_var:              dictionary with information about all variables
        :res_condNo:            optional dictionary with calculated condtion 
                                numbers, default = None 
              
    """
    
    # Decomposition
    if dict_options["decomp"]=='None':
        res_file.write("*****Original system*****\n\n")
        writePermutation(res_file, dict_eq, dict_var)
    if dict_options["decomp"]=='DM':
        res_file.write("*****Dulmage-Mendelsohn decomposed System*****\n\n")
        writePermutation(res_file, dict_eq, dict_var)
        
    if dict_options["decomp"]=='BBTF':
        res_file.write("*****Bordered Block Decomposed System*****\n\n")
        writePermutation(res_file, dict_eq, dict_var)
    # Scaling       
    if dict_options["scaling"]=='Inf RowSca and Mean ColSca':
        res_file.write("\n*****Variable Scaling by Mean Value*****\n\n")
        writeScaledVarAndVarScaling(res_file, dict_var)
        res_file.write("\n*****Equation Scaling by Infinite Norm*****\n\n")
        writeEqScaling(res_file, dict_eq)
                
    if dict_options["scaling"]=='Inf RowSca and gMean ColSca':
        res_file.write("\n*****Variable Scaling by GMean Value*****\n\n")
        writeScaledVarAndVarScaling(res_file, dict_var)
        
        res_file.write("\n*****Equation Scaling by Infinite Norm*****\n\n")
        writeEqScaling(res_file, dict_eq)
        
    if dict_options["scaling"]=='MC29':
        res_file.write("\n*****Variable Scaling by Curtis' and Reid's Method*****\n\n")
        writeScaledVarAndVarScaling(res_file, dict_var)
        
        res_file.write("\n*****Equation Scaling by Curtis' and Reid's Method*****\n\n")
        writeEqScaling(res_file, dict_eq)

    if dict_options["scaling"]=='MC77':
        res_file.write("\n*****Variable Scaling by Ruiz' Method*****\n\n")
        writeScaledVarAndVarScaling(res_file, dict_var)
        
        res_file.write("\n*****Equation Scaling by Ruiz' Method*****\n\n")
        writeEqScaling(res_file, dict_eq)
        
    # Condition Number
    if dict_options["condNo"]:
        res_file.write("\n*****Condition Number*****\n\n")
        writeCondNo(res_file, res_condNo, dict_options)


def getFileName(dict_options, solv_options=None):
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
        if dict_options["scalingProcedure"] =='tot_init': name = ''.join([name,'_totInit'])  
        if dict_options["scalingProcedure"] =='block_init': name = ''.join([name,'_blcInit']) 
        if dict_options["scalingProcedure"] =='block_iter': name = ''.join([name,'_blcIter']) 
    if solv_options:
        if solv_options["solver"]=='newton': name = ''.join([name,'_newton'])
        if solv_options["solver"]=='SLSQP': name = ''.join([name,'_SLSQP_', str(solv_options["mode"])])
        if solv_options["solver"]=='trust-constr': name = ''.join([name,'_trust-constr_', str(solv_options["mode"])])
        if solv_options["solver"]=='TNC': name = ''.join([name,'TNC_', str(solv_options["mode"])])
        if solv_options["solver"]=='ipopt': name = ''.join([name,'ipopt_', str(solv_options["mode"])])
    
    return name


def writeDict(res_file, DICT):
    """ write dictionary information to text file
    
    Args:
        :res_file:              opened text file
        :dict:                  dictionary
        
    """
    
    for i in list(DICT.keys()):
        res_file.write("%s = %s\n"%(i, DICT[i]))


def writeBBTFResults(res_file, rowperm, colperm, borderWidth, info):
    """ write results from BBTF decomposition into text file
    
    Args:
        :res_file:       opened text file
        :rowperm:        list with permuted global row indices 
        :colperm:        list with permuted global column indices
        :borderWidth:    size of border width as integer
        :info:           error flag
        
    """    
    
    res_file.write("\nResults from Decomposition:\n")
    res_file.write("Row Permutation: %s\n"%(rowperm))
    res_file.write("Column Perumtation: %s\n"%(colperm))
    res_file.write("Border Width: %s\n"%(borderWidth))
    res_file.write("Number of Nonzero Rows: %s\n"%(list(info.values())[0]))
    res_file.write("Number of Nonzero Columns: %s\n"%(list(info.values())[1]))
    res_file.write("Error Message: %s\n"%(list(info.values()[2])))
    res_file.write("Number of Duplicates: %s\n"%(list(info.values()[3])))


def writePermutation(res_file, dict_eq, dict_var):
    """  write results from permutatuion into text files
    
    Args:
        :res_file:      opened text file        
        :dict_eq:       dictionary with information about all equations
        :dict_var:      dictionary with information about all variables
              
    """
    
    res_file.write("*****Equation Index after Permutation*****\n\n")
    for eq in list(dict_eq.keys()):
          res_file.write("EQ_PERM_%s %s\n"%(dict_eq[eq][1], dict_eq[eq][2]))
    res_file.write("\n")


    res_file.write("\n*****Variable Index after Permutation*****\n\n")
    for var in list(dict_var.keys()):
        res_file.write("%s_PERM %s\n"%(var,dict_var[var][2]))
    res_file.write("\n")


def writeEqScaling(res_file, dict_eq):
    """  write results from equation scaling into text files
    
    Args:
        :res_file:      opened text file        
        :dict_eq:       dictionary with information about all equations
    
    """
         
    for eq in list(dict_eq.keys()):
        res_file.write("EQ_SCA_%s %s\n"%(dict_eq[eq][1], dict_eq[eq][3]))
    res_file.write("\n")


def writeScaledVarAndVarScaling(res_file, dict_var):
    """  write results from variable scaling into text files
    
    Args:
        :res_file:      opened text file        
        :dict_var:      dictionary with information about all state variables
    
    """
    
    for var in list(dict_var.keys()):
        res_file.write("%s_SCA %s %s\n"%(var,dict_var[var][0], dict_var[var][3]))
    res_file.write("\n")


def writeCondNo(res_file, res_condNo, dict_options):
    """  write condition number(s) into text file
    
    Args:
        :res_file:        opened text file        
        :res_condNo:      optional dictionary with calculated condtion numbers, 
                          default = None 
        :dict_options:    dictionary with user specified settings    
        
    """
    
    res_file.write("Original System: %s\n"%(res_condNo["Original"]))

    if dict_options["scaling"]!='None' or dict_options["decomp"]!='None':
        res_file.write("\nRestructured System: %s"%(res_condNo["Restructured"]))



def plotIncidence(model, dict_options, plot_options):
    """  plot incidence matrix of equation system and store it in PDF file
    
    Args:
        :model:           object of type model containig all equation system 
                          evaluation information        
        :dict_options:    dictionary with user specified settings    
        :plot_options:    dictionary with user specified plotting options    
                
    """
    
    plotLabelsAndGrid = plot_options["labelsAndGrid"]
    fileName = getFileName(dict_options)
    rowPerm = model.rowPerm
    colPerm = model.colPerm
    Jperm = model.getPermutedJacobian()
    
    varNames = numpy.array(model.xSymbolic)[colPerm]
    
    incidence = numpy.zeros(Jperm.shape)
    
    #cm = cl.ListedColormap(["white","red","black", "blue"], name='from_list', N=None)
    cm = cl.ListedColormap(["white","red","black", "black"], name='from_list', N=None)
    nz_row = numpy.array(Jperm.row())
    nz_col = numpy.array(Jperm.sparsity().get_col())
    nz_count = len(nz_row)
    nz = numpy.repeat(1.5, nz_count)
    
    incidence[nz_row, nz_col] = nz
    
    if plotLabelsAndGrid == True: 
        plt.grid(linestyle='-', linewidth=0.1)
        plt.xticks(range(0,Jperm.shape[0]), varNames, fontsize=int(210/Jperm.shape[0]))
        plt.yticks(range(0,Jperm.shape[0]), rowPerm, fontsize=int(210/Jperm.shape[0]))
        locs, labels = plt.xticks()  
        plt.setp(labels, rotation=-90) 
        plt.ylabel("Equation No.")
    else:
        plt.ylabel("Equations")
        plt.xlabel("Variables")
    plt.imshow(incidence, cmap=cm, vmin=0, vmax=1.5)

    plt.savefig(''.join([fileName, "_incidence.pdf"]), bbox_inches='tight',dpi=1000)


def plotColoredIncidence(res_solver, dict_options, plot_options, solv_options):
    """ plot incidence matrix of non linear algebraic system with solver results.
    
    blue  = (sub) system solved
    red   = iteration of (sub) system failed
    black = (sub) system solved but for wrong input data i.e. failed results from 
            sub system before
            
    Args:
        :res_solver:        dictionary with resutls from solver
        :dict_options:      dictionary with user specified settings
        :solv_options:      dictionary with user specified solver settings
        :plot_options:      dictionary with user specified plot settings
    
    """
    if res_solver == []: return False
    J, ex, model = getResults(res_solver) 
    
    incidence = prepareIncidence(J, ex, dict_options)
    
    plot_options["color"] = cl.ListedColormap(["white","red","black", (23/255.0, 113.0/255.0, 145.0/255.0)], name = 'from_list', N = None)
    plot_options["lineStyle"] = '-'
    plot_options["lineWidth"] = 0.1
    plot_options["fontSize"] = int(210/J.shape[0])
    
    plotPDF(incidence, model, dict_options, plot_options, solv_options)
    

def getResults(res_solver):
    """ get results from solver needed for incidence plot
    
    Args:
        :res_solver:        dictionary with solver output
    
    """
    
    model = res_solver["Model"]
    #exblock = res_solver["Exitflag"]
    blockID = model.getBlockID()
    ex = getQuantityForFunction(res_solver, "Exitflag", blockID) # in global order
    ex = getPermID(ex, model.rowPerm)
    
    
    J = model.getCasadiJacobian()[model.rowPerm, model.colPerm]
    
    return J, ex, model
    
    
def getQuantityForFunction(res_solver, key, blockID):
    """ get a quantities of the functions in global order referring to their block ID
    
    Args:
        :blocklist:         list with block quanities
        :blockID:           list with blockID of functions in global order
    
    Return:
        :functionList:      list with function quantities in global order
    
    """ 
    return [res_solver[b][key] for b in blockID]


def getPermID(quantityInGlbOrder, permOrder):
    """ get quantities in permuted order
 
    Args:
        :QuantityInGlbOrder:       list with quantities in global order
        :permOrder:                list with permuted order

    Return:
        :QuantityInPermOrder:       list with quantites in permuted order
    
    """
   
    quantityInPermOrder = numpy.zeros(len(quantityInGlbOrder))
   
    for i in range(0, len(quantityInGlbOrder)):
        j = permOrder.index(i)
        quantityInPermOrder[j] =quantityInGlbOrder[i]
        
    return quantityInPermOrder


def prepareIncidence(J, ex, dict_options):
    """ returns incidence matrix with different nonzero values, i.e.: 0.5 for 
    failed blocks, 1.5 for solved blocks and 1 for blocks that were solved but
    based on wrong input variables from blocks before.
    
    Args:
        :J:                     matrix in casadi formate
        :ex:                    list with exit flags of all functions
        :deict_options:         dictionary with user specified settings
        
    """
    incidence = numpy.zeros(J.shape)
    
    if dict_options["decomp"] == 'None':
        nz_row = numpy.array(J.row())
        nz_col = numpy.array(J.sparsity().get_col())
        nz_count = len(nz_row)
        
        if ex[0] == 1: nz = numpy.repeat(1.5, nz_count)
        if ex[0] <= 0: nz = numpy.repeat(0.5, nz_count)
        incidence[nz_row, nz_col] = nz
    
    if dict_options["decomp"] != 'None':
        dim = J.shape[0]

        
        for k in range(0, dim):
            col_nz_index = J[:,k].row()
            row_nz_index = J[k,:].sparsity().get_col()

            if ex[k]==1:
                if all(x!=0.5 for x in incidence[k,row_nz_index]): 
                    incidence[col_nz_index,k]=1.5
                else: 
                    incidence[col_nz_index,k]=0.5
                    if not incidence[k,k] == 0: incidence[k,k] = 1
            if ex[k]<=0: incidence[col_nz_index,k]=0.5          
    return incidence


def plotPDF(incidence, model, dict_options, plot_options, solv_options):
    """ plots colored incidence matrix and stores it in a PDF
    
    Args:
        :incidence:         matrix with different nonzero values due to the solver
                            output
        :model:             object of type model
        :dict_options:      dictionary with user specificed settings
        :plot_options:      dictionary with plot style settings
        :solv_options:      dictionary with user specified solver settings
    
    """
    fileName = getFileName(dict_options, solv_options)
    xSymbolicPerm = [model.xSymbolic[i] for i in model.colPerm]
    
    
    
    if plot_options["labelsAndGrid"]: 
        plt.grid(linestyle='-', linewidth=0.1)    
        plt.grid(linestyle = plot_options["lineStyle"], linewidth= plot_options["lineWidth"] )
        plt.imshow(incidence, cmap=plot_options["color"], vmin=0, vmax=1.5)
        plt.xticks(range(0,len(model.colPerm)), xSymbolicPerm, fontsize = plot_options["fontSize"])
        plt.yticks(range(0,len(model.rowPerm)), model.rowPerm, fontsize = plot_options["fontSize"])
        locs, labels = plt.xticks()  
        plt.setp(labels, rotation=-90) 
        plt.ylabel("Equation No.")
        
    else:
        plt.ylabel("Equations")
        plt.xlabel("Variables")
        plt.imshow(numpy.matrix(incidence), cmap=plot_options["color"], vmin=0, vmax=1.5)
        
    plt.savefig(''.join([fileName, "_incidence.pdf"]),bbox_inches='tight')       
