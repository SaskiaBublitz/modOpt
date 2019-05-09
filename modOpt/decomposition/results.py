"""
***************************************************
Import packages
***************************************************
"""
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import numpy

"""
***************************************************
Output
***************************************************
"""

__all__ = ['writeResults', 'plotIncidence']


def writeResults(dict_equations, dict_variables, dict_options, res_conditionNumber = None):
    """  write results from dictionaries into text files
    
    Args:
        :dict_equations:       dictionary with information about all equations
        :dict_variables:       dictionary with information about all variables
        :dict_options:         dictionary with user specified settings
        :res_conditionNumber:  optional dictionary with calculated condtion 
                               numbers, default = None 
              
    """

    fileName = getFileName(dict_options)
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


def getFileName(dict_options):
    """ creates name for text file
    
    Args:
        :dict_options:    dictionary with user specified settings  
    
    Return:               file name as string
        
    """
    
    name = dict_options["fileName"]
    if dict_options["decomp"] =='None': name = ''.join([name, '_org'])
    if dict_options["decomp"] =='DM': name = ''.join([name,'_DM'])
    if dict_options["decomp"] =='BBTF': name = ''.join([name,'_BBTF'])
    if dict_options["scaling"] == 'MC29': name = ''.join([name,'_sca29'])
    if dict_options["scaling"] == 'Inf RowSca and Mean ColSca': name = ''.join([name,'_scaRInfCMean'])
    if dict_options["scaling"] == 'Inf RowSca and gMean ColSca': name = ''.join([name,'_scaRInfCgMean'])
    if dict_options["condNo"]: name = ''.join([name, '_cond'])
    return name


def writeDict(res_file, DICT):
    """ write dictionary information to text file
    
    Args:
        :res_file:              opened text file
        :dict:                  dictionary
        
    """
    
    for i in DICT.keys():
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
    res_file.write("Number of Nonzero Rows: %s\n"%(info.values()[0]))
    res_file.write("Number of Nonzero Columns: %s\n"%(info.values()[1]))
    res_file.write("Error Message: %s\n"%(info.values()[2]))
    res_file.write("Number of Duplicates: %s\n"%(info.values()[3]))


def writePermutation(res_file, dict_eq, dict_var):
    """  write results from permutatuion into text files
    
    Args:
        :res_file:      opened text file        
        :dict_eq:       dictionary with information about all equations
        :dict_var:      dictionary with information about all variables
              
    """
    
    res_file.write("*****Equation Index after Permutation*****\n\n")
    for eq in dict_eq.keys():
          res_file.write("EQ_PERM_%s %s\n"%(dict_eq[eq][1], dict_eq[eq][2]))
    res_file.write("\n")


    res_file.write("\n*****Variable Index after Permutation*****\n\n")
    for var in dict_var.keys():
        res_file.write("%s_PERM %s\n"%(var,dict_var[var][2]))
    res_file.write("\n")


def writeEqScaling(res_file, dict_eq):
    """  write results from equation scaling into text files
    
    Args:
        :res_file:      opened text file        
        :dict_eq:       dictionary with information about all equations
    
    """
         
    for eq in dict_eq.keys():
        res_file.write("EQ_SCA_%s %s\n"%(dict_eq[eq][1], dict_eq[eq][3]))
    res_file.write("\n")


def writeScaledVarAndVarScaling(res_file, dict_var):
    """  write results from variable scaling into text files
    
    Args:
        :res_file:      opened text file        
        :dict_var:      dictionary with information about all state variables
    
    """
    
    for var in dict_var.keys():
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


def plotIncidence(model, dict_var, dict_options, plot_options):
    """  plot incidence matrix of equation system and store it in PDF file
    
    Args:
        :model:           object of type model containig all equation system 
                          evaluation information        
        :dict_var:        dictionary with information about all state variables
        :dict_options:    dictionary with user specified settings    
        :plot_options:    dictionary with user specified plotting options    
                
    """
    
    plotLabelsAndGrid = plot_options["labelsAndGrid"]
    fileName = getFileName(dict_options)
    rowPerm = model.rowPerm
    colPerm = model.colPerm
    Jperm = model.getPermutedJacobian()
    
    #varNames = getKeyByGlobalId(dict_var)
    varNames = numpy.array(model.xSymbolic)[colPerm]
    
    incidence = numpy.zeros(Jperm.shape)
    
    cm = cl.ListedColormap(["white","red","black", "blue"], name='from_list', N=None)
    
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
    