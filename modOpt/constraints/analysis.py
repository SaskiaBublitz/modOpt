"""
***************************************************
Import packages
***************************************************
"""
import mpmath
import modOpt.decomposition as mod
import sympy

"""
***************************************************
analysis tools
***************************************************
"""

__all__ = ['analyseResults', 'trackErrors']

def analyseResults(dict_options, initialModel, res_solver):
    """ volume fractions of resulting soltuion area(s) to initial volume are
    calculated and stored in a textfile <fileName>_analysis.txt
    
    Args:
        :dict_options:     dictionary with user settings
        :initialModel:     instance of type model with initial bounds 
        :res_solver:       dictionary with resulting model after variable bounds 
                           reduction 
        
    """
    
    modelWithReducedBounds = res_solver["Model"]
    varSymbolic = initialModel.xSymbolic
    initVarBounds = initialModel.xBounds[0]
    initLength = calcVolumeLength(initVarBounds, len(varSymbolic)) # volume calculation failed for large systems with large initial volumes

    if modelWithReducedBounds != []:
        reducedVarBounds = modelWithReducedBounds.xBounds
        solvedVarsID, solvedVarsNo = getSolvedVars(reducedVarBounds)
        
        dim_reduced = getReducedDimensions(solvedVarsNo, len(varSymbolic))
        boundsRatios = getVarBoundsRatios(initVarBounds, reducedVarBounds)

        lengths = calcHypercubicLength(reducedVarBounds, dim_reduced)        
        boundRatiosOfVars = getBoundRatiosOfVars(boundsRatios)

        lengthFractions = getLengthFractions(initLength, lengths)
     
        hypercubicLFraction = getHyperCubicLengthFraction(initLength, lengths, dim_reduced)
        density = getDensityOfJacoboan(modelWithReducedBounds)
        nonLinRatio = getNonLinearityRatio(modelWithReducedBounds)
        writeAnalysisResults(dict_options["fileName"], varSymbolic, boundsRatios, 
                             boundRatiosOfVars, initLength, lengthFractions, 
                             hypercubicLFraction, solvedVarsID, density, nonLinRatio)


def calcVolumeLength(box, dim):
    """ calculates box edge length assuming it as a hypercube
    
    Args:
        :box:      list with variable bounds in mpmath.mpi logic
        :dim:      box dimension as integer
    
    Returns:
        :length:   box edge length as float
    
    """
    
    length = 1.0
    
    solvedID, dim = getSolvedVars([box])
    dim =dim[0]
    
    for interval in box:
        width = float(mpmath.mpf(interval.delta)) 
        if width != 0.0:
            length*=(width)**(1.0/dim)
    return length


def getSolvedVars(boxes):
    """ filters out variable intervals with zero width (solved)
    
    Args:
        :boxes:     list with reduced boxes (numpy.array)
    
    Returns:
    :solvedVarsID:      Nested list [[i,j],...] with i box-ID and j variable-ID 
                        of solved interval as integer
    solvedVarsNo :      List with numbers of solved intervals in the boxes (int)

    """
    
    solvedVarsID =[]
    solvedVarsNo =[]
    
    for i in range(0, len(boxes)):
        curBox = boxes[i]
        soledVarsNoBox = 0
        for j in range(0, len(curBox)):
            width = float(mpmath.mpf(curBox[j].delta))
            if width == 0.0:
                solvedVarsID.append([j,i])
                soledVarsNoBox += 1
                
        solvedVarsNo.append(soledVarsNoBox) 
    return solvedVarsID, solvedVarsNo


def getReducedDimensions(solvedVarsNo, dim):
    """ determines the dimension of the reduced boxes where solvedVarsNo variables
    have been solved.
    
    Args:
        :solvedVarsNo:      list with number of solved variables per box 
        :dim:               integer with dimension of initial box
    
    Returns:
        :dim_reduced:       list with dimension of reduced boxes as integer

    """
    
    if solvedVarsNo == []: return [dim]
    
    dim_reduced = []
    for soledVarsNoBox in solvedVarsNo:
        dim_reduced.append(dim - soledVarsNoBox)
    return dim_reduced
            
    
def getVarBoundsRatios(initBox, reducedVarBounds):
    """ calculates ratios of reduced variable bounds to initial variable bounds
    
    Args:
        :initBox:             list with initial variable bounds
        :reducedVarBounds:    list with reduced variable bound sets
    
    Returns:                   list with variable bounds ratios
        
    """
    
    varBoundsRatios = []
    
    for curBox in reducedVarBounds:
        varBoundsRatios.append(calcBoxRatios(initBox, curBox))
    return varBoundsRatios


def calcHypercubicLength(boxes, dim):
    """ calculates hypercubic lengths of boxes
    
    Args:
        :boxes:     list with boxes and that contain intervals in mpmath.mpi formate
        :dim:       list with dimensions of boxes as integer
        
    Returns:
        :lengths:   list with hypercubic lengths as floats

    """
    
    lengths = []
    for i in range(0, len(boxes)):
        lengths.append(calcVolumeLength(boxes[i], dim[i]))
    return lengths


def getBoundRatiosOfVars(boundsRatios):
    """ calculates total variable bound ratios through summing up all different
    variable intervals of the boxes.
    
    Args:
        :boundsRatios:         nested list with bound ratios as float values in 
                               reduced boxes

    Returns:
        :boundRatiosOfVars:   list with sums of unique variable bound ratios

    """
    boundRatiosOfVars = []
    
    boxNo = len(boundsRatios)
    dim = len(boundsRatios[0])
    
    for i in range(0, dim):
        varBoundRatios = []
        ratioOfVar = 0.0
        for j in range(0, boxNo):
            if not boundsRatios[j][i] in varBoundRatios:
                varBoundRatios.append(boundsRatios[j][i])
             
        for ratio in varBoundRatios:
            if type(ratio) is float: ratioOfVar += ratio
            
        if ratioOfVar == 0: ratioOfVar='solved'
        boundRatiosOfVars.append(ratioOfVar)
        
    return boundRatiosOfVars
    

def getLengthFractions(initLength, lengths):
    """ calculates hypercubic length fractions of box edge lengths referring 
    to the lngth of th initial box
   
    Args:
    :initLength:    float with edgie length of initial volume (as hypercube)
    :lengths:       list with edgie lengths of redced box volumes 
                    (as hypercubes) as floats

    Returns:
    :lengthFractions: list with hypercubic length fractions as float

    """
    
    lengthFractions = []
    for length in lengths:
        lengthFractions.append(length / initLength)
        
    return lengthFractions


def getHyperCubicLengthFraction(initLength, lengths, dim_reduced):
    """ calculates length fraction of all reduced boxes / intial box
    
    Args:
        :initLength:    float with hypercubic length of initial box
        :lengths:       list with hypercubic lengths of reduced boxes as floats
        :dim_reduced:   list with dimensions of reduced boxes as integer
        
    Returns:     hypercubic length fraction as mpmath.mpf (gets to higher nubmers than float)
    
    """
    
    reduced_volume =  0.0

    for i in range(0, len(lengths)):
        reduced_volume+=(mpmath.mpf(lengths[i]))**dim_reduced[i]
        
    return reduced_volume**(1.0/max(dim_reduced))/initLength
 

def getDensityOfJacoboan(model):
    """ returns nonzero density of jacobian matrix from equation system
    
    Args:
        :model:         instance of class Model
        
    Return:     ratio of nonzero entries to total number of entries in jacobian (mxm)
    
    """
    
    model.jacobian, f = mod.getCasadiJandF(model.xSymbolic, model.fSymbolic)
    return float(model.getJacobian().nnz()) / model.getModelDimension()**2
    

def getNonLinearityRatio(model):
    """ identifies nonlinear dependencies of variables in jacobian matrix by 
    second derrivate and counts all nonlinear entries to calculate the ratio
    of nonlinear entries to total number of entries in jacobian.
    
    Args:
        :model:     istance of class Model
    
    Return:         float of ratio: nonlinear entries / total entries
        
    """

    nonLin = 0

    for curX in model.xSymbolic:
        for curF in model.fSymbolic:
            if curX in curF.free_symbols:
                d2fdx = sympy.diff(sympy.diff(curF, curX), curX)
                if d2fdx != 0: nonLin = nonLin + 1
                
    return nonLin / float(model.getJacobian().nnz())            
    
    
def writeAnalysisResults(fileName, varSymbolic, boundRatios, boundRatioOfVars, initVolume,
                         hypercubicLFractions, hypercubicLFraction, solvedVars, density,
                         nonLinRatio):
    """ writes anaylsis results to a textfile  <fileName>_analysis.txt
    
    Args:
        :fileName:                    string with file name
        :varSymbolic:                 list with symbolic variables in sympy logic
        :boundRatios:                 list with reduced variable bound tp initial 
                                      variable bound ratio (has only one entry if 
                                      one set of variable bounds remains)
        :boundRatioOfVars:            sum of the unique bound ratios of one variable
        :initVolume:                  volume of initial variable bound set
        :hypercubicLFractions:        list with fractional length of each sub-hypercube
        :hypercubicLFraction:         If the volumes were hypercubic, the hypercubicLFraction
                                      equals their edge length reduction
        :solvedVars:                  list with indices of solved variables
        :density:                     float with nonzero density of jacobian
        :nonLinRatio:                 float with nonlinear entries / total entries of jacobian
        
    """
    
    res_file = open(''.join([fileName,"_analysis.txt"]), "w") 
    
    res_file.write("***** Results of Analysis *****\n\n") 
    
    noOfVarSets = len(boundRatios)
    res_file.write("System Dimension: \t%s\n"%(len(boundRatios[0]))) 
    res_file.write("Jacobian Nonzero Density: \t%s\n"%(density))
    res_file.write("Jacobian Nonlinearity Ratio: \t%s\n"%(nonLinRatio))
    res_file.write("Length of initial box: \t%s\n"%(initVolume))
    res_file.write("\nHypercubicLengthFractions\t ")
    for j in range(0, noOfVarSets):
        res_file.write("%s\t"%(hypercubicLFractions[j]))
    res_file.write("%s"%(hypercubicLFraction)) 
    
    if solvedVars != []:
        res_file.write("\n\nFollowing Variables have been solved:\n")
        for solvedVar in solvedVars:
            res_file.write("%s (VarBound_%s)\n" %(varSymbolic[solvedVar[0]], 
                                                               solvedVar[1]))
    res_file.write("\nVariables\t") 
    for j in range(0, noOfVarSets):
          res_file.write("VarBounds_%s\t"%(j)) 
    res_file.write("VarBoundFraction\n")
    
    for i in range(0, len(boundRatios[0])):
        res_file.write("%s\t"%(repr(varSymbolic[i])))        
        
        for j in range(0, noOfVarSets):
            res_file.write("%s \t"%(boundRatios[j][i]))  
        res_file.write("%s\n"%(boundRatioOfVars[i]))


def getBoundRatioOfVarBoundSet(boundRatios):
    """ multiplies all variable bounds of one variable bound set for all variable
    bound sets in order to calculate their volume specific volume frations.
    
    Args:
        :boundRatios:       list with bound ratios
    
    Return:                 list with volume fractions all variable bound sets
    
    """
    
    boundRatioOfVarBoundSet = []
    
    for j in range(0, len(boundRatios)):
        
        productOfBoundRatios = 1
        
        for i in range(0, len(boundRatios[0])):
            
            if boundRatios[j][i] != 'solved':
                productOfBoundRatios = productOfBoundRatios * boundRatios[j][i]

        boundRatioOfVarBoundSet.append(productOfBoundRatios)
    
    return boundRatioOfVarBoundSet
        
       
def calcBoxRatios(initVarBounds, box):
    """ calculates interval set bound ratio to initial box
  
    Args:
        :initVarBounds:       list with initial variable bounds
        :box:           current box as a list with mpmath.mpi values
    Return:

        :boxRatios:         list with reduced variable bounds ratios                          
    """  
    boxRatios = []
    for j in range(0, len(initVarBounds)):
        boxRatios.append(calcBoundFraction(initVarBounds[j], box[j]))
    return boxRatios


def calcBoundFraction(initVarBound, curVarBound):
    """ calculates current variable bound ratio
  
    Args:
        :initVarBounds:       list with initial variable bounds
        :curVarBound:         current variable bound in mpmath.mpi formate
        
    Return:                   current variable bound ratio as float value
    
    """
    if float(mpmath.mpf(initVarBound.delta)) != 0.0:
        bratio = float(mpmath.mpf(curVarBound.delta)) / float(mpmath.mpf(initVarBound.delta))
    else: 
        return "Warning: Initial interval with zero width"
    if bratio == 0: return 'solved'
    return bratio


def trackErrors(initialModel, res_solver, dict_options):
    """ proofs if current state of model is correctly and write the error that
    occured in case the model failed to an error text file.
    
    Args:
        :initialModel:      instance of type Model at initial point
        :res_solver:        dictionary with solver output
        :dict_options:      dictionary with user specified settings
        
    """
    if res_solver["Model"].failed:
        fileName = dict_options["fileName"] + "_errorAnalysis.txt"
        failedModel = res_solver["Model"]
        failedSystem = res_solver["noSolution"]
        fCrit = failedSystem.critF
        varCrit = failedSystem.critVar
        varsInF = failedSystem.varsInF
        
        xBoundsInitial = initialModel.getXBoundsOfCertainVariablesFromIntervalSet(varsInF, 0)
        xBoundsFailed = failedModel.getXBoundsOfCertainVariablesFromIntervalSet(varsInF, 0)
        
        writeErrorAnalysis(fileName, fCrit, varCrit, varsInF, xBoundsInitial, xBoundsFailed)


def writeErrorAnalysis(fileName, fCrit, varCrit, varsInF, xBoundsInitial, xBoundsFailed) :
    res_file = open(fileName, "w") 
    res_file.write("***** Error Analysis *****\n\n")    

    res_file.write("Algorithm failed because it could not find any solution for %s in equation:\n\n %s\n\n"%(varCrit, fCrit)) 
     
    res_file.write("The following table shows the initial and final bounds of all variables in this equation before termination:\n\n")
                    
    res_file.write("VARNAME \t INITBOUNDS \t FINALBOUNDS \n" )
     
    for i in range(0, len(varsInF)):
        res_file.write("%s \t %s \t %s \n"%(varsInF[i],  str(xBoundsInitial[i]), str(xBoundsFailed[i])))
     
    res_file.close()
    