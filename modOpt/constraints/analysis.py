"""
***************************************************
Import packages
***************************************************
"""
import mpmath
import copy
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
    if modelWithReducedBounds != []:
        reducedVarBounds = modelWithReducedBounds.xBounds
        boundRatios = getBoundRatios(initVarBounds, reducedVarBounds)
        boundRatioOfVars, solvedVars = getBoundRatioOfVars(boundRatios)
        initVolume = calcInitVolume(initVarBounds)
        hypercubicLFractions = getHypercubelengthFractionOfOneVarBoundSet(boundRatios,
                                                                         (len(varSymbolic)-len(solvedVars)))
        
        hypercubicLFraction = sum(hypercubicLFractions)
        density = getDensityOfJacoboan(modelWithReducedBounds)
        nonLinRatio = getNonLinearityRatio(modelWithReducedBounds)
        writeAnalysisResults(dict_options["fileName"], varSymbolic, boundRatios, 
                             boundRatioOfVars, initVolume, hypercubicLFractions, 
                             hypercubicLFraction, solvedVars, density, nonLinRatio)
 

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
        :boundRatioOfVars:            sum of the bound ratios of one variable
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
    res_file.write("Volume of initial set of varibale bounds: \t%s\n"%(initVolume))
    res_file.write("\n") 
    res_file.write("Variables\t") 
    for j in range(0, noOfVarSets):
          res_file.write("VarBounds_%s\t"%(j)) 
    res_file.write("VarBoundFraction\n")
    
    for i in range(0, len(boundRatios[0])):
        res_file.write("%s\t"%(repr(varSymbolic[i])))        
        
        for j in range(0, noOfVarSets):
            res_file.write("%s \t"%(boundRatios[j][i]))  
        res_file.write("%s\n"%(boundRatioOfVars[i]))

    res_file.write("\nHypercubicLengthFractions\t ")
    for j in range(0, noOfVarSets):
        res_file.write("%s\t"%(hypercubicLFractions[j]))
    res_file.write("%s"%(hypercubicLFraction)) 
    
    if solvedVars != []:
        res_file.write("\n\nFollowing Variables have been solved:\n")
        for solvedVar in solvedVars:
            res_file.write("%s (VarBound_%s)\n" %(varSymbolic[solvedVar[1]], 
                                                               solvedVar[0]))

def getHypercubelengthFractionOfOneVarBoundSet(boundRatios, dim):
    """ calculates edge fraction of each sub-hypercube that result from multiple
    solution interval sets
    
    Args:
        :boundRatios:      list with reduced variable bound tp initial variable 
                           bound ratio (has only one entry if one set of variable 
                           bounds remains) 
        :dim:              number of remaining iteration variable intervals
        
    Return:
        :hypercubeLengthFractions: list with sub-hypercubic length fractions
        
    """
    
    if dim == 0.0:
        return [0.0]
    
    hypercubeLengthFractions = []
    n = 1.0 / dim
    
    for j in range(0, len(boundRatios)):
        
        curFraction = 1
        
        for i in range(0, len(boundRatios[0])):
            
            if boundRatios[j][i] != 'solved':
                curFraction = curFraction * (boundRatios[j][i])**n

        hypercubeLengthFractions.append(curFraction)
    
    return hypercubeLengthFractions
    

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
    

def getBoundRatioOfVars(boundRatios):
    """ sums of variable bounds of different reduced variable bound sets
    
    Args:
        :boundRatios:       list with bound ratios
    
    Return:                 list bound ratios of all iteration variables
    
    """
    
    boundRatioOfVars = []
    solvedVars = []
    
    for i in range(0, len(boundRatios[0])):
        
        sumOfBoundRatios = 0
        
        for j in range(0, len(boundRatios)):
            
            if boundRatios[j][i] != 'solved':
                sumOfBoundRatios = sumOfBoundRatios + boundRatios[j][i]    
            else:
                solvedVars.append([j, i])   
                
        if sumOfBoundRatios != 0:
            boundRatioOfVars.append(sumOfBoundRatios)
        else: boundRatioOfVars.append('solved')
    
    return boundRatioOfVars, solvedVars
            
    
def getBoundRatios(initVarBounds, reducedVarBounds):
    """ calculates ratios of reduced variable bounds to initial variable bounds
    
    Args:
        :initVarBounds:       list with initial variable bounds
        :reducedVarBounds:    list with reduced variable bound sets
    
    Return:                   list with bound ratios
        
    """
    
    boundRatios = copy.deepcopy(reducedVarBounds)
    
    for i in range(0, len(boundRatios)):
        calcBoundRatios(initVarBounds, boundRatios[i])
    return boundRatios

        
def calcBoundRatios(initVarBounds, curBoundRatio):
    """ calculates current variable set bound ratio
  
    Args:
        :initVarBounds:       list with initial variable bounds
        :curBoundRatio:       current variable bound set as a list with mpmath.mpi
                              values
                              
    """  
    
    for j in range(0, len(initVarBounds)):
        curBoundRatio[j] = calcBoundFraction(initVarBounds[j], curBoundRatio[j])


def calcInitVolume(initVarBounds):
    """ calculates volume of initial variable bound set
    
    Args:
        :initVarBounds:     list with initial variable bounds
    
    Return:                 float with intiial volume
    
    """
    
    volume = 1
    for curBound in initVarBounds:
        width = curBound.delta
        volume = volume * width
    return volume
        

def calcBoundFraction(initVarBound, curVarBound):
    """ calculates current variable bound ratio
  
    Args:
        :initVarBounds:       list with initial variable bounds
        :curVarBound:         current variable bound in mpmath.mpi formate
        
    Return:                   current variable bound ratio as float value
    
    """
    
    bratio = float(mpmath.mpf(curVarBound.delta)) / float(mpmath.mpf(initVarBound.delta))
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
    