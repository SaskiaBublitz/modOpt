"""
***************************************************
Import packages
***************************************************
"""
import mpmath
import copy
"""
***************************************************
analysis tools
***************************************************
"""

__all__ = ['analyseResults']

def analyseResults(dict_options, initialModel, modelWithReducedBounds):
    """ volume fractions of resulting soltuion area(s) to initial volume are
    calculated and stored in a textfile <fileName>_analysis.txt
    
    Args:
        :dict_options:              dictionary with user settings
        :initialModel:              instance of type model with initial bounds 
        :modelWithReducedBounds:    instance of type model with reduced bounds 
        
    """
    
    varSymbolic = initialModel.xSymbolic
    initVarBounds = initialModel.xBounds[0]
    if modelWithReducedBounds != []:
        reducedVarBounds = modelWithReducedBounds.xBounds
        boundRatios =getBoundRatios(initVarBounds, reducedVarBounds)
        boundRatioOfVars, solvedVars = getBoundRatioOfVars(boundRatios)
        hypercubicLFractions = getHypercubelengthFractionOfOneVarBoundSet(boundRatios,
                                                                         (len(varSymbolic)-len(solvedVars)))

        hypercubicLFraction = sum(hypercubicLFractions)
        writeAnalysisResults(dict_options["fileName"], varSymbolic, boundRatios, boundRatioOfVars,
                             hypercubicLFractions, hypercubicLFraction, solvedVars)
 
    
def writeAnalysisResults(fileName, varSymbolic, boundRatios, boundRatioOfVars,
                         hypercubicLFractions, hypercubicLFraction, solvedVars):
    """ writes anaylsis results to a textfile  <fileName>_analysis.txt
    
    Args:
        :fileName:                    string with file name
        :varSymbolic:                 list with symbolic variables in sympy logic
        :boundRatios:                 list with reduced variable bound tp initial 
                                      variable bound ratio (has only one entry if 
                                      one set of variable bounds remains)
        :boundRatioOfVars:            sum of the bound ratios of one variable
        :hypercubicLFractions:        list with fractional length of each sub-hypercube
        :hypercubicLFraction:         If the volumes were hypercubic, the hypercubicLFraction
                                      equals their edge length reduction
                                    
    """
    
    res_file = open(''.join([fileName,"_analysis.txt"]), "w") 
    res_file.write("***** Results of Analysis *****\n\n") 
    
    noOfVarSets = len(boundRatios)
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


