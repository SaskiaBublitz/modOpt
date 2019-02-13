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

def analyseResults(fileName, varSymbolic, initVarBounds, reducedVarBounds):
    boundRatios =getBoundRatios(initVarBounds, reducedVarBounds)
    boundRatioOfVars = getBoundRatioOfVars(boundRatios)
    boundRatioOfVarBoundSet = getBoundRatioOfVarBoundSet(boundRatios)
    volumeFraction = sum(boundRatioOfVarBoundSet)
    writeAnalysisResults(fileName, varSymbolic, boundRatios, boundRatioOfVars,
                         boundRatioOfVarBoundSet,volumeFraction)
    
def writeAnalysisResults(fileName, varSymbolic, boundRatios, boundRatioOfVars,
                         boundRatioOfVarBoundSet, volumeFraction):

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

    res_file.write("\nVarboundSetFraction\t ")
    for j in range(0, noOfVarSets):
        res_file.write("%s\t"%(boundRatioOfVarBoundSet[j]))
    res_file.write("%s"%(volumeFraction)) 


    
            
            

    

def getBoundRatioOfVarBoundSet(boundRatios):
    boundRatioOfVarBoundSet = []

    for j in range(0, len(boundRatios)):
        
        productOfBoundRatios = 1
        
        for i in range(0, len(boundRatios[0])):
            productOfBoundRatios = productOfBoundRatios * boundRatios[j][i]
            
        boundRatioOfVarBoundSet.append(productOfBoundRatios)
    
    return boundRatioOfVarBoundSet
    
def getBoundRatioOfVars(boundRatios):
    boundRatioOfVars = []

    for i in range(0, len(boundRatios[0])):
        
        sumOfBoundRatios = 0
        
        for j in range(0, len(boundRatios)):
            sumOfBoundRatios = sumOfBoundRatios + boundRatios[j][i]
            
        boundRatioOfVars.append(sumOfBoundRatios)
    
    return boundRatioOfVars
            
    


def getBoundRatios(initVarBounds, reducedVarBounds):
    
    boundRatios = copy.deepcopy(reducedVarBounds)
    
    for i in range(0, len(boundRatios)):
        calcBoundRatios(initVarBounds, boundRatios[i])
    return boundRatios

        


def calcBoundRatios(initVarBounds, curBoundRatio):
    
    for j in range(0, len(initVarBounds)):
        curBoundRatio[j] = calcBoundFraction(initVarBounds[j], curBoundRatio[j])


def calcBoundFraction(initVarBound, curVarBound):
    bratio = float(mpmath.mpf(curVarBound.delta)) / float(mpmath.mpf(initVarBound.delta))
    return bratio


