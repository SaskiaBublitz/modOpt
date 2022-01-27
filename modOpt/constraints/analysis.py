"""
***************************************************
Import packages
***************************************************
"""

import mpmath
import modOpt.decomposition as mod
import sympy
import numpy
import modOpt.scaling as mosca

"""
***************************************************
analysis tools
***************************************************
"""

__all__ = ['analyseResults', 'trackErrors', 'get_hypercubic_length',
           'calc_hypercubic_length', 'calc_average_length', 'calc_residual']


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
    tol = dict_options["absTol"]
    initVarBounds = initialModel.xBounds
    initLength = calc_length(initVarBounds, tol)
    #initLength = calcVolumeLength(initVarBounds, len(varSymbolic),tol) # volume calculation failed for large systems with large initial volumes

    if modelWithReducedBounds != []:
        reducedVarBounds = modelWithReducedBounds.xBounds
        solvedVarsID, solvedVarsNo = getSolvedVars(reducedVarBounds)
        dim_reduced = getReducedDimensions(solvedVarsNo, len(varSymbolic))
        initLengths = calcInitLengths(initVarBounds, tol)
        
        boundsRatios = getVarBoundsRatios(initVarBounds[0], reducedVarBounds)

        lengths = calcHypercubicLength(reducedVarBounds, tol)        
        boundRatiosOfVars = getBoundRatiosOfVars(boundsRatios)

        lengthFractions = 0#getLengthFractions(initLengths, lengths)
        hypercubicLFraction = get_hypercubic_length(dict_options, 
                                                    initVarBounds, 
                                                    reducedVarBounds)
        #hypercubicLFraction = getHyperCubicLengthFraction(initLengths, lengths, dim_reduced)
        density = getDensityOfJacoboan(modelWithReducedBounds)
        nonLinRatio = getNonLinearityRatio(modelWithReducedBounds)
        writeAnalysisResults(dict_options["fileName"], varSymbolic, boundsRatios, 
                             boundRatiosOfVars, initLength, lengthFractions, 
                             hypercubicLFraction, solvedVarsID, density, nonLinRatio)


def get_hypercubic_length(dict_options, init_box, reduced_boxes):
    tol = dict_options["absTol"]
    initLength = calc_length(init_box, tol)
    length = calc_length(reduced_boxes, tol)
    return length / initLength


def calcInitLengths(initVarBounds, tol):#dim_reduced):
    """calculates initial edge lengths for each box (neglecting solved variables)
    
    Args:
        :initVarBounds:     numpy array with initial bounds
        :tol:       float with tolerance for interval width
    
    Returns:
        :initLengths:       list with floats of individual boxes initial lengths 

    """
    initLengths = []
        
    #for k in range(0, len(initVarBounds)):
        #notSolvedVarBounds = initVarBounds
        #curBoxsolvedVarsNo = solvedVarsNo[k]
        #if curBoxsolvedVarsNo !=0:
        #    for curSolvedVarId in solvedVarsID:
        #        if curSolvedVarId[0] == k: 
                    #notSolvedVarBounds = numpy.delete(notSolvedVarBounds,k)
        #            initVarBounds[k] = tol
                    
                    
            #initLengths.append(calcVolumeLength(notSolvedVarBounds, dim_reduced[k]))       
            #initLengths.append(calcVolumeLength(initVarBounds, len(initVarBounds)), tol)   
        #else:
    initLengths.append(calcVolumeLength(initVarBounds, len(initVarBounds), tol))
            
    return initLengths


def calcVolumeLength(box, dim, tol):
    """ calculates box edge length assuming it as a hypercube
    
    Args:
        :box:       list with variable bounds in mpmath.mpi logic
        :dim:       box dimension as integer
        :tol:       float with tolerance for interval width
    
    Returns:
        :length:   box edge length as float
    
    """
    
    length = 1.0
    
    #solvedID, solvedVarsNo = getSolvedVars([box])
    #dim =dim - solvedVarsNo[0]
    
    for interval in box:
        if isinstance(interval, mpmath.iv.mpf):
            width = float(mpmath.mpf(interval.delta))
        else:
            width = interval[1] - interval[0]
        if width >= tol:
            length*=(width)**(1.0/dim)
        else:
            length*=(tol)**(1.0/dim)
    return length


def calc_vol_fraction(box, init_box, tol):
    """ calculates box edge length assuming it as a hypercube
    
    Args:
        :box:       list with variable bounds in mpmath.mpi logic
        :dim:       box dimension as integer
        :tol:       float with tolerance for interval width
    
    Returns:
        :length:   box edge length as float
    
    """
    
    vol_frac = 1.0
    
    #solvedID, solvedVarsNo = getSolvedVars([box])
    #dim =dim - solvedVarsNo[0]
    
    for i, iv in enumerate(box):
        if isinstance(iv, mpmath.iv.mpf):
            width = float(mpmath.mpf(box[i].delta))
        else:
            width = iv[1] - iv[0]
        if isinstance(init_box[i], mpmath.iv.mpf):
            width_0 = float(mpmath.mpf(init_box[i].delta))
        else: 
            width_0 = init_box[i][1] - init_box[i][0]
                    
        if width >= tol:
            vol_frac*=(width/width_0)
        else:
            vol_frac*=(tol/width_0)
    return vol_frac

def calc_residual(model, solv_options): 
    residual = []
    
    
    if model.xBounds != []:
        x_old = model.stateVarValues[0]
        for i,box in  enumerate(model.xBounds):
            model.stateVarValues[0] =  numpy.array([float(mpmath.mpf(iv.mid)) for iv in box])
            mosca.main.scaleSystem(model, solv_options)
            #for j, iv in enumerate(box): x.append(float(mpmath.mpf(iv.mid)))
            try: 
                residual.append(sum([abs(fi)/model.rowSca[j] for j, fi 
                                     in enumerate(model.fLamb(*model.stateVarValues[0]))]))
            except: 
                residual.append(float('nan'))
        model.stateVarValues[0] = x_old
    return residual
        
        
def calc_average_length(dict_options, init_box, boxes):
    avLength = 0
    avLength_0 = 0
    
    for i, iv in enumerate(boxes[0]):
        for box in boxes:
            avLength += (box[i][1] - box[i][0])
            #else: avLength +=  tol
    
        avLength_0 +=  (init_box[0][i][1] - init_box[0][i][0]) 
      
    return avLength/len(boxes)/ avLength_0
    

def calc_hypercubic_length(dict_options, init_box, boxes):
    tol = dict_options["absTol"]
    tot_vol_fraction = 0
    
    for box in boxes:
        tot_vol_fraction += calc_vol_fraction(box, init_box[0], tol)
    
    return (tot_vol_fraction)**(1.0/len(init_box[0]))


def calcVolume(box, tol):
    volume = 1.0
    
    for interval in box:
        if isinstance(interval, mpmath.iv.mpf):
            width = float(mpmath.mpf(interval.delta))
        else:
            width = interval[1] - interval[0]
        if width >= tol:
            volume*=width
        else:
            volume*=tol
    return volume    


def calc_length(boxes, tol):
    dim = len(boxes[0])
    volume = 0.0
    for box in boxes:
        volume += calcVolume(box, tol)
    return volume**(1.0/dim)
    
    
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
                solvedVarsID.append([i,j])
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


def calcHypercubicLength(boxes, tol):
    """ calculates hypercubic lengths of boxes
    
    Args:
        :boxes:     list with boxes and that contain intervals in mpmath.mpi formate
        :tol:       float with tolerance for interval width
        
    Returns:
        :lengths:   list with hypercubic lengths as floats

    """
    
    lengths = []
    for box in boxes:
        lengths.append(calcVolumeLength(box, len(box), tol))
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
            if isinstance(ratio, float): ratioOfVar += ratio
            
        if ratioOfVar == 0: ratioOfVar='solved'
        boundRatiosOfVars.append(ratioOfVar)
        
    return boundRatiosOfVars
    

def getLengthFractions(initLengths, lengths):
    """ calculates hypercubic length fractions of box edge lengths referring 
    to the lngth of th initial box
   
    Args:
    :initLengths:   list with edge lengths of initial volume (as hypercube)
    :lengths:       list with edgie lengths of redced box volumes 
                    (as hypercubes) as floats

    Returns:
    :lengthFractions: list with hypercubic length fractions as float

    """
    
    lengthFractions = []
    for k in range(0, len(lengths)):
        lengthFractions.append(lengths[k] / initLengths[k])
        
    return lengthFractions


def getHyperCubicLengthFraction(initLengths, lengths, dim_reduced):
    """ calculates length fraction of all reduced boxes / intial box
    
    Args:
        :initLengths:   list with edge lengths of initial volume (as hypercube)
        :lengths:       list with hypercubic lengths of reduced boxes as floats
        :dim_reduced:   list with dimensions of reduced boxes as integer
        
    Returns:     hypercubic length fraction as mpmath.mpf (gets to higher nubmers than float)
    
    """
    
    reduced_volume =  0.0

    for i in range(0, len(lengths)):
        reduced_volume+=(mpmath.mpf(lengths[i]))**dim_reduced[i]
        
    return reduced_volume**(1.0/max(dim_reduced))/initLengths[i]
 

def getDensityOfJacoboan(model):
    """ returns nonzero density of jacobian matrix from equation system
    
    Args:
        :model:         instance of class Model
        
    Return:     ratio of nonzero entries to total number of entries in jacobian (mxm)
    
    """
    
    model.jacobian, f = mod.getCasadiJandF(model.xSymbolic, model.fSymbolic)
    return float(model.getCasadiJacobian().nnz()) / model.getModelDimension()**2
    

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
    res_file.write("\nHypercubicLengthFraction\t ")
    #for j in range(0, noOfVarSets):
    #    res_file.write("%s\t"%(hypercubicLFractions[j]))
    res_file.write("%s"%(hypercubicLFraction)) 
    
    if solvedVars != []:
        res_file.write("\n\nFollowing Variables have been solved:\n")
        for solvedVar in solvedVars:
            res_file.write("%s (VarBound_%s)\n" %(varSymbolic[solvedVar[1]], 
                                                               solvedVar[0]))
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
    