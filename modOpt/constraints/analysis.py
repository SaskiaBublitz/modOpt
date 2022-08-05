# -*- coding: utf-8 -*-
"""
***************************************************
analysis tools
***************************************************

This module contains all methods to calculate analysis parameters to 
characterize the performance of the hybrid approach. It also includes
methods to ouput the analysis data graphically or as text file.

"""

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
import modOpt.constraints as moc

"""
***************************************************
Methods
***************************************************
"""

__all__ = ['analyseResults', 'trackErrors', 'get_hypercubic_length',
           'calc_hypercubic_length', 'calc_average_length', 'calc_residual',
           'identify_average_box_reduction','initialize_with_boxFile',
           'initialize_with_boxFiles']

def analyseResults(bxrd_options, res_solver):
    """ main function to analyse results from hybrid approach and store output
    to text file.
    
    Args:
        bxrd_options (dict) :   contains user settings
        res_solver (dict) :     contains model and results from hybrid approach 
    
    """    
    modelWithReducedBounds = res_solver["Model"]
    varSymbolic = res_solver["Model"].xSymbolic
    tol = bxrd_options["absTol"]
    init_box = res_solver["init_box"]
    initLength = calc_length(init_box, tol)
    #initLength = calcVolumeLength(initVarBounds, len(varSymbolic),tol) # volume calculation failed for large systems with large initial volumes

    if modelWithReducedBounds != []:
        reduced_boxes = modelWithReducedBounds.xBounds
        solvedVarsID, solvedVarsNo = getSolvedVars(reduced_boxes)
        #dim_reduced = getReducedDimensions(solvedVarsNo, len(varSymbolic))
        #initLengths = calcInitLengths(init_box, tol)
        
        boundsRatios = getVarBoundsRatios(init_box[0], reduced_boxes)

        #lengths = calcHypercubicLength(reduced_boxes, tol)        
        boundRatiosOfVars = getBoundRatiosOfVars(boundsRatios)

        lengthFractions = 0#getLengthFractions(initLengths, lengths)
        #hypercubicLFraction = get_hypercubic_length(bxrd_options, 
        #                                            initVarBounds, 
        #                                            reducedVarBounds)
        abl = identify_average_box_reduction(reduced_boxes, init_box[0])
        #hypercubicLFraction = getHyperCubicLengthFraction(initLengths, lengths, dim_reduced)
        density = getDensityOfJacoboan(modelWithReducedBounds)
        nonLinRatio = getNonLinearityRatio(modelWithReducedBounds)
        
        block_dim_max = max([len(block) for block in 
                             modelWithReducedBounds.blocks])
        
        if "largest_block_analysis" in bxrd_options.keys():
            (nonzero_ratio, 
             nonlin_ratio) = get_largest_block_analytics(modelWithReducedBounds, 
                                                         block_dim_max)
            write_analysis_largestBlock(bxrd_options["fileName"], nonzero_ratio, 
                                        nonlin_ratio, block_dim_max)

            
        writeAnalysisResults(bxrd_options["fileName"], varSymbolic, 
                             boundsRatios, boundRatiosOfVars, initLength, 
                             lengthFractions, abl, solvedVarsID, density, 
                             nonLinRatio,block_dim_max)


def get_largest_block_analytics(model, block_dim_max):
    """Get nonzero and nonlinearity ratio of the largest block(s)
    
    Args:
        model (modOpt.model.Model) :   instance of class Model
        block_dim_max (int) :          largest block(s) dimension
        
    Returns:
        nonzero_ratio (float) :         ratio of nonzeros (nnz) to n x n 
                                        entries of largest block
        nonlinearity_ratio (float) :    ratio of nonlinear entries to nnz of
                                        largest block
    
    """ 
    block_dim_max_id =  [model.blocks.index(block) for block in 
                         model.blocks if len(block) == block_dim_max]
    complex_grade = 0
    for block_id in block_dim_max_id:
        cur_block = model.all_blocks[block_id]
        cur_nonzero_ratio = (cur_block.get_nonzeros_of_jacobian() / 
                             block_dim_max**2)
        cur_nonlinearity_ratio = getNonLinearityRatio_block(cur_block)
    
        if (cur_nonzero_ratio + cur_nonlinearity_ratio) > complex_grade: 
            complex_grade = cur_nonzero_ratio + cur_nonlinearity_ratio
            nonzero_ratio = cur_nonzero_ratio
            nonlinearity_ratio = cur_nonlinearity_ratio
    
    return nonzero_ratio, nonlinearity_ratio


def initialize_with_boxFile(model, textFile_name):
    """Read out bounds and initial values from text file and assigns them
    to model object
    
    Args:
        model (modOpt.model.Model) :    instance of class Model
        textFile_name (str) :           text file name

    """   
    box_to_init = open(textFile_name,'r').readlines()
    for l,line in enumerate(box_to_init):
        if l < 2: continue
        elements = line.split()
        i = list(model.xSymbolic).index(sympy.symbols(elements[0]))
        model.stateVarValues[0][i] = float(elements[1])
        model.xBounds[0][i] = mpmath.mpi(elements[2], elements[3])


def initialize_with_boxFiles(model, textFile_names):
    """Initialize model from text file wit variable bounds
    
    Args:
        model (modOpt.model.Model) :    instance of class Model
        textFile_name (str) :           text file name

    """
    x_init = model.stateVarValues[0]
    box_init = model.xBounds[0]
    model.stateVarValues = []
    model.xBounds = []
    for textFile_name in textFile_names:
        x = x_init
        box = box_init
        box_to_init = open(textFile_name,'r').readlines()
        for l,line in enumerate(box_to_init):
            if l < 2: continue
            elements = line.split()
            i = list(model.xSymbolic).index(sympy.symbols(elements[0]))
            x[i] = float(elements[1])
            box[i] = mpmath.mpi(elements[2], elements[3]) 
        model.xBounds.append(box)    
        model.stateVarValues.append(x)


def get_hypercubic_length(bxrd_options, init_box, reduced_boxes):
    """Calcuate hypercubic length ratio of reduced boxes
    
    Args:
        bxrd_options (dict) :   contains user settings
        init_box (list) :       contains list with initial variable bounds
        reduced_boxes (list) :  contains lists with reduced variable bounds
    
    Returns:
        hypercubic length (float)
        
    """
    tol = bxrd_options["absTol"]
    initLength = calc_length(init_box, tol)
    length = calc_length(reduced_boxes, tol)
    
    return length / initLength


def calc_vol_fraction(box, init_box, tol):
    """Calculate box edge length assuming it as a hypercube
    
    Args:
        box (list) :       list with variable bounds in mpmath.mpi formate
        dim (int) :        box dimension
        tol (float):       tolerance for interval width consideration 
    
    Returns:
        vol_frac (float) :   box edge length
    
    """    
    vol_frac = 1.0
    
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


def calc_residual(model, solv_options=None): 
    """Calculate residuals of current boxes contained in model
    
    Args:
        model (modOpt.model.Model) :    instance of class Model
        solv_options (dict) :           contains numerical solver settings
    
    Returns:
        residual (list) :     contains float values of boxes' residuals
    
    """   
    residual = []
     
    if model.xBounds != []:
        x_old = model.stateVarValues[0]
        for i,box in  enumerate(model.xBounds):
            model.stateVarValues[0] =  numpy.array([float(mpmath.mpf(iv.mid)) 
                                                    for iv in box])
            if solv_options:
                if "scaling" in solv_options.keys():
                    if solv_options["scaling"] != "None":
                        mosca.main.scaleSystem(model, solv_options)
            try: 
                fsum = sum([abs(fi)/model.rowSca[j] for j, fi in 
                            enumerate(model.fLamb(*model.stateVarValues[0]))])
            except: 
                fsum = numpy.inf
                
            if numpy.isnan(fsum): fsum = numpy.inf
            residual.append(fsum)
            
        model.stateVarValues[0] = x_old
        
    return residual


def calc_iv_residual(model): 
    """Check for degenerate, discontinuous intervals with undefined or 
    infinite function values. They get a large resiudal value.
    
    Args:
        model (modOpt.model.Model) :    instance of class Model
        solv_options (dict) :           contains numerical solver settings
    
    Returns:
        residual (list) :     contains float values of boxes' residuals
    
    """     
    
    
    residual = []
    if model.xBounds != []:

        for k,box in  enumerate(model.xBounds):
            r_box = 0            
            for f in model.functions:
                sbox = [box[i] for i in f.glb_ID]
                r_iv = float(mpmath.mpf(f.f_mpmath[0](*sbox).delta))
                if r_iv == numpy.nan or r_iv == numpy.inf:
                    break
                else: r_box += r_iv
            if r_iv == numpy.nan: residual.append(1.0e300)     
            else: residual.append(r_box)   
            
            
    if set(residual) == set([1.0e300]): return calc_residual(model)
    else: return residual        
        
def calc_average_length(bxrd_options, init_box, boxes):
    """ Sum up interval widths of each current box and divide it by the number
    of variables nvar to get the average interval width of th box. All average
    interval widths of the boxes are then sumed up and divided by the current
    number of boxes to get an overall average value. Similarly the average 
    interval width is determined for th initial box and the average
    value with respect to all current boxes is divided by the latter to get
    the so-caled average box length.
    

    
    Args:
        bxrd_options (dict) :   contains user settings
        init_box (list) :       contains numpy.array with initial box
        boxes (list) :          contains numpy.arrays with current boxes
        
    Returns:
        average box length (float)
        
    """
    avLength = 0
    avLength_0 = 0
    
    all_boxes = init_box + boxes

    for j, box in enumerate(all_boxes):
        nvar = len(box)
        al_box = 0
        
        if isinstance(box[0], mpmath.iv.mpf):
            box = [[float(mpmath.mpf(iv.a)), float(mpmath.mpf(iv.b))] 
                  for iv in list(box)]
        for i, iv in enumerate(box):
            if numpy.isclose(iv[0], iv[1], bxrd_options["relTol"], 
                             bxrd_options["absTol"]):
                nvar -= 1
            else:
                if not j == 0: al_box += (iv[1] - iv[0])
                else: avLength_0 += (iv[1] - iv[0])
        
        if nvar == 0 and not j==0: continue
        elif not j==0: avLength += al_box/nvar 
        elif nvar == 0 and j == 0: return 0.0     
        else: avLength_0 = avLength_0/nvar 
      
    return avLength/len(boxes)/ avLength_0
    

def calc_hypercubic_length(bxrd_options, init_box, boxes):
    """ Firstly, the volume fractions of all current boxes with respect to 
    the initial box are determined. Secondly, the geometric mean is calculated
    over the number of variables. 
        
    Args:
        bxrd_options (dict) :   contains user settings
        init_box (list) :       contains numpy.array with initial box
        boxes (list) :          contains numpy.arrays with current boxes
        
    Returns:
        hybpercubic box length (float)
        
    """
    tol = bxrd_options["absTol"]
    tot_vol_fraction = 0
    
    for box in boxes:
        tot_vol_fraction += calc_vol_fraction(box, init_box[0], tol)
    
    return (tot_vol_fraction)**(1.0/len(init_box[0]))


def calcVolume(box, tol):
    """ The absolute volume of the box is calculated
        
    Args:
        box (list) :        contains box with intervals in mpmath.mpi or as
                            list
        tol (float) :       dimensions with interval widths lower than tol
                            are ignored as these variables are solved and 
                            therefore constant
        
    Returns:
        box volume (float)
    
    """
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
    """ calculate hypercubic length which is the geometric mean of the sum of
    all current box volumes over all variables
        
    Args:
        boxes (list) :     contains current boxes as list
        tol (float) :      dimensions with interval widths lower than tol
                           are ignored as these variables are solved and 
                           therefore constant
        
    Returns:
        hypercubic length
    
    """
    dim = len(boxes[0])
    volume = 0.0
    for box in boxes:
        volume += calcVolume(box, tol)
        
    return volume**(1.0/dim)
    
    
def getSolvedVars(boxes):
    """ filter out variable intervals with zero width (solved)
    
    Args:
        boxes (list) :          contains numpy.arrays with current boxes
    
    Returns:
        solvedVarsID (list) :   nested list [[i,j],...] with i box-ID and j 
                                variable-ID of solved interval as integer
        solvedVarsNo (list) :   contains numbers of solved intervals in the 
                                boxes as integer

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
    """ determine the dimension of the reduced boxes where solvedVarsNo 
    variables have been solved.
    
    Args:
        solvedVarsNo (list) :   contains numbers of solved intervals in the 
                                boxes as int
        dim (int) :             integer with dimension of initial box
    
    Returns:
        dim_reduced (list):     contains dimension of reduced boxes as integer

    """
    if solvedVarsNo == []: return [dim]
    
    dim_reduced = []
    for soledVarsNoBox in solvedVarsNo:
        dim_reduced.append(dim - soledVarsNoBox)
    return dim_reduced
            
    
def getVarBoundsRatios(init_box, boxes):
    """ calculate ratios of reduced variable bounds to initial variable bounds
    
    Args:
        init_box (list) :       contains numpy.array with initial box
        boxes (list) :          contains numpy.arrays with current boxes
    
    Returns:                    
        boxes_ratio :           list with ratio of reduced variable bounds
        
    """    
    boxes_ratio = []
    
    for cur_box in boxes:
        boxes_ratio.append(calcBoxRatios(init_box, cur_box))

    return boxes_ratio


def getBoundRatiosOfVars(bound_ratios):
    """ calculates total variable bound ratios through summing up all different
    variable intervals of the boxes.
    
    Args:
        bound_ratios (list):   contains lists with bound ratios as float values 
                               for all variables in all reduced boxes 

    Returns:
        :boundRatiosOfVars:   list with sums of unique variable bound ratios

    """
    boundRatiosOfVars = []   
    boxNo = len(bound_ratios)
    dim = len(bound_ratios[0])
    
    for i in range(0, dim):
        varBoundRatios = []
        ratioOfVar = 0.0
        for j in range(0, boxNo):
            if not bound_ratios[j][i] in varBoundRatios:
                varBoundRatios.append(bound_ratios[j][i])
             
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
        model (modOpt.model.Model) :         instance of class Model
        
    Returns:     ratio of nonzero entries to total number of entries in jacobian (mxm)
    
    """
    
    model.jacobian, f = mod.getCasadiJandF(model.xSymbolic, model.fSymbolic)
    return float(model.getCasadiJacobian().nnz()) / model.getModelDimension()**2
    

def getNonLinearityRatio(model):
    """ identifies nonlinear dependencies of variables in jacobian matrix by 
    second derrivate and counts all nonlinear entries to calculate the ratio
    of nonlinear entries to total number of entries in jacobian.
    
    Args:
        model (modOpt.model.Model) :     istance of class Model
    
    Returns:         float of ratio: nonlinear entries / total entries
        
    """

    nonLin = 0

    for curX in model.xSymbolic:
        for curF in model.fSymbolic:
            if curX in curF.free_symbols:
                d2fdx = sympy.diff(sympy.diff(curF, curX), curX)
                if d2fdx != 0: nonLin = nonLin + 1
                
    return nonLin / float(model.getJacobian().nnz())            
    

def getNonLinearityRatio_block(block):
    """ identifies nonlinear dependencies of variables in jacobian matrix by 
    second derrivate and counts all nonlinear entries to calculate the ratio
    of nonlinear entries to total number of entries in jacobian.
    
    Args:
        model (modOpt.model.Model) :     istance of class Model
    
    Returns:         float of ratio: nonlinear entries / total entries
        
    """

    nonLin = 0

    for curX in block.getSymbolicVariablesOfBlock():
        for curF in block.functions_block:
            if curX in curF.x_sym:
                d2fdx = sympy.diff(sympy.diff(curF.f_sym, curX), curX)
                if d2fdx != 0: nonLin += 1
                
    return nonLin / float(block.get_nonzeros_of_jacobian()) 


def write_analysis_largestBlock(fileName, nonzero_ratio, nonlin_ratio, block_dim_max):
    res_file = open(''.join([fileName,"_largestBlock_analysis.txt"]), "w") 
    res_file.write("***** Results of Largest Block Analysis *****\n\n") 
    
    res_file.write("Largest Block dimension: \t%s\n"%(block_dim_max))
    res_file.write("Block's Jacobian Nonzero Density: \t%s\n"%(nonzero_ratio))
    res_file.write("block's Jacobian Nonlinearity Ratio: \t%s\n"%(nonlin_ratio))
   

def writeAnalysisResults(fileName, varSymbolic, boundRatios, boundRatioOfVars, initVolume,
                         hypercubicLFractions, hypercubicLFraction, solvedVars, density,
                         nonLinRatio,block_dim_max):
    """ writes anaylsis results to a textfile  <fileName>_analysis.txt
    
    Args:
        :fileName:                    string with file name
        :varSymbolic:                 list with symbolic variables in sympy logic
        boundRatios (list) :                 list with reduced variable bound tp initial 
                                      variable bound ratio (has only one entry if 
                                      one set of variable bounds remains)
        boundRatioOfVars (list) :            sum of the unique bound ratios of one variable
        initVolume (float) :                  volume of initial variable bound set
        hypercubicLFractions (list) :        list with fractional length of each sub-hypercube
        hypercubicLFraction (float) :         If the volumes were hypercubic, the hypercubicLFraction
                                      equals their edge length reduction
        solvedVars (list) :                  list with indices of solved variables
        density (float) :                     float with nonzero density of jacobian
        nonLinRatio (float) :                 float with nonlinear entries / total entries of jacobian
        
    """
    
    res_file = open(''.join([fileName,"_analysis.txt"]), "w") 
    
    res_file.write("***** Results of Analysis *****\n\n") 
    
    noOfVarSets = len(boundRatios)
    res_file.write("System Dimension: \t%s\n"%(len(boundRatios[0]))) 
    res_file.write("Largest Block dimension %s\n"%(block_dim_max))
    res_file.write("Jacobian Nonzero Density: \t%s\n"%(density))
    res_file.write("Jacobian Nonlinearity Ratio: \t%s\n"%(nonLinRatio))
    res_file.write("Length of initial box: \t%s\n"%(initVolume))
    res_file.write("AverageLengthFraction: \t%s"%(hypercubicLFraction))
        
    if solvedVars != []:
        res_file.write("\n\nFollowing Variables have been solved:\n")
        for solvedVar in solvedVars:
            res_file.write("%s (VarBound_%s)\n" %(varSymbolic[solvedVar[1]], 
                                                               solvedVar[0]))
    res_file.write("\n\nVariables\t") 
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
        boundRatios (list) :       list with bound ratios
    
    Returns:                 list with volume fractions all variable bound sets
    
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
        initVarBounds (list) :       list with initial variable bounds
        box (list) :           current box as a list with mpmath.mpi values
    Returns:

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
        
    Returns:                   current variable bound ratio as float value
    
    """
    if float(mpmath.mpf(initVarBound.delta)) != 0.0:
        bratio = float(mpmath.mpf(curVarBound.delta)) / float(mpmath.mpf(initVarBound.delta))
    else: 
        return "Warning: Initial interval with zero width"
    if bratio == 0: return 'solved'
    return bratio


def trackErrors(res_solver, bxrd_options):
    """ proofs if current state of model is correctly and write the error that
    occured in case the model failed to an error text file.
    
    Args:
        :initialModel:      instance of class Model at initial point
        res_solver (dict) :        dictionary with solver output
        bxrd_options (dict) :   dictionary with user specified settings
        
    """
    if res_solver["Model"].failed:
        file_name = bxrd_options["fileName"] + "_errorAnalysis.txt"
        failed_mod = res_solver["Model"]
        failed_sys = res_solver["noSolution"]
        f_crit_sym = failed_sys.critF    
        f_crit =  [f for f in failed_mod.functions if f.f_sym == f_crit_sym][0]
        var_crit = failed_sys.critVar 
        init_box = moc.analysis.convert_mpi_box_float(res_solver["init_box"][0])
        init_failed_box = numpy.array([list(init_box[glb_id]) for glb_id in f_crit.glb_ID])
        if (hasattr(failed_sys, "critBox")) : failed_box = failed_sys.critBox
        else: 
            failed_box = failed_mod.getXBoundsOfCertainVariablesFromIntervalSet(f_crit.x_sym, 0)
        
        writeErrorAnalysis(file_name, f_crit.f_sym, var_crit, f_crit.x_sym, 
                           init_failed_box, failed_box)


def writeErrorAnalysis(fileName, fCrit, varCrit, varsInF, xBoundsInitial, xBoundsFailed=None) :
    res_file = open(fileName, "w") 
    res_file.write("***** Error Analysis *****\n\n")    

    res_file.write("Algorithm failed because it could not find any solution for %s in equation:\n\n %s\n\n"%(varCrit, fCrit)) 
     
    res_file.write("The following table shows the initial and final bounds of all variables in this equation before termination:\n\n")
                    
    
    if not xBoundsFailed:
        for i in range(0, len(varsInF)):
            res_file.write("VARNAME \t INITBOUNDS \n" )
            res_file.write("%s \t %s \n"%(varsInF[i],  str(xBoundsInitial[i])))
    else:
        res_file.write("VARNAME \t INITBOUNDS \t FINALBOUNDS \n" )
        for i in range(0, len(varsInF)):
            res_file.write("%s \t %s \t %s \n"%(varsInF[i],  str(xBoundsInitial[i]), str(xBoundsFailed[i])))
     
    res_file.close()


def identify_interval_reduction(box_new,box_old):
    '''calculates the side length ratio of all intervals from a new box
    to an old box

    Args:
        :box_new:        new variable bounds of class momath.iv
        :box_old:        old variable bounds of class momath.iv
        
    Returns:
        :w_ratio:    list with side length ratio for all interval
        
    '''
    w_ratio = []
    
    for i, iv in enumerate(box_old):
        if (iv[1]-iv[0])>0: 
            w_ratio.append((box_new[i][1] - box_new[i][0]) / (iv[1]-iv[0]))
        else:
            w_ratio.append(0)
                                             
    return w_ratio    

def identify_average_box_reduction(boxes_new,box_old): 

    if isinstance(boxes_new[0][0], mpmath.iv.mpf):
            boxes_new = [convert_mpi_box_float(box) for box in list(boxes_new)]
    if isinstance(box_old[0], mpmath.iv.mpf):
            box_old = convert_mpi_box_float(box_old)
            
    return (sum([identify_box_reduction(box, box_old) for box in boxes_new])/
            len(boxes_new))


def identify_box_reduction(box_new,box_old):
    '''calculates the average side length reduction from old to new Box

    Args:
        :box_new:        new variable bounds of class momath.iv
        :box_old:        old variable bounds of class momath.iv
        
    Returns:
        :av_w_ratio:    average sidelength reduction
        
    '''
    av_w_ratio = 0
    n_vars = len(box_old)
    for i, iv in enumerate(box_old): 
        if (iv[1] -iv[0])>0:
            av_w_ratio += (box_new[i][1] - box_new[i][0])/(iv[1] - iv[0])    
        else:
            n_vars -= 1
    if n_vars == 0: return 0.0                                           
    return av_w_ratio/len(box_old)


def convert_mpi_box_float(box):
    return [[float(mpmath.mpf(iv.a)), float(mpmath.mpf(iv.b))] for iv in list(box)]
    
    
    
    