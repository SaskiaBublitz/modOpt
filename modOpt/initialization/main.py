""" Sample selection """

"""
***************************************************
Import packages
***************************************************
"""
import copy
import time
import numpy
from modOpt.initialization import parallelization, VarListType, Sampling, onePointInit #,axOptimization
import modOpt.storage as mostge

"""
********************************************************
All methods to specify a specific sample point selection
********************************************************
"""
__all__ = ['get_samples_with_n_lowest_residuals', 'doSampling', 'sample_box',
           'sample_box_in_block', 'func_optuna_timeout']


def create_tear_block(fun_tear, id_tear_fun, block):
    var_ID_in_tear_fun = []
    for f in fun_tear: var_ID_in_tear_fun += f.glb_ID
    var_ID_in_tear_fun = list(set(var_ID_in_tear_fun)) # unique values
    tear_block = copy.copy(block)
    tear_block.colPerm = var_ID_in_tear_fun
    tear_block.rowPerm = id_tear_fun
    return tear_block


def doSampling(model, bxrd_options, smpl_options):
    """ samples multiple boxes with selected method from smpl_options and stores
    the ones with the minimum functional residual.
    
    Args:
        :model:             instance of class model
        :bxrd_options:      dictionary with user-settings regarding parallelization
                            and box reduction steps applied before
        :smpl_options:  dictionary with number of samples to generate and 
                            sampleNo_min_residual which is the number of candidates
                            with lowest residuals that are stored

    Returns:                None

    """
    
    res = {}
    tic = time.time()
    

    if smpl_options["smplNo"] == 0:
        fileName = bxrd_options["fileName"]\
            +"_r"+str(bxrd_options["redStep"])\
            +"_s"+str(smpl_options["smplNo"])\
            +".npz"
        onePointInit.setStateVarValuesToMidPointOfIntervals({"Model": model}, bxrd_options)
        allx = copy.deepcopy(model.stateVarValues)
        for boxID in range(0, len(model.xBounds)):
            model.stateVarValues = numpy.array([allx[boxID]])
            print ("Function residuals of sample points:\t", model.getFunctionValuesResidual())
            mostge.store_list_in_npz_dict(fileName, model.stateVarValues, boxID)
    else:
        fileName = bxrd_options["fileName"]\
            +"_r"+str(bxrd_options["redStep"])\
            +"_s"+str(smpl_options["smplNo"])\
            +"_smin"+str(smpl_options["smplBest"])\
            +".npz"
        if bxrd_options["parallelization"]: 
            res =parallelization.sample_box(model, smpl_options, bxrd_options)
        else:
            for boxID in range(0, len(model.xBounds)):
                res = sample_box(model, boxID, smpl_options, bxrd_options, res)
                
        for boxID in range(0,len(model.xBounds)):     
            mostge.store_list_in_npz_dict(fileName, res[boxID], boxID)

    if bxrd_options["timer"]:  
        print("Time: ", time.time() - tic, " sec.")
        mostge.store_time(fileName, [time.time() - tic], len(model.xBounds))


def sample_box(model, boxID, smpl_options, bxrd_options, res):
    """ samples one box with ID boxID by selected method from smpl_options and 
    stores the samples with the minimum functional residual.
    
    Args:
        :model:             instance of class model
        :boxID:             ID of current box as integer
        :smpl_options:  dictionary with number of samples to generate and 
                            sampleNo_min_residual which is the number of candidates
                            with lowest residuals that are stored
        :bxrd_options:      dictionary with user-settings regarding parallelization
                            and box reduction steps applied before
        :res:               dictionary for storage of samples with minimum residual
                            for all boxes

    Returns:                updated dictionary res by samples of current box

    """

    iterVars = VarListType.VariableList(performSampling=True, 
                                  numberOfSamples=smpl_options["smplNo"],
                                  samplingMethod=smpl_options["smplMethod"], 
                                  samplingDistribution='uniform',
                                  seed=None,
                                  distributionParams=(),
                                  model=model, 
                                  boxID=boxID)
    iterVarsSampler = Sampling.Variable_Sampling(iterVars, number_of_samples=iterVars.numberOfSamples)

    iterVars.sampleData = numpy.array(iterVarsSampler.create_samples())
    sampleNo = smpl_options['sampleNo_min_resiudal']   
    
    res[boxID] = get_samples_with_n_lowest_residuals(model, sampleNo, iterVars.sampleData)
    return res
    

def do_optuna_optimization_in_block(block, boxID, smpl_options, bxrd_options):
    import optuna
    onePointInit.setStateVarValuesToMidPointOfIntervals({"Block": block}, bxrd_options) 
    residual =  numpy.linalg.norm(block.getFunctionValues())
    if residual > bxrd_options["absTol"]:  
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        #box = block.xBounds_tot
        var_names = [str(var_name)  for var_name in block.x_sym_tot]
        #sampler = optuna.samplers.CmaEsSampler()
        sampler = optuna.samplers.TPESampler()
        study = optuna.create_study(direction='minimize', 
                                    sampler=sampler)
        study.optimize(lambda trial: objective_2(trial, var_names, 
                                           block), 
                   n_trials=smpl_options["smplNo"])
        sample = [study.best_params[var_names[c]] for c in block.colPerm]
    else:
        sample = block.x_tot[block.colPerm]
    return sample


def func_optuna_timeout(block, boxID, smpl_options, bxrd_options):
    from func_timeout import func_timeout, FunctionTimedOut
    args = (block, boxID, smpl_options, bxrd_options)
    try:
        sample = func_timeout(0.5, do_optuna_optimization_in_block, args)
    except FunctionTimedOut:
        print("Sampling took too long, hence midpoint of box is taken")
        samples = {}
        old_val = smpl_options["smplNo"]
        smpl_options["smplNo"]= 0
        sample_box_in_block(block, boxID, smpl_options, bxrd_options, samples)
        sample =  samples[0][0]
        smpl_options["smplNo"]=old_val
        
    return sample

    
def objective(trial, box, var_names, functions):
    variables = []
    quadratic_error_sum = 0
    for i, var in enumerate(var_names):  
        variables.append(trial.suggest_float(var, box[i][0], box[i][1])) 
    
    for f in functions: 
        f_vars = [variables[i] for i in f.glb_ID]        
        quadratic_error_sum += f.f_numpy(*f_vars)**2
        
    return quadratic_error_sum


def objective_2(trial, var_names, block):
    iter_vars = []
    
    for c in block.colPerm:
        iter_vars.append(trial.suggest_float(var_names[c], 
                                                 block.xBounds_tot[c][0],
                                                 block.xBounds_tot[c][1])) 
    block.x_tot[block.colPerm] = iter_vars
    return sum([fi**2 for fi in block.get_functions_values()])
    #return sum([fi**2 for fi in block.getFunctionValues()])
    
            
def sample_box_in_block(block, boxID, smpl_options, bxrd_options, res):
    """ samples one box with ID boxID by selected method from smpl_options and 
    stores the samples with the minimum functional residual.
    
    Args:
        :model:             instance of class model
        :boxID:             ID of current box as integer
        :smpl_options:  dictionary with number of samples to generate and 
                            sampleNo_min_residual which is the number of candidates
                            with lowest residuals that are stored
        :bxrd_options:      dictionary with user-settings regarding parallelization
                            and box reduction steps applied before
        :res:               dictionary for storage of samples with minimum residual
                            for all boxes

    Returns:                updated dictionary res by samples of current box

    """
    onePointInit.setStateVarValuesToMidPointOfIntervals({"Block": block}, bxrd_options)
    residual =  numpy.linalg.norm(block.getFunctionValues())
    if smpl_options["smplNo"] == 0 or residual < 1e-6:
        
        res[boxID] = [block.x_tot[block.colPerm]]
        #print("Function resiudals of sample point: ", residual)

    else:        
        iterVars = VarListType.VariableList(performSampling=True, 
                                  numberOfSamples=smpl_options["smplNo"],
                                  samplingMethod=smpl_options["smplMethod"], 
                                  samplingDistribution='uniform',
                                  seed=None,
                                  distributionParams=(),
                                  block=block, 
                                  boxID=boxID)
        iterVarsSampler = Sampling.Variable_Sampling(iterVars, number_of_samples=iterVars.numberOfSamples)

        iterVars.sampleData = numpy.array(iterVarsSampler.create_samples())
        sampleNo = smpl_options['sampleNo_min_resiudal']   
    
        res[boxID] = get_samples_with_n_lowest_residuals_in_block(block, sampleNo, iterVars.sampleData)
    return res


def get_samples_with_n_lowest_residuals_in_block(block, n, sampleData):
    """ tests the first n samples that have the lowest function residuals from 
    sampleData
    
    Args:
        :model:             instance of class model
        :n:                 integer with real number of tested samples
        :sampleData:        numpy array sampling points 
    
    Returns:                numpy array with n samples with lowest residual
    
    """
    residuals = []
    
    # Calc residuals:
    #if bxrd_options != parallel: 
    for curSample in sampleData:
        block.x_tot[block.colPerm]=curSample
        
        residuals.append(numpy.linalg.norm(block.getFunctionValues()))
    
    # Sort samples by minimum residuals
    residuals = numpy.array(residuals)
    sample_index = numpy.argsort(residuals)
    residuals = residuals[sample_index]
    print ("Function residuals of sample points:\t", residuals[0:n])
    
    return sampleData[sample_index][0:n]



def get_samples_with_n_lowest_residuals(model, n, sampleData):
    """ tests the first n samples that have the lowest function residuals from 
    sampleData
    
    Args:
        :model:             instance of class model
        :n:                 integer with real number of tested samples
        :sampleData:        numpy array sampling points 
    
    Returns:                numpy array with n samples with lowest residual
    
    """
    residuals = []
    
    # Calc residuals:
    #if bxrd_options != parallel: 
    for curSample in sampleData:
        model.stateVarValues=[curSample]
        
        residuals.append(numpy.linalg.norm(model.getFunctionValues()))
    
    # Sort samples by minimum residuals
    residuals = numpy.array(residuals)
    sample_index = numpy.argsort(residuals)
    residuals = residuals[sample_index]
    print ("Function residuals of sample points:\t", residuals[0:n])
    
    return sampleData[sample_index][0:n]
