""" Sample selection """

"""
***************************************************
Import packages
***************************************************
"""
import copy
import time
import numpy
from modOpt.initialization import parallelization, VarListType, Sampling, arithmeticMean, axOptimization
import modOpt.storage as mostge

"""
********************************************************
All methods to specify a specific sample point selection
********************************************************
"""
__all__ = ['get_samples_with_n_lowest_residuals', 'doSampling', 'sample_box',
           'sample_box_in_block', 'do_ax_optimization_in_block', 'do_tear_sampling',
           'do_optuna_optimization_in_block', 'func_optuna_timeout']

def do_ax_optimization(model, dict_options, sampling_options):
    sampling_options["max_iter"] = 1000
    axObj = axOptimization.AxOptimization(model)
    axObj.tune_model_ax(sampling_options["max_iter"])


def do_tear_sampling(model, cur_block, box_id, sampling_options, dict_options,functions,solv_options):
    from modOpt.decomposition import MC33
    import modOpt.solver as mos
    samples = {}
    if (len(cur_block.colPerm) <= 1.0):      
        return sample_box_in_block(cur_block, box_id, sampling_options, dict_options, samples)
    else:
        block_model = copy.deepcopy(model)
        res_permutation = MC33.doMC33(cur_block.getPermutedJacobian())
        block_model.blocks = cur_block.createBlocks(res_permutation["Number of Row Blocks"])
                
        org_fun_id_resorted = [cur_block.rowPerm[i] for i in res_permutation["Row Permutation"] ]
        org_var_id_resorted = [cur_block.colPerm[i] for i in res_permutation["Column Permutation"]]
        id_tear_fun = [org_fun_id_resorted[i] for i in block_model.blocks[-1]]
        id_tear_var = [org_var_id_resorted[i] for i in block_model.blocks[-1]]
        block_model.rowPerm = org_fun_id_resorted
        block_model.colPerm = org_var_id_resorted       
        fun_tear = [functions[i] for i in id_tear_fun]
        sampling_options["sampleNo_min_resiudal"]=1
        tear_block = create_tear_block(fun_tear, id_tear_fun, cur_block)
        #samples = sample_box_in_block(cur_block, box_id, sampling_options, dict_options, samples)
        #samples = sample_box_in_block(tear_block, box_id, sampling_options, dict_options, samples)
        #tear_block.x_tot[tear_block.colPerm]=samples[0][0]
        #cur_block.x_tot[cur_block.colPerm] = samples[0][0]
        #print ("This is the residual after sampling of the block: ", 
        #       numpy.linalg.norm(cur_block.getFunctionValues()))
        #block_model.stateVarValues[0][cur_block.colPerm]=samples[0][0]
        #print ("This is the residual after sampling of the whole system: ", 
        #       numpy.linalg.norm(numpy.array(block_model.getFunctionValues())))
        #tear_block.colPerm = id_tear_var
        sample = do_ax_optimization_in_block(cur_block, 0, sampling_options, dict_options)
        #tear_block.x_tot[tear_block.colPerm]=sample
        cur_block.x_tot[cur_block.colPerm] =sample
        print ("This is the residual after using ax on the tear variables of the block: ", 
               numpy.linalg.norm(cur_block.getFunctionValues()))
        block_model.stateVarValues[0][cur_block.colPerm]=sample
        print ("This is the residual after sampling and using ax of the whole system: ", 
               numpy.linalg.norm(numpy.array(block_model.getFunctionValues())))
        del dict_options["sampling"]
        res_solver = mos.solveBlocksSequence(block_model, solv_options, dict_options) 
        block_model = res_solver["Model"] 
        print ("This is the residual of the block after sampling, using ax and solving all block functions", 0," :", 
               numpy.linalg.norm(numpy.array(block_model.getFunctionValues())[block_model.rowPerm])) 
        print ("This is the residual after sampling, using ax and solving all block functions of the whole system: ", 
               numpy.linalg.norm(numpy.array(block_model.getFunctionValues())))
        
        cur_block.x_tot =block_model.stateVarValues
        tear_block.x_tot=block_model.stateVarValues
        sample = do_ax_optimization_in_block(tear_block, 0, sampling_options, dict_options)
        
        tear_block.x_tot[tear_block.colPerm]=sample
        cur_block.x_tot[tear_block.colPerm]=sample
        block_model.stateVarValues[0][cur_block.colPerm]=sample
        
        print ("This is the residual of the block after sampling, using ax and solving all block functions", 0," :", 
               numpy.linalg.norm(numpy.array(block_model.getFunctionValues())[block_model.rowPerm])) 
        print ("This is the residual after sampling, using ax and solving all block functions of the whole system: ", 
               numpy.linalg.norm(numpy.array(block_model.getFunctionValues())))
        
        dict_options["sampling"] = True
        return [[block_model.stateVarValues[0][cur_block.colPerm]]]

def create_tear_block(fun_tear, id_tear_fun, block):
    var_ID_in_tear_fun = []
    for f in fun_tear: var_ID_in_tear_fun += f.glb_ID
    var_ID_in_tear_fun = list(set(var_ID_in_tear_fun)) # unique values
    tear_block = copy.copy(block)
    tear_block.colPerm = var_ID_in_tear_fun
    tear_block.rowPerm = id_tear_fun
    return tear_block


def doSampling(model, dict_options, sampling_options):
    """ samples multiple boxes with selected method from sampling_options and stores
    the ones with the minimum functional residual.
    
    Args:
        :model:             instance of class model
        :dict_options:      dictionary with user-settings regarding parallelization
                            and box reduction steps applied before
        :sampling_options:  dictionary with number of samples to generate and 
                            sampleNo_min_residual which is the number of candidates
                            with lowest residuals that are stored

    Returns:                None

    """
    
    res = {}
    tic = time.time()
    

    if sampling_options["number of samples"] == 0:
        fileName = dict_options["fileName"]\
            +"_r"+str(dict_options["redStep"])\
            +"_s"+str(sampling_options["number of samples"])\
            +".npz"
        arithmeticMean.setStateVarValuesToMidPointOfIntervals({"Model": model}, dict_options)
        allx = copy.deepcopy(model.stateVarValues)
        for boxID in range(0, len(model.xBounds)):
            model.stateVarValues = numpy.array([allx[boxID]])
            print ("Function residuals of sample points:\t", model.getFunctionValuesResidual())
            mostge.store_list_in_npz_dict(fileName, model.stateVarValues, boxID)
    else:
        fileName = dict_options["fileName"]\
            +"_r"+str(dict_options["redStep"])\
            +"_s"+str(sampling_options["number of samples"])\
            +"_smin"+str(sampling_options["sampleNo_min_resiudal"])\
            +".npz"
        if dict_options["parallelization"]: 
            res =parallelization.sample_box(model, sampling_options, dict_options)
        else:
            for boxID in range(0, len(model.xBounds)):
                res = sample_box(model, boxID, sampling_options, dict_options, res)
                
        for boxID in range(0,len(model.xBounds)):     
            mostge.store_list_in_npz_dict(fileName, res[boxID], boxID)

    if dict_options["timer"]:  
        print("Time: ", time.time() - tic, " sec.")
        mostge.store_time(fileName, [time.time() - tic], len(model.xBounds))


def sample_box(model, boxID, sampling_options, dict_options, res):
    """ samples one box with ID boxID by selected method from sampling_options and 
    stores the samples with the minimum functional residual.
    
    Args:
        :model:             instance of class model
        :boxID:             ID of current box as integer
        :sampling_options:  dictionary with number of samples to generate and 
                            sampleNo_min_residual which is the number of candidates
                            with lowest residuals that are stored
        :dict_options:      dictionary with user-settings regarding parallelization
                            and box reduction steps applied before
        :res:               dictionary for storage of samples with minimum residual
                            for all boxes

    Returns:                updated dictionary res by samples of current box

    """

    iterVars = VarListType.VariableList(performSampling=True, 
                                  numberOfSamples=sampling_options["number of samples"],
                                  samplingMethod=sampling_options["sampling method"], 
                                  samplingDistribution='uniform',
                                  seed=None,
                                  distributionParams=(),
                                  model=model, 
                                  boxID=boxID)
    iterVarsSampler = Sampling.Variable_Sampling(iterVars, number_of_samples=iterVars.numberOfSamples)

    iterVars.sampleData = numpy.array(iterVarsSampler.create_samples())
    sampleNo = sampling_options['sampleNo_min_resiudal']   
    
    res[boxID] = get_samples_with_n_lowest_residuals(model, sampleNo, iterVars.sampleData)
    return res
    

def do_ax_optimization_in_block(block, boxID, sampling_options, dict_options):
    arithmeticMean.setStateVarValuesToMidPointOfIntervals({"Block": block}, dict_options)
    residual =  numpy.linalg.norm(block.getFunctionValues())
    if not residual == 0:
        #sampling_options["max_iter"] = 10
        axObj = axOptimization.AxOptimization(block)
        sampling_options["max_iter"] = len(block.colPerm)*3
        sample = axObj.tune_model_ax(sampling_options["max_iter"])
        block.x_tot[block.colPerm] = sample
    else:
        sample = block.x_tot[block.colPerm]
    print("The residual of the sample point is: ", numpy.linalg.norm(block.getFunctionValues()))
    return sample
    

def do_optuna_optimization_in_block(block, boxID, sampling_options, dict_options):
    import optuna
    arithmeticMean.setStateVarValuesToMidPointOfIntervals({"Block": block}, dict_options) 
    residual =  numpy.linalg.norm(block.getFunctionValues())
    if residual > dict_options["absTol"]:  
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        box = block.xBounds_tot
        var_names = [str(var_name)  for var_name in block.x_sym_tot]
        sampler = optuna.samplers.CmaEsSampler()
        study = optuna.create_study(direction='minimize', 
                                    sampler=sampler)
        study.optimize(lambda trial: objective_2(trial, var_names, 
                                           block), 
                   n_trials=sampling_options["number of samples"])
        sample = [study.best_params[var_names[c]] for c in block.colPerm]
    else:
        sample = block.x_tot[block.colPerm]
    return sample


def func_optuna_timeout(block, boxID, sampling_options, dict_options):
    from func_timeout import func_timeout, FunctionTimedOut
    args = (block, boxID, sampling_options, dict_options)
    try:
        sample = func_timeout(0.5, do_optuna_optimization_in_block, args)
    except FunctionTimedOut:
        print("Sampling took too long, hence midpoint of box is taken")
        samples = {}
        old_val = sampling_options["number of samples"]
        sampling_options["number of samples"]= 0
        sample_box_in_block(block, boxID, sampling_options, dict_options, samples)
        sample =  samples[0][0]
        sampling_options["number of samples"]=old_val
        
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
    return sum([fi**2 for fi in block.getFunctionValues()])
    
        

    
def sample_box_in_block(block, boxID, sampling_options, dict_options, res):
    """ samples one box with ID boxID by selected method from sampling_options and 
    stores the samples with the minimum functional residual.
    
    Args:
        :model:             instance of class model
        :boxID:             ID of current box as integer
        :sampling_options:  dictionary with number of samples to generate and 
                            sampleNo_min_residual which is the number of candidates
                            with lowest residuals that are stored
        :dict_options:      dictionary with user-settings regarding parallelization
                            and box reduction steps applied before
        :res:               dictionary for storage of samples with minimum residual
                            for all boxes

    Returns:                updated dictionary res by samples of current box

    """
    arithmeticMean.setStateVarValuesToMidPointOfIntervals({"Block": block}, dict_options)
    residual =  numpy.linalg.norm(block.getFunctionValues())
    if sampling_options["number of samples"] == 0 or residual < 1e-6:
        
        res[boxID] = [block.x_tot[block.colPerm]]
        #print("Function resiudals of sample point: ", residual)

    else:        
        iterVars = VarListType.VariableList(performSampling=True, 
                                  numberOfSamples=sampling_options["number of samples"],
                                  samplingMethod=sampling_options["sampling method"], 
                                  samplingDistribution='uniform',
                                  seed=None,
                                  distributionParams=(),
                                  block=block, 
                                  boxID=boxID)
        iterVarsSampler = Sampling.Variable_Sampling(iterVars, number_of_samples=iterVars.numberOfSamples)

        iterVars.sampleData = numpy.array(iterVarsSampler.create_samples())
        sampleNo = sampling_options['sampleNo_min_resiudal']   
    
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
    #if dict_options != parallel: 
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
    #if dict_options != parallel: 
    for curSample in sampleData:
        model.stateVarValues=[curSample]
        
        residuals.append(numpy.linalg.norm(model.getFunctionValues()))
    
    # Sort samples by minimum residuals
    residuals = numpy.array(residuals)
    sample_index = numpy.argsort(residuals)
    residuals = residuals[sample_index]
    print ("Function residuals of sample points:\t", residuals[0:n])
    
    return sampleData[sample_index][0:n]