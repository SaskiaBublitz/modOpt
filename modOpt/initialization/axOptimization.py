""" Ax optimization """

"""
***************************************************
Import packages
***************************************************
"""
import time
import numpy as np
import warnings
import mpmath
from ax.service.managed_loop import optimize as ax_optimize
from ax.plot.trace import optimization_trace_single_method
from ax.utils.notebook.plotting import render
from ax.plot.contour import interact_contour, plot_contour



class AxOptimization():
    
    def __init__(self, block):
        self.block = block
        self.convert_bounds_of_model_to_ax_parameter()
  
        
    def convert_bounds_of_model_to_ax_parameter(self):
        self.parameter = []
        self.box = self.block.xBounds_tot[self.block.colPerm]
        self.variables = self.block.x_sym_tot[self.block.colPerm]
        
        for i, interval in enumerate(self.box):
            self.parameter.append({"name": str(self.variables[i]), 
                          "value_type": "float", "type": "range", 
                          "bounds": list(interval)})
   
            
    def tune_model_ax(self, max_iter, metric='MSE', plot=False):
        
        # a lot of pytorch warnings occure using ax. Those are ignored now.
        warnings.filterwarnings("ignore", module="ax")

        tic = time.time()
        best_parameters, _, experiment, model = ax_optimize(parameters=self.parameter,
                                                            evaluation_function=self.tuning_objective_ax_recurrent,
                                                            objective_name=metric,
                                                            minimize=True, 
                                                            total_trials=max_iter)
        toc = time.time()

        print(f'\nModel tuning finished. It took {toc-tic} seconds\nThe best settings are: {best_parameters}')
        
        best_objectives = np.array([trial.objective_mean for trial in experiment.trials.values()])
        
        iterations = [{'objective': trial.objective_mean,
                       'parameters': trial.arm.parameters} for trial in experiment.trials.values()]
        
        tuning_stats = {'best_parameters': best_parameters,
                        'tuning_metric': metric,
                        'n_iter': len(experiment.trials),
                        'best_objective_curve': np.minimum.accumulate(best_objectives),
                        'iter': iterations}
        
        return list(tuning_stats['best_parameters'].values()) 

        
    def tuning_objective_ax_recurrent(self, parameterisation):
        values = [parameterisation[var] for var in parameterisation.keys()]
        #for i,value in enumerate(parameterisation): 
        #    values.append(parameterisation[value]) 
            
        self.block.x_tot[self.block.colPerm] = values

        return np.sum(np.abs(self.block.get_functions_values()))
        #print(res)
        #if not np.isnan(res): return res
        #else: 
        #    print(np.nan_to_num(np.inf))
        #    return np.nan_to_num(np.inf)
 
        
    def convert_mpi_to_list(interval):
        return [float(mpmath.mpf(interval.a)), float(mpmath.mpf(interval.b))]
