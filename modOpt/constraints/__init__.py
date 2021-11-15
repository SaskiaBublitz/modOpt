"""
public functions and classes
"""
from modOpt.constraints.main import reduceVariableBounds, nestBlocks, sort_fId_to_varIds
from modOpt.constraints.update import updateDictToModel
from modOpt.constraints.analysis import (analyseResults, trackErrors, 
                                         get_hypercubic_length,calc_hypercubic_length, 
                                         calc_average_length,calc_residual)
from modOpt.constraints.results import writeResults

