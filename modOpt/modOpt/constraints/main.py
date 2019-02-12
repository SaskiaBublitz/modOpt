""" Interval union Newton method """

"""
***************************************************
Import packages
***************************************************
"""

import iNcomplete
import time
#TODO: import iNpartial

"""
***************************************************
Main that invokes methods for variable constraints reduction
***************************************************
"""
__all__ = ['reduceVariableBounds']

def reduceVariableBounds(model, options):
    """ variable bounds are reduced based on user-defined input
    
    Args: 
        model:       object of class model in modOpt.model that contains all
                     information of the NLE-evaluation from MOSAICm. 
        options:     dictionary with user-specified information
        
    Returns:
        model:       model with reduced XBounds
        iterNo:      number of outter iteration Steps
        t or []:     if timer-option has been selected the time the algorith takes
                     is returned, otherwise an empty list is returned
    
    """
    
    if options['method'] == 'complete':
        if options['timer'] == True: 
            tic = time.clock()
            reducedModel, iterNo = iNcomplete.doIntervalNesting(model, options)
            toc = time.clock()
            t = toc - tic
            return reducedModel, iterNo, t
        
        else:
            reducedModel, iterNo = iNcomplete.doIntervalNesting(model, options)
            return reducedModel, iterNo, []
    
#    TODO: if options['method'] == 'partial':
#        if options['timer'] == True: 
#            tic = time.clock()
#            reducedModel, iterNo = iNcomplete.doIntervalNesting(model, options)
#            toc = time.clock()
#            t = tic - toc
#            return reducedModel, iterNo, t
#        
#        else:
#            reducedModel, iterNo = iNcomplete.doIntervalNesting(model, options)
#            return reducedModel, iterNo, []
