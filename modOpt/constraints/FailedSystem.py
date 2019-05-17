"""
***************************************************
Imported packages
***************************************************
"""

import numpy
import casadi

"""
***************************************************
CritEquation implementation
***************************************************
"""

__all__ =['FailedSystem']

class FailedSystem:
    """Critical Equation
    
    This modul defines a failed System class for error analysis in variable
    bounds reduction
    
    Attributes:
        
    :stateVarValues:        list with arrays of dimension m with current values 
                            of state variables. (in global order)
    :xBounds:               list with arrays of dimension m with current sets of
                            state variable bounds. 
                            (in global order)                            
    :xSymbolic:             sympy array of dimension m with symbolic expression 
                            of state variables. (in global order)
    :parameter:             array with values for model parameter                                                    
    :rowPerm:               array of dimension m with permutation order of 
                            equations regarding their global index (1...m)
    :colPerm:               array of dimension n with permutation order of 
                            variables regarding their global index (1...n)
    :blocks:                List with blocks referring to permuted index (not 
                            global index) Example: 
                            For a 5x5 system a structure could be:
                            [[0], [1, 2], [3], [4]] with one 2-dimensional block 
                            consisting of the permuted equations and variables 
                            2 and 3.
    :fValues:               optional 1D-array with current function values in
                            global order, default = None
    :jValues:               optional 2D.array with current jacobian entries in
                            global order, default = None
                            
    """
    
    def __init__(self, CRITF, CRITVAR):
        """ Initialization method for class model
        
        Args:
            :X:            array with m initial values for the state variables. 
                           (in global order)
            :BOUNDS:       array of dimension m with current values of the
                           interval bounds of the state variables. 
                           (in global order)                            
            :XSYMBOLIC:    sympy array of dimension m with symbolic expression
                           of state variables. (in global order)
            :PARAMS:       array with values of model parameter 
            :JACOBIAN:     if used, array with symbolic jacobian            
            :FVALUES:      optional 1D-array with current function values in
                           global order, default = None
            :JVALUES:      optional 2D.array with current jacobian entries in
                           global order, default = None
            
        """
        
        self.critF = CRITF # critical equation
        self.critVar = CRITVAR # critical variable
        self.varsInF = self.getVarsInF()

        
    def getVarsInF(self):
        """returns all variable names that the function contains"
            
        Return:
            :varsInF:       list of Strings with all variable names contained in f

        """
        varsInF= []

        for variable in self.critF.free_symbols:
            varsInF.append(variable)
        return varsInF