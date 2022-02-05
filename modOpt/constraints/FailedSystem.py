"""
***************************************************
Failed System implementation
***************************************************
"""

__all__ =['FailedSystem']

class FailedSystem:
    """Failed System
    
    This modul defines a failed System class for error analysis in variable
    bounds reduction
    
    Attributes:
        
    :critF:         function, where the solver terminated in sypmpy formate  
    :critVar:       variable with resulting empty interval in sypmpy formate 
    :varInF:        all symbolic variables in critF in sympy formate
                            
    """
    
    def __init__(self, CRITF, CRITVAR):
        """ Initialization method for class model
        
        Args:
            :CRITF:         function, where the solver terminated in sypmpy formate  
            :CRITVAR:       variable with resulting empty interval in sypmpy formate                          
            
        """
        
        self.critF = CRITF # critical equation
        self.critVar = CRITVAR # critical variable
        #self.varsInF = self.getVarsInF()

        
    def getVarsInF(self):
        """returns all variable names that the function contains"
            
        Return:
            :varsInF:       list of Strings with all variable names contained in f

        """
        varsInF= []

        for variable in self.critF.free_symbols:
            varsInF.append(variable)
        return varsInF