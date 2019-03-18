"""
***************************************************
Imported packages
***************************************************
"""

import numpy

"""
***************************************************
Model implementation
***************************************************
"""

__all__ =['Model']

class Model:
    """Mathematical Model
    
    This module stores all information about state variable bounds, parameters, 
    and the permutation arrays referring to decomposition methods like 
    Dulmage-Mendelsohn.
    
    Attributes:
        
    :stateVarValues:         list with arrays of dimension m with current values 
                            of state variables. (in global order)
    :xBounds:                list with arrays of dimension m with current sets of
                            state variable bounds. 
                            (in global order)                            
    :xSymbolic:              sympy array of dimension m with symbolic expression 
                            of state variables. (in global order)
    :parameter:              array with values for model parameter                                                    
    :rowPerm:                array of dimension m with permutation order of 
                            equations regarding their global index (1...m)
    :colPerm:                array of dimension n with permutation order of 
                            variables regarding their global index (1...n)
    :blocks:                  List with blocks referring to permuted index (not 
                            global index) Example: 
                            For a 5x5 system a structure could be:
                            [[0], [1, 2], [3], [4]] with one 2-dimensional block 
                            consisting of the permuted equations and variables 
                            2 and 3.
                            
    """
    
    def __init__(self, X, BOUNDS, XSYMBOLIC, FSYMBOLIC, PARAMS, JACOBIAN):
        """ Initialization method for class model
        
        Args:
            :X:             array with m initial values for the state variables. 
                           (in global order)
            :BOUNDS:        array of dimension m with current values of the
                           interval bounds of the state variables. 
                           (in global order)                            
            :XSYMBOLIC:     sympy array of dimension m with symbolic expression
                           of state variables. (in global order)
            :PARAMS:        array with values of model parameter 
            :JACOBIAN:      if used, array with symbolic jacobian 
            
        """
        
        self.stateVarValues = [X]
        self.xBounds = [BOUNDS]
        self.xSymbolic = XSYMBOLIC
        self.fSymbolic = FSYMBOLIC
        self.jacobian =  JACOBIAN
        self.parameter = PARAMS
        self.rowPerm = range(0,len(X))
        self.colPerm = range(0,len(X))
        self.blocks = [self.rowPerm]


    def getSymbolicFunctions(self):
        """ :Return: symbolic equations of equation system"""
        return self.fSymbolic


    def getBoundsOfPermutedModel(self, xBounds, xSymbolic, parameter):
        """ Calculates bounds of the Function residuals based on 
        current state variable bounds.
        
        Args:
            :xBounds:          numpy array with state variables bounds
            :xSymbolic:        sympy array with symbolic state variables
            :parameter:        numpy array with parameter values
        
        :Return:              numpy array with residual bounds
            
        """
        
        Fsym = self.getSymbolicFunctions()
        
        FsymPerm = reorderList(Fsym, self.rowPerm)
        xSymbolicPerm = reorderList(xSymbolic, self.colPerm)
        xBoundsPerm = reorderList(xBounds, self.colPerm)
  
        return FsymPerm, xSymbolicPerm, xBoundsPerm

      
    def getJacobian(self, X):
        """ :Return: jacobian evaluated at X"""
        return self.jacobian(*numpy.append(X, X))
    
         
    def getParameter(self):
        """ :Return: paramter values """
        return self.parameter


    def getXBounds(self):
        """ :Return: current state variable bounds"""
        return self.xBounds


    def getXSymbolic(self):
        """ :Return: symbolic variables"""        
        return self.xSymbolic


    def getXValues(self):
        """ :Return: current state variable values"""        
        return self.stateVarValues


    def setXBounds(self, xBounds):
        """ sets the state variable bounds to the intervals given by the array
        xBounds
        
        """
        
        self.xBounds = xBounds
        

    def updateToPermutation(self, rowPerm, colPerm, blocks):
        """ updates the row and column order of the Jacobian to the indices 
        given by the lists rowPerm and colPerm.
        
        """        
        
        self.colPerm = colPerm
        self.rowPerm = rowPerm
        self.blocks =  createBlocks(blocks)
        

def createBlocks(blocks):
    """ Creates blocks for permutation order 1 to n based on block border list
    from preordering algorithm
    
    Args:
        :blocks:         list with block border indices as integers
    
    :Return:             list with block elements in sublist
    
    """
    
    nestedBlocks = []
    for i in range(1, len(blocks)):
        curBlock = range(blocks[i-1], blocks[i])
        nestedBlocks.append(curBlock)
        
    return nestedBlocks

        
def reorderList(myList, newOrder):
    """ reorders given list myList
    
    Args:
        :myList:      list with objects
        :newOrder:    list with new order indices
        
    :Return:          sorted list
        
    """
    
    newList = []

    for i in newOrder:
        newList.append(myList[i])
    return newList