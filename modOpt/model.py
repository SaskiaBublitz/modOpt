"""
***************************************************
Imported packages
***************************************************
"""

import numpy
import casadi

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
        
    :stateVarValues:        list with arrays of dimension m with current values 
                            of state variables. (in global order)
    :xBounds:               list with arrays of dimension m with current sets of
                            state variable bounds. 
                            (in global order)                            
    :xSymbolic:             sympy array of dimension m with symbolic expression 
    :fSymbolic:             sympy array with functions of equation system
    :jacobian:              casadi array with symbolic jacobian matrix
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
    :fSymCasadi:            symbolic functions in casadi logic
                            global order, default = None
    :rowSca:                numpy array with equation scaling factors
    :colSca:                numpy array with variable scaling factors
    :failed:                if true model status is well, if false procedure
                            failed to find a proper solution for the system
                            
    """
    
    def __init__(self, X, BOUNDS, XSYMBOLIC, FSYMBOLIC, PARAMS, JACOBIAN, FSYMCASADI = None, CONSTRAINTS = None):
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
            :FSYMCASADI:   optional 1D-array with current symbolic functions in
                           global order, default = None
            :CONSTRAINTS:  optional function that returns list with sympy functions
            
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
        self.fSymCasadi = FSYMCASADI 
        self.rowSca = numpy.ones(len(X))
        self.colSca = numpy.ones(len(X))
        self.failed = False
        self.constraints = CONSTRAINTS


    def getBoundsOfPermutedModel(self, xBounds, xSymbolic, parameter):
        """ Calculates bounds of the Function residuals based on 
        current state variable bounds.
        
        Args:
            :xBounds:          numpy array with state variables bounds
            :xSymbolic:        sympy array with symbolic state variables
            :parameter:        numpy array with parameter values
        
        Return:              numpy array with residual bounds
            
        """
        
        Fsym = self.fSymbolic
        
        FsymPerm = reorderList(Fsym, self.rowPerm)
        xSymbolicPerm = reorderList(xSymbolic, self.colPerm)
        xBoundsPerm = reorderList(xBounds, self.colPerm)
  
        return FsymPerm, xSymbolicPerm, xBoundsPerm


    def getConditionNumber(self):
        """ Return: Condition number of original system """
        return numpy.linalg.cond(self.getJacobian())


    def getPermutedConditionNumber(self):
        """ Return: Condition number of permuted system """
        return numpy.linalg.cond(self.getPermutedJacobian())    

    
    def getPermutedAndScaledConditionNumber(self):
        """ Return: Condition number of permuted and scaled system """
        return numpy.linalg.cond(self.getPermutedAndScaledJacobian())     

    
    def getFunctionValues(self):
        """ Return: Function Values at current state variable values"""
        return numpy.array(self.fSymCasadi(*self.stateVarValues[0]))
    
    
    def getScaledFunctionValues(self):
        """ Return: scaled function Values at current state variable values"""        
        return numpy.dot(numpy.diag(1.0 / self.rowSca), self.getFunctionValues())
        
    
    def getPermutedFunctionValues(self):
        """ Return: permuted and scaled function Values at current state variable values"""
        return self.getFunctionValues()[self.rowPerm] 


    def getPermutedAndScaledFunctionValues(self):
        """ Return: permuted and scaled function Values at current state variable values"""
        return self.getScaledFunctionValues()[self.rowPerm] 


    def getFunctionValuesResidual(self):    
        """ Return: euclydian norm of current function values"""
        return numpy.linalg.norm(self.getPermutedFunctionValues()) 

    
    def getJacobian(self):
        """ Return: jacobian evaluated at current state variable values"""
        return self.jacobian(*numpy.append(self.stateVarValues[0], self.stateVarValues[0]))


    def getScaledJacobian(self):
        """ Return: scaled jacobian evaluated at current state variable values"""
        return casadi.mtimes(casadi.mtimes(casadi.diag(1.0 / self.rowSca), self.getJacobian()), 

                             casadi.diag(self.colSca))

    def getPermutedJacobian(self):
        """ Return: permuted jacobian evaluated at current state variable values"""
        return self.getJacobian()[self.rowPerm, self.colPerm]

 
    def getPermutedAndScaledJacobian(self):
        """ Return: permuted and scaled jacobian evaluated at current state variable values"""
        return self.getScaledJacobian()[self.rowPerm, self.colPerm]

                           
    def getScaledXValues(self):
         """ Return: Scaled state variable values in global order"""
         return self.stateVarValues[0] / self.colSca          

    
    def getPermutedXValues(self):
         """ Return: Permuted state variable values"""
         return self.getXValues()[self.colPerm] 


    def getPermutedAndScaledXValues(self):
         """ Return: Permuted and scaled state variable values"""
         return self.getScaledXValues()[self.colPerm] 

     
    def getBlockID(self):
        blockIDinGlbOrder =[]
        for glbID in range(0, len(self.xSymbolic)):
            permID = self.rowPerm.index(glbID)
            
            for b in range(0, len(self.blocks)):
                if permID in self.blocks[b]:
                    blockIDinGlbOrder.append(b)
                    break
        return blockIDinGlbOrder        


    def getXBoundsOfCertainVariablesFromIntervalSet(self, certainVariables, intervalID):
        """ returns bounds of certain variables from the model for a specific interval
        set accessed by intervalID
        
        Args:
            :certainVariables:          list with variable names as sympy.symbols
            :intervalID:                Position of interval set in xBounds list
                                        of the model
        Returns:
            :certainXBounds:            numpy array with referring xBounds         
        
        """
        
        certainXBounds = numpy.empty((len(certainVariables)), dtype=object)
        curXBounds = self.xBounds[intervalID]
        
        for i in range(0, len(certainVariables)):
            glbID = self.xSymbolic.index(certainVariables[i])
            certainXBounds[i] = curXBounds[glbID]
   
        return certainXBounds
        
    
    def setXBounds(self, xBounds):
        """ sets the state variable bounds to the intervals given by the array
        xBounds
        
        """
        
        self.xBounds = xBounds
        
    
    def updateToPermutation(self, rowPerm, colPerm, blocks):
        """ updates the row and column order and block shape list
        given by the lists rowPerm and colPer and blocks
        
        Args:
            :rowPerm:       list with row permutation indices
            :colPerm:       list with column permutation indices
            :blocks:        nested list indicating block structure referring
                            to permuted index, i.e. [[1,2], [3,4,5],...]
        
        """        
        
        self.colPerm = colPerm
        self.rowPerm = rowPerm
        self.blocks =  createBlocks(blocks)
 
       
    def updateToScaling(self, res_scaling):
        """ updates the jacobian entries jValues by the scaling factors. Mind 
        that jValues remains in global order.
        
        Args:
            :res_scaling:    dictionary containing scaling results
        
        """ 
        
        if res_scaling.has_key("Equations"): 
            self.rowSca[self.rowPerm] = res_scaling["Equations"]
        
        if res_scaling.has_key("Variables"):
            self.colSca[self.colPerm] = res_scaling["Variables"]

                        
    def getModelDimension(self):
        "returns system dimension"
        return len(self.xSymbolic)

            
def createBlocks(blocks):
    """ creates blocks for permutation order 1 to n based on block border list
    from preordering algorithm
    
    Args:
        :blocks:         list with block border indices as integers
    
    Return:             list with block elements in sublist
    
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
        
    Return:          sorted list
        
    """
    
    newList = []

    for i in newOrder:
        newList.append(myList[i])
    return newList