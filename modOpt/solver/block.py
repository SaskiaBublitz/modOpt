"""
***************************************************
Imported packages
***************************************************
"""

import numpy
import casadi

"""
***************************************************
Block implementation
***************************************************
"""

__all__ =['Block']

class Block:
    """Block respresentation
    
    This module stores all information about a certain block of a decomposed
    equation system
    
    Attributes:
        
    :rowPerm:         list with global function indices of the block 
    :colPerm:         list with global iteration variable indices of the block                          
    :yb_ID:         list with global variable indices of the block
    :J_sym_tot:     symbolic jacobian (casadi) not permuted of the total system                                                
    :F_sym_tot:     symbolic function (casadi) not permuted of the total system 

                            
    """
    
    def __init__(self, rBlock, cBlock, xInF, J_sym, F_sym, x):
        """ Initialization method for class Block
        
        Args:
            :rBlock:       list with global function indices of the block 
            :cBlock:       list with global iteration variable indices of the block                          
            :xInF:         list with global variable indices of all blocks
            :J_sym:     symbolic jacobian (casadi) not permuted of the total system                                                
            :F_sym:     symbolic function (casadi) not permuted of the total system
            :x:         vector with current state variable values of the total system
            
        """
        
        self.rowPerm = rBlock
        self.colPerm = cBlock
        self.yb_ID = self.getSubsystemVariableIDs(xInF)
        self.J_sym_tot = J_sym
        self.F_sym_tot = F_sym
        self.x_tot = x
        self.rowSca = numpy.ones(len(self.rowPerm))
        self.colSca = numpy.ones(len(self.colPerm))

        
    def getSubsystemVariableIDs(self, xInF):
        """ gets global id of all variables within a certain block
        
        Args:
            :xInF:      list with global variable indices of all blocks
       
        Return:         list with global variable indices of the block 
                        identified by self.rowPerm
            
        """        
        x_all_ID = []
    
        for i in self.rowPerm:
            curVars_ID = xInF[i]
            x_all_ID.append(curVars_ID) 
        return self.getUniqueValuesInList(x_all_ID)


    def getUniqueValuesInList(self, listObj):
        """ returns only unique variable indices of the block
        
        Args:
            :listObj:   nested list in the formate [[...], [...], ...]
       
        Return:         list with unique variable indices in the formate [...]
            
        """         
        newList = []
        for subList in listObj:
            for item in subList:
                if item not in newList:
                    newList.append(item)
        return newList       
        #self.J = J
        #self.F = F
        #self.x = x
        #self.y = y

        
    def getPermutedJacobian(self):
         """ Return: block jacobian evaluated at x_tot"""
         J_tot = self.J_sym_tot(*numpy.append(self.x_tot, self.x_tot))
         return J_tot[self.rowPerm, self.colPerm]

    
    def getScaledJacobian(self):
         """ Return: scaled block jacobian evaluated at x_tot """
         return casadi.mtimes(casadi.mtimes(casadi.diag(1.0 / self.rowSca), self.getPermutedJacobian()), 
                             casadi.diag(self.colSca))


    def getFunctionValues(self):
        """ Return: block function values evaluated at x_tot """
        F_tot = numpy.array(self.F_sym_tot(*self.x_tot))
        return F_tot[self.rowPerm]


    def getPermutedFunctionValues(self):
        """ Return: block function values evaluated at x_tot """
        F_tot = numpy.array(self.F_sym_tot(*self.x_tot))
        return F_tot[self.rowPerm]


    def getScaledFunctionValues(self):
        """ Return: scaled block function values evaluated at x_tot """
        return numpy.dot(numpy.diag(1.0 / self.rowSca), self.getPermutedFunctionValues())

        
    def getIterVarValues(self):
        """ Return: block iteration variable values """       
        return self.x_tot[self.colPerm]


    def getScaledIterVarValues(self):
        """ Return: scaled block iteration variable values """       
        return self.getIterVarValues() / self.colSca 


    def setScaling(self, model):
        """ sets scaling of block due to the complete model
        
        Args:
            :model:         instance of class Model
            
        """
        
        self.rowSca = model.rowSca[self.rowPerm]
        self.colSca = model.colSca[self.colPerm]


    def updateToScaling(self, res_scaling):
        """ updates the jacobian entries jValues by the scaling factors. Mind 
        that jValues remains in global order.
        
        Args:
            :res_scaling:    dictionary containing scaling results
        
        """ 
        
       # self.jValues[self.rowPerm, self.colPerm] = res_scaling["Matrix"]
        
        if res_scaling.has_key("Equations"): 
            self.rowSca = res_scaling["Equations"]
        
        if res_scaling.has_key("Variables"):
            self.colSca = res_scaling["Variables"]