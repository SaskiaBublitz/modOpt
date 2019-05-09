"""
***************************************************
Imported packages
***************************************************
"""

import numpy

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
        
    :fb_ID:         list with global function indices of the block 
    :xb_ID:         list with global iteration variable indices of the block                          
    :yb_ID:         list with global variable indices of the block
    :J_sym_tot:     symbolic jacobian (casadi) not permuted of the total system                                                
    :F_sym_tot:     symbolic function (casadi) not permuted of the total system 

                            
    """
    
    def __init__(self, rBlock, cBlock, xInF, J_sym, F_sym):
        """ Initialization method for class Block
        
        Args:
            :rBlock:       list with global function indices of the block 
            :cBlock:       list with global iteration variable indices of the block                          
            :xInF:         list with global variable indices of all blocks
            :J_sym:     symbolic jacobian (casadi) not permuted of the total system                                                
            :F_sym:     symbolic function (casadi) not permuted of the total system 
            
        """
        
        self.fb_ID = rBlock
        self.xb_ID = cBlock
        self.yb_ID = self.getSubsystemVariableIDs(xInF)
        self.J_sym_tot = J_sym
        self.F_sym_tot = F_sym

    
    def getSubsystemVariableIDs(self, xInF):
        """ gets global id of all variables within a certain block
        
        Args:
            :xInF:      list with global variable indices of all blocks
       
        Return:         list with global variable indices of the block 
                        identified by self.fb_ID
            
        """        
        x_all_ID = []
    
        for i in self.fb_ID:
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
    
    
    def getJacobian(self, x_tot):
        """ Return: jacobian evaluated at x_tot"""
        
        J_tot = self.J_sym_tot(*numpy.append(x_tot, x_tot))
        return J_tot[self.fb_ID, self.xb_ID]
     
        
    def getFunctionValues(self, x_tot):
        """ Return: function values evaluated at x"""
        
        F_tot = numpy.array(self.F_sym_tot(*x_tot))
        return F_tot[self.fb_ID]
        
    def getIterVarValues(self, x_tot):
        """ Return: iteration variable values"""
        
        return x_tot[self.xb_ID]
