"""
***************************************************
Imported packages
***************************************************
"""
import sympy
import numpy
import casadi
import copy

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
    
    def __init__(self, rBlock, cBlock, xInF, J_sym, F_sym, x, xBounds, 
                 parameter=None, constraints=None, xSymbolic=None,
                 jacobianSympy =None, functions=None):
        """ Initialization method for class Block
        
        Args:
            :rBlock:       list with global function indices of the block 
            :cBlock:       list with global iteration variable indices of the block                          
            :xInF:         list with global variable indices of all blocks
            :J_sym:        symbolic jacobian (casadi) not permuted of the total system                                                
            :F_sym:        symbolic function (casadi) not permuted of the total system
            :x:            vector with current state variable values of the total system
            :xBounds:      list with bounds
            :constraints:  optional function that returns list with sympy functions
            
        """
        
        self.rowPerm = rBlock
        self.colPerm = cBlock
        self.yb_ID = self.getSubsystemVariableIDs(xInF)
        self.J_sym_tot = J_sym
        self.F_sym_tot = F_sym
        self.x_sym_tot = numpy.array(xSymbolic)
        self.x_tot = x
        self.xBounds_tot = xBounds
        self.parameter = parameter
        self.rowSca = numpy.ones(len(self.rowPerm))
        self.colSca = numpy.ones(len(self.colPerm))
        self.allConstraints = constraints
        self.jacobianSympy = jacobianSympy
        self.FoundSolutions = []
        if functions: self.functions_block = self.get_functions_block(functions)
        
    def getSubsystemVariableIDs(self, xInF):
        """ gets global id of all variables within a certain block
        
        Args:
            :xInF:      list with global variable indices of current blocks
       
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

        
    def getPermutedJacobian(self):
         """ Return: block jacobian evaluated at x_tot"""
         J_tot = self.J_sym_tot(*numpy.append(self.x_tot, self.x_tot))
         J_perm = J_tot[self.rowPerm, self.colPerm]
         
         if numpy.all(numpy.array(J_perm == J_perm)): return J_perm
         else: return self.getPermutedSympyJacobian()

         
    def getPermutedSympyJacobian(self):
        if self.jacobianSympy == []: self.jacobianSympy = self.getSympySymbolicJacobian()
        array2mat = [{'ImmutableDenseMatrix': numpy.array}, 'numpy']
        lam_f_mat = sympy.lambdify(self.x_sym_tot, self.jacobianSympy, modules=array2mat)
        jac = lam_f_mat(*self.x_tot)
        return casadi.DM(jac)[self.rowPerm, self.colPerm]


    def getSympySymbolicJacobian(self):
        f_sym = self.allConstraints(self.x_sym_tot, self.parameter)
        return sympy.Matrix(f_sym).jacobian(self.x_sym_tot)
        
    
    def getScaledJacobian(self):
         """ Return: scaled block jacobian evaluated at x_tot """
         return casadi.mtimes(casadi.mtimes(casadi.diag(1.0 / self.rowSca), self.getPermutedJacobian()), 
                             casadi.diag(self.colSca))

    def get_functions_block(self, functions):
        return [functions[i] for i in self.rowPerm]
              
        
    def get_functions_values(self):
        """ Return: block function values evaluated at x_tot """
        return numpy.array([f.f_numpy(*self.x_tot[f.glb_ID]) for f in self.functions_block])

    def getFunctionValues(self):
        """ Return: block function values evaluated at x_tot """
        F_tot = numpy.array(self.F_sym_tot(*self.x_tot))[self.rowPerm]   

        if numpy.all(F_tot==F_tot): return F_tot
        else:
            F_tot = []
            for fun in self.allConstraints(self.x_sym_tot, self.parameter):
                fun = sympy.lambdify(self.x_sym_tot, fun)
                F_tot.append(fun(*self.x_tot))
        
            return numpy.array(F_tot)[self.rowPerm]


    
    def getPermutedFunctionValues(self):
        """ Return: block function values evaluated at x_tot """
        return self.getFunctionValues()


    def getScaledFunctionValues(self):
        """ Return: scaled block function values evaluated at x_tot """
        return numpy.dot(numpy.diag(1.0 / self.rowSca), self.get_functions_values())


    def getIterVarBoundValues(self):
        """ Return: block iteration variable values """       
        return self.xBounds_tot[self.colPerm]

        
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


    def getSymbolicFunctions(self, x):
        xWithIterVars = self.subscribeXwithIterVars(x)
        return self.allConstraints(xWithIterVars, self.parameter)

    def getSymbolicVariablesOfBlock(self):
        return self.x_sym_tot[self.colPerm]
    
    def subscribeXwithIterVars(self, x):
        xWithIterVars = copy.deepcopy(self.x_tot)
        xWithIterVars[self.colPerm] = x
        return xWithIterVars
    
    
    def updateToScaling(self, res_scaling):
        """ updates the jacobian entries jValues by the scaling factors. Mind 
        that jValues remains in global order.
        
        Args:
            :res_scaling:    dictionary containing scaling results
        
        """ 
        
       # self.jValues[self.rowPerm, self.colPerm] = res_scaling["Matrix"]
        
        if res_scaling.__contains__("Equations"): 
            self.rowSca = res_scaling["Equations"]
        
        if res_scaling.__contains__("Variables"):
            self.colSca = res_scaling["Variables"]
            
    def createBlocks(self,blocks):
        """ creates blocks for permutation order 1 to n based on block border list
        from preordering algorithm
        
        Args:
            :blocks:         list with block border indices as integers
    
        Return:             list with block elements in sublist
    
        """
    
        nestedBlocks = []
        for i in range(1,len(blocks)):
            curBlock = range(blocks[i-1], blocks[i])
            nestedBlocks.append(curBlock)
            
        return nestedBlocks