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

__all__ =['Function']

class Function:
    """Block respresentation
    
    This module stores all information about a certain function for the constrainted reduction
    
    Attributes:
        
        :f_sym:        function in sympy logic
        :g_sym:        list with functions of the form g(x,y) in global order
        :b_sym:        list with dicitionaries of the form {var_ID(a): 
                       function of the form b(a,z)} in global order
        :var_ID:       list with global variable indices that are part of the function
        :dgdx_sym:     list with derrivatives dgdx(x,y) in global order
        :dbda_sym:     list with dicitionaries of the form {var_ID(a): 
                       derrivative of the form dbda(a,z)} in global order    

                            
    """
    
    def __init__(self, f_sym, x_symbolic):
        """ Initialization method for class Block
        
        Args:
            :f_sym:        function in sympy logic
            :g_sym:        list with functions of the form g(x,y) in global order
            :b_sym:        list with dicitionaries of the form {var_ID(a): 
                           function of the form b(a,z)} in global order
            :var_ID:       list with global variable indices that are part of the function
            :dgdx_sym:     list with derrivatives dgdx(x,y) in global order
            :dbda_sym:     list with dicitionaries of the form {var_ID(a): 
                           derrivative of the form dbda(a,z)} in global order                                               
            
            
        """
        
        self.f_sym = f_sym
        self.x_sym = list(f_sym.free_symbols)
        self.glb_ID = self.get_glb_ID(x_symbolic)
        self.g_sym, self.b_sym = self.get_g_b_functions()
        #self.b_sym = self.get_b_functions(b, self.x_sym)
        #self.dgdx_sym = self.get_dgdx_functions(self.g_sym, self.x_sym)
        self.dgdx_sym, self.dbdx_sym = self.get_deriv_functions()

    def get_glb_ID(self,x_symbolic):
        glb_ID = []
        for x in self.x_sym:
            glb_ID.append(x_symbolic.index(x))
        return glb_ID
    
    
    def get_deriv_functions(self):
        dgdx =[]
        dbdx = []
        
        for i in range(0, len(self.g_sym)):
            dgdx.append(sympy.diff(self.g_sym[i], self.x_sym[i])) 
            
            dbdxi = [] 
            
            for x in self.x_sym:
                dbdxi.append(sympy.diff(self.b_sym[i], x)) 
                
            dbdx.append(dbdxi)  
        
        return dgdx, dbdx
    
    
    def get_g_b_functions(self):
        """ splits function in x dependent and x indepenendt parts for all x in
        the function and returns list with x in global order 
 
        Return:
            g: list with sympy functions depending on x
                fWithoutVar:    sum of x independent arguments
            
        """        
        #[g(x1,y(x1)), ..., g(xn, y(xn))]
        g = []
        b = []
        
        for x in self.x_sym :
            cg, cb = self.splitFunctionByVariableDependency(x)
            g.append(cg)
            b.append(cb)
        
        return g, b


    def splitFunctionByVariableDependency(self, x):
        """ a sympy function expression f is splitted into a sum from x depended terms
        and a sum from x independent terms. If the expressions consists of a product or
        quotient, it is only checked if this one contains x or not. If one or both of the
        resulting parts are empty they are returned as 0.
    
        Args:
            f:              mathematical expression in sympy logic
            x:              symbolic variable x
            
            Return:
                fvar:           sum of x dependent arguments
                fWithoutVar:    sum of x independent arguments
        
        """    
        f =self.f_sym
        allArguments = f.args
        allArgumentsWithVariable = []
        allArgumentsWithoutVariable = []
        
        if f.func.class_key()[2]=='Mul':f = f + numpy.finfo(numpy.float).eps/2
    
        if f.func.class_key()[2]=='Add':
                
            for i in range(0, len(allArguments)):
                if x in allArguments[i].free_symbols:
                    allArgumentsWithVariable.append(allArguments[i])
                else: allArgumentsWithoutVariable.append(allArguments[i])
                 
            fvar = sympy.Add(*allArgumentsWithVariable)
            fWithoutVar = sympy.Add(*allArgumentsWithoutVariable)
            return fvar, -fWithoutVar
        
        #if f.func.class_key()[2]=='Mul':
        #    if x in f.free_symbols: return f, 0
        #    else: return 0, f
    
        else: 
            print("Problems occured during function parsing")
            return 0, 0


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
        
        if res_scaling.has_key("Equations"): 
            self.rowSca = res_scaling["Equations"]
        
        if res_scaling.has_key("Variables"):
            self.colSca = res_scaling["Variables"]