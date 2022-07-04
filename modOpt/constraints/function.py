"""
***************************************************
Imported packages
***************************************************
"""
import pyibex
import mpmath
import sympy
import numpy
import casadi
from modOpt.constraints import iNes_procedure, affineArithmetic

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
    
    def __init__(self, f_sym, x_symbolic, aff=None, numpy=None, casadi=None, dfdx=None):
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
        self.var_count = [self.f_sym.count(x) for x in self.x_sym]
        self.glb_ID = self.get_glb_ID(x_symbolic)
        self.g_sym, self.b_sym = self.get_g_b_functions()
        self.dgdx_sym, self.dbdx_sym = self.get_deriv_functions(dfdx)
        self.vars_of_deriv = self.get_vars_of_deriv()
        self.deriv_is_constant = self.is_deriv_constant()
        self.f_mpmath = self.get_mpmath_functions(self.x_sym, [self.f_sym])
        self.g_mpmath = self.get_mpmath_functions(self.x_sym, self.g_sym)
        self.b_mpmath = self.get_mpmath_functions(self.x_sym, self.b_sym)
        self.dgdx_mpmath = self.get_mpmath_functions(self.x_sym, self.dgdx_sym)
        self.dbdx_mpmath = self.get_mpmath_functions(self.x_sym, self.dbdx_sym)
        self.f_pibex = self.get_pybex_function(self.x_sym, self.f_sym)
        if casadi:
            self.f_casadi = self.lambdify_to_casadi(self.x_sym, self.f_sym)
        if numpy:
            self.f_numpy = sympy.lambdify(self.x_sym, self.f_sym,'numpy')
        
        if aff:
            self.f_aff = self.get_aff_functions(self.x_sym, [self.f_sym])
            self.g_aff = self.get_aff_functions(self.x_sym, self.g_sym)
            self.b_aff = self.get_aff_functions(self.x_sym, self.b_sym)
            self.dgdx_aff = self.get_aff_functions(self.x_sym, self.dgdx_sym)
            self.dbdx_aff = self.get_aff_functions(self.x_sym, self.dbdx_sym)      
        else:
            self.f_aff = [None] * len(self.f_mpmath)
            self.g_aff = [None] * len(self.g_mpmath)
            self.b_aff = [None] * len(self.b_mpmath)
            self.dgdx_aff = [None] * len(self.dgdx_mpmath)
            self.dbdx_aff = [None] * len(self.dbdx_mpmath)
            
    def get_var_occurences(self, f_sym):      
        return [f_sym.count(x) for x in self.f_sym.free_symbols]
    
    def lambdify_to_casadi(self, x_sym, f_sym):
        """Converting operations of symoblic equation system f (simpy) to 
        arithmetic interval functions (mpmath.iv)
    
        """
    
        toCasadi = {"exp" : casadi.SX.exp,
                    "sin" : casadi.SX.sin,
                    "cos" : casadi.SX.cos,
                    "log" : casadi.SX.log,
                    "abs":  casadi.SX.fabs,
                    "sqrt": casadi.SX.sqrt,}   
        return sympy.lambdify(x_sym,f_sym, toCasadi) 
            
    def get_pybex_function(self, x_sym, f_sym):
        stringF = str(f_sym).replace('log(', 'ln(').replace('**', '^')
        string_x = [str(s) for s in x_sym]
        
        
        for s in tuple(sorted(string_x,key=len,reverse=True)):
            stringF = stringF.replace(s, 'x['+str(string_x.index(s))+']')

        pyibexFun = pyibex.Function('x['+str(len(x_sym))+']', stringF)
        return pyibex.CtcFwdBwd(pyibexFun)
    
       
    def is_deriv_constant(self):
        deriv_is_constant = []

        for i, x in enumerate(self.x_sym):
            if not x in self.vars_of_deriv[i] and self.dgdx_sym[i] * self.x_sym[i] == self.g_sym[i]:
                #and self.dgdx_sym[i] * self.x_sym[i] == self.g_sym[i]:
                # TODO: One could include the case self.dgdx_sym[i] * self.x_sym[i] != self.g_sym[i] 
                # but then had to add the constant bit dgdx*x - dgdx to bInterval
                #if not self.dgdx_sym[i] * self.x_sym[i] == self.g_sym[i]:
                #    self.b_sym[i] = sympy.simplify(self.dgdx_sym[i]*self.x_sym[i]-self.f_sym)
                #    self.g_sym[i] = self.dgdx_sym[i] * self.x_sym[i]
                #    self.dbdx_sym[i] = []
                #    for x in self.x_sym:                    
                #        self.dbdx_sym[i] += [sympy.diff(self.b_sym[i], x)]      
                deriv_is_constant.append(True)
            else: deriv_is_constant.append(False)
        return deriv_is_constant
                
                   
    def get_vars_of_deriv(self):
        vars_of_deriv = []
        for cur_deriv in self.dgdx_sym: vars_of_deriv.append(list(cur_deriv.free_symbols))
        return vars_of_deriv
    
    
    def eval_aff_function(self, box, f_aff):
        if isinstance(f_aff, mpmath.ctx_iv.ivmpf): return f_aff    
        
        try:
            box_aff = affineArithmetic.Affine.mpiList2affList(box)
            return affineArithmetic.Affine.aff2mpi(f_aff(*box_aff))
                  
        except:
            return []
    

    def eval_mpmath_function(self, box, f_mpmath):
        if isinstance(f_mpmath, mpmath.ctx_iv.ivmpf): return f_mpmath    
        try:         
            fInterval = mpmath.mpi(f_mpmath(*box))#timeout(fMpmathIV, xBounds)
            if fInterval == []: return numpy.nan_to_num(numpy.inf)*mpmath.mpi(-1,1)
            elif -numpy.inf in fInterval and numpy.inf in fInterval: return numpy.nan_to_num(numpy.inf)*mpmath.mpi(-1,1)
            elif -numpy.inf in fInterval: return mpmath.mpi(-numpy.nan_to_num(numpy.inf), fInterval.b)
            elif numpy.inf in fInterval: return mpmath.mpi(fInterval.a, numpy.nan_to_num(numpy.inf))
            
            else: return fInterval
        
        except:
            return []

    def get_aff_functions(self, x, f_sym):
        f_aff = []
        for f in f_sym:
            if isinstance(f, sympy.Float) and len(str(f)) > 15:
                f_aff.append(iNes_procedure.roundValue(f, 16))
            else: f_aff.append(iNes_procedure.lambdifyToAffapyAffine(x, f))
        return f_aff

    
    def get_mpmath_functions(self, x, f_sym):
        f_mpmath = []
        for f in f_sym:
            if isinstance(f, sympy.Float) and len(str(f)) > 15:
                f_mpmath.append(iNes_procedure.roundValue(f, 16))
            else: f_mpmath.append(iNes_procedure.lambdifyToMpmathIvComplex(x, f))
        return f_mpmath
                  

    def get_glb_ID(self,x_symbolic):
        glb_ID = []
        for x in self.x_sym:
            glb_ID.append(x_symbolic.index(x))
        return glb_ID
    
    
    def get_deriv_functions(self, dfdx=None):
        dgdx =[]
        dbdx = []
        if dfdx: dgdx = [dfdx[i] for i in self.glb_ID]
        
        for i, g in enumerate(self.g_sym):
            if not dfdx: dgdx.append(sympy.diff(g, self.x_sym[i]))          
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
        
        if f.func.class_key()[2] in ['Mul', 'sin', 'cos', 'exp', 'log', 'Pow'] :
            return f, 0.0
    
        if f.func.class_key()[2]=='Add':
            for arg in allArguments:
                if x in arg.free_symbols:
                    allArgumentsWithVariable.append(arg)
                else: allArgumentsWithoutVariable.append(arg)
                 
            fvar = sympy.Add(*allArgumentsWithVariable)
            fWithoutVar = sympy.Add(*allArgumentsWithoutVariable)
            return fvar, -fWithoutVar

    	elif f.func.class_key()[2]=='Symbol': return f, 0.0
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
        xWithIterVars = list(self.x_tot)#copy.deepcopy(self.x_tot)
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
