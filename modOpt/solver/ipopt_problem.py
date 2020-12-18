import numpy

class Ipopt_problem(object):
    def __init__(self, curBlock):
        self.curBlock = curBlock
    
    def objective(self, x): 
        self.curBlock.x_tot[self.curBlock.colPerm] = x
        f = self.curBlock.getFunctionValues()
        m = 0
        
        for fi in f: m += fi**2.0 
             
        return m
    def gradient(self, x):    
        grad_m = numpy.zeros(len(x)+1)
    
        f = self.curBlock.getFunctionValues()
        jac = self.curBlock.getPermutedJacobian()
    
        for j in range(0, len(x)):
            for i in range(0, len(f)):
                grad_m[j+1] += 2 * f[i] * jac[i, j]
            
        return numpy.array(grad_m, dtype=float)
    
    def constraints(self, x):
        return numpy.array([]) 