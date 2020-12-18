function [RES, fval, exitflag, output] = matlabScript(fSymbolic, xSymbolic, x0, FTOL, iterMax)
    
   options = optimset('MaxIter',iterMax,'TolFun',FTOL,'Display','iter-detailed');
   [RES,fval,exitflag,output] = fsolve (@(x) getFunctionValue(x,fSymbolic, xSymbolic), x0, options);

end
    

function x = getFunctionValue(x, fSymbolic, xSymbolic)
   x = py.modOpt.solver.matlabSolver.systemToSolve(x,fSymbolic, xSymbolic);
   x = cellfun(@double,cell(x));
end
