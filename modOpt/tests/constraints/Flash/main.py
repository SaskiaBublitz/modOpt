#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a test main file to proof that the box reduction works as expected.
Firstly, it is checked if the algorithm runs without terminating by an error.
Secondly, the analysis text files (part of the algorithm's output') are compared
as they should be identical for a specific reduction method but different 
program processing case (parallel / sequential).

The script can easily be adapted to other non linear algebraic systems.

Basic requirement:
    python file with NLE in the required format (according to vanDerWaals.py)

Output:
    Report with tests results from single case or all cases that is structured
    as foloows:
    tolerance:      used to check if resulting boxes are the same if different
                    program processing is used
    case:           reduction method + program processing (sequential, parallel)
    status:         Ok := case ran as expected, Failed := Bug in the algorithm
    cpu:            CPU time for case run
    comparison:     comparison with analysis results from sequential processing
                    (sequential case := -) Ok:= hypercubic lengths and number of boxes
                    are identical, Failed := differences in number of boxes or 
                    hypercubic length of boxes to reference case (sequential case)
    
@author: sassibub
"""

import os
import mpmath

"""
*******************************************************************************
User settings
*******************************************************************************
"""

def main():
    dict_file = {'modul_name': "flash",   # requires python module with this name
                 'modus': "single",                # "all" := testing all cases, "single" :=testing a specific case
                 'tol': 1e-10,                   # for hypercubic length comparison
                 } 
    
    dict_options_all = {'bc_method': ["None", "b_normal", "b_tight"],
                    'newton_method': ["None", "newton", "detNewton", "newton3P"],
                    'parallel_branches': [False, True],
                    'parallel_variables': [False, True],
		    'hc_method': ["None", "HC4"],
		    'Affine_arithmetic': [False, True],
		    'InverseOrHybrid': ["None", "Hybrid", "both"],
		    'cut_Box': [False, True],
		    'split_Box': ["TearVar", "Largester", "forecastSplit"]
}

    dict_options_single = {'bc_method': ["None"],
                    'newton_method': ["None"],
                    'parallel_branches': [False],
                    'parallel_variables': [False],
        		    'hc_method': ["HC4"],
        		    'Affine_arithmetic': [True],
        		    'InverseOrHybrid': ["both"],
        		    'cut_Box': [True],
        		    'split_Box': ["forecastSplit"]
}

    if dict_file['modus'] == "single": testMethods(dict_file, dict_options_single)
    if dict_file['modus'] == "all": testMethods(dict_file, dict_options_all)

"""
*******************************************************************************
Further methods
*******************************************************************************
"""

def testMethods(dict_file, dict_options):
    """
    Args:
        :dict_file:        dictionary with options concerning test output files
        :dict_options:     dictionary with test run options (different cases)
        
    """
    
    results ={}
        
    for bc_method in dict_options['bc_method']:
        for newton_method in dict_options['newton_method']:
            for hc_method in dict_options['hc_method']:
            
                if bc_method == "None" and newton_method == "None" and hc_method == "None": continue
                dirName = createMethodDirectory(bc_method, newton_method, hc_method)                    
            
                for par_var in dict_options["parallel_variables"]: 
                    for par_branch in dict_options["parallel_branches"]:
                        for aa in dict_options["Affine_arithmetic"]:
                            for invNewton in dict_options["InverseOrHybrid"]:
                                for cut in dict_options["cut_Box"]: 
                                    for split in dict_options["split_Box"]:
                                                               filePath = dirName + "/" + dict_file['modul_name']
                                                               caseName = getResultFileName(filePath, par_var, par_branch, aa, invNewton, cut, split)
                                                               refCaseName = getResultFileName(filePath, False,  False, aa, invNewton, cut, split)
                                                               try: 
                                                                   testOneCase(dict_file['modul_name'], [caseName, bc_method, newton_method, 
                                                                                                         par_var, par_branch, hc_method, aa, invNewton, cut, split])
                                                                        
                                                                   results[caseName]=["Ok"]
                                                                   results = compareOutput(caseName, refCaseName, 
                                                                                                results, dict_file["tol"])
                                                               except: 
                                                                   results[caseName]=["Failed", "Failed", "Failed"]
    
                                                               print(caseName + " finished.")
                    
    writeReport(dict_file, results)


def createMethodDirectory(bc_method, newton_method, hc_method):
    """ creates directory for test case.
    
    Args:
        :bc_method:         string with box consistency method or "None"
        :newton_method:     string with interval newon method or "None"
        :newton_method:     string with hull consistency method or "None"  
    
    Return:
        :path:              string with relative case directory path

    """
    dirName = ""

    if bc_method != "None": dirName = bc_method
    if newton_method != "None" and dirName != "": dirName += "_" + newton_method
    if newton_method != "None" and dirName == "": dirName = newton_method
    if hc_method != "None" and dirName != "": dirName += "_" + hc_method
    if hc_method != "None" and dirName == "": dirName = hc_method
    
    return createDirectory([dirName])
    
    
def createDirectory(args):
    """ creates directory from args (each element of args is the name of a further
    nested subdirectory, e.g.: [a,b,c] -> a/b/c)
    
    Args:
        :args:      list with nested directory names as strings
    
    Return:
        :path:      string with relative path to last element of args
    
    """
    
    path = args[0]
    
    for i in range(1,len(args)):
        path = path + '/' + args[i]
           
    if not os.path.exists(path):
        os.makedirs(path)    
    return path


def getResultFileName(fileName, par_var, par_branch, aa, invNewton, cut, split):
    """ creates result file name for investigated case.
    
    Args:
        :fileName:      string with general file name
        :par_var:       boolean True/False variable reduction parallelization
        :par_branch:    boolean True/False branch process parallelization
	:aa: 		booliean True/False affine arithmetic
	:invNewton: 	string how the inverse of newton is build
	:cut:		boolean cut box edges on/off
	:split: 	string method to split the box if consistency is reached

    Return:
        case specific result file name as string

    """
    
    if par_var and par_branch: fileName += "_par_vb"
    if par_var and not par_branch: fileName += "_par_v"
    if not par_var and par_branch: fileName += "_par_b"
    #if not par_var and not par_branch: fileName    

    if aa: fileName += "_aa"
    if invNewton == "Hybrid": fileName += "_hybrid"
    if invNewton == "both": fileName += "_both"
    if split == "TearVar": fileName += "_tearVar"  
    if split == "Largester": fileName += "_largester"
    if split == "forecastSplit": fileName += "_forecast"
    if cut: fileName += "_cut"
    return fileName  


def testOneCase(modulName, args): 
    """ tests one case (reduction method + parallelization (yes/no))
    
    Args:
        :modulName:   string with module name
        :args:        list with input parameters for module call: 
                      [caseName, bc_method, newton_method, par_var, par_branch]

    """
    
    modul_call = getModulCall(modulName, args)
    os.system(modul_call) 


def getModulCall(modulName, args):
    """ creates string for module call
    
    Args:
        :modulName:   string with module name
        :args:        list with input parameters for module call: 
                      [caseName, bc_method, newton_method, par_var, par_branch]

    """
    
    modul_call = "python " + modulName + ".py"
    for arg in args:
        modul_call += " "+ str(arg)
    return modul_call


def compareOutput(caseName, refCaseName, results, tol):
    """ compares results of investigated case with reference case (generally the
    sequentially processed box reduction). Only applicable for successfully 
    terminating cases.
    
    Args:
        :caseName:      string with current case
        :refCaseName:   string with current reference case name (different for each method)
        :resutls:       dictionary with test results of each case. 
                        Key := case, Value := [Status]
    
    Return:
        :results:       Updated dictionary results
                        Key := case, Value := [Status, CPU, Comparison with Ref.]

    """
    
    efficiencyFile = caseName + "_efficiency.txt"
    cpu = open(efficiencyFile).readlines()[0].split()[5]
    results[caseName].append(cpu)  
    
    if refCaseName == caseName:
        results[caseName].append("-")
        return results
    
    analysisFile = caseName + "_analysis.txt"
    ref_analysisFile = refCaseName + "_analysis.txt"     
        
    hypercubicLengths = open(analysisFile).readlines()[7].split()
    hypercubicLengths.pop(0)
    ref_hypercubicLengths = open(ref_analysisFile).readlines()[7].split()
    ref_hypercubicLengths.pop(0)
    
    
    results = compareHypercubicLenghts(hypercubicLengths, ref_hypercubicLengths,
                                       caseName, results, tol)
    return results


def compareHypercubicLenghts(case_values, ref_values, caseName, results, tol):
    """ compares hypercubic lengths of current case to those of the 
    reference case by checking their value on almost equality.
    
    Args:
        :case_values:   list with current case hypercubic length values
        :ref_values:    list with reference case hypercubic length values
        :caseName:      string with current case name
        :results:       dictionary with test resutls
        :tol:           float value with tolerance for almost equality check
        
    Return:
        :results:       updated dictionary with results from hypercubic length
                        comparison
         
    """
    
    if not len(case_values) == len(ref_values):
        results[caseName].append("Failed (No. of boxes)")
        
    else:
        curBoxInRefBox = []
        
        for cur_val in case_values:
            curBoxInRefBox.append(checkIfBoxInReferenceCase(cur_val, ref_values, tol))
    
    if all(curBoxInRefBox): results[caseName].append("Ok")
    else: results[caseName].append("Failed (Box sizes)")
    
    return results


def checkIfBoxInReferenceCase(hypercubicLength, ref_hypercubicLengths, tol):
    """ checks if certain box (identified by its hypercubic length) of the 
    current case matches a hypercubic length from the reference case in the 
    required tolerance (tol). It removes the hypercubic length of the reference
    to avoid matching hypercubic length from the current case to the same 
    reference hypercubic length.
    
    Args:
        :hypercubicLength:      float value with current boxe's hypercubic length
        :ref_hypercubicLengths: list with reference case hypercubic lengths
        :tol:                   float value with tolerance for almost equality check

    Return:
        boolean that is True for almost equality and False otherwise
        
    """
    
    for ref_val in ref_hypercubicLengths:
        if mpmath.almosteq(mpmath.mpf(hypercubicLength), mpmath.mpf(ref_val), tol, tol):
            ref_hypercubicLengths.remove(ref_val)
            return True
    return False


def writeReport(dict_file, results):
    """ generates report from test results
    
    Args:
        :dict_file:         dictionary with file option settings
        :results:           dictionary with test results
    
    """
    modulName = dict_file['modul_name']
    
    if dict_file["modus"] == "all": report = open(modulName + "_all_test_report.txt","w+")
    if dict_file["modus"] == "single": report = open(modulName + "_single_test_report.txt","w+")
    
    report.write("Test report from module: " + modulName + "\n\n")
    report.write("Tolerance:\t %s\n\n" % (str(dict_file["tol"])))
    report.write("Case\t Status\t CPU in s\t Comparison with seq. process\n")
    
    for case in results:
        report.write("%s\t %s\t %s\t %s\n" % (case, results[case][0], 
                                       str(results[case][1]),
                                       results[case][2]))
    report.close()
    

"""
*******************************************************************************
Call of main method
*******************************************************************************
"""
# Invoke main method:
if __name__ == "__main__": main() 
