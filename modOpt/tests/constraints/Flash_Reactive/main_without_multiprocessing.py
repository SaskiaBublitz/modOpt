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
    as follows:
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
import copy
import os
import mpmath
from modOpt.constraints import results
"""
*******************************************************************************
User settings
*******************************************************************************
"""

def main():
    dict_file = {
        'modus': "all",                # "all" := testing all cases, "single" :=testing a specific case     
        }
    dict_options_all = {
                    'fileName': ["Flash_Reactive"],   # requires python module with this name                  
                    'tol': [1e-9],
                    'redStepMax': [20],
                    'resolution': [8],
                    'parallelBoxes': [0],
                    'bcMethod': ["None", "bnormal"],
                    "affineArithmetic": [0],
                    "tightBounds": [0, 1],
                    'newtonMethod': ["None", "newton"],
                    "newtonPoint": ["center"],
                    "preconditioning": ["pivotAll"],
                    'hcMethod': ["None", "HC4"],
                    'splitBox': ["tearVar", "leastChanged", "forecastSplit"],
                    "considerDisconti": [0],
                    'cutBox': ["tear", "all"],
                    "cpuCountBoxes": [2],
                    "decomp": ["DM"],
}

    dict_options_ref = {
                    'fileName': "Flash_Reactive",   # requires python module with this name
                    'tol': 1e-9,
                    'redStepMax': 20,
                    'resolution': 8,
                    'parallelBoxes': 0,
                    'bcMethod': "bnormal",
                    "affineArithmetic": 0,
                    "tightBounds": 1,
                    'newtonMethod': "newton",
                    "newtonPoint": "center",
                    "preconditioning": "pivotAll",
                    'hcMethod': "HC4",
                    'splitBox': "leastChanged",
                    "considerDisconti": 0,
                    'cutBox': "tear",
                    "cpuCountBoxes": 2,
                    "decomp": "DM",
}

    if dict_file['modus'] == "single": testMethods(dict_options_ref, dict_file)
    if dict_file['modus'] == "all": testMethods(dict_options_ref, dict_file,
                                                dict_options_all)

"""
*******************************************************************************
Further methods
*******************************************************************************
"""

def testMethods(dict_options_ref, dict_file, dict_options=None):
    """ Method to test all or a single case
    
    Args:
        :dict_options_ref: dictionary with test run options (reference case)
        :dict_file:        dictionary with modus (single / all case(s))
        :dict_options:     dictionary with test run options (all tested cases) 
    """
    
    res ={}
    noOfCase = 1
    res, ref_caseName = executeOneCase(dict_options_ref, res)
    print(ref_caseName + " finished.")
    
    if dict_options == None:
        print("This is case number " + str(noOfCase) +" of 1 tested case.") 
    else:
        ref_case = dict_options_ref.copy()
        noOfCases = calc_noOfcases(dict_options)
        print("This is case number " + str(noOfCase) +" of " + str(noOfCases))
        
        for key in dict_options:
            for value in dict_options[key]: 
            
                dict_options_ref[key] = value
                res, caseName = executeOneCase(dict_options_ref, res, ref_caseName)
                
                if caseName != ref_caseName:
                    noOfCase += 1
                    if key == "parallelBoxes":
                        res = compareHypercubicLenghts(res, ref_caseName, caseName,
                                                       dict_options_ref["tol"])
                        
                    dict_options_ref[key] = ref_case[key]
                    print(caseName + " finished.")
                    print("This is case number " + str(noOfCase) +" of " 
                          + str(noOfCases))
            
    writeReport(dict_file, dict_options_ref, res)


def executeOneCase(cur_dict_options, res, ref_caseName=None):
    """ generates case name, test the current case based on settings in 
    cur_dict_options and stores all results in res.
    
    Args:
        :cur_dict_options:      dictionary with current case settings
        :res:                   dictionary with results from testing
        :ref_caseName:          addionally required to test the reference case
                                only once
    
    Returns:
        :res:                   dictionary with updated results from tested case
        :caseName:              string with name of current case
    """
    
    caseName = results.get_file_name(cur_dict_options)
    if ref_caseName and ref_caseName == caseName: return res, caseName

    try: 
        testOneCase(cur_dict_options['fileName'], 
                    [cur_dict_options['fileName'],
                    cur_dict_options['redStepMax'], 
                    cur_dict_options['resolution'], 
                    cur_dict_options['parallelBoxes'],
                    cur_dict_options['bcMethod'],
                    cur_dict_options['affineArithmetic'], 
                    cur_dict_options['tightBounds'],
                    cur_dict_options['hcMethod'],                                                
                    cur_dict_options['newtonMethod'],
                    cur_dict_options['newtonPoint'], 
                    cur_dict_options['preconditioning'],
                    cur_dict_options['splitBox'], 
                    cur_dict_options['considerDisconti'],
                    cur_dict_options['cutBox'],
                    cur_dict_options["cpuCountBoxes"]])
        res = storeOutput(res, caseName)                                                  
    except: 
        res[caseName]=["Failed", "Failed", "Failed"]
    
    return res, caseName


def calc_noOfcases(dict_options):  
    """ precalculate the number of test cases according to the set options in 
    dict_options.
    
    Args:
        :dict_options:     dictionary with test run options (all tested cases)
        
    Returns:
        The total number of test cases
    """
    
    noOfValues = len([item for vallist in list(dict_options.values()) 
                      for item in vallist])
    noOfKeys = len(dict_options.keys())
    return 1 + noOfValues - noOfKeys


def testOneCase(modulName, args): 
    """ tests one case (reduction method + parallelization (yes/no))
    
    Args:
        :modulName:   string with module name
        :args:        list with input parameters for module call: 
                      [caseName, bcMethod, newtonMethod, par_var, par_branch]
    """
    
    modul_call = getModulCall(modulName, args)
    os.system(modul_call) 


def compareHypercubicLenghts(res, refCaseName, caseName, tol):
    """ compares hypercubic lengths of sequential to parallel case
    
    Args:
        :res:           dictionary with different case results
        :refCaseName:   string with reference case's name
        :caseName:      string with current case 
        :tol:           float value with tolerance for almost equality check
        
    Return:
        :res:           updated dictionary with results from hypercubic length
                        comparison        
    """  

    if checkIfBoxInReferenceCase(res[refCaseName][2], 
                                 res[caseName][2], tol):
        res[caseName][2] = "Identical to reference"
    else:
        res[caseName][2] = "Differs from reference"        
   
    return res


def storeOutput(res, caseName):
    """ stores all results from current case, e.g.: successful run, cpu, 
    hypercubic length, in res.
    
    Args:
        :res:           dictionary with different case results
        :caseName:      string with current case
    
    Return:
        :res:           updated dictionary results
    """
    
    res[caseName]=["Ok"] 
    efficiencyFile = "./results/" + caseName + "_nwt_efficiency.txt"
    cpu = open(efficiencyFile).readlines()[0].split()[5]
    res[caseName].append(cpu) 
    analysisFile = "./results/" + caseName + "_nwt_analysis.txt"
    hypercubicLength = open(analysisFile).readlines()[7].split()
    hypercubicLength.pop(0)   
    res[caseName].append(hypercubicLength[0]) 
    return res   


def getModulCall(modulName, args):
    """ creates string for module call
    
    Args:
        :modulName:   string with module name
        :args:        list with input parameters for module call: 
                      [caseName, bcMethod, newtonMethod, par_var, par_branch]

    """
    
    modul_call = "python " + modulName + ".py"
    for arg in args:
        modul_call += " "+ str(arg)
    return modul_call


def checkIfBoxInReferenceCase(hL_1, hL_2, tol):
    """ checks if certain box (identified by its hypercubic length) of the 
    current case matches a hypercubic length from the reference case in the 
    required tolerance (tol). It removes the hypercubic length of the reference
    to avoid matching hypercubic length from the current case to the same 
    reference hypercubic length.
    <e
    Args:
        :hL_1:      float value with boxe's hypercubic length of first case
        :hL_2:      float value with boxe's hypercubic length of second case
        :tol:       float value with tolerance for almost equality check

    Return:
        boolean that is True for almost equality and False otherwise
    """   
    
    if mpmath.almosteq(mpmath.mpf(hL_1), mpmath.mpf(hL_2), tol, tol):
        return True
    return False


def writeReport(dict_file, dict_options_ref, res):
    """ generates report from test results
    
    Args:
        :dict_file:         dictionary with file option settings
        :dict_options:      dictionary with module option settings
        :res:               dictionary with test results
    """
    
    modulName = dict_options_ref['fileName']
    
    if dict_file["modus"] == "all": report = open(modulName + "_all_test_report.txt","w+")
    if dict_file["modus"] == "single": report = open(modulName + "_single_test_report.txt","w+")
    
    report.write("Test report from module: " + modulName + "\n\n")
    report.write("Tolerance:\t %s\n\n" % (str(dict_options_ref["tol"])))
    report.write("Case\t Status\t CPU in s\t Hypercubic Length\n")
    
    for case in res:
        report.write("%s\t %s\t %s\t %s\n" % (case, res[case][0], 
                                       str(res[case][1]),
                                       res[case][2]))
    report.close()
    

"""
*******************************************************************************
Call of main method
*******************************************************************************
"""
# Invoke main method:
if __name__ == "__main__": main() 
