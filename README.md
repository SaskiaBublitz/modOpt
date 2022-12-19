

# Introduction

modOpt is a python package for solving nonlinear algebraic systems

Following subpackages are included:
* Constraints (Interval and affine arithmetic based variable bounds reduction)
* Decomposition (Subsystem identification by block matrix decomposition methods)
* Initialization (Sampling methods for multi start procedures)
* Scaling (Row/colum scaling methods for linear systems, i.e. equation scaling and variable unit transformation)
* Solver (Connection to state-of-the-art numerical solvers such as ipopt, scipy's SLSQP, matlab's fsolve method and self-implemented Newton)

# Installation Guide for Users

##  Prerequisites
* Python 3.9 with pip installed 


## Instructions for Installation of
### ModOpt

1. Open the terminal and type in following command (replace branch_name by the branch name of interest):

        pip install git+https://git.tu-berlin.de/dbta/simulation/modOpt@branch_name

2. Open python in terminal and check if package is imported:

        python
        import modOpt

### Dynamic linked libraries from HSL
You will probably get an "*ImportError: DLL load failed"* when you try to import modOpt because the provided **.pyd** and **.so** files might not
be compatible to your operating system and python version. In this case:
1. Get acess and download the fortran routines: MC29, MC33 and MC77 from the HSL website https://www.hsl.rl.ac.uk/
2. Copy the mc29d.f, MC33AD.f, mc77d.f from the routine packages to the hsl_for_modOpt directory in your modOpt installation package, for example in
 
	  <path-to-your-envs>`/envs/your_env/libs/site-packages/modOpt/hsl_for_modOpt`  

Note: you also get the path to the modOpt package from the ImportError

3. Open the README.md in the hsl_for_modOpt directory and follow the installation instructions to create **.pyd**-files for Windows and **.so**-files for Linux and iOS.
They are automatically moved to the required position in modOpt
4. Check if ImportError is fixed by repeating step 2 from the "Installation of modOpt section"
 
## Create scripts for your MOSAICmodeling evaluations  
1. Open, initialize and save your model as an evaluation in MOSAICmodeling > Simulation
2. Change to MOSAICmodeling > Simulation > Evaluation > Code Generation and select *User-defined LangSepc*
3. Open desired UDLS for Code Generation
4. Specify Code Generator and Solver Properties
5. Click on *Generate Code*
6. Go to *View Code*
7. *Export Code* or copy+paste the code into your python running environment

Current **UDLS-ID's** for public use:
| ID   |      Name      | Description |
|----------|:-------------:|:------:|
| 167855 | UDLS_modOpt_constraints_V2.9.lsp  | Box Redcution + Sampling + Solver |

## Tests to check modOpt is working as expected (MOSAICmodeling is not required)
To check wether modOpt is working correctly, there is a test package included with some examples from chemical engineering. Different settings in the bxrd_options dictionaries are sequentially checked. If no error has occured in a test run an "OK" is written into the *status* cell of the corresponding test case row of the tests' output file. The latter ends on "*test_all.txt". If an error occured the *status* of the associated test run is "Failed". This is exemplary explainded for the VanDerWaals.py example but works the same way for the other larger examples in the tests/constraints directory.

1. In the modOpt package go to 

	`modOpt/tests/constraints/VanDerWaals`

2. Run the python file main.py

        python main.py

3. Open the output textfile ending with test_all.text and check the each test run's status

You can also use the .py files of the test examples as a template for writing your own scripts apart from MOSAICmodeling. Simply replace the system arguments by an argument of the required type such as string, integer, boolean, float. Discrete settings from a set of valid options for example strings with numerical solver names you find in the comments behind the assignment starting with #

# Installation Guide for Developers

## Prerequisites for Windows
* An environment that understands git-commands must be installed such as [gitBash](https://gitforwindows.org/)

## Instructions
1. In git change to the page with the [overview of the current branches](https://git.tu-berlin.de/dbta/simulation/modOpt/-/branches)
2. Create a new branch (green button in the right top corner) from the current master-branch. Please, follow this project's naming policy for your branch name which is:

        dev_YourName
        
3. Open a terminal (Linux/Mac) or gitBash (Windows) and navigate to the location where the repository shall be stored at:

        cd <path_to_location>
        
4. Clone git repository:

        git clone https://git.tu-berlin.de/dbta/simulation/modOpt
        
5. Change to modOpt directory:

        cd modOpt
        
6. Change to your branch by:
	
        git checkout dev_YourName
	
    The remote branch should then automatically be tracked.

7. For Windows-users: Open a new python-understanding terminal and change to the modOpt directory:
	
        cd <path_to_location>/modOpt
	
8. Install python package by:
	
        pip install .

	*Notes*:
	* With the additional option -e changes in the modOpt package are automatically updated to your installation
	* If you have no installation rights use the option --user 
	
9. Check in terminal if package is imported correctly in python:

        python 
        import modOpt

## Commit and Push Starter Guide

Before changes can be pushed to the remote project you need to commit them. It is recommended to update changes regulary to prevent merge conflicts later on. Whenever a certain task is fulfilled and tests have been successfully executed it is recommended to commit the changes. This is done the follwing way:

1. Open the terminal, move to the modOpt package and check that you are in your branch
2. Use 
        
        git status 

	to get an overview if there are changes to commit.
3. If the status shows untracked files, you have created new files they first need to be added to git, by:

        git add .

4. If all files are tracked, you can create the commit message by:
        
        git commit -a

	A text editor will open where you can type in your message.
5. By *Strg+x* you close the finished commit message 
6. Check the status again if the commit has been successfully saved.
7. Push your local changes to the repository by:

        git push

## Commit Style Guide

I use the following keywords infront of the commit message to group the changes:
* Add: New functionality/files are added to project code
* Fix: A bug has been fixed
* Style: Some changes on the documentation level or refurbishing existing methods for reusability

Here for a example a commit with multiple changes:

    Add: New file <xyz> for the usage of method <abc>
    Add: A new subpackage <stu> to modOpt for a very nice feature
    Fix: In the existing method <cde> fixed subtraction error
    Style: Add function descripton to method <cde>
    Style: Rewrote method <fgh> in order to use it in methd <hij>

# Output of modOpt
The main function of your python script contains the dictionary bxrd_options. Here you must specify name and path, where the program should store the output files:
    
     bxrd_options={"fileName": "your_fileName", 
				   "savePath":  "path_to_your_files"}
    
The program will then generate at least two output files:
    
     1. your_fileName_methodsApplied_1.txt
     2. your_fileName_methodsApplied.npz
    
Below is a legend for all methodsApplied, which are separated by _ :
    
   |Shortcut|Description  |
    |--|--|
    | hc | HC4 contraction |
    | nc | Interval Newton with center as point of expansion |
    | bc | Bnormal applied |
    | ncaf | All equations used in Interval Newton |
    | tb | Tight bounds applied |
    | aff | Affine arithmetic|
    | dm | Dulmage Mendelsohn applied for system decomposition |
    | gap |Priority given to discontinuities for splitting |
    | cba | Cutting all variables |
    | cbtv | Cutting tearing variables |
    | tv | Split tearing variables |
    | lc | Split least changed variables |
    | ftv | Split forecast tearing variables |
    | fcs | Split forecast all variables |
    | rsX | Resolution X |
    | rX | X box redcution steps |
    | pbX | X number of parallel boxes|
    | sb | Sobol sampling|
    | hs | Hammersley sampling|
    | lhc | Latin hypercube sampling|
    | sX | X numer of samples|
    | sminX | X numer of best samples tested in root-finding|
    | nwt | Root-finder newton|
    | casipopt | Root-finder casadi ipopt|
    | slsqp | Root-finder slsqp|
    | tnc | Root-finder TNC solver|
    | fslv | Root-finder fsolve|
    | mfslv | Root-finder matlab fsolve|
    | mfslvscrpt | Root-finder matlab fsolve|
    | mc29 | Scaling method MC29 (HSL) for root-finder newton|
    | mc77 | Scaling method MC77 (HSL) for root-finder newton|
    
If the initial box does not contain any solution, there is an additional text file besides the files just listed:
    
   your_fileName_methodsApplied_error.txt
    
This shows the equation including the variable intervals, in which no solution could be found as well as their initial intervals. This should help the user to quickly find errors in the implementation of the equations or initialization of the variables, which caused the empty box.

