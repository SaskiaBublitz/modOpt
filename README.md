# Introduction

modOpt is a python package for solving nonlinear algebraic systems

Following subpackages are included:
* Constraints (Interval and affine arithmetic based variable bounds reduction)
* Decomposition (Subsystem identification by block matrix decomposition methods)
* Initialization (Sampling methods for multi start procedures)
* Scaling (Row/colum scaling methods for linear systems, i.e. equation scaling and variable unit transformation)
* Solver (State-of-the-art numerical solvers such as ipopt, scipy's SLSQP, matlab's fsolve method and self-implemented Newton)

# Installation Guide for Users

	1. Prerequisites
	* Python 3.7 with pip installed 

	2. Open the terminal and type in following command:	
	``` pip install git+https://git.tu-berlin.de/dbta/simulation/modOpt@modOpt_v_3.0 ```
	
	3. Open python in terminal and check if package is imported:
	
	> python
	> import modOpt
	
# Installation Guide for Developers

	1. Prerequisites for Windows
		* An environment that understands git-commands must be installed such as [gitBash](https://gitforwindows.org/)
	2. In git change to the page with the [overview of the current branches](https://git.tu-berlin.de/dbta/simulation/modOpt/-/branches)
	3. Create a new branch (green button in the right top corner) from the current master-branch. Please, follow this project's naming policy for your branch name which is:
	> dev_YourName
	4. Open a terminal (Linux/Mac) or gitBash (Windows) and navigate to the location where the repository shall be stored at:
	> cd <path_to_location>
	5. Clone git repository:
	> git clone https://git.tu-berlin.de/dbta/simulation/modOpt
	6. Change to modOpt directory:
	> cd modOpt
	7. Change to your branch by:
	
	> git checkout dev_YourName
	
	The remote branch should then automatically be tracked.
	8. For Windows-users: Open a new python-understanding terminal and change to the modOpt directory:
	
	> cd <path_to_location>/modOpt
	
	9. Install python package by:
	
	> pip install .
		* *Notes*: 
			* With the additional option -e changes in the modOpt package are automatically updated to your installations
			* If you have no installation rights use the option --user 
	
	10. Check in terminal if package is imported correctly in python:
	
	> python
	>
	> import modOpt

# Commit and Push Starter Guide

Before changes can be pushed to the remote project you need to commit them. It is recommended to update changes regulary to prevent merge conflicts later on.
Whenever a certain task is full filed and tests have been successfully executed it is recommended to commit the changes. This is done the follwing way:

	1. Open the terminal, move to the modOpt package and check that you are in your branch
	2. Use 
	> git status 
	to get an overview if there are changes to commit.
	3. If the status shows untracked files, you have created new files they first need to be added to git, by:
	> git add .
	4. If all files are tracked, you can create the commit message by:
	> git commit -a
	A text editor will open where you can type in your message.
	5. By *Strg+x* you close the finished commit message 
	6. Check the status again if the commit has been successfully saved.
	7. Push your local changes to the repository by:
	> git push

# Commit Style Guide

I use the following keywords infront of the commit message to group the changes:
* Add: New functionality/files are added to project code
* Fix: A bug has been fixed
* Style: Some changes on the documentation level or refurbishing existing methods for reusability

Here for a example a commit with multiple changes:

> Add: New file <xyz> for the usage of method <abc>
> Add: A new subpackage <stu> to modOpt for a very nice feature
> Fix: In the existing method <cde> fixed subtraction error
> Style: Add function descripton to method <cde>
> Style: Rewrote method <fgh> in order to use it in methd <hij>


