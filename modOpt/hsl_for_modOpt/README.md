# Instructions to generate dynamic link libraries in Python from fortran routines

## Prerequisities
* Python package f2py needs to be installed
* Having a fortran compiler installed

## Generation of .pyd-files (Windows) and .so-files (Linux/iOS) for modOpt

This directory contains all signature files to generate the dynamic link libraries
1. Get the following associated fortran files from the HSL library:

MC33AD.f
mc29d.f
mc77d.f

2. Put all .f-Files into this directory 
3. Open a terminal and run
3. a) For Linux and IOS:
./make_so_files.sh
3. b) For Windows:  
make_pyd_files.bat

The shell script creates associated .so or .pyd files and move them into the directories in modOpt where they are invoked.
