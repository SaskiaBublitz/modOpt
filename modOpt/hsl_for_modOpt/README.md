This directory contains all signature files to generate dynamic libraries executable in python from HSL fortran routines. To get them working in Python you need to:

1. Get all associated fortran files from the HSL library TODOlink:

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
