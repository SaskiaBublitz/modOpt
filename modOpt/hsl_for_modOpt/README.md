
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

Tip: If you get an error such as: No module named 'distutils.msvccompiler' please make sure that you use an older Version of setuptools. You can instal it by: pip install "setuptools<65.0"


The shell script creates associated .so or .pyd files and move them into the directories in modOpt where they are invoked.

## Debugging
* After each installation of the .pyd or .so file you get informed about its success in the terminal by the text output of for example the routine MC33AD on Linux:
		
		MC33AD.so has been successfully installed.

If you cannot find this statement in your console output the reason might be the missing fortran compiler. On Windows you shold use mingw32 , please go to the sub section: Missing fortran compiler on Windows to follow the install instructions.

### Missing fortran compiler on Windows
You need to install Gcc-compiler from thin mingw.org project for Windows manually by:
1. Download a specific 64-bit build of mingw-w64 7.2.0 from sourceforge: https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/7.2.0/threads-posix/seh/, specifically x86_64-7.2.0-release-posix-seh-rt_v5-rev1.7z.
2. Install new version through the .exe-file and store files to (C:\mingw64) or another directory.
3. Add C:\mingw64\bin to System variable Path.
