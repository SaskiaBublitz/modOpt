echo on 
f2py -c MC33AD.pyf MC33AD.f -m MC33AD --compiler=mingw32
echo "MC33AD.pyd successfully installed."
f2py -c mc29.pyf mc29d.f -m MC29AD --compiler=mingw32
echo "MC29AD.pyd successfully installed."
f2py -c MC77D.pyf mc77d.f -m MC77D --compiler=mingw32
echo "MC77D.pyd successfully installed."
 
move MC33AD.pyd ..\decomposition
move MC29AD.pyd ..\scaling
move MC77D.pyd ..\scaling
