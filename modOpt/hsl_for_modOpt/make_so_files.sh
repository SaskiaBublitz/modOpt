#!/bin/bash

f2py -c MC33AD.pyf MC33AD.f -m MC33AD
f2py -c mc29.pyf mc29d.f -m MC29AD
f2py -c MC77D.pyf mc77d.f -m MC77D

for f in MC33AD*.so; do mv $f ./decomposition/MC33AD.so; done
for f in MC29AD*.so; do mv $f ./scaling/MC29AD.so; done
for f in MC77D*.so; do mv $f ./scaling/MC77D.so; done

