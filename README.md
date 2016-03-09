#implementations of Eigenface, Fisherface, SRC and SVM 
---
###experimental setups are in expr.py

###required python packages
* numpy
* scipy
* matplotlib
* scikit-learn

###libsvm needs to be recompiled before use:
```shell
cd libsvm/
rm libsvm/svm-train 
   libsvm/svm-predict 
   libsvm/svm-scale 
   libsvm/svm.o 
   libsvm/libsvm.so.2 
make
cd python/
make
```
###spgl1 solver
* compile a c-version by running spgsetup.m for faster computation"
