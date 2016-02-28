import mthds as md
import numpy as np
import my_func as mf
import matplotlib.pyplot as plt

# initialization
expr_time = 10
tr = 29
te = 29
ac = 0

# fisherface
# expr1, show fisherface
if 0:
    k = 100
    sam_tr, sam_te, c = mf.predat(tr, te)
    a, co, w = md.lda(k, sam_tr, sam_te, c)
    mf.showface(w)

# expr2, influence of k
if 1:
    for k in np.array([35]):
        for i in range(1, expr_time+1):
            sam_tr, sam_te, c = mf.predat(tr, te)
            a, co, w = md.lda(k, sam_tr, sam_te, c)
            ac += a
    accu = ac/expr_time
    print "accuracy is %0.2f%%" % (accu*100)

# eigenface
# expr1, influence of k
if 0:
    for k in np.array([100]):
        for i in range(1, expr_time+1):
            sam_tr, sam_te, c = mf.predat(tr, te)
            a, co, w = md.pca(k, sam_tr, sam_te, c)
            ac += a
    accu = ac/expr_time
    # mf.showface(w[:, 3:-1])
    print "accuracy is %0.2f%%" % (accu*100)
