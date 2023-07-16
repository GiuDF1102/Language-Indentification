import numpy as np
import matplotlib.pyplot as plt
import data_utils as du
import gaussian_classifiers as gc
from sklearn.utils import shuffle
import dimensionality_reduction as dr
import logistic_regression_classifiers as lrc
import math_utils as mu
import SVM_classifiers as svm
import data_visualization as dv
import validation2 as val

L, D = du.load("..\PROJECTS\Language_detection\Train.txt")
DPCA = dr.PCA(D, 5)
print("RBF, True, gamma=0.01, C=0.1, K=0.01, piT=0.2")
gc = gc.naive_multivariate_cl([0.9, 0.1])
actualDCF, minDCF, scores = val.k_fold_bayes_plot(gc, DPCA, L, 5, (0.1, 1, 1), "GC")
print(f"Not Calibrated, aDCF: {actualDCF}, minDCF: {minDCF}")
print("scores: ",scores)

print("RBF, True, gamma=0.01, C=0.1, K=0.01, piT=0.2")
svmc = svm.SVM("RBF", True, gamma=0.01, C=0.1, K=0.01, piT=0.2)
actualDCF, minDCF, scores = val.k_fold_bayes_plot(svmc, DPCA, L, 5, (0.1, 1, 1), "SVMC")
print(f"Not Calibrated, aDCF: {actualDCF}, minDCF: {minDCF}")
print("scores: ",scores)
