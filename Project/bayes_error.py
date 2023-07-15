import gaussian_classifiers as gc
import SVM_classifiers as svm
import logistic_regression_classifiers as lr
import GMM as gmm
import validation as val
import dimensionality_reduction as dr
import data_utils as du
import numpy as np
import SVM_classifiers as svm 

#IMPORT DATASET
labels, features = du.load("..\PROJECTS\Language_detection\Train.txt")

# #BAYES
# print("BAYES")
# dataPCA5 = dr.PCA(features, 5)
# actualDCFlist = []
# minDCFlist = []
# for piTilde in [0.1, 0.5]:
#     GC = gc.naive_multivariate_cl([1-piTilde, piTilde])
#     actualDCF, minDCF = val.k_fold(GC, dataPCA5, labels, 5, (piTilde, 1,1))
#     print(f"actualDCF{actualDCF}, minDCF{minDCF}")
#     actualDCFlist.append(actualDCF)
#     minDCFlist.append(minDCF)

# actualDCFlist = np.array(actualDCFlist)
# minDCFlist = np.array(minDCFlist)
# print("Cprim:", actualDCFlist.mean(axis=0))
# print("min Cprim:", minDCFlist.mean(axis=0))

# #QLOGREG
# print("QLOGREG")
# QuadraticFeatures = du.features_expansion(features)
# actualDCFlist = []
# minDCFlist = []
# for piTilde in [0.1, 0.5]:
#     LR = lr.logReg(126, 0.17, "balanced")
#     actualDCF, minDCF = val.k_fold(LR, QuadraticFeatures, labels, 5, (piTilde, 1,1))
#     print(f"actualDCF{actualDCF}, minDCF{minDCF}")
#     actualDCFlist.append(actualDCF)
#     minDCFlist.append(minDCF)

# actualDCFlist = np.array(actualDCFlist)
# minDCFlist = np.array(minDCFlist)
# print("Cprim:", actualDCFlist.mean(axis=0))
# print("min Cprim:", minDCFlist.mean(axis=0))

# #QLOGREG
# print("GMM")
# actualDCFlist = []
# minDCFlist = []
# for piTilde in [0.1, 0.5]:
#     GMMObj = gmm.GMM(2,32,"MVG", "tied")
#     actualDCF, minDCF = val.k_fold(GMMObj, features, labels, 5, (piTilde, 1,1))
#     print(f"actualDCF{actualDCF}, minDCF{minDCF}")
#     actualDCFlist.append(actualDCF)
#     minDCFlist.append(minDCF)

# actualDCFlist = np.array(actualDCFlist)
# minDCFlist = np.array(minDCFlist)
# print("Cprim:", actualDCFlist.mean(axis=0))
# print("min Cprim:", minDCFlist.mean(axis=0))

pi = 0.1
GC = gc.naive_multivariate_cl([1-pi, pi])
dataPCA5 = dr.PCA(features, 5)
print(val.k_fold_bayes_plot(GC, features, labels, 5, (pi,1,1), "Naive PCA5 plot"))