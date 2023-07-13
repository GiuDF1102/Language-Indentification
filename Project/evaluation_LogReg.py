import validation as val
import data_utils as du
import dimensionality_reduction as dr
import logistic_regression_classifiers as lrc
import math_utils as mu
import numpy as np
import data_visualization as dv

#LOADING DATASET
labels, features = du.load("..\PROJECTS\Language_detection\Train.txt")
labels_test, features_test = du.load("..\PROJECTS\Language_detection\Test.txt")

#FEATURES EXPANSION
featuresTrainQuadratic = du.features_expansion(features)
featuresTrainQuadraticZNorm = mu.z_score(featuresTrainQuadratic)
featuresZNorm = mu.z_score(features)
lambdas = np.logspace(-3, 5, num=30)

# #LINEAR
CprimLogReg = np.zeros((2, len(lambdas)))

# ##LINEAR NOT NORMALIZED
# minDCFList = np.zeros((2, len(lambdas)))
# for index, pi in enumerate([0.1,0.5]):
#     for lIndex, l in enumerate(lambdas):
#         logRegObj = lrc.logReg(l, pi, True) #Balanced
#         _, minDCF = val.k_fold(logRegObj, features, labels, 5, (pi, 1, 1))
#         print("Linear LogReg Balanced Not Normalized, minDCF with pi {} and lambda {} is {}".format(pi, l, minDCF))
#         minDCFList[index, lIndex] = minDCF
# CprimLogReg[0] = minDCFList.mean(axis=0)

# ##LINEAR NORMALIZED
# minDCFList = np.zeros((2, len(lambdas)))
# for index, pi in enumerate([0.1,0.5]):
#     for lIndex, l in enumerate(lambdas):
#         logRegObj = lrc.logReg(l, pi, True)
#         _, minDCF = val.k_fold(logRegObj, featuresZNorm, labels, 5, (pi, 1, 1))
#         print("Linear LogReg Balanced Normalized, minDCF with pi {} and lambda {} is {}".format(pi, l, minDCF))
#         minDCFList[index, lIndex] = minDCF
# CprimLogReg[1] = minDCFList.mean(axis=0)

# dv.plotCPrim(lambdas, CprimLogReg, ["Linear Log-Reg Balanced", "Linear Log-Reg Balanced z-norm"] , "λ", "LinearLogReg_LinearLogRegNorm")


# #LINEAR PCA NOT NORMALIZED 
# CprimLogReg = np.zeros((3, len(lambdas)))

# ##LINEAR PCA NOT NORMALIZED
# for PCAIndex, nPCA in enumerate([5,4]):
#     minDCFList = np.zeros((2, len(lambdas)))
#     dataPCA = dr.PCA(features, nPCA)
#     for index, pi in enumerate([0.1,0.5]):
#         for lIndex, l in enumerate(lambdas):
#             logRegObj = lrc.logReg(l, pi, True)
#             _, minDCF = val.k_fold(logRegObj, dataPCA, labels, 5, (pi, 1, 1))
#             print("Linear LogReg Balanced Not Normalized, minDCF with pi {}, PCA {} and lambda {} is {}".format(pi, nPCA, l, minDCF))
#             minDCFList[index, lIndex] = minDCF
#     CprimLogReg[PCAIndex] = minDCFList.mean(axis=0)


# ##LINEAR NOT NORMALIZED
# minDCFList = np.zeros((2, len(lambdas)))
# for index, pi in enumerate([0.1,0.5]):
#     for lIndex, l in enumerate(lambdas):
#         logRegObj = lrc.logReg(l, pi, True)
#         _, minDCF = val.k_fold(logRegObj, features, labels, 5, (pi, 1, 1))
#         print("Linear LogReg Balanced Not Normalized, minDCF with pi {} no PCA and lambda {} is {}".format(pi, l, minDCF))
#         minDCFList[index, lIndex] = minDCF
# CprimLogReg[2] = minDCFList.mean(axis=0)

# dv.plotCPrim(lambdas, CprimLogReg, ["Linear Log-Reg Balanced PCA 5", "Linear Log-Reg Balanced PCA 4", "Linear Log-Reg Balanced no PCA"] , "λ", "LinearLogRegBalancedPCAs")


# #QUADRATIC
# CprimLogReg = np.zeros((2, len(lambdas)))

# ##QUADRATIC NOT NORMALIZED
# minDCFList = np.zeros((2, len(lambdas)))
# for index, pi in enumerate([0.1,0.5]):
#     for lIndex, l in enumerate(lambdas):
#         logRegObj = lrc.logReg(l, pi, True) #Balanced
#         _, minDCF = val.k_fold(logRegObj, featuresTrainQuadratic, labels, 5, (pi, 1, 1))
#         print("Quadratic LogReg Balanced Not Normalized, minDCF with pi {} and lambda {} is {}".format(pi, l, minDCF))
#         minDCFList[index, lIndex] = minDCF
# CprimLogReg[0] = minDCFList.mean(axis=0)

# ##QUADRATIC NORMALIZED
# minDCFList = np.zeros((2, len(lambdas)))
# for index, pi in enumerate([0.1,0.5]):
#     for lIndex, l in enumerate(lambdas):
#         logRegObj = lrc.logReg(l, pi, True)
#         _, minDCF = val.k_fold(logRegObj, featuresTrainQuadraticZNorm, labels, 5, (pi, 1, 1))
#         print("Quadratic Z-Norm, minDCF with pi {} and lambda {} is {}".format(pi, l, minDCF))
#         minDCFList[index, lIndex] = minDCF
# CprimLogReg[1] = minDCFList.mean(axis=0)

# dv.plotCPrim(lambdas, CprimLogReg, ["Quadratic Log-Reg Balanced", "Quadratic Log-Reg Balanced z-norm"] , "λ", "QuadraticLogReg_QuadraticLogRegNorm")


# #QUADRATIC PCA NOT NORMALIZED 
# CprimLogReg = np.zeros((3, len(lambdas)))

# ##QUADRATIC PCA NOT NORMALIZED
# for PCAIndex, nPCA in enumerate([5,4]):
#     minDCFList = np.zeros((2, len(lambdas)))
#     dataPCA = dr.PCA(features, nPCA)
#     expandedDataPCA = du.features_expansion(dataPCA)
#     for index, pi in enumerate([0.1,0.5]):
#         for lIndex, l in enumerate(lambdas):
#             logRegObj = lrc.logReg(l, pi, True)
#             _, minDCF = val.k_fold(logRegObj, expandedDataPCA, labels, 5, (pi, 1, 1))
#             print("Quadratic LogReg Balanced Not Normalized, minDCF with pi {}, PCA {} and lambda {} is {}".format(pi, nPCA, l, minDCF))
#             minDCFList[index, lIndex] = minDCF
#     CprimLogReg[PCAIndex] = minDCFList.mean(axis=0)


# ##QUADRATIC NOT NORMALIZED
# minDCFList = np.zeros((2, len(lambdas)))
# for index, pi in enumerate([0.1,0.5]):
#     for lIndex, l in enumerate(lambdas):
#         logRegObj = lrc.logReg(l, pi, True)
#         _, minDCF = val.k_fold(logRegObj, featuresTrainQuadratic, labels, 5, (pi, 1, 1))
#         print("Quadratic LogReg Balanced Not Normalized, minDCF with pi {} no PCA and lambda {} is {}".format(pi, l, minDCF))
#         minDCFList[index, lIndex] = minDCF
# CprimLogReg[2] = minDCFList.mean(axis=0)

# dv.plotCPrim(lambdas, CprimLogReg, ["Quadratic Log-Reg Balanced PCA 5", "Quadratic Log-Reg Balanced PCA 4", "Quadratic Log-Reg Balanced no PCA"] , "λ", "QuadraticLogRegBalancedPCAs")

#QUADRATIC BEST PCA NOT NORMALIZED 
CprimLogReg = np.zeros((2, len(lambdas)))

##QUADRATIC BEST PCA NOT NORMALIZED
for PCAIndex, nPCA in enumerate([5]):
    minDCFList = np.zeros((2, len(lambdas)))
    dataPCA = dr.PCA(features, nPCA)
    expandedDataPCA = du.features_expansion(dataPCA)
    for index, pi in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, pi, True)
            _, minDCF = val.k_fold(logRegObj, expandedDataPCA, labels, 5, (pi, 1, 1))
            print("Quadratic LogReg Balanced Not Normalized, minDCF with pi {}, PCA {} and lambda {} is {}".format(pi, nPCA, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[PCAIndex] = minDCFList.mean(axis=0)


##QUADRATIC BEST PCA NORMALIZED
for PCAIndex, nPCA in enumerate([5]):
    minDCFList = np.zeros((2, len(lambdas)))
    dataPCA = dr.PCA(features, nPCA)
    expandedDataPCA = du.features_expansion(dataPCA)
    zNormExpanded = mu.z_score(expandedDataPCA)
    for index, pi in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, pi, True)
            _, minDCF = val.k_fold(logRegObj, zNormExpanded, labels, 5, (pi, 1, 1))
            print("Quadratic LogReg Balanced Normalized, minDCF with pi {}, PCA {} and lambda {} is {}".format(pi, nPCA, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[1] = minDCFList.mean(axis=0)


dv.plotCPrim(lambdas, CprimLogReg, ["Quadratic Log-Reg Balanced PCA 5", "Quadratic Log-Reg Balanced PCA 5 z-norm"] , "λ", "QuadraticLogRegBalancedPCAsNorm")

