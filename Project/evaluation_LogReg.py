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
#Z-Norm---->PCA------>Expansion
featuresZNorm = mu.z_score(features)
featuresTrainQuadratic = du.features_expansion(features)
featuresTrainQuadraticZNorm = du.features_expansion(featuresZNorm)
featuresPCA5 = dr.PCA(features,5)
featuresPCA4 = dr.PCA(features,4)
featuresPCA5ZNorm = mu.z_score(featuresPCA5)
featuresPCA5Exapanded = du.features_expansion(featuresPCA5)
featuresPCA4Exapanded = du.features_expansion(featuresPCA4)
featuresPCA5ZNormExapanded = du.features_expansion(dr.PCA(mu.z_score(features), 5))

lambdas = np.logspace(-2, 5, num=30)
k = 5

#-------------------------------------
##LINEAR NOT NORM + LINEAR Z-NORM EMPIRICAL
# labels_text = [
#     "Linear LogReg",
#     "Linear LogReg Z-norm"
# ]
# CprimLogReg = np.zeros((2, len(lambdas)))
# piT = 0.17
# with open("LinearLogRegEmpiricalNoNorm_Norm.txt", "w") as f:
#     ##LINEAR NOT NORMALIZED piT = empirical
#     minDCFList = np.zeros((2, len(lambdas)))
#     for index, piTilde in enumerate([0.1,0.5]):
#         for lIndex, l in enumerate(lambdas):
#             logRegObj = lrc.logReg(l, piT, "balanced") 
#             _, minDCF = val.k_fold(logRegObj, features, labels, k, (piTilde, 1,1))
#             print("Linear LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
#             print("Linear LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
#             minDCFList[index, lIndex] = minDCF
#     CprimLogReg[0] = minDCFList.mean(axis=0)
#     print("Cprim list {}".format(CprimLogReg[0]), file=f)
#     print("Cprim list {}".format(CprimLogReg[0]))


#     ##LINEAR NORMALIZED piT = empirical
#     minDCFList = np.zeros((2, len(lambdas)))
#     for index, piTilde in enumerate([0.1,0.5]):
#         for lIndex, l in enumerate(lambdas):
#             logRegObj = lrc.logReg(l, piT, "balanced")
#             _, minDCF = val.k_fold(logRegObj, featuresZNorm, labels, k, (piTilde, 1, 1))
#             print("Linear LogReg Empirical Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
#             print("Linear LogReg Empirical Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
#             minDCFList[index, lIndex] = minDCF
#     CprimLogReg[1] = minDCFList.mean(axis=0)
#     print("Cprim list {}".format(CprimLogReg[1]), file=f)
#     print("Cprim list {}".format(CprimLogReg[1]))

#     dv.plotCPrim(lambdas, CprimLogReg, labels_text, "λ", "LinearLogRegEmpiricalNoNorm_Norm")

# #---------------------------------
# ##LINEAR NOT NORM PCAs EMPIRICAL
# labels_text = [
#     "Linear LogReg NO PCA",
#     "Linear LogReg PCA 5",
#     "Linear LogReg PCA 4"
# ]
# CprimLogReg = np.zeros((3, len(lambdas)))
# piT = 0.17
# with open("LinearLogRegEmpiricalPCA.txt", "w") as f:
#     ##LINEAR NOT NORMALIZED piT = empirical No PCA
#     minDCFList = np.zeros((2, len(lambdas)))
#     for index, piTilde in enumerate([0.1,0.5]):
#         for lIndex, l in enumerate(lambdas):
#             logRegObj = lrc.logReg(l, piT, "balanced") 
#             _, minDCF = val.k_fold(logRegObj, features, labels, k, (piTilde, 1,1))
#             print("Linear LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
#             print("Linear LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
#             minDCFList[index, lIndex] = minDCF
#     CprimLogReg[0] = minDCFList.mean(axis=0)
#     print("Cprim list {}".format(CprimLogReg[0]), file=f)
#     print("Cprim list {}".format(CprimLogReg[0]))

#     ##LINEAR NOT NORMALIZED piT = empirical PCA 5
#     minDCFList = np.zeros((2, len(lambdas)))
#     for index, piTilde in enumerate([0.1,0.5]):
#         for lIndex, l in enumerate(lambdas):
#             logRegObj = lrc.logReg(l, piT, "balanced") 
#             _, minDCF = val.k_fold(logRegObj, featuresPCA5, labels, k, (piTilde, 1,1))
#             print("Linear LogReg Empirical Not Normalized PCA 5, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
#             print("Linear LogReg Empirical Not Normalized PCA 5, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
#             minDCFList[index, lIndex] = minDCF
#     CprimLogReg[1] = minDCFList.mean(axis=0)
#     print("Cprim list {}".format(CprimLogReg[1]), file=f)
#     print("Cprim list {}".format(CprimLogReg[1]))

#     ##LINEAR NOT NORMALIZED piT = empirical PCA 4
#     minDCFList = np.zeros((2, len(lambdas)))
#     for index, piTilde in enumerate([0.1,0.5]):
#         for lIndex, l in enumerate(lambdas):
#             logRegObj = lrc.logReg(l, piT, "balanced") 
#             _, minDCF = val.k_fold(logRegObj, featuresPCA4, labels, k, (piTilde, 1,1))
#             print("Linear LogReg Empirical Not Normalized PCA 4, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
#             print("Linear LogReg Empirical Not Normalized PCA 4, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
#             minDCFList[index, lIndex] = minDCF
#     CprimLogReg[2] = minDCFList.mean(axis=0)
#     print("Cprim list {}".format(CprimLogReg[2]), file=f)
#     print("Cprim list {}".format(CprimLogReg[2]))

#     dv.plotCPrim(lambdas, CprimLogReg, labels_text, "λ", "LinearLogRegEmpiricalPCA")

#----------------------------------------------
# #QUADRATIC
# labels_text = [
#     "Quadratic LogReg",
#     "Quadratic LogReg Z-norm"
# ]
# piT = 0.17
# with open("QLogRegEmpiricalNoNorm_Norm.txt", "w") as f:
#     CprimLogReg = np.zeros((2, len(lambdas)))
#     ##QUADRATIC NOT NORMALIZED piT = empirical
#     minDCFList = np.zeros((2, len(lambdas)))
#     for index, piTilde in enumerate([0.1,0.5]):
#         for lIndex, l in enumerate(lambdas):
#             logRegObj = lrc.logReg(l, piT, "balanced") 
#             _, minDCF = val.k_fold(logRegObj, featuresTrainQuadratic, labels, k, (piTilde, 1,1))
#             print("Quadratic LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
#             print("Quadratic LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
#             minDCFList[index, lIndex] = minDCF
#     CprimLogReg[0] = minDCFList.mean(axis=0)
#     print("Cprim list {}".format(CprimLogReg), file=f)
#     print("Cprim list {}".format(CprimLogReg))

#     ##QUADRATIC NORMALIZED piT = empirical
#     minDCFList = np.zeros((2, len(lambdas)))
#     for index, piTilde in enumerate([0.1,0.5]):
#         for lIndex, l in enumerate(lambdas):
#             logRegObj = lrc.logReg(l, piT, "balanced")
#             _, minDCF = val.k_fold(logRegObj, featuresTrainQuadraticZNorm, labels, k, (piTilde, 1, 1))
#             print("Quadratic LogReg Empirical Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF), file=f)
#             print("Quadratic LogReg Empirical Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
#             minDCFList[index, lIndex] = minDCF
#     CprimLogReg[1] = minDCFList.mean(axis=0)
#     print("Cprim list {}".format(CprimLogReg), file=f)
#     print("Cprim list {}".format(CprimLogReg))

#     dv.plotCPrim(lambdas, CprimLogReg, labels_text, "λ", "QLogRegEmpiricalNoNorm_Norm")

#---------------------------------
##QUADRATIC NOT NORM PCAs EMPIRICAL
labels_text = [
    "Quadratic LogReg NO PCA",
    "Quadratic LogReg NO PCA Z-Norm",
    "Quadratic LogReg PCA 5",
    "Quadratic LogReg PCA 5 Z-Norm",
]
CprimLogReg = np.zeros((4, len(lambdas)))
piT = 0.17
with open("QuadraticLogRegEmpiricalPCANorm.txt", "w") as f:
    #QUADRATIC NOT NORMALIZED piT = empirical No PCA
    minDCFList = np.zeros((2, len(lambdas)))
    for index, piTilde in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, piT, "balanced") 
            _, minDCF = val.k_fold(logRegObj, featuresTrainQuadratic, labels, k, (piTilde, 1,1))
            print("Quadratic LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
            print("Quadratic LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[0] = minDCFList.mean(axis=0)
    print("Cprim list {}".format(CprimLogReg[0]), file=f)
    print("Cprim list {}".format(CprimLogReg[0]))

    #QUADRATIC NORMALIZED piT = empirical No PCA
    minDCFList = np.zeros((2, len(lambdas)))
    for index, piTilde in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, piT, "balanced") 
            _, minDCF = val.k_fold(logRegObj, featuresTrainQuadraticZNorm, labels, k, (piTilde, 1,1))
            print("Quadratic LogReg Empirical Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
            print("Quadratic LogReg Empirical Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[1] = minDCFList.mean(axis=0)
    print("Cprim list {}".format(CprimLogReg[1]), file=f)
    print("Cprim list {}".format(CprimLogReg[1]))

    #QUADRATIC NOT NORMALIZED piT = empirical PCA 5
    minDCFList = np.zeros((2, len(lambdas)))
    for index, piTilde in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, piT, "balanced") 
            _, minDCF = val.k_fold(logRegObj, featuresPCA5Exapanded, labels, k, (piTilde, 1,1))
            print("Quadratic LogReg Empirical Not Normalized PCA 5, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
            print("Quadratic LogReg Empirical Not Normalized PCA 5, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[2] = minDCFList.mean(axis=0)
    print("Cprim list {}".format(CprimLogReg[2]), file=f)
    print("Cprim list {}".format(CprimLogReg[2]))

    #QUADRATIC NORMALIZED piT = empirical PCA 5
    minDCFList = np.zeros((2, len(lambdas)))
    for index, piTilde in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, piT, "balanced") 
            _, minDCF = val.k_fold(logRegObj, featuresPCA5ZNormExapanded, labels, k, (piTilde, 1,1))
            print("Quadratic LogReg Empirical Normalized PCA 5, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
            print("Quadratic LogReg Empirical Normalized PCA 5, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[3] = minDCFList.mean(axis=0)
    print("Cprim list {}".format(CprimLogReg[3]), file=f)
    print("Cprim list {}".format(CprimLogReg[3]))

    dv.plotCPrim(lambdas, CprimLogReg, labels_text, "λ", "QuadraticLogRegEmpiricalPCANorm")


#---------------------------------
##QUADRATIC NOT NORM PCAs EMPIRICAL
# labels_text = [
#     "Quadratic LogReg NO PCA",
#     "Quadratic LogReg PCA 5",
#     "Quadratic LogReg PCA 4"
# ]
# CprimLogReg = np.zeros((3, len(lambdas)))
# piT = 0.17
# with open("QuadraticLogRegEmpiricalPCATEST2.txt", "w") as f:
#     #QUADRATIC NOT NORMALIZED piT = empirical No PCA
#     minDCFList = np.zeros((2, len(lambdas)))
#     for index, piTilde in enumerate([0.1,0.5]):
#         for lIndex, l in enumerate(lambdas):
#             logRegObj = lrc.logReg(l, piT, "balanced") 
#             _, minDCF = val.k_fold(logRegObj, featuresTrainQuadratic, labels, k, (piTilde, 1,1))
#             print("Quadratic LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
#             print("Quadratic LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
#             minDCFList[index, lIndex] = minDCF
#     CprimLogReg[0] = minDCFList.mean(axis=0)
#     print("Cprim list {}".format(CprimLogReg[0]), file=f)
#     print("Cprim list {}".format(CprimLogReg[0]))

#     #QUADRATIC NOT NORMALIZED piT = empirical PCA 5
#     minDCFList = np.zeros((2, len(lambdas)))
#     for index, piTilde in enumerate([0.1,0.5]):
#         for lIndex, l in enumerate(lambdas):
#             logRegObj = lrc.logReg(l, piT, "balanced") 
#             _, minDCF = val.k_fold(logRegObj, featuresPCA5Exapanded, labels, k, (piTilde, 1,1))
#             print("Quadratic LogReg Empirical Not Normalized PCA 5, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
#             print("Quadratic LogReg Empirical Not Normalized PCA 5, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
#             minDCFList[index, lIndex] = minDCF
#     CprimLogReg[1] = minDCFList.mean(axis=0)
#     print("Cprim list {}".format(CprimLogReg[1]), file=f)
#     print("Cprim list {}".format(CprimLogReg[1]))

#     #QUADRATIC NOT NORMALIZED piT = empirical PCA 4
#     minDCFList = np.zeros((2, len(lambdas)))
#     for index, piTilde in enumerate([0.1,0.5]):
#         for lIndex, l in enumerate(lambdas):
#             logRegObj = lrc.logReg(l, piT, "balanced") 
#             _, minDCF = val.k_fold(logRegObj, featuresPCA4Exapanded, labels, k, (piTilde, 1,1))
#             print("Quadratic LogReg Empirical Not Normalized PCA 4, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
#             print("Quadratic LogReg Empirical Not Normalized PCA 4, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
#             minDCFList[index, lIndex] = minDCF
#     CprimLogReg[2] = minDCFList.mean(axis=0)
#     print("Cprim list {}".format(CprimLogReg[2]), file=f)
#     print("Cprim list {}".format(CprimLogReg[2]))

#     dv.plotCPrim(lambdas, CprimLogReg, labels_text, "λ", "QuadraticLogRegEmpiricalPCATEST2")

