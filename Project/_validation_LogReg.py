import numpy as np
import data_utils as du
import dimensionality_reduction as dr
import logistic_regression_classifiers as lrc
import math_utils as mu
import data_visualization as dv
import validation as val

labels, features = du.load(".\Data\Train.txt")  
lambdas = np.logspace(-2, 3, num=30)
k = 5

featuresTrainQuadratic = du.features_expansion(features)
#Normalization first
featuresZNorm = mu.z_score(features)
featuresTrainQuadraticZNorm = du.features_expansion(mu.z_score(features))

featuresPCA5 = dr.PCA(features,5)
featuresPCA5Exapanded = du.features_expansion(dr.PCA(features,5))
featuresPCA5ZNorm = dr.PCA(mu.z_score(features),5)
featuresPCA4 = dr.PCA(features,4)
featuresPCA4Exapanded = du.features_expansion(dr.PCA(features,4))
featuresPCA5ZNormExapanded = du.features_expansion(dr.PCA(mu.z_score(features), 5))

#LINEAR NOT NORM + LINEAR Z-NORM EMPIRICAL
labels_text = [
    "Linear LogReg",
    "Linear LogReg Z-norm"
]
CprimLogReg = np.zeros((2, len(lambdas)))
piT = 0.17
with open("LinearLogRegEmpiricalNoNorm_NormGIUSTO.txt", "w") as f:
    ##LINEAR NOT NORMALIZED piT = empirical
    minDCFList = np.zeros((2, len(lambdas)))
    for index, piTilde in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, piT, "balanced") 
            actualDCF, minDCF,_,_ = val.k_fold_bayes_plot(logRegObj, features, labels, k, (piTilde, 1,1), "LinearLogRegEmpiricalNoNorm_Norm", False)
            print("Linear LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
            print("Linear LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[0] = minDCFList.mean(axis=0)
    print("Cprim list {}".format(CprimLogReg[0]), file=f)
    print("Cprim list {}".format(CprimLogReg[0]))


    ##LINEAR NORMALIZED piT = empirical
    minDCFList = np.zeros((2, len(lambdas)))
    for index, piTilde in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, piT, "balanced")
            actualDCF, minDCF,_,_ = val.k_fold_bayes_plot(logRegObj, featuresZNorm, labels, k, (piTilde, 1, 1), "LinearLogRegEmpiricalNoNorm_Norm", False)
            print("Linear LogReg Empirical Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
            print("Linear LogReg Empirical Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[1] = minDCFList.mean(axis=0)
    print("Cprim list {}".format(CprimLogReg[1]), file=f)
    print("Cprim list {}".format(CprimLogReg[1]))

    dv.plotCPrim(lambdas, CprimLogReg, labels_text, "λ", "LinearLogRegEmpiricalNoNorm_NormGIUSTO")

##LINEAR NOT NORM PCAs EMPIRICAL
labels_text = [
    "Linear LogReg NO PCA",
    "Linear LogReg PCA 5",
    "Linear LogReg PCA 4"
]
CprimLogReg = np.zeros((3, len(lambdas)))
piT = 0.17
with open("LinearLogRegEmpiricalPCA.txt", "w") as f:
    ##LINEAR NOT NORMALIZED piT = empirical No PCA
    minDCFList = np.zeros((2, len(lambdas)))
    for index, piTilde in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, piT, "balanced") 
            actualDCF, minDCF,_,_ = val.k_fold_bayes_plot(logRegObj, features, labels, k, (piTilde, 1,1), "LinearLogRegEmpiricalPCA", False)
            print("Linear LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
            print("Linear LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[0] = minDCFList.mean(axis=0)
    print("Cprim list {}".format(CprimLogReg[0]), file=f)
    print("Cprim list {}".format(CprimLogReg[0]))

    ##LINEAR NOT NORMALIZED piT = empirical PCA 5
    minDCFList = np.zeros((2, len(lambdas)))
    for index, piTilde in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, piT, "balanced") 
            actualDCF, minDCF,_,_ = val.k_fold_bayes_plot(logRegObj, featuresPCA5, labels, k, (piTilde, 1,1), "LinearLogRegEmpiricalPCA", False)
            print("Linear LogReg Empirical Not Normalized PCA 5, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
            print("Linear LogReg Empirical Not Normalized PCA 5, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[1] = minDCFList.mean(axis=0)
    print("Cprim list {}".format(CprimLogReg[1]), file=f)
    print("Cprim list {}".format(CprimLogReg[1]))

    ##LINEAR NOT NORMALIZED piT = empirical PCA 4
    minDCFList = np.zeros((2, len(lambdas)))
    for index, piTilde in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, piT, "balanced") 
            actualDCF, minDCF,_,_ = val.k_fold_bayes_plot(logRegObj, featuresPCA4, labels, k, (piTilde, 1,1), "LinearLogRegEmpiricalPCA", False)
            print("Linear LogReg Empirical Not Normalized PCA 4, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
            print("Linear LogReg Empirical Not Normalized PCA 4, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[2] = minDCFList.mean(axis=0)
    print("Cprim list {}".format(CprimLogReg[2]), file=f)
    print("Cprim list {}".format(CprimLogReg[2]))

    dv.plotCPrim(lambdas, CprimLogReg, labels_text, "λ", "LinearLogRegEmpiricalPCA")


# QUADRATIC
labels_text = [
    "Quadratic LogReg",
    "Quadratic LogReg Z-norm"
]
lambdas = np.logspace(-5, 3, num=40)
piT = 0.17
with open("QLogRegEmpiricalNoNorm_NormGIUSTO.txt", "w") as f:
    CprimLogReg = np.zeros((2, len(lambdas)))
    #QUADRATIC NOT NORMALIZED piT = empirical
    minDCFList = np.zeros((2, len(lambdas)))
    for index, piTilde in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, piT, "balanced") 
            actualDCF, minDCF,_,_ = val.k_fold_bayes_plot(logRegObj, featuresTrainQuadratic, labels, k, (piTilde, 1,1), "QLogRegEmpiricalNoNorm_Norm", False)
            print("Quadratic LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
            print("Quadratic LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[0] = minDCFList.mean(axis=0)
    print("Cprim list {}".format(CprimLogReg), file=f)
    print("Cprim list {}".format(CprimLogReg))

    #QUADRATIC NORMALIZED piT = empirical
    minDCFList = np.zeros((2, len(lambdas)))
    for index, piTilde in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, piT, "balanced")
            actualDCF, minDCF,_,_ = val.k_fold_bayes_plot(logRegObj, featuresTrainQuadraticZNorm, labels, k, (piTilde, 1, 1), "QLogRegEmpiricalNoNorm_Norm", False)
            print("Quadratic LogReg Empirical Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF), file=f)
            print("Quadratic LogReg Empirical Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[1] = minDCFList.mean(axis=0)
    print("Cprim list {}".format(CprimLogReg), file=f)
    print("Cprim list {}".format(CprimLogReg))

    dv.plotCPrim(lambdas, CprimLogReg, labels_text, "λ", "QLogRegEmpiricalNoNorm_NormGIUSTO")


# QUADRATIC NOT NORM PCAs EMPIRICAL
labels_text = [
    "Quadratic LogReg NO PCA",
    "Quadratic LogReg NO PCA Z-Norm",
    "Quadratic LogReg PCA 5",
    "Quadratic LogReg PCA 5 Z-Norm"
]
lambdas = np.logspace(-5, 3, num=40)
CprimLogReg = np.zeros((4, len(lambdas)))
piT = 0.17
with open("QuadraticLogRegEmpiricalPCANormGIUSTO.txt", "w") as f:
    # QUADRATIC NOT NORMALIZED piT = empirical No PCA
    minDCFList = np.zeros((2, len(lambdas)))
    for index, piTilde in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, piT, "balanced") 
            actualDCF, minDCF,_,_ = val.k_fold_bayes_plot(logRegObj, featuresTrainQuadratic, labels, k, (piTilde, 1,1), "QuadraticLogRegEmpiricalPCANorm", False)
            print("Quadratic LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
            print("Quadratic LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[0] = minDCFList.mean(axis=0)
    print("Cprim list {}".format(CprimLogReg[0]), file=f)
    print("Cprim list {}".format(CprimLogReg[0]))

    # QUADRATIC NORMALIZED piT = empirical No PCA
    minDCFList = np.zeros((2, len(lambdas)))
    for index, piTilde in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, piT, "balanced") 
            actualDCF, minDCF,_,_ = val.k_fold_bayes_plot(logRegObj, featuresTrainQuadraticZNorm, labels, k, (piTilde, 1,1), "QuadraticLogRegEmpiricalPCANorm", False)
            print("Quadratic LogReg Empirical Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
            print("Quadratic LogReg Empirical Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[1] = minDCFList.mean(axis=0)
    print("Cprim list {}".format(CprimLogReg[1]), file=f)
    print("Cprim list {}".format(CprimLogReg[1]))

    # QUADRATIC NOT NORMALIZED piT = empirical PCA 5
    minDCFList = np.zeros((2, len(lambdas)))
    for index, piTilde in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, piT, "balanced") 
            actualDCF, minDCF,_,_ = val.k_fold_bayes_plot(logRegObj, featuresPCA5Exapanded, labels, k, (piTilde, 1,1), "QuadraticLogRegEmpiricalPCANorm", False)
            print("Quadratic LogReg Empirical Not Normalized PCA 5, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
            print("Quadratic LogReg Empirical Not Normalized PCA 5, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[2] = minDCFList.mean(axis=0)
    print("Cprim list {}".format(CprimLogReg[2]), file=f)
    print("Cprim list {}".format(CprimLogReg[2]))

    # QUADRATIC NORMALIZED piT = empirical PCA 5
    minDCFList = np.zeros((2, len(lambdas)))
    for index, piTilde in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, piT, "balanced") 
            actualDCF, minDCF,_,_ = val.k_fold_bayes_plot(logRegObj, featuresPCA5ZNormExapanded, labels, k, (piTilde, 1,1), "QuadraticLogRegEmpiricalPCANorm", False)
            print("Quadratic LogReg Empirical Normalized PCA 5, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
            print("Quadratic LogReg Empirical Normalized PCA 5, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[3] = minDCFList.mean(axis=0)
    print("Cprim list {}".format(CprimLogReg[3]), file=f)
    print("Cprim list {}".format(CprimLogReg[3]))

    dv.plotCPrim(lambdas, CprimLogReg, labels_text, "λ", "QuadraticLogRegEmpiricalPCANormGIUSTO")

#QUADRATIC NOT NORM PCAs EMPIRICAL
labels_text = [
    "Quadratic LogReg NO PCA",
    "Quadratic LogReg PCA 5",
    "Quadratic LogReg PCA 4"
]
CprimLogReg = np.zeros((3, len(lambdas)))
piT = 0.17
with open("QuadraticLogRegEmpiricalPCATEST2.txt", "w") as f:
    #QUADRATIC NOT NORMALIZED piT = empirical No PCA
    minDCFList = np.zeros((2, len(lambdas)))
    for index, piTilde in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, piT, "balanced") 
            actualDCF, minDCF,_,_ = val.k_fold_bayes_plot(logRegObj, featuresTrainQuadratic, labels, k, (piTilde, 1,1), "QuadraticLogRegEmpiricalPCATEST2", False)
            print("Quadratic LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
            print("Quadratic LogReg Empirical Not Normalized, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[0] = minDCFList.mean(axis=0)
    print("Cprim list {}".format(CprimLogReg[0]), file=f)
    print("Cprim list {}".format(CprimLogReg[0]))

    #QUADRATIC NOT NORMALIZED piT = empirical PCA 5
    minDCFList = np.zeros((2, len(lambdas)))
    for index, piTilde in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, piT, "balanced") 
            actualDCF, minDCF,_,_ = val.k_fold_bayes_plot(logRegObj, featuresPCA5Exapanded, labels, k, (piTilde, 1,1), "QuadraticLogRegEmpiricalPCATEST2", False)
            print("Quadratic LogReg Empirical Not Normalized PCA 5, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
            print("Quadratic LogReg Empirical Not Normalized PCA 5, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[1] = minDCFList.mean(axis=0)
    print("Cprim list {}".format(CprimLogReg[1]), file=f)
    print("Cprim list {}".format(CprimLogReg[1]))

    #QUADRATIC NOT NORMALIZED piT = empirical PCA 4
    minDCFList = np.zeros((2, len(lambdas)))
    for index, piTilde in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, piT, "balanced") 
            actualDCF, minDCF,_,_ = val.k_fold_bayes_plot(logRegObj, featuresPCA4Exapanded, labels, k, (piTilde, 1,1), "QuadraticLogRegEmpiricalPCATEST2", False)
            print("Quadratic LogReg Empirical Not Normalized PCA 4, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF),file=f)
            print("Quadratic LogReg Empirical Not Normalized PCA 4, minDCF with piTilde {} and lambda {} is {}".format(piTilde, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[2] = minDCFList.mean(axis=0)
    print("Cprim list {}".format(CprimLogReg[2]), file=f)
    print("Cprim list {}".format(CprimLogReg[2]))

    dv.plotCPrim(lambdas, CprimLogReg, labels_text, "λ", "QuadraticLogRegEmpiricalPCATEST2")

# BEST MODEL PITILDE
lambdaBest=10
with open("QLOGREG_PI_PITILDE.txt","w") as f:
    for piT in [0.1,0.17,0.2,0.3, 0.4, 0.5]:
        CprimList=[]
        for piTilde in [0.1,0.5]:
            logRegObj = lrc.logReg(lambdaBest, piT, "balanced")
            actualDCF, minDCF,_,_ = val.k_fold_bayes_plot(logRegObj, featuresTrainQuadratic, labels, 5, (piTilde, 1, 1), "QLogRegEmpiricalNoNorm_Norm", False)
            print("Quadratic LogReg Balanced Not Normalized, minDCF with piT {} and piTilde {} no PCA and  is {}".format(piT, piTilde, minDCF))
            print("Quadratic LogReg Balanced Not Normalized, minDCF with piT {} and piTilde {} no PCA and  is {}".format(piT, piTilde, minDCF),file=f)
            CprimList.append(minDCF)
        Cprim=np.array(CprimList).mean(axis=0)
        print("Quadratic LogReg Balanced Not Normalized, Cprim with piT {}  no PCA and  is {}".format(piT,Cprim ))
        print("Quadratic LogReg Balanced Not Normalized, Cprim with piT {}  no PCA and  is {}".format(piT,Cprim ),file=f)
