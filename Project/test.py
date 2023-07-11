import data_utils as du
import data_visualization as dv
import dimensionality_reduction as dr
import gaussian_classifiers as gc
import validation as val
import math_utils as mu
import logistic_regression_classifiers as lrc
import GMM as gmm
import SVM_classifiers as svmc
from datetime import datetime
import numpy as np
import sys

if __name__ == "__main__":
    start_time = datetime.now()

    orig_stdout = sys.stdout
    f = open('GMM_tests.txt', 'w')
    sys.stdout = f

    #LOADING DATASET
    labels, features = du.load("..\PROJECTS\Language_detection\Train.txt")
    labels_test, features_test = du.load("..\PROJECTS\Language_detection\Test.txt")

    #DICTIONARIES
    labels_dict = {
        "Not-Italian": 0,
        "Italian": 1
    }
    features_dict = {
        "feature 0": 0,
        "feature 1": 1,
        "feature 2": 2,
        "feature 3": 3,
        "feature 4": 4,
        "feature 5": 5        
    }
    features_dict_PCA = {
        "PC-0": 0,
        "PC-1": 1,
        "PC-2": 2,
        "PC-3": 3,
        "PC-4": 4   
    }

    """
    #MVG
    for pi in [0.1,0.5]:
        #NO PCA
        mvgObj = gc.multivariate_cl([1-pi, pi])
        _, minDCF = val.k_fold(mvgObj, features, labels, 5, (pi, 1, 1))
        print("MVG, minDCF with pi {} is {}".format(pi, minDCF))
        
        #PCA
        for nPCA in [5,4,3]:
            dataPCA = dr.PCA(features, nPCA)
            mvgObj = gc.multivariate_cl([1-pi, pi])
            _, minDCF = val.k_fold(mvgObj, dataPCA, labels, 5, (pi, 1, 1))
            print("MVG, minDCF with pi {} and {} PCA is {}".format(pi, nPCA, minDCF))

    #NAIVE MVG
    for pi in [0.1,0.5]:
        #NO PCA
        naiveMvgObj = gc.naive_multivariate_cl([1-pi, pi])
        _, minDCF = val.k_fold(naiveMvgObj, features, labels, 5, (pi, 1, 1))
        print("Naive MVG, minDCF with pi {} is {}".format(pi, minDCF))
        
        #PCA
        for nPCA in [5,4,3]:
            dataPCA = dr.PCA(features, nPCA)
            naiveMvgObj = gc.naive_multivariate_cl([1-pi, pi])
            _, minDCF = val.k_fold(naiveMvgObj, dataPCA, labels, 5, (pi, 1, 1))
            print("Naive MVG, minDCF with pi {} and {} PCA is {}".format(pi, nPCA, minDCF))

    #TIED MVG
    for pi in [0.1,0.5]:
        #NO PCA
        tiedMvgObj = gc.tied_multivariate_cl([1-pi, pi])
        _, minDCF = val.k_fold(tiedMvgObj, features, labels, 5, (pi, 1, 1))
        print("Tied MVG, minDCF with pi {} is {}".format(pi, minDCF))
        
        #PCA
        for nPCA in [5,4,3]:
            dataPCA = dr.PCA(features, nPCA)
            tiedMvgObj = gc.tied_multivariate_cl([1-pi, pi])
            _, minDCF = val.k_fold(tiedMvgObj, dataPCA, labels, 5, (pi, 1, 1))
            print("Tied MVG, minDCF with pi {} and {} PCA is {}".format(pi, nPCA, minDCF))

    #TIED NAIVE MVG
    for pi in [0.1,0.5]:
        #NO PCA
        tiedNaiveMvgObj = gc.tied_naive_multivariate_cl([1-pi, pi])
        _, minDCF = val.k_fold(tiedNaiveMvgObj, features, labels, 5, (pi, 1, 1))
        print("Tied Naive MVG, minDCF with pi {} is {}".format(pi, minDCF))
        
        #PCA
        for nPCA in [5,4,3]:
            dataPCA = dr.PCA(features, nPCA)
            tiedNaiveMvgObj = gc.tied_naive_multivariate_cl([1-pi, pi])
            _, minDCF = val.k_fold(tiedNaiveMvgObj, dataPCA, labels, 5, (pi, 1, 1))
            print("Tied Naive MVG, minDCF with pi {} and {} PCA is {}".format(pi, nPCA, minDCF))
    """

    #LOGISTIC REGRESSION
    """
    featuresTrainQuadratic = du.features_expansion(features)
    featuresTrainQuadraticZNorm = mu.z_score(featuresTrainQuadratic)
    featuresZNorm = mu.z_score(features)
    """
    """
    #QLOG REG NO NORMALIZATION
    lambdas = np.logspace(-3, 5, num=50)
    CprimLogReg = np.zeros((2, len(lambdas)))
    minDCFList = np.zeros((2, len(lambdas)))
    for index, pi in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, pi, False)
            _, minDCF = val.k_fold(logRegObj, featuresTrainQuadratic, labels, 5, (pi, 1, 1))
            print("LogReg, minDCF with pi {} and lambda {} is {}".format(pi, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[0] = minDCFList.mean(axis=0)

    #QLOG REG Z-NORM
    minDCFList = np.zeros((2, len(lambdas)))
    for index, pi in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, pi, True)
            _, minDCF = val.k_fold(logRegObj, featuresTrainQuadraticZNorm, labels, 5, (pi, 1, 1))
            print("LogReg Z-Norm, minDCF with pi {} and lambda {} is {}".format(pi, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[1] = minDCFList.mean(axis=0)

    dv.plotCPrim(lambdas, CprimLogReg, ["QLog-Reg", "QLog-Reg z-norm"] , "λ", "QLogReg_QLogRegNorm")

    #LOG REG NO NORMALIZATION
    lambdas = np.logspace(-3, 5, num=50)
    CprimLogReg = np.zeros((2, len(lambdas)))
    minDCFList = np.zeros((2, len(lambdas)))
    for index, pi in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, pi, False)
            _, minDCF = val.k_fold(logRegObj, features, labels, 5, (pi, 1, 1))
            print("LogReg, minDCF with pi {} and lambda {} is {}".format(pi, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[0] = minDCFList.mean(axis=0)

    #LOG REG Z-NORM
    minDCFList = np.zeros((2, len(lambdas)))
    for index, pi in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, pi, True)
            _, minDCF = val.k_fold(logRegObj, featuresZNorm, labels, 5, (pi, 1, 1))
            print("LogReg Z-Norm, minDCF with pi {} and lambda {} is {}".format(pi, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[1] = minDCFList.mean(axis=0)

    dv.plotCPrim(lambdas, CprimLogReg, ["Log-Reg", "Log-Reg z-norm"] , "λ", "LogReg_LogRegNorm")
    """
    """
    #QLOG REG PCA
    lambdas = np.logspace(-3, 5, num=50)
    CprimLogReg = np.zeros((4, len(lambdas)))
    
    for PCAIndex, nPCA in enumerate([5,4,3]):
        minDCFList = np.zeros((2, len(lambdas)))
        dataPCA = dr.PCA(featuresTrainQuadratic, nPCA)
        dataPCAExpanded = du.features_expansion(dataPCA)
        for index, pi in enumerate([0.1,0.5]):
            for lIndex, l in enumerate(lambdas):
                logRegObj = lrc.logReg(l, pi, False)
                _, minDCF = val.k_fold(logRegObj, dataPCAExpanded, labels, 5, (pi, 1, 1))
                print("LogReg, minDCF with pi {}, PCA {} and lambda {} is {}".format(pi, nPCA, l, minDCF))
                minDCFList[index, lIndex] = minDCF
        CprimLogReg[PCAIndex] = minDCFList.mean(axis=0)

    #QLOG REG NO NORMALIZATION
    minDCFList = np.zeros((2, len(lambdas)))
    for index, pi in enumerate([0.1,0.5]):
        for lIndex, l in enumerate(lambdas):
            logRegObj = lrc.logReg(l, pi, False)
            _, minDCF = val.k_fold(logRegObj, featuresTrainQuadratic, labels, 5, (pi, 1, 1))
            print("LogReg, minDCF with pi {} and lambda {} is {}".format(pi, l, minDCF))
            minDCFList[index, lIndex] = minDCF
    CprimLogReg[3] = minDCFList.mean(axis=0)

    print(CprimLogReg)
    dv.plotCPrim(lambdas, CprimLogReg, ["QLog-Reg PCA-5", "QLog-Reg PCA-4", "QLog-Reg PCA-3", "QLog-Reg no PCA"] , "λ", "QLogRegPCAs")
    """


    #TEST GMM

    #GMM no PCA
    for nTarget in [1,2,4,8]:
        for nNonTarget in [1,2,4,8,16,32]:
            for MtypeTarget in ["mvg", "tied", "diagonal", "tied diagonal"]:
                for MtypeNonTarget in ["mvg", "tied", "diagonal", "tied diagonal"]:
                    minDCFSum = 0
                    for pi in [0.1,0.5]:
                        GMMclass=gmm.GMM(nTarget,nNonTarget,MtypeTarget,MtypeNonTarget)
                        _,minDCF = val.k_fold(GMMclass,features,labels,5, (pi,1,1))
                        minDCFSum += minDCF
                    Cprim = minDCFSum/2
                    print("GMM Cprim no PCA, nTarget({}), nNonTarget({}), MTypeTarget({}), MTypeNonTarget({}): {}".format(nTarget,nNonTarget,MtypeTarget,MtypeNonTarget,Cprim))    

    #GMM
    for nPCA in [5,4,3]:
        dataPCA = dr.PCA(features, nPCA)
        for nTarget in [1,2,4,8]:
            for nNonTarget in [1,2,4,8,16,32]:
                for MtypeTarget in ["mvg", "tied", "diagonal", "tied diagonal"]:
                    for MtypeNonTarget in ["mvg", "tied", "diagonal", "tied diagonal"]:
                        minDCFSum = 0
                        for pi in [0.1,0.5]:
                            GMMclass=gmm.GMM(nTarget,nNonTarget,MtypeTarget,MtypeNonTarget)
                            _,minDCF = val.k_fold(GMMclass,dataPCA,labels,5, (pi,1,1))
                            minDCFSum += minDCF
                        Cprim = minDCFSum/2
                        print("GMM Cprim PCA({}), nTarget({}), nNonTarget({}), MTypeTarget({}), MTypeNonTarget({}): {}".format(nPCA,nTarget,nNonTarget,MtypeTarget,MtypeNonTarget,Cprim))                            
    
    end_time = datetime.now()

    print("--------- TIME ----------")
    print(f"Time elapsed: {end_time - start_time}")


    sys.stdout = orig_stdout
    f.close()
        