import data_utils as du
import data_visualization as dv
import dimensionality_reduction as dr
import gaussian_classifiers as gc
import validation2 as val
import math_utils as mu
import logistic_regression_classifiers as lrc
import GMM as gmm
import SVM_classifiers as svmc
from datetime import datetime
import numpy as np
import sys
from itertools import repeat
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import fusion

"""
    Simply train the model with the chosen configuration on the WHOLE training set and evaluate it with the test set.
    
    Do not test linear models, not needed.

    For simplicity consider minCprim.

    What's the point: 
        - From validation we found our best systems which are characterized by a configuration of parameters.
        - - GMM: 2 FC 32 FC-T NO PCA
        - - SVM: RBF Balanced gamma = 0.01, C = 0.1, K = 0.01 piT = 0.2 PCA 5
        - - QLR: Quadratic Logistic Regression NO PCA piT = 0.1, lambda = 10
        - - Best fusion GMM + SVM 
        - Now we want to evaluate them on the test set to see how they perform on unseen data.

"""

if __name__ == "__main__":
    # LOADING DATASET
    labels, features = du.load("Train.txt")
    labels_test, features_test = du.load("Test.txt")
    n_class0 = np.sum(labels==0)
    n_class1 = np.sum(labels==1)
    n_tot = n_class0 + n_class1
    print("Class 0: ", n_class0)
    print("Class 1: ", n_class1)
    print("Total: ", n_tot)
    print("P(Class 0): ", n_class0/n_tot)
    print("P(Class 1): ", n_class1/n_tot)
    
    """
    # DATA MODELLING
    features_exp = du.features_expansion(features) 
    features_PCA_5 = dr.PCA(features, 5)

    features_test_exp = du.features_expansion(features_test)
    features_test_PCA_5 = dr.PCA(features_test, 5)

    # BEST MODELS
    QLR = lrc.logReg(10, 0.1, "balanced")
    SVMC = svmc.SVM('RBF', balanced=True, gamma=0.01, K=0.01, C=0.1, piT=0.2)
    GMM = gmm.GMM(2, 32, "mvg", "tied")

    # TRAINING
    QLR.train(features_exp, labels)
    SVMC.train(features_PCA_5, labels)
    GMM.train(features, labels)

    # TRANSFORM
    predictedQLR = QLR.transform(features_test_exp)
    predicetedSVM = SVMC.transform(features_test_PCA_5)
    predictedGMM = GMM.transform(features_test)

    # OBTAINING SCORES
    scoresQLR = QLR.get_scores()
    scoresSVM = SVMC.get_scores()
    scoresGMM = GMM.get_scores()

    # OBTAINING Cprim
    minDCFQLR1 = val.min_DCF(scoresQLR, 0.5, 1, 1, labels_test, predictedQLR)
    minDCFQLR2 = val.min_DCF(scoresQLR, 0.1, 1, 1, labels_test, predictedQLR)
    minCprimQLR = (minDCFQLR1 + minDCFQLR2)/2

    minDCFSVM1 = val.min_DCF(scoresSVM, 0.5, 1, 1, labels_test, predicetedSVM)
    minDCFSVM2 = val.min_DCF(scoresSVM, 0.1, 1, 1, labels_test, predicetedSVM)
    minCprimSVM = (minDCFSVM1 + minDCFSVM2)/2

    minDCFGMM1 = val.min_DCF(scoresGMM, 0.5, 1, 1, labels_test, predictedGMM)
    minDCFGMM2 = val.min_DCF(scoresGMM, 0.1, 1, 1, labels_test, predictedGMM)
    minCprimGMM = (minDCFGMM1 + minDCFGMM2)/2

    # OBTAINING DET
    val.get_multi_DET([scoresQLR, scoresSVM, scoresGMM], labels_test, ["QLR", "SMV", "GMM"], "Best Models EVAL")

    # PRINTING RESULTS OF BEST MODEL WITH CHOSEN CONFIGURATION
    print("QLR minCprim: ", minCprimQLR)
    print("SVM minCprim: ", minCprimSVM)
    print("GMM minCprim: ", minCprimGMM)
    """

    """
    Results of the best models with the chosen configuration:
        - QLR minCprim: 0.271
        - SVM minCprim: 0.270
        - GMM minCprim: 0.256
    While during training (calibrated) we got:
        - QLR minCprim: 0.241
        - SVM minCprim: 0.230
        - GMM minCprim: 0.208

    We can see that there might be some overfitting. Therefore we look for a possible optimal solution. Repeating the analysis,
    but this time training on the whole trainig set and testing on the whole test set.
    """

    """
    ### GAUSSIAN MODELS
    # FULL COVARIANCE
    print("FULL COVARIANCE")
    for nPCA in [3, 4, 5, 6]:
        if nPCA == 6:
            data_train = features
            data_test = features_test
        else:
            data_train = dr.PCA(features, nPCA)
            data_test = dr.PCA(features_test, nPCA)
        minDCFList = []
        for pi in [0.5, 0.1]:
            MVG = gc.multivariate_cl([n_class0/n_tot, n_class1/n_tot])
            MVG.train(data_train, labels)
            predictedMVG = MVG.transform(data_test)
            scoresMVG = MVG.get_scores()
            minDCF = val.min_DCF(scoresMVG, pi, 1, 1, labels_test, predictedMVG)
            minDCFList.append(minDCF)
        minCprim = (minDCFList[0] + minDCFList[1])/2
        print("MVG PCA", nPCA, ", minCprim:", minCprim)

    # NAIVE
    print("NAIVE")
    for nPCA in [3, 4, 5, 6]:
        if nPCA == 6:
            data_train = features
            data_test = features_test
        else:
            data_train = dr.PCA(features, nPCA)
            data_test = dr.PCA(features_test, nPCA)
        minDCFList = []
        for pi in [0.5, 0.1]:
            MVG = gc.naive_multivariate_cl([n_class0/n_tot, n_class1/n_tot])
            MVG.train(data_train, labels)
            predictedMVG = MVG.transform(data_test)
            scoresMVG = MVG.get_scores()
            minDCF = val.min_DCF(scoresMVG, pi, 1, 1, labels_test, predictedMVG)
            minDCFList.append(minDCF)
        minCprim = (minDCFList[0] + minDCFList[1])/2
        print("NAIVE PCA", nPCA, ", minCprim:", minCprim)

    # TIED
    print("TIED")
    for nPCA in [3, 4, 5, 6]:
        if nPCA == 6:
            data_train = features
            data_test = features_test
        else:
            data_train = dr.PCA(features, nPCA)
            data_test = dr.PCA(features_test, nPCA)
        minDCFList = []
        for pi in [0.5, 0.1]:
            MVG = gc.tied_multivariate_cl([n_class0/n_tot, n_class1/n_tot])
            MVG.train(data_train, labels)
            predictedMVG = MVG.transform(data_test)
            scoresMVG = MVG.get_scores()
            minDCF = val.min_DCF(scoresMVG, pi, 1, 1, labels_test, predictedMVG)
            minDCFList.append(minDCF)
        minCprim = (minDCFList[0] + minDCFList[1])/2
        print("TIED PCA", nPCA, ", minCprim:", minCprim)

    # TIED NAIVE
    print("TIED NAIVE")
    for nPCA in [3, 4, 5, 6]:
        if nPCA == 6:
            data_train = features
            data_test = features_test
        else:
            data_train = dr.PCA(features, nPCA)
            data_test = dr.PCA(features_test, nPCA)
        minDCFList = []
        for pi in [0.5, 0.1]:
            MVG = gc.tied_naive_multivariate_cl([n_class0/n_tot, n_class1/n_tot])
            MVG.train(data_train, labels)
            predictedMVG = MVG.transform(data_test)
            scoresMVG = MVG.get_scores()
            minDCF = val.min_DCF(scoresMVG, pi, 1, 1, labels_test, predictedMVG)
            minDCFList.append(minDCF)
        minCprim = (minDCFList[0] + minDCFList[1])/2
        print("TIED NAIVE PCA", nPCA, ", minCprim:", minCprim)
    """

    """
        BEST MVG MODEL ON TEST DATA:
        FULL COVARIANCE
            PCA 3 , minCprim: 0.376
            PCA 4 , minCprim: 0.360
            PCA 5 , minCprim: 0.3244 <--- BEST MODEL
            no PCA , minCprim: 0.325
        NAIVE
            PCA 3 , minCprim: 0.374
            PCA 4 , minCprim: 0.361
            PCA 5 , minCprim: 0.3249
            no PCA , minCprim: 0.353
        TIED
            PCA 3 , minCprim: 0.788
            PCA 4 , minCprim: 0.789
            PCA 5 , minCprim: 0.796
            no PCA , minCprim: 0.798
        TIED NAIVE
            PCA 3 , minCprim: 0.788
            PCA 4 , minCprim: 0.788
            PCA 5 , minCprim: 0.795
            no PCA , minCprim: 0.798

        On the test set it seems that the best configuration is given by the FC+PCA5. But it is merely over 
        naive bayes (best model, but not chosen)
    """

    ### SVM
    ## POLI
    print("POLI")
    for nPCA in [5, 6]:
        if nPCA == 6:
            data_train = features
            data_test = features_test
        else:
            data_train = dr.PCA(features, nPCA)
            data_test = dr.PCA(features_test, nPCA)
        for _k in [1, 10, 100]:
            minCprimLists = []
            for _c in [0.1, 1, 10]:
                minCprimList = []
                for _C in [0.001, 0.01, 0.1, 1, 10, 100]:
                    minDCFList = []
                    for pi in [0.1, 0.5]:
                        SVMC = svmc.SVM('Polinomial', balanced=False, d=2, K=_k, C=_C, c=_c)
                        SVMC.train(data_train, labels)
                        predictedSVM = SVMC.transform(data_test)
                        scoresSVM = SVMC.get_scores()
                        minDCF = val.min_DCF(scoresSVM, 0.5, 1, 1, labels_test, predictedSVM)
                        minDCFList.append(minDCF)
                    minCprim = (minDCFList[0]+minDCFList[1])/2
                    minCprimList.append(minCprim)
                minCprimLists.append(minCprimList)
            