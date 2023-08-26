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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

if __name__ == "__main__":
    start_time = datetime.now()

    #LOADING DATASET
    labels, features = du.load("Train.txt")
    labels_test, features_test = du.load("Test.txt")
    labels_sh = shuffle(labels,random_state=0)
    labels_sh_sh = shuffle(labels_sh,random_state=0)

    features_expanded = du.features_expansion(features)
    features_PCA_5 = dr.PCA(features, 5)

    #BEST MODELS
    QLR = lrc.logReg(10, 0.1, "balanced")
    SVMC = svmc.SVM('RBF', balanced=True, gamma=0.01, K=0.01, C=0.1, piT=0.2)
    GMM = gmm.GMM(2, 32, "mvg", "tied")

    """
    ### DET PLOTS
    # BEST MODELS
    ## Quadratic Logistic Regression NO PCA piT = 0.1, lambda = 10 
    _,_,scoresQLR, _ = val.k_fold_bayes_plot(QLR, features_expanded, labels, 5, (0.5, 1, 1), "QLR")

    ## SVM RBF Balanced gamma = 0.01, C = 0.1, K = 0.01 piT = 0.2 PCA 5
    _,_,scoresSVM, _ = val.k_fold_bayes_plot(SVMC, features_PCA_5, labels, 5, (0.5, 1, 1), "SVM")
    
    ## GMM 2 FC 32 FC - T NO PCA
    _,_,scoresGMM, _ = val.k_fold_bayes_plot(GMM, features, labels, 5, (0.5, 1, 1), "GMM")

    val.get_multi_DET([scoresQLR, scoresSVM, scoresGMM], labels_sh, ["QLR", "SMV", "GMM"], "Best Models")
    """

    """
    Fusion results
        cal_l = 0 pi = 0.5
        QSG minDCF: 0.085 actualDCF: 0.198
        QS minDCF: 0.091 actualDCF: 0.219
        QG minDCF: 0.085 actualDCF: 0.198
        SG minDCF: 0.084 actualDCF: 0.199

        cal_l = 0 pi = 0.1
            QSG minDCF: 0.348 actualDCF: 0.583
            QS minDCF: 0.370 actualDCF: 0.655
            QG minDCF: 0.346 actualDCF: 0.588
            SG minDCF: 0.346 actualDCF: 0.561

        
        QSG minCprim: 0.216 Cprim: 0.390
        QS minCprim: 0.230 Cprim: 0.442
        QG minCprim: 0.216 Cprim: 0.400
        SG minCprim: 0.215 Cprim: 0.380

        BEST MODEL SVM + GMM
    """
    # FUSER 
    FUSER = lrc.logReg(0.1, 0.3, "balanced")
    labels_sh = shuffle(labels,random_state=0)
    labels_sh_sh = shuffle(labels_sh,random_state=0)

    # MODELS 0.5
    _, _, scoresQLR, _ = val.k_fold_bayes_plot(QLR, features_expanded, labels, 5, (0.5, 1, 1), "QLR", False)
    _, _, scoresGMM, _ = val.k_fold_bayes_plot(GMM, features, labels, 5, (0.5, 1, 1), "GMM", False)
    _, _, scoresSVM, _ = val.k_fold_bayes_plot(SVMC, features_PCA_5, labels, 5, (0.5, 1, 1), "SVM", False)
    scoresQGS1 = np.vstack((scoresQLR, scoresGMM, scoresSVM))
    scoresQG1 = np.vstack((scoresQLR, scoresGMM))
    scoresSG1 = np.vstack((scoresSVM, scoresGMM))
    scoresQS1 = np.vstack((scoresQLR, scoresSVM))


    # MODELS 0.1
    _, _, scoresQLR, _ = val.k_fold_bayes_plot(QLR, features_expanded, labels, 5, (0.1, 1, 1), "QLR", False)
    _, _, scoresGMM, _ = val.k_fold_bayes_plot(GMM, features, labels, 5, (0.1, 1, 1), "GMM", False)
    _, _, scoresSVM, _ = val.k_fold_bayes_plot(SVMC, features_PCA_5, labels, 5, (0.1, 1, 1), "SVM", False)
    scoresQGS2 = np.vstack((scoresQLR, scoresGMM, scoresSVM))
    scoresQG2 = np.vstack((scoresQLR, scoresGMM))
    scoresSG2 = np.vstack((scoresSVM, scoresGMM))
    scoresQS2 = np.vstack((scoresQLR, scoresSVM))

    # FUSION
    # - QLR GMM
    aDCFQG1, minDCFQG1, scoresQG1, predictedQG1 = val.k_fold_bayes_plot(FUSER, scoresQG1, labels_sh, 5, (0.5, 1, 1), "FUSER", False)
    aDCFQG2, minDCFQG2, scoresQG2, predictedQG2 = val.k_fold_bayes_plot(FUSER, scoresQG2, labels_sh, 5, (0.1, 1, 1), "FUSER", False)

    # - SVM GMM
    aDCFSG1, minDCFSG1, scoresSG1, predictedSG1 = val.k_fold_bayes_plot(FUSER, scoresSG1, labels_sh, 5, (0.5, 1, 1), "FUSER", False)
    aDCFSG2, minDCFSG2, scoresSG2, predictedSG2 = val.k_fold_bayes_plot(FUSER, scoresSG2, labels_sh, 5, (0.1, 1, 1), "FUSER", False)

    # - QLR SVM
    aDCFQS1, minDCFQS1, scoresQS1, predictedQS1 = val.k_fold_bayes_plot(FUSER, scoresQS1, labels_sh, 5, (0.5, 1, 1), "FUSER", False)
    aDCFQS2, minDCFQS2, scoresQS2, predictedQS2 = val.k_fold_bayes_plot(FUSER, scoresQS2, labels_sh, 5, (0.1, 1, 1), "FUSER", False)

    # - QLR SVM GMM
    aDCFQGS1, minDCFQGS1, scoresQGS1, predictedQGS1 = val.k_fold_bayes_plot(FUSER, scoresQGS1, labels_sh, 5, (0.5, 1, 1), "FUSER", False)
    aDCFQGS2, minDCFQGS2, scoresQGS2, predictedQGS2 = val.k_fold_bayes_plot(FUSER, scoresQGS2, labels_sh, 5, (0.1, 1, 1), "FUSER", False)

    # print minCprim and actualCprim Cprim = (minDCF1 + minDCF2)/2
    print("QSG minCprim: {} Cprim: {}".format((minDCFQGS1 + minDCFQGS2)/2, (aDCFQGS1 + aDCFQGS2)/2))
    print("QS minCprim: {} Cprim: {}".format((minDCFQS1 + minDCFQS2)/2, (aDCFQS1 + aDCFQS2)/2))
    print("QG minCprim: {} Cprim: {}".format((minDCFQG1 + minDCFQG2)/2, (aDCFQG1 + aDCFQG2)/2))
    print("SG minCprim: {} Cprim: {}".format((minDCFSG1 + minDCFSG2)/2, (aDCFSG1 + aDCFSG2)/2))

    # PLOTS best model
    val.get_error_plot_Cprim(scoresQG1, scoresQG2, 1, 1, labels_sh_sh, predictedQG1, predictedQG2, "FUSER")

    """
        QSG minCprim: 0.22012937595129378 Cprim: 0.27794520547945206
        QS minCprim: 0.23437975646879758 Cprim: 0.4345554287163876
        QG minCprim: 0.21940512430238457 Cprim: 0.2821657787924911
        SG minCprim: 0.21869165398274987 Cprim: 0.28917681380010146
    """
    end_time = datetime.now()
    print("--------- TIME ----------")
    print(f"Time elapsed: {end_time - start_time}")

