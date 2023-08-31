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
from itertools import repeat
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

if __name__ == "__main__":
    start_time = datetime.now()

    #LOADING DATASET
    labels, features = du.load(".\Data\Train.txt")
    labels_test, features_test = du.load(".\Data\Test.txt")
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
    ### FUSION 
    # FUSER 
    FUSER1 = lrc.logReg(0.1, 0.3, "balanced")
    FUSER2 = lrc.logReg(0.1, 0.6, "balanced")
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
    aDCFQG1, minDCFQG1, scoresQG1, predictedQG1 = val.k_fold_bayes_plot(FUSER1, scoresQG1, labels_sh, 5, (0.5, 1, 1), "FUSER", False)
    aDCFQG2, minDCFQG2, scoresQG2, predictedQG2 = val.k_fold_bayes_plot(FUSER2, scoresQG2, labels_sh, 5, (0.1, 1, 1), "FUSER", False)

    # - SVM GMM
    aDCFSG1, minDCFSG1, scoresSG1, predictedSG1 = val.k_fold_bayes_plot(FUSER1, scoresSG1, labels_sh, 5, (0.5, 1, 1), "FUSER", False)
    aDCFSG2, minDCFSG2, scoresSG2, predictedSG2 = val.k_fold_bayes_plot(FUSER2, scoresSG2, labels_sh, 5, (0.1, 1, 1), "FUSER", False)

    # - QLR SVM
    aDCFQS1, minDCFQS1, scoresQS1, predictedQS1 = val.k_fold_bayes_plot(FUSER1, scoresQS1, labels_sh, 5, (0.5, 1, 1), "FUSER", False)
    aDCFQS2, minDCFQS2, scoresQS2, predictedQS2 = val.k_fold_bayes_plot(FUSER2, scoresQS2, labels_sh, 5, (0.1, 1, 1), "FUSER", False)

    # - QLR SVM GMM
    aDCFQGS1, minDCFQGS1, scoresQGS1, predictedQGS1 = val.k_fold_bayes_plot(FUSER1, scoresQGS1, labels_sh, 5, (0.5, 1, 1), "FUSER", False)
    aDCFQGS2, minDCFQGS2, scoresQGS2, predictedQGS2 = val.k_fold_bayes_plot(FUSER2, scoresQGS2, labels_sh, 5, (0.1, 1, 1), "FUSER", False)

    # print minDCF and actualDCF 1
    print("p 0.5 QSG minDCF: {} actualDCF: {}".format(minDCFQGS1, aDCFQGS1))
    print("p 0.5 QS minDCF: {} actualDCF: {}".format(minDCFQS1, aDCFQS1))
    print("p 0.5 QG minDCF: {} actualDCF: {}".format(minDCFQG1, aDCFQG1))
    print("p 0.5 SG minDCF: {} actualDCF: {}".format(minDCFSG1, aDCFSG1))

    # print minDCF and actualDCF 2
    print("p 0.1 QSG minDCF: {} actualDCF: {}".format(minDCFQGS2, aDCFQGS2))
    print("p 0.1 QS minDCF: {} actualDCF: {}".format(minDCFQS2, aDCFQS2))
    print("p 0.1 QG minDCF: {} actualDCF: {}".format(minDCFQG2, aDCFQG2))
    print("p 0.1 SG minDCF: {} actualDCF: {}".format(minDCFSG2, aDCFSG2))

    # print minCprim and actualCprim 
    print("QSG minCprim: {} Cprim: {}".format((minDCFQGS1 + minDCFQGS2)/2, (aDCFQGS1 + aDCFQGS2)/2))
    print("QS minCprim: {} Cprim: {}".format((minDCFQS1 + minDCFQS2)/2, (aDCFQS1 + aDCFQS2)/2))
    print("QG minCprim: {} Cprim: {}".format((minDCFQG1 + minDCFQG2)/2, (aDCFQG1 + aDCFQG2)/2))
    print("SG minCprim: {} Cprim: {}".format((minDCFSG1 + minDCFSG2)/2, (aDCFSG1 + aDCFSG2)/2))

    # PLOTS best model
    val.get_error_plot_Cprim(scoresQGS1, scoresQGS2, 1, 1, labels_sh_sh, predictedQG1, predictedQG2, "FUSER QGS")
    val.get_error_plot_Cprim(scoresQS1, scoresQS2, 1, 1, labels_sh_sh, predictedQS1, predictedQS2, "FUSER QS")
    val.get_error_plot_Cprim(scoresQG1, scoresQG2, 1, 1, labels_sh_sh, predictedQG1, predictedQG2, "FUSER QG")
    val.get_error_plot_Cprim(scoresSG1, scoresSG2, 1, 1, labels_sh_sh, predictedSG1, predictedSG2, "FUSER SG")
    

        # p 0.5 QSG minDCF: 0.08664003044140031 actualDCF: 0.08816210045662101
        # p 0.5 QS minDCF: 0.1012138508371385 actualDCF: 0.10584601725012684
        # p 0.5 QG minDCF: 0.08769152714358194 actualDCF: 0.08866945712836125
        # p 0.5 SG minDCF: 0.08682394723490613 actualDCF: 0.09269152714358193

        # p 0.1 QSG minDCF: 0.35818493150684927 actualDCF: 0.3754794520547945
        # p 0.1 QS minDCF: 0.36754566210045664 actualDCF: 0.417648401826484
        # p 0.1 QG minDCF: 0.3536187214611872 actualDCF: 0.3779794520547945
        # p 0.1 SG minDCF: 0.3505593607305936 actualDCF: 0.35894977168949777

        # QSG minCprim: 0.2224124809741248 Cprim: 0.23182077625570774
        # QS minCprim: 0.23437975646879758 Cprim: 0.26174720953830544
        # QG minCprim: 0.2206551243023846 Cprim: 0.23332445459157786
        # SG minCprim: 0.21869165398274987 Cprim: 0.22582064941653984
    """

    aDCF, minDCF, scores, predicted = val.k_fold_bayes_plot(GMM, features, labels, 5, (0.5, 1, 1), "GMM", False)
    print("minDCF: {} actualDCF: {}".format(minDCF, aDCF))
    
    features_norm = mu.z_score(features)

    aDCFNorm, minDCNorm, scoresNorm, predictedNorm = val.k_fold_bayes_plot(GMM, features_norm, labels, 5, (0.5, 1, 1), "GMM", False)
    print("NORM minDCF: {} actualDCF: {}".format(minDCNorm, aDCFNorm))

    end_time = datetime.now()
    print("--------- TIME ----------")
    print(f"Time elapsed: {end_time - start_time}")

