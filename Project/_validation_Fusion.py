import data_utils as du
import dimensionality_reduction as dr
import validation as val
import math_utils as mu
import logistic_regression_classifiers as lrc
import GMM as gmm
import SVM_classifiers as svmc
from datetime import datetime
import numpy as np
from sklearn.utils import shuffle

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

    with open("fusion.txt", "w") as f:
        for l1 in [0.001, 0.01, 0.1, 1, 10]:
            for l2 in [0.001, 0.01, 0.1, 1, 10]:
                for pi1 in [0.1, 0.17, 0.2, 0.3, 0.4, 0.5]:
                    for pi2 in [0.1, 0.17, 0.2, 0.3, 0.4, 0.5]: 

                        ### FUSION 
                        # FUSER 
                        FUSER1 = lrc.logReg(l1, pi1, "balanced")
                        FUSER2 = lrc.logReg(l2, pi2, "balanced")
                        print("l1: {} l2: {} pi1: {} pi2: {}".format(l1, l2, pi1, pi2))
                        print("l1: {} l2: {} pi1: {} pi2: {}".format(l1, l2, pi1, pi2), file=f)

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
                        print("p 0.5 QSG minDCF: {} actualDCF: {}".format(minDCFQGS1, aDCFQGS1), file=f)
                        print("p 0.5 QS minDCF: {} actualDCF: {}".format(minDCFQS1, aDCFQS1), file=f)
                        print("p 0.5 QG minDCF: {} actualDCF: {}".format(minDCFQG1, aDCFQG1), file=f)
                        print("p 0.5 SG minDCF: {} actualDCF: {}".format(minDCFSG1, aDCFSG1), file=f)

                        # print minDCF and actualDCF 2
                        print("p 0.1 QSG minDCF: {} actualDCF: {}".format(minDCFQGS2, aDCFQGS2), file=f)
                        print("p 0.1 QS minDCF: {} actualDCF: {}".format(minDCFQS2, aDCFQS2), file=f)
                        print("p 0.1 QG minDCF: {} actualDCF: {}".format(minDCFQG2, aDCFQG2), file=f)
                        print("p 0.1 SG minDCF: {} actualDCF: {}".format(minDCFSG2, aDCFSG2), file=f)

                        # print minCprim and actualCprim 
                        print("QSG minCprim: {} Cprim: {}".format((minDCFQGS1 + minDCFQGS2)/2, (aDCFQGS1 + aDCFQGS2)/2), file=f)
                        print("QS minCprim: {} Cprim: {}".format((minDCFQS1 + minDCFQS2)/2, (aDCFQS1 + aDCFQS2)/2), file=f)
                        print("QG minCprim: {} Cprim: {}".format((minDCFQG1 + minDCFQG2)/2, (aDCFQG1 + aDCFQG2)/2), file=f)
                        print("SG minCprim: {} Cprim: {}".format((minDCFSG1 + minDCFSG2)/2, (aDCFSG1 + aDCFSG2)/2), file=f)
                        print("QSG Cprim - minCprim: {}".format(((aDCFQGS1 + aDCFQGS2)/2 - (minDCFQGS1 + minDCFQGS2)/2), file=f))
                        print("QS Cprim - minCprim: {}".format(((aDCFQS1 + aDCFQS2)/2 - (minDCFQS1 + minDCFQS2)/2), file=f))
                        print("QG Cprim - minCprim: {}".format(((aDCFQG1 + aDCFQG2)/2 - (minDCFQG1 + minDCFQG2)/2), file=f))
                        print("SG Cprim - minCprim: {}".format(((aDCFSG1 + aDCFSG2)/2 - (minDCFSG1 + minDCFSG2)/2), file=f))

#BEST FUSION MODEL CHOSEN l1: 0.001 l2: 0.001 pi1: 0.4 pi2: 0.4
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
FUSER = lrc.logReg(0.001, 0.4, "balanced")
# - QLR GMM
aDCFQG1, minDCFQG1, scoresQG1, predictedQG1 = val.k_fold_bayes_plot(FUSER, scoresQG1, labels_sh, 5, (0.5, 1, 1), "FUSERQG", True)
aDCFQG2, minDCFQG2, scoresQG2, predictedQG2 = val.k_fold_bayes_plot(FUSER, scoresQG2, labels_sh, 5, (0.1, 1, 1), "FUSER", False)

# - SVM GMM
aDCFSG1, minDCFSG1, scoresSG1, predictedSG1 = val.k_fold_bayes_plot(FUSER, scoresSG1, labels_sh, 5, (0.5, 1, 1), "FUSERSG", True)
aDCFSG2, minDCFSG2, scoresSG2, predictedSG2 = val.k_fold_bayes_plot(FUSER, scoresSG2, labels_sh, 5, (0.1, 1, 1), "FUSER", False)

# - QLR SVM
aDCFQS1, minDCFQS1, scoresQS1, predictedQS1 = val.k_fold_bayes_plot(FUSER, scoresQS1, labels_sh, 5, (0.5, 1, 1), "FUSERQS", True)
aDCFQS2, minDCFQS2, scoresQS2, predictedQS2 = val.k_fold_bayes_plot(FUSER, scoresQS2, labels_sh, 5, (0.1, 1, 1), "FUSER", False)

# - QLR SVM GMM
aDCFQGS1, minDCFQGS1, scoresQGS1, predictedQGS1 = val.k_fold_bayes_plot(FUSER, scoresQGS1, labels_sh, 5, (0.5, 1, 1), "FUSERQGS", True)
aDCFQGS2, minDCFQGS2, scoresQGS2, predictedQGS2 = val.k_fold_bayes_plot(FUSER, scoresQGS2, labels_sh, 5, (0.1, 1, 1), "FUSER", False)

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
