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

if __name__ == "__main__":
    # LOADING DATASET
    labels, features = du.load(".\Data\Train.txt")
    labels_test, features_test = du.load(".\Data\Test.txt")
    n_class0 = np.sum(labels==0)
    n_class1 = np.sum(labels==1)
    n_tot = n_class0 + n_class1
    
    n_class0_test = np.sum(labels_test==0)
    n_class1_test = np.sum(labels_test==1)
    n_tot_test = n_class0_test + n_class1_test

    print("Training set:")
    print("Class 0: ", n_class0)
    print("Class 1: ", n_class1)
    print("Total: ", n_tot)
    print("P(Class 0): ", n_class0/n_tot)
    print("P(Class 1): ", n_class1/n_tot)

    print("Test set:")
    print("Class 0: ", n_class0_test)
    print("Class 1: ", n_class1_test)
    print("Total: ", n_tot_test)
    print("P(Class 0): ", n_class0_test/n_tot_test)
    print("P(Class 1): ", n_class1_test/n_tot_test)
    
    # DATA MODELLING
    features_exp = du.features_expansion(features) 
    features_PCA_5 = dr.PCA(features, 5)
    features_PCA_5_exp = du.features_expansion(features_PCA_5)

    features_test_exp = du.features_expansion(features_test)
    features_test_PCA_5 = dr.PCA(features_test, 5)
    features_test_PCA_5_exp = du.features_expansion(features_test_PCA_5)
    

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

    ### FUSION MODELS
    # FUSER 
    FUSER = lrc.logReg(0.1, 0.3, "balanced")

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
    
    scoresQGS = np.vstack((scoresQLR, scoresGMM, scoresSVM))
    scoresQG = np.vstack((scoresQLR, scoresGMM))
    scoresSG = np.vstack((scoresSVM, scoresGMM))
    scoresQS = np.vstack((scoresQLR, scoresSVM))

    # FUSION
    FUSER.train(scoresQGS, labels_test)
    predictedQGS = FUSER.transform(scoresQGS)
    scoresQGS = FUSER.get_scores()

    FUSER.train(scoresQG, labels_test)
    predictedQG = FUSER.transform(scoresQG)
    scoresQG = FUSER.get_scores()

    FUSER.train(scoresSG, labels_test)
    predictedSG = FUSER.transform(scoresSG)
    scoresSG = FUSER.get_scores()

    FUSER.train(scoresQS, labels_test)
    predictedQS = FUSER.transform(scoresQS)
    scoresQS = FUSER.get_scores()

    # OBTAINING minCprim
    minDCFQGS1 = val.min_DCF(scoresQGS, 0.5, 1, 1, labels_test, predictedQGS)
    minDCFQGS2 = val.min_DCF(scoresQGS, 0.1, 1, 1, labels_test, predictedQGS)

    minDCFQG1 = val.min_DCF(scoresQG, 0.5, 1, 1, labels_test, predictedQG)
    minDCFQG2 = val.min_DCF(scoresQG, 0.1, 1, 1, labels_test, predictedQG)

    minDCFSG1 = val.min_DCF(scoresSG, 0.5, 1, 1, labels_test, predictedSG)
    minDCFSG2 = val.min_DCF(scoresSG, 0.1, 1, 1, labels_test, predictedSG)

    minDCFQS1 = val.min_DCF(scoresQS, 0.5, 1, 1, labels_test, predictedQS)
    minDCFQS2 = val.min_DCF(scoresQS, 0.1, 1, 1, labels_test, predictedQS)

    minCprimQGS = (minDCFQGS1 + minDCFQGS2)/2
    minCprimQG = (minDCFQG1 + minDCFQG2)/2
    minCprimSG = (minDCFSG1 + minDCFSG2)/2
    minCprimQS = (minDCFQS1 + minDCFQS2)/2

    print("QGS minCprim: ", minCprimQGS)
    print("QG minCprim: ", minCprimQG)
    print("SG minCprim: ", minCprimSG)
    print("QS minCprim: ", minCprimQS)

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

    ### SVM
    # POLI
    with open("SVM_Poli.txt", "w") as f:
        start = datetime.now()
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
                minCprimList = np.zeros((3, 6))
                for ic, _c in enumerate([0.1, 1, 10]):
                    for iC, _C in enumerate([0.001, 0.01, 0.1, 1, 10, 100]):
                        minDCFList = []
                        for pi in [0.1, 0.5]:
                            print("POLI nPCA:", nPCA, "k:", _k, "c:", _c, "C:", _C, "pi:", pi, "time:", datetime.now()-start)
                            SVMC = svmc.SVM('Polinomial', balanced=False, d=2, K=_k, C=_C, c=_c)
                            SVMC.train(data_train, labels)
                            predictedSVM = SVMC.transform(data_test)
                            scoresSVM = SVMC.get_scores()
                            minDCF = val.min_DCF(scoresSVM, pi, 1, 1, labels_test, predictedSVM)
                            print("nPCA:", nPCA, "k:", _k, "c:", _c, "C:", _C, "pi:", pi, "minDCF:", minDCF, file=f)
                            minDCFList.append(minDCF)
                        minCprim = (minDCFList[0]+minDCFList[1])/2
                        print("minCprim:", minCprim, file=f)
                        print("minCprim:", minCprim)
                        minCprimList[ic, iC] = minCprim
                # Save as npy file for later plotting, so in case of errors, there is no need of re-executions
                np.save("minCprimList_Poli_nPCA_{}_k_{}.npy".format(nPCA, _k), minCprimList)

    # load npy files
    C = [0.001, 0.01, 0.1, 1, 10, 100]
    minCprimList_Poli_nPCA_5_k_1 = np.load("figures/evaluation_SVM/Polinomial/minCprimList_Poli_nPCA_5_k_1.npy") 
    minCprimList_Poli_nPCA_5_k_10 = np.load("figures/evaluation_SVM/Polinomial/minCprimList_Poli_nPCA_5_k_10.npy")
    minCprimList_Poli_nPCA_5_k_100 = np.load("figures/evaluation_SVM/Polinomial/minCprimList_Poli_nPCA_5_k_100.npy")
    minCprimList_Poli_nPCA_6_k_1 = np.load("figures/evaluation_SVM/Polinomial/minCprimList_Poli_nPCA_6_k_1.npy")
    minCprimList_Poli_nPCA_6_k_10 = np.load("figures/evaluation_SVM/Polinomial/minCprimList_Poli_nPCA_6_k_10.npy")
    minCprimList_Poli_nPCA_6_k_100 = np.load("figures/evaluation_SVM/Polinomial/minCprimList_Poli_nPCA_6_k_100.npy")

    dv.plotCPrim(C, minCprimList_Poli_nPCA_5_k_1, ["Polinomial(2) SVM c = 0.1", "Polinomial(2) SVM c = 1", "Polinomial(2) SVM c = 10"], "C", "minCprimList_Poli_nPCA_5_k_1")
    dv.plotCPrim(C, minCprimList_Poli_nPCA_5_k_10, ["Polinomial(2) SVM c = 0.1", "Polinomial(2) SVM c = 1", "Polinomial(2) SVM c = 10"], "C", "minCprimList_Poli_nPCA_5_k_10")
    dv.plotCPrim(C, minCprimList_Poli_nPCA_5_k_100, ["Polinomial(2) SVM c = 0.1", "Polinomial(2) SVM c = 1", "Polinomial(2) SVM c = 10"], "C", "minCprimList_Poli_nPCA_5_k_100")
    dv.plotCPrim(C, minCprimList_Poli_nPCA_6_k_1, ["Polinomial(2) SVM c = 0.1", "Polinomial(2) SVM c = 1", "Polinomial(2) SVM c = 10"], "C", "minCprimList_Poli_nPCA_6_k_1")
    dv.plotCPrim(C, minCprimList_Poli_nPCA_6_k_10, ["Polinomial(2) SVM c = 0.1", "Polinomial(2) SVM c = 1", "Polinomial(2) SVM c = 10"], "C", "minCprimList_Poli_nPCA_6_k_10")
    dv.plotCPrim(C, minCprimList_Poli_nPCA_6_k_100, ["Polinomial(2) SVM c = 0.1", "Polinomial(2) SVM c = 1", "Polinomial(2) SVM c = 10"], "C", "minCprimList_Poli_nPCA_6_k_100")

    # RBF
    ### SVM
    with open("SVM_RBF.txt", "w") as f:
        start = datetime.now()
        print("RBF")
        for nPCA in [5, 6]:
            if nPCA == 6:
                data_train = features
                data_test = features_test
            else:
                data_train = dr.PCA(features, nPCA)
                data_test = dr.PCA(features_test, nPCA)
            for _k in [0.01, 0.1, 1]:
                minCprimLists = []
                minCprimList = np.zeros((3, 6))
                for ic, _c in enumerate([0.001, 0.01, 0.1]):
                    for iC, _C in enumerate([0.001, 0.01, 0.1, 1, 10, 100]):
                        minDCFList = []
                        for pi in [0.1, 0.5]:
                            print("POLI nPCA:", nPCA, "k:", _k, "gamma:", _c, "C:", _C, "pi:", pi, "time:", datetime.now()-start)
                            SVMC = svmc.SVM('RBF', balanced=False, K=_k, C=_C, gamma=_c)
                            SVMC.train(data_train, labels)
                            predictedSVM = SVMC.transform(data_test)
                            scoresSVM = SVMC.get_scores()
                            minDCF = val.min_DCF(scoresSVM, pi, 1, 1, labels_test, predictedSVM)
                            print("nPCA:", nPCA, "k:", _k, "gamma:", _c, "C:", _C, "pi:", pi, "minDCF:", minDCF, file=f)
                            minDCFList.append(minDCF)
                        minCprim = (minDCFList[0]+minDCFList[1])/2
                        print("minCprim:", minCprim, file=f)
                        print("minCprim:", minCprim)
                        minCprimList[ic, iC] = minCprim
                # Save as npy file for later plotting, so in case of errors, there is no need of re-executions
                np.save("minCprimList_RBF_nPCA_{}_k_{}.npy".format(nPCA, _k), minCprimList)

    C = [0.001, 0.01, 0.1, 1, 10, 100]
    minCprimList_RBF_nPCA_5_k_001 = np.load("figures/evaluation_SVM/RBF/minCprimList_RBF_nPCA_5_k_0.01.npy") 
    minCprimList_RBF_nPCA_5_k_01 = np.load("figures/evaluation_SVM/RBF/minCprimList_RBF_nPCA_5_k_0.1.npy")
    minCprimList_RBF_nPCA_5_k_1 = np.load("figures/evaluation_SVM/RBF/minCprimList_RBF_nPCA_5_k_1.npy")
    minCprimList_RBF_nPCA_6_k_001 = np.load("figures/evaluation_SVM/RBF/minCprimList_RBF_nPCA_6_k_0.01.npy")
    minCprimList_RBF_nPCA_6_k_01 = np.load("figures/evaluation_SVM/RBF/minCprimList_RBF_nPCA_6_k_0.1.npy")
    minCprimList_RBF_nPCA_6_k_1 = np.load("figures/evaluation_SVM/RBF/minCprimList_RBF_nPCA_6_k_1.npy")

    dv.plotCPrim(C, minCprimList_RBF_nPCA_5_k_001, ["RBF SVM γ = 0.001", "RBF SVM γ = 0.01", "RBF SVM γ = 0.1"], "C", "minCprimList_RBF_nPCA_5_k_001")
    dv.plotCPrim(C, minCprimList_RBF_nPCA_5_k_01, ["RBF SVM γ = 0.001", "RBF SVM γ = 0.01", "RBF SVM γ = 0.1"], "C", "minCprimList_RBF_nPCA_5_k_01")
    dv.plotCPrim(C, minCprimList_RBF_nPCA_5_k_1, ["RBF SVM γ = 0.001", "RBF SVM γ = 0.01", "RBF SVM γ = 0.1"], "C", "minCprimList_RBF_nPCA_5_k_1")
    dv.plotCPrim(C, minCprimList_RBF_nPCA_6_k_001, ["RBF SVM γ = 0.001", "RBF SVM γ = 0.01", "RBF SVM γ = 0.1"], "C", "minCprimList_RBF_nPCA_6_k_001")
    dv.plotCPrim(C, minCprimList_RBF_nPCA_6_k_01, ["RBF SVM γ = 0.001", "RBF SVM γ = 0.01", "RBF SVM γ = 0.1"], "C", "minCprimList_RBF_nPCA_6_k_01")
    dv.plotCPrim(C, minCprimList_RBF_nPCA_6_k_1, ["RBF SVM γ = 0.001", "RBF SVM γ = 0.01", "RBF SVM γ = 0.1"], "C", "minCprimList_RBF_nPCA_6_k_1")

    # Balanced RBF
    for _piT in [0.17, 0.1, 0.2, 0.5]:
        #RBF 
        print("######## piT = {} #######".format(_piT))
        print("------ RBF -------")
        RBFObj = svmc.SVM('RBF', balanced=True, gamma=0.01, K=0.1, C=1, piT=_piT)
        RBFObj.train(features, labels)
        predictedRBF = RBFObj.transform(features_test)
        scoresRBF = RBFObj.get_scores()
        minDCF5 = val.min_DCF(scoresRBF, 0.5, 1, 1, labels_test, predictedRBF)
        minDCF1 = val.min_DCF(scoresRBF, 0.1, 1, 1, labels_test, predictedRBF)
        print("minDCF 0.5 RBF: ", minDCF5)
        print("minDCF 0.1 RBF: ", minDCF1)
        print("Cprim RBF:", (minDCF5+minDCF1)/2)
  


    ### Log Reg
    lambdas = np.logspace(-2, 3, num=30)
    with open("QLogReg.txt", "w") as f:
        start = datetime.now()
        CprimList = np.zeros((2, len(lambdas)))
        for nPCA in [5, 6]:
            if nPCA == 6:
                data_train = features_exp
                data_test = features_test_exp
            else:
                data_train = features_PCA_5_exp
                data_test = features_test_PCA_5_exp
            for il, _lambda in enumerate(lambdas):
                QLR = lrc.logReg(_lambda, 0.17, "balanced")
                QLR.train(data_train, labels)
                predictedQLR = QLR.transform(data_test)
                scoresQLR = QLR.get_scores()
                minDCF1 = val.min_DCF(scoresQLR, 0.5, 1, 1, labels_test, predictedQLR)
                minDCF2 = val.min_DCF(scoresQLR, 0.1, 1, 1, labels_test, predictedQLR)
                minCprim = (minDCF1 + minDCF2)/2
                print("nPCA:", nPCA, "lambda:", _lambda, "minCprim:", minCprim, file=f)
                print("nPCA:", nPCA, "lambda:", _lambda, "minCprim:", minCprim)
                CprimList[nPCA-5, il] = minCprim
        np.save("CprimList_QLogReg.npy", CprimList)

    # load npy files
    lambdas = np.logspace(-2, 3, num=30)
    CprimList_QLogReg = np.load("figures/evaluation_QLogReg/CprimList_QLogReg.npy")

    dv.plotCPrim(lambdas, CprimList_QLogReg, ["LogReg PCA 5", "LogReg PCA 6"], "lambda", "CprimList_QLogReg")
    
    print("Working on piT")
    lambdaBest=0.1
    for piT in [0.1,0.17,0.2,0.5]:
        CprimList=[]
        for piTilde in [0.1,0.5]:
            logRegObj = lrc.logReg(lambdaBest, piT, "balanced")
            logRegObj.train(features_exp, labels)
            predictedLogReg = logRegObj.transform(features_test_exp)
            scoresLogReg = logRegObj.get_scores()
            minDCF = val.min_DCF(scoresLogReg, piTilde, 1, 1, labels_test, predictedLogReg)
            print("Quadratic LogReg Balanced Not Normalized, minDCF with piT {} and piTilde {} no PCA and  is {}".format(piT, piTilde, minDCF))
            CprimList.append(minDCF)
        Cprim=np.array(CprimList).mean(axis=0)
        print("Quadratic LogReg Balanced Not Normalized, Cprim with piT {}  no PCA and  is {}".format(piT,Cprim ))


    ### FUSION WITH OPTIMAL CONFIGURATIONS
    # Q-Log-Reg
    QLR = lrc.logReg(0.1, 0.17, "balanced")
    
    # SVM
    SVMC = svmc.SVM('RBF', balanced=True, gamma=0.01, K=0.1, C=1, piT=0.1)

    # GMM
    GMM = gmm.GMM(2, 32, "diagonal", "tied diagonal")

    # TRAINING
    featuresPCA = dr.PCA(features, 5)
    features_testPCA = dr.PCA(features_test, 5)
    QLR.train(features_exp, labels)
    SVMC.train(featuresPCA, labels)
    GMM.train(features, labels)

    # TRANSFORM
    predictedQLR = QLR.transform(features_test_exp)
    predicetedSVM = SVMC.transform(features_testPCA)
    predictedGMM = GMM.transform(features_test)

    # OBTAINING SCORES
    scoresQLR = QLR.get_scores()
    scoresSVM = SVMC.get_scores()
    scoresGMM = GMM.get_scores()

    # FUSER
    FUSER = lrc.logReg(0.001, 0.4, "balanced")

    # FUSION
    scoresQGS = np.vstack((scoresQLR, scoresGMM, scoresSVM))
    scoresQG = np.vstack((scoresQLR, scoresGMM))
    scoresSG = np.vstack((scoresSVM, scoresGMM))
    scoresQS = np.vstack((scoresQLR, scoresSVM))

    FUSER.train(scoresQGS, labels_test)
    predictedQGS = FUSER.transform(scoresQGS)
    scoresQGS = FUSER.get_scores()

    FUSER.train(scoresQG, labels_test)
    predictedQG = FUSER.transform(scoresQG)
    scoresQG = FUSER.get_scores()

    FUSER.train(scoresSG, labels_test)
    predictedSG = FUSER.transform(scoresSG)
    scoresSG = FUSER.get_scores()

    FUSER.train(scoresQS, labels_test)
    predictedQS = FUSER.transform(scoresQS)
    scoresQS = FUSER.get_scores()

    # OBTAINING minCprim
    minDCFQGS1 = val.min_DCF(scoresQGS, 0.5, 1, 1, labels_test, predictedQGS)
    minDCFQGS2 = val.min_DCF(scoresQGS, 0.1, 1, 1, labels_test, predictedQGS)
    actDCFQGS1 = val.act_DCF(scoresQGS, 0.1, 1, 1, labels_test, None)
    actDCFQGS2 = val.act_DCF(scoresQGS, 0.5, 1, 1, labels_test, None)

    minDCFQG1 = val.min_DCF(scoresQG, 0.5, 1, 1, labels_test, predictedQG)
    minDCFQG2 = val.min_DCF(scoresQG, 0.1, 1, 1, labels_test, predictedQG)
    actDCFQG1 = val.act_DCF(scoresQG, 0.1, 1, 1, labels_test, None)
    actDCFQG2 = val.act_DCF(scoresQG, 0.5, 1, 1, labels_test, None)

    minDCFSG1 = val.min_DCF(scoresSG, 0.5, 1, 1, labels_test, predictedSG)
    minDCFSG2 = val.min_DCF(scoresSG, 0.1, 1, 1, labels_test, predictedSG)
    actDCFSG1 = val.act_DCF(scoresSG, 0.1, 1, 1, labels_test, None)
    actDCFSG2 = val.act_DCF(scoresSG, 0.5, 1, 1, labels_test, None)

    minDCFQS1 = val.min_DCF(scoresQS, 0.5, 1, 1, labels_test, predictedQS)
    minDCFQS2 = val.min_DCF(scoresQS, 0.1, 1, 1, labels_test, predictedQS)
    actDCFQS1 = val.act_DCF(scoresQS, 0.1, 1, 1, labels_test, None)
    actDCFQS2 = val.act_DCF(scoresQS, 0.5, 1, 1, labels_test, None)

    minCprimQGS = (minDCFQGS1 + minDCFQGS2)/2
    minCprimQG = (minDCFQG1 + minDCFQG2)/2
    minCprimSG = (minDCFSG1 + minDCFSG2)/2
    minCprimQS = (minDCFQS1 + minDCFQS2)/2

    CprimQGS = (actDCFQGS1 + actDCFQGS2)/2
    CprimQG = (actDCFQG1 + actDCFQG2)/2
    CprimSG = (actDCFSG1 + actDCFSG2)/2
    CprimQS = (actDCFQS1 + actDCFQS2)/2

    print("QGS minCprim: ", minCprimQGS)
    print("QG minCprim: ", minCprimQG)
    print("SG minCprim: ", minCprimSG)
    print("QS minCprim: ", minCprimQS)
    print("QGS Cprim: ", CprimQGS)
    print("QG Cprim: ", CprimQG)
    print("SG Cprim: ", CprimSG)
    print("QS Cprim: ", CprimQS)



    # Balanced RBF
    for _piT in [0.17, 0.1, 0.2, 0.5]:
        #RBF 
        print("######## piT = {} #######".format(_piT))
        print("------ RBF -------")
        RBFObj = svmc.SVM('RBF', balanced=True, gamma=0.01, K=0.01, C=1, piT=_piT)
        RBFObj.train(features, labels)
        predictedRBF = RBFObj.transform(features_test)
        scoresRBF = RBFObj.get_scores()
        minDCF5 = val.min_DCF(scoresRBF, 0.5, 1, 1, labels_test, predictedRBF)
        minDCF1 = val.min_DCF(scoresRBF, 0.1, 1, 1, labels_test, predictedRBF)
        print("minDCF 0.5 RBF: ", minDCF5)
        print("minDCF 0.1 RBF: ", minDCF1)
        print("Cprim RBF:", (minDCF5+minDCF1)/2)

# GMM
    GMM1 = gmm.GMM(2, 32, "MVG", "MVG")
    GMM2 = gmm.GMM(2, 32, "MVG", "diagonal")
    GMM3 = gmm.GMM(2, 32, "MVG", "tied")
    GMM4 = gmm.GMM(2, 32, "MVG", "tied diagonal")

    GMM5 = gmm.GMM(2, 32, "diagonal", "MVG")
    GMM6 = gmm.GMM(2, 32, "diagonal", "diagonal")
    GMM7 = gmm.GMM(2, 32, "diagonal", "tied")
    GMM8 = gmm.GMM(2, 32, "diagonal", "tied diagonal")

    # TRAINING
    GMM1.train(features, labels)
    GMM2.train(features, labels)
    GMM3.train(features, labels)
    GMM4.train(features, labels)
    GMM5.train(features, labels)
    GMM6.train(features, labels)
    GMM7.train(features, labels)
    GMM8.train(features, labels)

    # TRANSFORM
    predictedGMM1 = GMM1.transform(features_test)
    predictedGMM2 = GMM2.transform(features_test)
    predictedGMM3 = GMM3.transform(features_test)
    predictedGMM4 = GMM4.transform(features_test)
    predictedGMM5 = GMM5.transform(features_test)
    predictedGMM6 = GMM6.transform(features_test)
    predictedGMM7 = GMM7.transform(features_test)
    predictedGMM8 = GMM8.transform(features_test)

    # OBTAINING SCORES
    scoresGMM1 = GMM1.get_scores()
    scoresGMM2 = GMM2.get_scores()
    scoresGMM3 = GMM3.get_scores()
    scoresGMM4 = GMM4.get_scores()
    scoresGMM5 = GMM5.get_scores()
    scoresGMM6 = GMM6.get_scores()
    scoresGMM7 = GMM7.get_scores()
    scoresGMM8 = GMM8.get_scores()

    # OBTAINING minCprim
    minDCF_05_1 = val.min_DCF(scoresGMM1, 0.5, 1, 1, labels_test, predictedGMM1)
    minDCF_01_1 = val.min_DCF(scoresGMM1, 0.1, 1, 1, labels_test, predictedGMM1)
    minCprim1 = (minDCF_05_1 + minDCF_01_1)/2

    minDCF_05_2 = val.min_DCF(scoresGMM2, 0.5, 1, 1, labels_test, predictedGMM2)
    minDCF_01_2 = val.min_DCF(scoresGMM2, 0.1, 1, 1, labels_test, predictedGMM2)
    minCprim2 = (minDCF_05_2 + minDCF_01_2)/2

    minDCF_05_3 = val.min_DCF(scoresGMM3, 0.5, 1, 1, labels_test, predictedGMM3)
    minDCF_01_3 = val.min_DCF(scoresGMM3, 0.1, 1, 1, labels_test, predictedGMM3)
    minCprim3 = (minDCF_05_3 + minDCF_01_3)/2

    minDCF_05_4 = val.min_DCF(scoresGMM4, 0.5, 1, 1, labels_test, predictedGMM4)
    minDCF_01_4 = val.min_DCF(scoresGMM4, 0.1, 1, 1, labels_test, predictedGMM4)
    minCprim4 = (minDCF_05_4 + minDCF_01_4)/2

    minDCF_05_5 = val.min_DCF(scoresGMM5, 0.5, 1, 1, labels_test, predictedGMM5)
    minDCF_01_5 = val.min_DCF(scoresGMM5, 0.1, 1, 1, labels_test, predictedGMM5)
    minCprim5 = (minDCF_05_5 + minDCF_01_5)/2

    minDCF_05_6 = val.min_DCF(scoresGMM6, 0.5, 1, 1, labels_test, predictedGMM6)
    minDCF_01_6 = val.min_DCF(scoresGMM6, 0.1, 1, 1, labels_test, predictedGMM6)
    minCprim6 = (minDCF_05_6 + minDCF_01_6)/2

    minDCF_05_7 = val.min_DCF(scoresGMM7, 0.5, 1, 1, labels_test, predictedGMM7)
    minDCF_01_7 = val.min_DCF(scoresGMM7, 0.1, 1, 1, labels_test, predictedGMM7)
    minCprim7 = (minDCF_05_7 + minDCF_01_7)/2

    minDCF_05_8 = val.min_DCF(scoresGMM8, 0.5, 1, 1, labels_test, predictedGMM8)
    minDCF_01_8 = val.min_DCF(scoresGMM8, 0.1, 1, 1, labels_test, predictedGMM8)
    minCprim8 = (minDCF_05_8 + minDCF_01_8)/2  

    print("MVG - MVG minDCF_05: {}, minDCF_01: {}, minCprim: {}".format(minDCF_05_1, minDCF_01_1, minCprim1))
    print("MVG - diagonal minDCF_05: {}, minDCF_01: {}, minCprim: {}".format(minDCF_05_2, minDCF_01_2, minCprim2))
    print("MVG - tied minDCF_05: {}, minDCF_01: {}, minCprim: {}".format(minDCF_05_3, minDCF_01_3, minCprim3))
    print("MVG - tied diagonal minDCF_05: {}, minDCF_01: {}, minCprim: {}".format(minDCF_05_4, minDCF_01_4, minCprim4))
    print("diagonal - MVG minDCF_05: {}, minDCF_01: {}, minCprim: {}".format(minDCF_05_5, minDCF_01_5, minCprim5))
    print("diagonal - diagonal minDCF_05: {}, minDCF_01: {}, minCprim: {}".format(minDCF_05_6, minDCF_01_6, minCprim6))
    print("diagonal - tied minDCF_05: {}, minDCF_01: {}, minCprim: {}".format(minDCF_05_7, minDCF_01_7, minCprim7))
    print("diagonal - tied diagonal minDCF_05: {}, minDCF_01: {}, minCprim: {}".format(minDCF_05_8, minDCF_01_8, minCprim8))

