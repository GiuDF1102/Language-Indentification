import data_utils as du
import data_visualization as dv
import dimensionality_reduction as dr
import gaussian_classifiers as gc
import validation as val
import math_utils as mu
import logistic_regression_classifiers as lrc
import SVM_classifiers as svmc
from datetime import datetime
import numpy as np

if __name__=="__main__":
    start_time = datetime.now()

    #LOADING DATASET
    labels, features = du.load("..\PROJECTS\Language_detection\Train.txt")
    labels_test, features_test = du.load("..\PROJECTS\Language_detection\Test.txt")

    #DICRIONARIES
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

    """#MEANS AND VARIANCE
    no_reduction_means = mu.calcmean_classes(features, labels)
    no_reduction_variance = mu.calcvariance_classes(features, labels)
    """
    
    #PCA DATA
    PCA_5 = dr.PCA(features,5)
    PCA_5_TEST = dr.PCA(features_test,5)


    #QUADRATIC FEATURES FOR REGRESSION
    featuresTrainQuadratic = du.features_expansion(features)
    featuresTestQuadratic = du.features_expansion(features_test)
    featuresTrainQuadraticPCA = du.features_expansion(PCA_5)
    featuresTestQuadraticPCA = du.features_expansion(PCA_5_TEST)

    """
    #PRINTING SCATTERPLOTS
    dv.get_scatter(features,labels,labels_dict, features_dict)
    #dv.get_scatter(PCA_5,labels,labels_dict, features_dict_PCA)

    #PRINTING HISTOGRAMS
    dv.get_hist(features,labels,labels_dict, features_dict)
    #dv.get_hist(PCA_5,labels,labels_dict, features_dict_PCA)
    
    #CORRELATION MATRICES
    dv.calc_correlation_matrix(features, "Dataset")
    dv.calc_correlation_matrix(features.T[ labels == 1].T, "Dataset Italian")
    dv.calc_correlation_matrix(features.T[ labels == 0].T, "Dataset not Italian")

    #GAUSSIAN CLASSIFIERS
    mvg_cl = gc.multivariate_cl()
    mean, C = mvg_cl.fit(features, labels)
    predicted_mvg = mvg_cl.trasform(features_test, mean, C)
    mean_PCA, C_PCA = mvg_cl.fit(PCA_5, labels)
    predicted_mvg_PCA = mvg_cl.trasform(PCA_5_TEST, mean_PCA, C_PCA)

    tied_cl = gc.tied_multivariate_cl()
    mean, C = tied_cl.fit(features, labels)
    predicted_tied = tied_cl.trasform(features_test, mean, C)
    mean_PCA, C_PCA = tied_cl.fit(PCA_5, labels)
    predicted_tied_PCA = tied_cl.trasform(PCA_5_TEST, mean_PCA, C_PCA)

    naive_cl = gc.naive_multivariate_cl()
    mean, C = naive_cl.fit(features, labels)
    predicted_naive = naive_cl.trasform(features_test, mean, C)
    mean_PCA, C_PCA = naive_cl.fit(PCA_5, labels)
    predicted_naive_PCA = naive_cl.trasform(PCA_5_TEST, mean_PCA, C_PCA)
    """

    # tied_naive_cl = gc.tied_naive_multivariate_cl()
    # tied_naive_cl.train(features, labels)
    # predicted_tied_naive = tied_naive_cl.trasform(features_test)
    # mean_PCA, C_PCA = tied_naive_cl.fit(PCA_5, labels)
    # predicted_tied_naive_PCA = tied_naive_cl.trasform(PCA_5_TEST, mean_PCA, C_PCA)

    #LOGISTIC REGRESSION
    # logQuad =lrc.logReg(featuresTrainQuadratic,labels,0.001)
    # logQuad.train()
    # predicted_qlr = logQuad.transform(featuresTestQuadratic,0)
    # logQuadPCA =lrc.logReg(featuresTrainQuadraticPCA,labels,0.001)
    # logQuadPCA.train()
    # predicted_qlr_PCA = lrc.transform(featuresTestQuadraticPCA,w, b,0)

    # SVM
    # svm_lin = svmc.SVM('linear')
    # conf_matr = val.k_fold_SVM_linear(svm_lin, features, labels, 5, 0.1, 1)
    # print(conf_matr.get_confusion_matrix())

    """     #KFOLD 
    # - GAUSSIAN CLASSIFIERS
    learners = [gc.multivariate_cl(),gc.naive_multivariate_cl(), gc.tied_multivariate_cl(), gc.tied_naive_multivariate_cl()]
    accuracies = val.k_fold(learners, features, labels, len(labels))
    accuracies_PCA = val.k_fold(learners, features, labels, len(labels))  """

    # - LOGISTIC REGRESSION
    # pi1 = 0.5
    # C1 = np.array([[0, 1], [1, 0]]) 
    # pi2 = 0.1
    # C2 = np.array([[0, 1], [1, 0]]) 
    # ConfMatrClass = val.confusion_matrix(labels_test, predicted_qlr)
    # ConfMatr = ConfMatrClass.get_confusion_matrix()
    # FNR, FPR = ConfMatrClass.FNR_FPR_binary()
    # dcf = ConfMatrClass.DCF_binary(pi1, C1)
    # normDCF = ConfMatrClass.DCF_binary_norm(pi1, C1)
    # minDCF1, bestThresh1 = val.min_DCF(scores, labels_test, pi1, C1)
    # minDCF2, bestThresh2 = val.min_DCF(scores, labels_test, pi2, C2)

    # logQuad =lrc.logReg(featuresTrainQuadratic,labels,0.001)
    # w,b=logQuad.train()
    
    # print("--------- Logistic Regression  ---------- ")
    # print(f"FNR: {FNR}, FPR: {FPR}")
    # print("Confusion Matrix: \n{}".format(ConfMatr))
    # print("DCF: {}".format(dcf))
    # print("NormDCF: {}".format(normDCF))
    # print("minDCF1: {}".format(minDCF1))
    # print("bestThreshold1: {}".format(bestThresh1))
    # print("minDCF2: {}".format(minDCF2))
    # print("bestThreshold2: {}".format(bestThresh2))

    # print("Logistic Regression Qudratic with optimal threshold - prior=0.5")
    # (predicted_qlr,scores) = lrc.transform(featuresTestQuadratic,w, b,bestThresh1)
    # ConfMatrClass = val.confusion_matrix(labels_test, predicted_qlr)
    # ConfMatr = ConfMatrClass.get_confusion_matrix()
    # val.get_ROC(scores, labels_test, pi1, C1, "Logistic Regression Quadratic - prior=0.5") 
    # val.get_error_plot(scores, labels_test, C1, "Logistic Regression Quadratic - prior=0.5")
    # print("Confusion Matrix: \n{}".format(ConfMatr))

    # print("Logistic Regression Qudratic with optimal threshold - prior=0.1")
    # (predicted_qlr,scores) = lrc.transform(featuresTestQuadratic,w, b,bestThresh2)
    # ConfMatrClass = val.confusion_matrix(labels_test, predicted_qlr)
    # ConfMatr = ConfMatrClass.get_confusion_matrix()
    # val.get_ROC(scores, labels_test, pi2, C2, "Logistic Regression Quadratic - prior=0.1")
    # val.get_error_plot(scores, labels_test, C2, "Logistic Regression Quadratic - prior=0.1") 
    # print("Confusion Matrix: \n{}".format(ConfMatr))


    """     mvg_cl = gc.multivariate_cl)
    mvg_cl.fit(features, labels)
    predicted_mvg = mvg_cl.trasform(features_test)
    
    ConfMatrClass = val.confusion_matrix(labels_test, predicted_mvg)
    ConfMatr = ConfMatrClass.get_confusion_matrix()
    FNR, FPR = ConfMatrClass.FNR_FPR_binary()
    dcf = ConfMatrClass.DCF_binary(pi1, C1)
    normDCF = ConfMatrClass.DCF_binary_norm(pi1, C1)

    print("--------- GAU MVG FPR/FNR  ---------- ")
    print(f"FNR: {FNR}, FPR: {FPR}")
    print("Confusion Matrix: \n{}".format(ConfMatr))
    print("DCF: {}".format(dcf))
    print("NormDCF: {}".format(normDCF)) """

    """     #PRINTING RESULTS
    print("--------- Data Information ---------- ")
    print(" Number of italian samples: {}".format((labels == 1).sum()))
    print(" Number of not italian samples: {}".format((labels == 0).sum()))
    print(" Means italian:")
    for i in range(features.shape[0]):
        print(" - Feature {}: {:.6f}".format(i,no_reduction_means[0][i]))
    print(" Means not italian:")
    for i in range(features.shape[0]):
        print(" - Feature {}: {:.6f}".format(i,no_reduction_means[1][i]))
    print(" Variance italian:")
    for i in range(features.shape[0]):
        print(" - Feature {}: {:.6f}".format(i,no_reduction_variance[0][i]))
    print(" Variance not italian:")
    for i in range(features.shape[0]):
        print(" - Feature {}: {:.6f}".format(i,no_reduction_variance[1][i]))
 
    print("--------- CLASSIFIERS ACCURACY ----------")
    print(f"Multivarate: {round(val.calc_accuracy(labels_test, predicted_mvg)*100,2)}%")
    print(f"Tied Multivarate: {round(val.calc_accuracy(labels_test, predicted_tied)*100,2)}%")
    print(f"Naive Multivarate: {round(val.calc_accuracy(labels_test, predicted_naive)*100,2)}%")
    print(f"Tied Naive Multivarate: {round(val.calc_accuracy(labels_test, predicted_tied_naive)*100,2)}%")
    print(f"Logistic Regression: {round(val.calc_accuracy(labels_test,predicted_qlr)*100,2)}%")
    print(f"Multivarate +  PCA_5_DIM: {round(val.calc_accuracy(labels_test, predicted_mvg_PCA)*100,2)}%")
    print(f"Tied Multivarate + PCA_5_DIM: {round(val.calc_accuracy(labels_test, predicted_tied_PCA)*100,2)}%")
    print(f"Naive Multivarate +  PCA_5_DIM: {round(val.calc_accuracy(labels_test, predicted_naive_PCA)*100,2)}%")
    print(f"Tied Naive Multivarate +  PCA_5_DIM: {round(val.calc_accuracy(labels_test, predicted_tied_naive_PCA)*100,2)}%")
    print(f"Logistic Regression +  PCA_5_DIM: {round(val.calc_accuracy(labels_test,predicted_qlr_PCA)*100,2)}%")


    print("--------- KFOLD ----------")
    print("Gaussian Classifiers")
    for i in range(len(accuracies)):
        print(f" - {learners[i].name}: {round(accuracies[i],2)}%")    
    for i in range(len(accuracies_PCA)):
        print(f" - {learners[i].name} + PCA: {round(accuracies_PCA[i],2)}%")
    """

    #K-folds

    #scoresMVG = val.k_fold_Gaussian(gc.multivariate_cl([pi1, 1-pi1]), features, labels, 5, "MVG")
    # scoresTied = val.k_fold_Gaussian(gc.tied_multivariate_cl(priors=[0.5, 0.5]), features, labels, 5, "Tied")
    # scoresNaive = val.k_fold_Gaussian(gc.naive_multivariate_cl(priors=[0.5, 0.5]), features, labels, 5, "Naive")
    # scoresTiedNaive = val.k_fold_Gaussian(gc.tied_naive_multivariate_cl(priors=[0.5, 0.5]), features, labels, 5, "Tied Naive")
    
    #minDCFMVG_P1, bestThresholdMVG_P1 = val.min_DCF(scoresMVG, labels, pi1, C)
    #print("MinDCF MVG: {}".format(minDCFMVG_P1))
    # minDCFTied_P1, bestThresholdTied_P1 = val.min_DCF(scoresTied, labels, pi1, C)
    # minDCFNaive_P1, bestThresholdNaive_P1 = val.min_DCF(scoresNaive, labels, pi1, C)
    # minDCFTiedNaive_P1, bestThresholdTiedNaive_P1 = val.min_DCF(scoresTiedNaive, labels, pi1, C)

    # minDCFMVG_P2, bestThresholdMVG_P2 = val.min_DCF(scoresMVG, labels, pi2, C)
    # minDCFTied_P2, bestThresholdTied_P2 = val.min_DCF(scoresTied, labels, pi2, C)
    # minDCFNaive_P2, bestThresholdNaive_P2 = val.min_DCF(scoresNaive, labels, pi2, C)
    # minDCFTiedNaive_P2, bestThresholdTiedNaive_P2 = val.min_DCF(scoresTiedNaive, labels, pi2, C)

    # print("NO PCA minDCF prior MVG (minDCF: {}, best threshold: {})".format(minDCFMVG_P1, bestThresholdMVG_P1))
    # print("NO PCA minDCF prior Tied (minDCF: {}, best threshold: {})".format(minDCFTied_P1, bestThresholdTied_P1))
    # print("NO PCA minDCF prior Naive (minDCF: {}, best threshold: {})".format(minDCFNaive_P1, bestThresholdNaive_P1))
    # print("NO PCA minDCF prior Tied Naive (minDCF: {}, best threshold: {})".format(minDCFTiedNaive_P1, bestThresholdTiedNaive_P1))

    # print("NO PCA minDCF prior MVG (minDCF: {}, best threshold: {})".format(minDCFMVG_P2, bestThresholdMVG_P2))
    # print("NO PCA minDCF prior Tied (minDCF: {}, best threshold: {})".format(minDCFTied_P2, bestThresholdTied_P2))
    # print("NO PCA minDCF prior Naive (minDCF: {}, best threshold: {})".format(minDCFNaive_P2, bestThresholdNaive_P2))
    # print("NO PCA minDCF prior Tied Naive (minDCF: {}, best threshold: {})".format(minDCFTiedNaive_P2, bestThresholdTiedNaive_P2))

    pi1 = 0.5
    pi2 = 0.1

    workingPoint = (pi1, 1, 1)

    for i in [3,4,5]:
        print("PCA: {}".format(i))
        PCA = dr.PCA(features,i)
        mvg_test = gc.multivariate_cl([1-pi1, pi1]) #( class 0, class 1 )
        val.k_fold(mvg_test, PCA, labels, 5, workingPoint, "MVG")

        mvg_test_naive = gc.naive_multivariate_cl([1-pi1, pi1])
        val.k_fold(mvg_test_naive, PCA, labels, 5, workingPoint, "Naive")

        mvg_test_tied = gc.tied_multivariate_cl([1-pi1, pi1])
        val.k_fold(mvg_test_tied, PCA, labels, 5, workingPoint, "Tied")

        mvg_test_tied_naive = gc.tied_naive_multivariate_cl([1-pi1, pi1])
        val.k_fold(mvg_test_tied_naive, PCA, labels, 5, workingPoint, "Tied Naive")

        svm_lin = svmc.SVM('linear',  C=0.1, K=1)
        val.k_fold(svm_lin, PCA, labels, 5, workingPoint, "SMV linear")

        svm_quad = svmc.SVM('Polinomial',  C=1, K=1, d=2, c=0)
        val.k_fold(svm_quad, PCA, labels, 5, workingPoint, "SMV Quadratic")

        svm_rbf = svmc.SVM('RBF',  C=100, K=0, gamma=0.1)
        val.k_fold(svm_rbf, PCA, labels, 5, workingPoint, "SMV RBF")

    print("NO PCA")
    mvg_test = gc.multivariate_cl([1-pi1, pi1]) #( class 0, class 1 )
    val.k_fold(mvg_test, features, labels, 5, workingPoint, "MVG")

    mvg_test_naive = gc.naive_multivariate_cl([1-pi1, pi1])
    val.k_fold(mvg_test_naive, features, labels, 5, workingPoint, "Naive")

    mvg_test_tied = gc.tied_multivariate_cl([1-pi1, pi1])
    val.k_fold(mvg_test_tied, features, labels, 5, workingPoint, "Tied")

    mvg_test_tied_naive = gc.tied_naive_multivariate_cl([1-pi1, pi1])
    val.k_fold(mvg_test_tied_naive, features, labels, 5, workingPoint, "Tied Naive")

    svm_lin = svmc.SVM('linear',  C=0.1, K=1)
    val.k_fold(svm_lin, features, labels, 5, workingPoint, "SMV linear")

    svm_quad = svmc.SVM('Polinomial',  C=1, K=1, d=2, c=0)
    val.k_fold(svm_quad, features, labels, 5, workingPoint, "SMV Quadratic")

    svm_rbf = svmc.SVM('RBF',  C=100, K=0, gamma=0.1)
    val.k_fold(svm_rbf, features, labels, 5, workingPoint, "SMV RBF")

    log_reg = lrc.logReg(0.1)
    val.k_fold(log_reg, featuresTrainQuadratic, labels, 5, workingPoint, "Logistic Regression")
    
    
    end_time = datetime.now()

    print("--------- TIME ----------")
    print(f"Time elapsed: {end_time - start_time}") 