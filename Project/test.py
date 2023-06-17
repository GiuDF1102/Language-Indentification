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
import sys

if __name__ == "__main__":
    orig_stdout = sys.stdout
    f = open('out_log_reg.txt', 'w')
    sys.stdout = f

    start_time = datetime.now()

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
    """

    minDCF_list_01_logReg = []
    minDCF_list_05_logReg = []
    print("#####################PRIOR 0.1 ######################\n")
    pi = 0.1
    workingPoint = (pi, 1, 1)
    K = 5
    for i in [3,4,5]:
        print("\n#################PCA: {} ##################".format(i))
        PCA = dr.PCA(features,i)

        print("###########LOGREG l = 100 #############\n")
        expanded_f = du.features_expansion(PCA)
        log_reg = lrc.logReg(100)
        val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 100")

        print("###########LOGREG l = 10 #############\n")
        expanded_f = du.features_expansion(PCA)
        log_reg = lrc.logReg(10)
        val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 10")

        print("###########LOGREG l = 1 #############\n")
        expanded_f = du.features_expansion(PCA)
        log_reg = lrc.logReg(1)
        val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 1")

        print("###########LOGREG l = 0.1 #############\n")
        expanded_f = du.features_expansion(PCA)
        log_reg = lrc.logReg(0.1)
        val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 0.1")

        print("###########LOGREG l = 0.01 #############\n")
        expanded_f = du.features_expansion(PCA)
        log_reg = lrc.logReg(0.01)
        val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 0.01")

        print("###########LOGREG l = 0.001 #############\n")
        expanded_f = du.features_expansion(PCA)
        log_reg = lrc.logReg(0.001)
        val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 0.001")

        print("###########LOGREG l = 0.0001 #############\n")
        expanded_f = du.features_expansion(PCA)
        log_reg = lrc.logReg(0.0001)
        val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 0.0001")

        print("###########LOGREG l = 0.00001 #############\n")
        expanded_f = du.features_expansion(PCA)
        log_reg = lrc.logReg(0.00001)
        val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 0.00001")

        # print("#############MVG ##############\n")
        # mvg_test = gc.multivariate_cl([1-pi, pi]) #( class 0, class 1 )
        # val.k_fold(mvg_test, PCA, labels, K, workingPoint, "MVG")

        # mvg_test_naive = gc.naive_multivariate_cl([1-pi, pi])
        # val.k_fold(mvg_test_naive, PCA, labels, K, workingPoint, "Naive")

        # mvg_test_tied = gc.tied_multivariate_cl([1-pi, pi])
        # val.k_fold(mvg_test_tied, PCA, labels, K, workingPoint, "Tied")

        # mvg_test_tied_naive = gc.tied_naive_multivariate_cl([1-pi, pi])
        # val.k_fold(mvg_test_tied_naive, PCA, labels, K, workingPoint, "Tied Naive")

        # svm_lin = svmc.SVM('linear',  C=0.1, K=1)
        # val.k_fold(svm_lin, PCA, labels, K, workingPoint, "SMV linear C=0.1, K=1")

        # svm_quad = svmc.SVM('Polinomial',  C=1, K=1, d=2, c=0)
        # val.k_fold(svm_quad, PCA, labels, K, workingPoint, "SMV Quadratic C=1, K=1, d=2, c=0")

        # svm_rbf = svmc.SVM('RBF',  C=100, K=0, gamma=0.1)
        # val.k_fold(svm_rbf, PCA, labels, K, workingPoint, "SMV RBF C=100, K=0, gamma=0.1")

    print("#################NO PCA ##################\n")

    print("###########LOGREG l = 100 #############\n")
    expanded_f = du.features_expansion(features)
    log_reg = lrc.logReg(100)
    val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 100")

    print("###########LOGREG l = 10 #############\n")
    expanded_f = du.features_expansion(features)
    log_reg = lrc.logReg(10)
    val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 10")

    print("###########LOGREG l = 1 #############\n")
    expanded_f = du.features_expansion(features)
    log_reg = lrc.logReg(1)
    val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 1")

    print("###########LOGREG l = 0.1 #############\n")
    expanded_f = du.features_expansion(features)
    log_reg = lrc.logReg(0.1)
    val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 0.1")

    print("###########LOGREG l = 0.01 #############\n")
    expanded_f = du.features_expansion(features)
    log_reg = lrc.logReg(0.01)
    val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 0.01")

    print("###########LOGREG l = 0.001 #############\n")
    expanded_f = du.features_expansion(features)
    log_reg = lrc.logReg(0.001)
    val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 0.001")

    print("###########LOGREG l = 0.0001 #############\n")
    expanded_f = du.features_expansion(features)
    log_reg = lrc.logReg(0.0001)
    val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 0.0001")

    print("###########LOGREG l = 0.00001 #############\n")
    expanded_f = du.features_expansion(features)
    log_reg = lrc.logReg(0.00001)
    val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 0.00001")

    # print("#############MVG ##############\n")
    # mvg_test = gc.multivariate_cl([1-pi, pi]) #( class 0, class 1 )
    # val.k_fold(mvg_test, features, labels, K, workingPoint, "MVG")

    # mvg_test_naive = gc.naive_multivariate_cl([1-pi, pi])
    # val.k_fold(mvg_test_naive, features, labels, K, workingPoint, "Naive")

    # mvg_test_tied = gc.tied_multivariate_cl([1-pi, pi])
    # val.k_fold(mvg_test_tied, features, labels, K, workingPoint, "Tied")

    # mvg_test_tied_naive = gc.tied_naive_multivariate_cl([1-pi, pi])
    # val.k_fold(mvg_test_tied_naive, features, labels, K, workingPoint, "Tied Naive")

    # print("#############SVM ##############\n")
    # svm_lin = svmc.SVM('linear',  C=0.1, K=1)
    # val.k_fold(svm_lin, features, labels, K, workingPoint, "SMV linear C=0.1, K=1")

    # svm_quad = svmc.SVM('Polinomial',  C=1, K=1, d=2, c=0)
    # val.k_fold(svm_quad, features, labels, K, workingPoint, "SMV Quadratic C=1, K=1, d=2, c=0")

    # svm_rbf = svmc.SVM('RBF',  C=100, K=0, gamma=0.1)
    # val.k_fold(svm_rbf, features, labels, K, workingPoint, "SMV RBF C=100, K=0, gamma=0.1")
    
    print("\n\n#####################PRIOR 0.5 ######################\n")
    pi = 0.5
    workingPoint = (pi, 1, 1)
    K = 3
    for i in [3,4,5]:
        print("\n#################PCA: {} ##################".format(i))
        PCA = dr.PCA(features,i)

        print("###########LOGREG l = 100 #############\n")
        expanded_f = du.features_expansion(PCA)
        log_reg = lrc.logReg(100)
        val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 100")

        print("###########LOGREG l = 10 #############\n")
        expanded_f = du.features_expansion(PCA)
        log_reg = lrc.logReg(10)
        val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 10")

        print("###########LOGREG l = 1 #############\n")
        expanded_f = du.features_expansion(PCA)
        log_reg = lrc.logReg(1)
        val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 1")

        print("###########LOGREG l = 0.1 #############\n")
        expanded_f = du.features_expansion(PCA)
        log_reg = lrc.logReg(0.1)
        val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 0.1")

        print("###########LOGREG l = 0.01 #############\n")
        expanded_f = du.features_expansion(PCA)
        log_reg = lrc.logReg(0.01)
        val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 0.01")

        print("###########LOGREG l = 0.001 #############\n")
        expanded_f = du.features_expansion(PCA)
        log_reg = lrc.logReg(0.001)
        val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 0.001")

        print("###########LOGREG l = 0.0001 #############\n")
        expanded_f = du.features_expansion(PCA)
        log_reg = lrc.logReg(0.0001)
        val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 0.0001")

        print("###########LOGREG l = 0.00001 #############\n")
        expanded_f = du.features_expansion(PCA)
        log_reg = lrc.logReg(0.00001)
        val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 0.00001")

        # print("#############MVG ##############\n")
        # mvg_test = gc.multivariate_cl([1-pi, pi]) #( class 0, class 1 )
        # val.k_fold(mvg_test, PCA, labels, K, workingPoint, "MVG")

        # mvg_test_naive = gc.naive_multivariate_cl([1-pi, pi])
        # val.k_fold(mvg_test_naive, PCA, labels, K, workingPoint, "Naive")

        # mvg_test_tied = gc.tied_multivariate_cl([1-pi, pi])
        # val.k_fold(mvg_test_tied, PCA, labels, K, workingPoint, "Tied")

        # mvg_test_tied_naive = gc.tied_naive_multivariate_cl([1-pi, pi])
        # val.k_fold(mvg_test_tied_naive, PCA, labels, K, workingPoint, "Tied Naive")

        # print("#############SVM ##############\n")
        # svm_lin = svmc.SVM('linear',  C=0.1, K=1)
        # val.k_fold(svm_lin, PCA, labels, K, workingPoint, "SMV linear C=0.1, K=1")

        # svm_quad = svmc.SVM('Polinomial',  C=1, K=1, d=2, c=0)
        # val.k_fold(svm_quad, PCA, labels, K, workingPoint, "SMV Quadratic C=1, K=1, d=2, c=0")

        # svm_rbf = svmc.SVM('RBF',  C=100, K=0, gamma=0.1)
        # val.k_fold(svm_rbf, PCA, labels, K, workingPoint, "SMV RBF C=100, K=0, gamma=0.1")

    print("#################NO PCA ##################\n")

    print("###########LOGREG l = 100 #############\n")
    expanded_f = du.features_expansion(features)
    log_reg = lrc.logReg(100)
    val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 100")

    print("###########LOGREG l = 10 #############\n")
    expanded_f = du.features_expansion(features)
    log_reg = lrc.logReg(10)
    val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 10")

    print("###########LOGREG l = 1 #############\n")
    expanded_f = du.features_expansion(features)
    log_reg = lrc.logReg(1)
    val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 1")

    print("###########LOGREG l = 0.1 #############\n")
    expanded_f = du.features_expansion(features)
    log_reg = lrc.logReg(0.1)
    val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 0.1")

    print("###########LOGREG l = 0.01 #############\n")
    expanded_f = du.features_expansion(features)
    log_reg = lrc.logReg(0.01)
    val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 0.01")

    print("###########LOGREG l = 0.001 #############\n")
    expanded_f = du.features_expansion(features)
    log_reg = lrc.logReg(0.001)
    val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 0.001")

    print("###########LOGREG l = 0.0001 #############\n")
    expanded_f = du.features_expansion(features)
    log_reg = lrc.logReg(0.0001)
    val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 0.0001")

    print("###########LOGREG l = 0.00001 #############\n")
    expanded_f = du.features_expansion(features)
    log_reg = lrc.logReg(0.00001)
    val.k_fold(log_reg, expanded_f, labels, K, workingPoint, "Logistic Regression l = 0.00001") 

    # print("#############MVG ##############\n")
    # mvg_test = gc.multivariate_cl([1-pi, pi]) #( class 0, class 1 )
    # val.k_fold(mvg_test, features, labels, K, workingPoint, "MVG")

    # mvg_test_naive = gc.naive_multivariate_cl([1-pi, pi])
    # val.k_fold(mvg_test_naive, features, labels, K, workingPoint, "Naive")

    # mvg_test_tied = gc.tied_multivariate_cl([1-pi, pi])
    # val.k_fold(mvg_test_tied, features, labels, K, workingPoint, "Tied")

    # mvg_test_tied_naive = gc.tied_naive_multivariate_cl([1-pi, pi])
    # val.k_fold(mvg_test_tied_naive, features, labels, K, workingPoint, "Tied Naive")

    # print("#############SVM ##############\n")
    # svm_lin = svmc.SVM('linear',  C=0.1, K=1)
    # val.k_fold(svm_lin, features, labels, K, workingPoint, "SMV linear C=0.1, K=1")

    # svm_quad = svmc.SVM('Polinomial',  C=1, K=1, d=2, c=0)
    # val.k_fold(svm_quad, features, labels, K, workingPoint, "SMV Quadratic C=1, K=1, d=2, c=0")

    # svm_rbf = svmc.SVM('RBF',  C=100, K=0, gamma=0.1)
    # val.k_fold(svm_rbf, features, labels, K, workingPoint, "SMV RBF C=100, K=0, gamma=0.1")

    end_time = datetime.now()

    print("--------- TIME ----------")
    print(f"Time elapsed: {end_time - start_time}")
    
    sys.stdout = orig_stdout
    f.close()
        