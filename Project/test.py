import data_utils as du
import data_visualization as dv
import dimensionality_reduction as dr
import gaussian_classifiers as gc
import validation as val
import math_utils as mu
import logistic_regression_classifiers as lrc
from datetime import datetime

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

    #MEANS AND VARIANCE
    no_reduction_means = mu.calcmean_classes(features, labels)
    no_reduction_variance = mu.calcvariance_classes(features, labels)

    #PCA DATA
    PCA_5 = dr.PCA(features,5)
    PCA_5_TEST = dr.PCA(features_test,5)

    #QUADRATIC FEATURES FOR REGRESSION
    featuresTrainQuadratic = du.features_expansion(features)
    featuresTestQuadratic = du.features_expansion(features_test)
    featuresTrainQuadraticPCA = du.features_expansion(PCA_5)
    featuresTestQuadraticPCA = du.features_expansion(PCA_5_TEST)

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

    tied_naive_cl = gc.tied_naive_multivariate_cl()
    mean, C = tied_naive_cl.fit(features, labels)
    mean_PCA, C_PCA = tied_naive_cl.fit(features, labels)
    predicted_tied_naive = tied_naive_cl.trasform(features_test, mean, C)
    mean_PCA, C_PCA = tied_naive_cl.fit(PCA_5, labels)
    predicted_tied_naive_PCA = tied_naive_cl.trasform(PCA_5_TEST, mean_PCA, C_PCA)

    #LOGISTIC REGRESSION
    logQuad =lrc.logReg(featuresTrainQuadratic,labels,0.001)
    w,b=logQuad.train()
    (predicted_qlr,scores) = lrc.transform(featuresTestQuadratic,w, b,0)
    logQuadPCA =lrc.logReg(featuresTrainQuadraticPCA,labels,0.001)
    w,b=logQuadPCA.train()
    (predicted_qlr_PCA,scores_PCA) = lrc.transform(featuresTestQuadraticPCA,w, b,0)

    #KFOLD 
    # - GAUSSIAN CLASSIFIERS
    learners = [gc.multivariate_cl(),gc.naive_multivariate_cl(), gc.tied_multivariate_cl(), gc.tied_naive_multivariate_cl()]
    accuracies = val.k_fold(learners, features, labels, len(labels))
    accuracies_PCA = val.k_fold(learners, features, labels, len(labels))

    end_time = datetime.now()

    #PRINTING RESULTS
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
    print(f"Multivarate + PCA: {round(val.calc_accuracy(labels_test, predicted_mvg_PCA)*100,2)}%")
    print(f"Tied Multivarate + PCA: {round(val.calc_accuracy(labels_test, predicted_tied_PCA)*100,2)}%")
    print(f"Naive Multivarate + PCA: {round(val.calc_accuracy(labels_test, predicted_naive_PCA)*100,2)}%")
    print(f"Tied Naive Multivarate + PCA: {round(val.calc_accuracy(labels_test, predicted_tied_naive_PCA)*100,2)}%")
    print(f"Logistic Regression + PCA: {round(val.calc_accuracy(labels_test,predicted_qlr_PCA)*100,2)}%")


    print("--------- KFOLD ----------")
    print("Gaussian Classifiers")
    for i in range(len(accuracies)):
        print(f" - {learners[i].name}: {round(accuracies[i],2)}%")    
    for i in range(len(accuracies_PCA)):
        print(f" - {learners[i].name} + PCA: {round(accuracies_PCA[i],2)}%")

    print("--------- TIME ----------")
    print(f"Time elapsed: {end_time - start_time}")