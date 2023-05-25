import data_utils as du
import data_visualization as dv
import dimensionality_reduction as dr
import gaussian_classifiers as gc
import validation as val
import math_utils as mu
import logistic_regression_classifiers as lrc

if __name__=="__main__":
    labels, features = du.load("..\PROJECTS\Language_detection\Train.txt")
    labels_test, features_test = du.load("..\PROJECTS\Language_detection\Test.txt")

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

    featuresTrainQuadratic = du.features_expansion(features)
    featuresTestQuadratic = du.features_expansion(features_test)
    logQuad =lrc.logReg(featuresTrainQuadratic,labels,0.001)
    w,b=logQuad.train()
    test = lrc.transform(featuresTestQuadratic,w, b)
    print("Logistic Regression: ", val.calc_accuracy(labels_test,test)*100)


    # print("Number of italian samples:", (labels == 1).sum())
    # print("Number of not italian samples:", (labels == 0).sum())
    # print((labels == 0).sum()/((labels == 1).sum()+(labels == 0).sum()))
    # no_reduction_means = mu.calcmean_classes(features, labels)
    # no_reduction_variance = mu.calcvariance_classes(features, labels)
    # print("No reduction means italian: ", no_reduction_means[1])
    # print("No reduction means not italian: ", no_reduction_means[0])
    # print("No reduction variance italian: ", no_reduction_variance[1])
    # print("No reduction variance not italian: ", no_reduction_variance[0])
    #The means of the classes 
    #dv.get_hist(features,labels,labels_dict, features_dict)
    #dv.get_scatter(features,labels,labels_dict, features_dict)

    
    #z_scored = mu.z_score(features)
    #z_scored_test = mu.z_score(features_test)
    #l2_normed = mu.l2_norm(features)
    #l2_normed_test = mu.l2_norm(features_test) # NORMALIZATION IS NOT USEFUL
    
    #DP = dr.PCA(features,5)
    #DPT = dr.PCA(features_test,5)
    #DP = dr.LDA(features,labels,3)

    #dv.get_scatter_3d(DP,2,labels)
    #dv.get_scatter(features,labels,labels_dict, features_dict)
    # dv.get_hist(features,labels,labels_dict, features_dict)

    # dv.calc_correlation_matrix(features, "Dataset")
    # dv.calc_correlation_matrix(features.T[ labels == 1].T, "Dataset Italian")
    # dv.calc_correlation_matrix(features.T[ labels == 0].T, "Dataset not Italian")
    #dv.calc_correlation_matrix(z_scored, "Dataset-zscore")
    #dv.calc_correlation_matrix(DP, "Dataset-PCA")

    # print("\n------- WITH PCA -------")
    # mvg_cl = gc.multivariate_cl()
    # mean, C = mvg_cl.fit(DP, labels)
    # predicted = mvg_cl.trasform(DPT, mean, C)
    # print(f"Multivarate + PCA: {round(val.calc_accuracy(labels_test, predicted)*100,2)}%")

    # tied_cl = gc.tied_multivariate_cl()
    # mean, C = tied_cl.fit(DP, labels)
    # predicted = tied_cl.trasform(DPT, mean, C)
    # print(f"Tied Multivarate + PCA: {round(val.calc_accuracy(labels_test, predicted)*100,2)}%")

    # naive_cl = gc.naive_multivariate_cl()
    # mean, C = naive_cl.fit(DP, labels)
    # predicted = naive_cl.trasform(DPT, mean, C)
    # print(f"Naive Multivarate + PCA: {round(val.calc_accuracy(labels_test, predicted)*100,2)}%")

    # tied_naive_cl = gc.tied_naive_multivariate_cl()
    # mean, C = tied_naive_cl.fit(DP, labels)
    # predicted = tied_naive_cl.trasform(DPT, mean, C)
    # print(f"Tied Naive Multivarate + PCA: {round(val.calc_accuracy(labels_test, predicted)*100,2)}%")

    # print("\n------- WITHOUT PCA -------")
    mvg_cl = gc.multivariate_cl()
    mean, C = mvg_cl.fit(features, labels)
    predicted = mvg_cl.trasform(features_test, mean, C)
    print(f"Multivarate: {round(val.calc_accuracy(labels_test, predicted)*100,2)}%")

    # tied_cl = gc.tied_multivariate_cl()
    # mean, C = tied_cl.fit(features, labels)
    # predicted = tied_cl.trasform(features_test, mean, C)
    # print(f"Tied Multivarate: {round(val.calc_accuracy(labels_test, predicted)*100,2)}%")

    # naive_cl = gc.naive_multivariate_cl()
    # mean, C = naive_cl.fit(features, labels)
    # predicted = naive_cl.trasform(features_test, mean, C)
    # print(f"Naive Multivarate: {round(val.calc_accuracy(labels_test, predicted)*100,2)}%")

    # tied_naive_cl = gc.tied_naive_multivariate_cl()
    # mean, C = tied_naive_cl.fit(features, labels)
    # predicted = tied_naive_cl.trasform(features_test, mean, C)
    # print(f"Tied Naive Multivarate: {round(val.calc_accuracy(labels_test, predicted)*100,2)}%\n")

    #print("\n------- WITH ZSCORE -------")
    #mvg_cl = gc.multivariate_cl()
    #mean, C = mvg_cl.fit(z_scored, labels)
    #predicted = mvg_cl.trasform(z_scored_test, mean, C)
    #print(f"Multivarate + zscore: {round(val.calc_accuracy(labels_test, predicted)*100,2)}%")

    #other classifiers produce singular matrices

    #learners = [gc.multivariate_cl(),gc.naive_multivariate_cl(), gc.tied_multivariate_cl(), gc.tied_naive_multivariate_cl()]
    #print(val.k_fold(learners, DP, labels, len(labels)))
    #NO PCA [93.71573175875159, 92.74567692956558, 83.12948123154787, 83.12948123154787]
    #PCA    [93.92661324335724, 93.63137916490932, 83.12948123154787, 83.12948123154787]
    
