import data_utils as du
import data_visualization as dv
import dimensionality_reduction as dr
import gaussian as gg
import validation as val

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
        "PC-1": 1   
    }

    #no_reduction_means = mu.calcmean_classes(features, labels)
    #no_reduction_variance = mu.calcmean_variance(features, labels)
    #dv.get_hist(features,labels,labels_dict, features_dict)
    #dv.get_scatter(features,labels,labels_dict, features_dict)

    DP = dr.PCA(features,5)
    DPT = dr.PCA(features_test,5)

    #DP = LDA(features,labels,3)
    #dv.get_scatter_3d(DP,2,labels)
    #dv.get_scatter(DP,labels,labels_dict, features_dict_PCA)
    #dv.get_hist(DP,labels,labels_dict, features_dict_PCA)

    # #WITH PCA
    # mvg_cl = gg.multivariate_cl()
    # mean, C = mvg_cl.fit(DP, labels)
    # predicted = mvg_cl.trasform(DPT, mean, C)
    # print("PCA Multivarate:", gg.calc_accuracy(labels_test, predicted))

    # tied_cl = gg.tied_multivariate_cl()
    # mean, C = tied_cl.fit(DP, labels)
    # predicted = tied_cl.trasform(DPT, mean, C)
    # print("PCA Tied MVG:", gg.calc_accuracy(labels_test, predicted))

    # naive_cl = gg.naive_multivariate_cl()
    # mean, C = naive_cl.fit(DP, labels)
    # predicted = naive_cl.trasform(DPT, mean, C)
    # print("PCA Naive MVG:", gg.calc_accuracy(labels_test, predicted))

    # tied_naive_cl = gg.tied_naive_multivariate_cl()
    # mean, C = tied_naive_cl.fit(DP, labels)
    # predicted = tied_naive_cl.trasform(DPT, mean, C)
    # print("PCA Tied Naive MVG:", gg.calc_accuracy(labels_test, predicted))
    dv.calc_correlation_matrix(features, "Dataset")
    dv.calc_correlation_matrix(DP, "Dataset-PCA")
    #WITHOUT
    print("\n------- WITHOUT PCA -------")
    mvg_cl = gg.multivariate_cl()
    mean, C = mvg_cl.fit(features, labels)
    predicted = mvg_cl.trasform(features_test, mean, C)
    print("Multivarate:", gg.calc_accuracy(labels_test, predicted))

    tied_cl = gg.tied_multivariate_cl()
    mean, C = tied_cl.fit(features, labels)
    predicted = tied_cl.trasform(features_test, mean, C)
    print("Tied MVG:", gg.calc_accuracy(labels_test, predicted))

    naive_cl = gg.naive_multivariate_cl()
    mean, C = naive_cl.fit(features, labels)
    predicted = naive_cl.trasform(features_test, mean, C)
    print("Naive MVG:", gg.calc_accuracy(labels_test, predicted))

    tied_naive_cl = gg.tied_naive_multivariate_cl()
    mean, C = tied_naive_cl.fit(features, labels)
    predicted = tied_naive_cl.trasform(features_test, mean, C)
    print("Tied Naive MVG:", gg.calc_accuracy(labels_test, predicted))
    # print(val.k_fold([mvg_cl,tied_cl,naive_cl,tied_naive_cl],features,labels,5))