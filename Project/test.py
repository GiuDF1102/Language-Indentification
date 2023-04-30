import data_utils as du
import data_visualization as dv
import dimensionality_reduction as dr
import math_utils as mu

if __name__=="__main__":
    labels, features = du.load("..\PROJECTS\Language_detection\Train.txt")

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

    no_reduction_means = mu.calcmean_classes(features, labels)
    no_reduction_variance = mu.calcmean_variance(features, labels)
    dv.get_hist(features,labels,labels_dict, features_dict)
    dv.get_scatter(features,labels,labels_dict, features_dict)

    DP = dr.PCA(features,2)
    #DP = LDA(features,labels,3)
    dv.get_scatter_3d(DP,2,labels)
    dv.get_scatter(DP,labels,labels_dict, features_dict_PCA)
    dv.get_hist(DP,labels,labels_dict, features_dict_PCA)