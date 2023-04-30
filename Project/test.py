import data_utils as du
import data_visualization as dv
import dimensionality_reduction as dr

if __name__=="__main__":
    labels, features = du.load("..\PROJECTS\Language_detection\Train.txt")

    labels_dict = {
        "Not-Italian": 0,
        "Italian": 1
    }
    features_dict = {
        "elem 0": 0,
        "elem 1": 1,
        "elem 2": 2,
        "elem 3": 3,
        "elem 4": 4,
        "elem 5": 5        
    }
    features_dict_PCA = {
        "PCA elem 0": 0,
        "PCA elem 1": 1   
    }

    #get_hist(features,labels,labels_dict, features_dict)
    #get_scatter(features,labels,labels_dict, features_dict)
    DP = dr.PCA(features,3)
    #DP = LDA(features,labels,3)
    dv.get_scatter_3d(DP,2,labels)
    dv.get_scatter(DP,labels,labels_dict, features_dict_PCA)
    dv.get_hist(DP,labels,labels_dict, features_dict_PCA)