import validation as val
import data_utils as du
import dimensionality_reduction as dr
import gaussian_classifiers as gc

#LOADING DATASET
labels, features = du.load("..\PROJECTS\Language_detection\Train.txt")
labels_test, features_test = du.load("..\PROJECTS\Language_detection\Test.txt")

#MVG
for pi in [0.1,0.5]:
    #NO PCA
    mvgObj = gc.multivariate_cl([1-pi, pi])
    _, minDCF = val.k_fold(mvgObj, features, labels, 5, (pi, 1, 1))
    print("MVG, minDCF with pi {} is {}".format(pi, minDCF))
    
    #PCA
    for nPCA in [5,4,3]:
        dataPCA = dr.PCA(features, nPCA)
        mvgObj = gc.multivariate_cl([1-pi, pi])
        _, minDCF = val.k_fold(mvgObj, dataPCA, labels, 5, (pi, 1, 1))
        print("MVG, minDCF with pi {} and {} PCA is {}".format(pi, nPCA, minDCF))

#NAIVE MVG
for pi in [0.1,0.5]:
    #NO PCA
    naiveMvgObj = gc.naive_multivariate_cl([1-pi, pi])
    _, minDCF = val.k_fold(naiveMvgObj, features, labels, 5, (pi, 1, 1))
    print("Naive MVG, minDCF with pi {} is {}".format(pi, minDCF))
    
    #PCA
    for nPCA in [5,4,3]:
        dataPCA = dr.PCA(features, nPCA)
        naiveMvgObj = gc.naive_multivariate_cl([1-pi, pi])
        _, minDCF = val.k_fold(naiveMvgObj, dataPCA, labels, 5, (pi, 1, 1))
        print("Naive MVG, minDCF with pi {} and {} PCA is {}".format(pi, nPCA, minDCF))

#TIED MVG
for pi in [0.1,0.5]:
    #NO PCA
    tiedMvgObj = gc.tied_multivariate_cl([1-pi, pi])
    _, minDCF = val.k_fold(tiedMvgObj, features, labels, 5, (pi, 1, 1))
    print("Tied MVG, minDCF with pi {} is {}".format(pi, minDCF))
    
    #PCA
    for nPCA in [5,4,3]:
        dataPCA = dr.PCA(features, nPCA)
        tiedMvgObj = gc.tied_multivariate_cl([1-pi, pi])
        _, minDCF = val.k_fold(tiedMvgObj, dataPCA, labels, 5, (pi, 1, 1))
        print("Tied MVG, minDCF with pi {} and {} PCA is {}".format(pi, nPCA, minDCF))

#TIED NAIVE MVG
for pi in [0.1,0.5]:
    #NO PCA
    tiedNaiveMvgObj = gc.tied_naive_multivariate_cl([1-pi, pi])
    _, minDCF = val.k_fold(tiedNaiveMvgObj, features, labels, 5, (pi, 1, 1))
    print("Tied Naive MVG, minDCF with pi {} is {}".format(pi, minDCF))
    
    #PCA
    for nPCA in [5,4,3]:
        dataPCA = dr.PCA(features, nPCA)
        tiedNaiveMvgObj = gc.tied_naive_multivariate_cl([1-pi, pi])
        _, minDCF = val.k_fold(tiedNaiveMvgObj, dataPCA, labels, 5, (pi, 1, 1))
        print("Tied Naive MVG, minDCF with pi {} and {} PCA is {}".format(pi, nPCA, minDCF))
