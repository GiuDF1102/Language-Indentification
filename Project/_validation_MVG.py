import validation as val
import data_utils as du
import dimensionality_reduction as dr
import gaussian_classifiers as gc
import math_utils as mu

#LOADING DATASET
labels, features = du.load(".\Data\Train.txt")
labels_test, features_test = du.load(".\Data\Test.txt")

#MVG
for pi in [0.1,0.5]:
    #NO PCA
    mvgObj = gc.multivariate_cl([1-pi, pi])
    _, minDCF = val.k_fold(mvgObj, features, labels, 5, (pi, 1, 1))
    print("MVG, minDCF with pi {} is {}".format(pi, minDCF))
    
    #PCA
    for nPCA in [5,4]:
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
    for nPCA in [5,4]:
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
    for nPCA in [5,4]:
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
    for nPCA in [5,4]:
        dataPCA = dr.PCA(features, nPCA)
        tiedNaiveMvgObj = gc.tied_naive_multivariate_cl([1-pi, pi])
        _, minDCF = val.k_fold(tiedNaiveMvgObj, dataPCA, labels, 5, (pi, 1, 1))
        print("Tied Naive MVG, minDCF with pi {} and {} PCA is {}".format(pi, nPCA, minDCF))


BESTMVG = gc.naive_multivariate_cl([0.9, 0.1])

# BEST MODEL WITH Z-NORMALIZATION
featuresZNorm = mu.z_score(features)
featuresPCA5ZNorm = dr.PCA(mu.z_score(features),5)
featuresPCA4ZNorm = dr.PCA(mu.z_score(features),4)
_, minDCF1, scores, pred = val.k_fold_bayes_plot(BESTMVG, featuresZNorm, labels, 5, (0.1, 1, 1), "BESTMVG", None)
print("minDCF NAIVE Z-NORM 0.1:", minDCF1)
_, minDCF2, scores, pred = val.k_fold_bayes_plot(BESTMVG, featuresZNorm, labels, 5, (0.5, 1, 1), "BESTMVG", None)
print("minDCF NAIVE Z-NORM 0.5:", minDCF2)
print("minCprim NAIVE Z-NORM:", (minDCF1+minDCF2)/2)

_, minDCF1, scores, pred = val.k_fold_bayes_plot(BESTMVG, featuresPCA5ZNorm, labels, 5, (0.1, 1, 1), "BESTMVG", None)
print("minDCF NAIVE PCA 5 Z-NORM 0.1:", minDCF1)
_, minDCF2, scores, pred = val.k_fold_bayes_plot(BESTMVG, featuresPCA5ZNorm, labels, 5, (0.5, 1, 1), "BESTMVG", None)
print("minDCF NAIVE PCA 5 Z-NORM 0.5:", minDCF2)
print("minCprim NAIVE PCA 5 Z-NORM:", (minDCF1+minDCF2)/2)

_, minDCF1, scores, pred = val.k_fold_bayes_plot(BESTMVG, featuresPCA4ZNorm, labels, 5, (0.1, 1, 1), "BESTMVG", None)
print("minDCF NAIVE PCA 4 Z-NORM 0.1:", minDCF1)
_, minDCF2, scores, pred = val.k_fold_bayes_plot(BESTMVG, featuresPCA4ZNorm, labels, 5, (0.5, 1, 1), "BESTMVG", None)
print("minDCF NAIVE PCA 4 Z-NORM 0.5:", minDCF2)
print("minCprim NAIVE PCA 4 Z-NORM:", (minDCF1+minDCF2)/2)