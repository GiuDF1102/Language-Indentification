import data_utils as du
import validation as val
import math_utils as mu
import GMM as gmm

#DATA
labels, features = du.load(".\Data\Train.txt")
DNORM = mu.z_score(features)

with open("output_GMM_minDCF.txt", "w") as f:

    for pi in [0.1,0.5]:
        for nTarget in [2,4,8]:
            for nNonTarget in [2,4,8,16,32]:
                for MtypeTarget in ["mvg","tied","diagonal","tied diagonal"]:
                    for MtypeNonTarget in ["mvg","tied","diagonal","tied diagonal"]:
                        GMMClass = gmm.GMM(nTarget,nNonTarget,MtypeTarget,MtypeNonTarget)
                        _, minDCF = val.k_fold_bayes_plot(GMMClass, features, labels, 5, (pi, 1, 1), "PROVA", False)
                        print("GMM, minDCF NO PCA with nTarget {},nNonTarget{},MTypeTarget{},MtypeNonTargte {}, and prior {} is {}".format(nTarget,nNonTarget,MtypeTarget,MtypeNonTarget, pi, minDCF),file=f)

    for pi in [0.1,0.5]:
        for nPCA in [5,4]:
                dataPCA = dr.PCA(features, nPCA)
                for nTarget in [2,4,8]:
                    for nNonTarget in [2,4,8,16,32]:
                        for MtypeTarget in ["mvg","tied","diagonal","tied diagonal"]:
                            for MtypeNonTarget in ["mvg","tied","diagonal","tied diagonal"]:
                                GMMClass = gmm.GMM(nTarget,nNonTarget,MtypeTarget,MtypeNonTarget)
                                _, minDCF = val.k_fold(GMMClass, dataPCA, labels, 5, (pi, 1, 1), "PROVA", False)
                                print("GMM, minDCF with PCA {}, with nTarget {},nNonTarget{},MTypeTarget{},MtypeNonTargte {}, and prior {} is {}".format(nPCA,nTarget,nNonTarget,MtypeTarget,MtypeNonTarget, pi, minDCF),file=f)

#MVG
GMM1 = gmm.GMM(2,32,"MVG", "MVG")
GMM2 = gmm.GMM(2,32,"MVG", "diagonal")
GMM3 = gmm.GMM(2,32,"MVG", "tied")
GMM4 = gmm.GMM(2,32,"MVG", "tied diagonal")

#NAIVE
GMM5 = gmm.GMM(2,32,"diagonal", "MVG")
GMM6 = gmm.GMM(2,32,"diagonal", "diagonal")
GMM7 = gmm.GMM(2,32,"diagonal", "tied")
GMM8 = gmm.GMM(2,32,"diagonal", "tied diagonal")

_, minDCF_05_1, _, _ = val.k_fold_bayes_plot(GMM1, D, L, 5, (0.5, 1, 1), "PROVA", False)
_, minDCF_05_2, _, _ = val.k_fold_bayes_plot(GMM2, D, L, 5, (0.5, 1, 1), "PROVA", False)
_, minDCF_05_3, _, _ = val.k_fold_bayes_plot(GMM3, D, L, 5, (0.5, 1, 1), "PROVA", False)
_, minDCF_05_4, _, _ = val.k_fold_bayes_plot(GMM4, D, L, 5, (0.5, 1, 1), "PROVA", False)
_, minDCF_05_5, _, _ = val.k_fold_bayes_plot(GMM5, D, L, 5, (0.5, 1, 1), "PROVA", False)
_, minDCF_05_6, _, _ = val.k_fold_bayes_plot(GMM6, D, L, 5, (0.5, 1, 1), "PROVA", False)
_, minDCF_05_7, _, _ = val.k_fold_bayes_plot(GMM7, D, L, 5, (0.5, 1, 1), "PROVA", False)
_, minDCF_05_8, _, _ = val.k_fold_bayes_plot(GMM8, D, L, 5, (0.5, 1, 1), "PROVA", False)

_, minDCF_01_1, _, _ = val.k_fold_bayes_plot(GMM1, D, L, 5, (0.1, 1, 1), "PROVA", False)
_, minDCF_01_2, _, _ = val.k_fold_bayes_plot(GMM2, D, L, 5, (0.1, 1, 1), "PROVA", False)
_, minDCF_01_3, _, _ = val.k_fold_bayes_plot(GMM3, D, L, 5, (0.1, 1, 1), "PROVA", False)
_, minDCF_01_4, _, _ = val.k_fold_bayes_plot(GMM4, D, L, 5, (0.1, 1, 1), "PROVA", False)
_, minDCF_01_5, _, _ = val.k_fold_bayes_plot(GMM5, D, L, 5, (0.1, 1, 1), "PROVA", False)
_, minDCF_01_6, _, _ = val.k_fold_bayes_plot(GMM6, D, L, 5, (0.1, 1, 1), "PROVA", False)
_, minDCF_01_7, _, _ = val.k_fold_bayes_plot(GMM7, D, L, 5, (0.1, 1, 1), "PROVA", False)
_, minDCF_01_8, _, _ = val.k_fold_bayes_plot(GMM8, D, L, 5, (0.1, 1, 1), "PROVA", False)

minCprim1 = (minDCF_05_1 + minDCF_01_1)/2
minCprim2 = (minDCF_05_2 + minDCF_01_2)/2
minCprim3 = (minDCF_05_3 + minDCF_01_3)/2
minCprim4 = (minDCF_05_4 + minDCF_01_4)/2
minCprim5 = (minDCF_05_5 + minDCF_01_5)/2
minCprim6 = (minDCF_05_6 + minDCF_01_6)/2
minCprim7 = (minDCF_05_7 + minDCF_01_7)/2
minCprim8 = (minDCF_05_8 + minDCF_01_8)/2

print("MVG - MVG minDCF_05: {}, minDCF_01: {}, minCprim: {}".format(minDCF_05_1, minDCF_01_1, minCprim1))
print("MVG - diagonal minDCF_05: {}, minDCF_01: {}, minCprim: {}".format(minDCF_05_2, minDCF_01_2, minCprim2))
print("MVG - tied minDCF_05: {}, minDCF_01: {}, minCprim: {}".format(minDCF_05_3, minDCF_01_3, minCprim3))
print("MVG - tied diagonal minDCF_05: {}, minDCF_01: {}, minCprim: {}".format(minDCF_05_4, minDCF_01_4, minCprim4))
print("diagonal - MVG minDCF_05: {}, minDCF_01: {}, minCprim: {}".format(minDCF_05_5, minDCF_01_5, minCprim5))
print("diagonal - diagonal minDCF_05: {}, minDCF_01: {}, minCprim: {}".format(minDCF_05_6, minDCF_01_6, minCprim6))
print("diagonal - tied minDCF_05: {}, minDCF_01: {}, minCprim: {}".format(minDCF_05_7, minDCF_01_7, minCprim7))
print("diagonal - tied diagonal minDCF_05: {}, minDCF_01: {}, minCprim: {}".format(minDCF_05_8, minDCF_01_8, minCprim8))

#NORM

_, minDCF_05_1, _, _ = val.k_fold_bayes_plot(GMM1, DNORM, L, 5, (0.5, 1, 1), "PROVA", False)
_, minDCF_05_2, _, _ = val.k_fold_bayes_plot(GMM2, DNORM, L, 5, (0.5, 1, 1), "PROVA", False)
_, minDCF_05_3, _, _ = val.k_fold_bayes_plot(GMM3, DNORM, L, 5, (0.5, 1, 1), "PROVA", False)
_, minDCF_05_4, _, _ = val.k_fold_bayes_plot(GMM4, DNORM, L, 5, (0.5, 1, 1), "PROVA", False)
_, minDCF_05_5, _, _ = val.k_fold_bayes_plot(GMM5, DNORM, L, 5, (0.5, 1, 1), "PROVA", False)
_, minDCF_05_6, _, _ = val.k_fold_bayes_plot(GMM6, DNORM, L, 5, (0.5, 1, 1), "PROVA", False)
_, minDCF_05_7, _, _ = val.k_fold_bayes_plot(GMM7, DNORM, L, 5, (0.5, 1, 1), "PROVA", False)
_, minDCF_05_8, _, _ = val.k_fold_bayes_plot(GMM8, DNORM, L, 5, (0.5, 1, 1), "PROVA", False)

_, minDCF_01_1, _, _ = val.k_fold_bayes_plot(GMM1, DNORM, L, 5, (0.1, 1, 1), "PROVA", False)
_, minDCF_01_2, _, _ = val.k_fold_bayes_plot(GMM2, DNORM, L, 5, (0.1, 1, 1), "PROVA", False)
_, minDCF_01_3, _, _ = val.k_fold_bayes_plot(GMM3, DNORM, L, 5, (0.1, 1, 1), "PROVA", False)
_, minDCF_01_4, _, _ = val.k_fold_bayes_plot(GMM4, DNORM, L, 5, (0.1, 1, 1), "PROVA", False)
_, minDCF_01_5, _, _ = val.k_fold_bayes_plot(GMM5, DNORM, L, 5, (0.1, 1, 1), "PROVA", False)
_, minDCF_01_6, _, _ = val.k_fold_bayes_plot(GMM6, DNORM, L, 5, (0.1, 1, 1), "PROVA", False)
_, minDCF_01_7, _, _ = val.k_fold_bayes_plot(GMM7, DNORM, L, 5, (0.1, 1, 1), "PROVA", False)
_, minDCF_01_8, _, _ = val.k_fold_bayes_plot(GMM8, DNORM, L, 5, (0.1, 1, 1), "PROVA", False)

minCprim1 = (minDCF_05_1 + minDCF_01_1)/2
minCprim2 = (minDCF_05_2 + minDCF_01_2)/2
minCprim3 = (minDCF_05_3 + minDCF_01_3)/2
minCprim4 = (minDCF_05_4 + minDCF_01_4)/2
minCprim5 = (minDCF_05_5 + minDCF_01_5)/2
minCprim6 = (minDCF_05_6 + minDCF_01_6)/2
minCprim7 = (minDCF_05_7 + minDCF_01_7)/2
minCprim8 = (minDCF_05_8 + minDCF_01_8)/2

print("NORM MVG - MVG minDCF_05: {}, minDCF_01: {}, minCprim: {}".format(minDCF_05_1, minDCF_01_1, minCprim1))
print("NORM MVG - diagonal minDCF_05: {}, minDCF_01: {}, minCprim: {}".format(minDCF_05_2, minDCF_01_2, minCprim2))
print("NORM MVG - tied minDCF_05: {}, minDCF_01: {}, minCprim: {}".format(minDCF_05_3, minDCF_01_3, minCprim3))
print("NORM MVG - tied diagonal minDCF_05: {}, minDCF_01: {}, minCprim: {}".format(minDCF_05_4, minDCF_01_4, minCprim4))
print("NORM diagonal - MVG minDCF_05: {}, minDCF_01: {}, minCprim: {}".format(minDCF_05_5, minDCF_01_5, minCprim5))
print("NORM diagonal - diagonal minDCF_05: {}, minDCF_01: {}, minCprim: {}".format(minDCF_05_6, minDCF_01_6, minCprim6))
print("NORM diagonal - tied minDCF_05: {}, minDCF_01: {}, minCprim: {}".format(minDCF_05_7, minDCF_01_7, minCprim7))
print("NORM diagonal - tied diagonal minDCF_05: {}, minDCF_01: {}, minCprim: {}".format(minDCF_05_8, minDCF_01_8, minCprim8))
