import numpy as np
import dimensionality_reduction as dr
import data_utils as du
import gaussian_classifiers as gc
import matplotlib.pyplot as plt
import logistic_regression_classifiers as lr
import GMM as gmm

def bayes_optimal_decisions(llr, pi1, cfn, cfp):
    
    threshold = -np.log(pi1*cfn/((1-pi1)*cfp))
    predictions = (llr > threshold ).astype(int)
    return predictions


def detection_cost_function (M, pi1, cfn, cfp):
    FNR = M[0][1]/(M[0][1]+M[1][1])
    FPR = M[1][0]/(M[0][0]+M[1][0])
    
    return (pi1*cfn*FNR +(1-pi1)*cfp*FPR)

def normalized_detection_cost_function (DCF, pi1, cfn, cfp):
    dummy = np.array([pi1*cfn, (1-pi1)*cfp])
    index = np.argmin (dummy) 
    return DCF/dummy[index]

def minimum_detection_costs (llr, LTE, pi1, cfn, cfp):
    
    sorted_llr = np.sort(llr)
    
    NDCF= []
    
    for t in sorted_llr:
        predictions = (llr > t).astype(int)
        
        confMatrix =  confusionMatrix(predictions, LTE, LTE.max()+1)
        uDCF = detection_cost_function(confMatrix, pi1, cfn, cfp)
        
        NDCF.append(normalized_detection_cost_function(uDCF, pi1, cfn, cfp))
        
    index = np.argmin(NDCF)
    
    return NDCF[index]

def compute_actual_DCF(llr, LTE, pi1, cfn, cfp):
    
    predictions = (llr > (-np.log(pi1/(1-pi1)))).astype(int)
    
    confMatrix =  confusionMatrix(predictions, LTE, LTE.max()+1)
    uDCF = detection_cost_function(confMatrix, pi1, cfn, cfp)
        
    NDCF=(normalized_detection_cost_function(uDCF, pi1, cfn, cfp))
        
    return NDCF

def Ksplit(D, L, seed=0, K=3):
    folds = []
    labels = []
    numberOfSamplesInFold = int(D.shape[1]/K)
    # Generate a random seed
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    for i in range(K):
        folds.append(D[:,idx[(i*numberOfSamplesInFold): ((i+1)*(numberOfSamplesInFold))]])
        labels.append(L[idx[(i*numberOfSamplesInFold): ((i+1)*(numberOfSamplesInFold))]])
    return folds, labels

def KfoldActualDCF(D, L, model, K=3, prior=0.5):
    if (K>1):
        folds, labels = Ksplit(D, L, seed=0, K=K)
        orderedLabels = []
        scores = []
        for i in range(K):
            trainingSet = []
            labelsOfTrainingSet = []
            for j in range(K):
                if j!=i:
                    trainingSet.append(folds[j])
                    labelsOfTrainingSet.append(labels[j])
            evaluationSet = folds[i]
            orderedLabels.append(labels[i])
            trainingSet=np.hstack(trainingSet)
            labelsOfTrainingSet=np.hstack(labelsOfTrainingSet)
            model.train(trainingSet, labelsOfTrainingSet)
            getpredicted = model.transform(evaluationSet)
            scores.append(model.get_scores())
        scores=np.hstack(scores)
        orderedLabels=np.hstack(orderedLabels)
        labels = np.hstack(labels)
        return compute_actual_DCF(scores, orderedLabels, prior, 1, 1)
    else:
        print("K cannot be <=1")
    return

def Kfold(D, L, model, K=5, prior=0.5):
    if (K>1):
        folds, labels = Ksplit(D, L, seed=0, K=K)
        orderedLabels = []
        scores = []
        for i in range(K):
            trainingSet = []
            labelsOfTrainingSet = []
            for j in range(K):
                if j!=i:
                    trainingSet.append(folds[j])
                    labelsOfTrainingSet.append(labels[j])
            evaluationSet = folds[i]
            orderedLabels.append(labels[i])
            trainingSet=np.hstack(trainingSet)
            labelsOfTrainingSet=np.hstack(labelsOfTrainingSet)
            model.train(trainingSet, labelsOfTrainingSet)
            getpredicted = model.transform(evaluationSet)
            scores.append(model.get_scores())
        scores=np.hstack(scores)
        orderedLabels=np.hstack(orderedLabels)
        labels = np.hstack(labels)
        return minimum_detection_costs(scores, orderedLabels, prior, 1, 1)
    else:
        print("K cannot be <=1")
    return

def bayesErrorPlot(dcf, mindcf, effPriorLogOdds, model):
    plt.figure()
    plt.plot(effPriorLogOdds, dcf, label='act DCF', color='r')
    plt.plot(effPriorLogOdds, mindcf, label='min DCF', color='b', linestyle="--")
    plt.xlim([min(effPriorLogOdds), max(effPriorLogOdds)])
    plt.legend([model + " - act DCF", model+" - min DCF"])
    plt.xlabel("prior log-odds")
    plt.ylabel("DCF")
    plt.savefig(model + "_DCF.png")
    return

def confusionMatrix(predictedLabels, actualLabels, K):
    # Initialize matrix of K x K zeros
    matrix = np.zeros((K, K)).astype(int)
    # We're computing the confusion
    # matrix which "counts" how many times we get prediction i when the actual
    # label is j.
    for i in range(actualLabels.size):
        matrix[predictedLabels[i], actualLabels[i]] += 1
    return matrix
if __name__ == '__main__':
    # Load data
    labels, features = du.load("..\PROJECTS\Language_detection\Train.txt")

    #NAIVE PCA 5
    actualDCFs = []
    minDCFs = []
    numberOfPoints=21
    effPriorLogOdds = np.linspace(-3, 3, numberOfPoints)
    effPriors = 1/(1+np.exp(-1*effPriorLogOdds))
    gmmc = gmm.GMM(2,32,'MVG','tied')
    for i in range(numberOfPoints):
        actualDCFs.append(KfoldActualDCF(features, labels, gmmc, prior=effPriors[i]))
        minDCFs.append(Kfold(features, labels,gmmc, prior=effPriors[i]))
        print("At iteration", i, "the min DCF is", minDCFs[i], "and the actual DCF is", actualDCFs[i])
    bayesErrorPlot(actualDCFs, minDCFs, effPriorLogOdds, "GMM")

