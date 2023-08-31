import data_utils as du
import dimensionality_reduction as dr
import validation as val
import logistic_regression_classifiers as lrc
import GMM as gmm
import SVM_classifiers as svmc
from datetime import datetime
from itertools import repeat
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

if __name__ == "__main__":
    start_time = datetime.now()

    #LOADING DATASET
    labels, features = du.load(".\Data\Train.txt")
    labels_test, features_test = du.load(".\Data\Test.txt")
    labels_sh = shuffle(labels,random_state=0)
    labels_sh_sh = shuffle(labels_sh,random_state=0)

    features_expanded = du.features_expansion(features)
    features_PCA_5 = dr.PCA(features, 5)

    #BEST MODELS
    QLR = lrc.logReg(10, 0.1, "balanced")
    SVMC = svmc.SVM('RBF', balanced=True, gamma=0.01, K=0.01, C=0.1, piT=0.2)
    GMM = gmm.GMM(2, 32, "mvg", "tied")

    ### DET/ROC PLOTS
    # BEST MODELS
    ## Quadratic Logistic Regression NO PCA piT = 0.1, lambda = 10 
    _,_,scoresQLR, _ = val.k_fold_bayes_plot(QLR, features_expanded, labels, 5, (0.5, 1, 1), "QLR")

    ## SVM RBF Balanced gamma = 0.01, C = 0.1, K = 0.01 piT = 0.2 PCA 5
    _,_,scoresSVM, _ = val.k_fold_bayes_plot(SVMC, features_PCA_5, labels, 5, (0.5, 1, 1), "SVM")
    
    ## GMM 2 FC 32 FC - T NO PCA
    _,_,scoresGMM, _ = val.k_fold_bayes_plot(GMM, features, labels, 5, (0.5, 1, 1), "GMM")

    val.get_multi_DET([scoresQLR, scoresSVM, scoresGMM], labels_sh, ["QLR", "SMV", "GMM"], "Best Models")

    ## GMM plots
    df = pd.read_csv('GMM_BestPCA_FC.csv')
    #sort by NonTarget
    df = df.sort_values(by=['NonTarget', 'Target'])

    Cprim = df['Cprim'].values

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.tight_layout()
    g = sns.barplot(x = 'Target', y = 'Cprim', hue = 'NonTarget', data = df, palette = 'coolwarm')
    for i, bar in enumerate(g.patches):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2. -0.05, 1.05 * height, Cprim[i], rotation=45)

    # max height of the figure
    plt.ylim(0, 0.5)
    # show legend outside the figure with NonTarget labels
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='NonTarget')

    #margin
    plt.subplots_adjust(right=0.8)
    plt.savefig("figures_gmm/{}.svg".format("GMM_BestPCA_FC"))


    end_time = datetime.now()
    print("--------- TIME ----------")
    print(f"Time elapsed: {end_time - start_time}")

