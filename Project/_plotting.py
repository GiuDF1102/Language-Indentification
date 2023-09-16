import data_utils as du
import dimensionality_reduction as dr
import validation as val
import logistic_regression_classifiers as lrc
import GMM as gmm
import SVM_classifiers as svmc
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import data_visualization as dv

if __name__ == "__main__":
    #LOADING DATASET
    labels, features = du.load(".\Data\Train.txt")
    labels_sh = shuffle(labels,random_state=0)
    features_expanded = du.features_expansion(features)
    features_PCA_5 = dr.PCA(features, 5)
    features_PCA_2 = dr.PCA(features, 2)

    #BEST MODELS
    QLR = lrc.logReg(10, 0.1, "balanced")
    SVMC = svmc.SVM('RBF', balanced=True, gamma=0.01, K=0.01, C=0.1, piT=0.2)
    GMM = gmm.GMM(2, 32, "mvg", "tied")

    ### DET/ROC PLOTS
    _,_,scoresQLR, _ = val.k_fold_bayes_plot(QLR, features_expanded, labels, 5, (0.5, 1, 1), "QLR")
    _,_,scoresSVM, _ = val.k_fold_bayes_plot(SVMC, features_PCA_5, labels, 5, (0.5, 1, 1), "SVM")
    _,_,scoresGMM, _ = val.k_fold_bayes_plot(GMM, features, labels, 5, (0.5, 1, 1), "GMM")

    val.get_multi_DET([scoresQLR, scoresSVM, scoresGMM], labels_sh, ["QLR", "SMV", "GMM"], "Best Models")

    ### GMM plots
    df = pd.read_csv('GMM_BestPCA_FC.csv')
    df = df.sort_values(by=['NonTarget', 'Target'])

    Cprim = df['Cprim'].values

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.tight_layout()
    g = sns.barplot(x = 'Target', y = 'Cprim', hue = 'NonTarget', data = df, palette = 'coolwarm')
    for i, bar in enumerate(g.patches):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2. -0.05, 1.05 * height, Cprim[i], rotation=45)

    plt.ylim(0, 0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='NonTarget')
    plt.subplots_adjust(right=0.8)
    plt.savefig("figures_gmm/{}.svg".format("GMM_BestPCA_FC"))


    ### LDA PLOT
    DATALDA = dr.LDA(features, labels, 1).T
    plt.figure(figsize=(10, 4))
    plt.hist(DATALDA[labels == 0], bins=100, alpha=0.5, label='NonTarget')
    plt.hist(DATALDA[labels == 1], bins=100, alpha=0.5, label='Target')
    plt.legend(loc='upper right')
    plt.savefig("{}.svg".format("LDA_figure"))
    plt.show()

    ### HISTOGRAMS and SCATTER PLOTS
    map_classes = {"Not-Italian": 0,"Italian": 1}
    map_features = {"feature 1":0, "feature 2":1}
    dv.get_scatter_total(features, labels, map_classes, map_features)

    ### correlation matrix
    italian = features[:, labels == 1]
    italian_CM = dv.calc_correlation_matrix(italian, "italian_CM")
    not_italian = features[:, labels == 0]
    not_italian_CM = dv.calc_correlation_matrix(not_italian, "not_italian_CM")
    dv.calc_correlation_matrix(features, "italian_CM")

    ### exaplined variance
    du.explained_variance(features)