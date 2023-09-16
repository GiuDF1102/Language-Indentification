# Italian Language Detection from Utterance Embeddings: A Comparative Study of SVM, Gaussian Models, Logistic Regression, and GMM

This study presents an investigation into the performance of some common machine learning algorithms for the task of identifying the Italian language among a set of 26 languages. The algorithms are trained on a dataset of synthetic language embeddings extracted from audio sources. The study also explores the effects of dimensionality reduction, score calibration, and fusion on the classification results.

The main contributions and findings of the study are:

- The study provides a comprehensive analysis of the behaviour and performance of four classifiers: Support Vector Machines (SVM), Gaussian Models, Logistic Regression, and Gaussian Mixture Models (GMM).
- The study also applies Z-normalization to see the effects of normalization in the models.
- The study shows that quadratic classification rules are more effective than linear ones for this task, as the data is not linearly separable.
- The study demonstrates that GMM and SVM are the best performing classifiers among the four. The study also shows that fusion of different classifiers can improve the results further.
- The study evaluates the models on two working points with different prior probabilities and costs, and uses the minCprim metric as the primary measure of performance.
- The study applies Principal Component Analysis (PCA) to reduce the dimensionality of the data and observes that it slightly improves the performance of some models, but not significantly.
- The study applies score calibration to adjust the scores of the models to better reflect the posterior probabilities and observes that it reduces the classification cost for some models, especially for SVM and Logistic Regression.
