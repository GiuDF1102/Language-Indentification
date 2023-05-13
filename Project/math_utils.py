import numpy as np

def FromRowToColumn(v):
    return v.reshape((v.size, 1))

def vrow(x):
    return x.reshape((1,x.shape[0]))

def calcmean(D):
    return D.mean(1) #ritorno media sulle colonne

def calcmean_classes(Data, labels):
    classes_list = []
    classes_means = []
    num_classes = len(np.unique(labels))
    num_feats = len(Data)

    for i in range(num_classes):
        classes_list.append(np.array(Data[:, (labels == i)]))
    
    for i in range(num_classes):
        means = []
        for j in range(num_feats):
            means.append(classes_list[i][j].mean())
        classes_means.append(means)

    return classes_means

def calcvariance_classes(Data, labels):
    classes_list = []
    classes_variance = []
    num_classes = len(np.unique(labels))
    num_feats = len(Data)

    for i in range(num_classes):
        classes_list.append(np.array(Data[:, (labels == i)]))
    
    for i in range(num_classes):
        variance_class = []
        for j in range(num_feats):
            variance_class.append(classes_list[i][j].var())
        classes_variance.append(variance_class)

    return classes_variance

def z_score(X):
    return (X - np.mean(X, axis=0))/np.std(X, axis=0)

def l2_norm(X):
    return X / np.sqrt(np.sum(X**2, axis=0))

def cov_mat(D,mu):
    DC = D - mu.reshape((mu.size, 1))
    return (1/D.shape[1])*np.dot(DC,DC.T)   

def exp_gaussian_univariate(x, sigma, mu):
    """
    This function takes a value x and returns the value of the distribution.
    """
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-((x-mu)**2/(2*sigma**2)))

def log_gaussian_univariate(x, sigma, mu):
    """
    This function takes a value x and returns the value of the distribution
    using the log multivariate formula. To avoid numerical issues due to 
    exponentiation of large numbers, in many practical cases it’s more 
    convenient to work with the logarithm of the density.
    """
    k_1 = -0.5*np.log(2*np.pi*sigma)
    k_2 = 0.5*np.log(1/sigma)
    k_3 = -(x-mu)**2/(2*(sigma**2))
    return np.exp(k_1+k_2+k_3)

def log_gaussian_multivariate(x, mu, C):
    M = x.shape[0]
    k_1 = (M*0.5)*np.log(2*np.pi)

    _,log_C = np.linalg.slogdet(C)
    k_2 = 0.5*log_C
    C_inv = np.linalg.inv(C)
    x_m = x - mu
    k_3 = 0.5*(x_m*np.dot(C_inv,x_m))
    
    return -k_1-k_2-k_3.sum(0)

def log_likelihood(x,m_ML,C_ML):
    return np.sum(log_gaussian_multivariate(x,m_ML,C_ML))