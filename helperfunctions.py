import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt



def random_inputs_outputs_test_generator(N = 12, var = 1.0, N_test = 5, seed = None):   
    
    np.random.seed(seed)
    inputs = 5 * np.random.rand(N,1)
    outputs = np.sin(12*inputs) + 0.66*np.cos(25*inputs) + np.random.randn(N,1)*0.1
    test_inputs = np.random.rand(N_test,1)
    test_outputs = np.sin(12*test_inputs) + 0.66*np.cos(25*test_inputs) + np.random.randn(N_test,1)*0.1
    
    return inputs, outputs, test_inputs, test_outputs


def RBF(X, Y, var = 1.0, lengthscale = 1.0):
    
#    kx1x2 = var * np.exp(-0.5 * cdist(X, Y, 'sqeuclidean') / lengthscale**2)
    kx1x2 = var * np.exp(-0.5 * np.square(cdist(X, X)) / lengthscale**2)
    
    return kx1x2


def prior(kernel, n_samples = 5):
    
    # compute prior  
    Xplot = np.expand_dims(np.linspace(0, 5, 100), 1) # 100 points in each sample by default
    mean = np.zeros(len(Xplot))
    Kprior = RBF(Xplot, Xplot)
    prior = np.random.multivariate_normal(mean, Kprior, n_samples)    
    
    # plot samples

    for i in range(n_samples):
        plt.plot(Xplot, prior[i], linestyle = '-', marker = 'x', markersize = 0)
    
    return None


def posterior (kernel, outputs, var = 1.0, lengthscale = 1.0, n_samples = 5):
    
    global local_vars
    
    Xplot = np.expand_dims(np.linspace(0, 5, 100), 1) # 100 points in each sample by default
    Xtest = np.expand_dims(np.linspace(0, 5, 100), 1) # 80 test input points in by default
    
    KXX = RBF(Xplot, Xplot)
    KXXt = RBF(Xplot, Xtest)
    KXtXt = RBF(Xtest, Xtest)
    
        
    mean_posterior = np.matmul(np.matmul(KXXt, np.linalg.inv(KXX)), outputs)
    cov_posterior = KXtXt - np.matmul( np.matmul(KXtXt, KXX), KXXt )
       
    posterior = np.random.multivariate_normal(mean_posterior.T[0], cov_posterior, 5, 'warn')
    
    Xplot = np.expand_dims(np.linspace(0, 5, 50), 1)

    for i in range(n_samples):
        plt.plot(Xplot, posterior[i], linestyle = '-', marker = 'x', markersize = 0)

    
    return mean_posterior, cov_posterior
