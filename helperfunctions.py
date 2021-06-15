import numpy as np
from scipy.spatial.distance import cdist
from scipy import linalg



def random_inputs_outputs_test_generator (N = 12, var = 1.0, N_test = 10, seed = None):   
    
    np.random.seed(seed)
    inputs = 5 * np.random.rand(N,1)
    outputs = np.sin(2*inputs) + np.cos(inputs) +np.random.randn(N,1)*0.4 #☻ np.sin(12*inputs) + 0.66*np.cos(25*inputs) + np.random.randn(N,1)*0.1
    test_inputs = 5 * np.random.rand(N_test,1)
    test_outputs = np.sin(2*inputs) + np.cos(inputs) +np.random.randn(N,1)*0.4
    
    return inputs, outputs, test_inputs, test_outputs



def RBF(X, Y, var = 1.0, lengthscale = 1.0):
    
    kx1x2 = var * np.exp(-0.5 * np.square(cdist(X, Y)) / lengthscale**2) #0.6 cool
    
    return kx1x2



def prior (kernel, inputs, var = 1.0, lengthscale = 1.0, n_samples = 6):
    
    # compute prior  
    var = 1.0
    lenngthscale = 1.0
    mean = np.zeros(len(inputs))
    Kprior = RBF(inputs, inputs, var, lenngthscale)
    prior = np.random.multivariate_normal(mean, Kprior, n_samples)    


    return prior



def posterior (kernel, inputs, outputs, test_inputs, var = 1.0, lengthscale = 1.0, n_samples = 5):    
    
    
    KXX = kernel(inputs, inputs, var, lengthscale)
    KXXt = kernel(inputs, test_inputs, var, lengthscale)
    #print(KXXt.T == KXtX)
    #print('Kxxt', KXXt.shape)
    KXtXt = RBF(test_inputs, test_inputs, var, lengthscale)
    
    sigma = 0.1
           
    mean_posterior = np.matmul(linalg.solve((KXX + (sigma**2) * np.identity(KXX.shape[0])), KXXt).T, outputs) #np.matmul(a, outputs)
                    # KXXt.T.dot(K_inv).dot(Y_train)
    cov_posterior = KXtXt - np.matmul(linalg.solve((KXX + (sigma**2) * np.identity(KXX.shape[0])), KXXt).T, KXXt)# + sigma**2 * np.identity(KXtXt.shape[0]) - np.matmul(np.matmul(KXXt.T, KXX + sigma**2 * np.identity(KXX.shape[0])), KXXt )

    
    return mean_posterior, cov_posterior
