import numpy as np
from matplotlib import pyplot as plt
from helperfunctions import random_inputs_outputs_test_generator, RBF, prior, posterior

# Dataset and cov matrix (for posterior)

N = 10 # Number of data points
gen_var = 1.0 # variance in the generation of data points

X, Y, Xt, Yt = random_inputs_outputs_test_generator(N, gen_var, seed = 4)

plt.plot(X, Y, 'ro', mew=2)



# Prior and samples

samples = 10 # Number of samples to plot

#Xplot = np.expand_dims(np.linspace(-7, 7, 80), 1) # 80 points in each sample by default
#Kprior = RBF(Xplot, Xplot)
#mean = np.zeros(len(Xplot))

prior(RBF, samples)



# Posterior

# Covariance Matrices

#K_XX = Kprior # , var = 1.0, lengthscale = 1.0)
K_XtXt = RBF(Xt, Xt) # , var = 1.0, lengthscale = 1.0)
K_XXt = RBF(X, Xt) # , var = 1.0, lengthscale = 1.0)

posterior(RBF, Y) #, var = 1.0, lengthscale = 1.0, n_samples = 5)

# mean_posterior, cov_posterior = posterior (K_XX, K_XXt, K_XtXt, Y, var = 1.0, lengthscale = 1.0)
# print(mean_posterior.T[0])
# print(mean)
    


