import numpy as np
from matplotlib import pyplot as plt
from helperfunctions import random_inputs_outputs_test_generator, RBF, prior, posterior

# Dataset and cov matrix (for posterior)

N = 3   # Number of data points
N_test = 40 # Number of test points
gen_var = 1.0 # variance in the generation of data points

X, Y, Xt, Yt = random_inputs_outputs_test_generator(N, gen_var, N_test)

plt.figure(0)
plt.plot(X, Y, 'ro', mew=2)


# Prior and samples

n_samples = 5 # Number of samples to plot
var = 1.0
lengthscale = 1.0

plt.figure(1)
plt.title("Prior samples")
plt.plot(X, Y, 'ro', mew=2)

Xplot = np.expand_dims(np.linspace(0, 5, 100), 1) # 100 points in each sample by default

prior = prior(RBF, Xplot, var, lengthscale, n_samples)

plt.ylim([-3, 3])
    

# plot prior samples
    
for i in range(n_samples):
    plt.plot(Xplot, prior[i], linestyle = '-', marker = 'x', markersize = 0)
 
plt.plot(Xplot, np.zeros(len(Xplot)), linestyle = '-', color ='blue')       
plt.plot(Xplot, var * np.ones(len(Xplot)), linestyle = 'dashed', color = 'red')
plt.plot(Xplot, -1.0 * var * np.ones(len(Xplot)), linestyle = 'dashed', color = 'red')
plt.legend(['training points', "mean", 'standard dev.'], loc ="upper right")


# Posterior

# Covariance Matrices

K_XX = RBF(X, X)
K_XtXt = RBF(Xt, Xt)
K_XXt = RBF(X, Xt)

mean_posterior, cov_posterior = posterior(RBF, X, Y, Xt, var = 1.0, lengthscale = 1.0)

mean_posterior_plot, cov_posterior_plot = posterior(RBF, X, Y, Xplot)

posterior_plot = np.random.multivariate_normal(mean_posterior_plot.T[0], cov_posterior_plot, n_samples)

yplot_var = cov_posterior_plot.diagonal
print(yplot_var)

plt.figure(2)
plt.title("Mean posterior")
plt.plot(X, Y, 'ro', mew=2)
plt.plot(Xplot, mean_posterior_plot.T[0], linestyle = '-', marker = 'x', markersize = 0, color = 'blue')

plt.figure(3)
plt.title("Posterior samples")
plt.plot(X, Y, 'ro', mew=2)
for i in range(n_samples):    
    plt.plot(Xplot, posterior_plot[i], linestyle = '-', marker = 'x', markersize = 0)
    
covariances = cov_posterior_plot.diagonal()

f = np.sin(2*Xplot) + np.cos(Xplot)

plt.figure(4)
plt.title("Posterior distribution")
plt.plot(X, Y, 'ro', mew=2,markersize = 7)# label='_nolegend_')                
plt.plot(Xplot, f, linestyle = '-' ,color = 'green', markersize = 30)
plt.plot(Xplot, mean_posterior_plot.T[0], linestyle = '-', marker = 'x', markersize = 0 ,color ='blue')
plt.plot(Xplot, mean_posterior_plot.T[0] + covariances, linestyle = 'dashed', color = 'red')
plt.plot(Xplot, mean_posterior_plot.T[0] - covariances, linestyle = 'dashed', color = 'red')
plt.legend(['training points', 'f(x)', "mean function", 'standard dev.'], loc ="upper right")
