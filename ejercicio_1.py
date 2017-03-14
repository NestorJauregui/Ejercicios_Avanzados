import numpy as np
import matplotlib.pyplot as plt
import random as rm

pos = np.array([1.5, 1.7, 2.0])
a = 1.0
b = 20.0
lamb = np.linspace(0.01, 1000.0, 10000)
evidence = 1.0
prior = 1.0/1000.0

def likelihood(pos, z, lamb):
    mult = 1.0
    for i in range(np.size(pos)):
        mult = mult*np.exp(-pos[i]/lamb)/z
    return mult

def z(lamb, a, b):
    return lamb*np.exp(-a/lamb)-lamb*np.exp(-b/lamb)

def proba(pos, a, b, lamb, prior):
    return likelihood(pos, z(lamb, a, b), lamb)*prior/evidence

func = proba(pos, a, b, lamb, prior)
evidence = 0.5*np.sum(func)*(lamb[1]-lamb[0])
func = func/evidence

lambda_walk = np.empty((0))
l_walk = np.empty((0))

lambda_walk = np.append(lambda_walk, rm.random())
#l_walk = np.append(l_walk, rm.random())

niter = 20000
for i in range(niter):
    lambda_prime = np.random.normal(lambda_walk[i], 0.4)
    alpha = proba(pos, a, b, lambda_prime, prior)/proba(pos, a, b, lambda_walk[i], prior)
    if alpha >= 1.0:
        lambda_walk = np.append(lambda_walk, lambda_prime)
    else:
        beta = rm.random()
        if beta <= alpha:
            lambda_walk = np.append(lambda_walk, lambda_prime)
        else:
            lambda_walk = np.append(lambda_walk, lambda_walk[i])

plt.plot(lamb, func)
count, bins, ignored = plt.hist(lambda_walk, 1000, normed=True)
ax = plt.axes()
ax.set_xlim([0.0, 20.0])
plt.show()
