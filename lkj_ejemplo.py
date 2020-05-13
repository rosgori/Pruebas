#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import os
# #   # This must happen before pymc is imported, so you might
# #   # need to restart the kernel for it to take effect.
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'


import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import scipy
import theano
import arviz as az


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

print('Versión de scipy {}'.format(scipy.__version__))
print('Versión de numpy {}'.format(np.__version__))
print(np.__config__.show())


SEED = 3264602 # from random.org
np.random.seed(SEED)

N = 1_000

μ_actual = np.array([1, -2])
Σ_actual = np.array([[0.5, -0.3],
                     [-0.3, 1.]])

x = np.random.multivariate_normal(μ_actual, Σ_actual, size=N)

with pm.Model() as model:
    packed_L = pm.LKJCholeskyCov('packed_L', n=2,
                                  eta=2., sd_dist=pm.HalfCauchy.dist(2.5))

    L = pm.expand_packed_triangular(2, packed_L)
    Σ = pm.Deterministic('Σ', L.dot(L.T))

    μ = pm.Normal('μ', 0., 10., shape=2,
                  testval=x.mean(axis=0))
    obs = pm.MvNormal('obs', μ, chol=L, observed=x)


with model:
    #trace = pm.sample(random_seed=SEED, cores=1, chains=4)
    trace = pm.sample(draws=2000, random_seed=SEED)


print(pm.summary(trace))


# x = np.random.normal(loc=0, scale=2, size=N)

# with pm.Model() as model:
    
#     mu = pm.Normal('mu', 0, 5)
#     sigma = pm.HalfNormal('sigma', sd=5)
    
#     obs = pm.Normal('obs', mu=mu, sd=sigma, observed=x)
    
# with model:
#     trace = pm.sample(draws=100, tune=200, init='nuts', n_init=100)
    
# print(pm.summary(trace))
    



#func = model.logp_dlogp_function()
#the = theano.printing.debugprint(func._theano_function)
#print(the)






#
#import os
## This must happen before pymc is imported, so you might
## need to restart the kernel for it to take effect.
#os.environ['MKL_NUM_THREADS'] = '1'
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#
#
#import matplotlib.pyplot as plt
#import numpy as np
#import pymc3 as pm
#import scipy
#
#print('Versión numpy:', np.__version__)
#print('Versión scipy:', scipy.__version__)
#
#SEED = 3264602
#np.random.seed(SEED)
#
#N = 10000
#
#μ_actual = np.array([1, -2])
#Σ_actual = np.array([[0.5, -0.3],
#                     [-0.3, 1.]])
#
#x = np.random.multivariate_normal(μ_actual, Σ_actual, size=N)
#
#with pm.Model() as model:
#    packed_L = pm.LKJCholeskyCov('packed_L', n=2,
#                                 eta=2., sd_dist=pm.HalfCauchy.dist(2.5))
#
#    L = pm.expand_packed_triangular(2, packed_L)
#    Σ = pm.Deterministic('Σ', L.dot(L.T))
#
#    μ = pm.Normal('μ', 0., 10., shape=2,)
#                  #testval=x.mean(axis=0))
#    obs = pm.MvNormal('obs', μ, chol=L, observed=x)
#
#func = model.logp_dlogp_function(profile=True)
#
#func.set_extra_values({})
#vec = np.random.randn(func.size)
#for _ in range(10000):
#    func(vec)
#
#p = func.profile.summary_ops(N=10)
#print(p)
