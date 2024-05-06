#!/usr/bin/env python

import numpy as np
from sparsemodels.functions import parallel_sgcca

# Set random seed for reproducibility
np.random.seed(42)

# Sample Charaterizics
n_samples = 1000
n_vars = 10

# RandomSite
SITES = np.repeat(['Site1', 'Site2', 'Site3', 'Site4', 'Site5'], repeats=200)

# Generate data for each view
views_data = []
for i in range(8):
    # Generate random data for each view
    view_data = np.random.randn(n_samples, n_vars)
    # Introduce correlation between views
    view_data[:, :2] += i * 0.5  # Increasing correlation
    views_data.append(view_data)

# Ensure that the data is a list
views = list(views_data)


# Create a model object using parallel_sgcca from sparsemodels.
model = parallel_sgcca(n_jobs = 12, scheme = "factorial", n_permutations = 1000, design_matrix = None)

# Create nfolds in the data with approximately equal sampling across all eight sites. 
# The data will also  be split with randomly split 70% of data with designated as the training sample (for model fitting) with 30% of the sample
model.create_nfold(group=SITES, n_fold=10, holdout=0.3)

# With optimal sparsity and number of component selected, the training model is now calculated.
model.fit_model(views, n_components = 3, l1_sparsity = 0.3)

# model variance explained

print(model.model_obj_.AVE_inner_)
# array([0.00244925, 0.00217574, 0.00169972])

print(model.model_obj_.AVE_outer_)
# array([0.10163285, 0.1011095 , 0.10067549])

print(model.model_obj_.AVE_views_)
# array([[0.10176376, 0.1014266 , 0.10118082],
#        [0.10169056, 0.10005755, 0.10068283],
#        [0.10055597, 0.10104704, 0.10074827],
#        [0.10134403, 0.10102171, 0.10065261],
#        [0.10144607, 0.10146916, 0.10055249],
#        [0.10289514, 0.10126118, 0.10028722],
#        [0.10102153, 0.10097951, 0.10113573],
#        [0.10234573, 0.10161325, 0.10016396]])

# print component 1 canonical correlations
print(np.round(np.corrcoef(model.train_scores_[:,:,0]), 3))

# array([[ 1.   ,  0.024,  0.04 , -0.049,  0.077,  0.072, -0.046, -0.062],
#        [ 0.024,  1.   , -0.046,  0.091, -0.053,  0.046,  0.037, -0.012],
#        [ 0.04 , -0.046,  1.   ,  0.081, -0.01 , -0.008, -0.046,  0.049],
#        [-0.049,  0.091,  0.081,  1.   , -0.   , -0.022, -0.014,  0.08 ],
#        [ 0.077, -0.053, -0.01 , -0.   ,  1.   ,  0.037,  0.011,  0.017],
#        [ 0.072,  0.046, -0.008, -0.022,  0.037,  1.   ,  0.083,  0.042],
#        [-0.046,  0.037, -0.046, -0.014,  0.011,  0.083,  1.   , -0.039],
#        [-0.062, -0.012,  0.049,  0.08 ,  0.017,  0.042, -0.039,  1.   ]])

print(np.round(np.corrcoef(model.test_scores_[:,:,0]), 3))

# [[ 1.     0.015 -0.031  0.133 -0.018  0.067 -0.097  0.05 ]
#  [ 0.015  1.    -0.08   0.038  0.055  0.012 -0.09  -0.015]
#  [-0.031 -0.08   1.     0.016  0.072  0.017 -0.041  0.059]
#  [ 0.133  0.038  0.016  1.     0.026 -0.03  -0.005  0.036]
#  [-0.018  0.055  0.072  0.026  1.    -0.073  0.08  -0.05 ]
#  [ 0.067  0.012  0.017 -0.03  -0.073  1.    -0.039  0.064]
#  [-0.097 -0.09  -0.041 -0.005  0.08  -0.039  1.    -0.029]
#  [ 0.05  -0.015  0.059  0.036 -0.05   0.064 -0.029  1.   ]]

# The shape of the data: n_views, n_subjects, n_components
print(model.train_scores_.shape)
# (8, 700, 3)

print(model.test_scores_.shape)
# (8, 300, 3)
