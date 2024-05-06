#!/usr/bin/env python

import os
import sys
import pickle
import glob
import itertools
import matplotlib
import numpy as np
import nibabel as nib
import pandas as pd
from random import choices

#from pyggseg.functions import ggseg_plot_glasser_bluered, convert_rywlbb_gradient
from PyCerebralPlots.functions import create_rywlbb_gradient_cmap
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.stats import pearsonr, boxcox, norm
from sklearn.preprocessing import scale
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.preprocessing import scale
from sklearn.linear_model import TheilSenRegressor
from joblib import Parallel, delayed, wrap_non_picklable_objects
from tfce_mediation.pyfunc import stack_ones, dummy_code

from sparsemodels.functions import generate_seeds, sgcca_rwrapper, parallel_sgcca, pickle_save_model, pickle_load_model
from sparsemodels.cynumstats import cy_lin_lstsqr_mat

from sparsemodels.functions import plot_ncomponents, plot_parameter_selection, plot_prediction_bootstraps, plot_scatter_histogram, plot_pairscores, plot_permuted_model

def neglog_transformation(x):
	return(np.sign(x)*np.log10(np.abs(x)+1))

def _theilsenregression(t, X, y, n_jobs = 12,  seed = None):
	"""
	Selection of best number of components using permutation testing.
	"""
	if seed is None:
		np.random.seed(np.random.randint(4294967295))
	else:
		np.random.seed(seed)
	if t % 100 == 0:
		print(t)
	nonbiasreg = TheilSenRegressor(n_jobs = n_jobs).fit(X, y)
	return(y - nonbiasreg.predict(X))

def theilsenregression_reduce_data(X, y, n_jobs = 12):
	nsamples, ntargets = y.shape
	seeds = generate_seeds(ntargets)
	yresid = Parallel(n_jobs = n_jobs, backend='multiprocessing')(delayed(_theilsenregression)(t, X = X, y  = y[:,t], n_jobs = n_jobs, seed = seeds[t]) for t in range(ntargets))
	yresid = np.array(yresid).T
	return(yresid)

# import demographics data
CSV = 'subset_IMAGEN-FU3_SexAgeSite_checked.csv'
pdCSV = pd.read_csv(CSV, delimiter=',')

SITES = np.array(pdCSV.Site)
uSITES = np.unique(SITES)
SEX = np.array(pdCSV.Sex)
AGE = np.array(pdCSV.Age)
Diagnosis = np.array(pdCSV.Diagnosis)

# dummpy code covariates
covariates = np.column_stack((dummy_code(SEX, iscontinous=False, demean=True), dummy_code(SITES, iscontinous=False, demean=True)))
covariates = np.column_stack((covariates, dummy_code(AGE, iscontinous=True, demean=True)))

# create an empty list for appending each data-view
views = []

# Clinical data view
CSV = 'subset_IMAGEN-FU3_clincal_dawba-sdq-audit.csv'
pdCSV = pd.read_csv(CSV, delimiter=',')
Xraw = np.array(pdCSV.iloc[:, 1:])

# remove the effect of the covariates, and append the the data to the first data-view
Xreduced = scale(theilsenregression_reduce_data(covariates, neglog_transformation(Xraw), n_jobs = 12))
views.append(Xreduced)

# Emotional Face Task data view
pdCSV = pd.read_csv('subset_IMAGEN-FU3-TASK_ward300_EFT_ANGRY_CONTROL.csv', delimiter=',')
Xraw =np.array(pdCSV.iloc[:, 1:])
Xreduced = scale(theilsenregression_reduce_data(covariates, Xraw, n_jobs = 12))
views.append(Xreduced)

# Monetary Incentive Delay data view
pdCSV = pd.read_csv('subset_IMAGEN-FU3-TASK_ward300_MID_antLwin_Nowin.csv', delimiter=',')
Xraw = np.array(pdCSV.iloc[:, 1:])
Xreduced = scale(theilsenregression_reduce_data(covariates, Xraw, n_jobs = 12))
views.append(Xreduced)

# Stop-Signal Task data view
pdCSV = pd.read_csv('subset_IMAGEN-FU3-TASK_ward300_SST_stop_sucess.csv', delimiter=',')
Xraw = np.array(pdCSV.iloc[:, 1:])
Xreduced = scale(theilsenregression_reduce_data(covariates, Xraw, n_jobs = 12))
views.append(Xreduced)

# Cortical Thickness data view
pdCSV = pd.read_csv('subset_IMAGEN_FU3_surf-HCP-MMP1_CT.csv', delimiter=',')
Xraw =np.array(pdCSV.iloc[:, 1:])
Xreduced = scale(theilsenregression_reduce_data(covariates, Xraw, n_jobs = 12))
views.append(Xreduced)

# Surface Area data view
pdCSV = pd.read_csv('subset_IMAGEN_FU3_surf-HCP-MMP1_SA.csv', delimiter=',')
Xraw =np.array(pdCSV.iloc[:, 1:])
Xreduced = scale(theilsenregression_reduce_data(covariates, Xraw, n_jobs = 12))
views.append(Xreduced)

# Resting State Modes data view
pdCSV = pd.read_csv('subset_IMAGEN-FU3-REST_modes_normr.csv', delimiter=',')
Xraw =np.array(pdCSV.iloc[:, 1:])
Xreduced = scale(theilsenregression_reduce_data(covariates, Xraw, n_jobs = 12))
views.append(Xreduced)

# White Matter FA data view
pdCSV = pd.read_csv('subset_IMAGEN-FU3-DTI_ward100_fa.csv', delimiter=',')
ROI_names = np.array(pdCSV.columns[1:])
Xraw =np.array(pdCSV.iloc[:, 1:])
Xreduced = scale(theilsenregression_reduce_data(covariates, Xraw, n_jobs = 12))
views.append(Xreduced)

# Ensure that the data is a list
views = list(views)

# Save the view data as a pickle object
pickle_save_model(views, "views.pkl")

# Create a model object using parallel_sgcca from sparsemodels.
model = parallel_sgcca(n_jobs = 12, scheme = "factorial", n_permutations = 10000, design_matrix = None)

# Create nfolds in the data with approximately equal sampling across all eight sites. 
# The data will also  be split with randomly split 70% of data with designated as the training sample (for model fitting) with 30% of the sample
model.create_nfold(group=SITES, n_fold=10, holdout=0.3)

# Run the parallel_parameterselection which used to estimate the optimal l1 spartsity. Ranging from 0.1 to 1.0 at 0.1 increments, the model will be fitted
# in the training data, and then 1000 permutation with be performed. The metric values divided standard deviation of the permutated metric values is used
# to create a z-statistic. The largest z-statistic from all increments is set as the 'model.parameterselection_bestpenalties_'
model.run_parallel_parameterselection(views = views, verbose=True, n_perm_per_block=1000, metric = 'objective_function')

# Plot the parameter selection
plot_parameter_selection(model)

# For selecting the optimal number of components a 50 component SGCCA model was generated with the optimal sparsity and plotted to find the approximate 'elbow' in
# in the cululative variance explained 
plot_ncomponents(model, views = views,
								max_n_comp = 50,
								l1_sparsity = model.parameterselection_bestpenalties_,
								labels = labels,
								png_basename = m_name)

# With optimal sparsity and number of component selected, the training model is now calculated
model.fit_model(views, n_components = 10, l1_sparsity = model.parameterselection_bestpenalties_)

# After the initial model is created, stability selection. Stability selection subsamples 50% of the the training data randomly and the model is re-fit.
# Only variables that appear in 90% of the models are kept. 
model.run_parallel_stability_selection(n_bootstraps = 10000, consistency_threshold = 0.9, fit_feature_selected_model = True)

# After stability selection the model, the significance of the model components are assessed in two ways: (1) in the training data the average inner variance
# explained is compared to the compared to the variance explained in 10000 generated permuted models. (2) The average inner variance explained by the transformed components
# of the test data in the by the real model compared to average inner variance explained by 10000 permuted models. Significance is determined by rank-ordering
# the real model average inner variance explained compared to the null values for the training and test data, respectively.
model.run_parallel_permute_model(metric = 'AVE_inner')


# SGCCA-regression model in which the clinical data view component score is the response variables ('y') and the the neuroimaging scores are the predictor variable
# significance of the model and their coefficients are determined by bootstrapping.
model.bootstrap_prediction_model(response_index = 0, n_bootstraps = 10000)

# plot the SGCCA-regression model 
plot_prediction_bootstraps(model, png_basename = "prediction_model")

# calculate the significance of each model loading (correlation between component score and each item in the respective data view) using bootstraping
model.bootstrap_model_loadings()

# save the model
pickle_save_model(model, "model.pkl")

# import the view data for the stratify sample (similar to lines 58-127). 
views_data_stratify = pickle_load_model("views_data_stratify.pkl")
# calculate the component scores for stratify. After which the canonical correlation among the scores, and SGCCA-regression model can be calculated. 
scores_stratify = model.model_obj_.transform(model.views_)

# import the view data for the imagen sample at all timepoints (similar to lines 58-127). After which the canonical correlation among the scores, and SGCCA-regression model can be calculated. 
views_data_stratify = pickle_load_model("views_data_imagen_longitudinal.pkl")
scores_stratify = model.model_obj_.transform(model.views_)