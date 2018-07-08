
# Limit numpy's number of threads
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Base imports
import itertools
import json
import math
from multiprocessing import Pool
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from sklearn.model_selection import train_test_split

# Code imports
import sys
sys.path.insert(0, "/media/gregory/HDD/Projects/SLIM/Code/")

from SLIM import SLIM
from Misc import load_normalize_data, unpack_coefs

from lime import lime_tabular

###
# Run Experiments
###

datasets = ["autompgs", "happiness", "winequality-red", "housing", "day", "crimes", "music", "communities"]
trials = []
for i in range(25):
    trials.append(i + 1)
args = itertools.product(datasets, trials)

def run(args):
    # Hyperparamaters
    num_perturbations = 5
    
    # Fixes an issue where threads inherit the same rng state
    scipy.random.seed()
    
    # Arguments
    dataset = args[0]
    trial = args[1]
    
    # Output
    out = {}
    file = open("Trials/" + dataset + "_" + str(trial) + ".json", "w")

    # Load data
    X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_stddev = load_normalize_data("../Datasets/" + dataset + ".csv")
    n = X_test.shape[0]
    d = X_test.shape[1]
    
    # Load the noise scale parameters
    #with open("Sigmas/" + dataset + ".json", "r") as tmp:
        #scales = json.load(tmp)
    scales = [0.1, 0.25]
    scales_len = len(scales)
        
    # Fit SLIM model
    exp_slim = SLIM(X_train, y_train, X_valid, y_valid)
    
    # Fit LIME to explain SLIM
    exp_lime = lime_tabular.LimeTabularExplainer(X_train, discretize_continuous=False, mode="regression")

    # Evaluate model faithfullness on the test set
    rmse = 0.0 #SLIM accuracy on the dataset
    lime_rmse = np.zeros((scales_len))
    slim_rmse = np.zeros((scales_len))
    for i in range(n):
        x = X_test[i, :]
        
        #LIME's default parameter for num_samples is 500
        # 1) This is larger than any of the datasets we tested on
        # 2) It makes explaining SLIM impractically slow since the complexity of SLIM's predict() depends on the dataset size
        coefs_lime = unpack_coefs(exp_lime, x, exp_slim.predict, d, X_train, num_samples = 100)
        
        e_slim = exp_slim.explain(x)
        coefs_slim = e_slim["coefs"]
        
        rmse += (e_slim["pred"] - y_test[i])**2
        
        for j in range(num_perturbations):
            
            noise = np.random.normal(loc = 0.0, scale = 1.0, size = d)
            
            for k in range(scales_len):
                scale = scales[k]

                x_pert = x + scale * noise
            
                e_slim_pert = exp_slim.explain(x_pert)
                model_pred = e_slim_pert["pred"]
                lime_pred = np.dot(np.insert(x_pert, 0, 1), coefs_lime)
                slim_pred = np.dot(np.insert(x_pert, 0, 1), coefs_slim)

                lime_rmse[k] += (lime_pred - model_pred)**2
                slim_rmse[k] += (slim_pred - model_pred)**2

    rmse /= n
    lime_rmse /= n * num_perturbations
    slim_rmse /= n * num_perturbations

    rmse = np.sqrt(rmse)
    lime_rmse = np.sqrt(lime_rmse)
    slim_rmse = np.sqrt(slim_rmse)
    
    out["model_rmse"] = rmse[0]
    out["lime_rmse_0.1"] = lime_rmse[0]
    out["slim_rmse_0.1"] = slim_rmse[0]
    out["lime_rmse_0.25"] = lime_rmse[1]
    out["slim_rmse_0.25"] = slim_rmse[1]

    json.dump(out, file)
    file.close()

pool = Pool(12)
pool.map(run, args)

#for i in args:
    #run(i)

###
# Merge Results
###

with open("Trials/" + datasets[0] + "_" + str(trials[0]) + ".json") as f:
    data = json.load(f)

columns = list(data.keys())
df = pd.DataFrame(0, index = datasets, columns = columns)

for dataset in datasets:
    for trial in trials:
        with open("Trials/" + dataset + "_" + str(trial) + ".json") as f:
            data = json.load(f)
        for name in columns:
            df.ix[dataset, name] += data[name] / len(trials)

df.to_csv("results.csv")

###
# Stat Testing
###

file = open("stats.txt", "w")

with open("Trials/" + datasets[0] + "_" + str(trials[0]) + ".json") as f:
    data = json.load(f)

columns = list(data.keys())
df = pd.DataFrame(index = datasets, columns = columns)
df = df.apply(lambda x:x.apply(lambda x:[] if math.isnan(x) else x))

for dataset in datasets:
    for trial in trials:
        with open("Trials/" + dataset + "_" + str(trial) + ".json") as f:
            data = json.load(f)
        for name in columns:
            df.ix[dataset, name].append(data[name])

    file.write(dataset + "\n")
    file.write("Sigma = 0.1: " + str(stats.ttest_ind(df.ix[dataset, "lime_rmse_0.1"],df.ix[dataset, "slim_rmse_0.1"], equal_var = False).pvalue) + "\n")
    file.write("Sigma = 0.25: " + str(stats.ttest_ind(df.ix[dataset, "lime_rmse_0.25"],df.ix[dataset, "slim_rmse_0.25"], equal_var = False).pvalue) + "\n")

file.close()
