
# Limit numpy"s number of threads
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Code imports
import sys
sys.path.insert(0, "/media/gregory/HDD/Projects/MAPLE/Code/")
from MAPLE import MAPLE
from Misc import load_normalize_data

###
# Run Experiments
###

datasets = ["autompgs","happiness", "winequality-red", "housing", "day", "music", "crimes", "communities"]
trials = []
for i in range(50):
    trials.append(i + 1)
args = itertools.product(datasets, trials)

def run(args):
    
    # Fixes an issue where threads of inherit the same rng state
    scipy.random.seed()
    
    # Arguments
    dataset = args[0]
    trial = args[1]
    
    # Outpt
    out = {}
    file = open("Trials/" + dataset + "_" + str(trial) + ".json", "w")

    # Load Data
    X_train, y_train, X_val, y_val, X_test, y_test, train_mean, train_stddev = load_normalize_data("../Datasets/" + dataset + ".csv")

    # Linear model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    predictions = lr.predict(X_test)
    out["lm_rmse"] = np.sqrt(mean_squared_error(y_test, predictions))

    # RF
    maple_rf = MAPLE(X_train, y_train, X_val, y_val)

    predictions = maple_rf.predict_fe(X_test)
    out["rf_rmse"] = np.sqrt(mean_squared_error(y_test, predictions))

    predictions = maple_rf.predict_silo(X_test)
    out["silo_rf_rmse"] = np.sqrt(mean_squared_error(y_test, predictions))

    predictions = maple_rf.predict(X_test)
    out["maple_rf_rmse"] = np.sqrt(mean_squared_error(y_test, predictions))
    out["nf_rf"] = maple_rf.retain

    # GBRT
    maple_gbrt = MAPLE(X_train, y_train, X_val, y_val, fe_type = "gbrt")

    predictions = maple_gbrt.predict_fe(X_test)
    out["gbrt_rmse"] = np.sqrt(mean_squared_error(y_test, predictions))

    predictions = maple_gbrt.predict_silo(X_test)
    out["silo_gbrt_rmse"] = np.sqrt(mean_squared_error(y_test, predictions))
    
    predictions = maple_gbrt.predict(X_test)
    out["maple_gbrt_rmse"] = np.sqrt(mean_squared_error(y_test, predictions))
    out["nf_gbrt"] = maple_gbrt.retain

    # Save results
    json.dump(out, file)
    file.close()

pool = Pool(12)
pool.map(run, args)

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
    file.write("MAPLE vs RF: " + str(stats.ttest_ind(df.ix[dataset, "rf_rmse"],df.ix[dataset, "maple_rf_rmse"], equal_var = False).pvalue) + "\n")
    file.write("MAPLE vs SILO: " + str(stats.ttest_ind(df.ix[dataset, "maple_rf_rmse"],df.ix[dataset, "silo_rf_rmse"], equal_var = False).pvalue) + "\n")
    file.write("MAPLE vs GBRT: " + str(stats.ttest_ind(df.ix[dataset, "gbrt_rmse"],df.ix[dataset, "maple_gbrt_rmse"], equal_var = False).pvalue) + "\n")
    file.write("MAPLE vs SILO: " + str(stats.ttest_ind(df.ix[dataset, "maple_gbrt_rmse"],df.ix[dataset, "silo_gbrt_rmse"], equal_var = False).pvalue) + "\n")

file.close()
