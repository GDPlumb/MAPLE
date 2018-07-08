
# Base imports
import json
import math
import pandas as pd
import scipy
from scipy import stats

datasets = ['autompgs' , 'happiness', 'winequality-red', 'housing', 'day', 'crimes', 'music', 'communities']
trials = []
for i in range(25):
    trials.append(i + 1)

with open('SVR/Trials/' + datasets[0] + '_' + str(trials[0]) + '.json') as f:
    data = json.load(f)

columns = list(data.keys())
df = pd.DataFrame(index = datasets, columns = columns)
df = df.apply(lambda x:x.apply(lambda x:[] if math.isnan(x) else x))

for dataset in datasets:
    for trial in trials:
        with open('SVR/Trials/' + dataset + '_' + str(trial) + '.json') as f:
            data = json.load(f)
        for name in columns:
            df.ix[dataset, name].append(data[name])

    print(dataset)
    print('Sigma = 0.1', stats.ttest_ind(df.ix[dataset, 'lime_rmse_0.1'],df.ix[dataset, 'slim_rmse_0.1'], equal_var = False).pvalue)
    print('Sigma = 0.25', stats.ttest_ind(df.ix[dataset, 'lime_rmse_0.25'],df.ix[dataset, 'slim_rmse_0.25'], equal_var = False).pvalue)




