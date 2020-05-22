import numpy as np
from pyBKT.generate import random_model_uni
from pyBKT.fit import EM_fit
import json

# get data
with open('drive/My Drive/Colab files/bkt_data.json', 'r') as file:
    data = json.load(file)
    for key, value in data.items():
        if isinstance(value, list):
            data[key] = np.array(value, dtype=np.float_)
num_subparts = data['data'].shape[0]
num_resources = len(data['resources'])

# fit model
num_fit_initializations = 5
best_likelihood = float("-inf")
for i in range(num_fit_initializations):
    print('fit number %d' % i)
    fitmodel = random_model_uni.random_model_uni(num_resources, num_subparts)
    fitmodel, log_likelihoods = EM_fit.EM_fit(fitmodel, data)
    if log_likelihoods[-1] > best_likelihood:
        print('\tnew best model')
        best_likelihood = log_likelihoods[-1]
        best_model = fitmodel

# write output
for key, value in fitmodel.items():
    if isinstance(value, np.ndarray):
        fitmodel[key] = value.tolist()
with open('drive/My Drive/Colab files/bkt_output.json', 'w') as file:
    json.dump(fitmodel, file)

print('done')
