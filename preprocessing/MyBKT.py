import numpy as np
from pyBKT.generate import random_model_uni
from pyBKT.fit import EM_fit
import json

dataset_dir = 'C:/Users/Jonathan/Documents/BGU/Research/Thesis/DataSets/ednet'
with open('%s/bkt_data.json' % dataset_dir, 'r') as file:
    data = json.load(file)

for key, value in data.items():
    if isinstance(value, list):
        data[key] = np.array(value)

num_subparts = data['data'].shape[0]
num_resources = data['num_resources']

max_num_fit_initializations = 1
best_likelihood = float("-inf")

fitmodel = random_model_uni.random_model_uni(num_resources, num_subparts)
(fitmodel, log_likelihoods) = EM_fit.EM_fit(fitmodel, data)

# write output
for key, value in fitmodel.items():
    if isinstance(value, np.ndarray):
        fitmodel[key] = value.tolist()
with open('%s/bkt_output.json' % dataset_dir, 'w') as file:
    json.dump(fitmodel, file)

print('done')
