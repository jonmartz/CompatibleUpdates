import numpy as np
import random
import matplotlib.pyplot as plt

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def binomial(x):
    return 0.3* np.exp(-0.2*x**2) + 0.7 * np.exp(-0.2 * (x - 10)**2)

x = np.linspace(-10, 20, 100)
y = binomial(x)
y = (y / np.max(y)) * 0.17

samples = []
samples.append(np.random.normal(0, 1, 1)[0])

for i in range(0, 100000):
    x_i = samples[i]
    x_cand = np.random.normal(x_i, 1, 1)[0]
    accept = min(1, binomial(x_cand)/binomial(x_i))

    p = random.uniform(0, 1)
    if p < accept:
        samples.append(x_cand)
    else:
        samples.append(x_i)

plt.plot(x, y)
plt.hist(samples, bins=100, density=True)
plt.show()