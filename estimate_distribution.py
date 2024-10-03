import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

def find_best_fit_distribution(data):
    # Fit distributions
    normal_params = stats.norm.fit(data)     # Fit Normal distribution
    uniform_params = stats.uniform.fit(data)  # Fit Uniform distribution
    lambda_exp = 1 / np.mean(data)

    # Generate PDFs from fitted parameters
    x = np.linspace(min(data), max(data), 100)
    normal_pdf = stats.norm.pdf(x, *normal_params)
    p_exp = stats.expon.pdf(x, scale=1/lambda_exp)
    poisson_pmf = stats.poisson.pmf(np.round(x), np.mean(data))
    uniform_pdf = stats.uniform.pdf(x, *uniform_params)

    # Kolmogorov-Smirnov test for normal distribution
    ks_normal = stats.kstest(data, 'norm', args=normal_params)
    print("K-S Test for Normal:", ks_normal)

    best_fit_distribution = stats.norm(normal_params)
    best_p_value = ks_normal.pvalue
    # K-S test for Poisson
    ks_poisson = stats.kstest(data, 'poisson', args=[np.mean(data)])
    print("K-S Test for Poisson:", ks_poisson)
    if(ks_poisson.pvalue < best_p_value):
        best_fit_distribution = stats.poisson(np.mean(data))
        best_p_value = ks_poisson.pvalue

    # K-S test for Exponential
    ks_exponential = stats.kstest(data, 'expon', args=[lambda_exp])
    print("K-S Test for Exponential:", ks_exponential)
    if (ks_exponential.pvalue < best_p_value):
        best_fit_distribution = stats.expon(scale=1/lambda_exp)
        best_p_value = ks_exponential.pvalue


    # K-S test for Uniform
    ks_uniform = stats.kstest(data, 'uniform', args=uniform_params)
    print("K-S Test for Uniform:", ks_uniform)
    if (ks_uniform.pvalue < best_p_value):
        best_fit_distribution = stats.uniform(np.mean(data))
        best_p_value = ks_uniform.pvalue

    return best_fit_distribution

print(find_best_fit_distribution([1,2,3,4,6,5,5,8,63,4]))