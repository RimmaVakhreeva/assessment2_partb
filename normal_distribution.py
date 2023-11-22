import random
import math
import matplotlib.pyplot as plt

import numpy as np
from scipy import stats

#Generate random data
def generate_random_data(num_points):
    return [random.gauss(0, 1) for _ in range(num_points)]


def calculate_mean(data):
    return sum(data) / len(data)


def calculate_std_dev(data, mean_value):
    return math.sqrt(sum((x - mean_value) ** 2 for x in data) / len(data))


# Calculate the Empirical Distribution Function
def empirical_distribution_function(data, x):
    sorted_data = sorted(data)
    n = len(data)
    # Counts how many data points are less than or equal to x
    return sum(value <= x for value in sorted_data) / n


# Approximate the Cumulative Distribution Function of the Normal Distribution
def phi(x, mu, sigma):
    return (1 + math.erf((x - mu) / (sigma * math.sqrt(2)))) / 2


#Apply the Kolmogorov Test
def kolmogorov_test(data, mu, sigma):
    d_max = max(abs(empirical_distribution_function(data, x)) - phi(x, mu, sigma) for x in data)
    return d_max


# Creating the histogram
def visualize_data(data):
    plt.hist(data, bins=20, edgecolor='black')  # Adjust the number of bins as needed

    # Adding titles and labels
    plt.title('Histogram of Generated Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Show the plot
    return plt.show()


# Generating 1000 random data points
data = generate_random_data(1000)

# Calculate mean and standard deviation of the sample
sample_mean = calculate_mean(data)
sample_std_dev = calculate_std_dev(data, sample_mean)


# Calculate Kolmogorov test

d_max = kolmogorov_test(data, sample_mean, sample_std_dev)
print(f'Kolmogorov statistic: {round(d_max, 4)}')

# Interpret the results

alpha = 0.05 # significance level
critical_value = 1.36

d = round(d_max * math.sqrt(len(data)), 4)

if d < critical_value:
    print(f'Data looks normal (fail to reject H_0), critical_value is {critical_value},  d is {d}')
else:
    print(f'Data does not look normal (reject H_0), critical_value is {critical_value},  d is {d}')

visualize_data(data)


# Perform the Kolmogorov-Smirnov test using libraries
ks_statistic, p_value = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))

print(f"KS Statistic: {ks_statistic}")
print(f"P-Value: {p_value}")