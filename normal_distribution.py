import math
import random
import unittest
from typing import List

import matplotlib.pyplot as plt


def generate_random_data(num_points: int) -> List[float]:
    """
    Generate a list of random data points following a gamma distribution.

    Args:
        num_points (int): The number of data points to generate.

    Returns:
        List[float]: A list of randomly generated data points.
    """
    # Generate random data points from a gamma distribution with shape=100 and scale=2
    return [random.gammavariate(100, 2) for _ in range(num_points)]


def empirical_distribution_function(data: List[float], x: float) -> float:
    """
    Calculate the empirical distribution function (EDF) for a given dataset at a specific point.

    Args:
        data (List[float]): The dataset for which to calculate the EDF.
        x (float): The point at which to calculate the EDF.

    Returns:
        float: The value of the empirical distribution function at the given point.
    """
    # Calculate the proportion of data points less than or equal to x
    n = len(data)
    return sum(value <= x for value in data) / n


def theoretical_distribution_function(x: float, mean: float, std_dev: float) -> float:
    """
    Calculate the theoretical cumulative distribution function (CDF) for a normal distribution.

    Args:
        x (float): The point at which to evaluate the CDF.
        mean (float): The mean of the normal distribution.
        std_dev (float): The standard deviation of the normal distribution.

    Returns:
        float: The value of the CDF at the given point.
    """
    # Compute the theoretical CDF of the normal distribution at point x
    return (1 + math.erf((x - mean) / (std_dev * math.sqrt(2)))) / 2


def kolmogorov_statistic(data: List[float], critical_value: float = 1.36, alpha: float = 0.05) -> float:
    """
    Calculate and print the Kolmogorov-Smirnov statistic for a given dataset.

    Args:
        data (List[float]): The dataset to analyze.
        critical_value (float): The critical value for the Kolmogorov-Smirnov test.
        alpha (float): The significance level.

    Returns:
        float: The Kolmogorov-Smirnov statistic for the given data.
    """
    # Calculate the mean and standard deviation of the data
    n = len(data)
    mean = sum(data) / len(data)
    std_dev = math.sqrt(sum((x - mean) ** 2 for x in data) / len(data))

    # Sort the data
    sorted_data = sorted(data)

    # Calculate the Kolmogorov-Smirnov statistic
    d_max = max(
        abs(empirical_distribution_function(sorted_data, x) - theoretical_distribution_function(x, mean, std_dev)) for x
        in sorted_data)
    print(f'Kolmogorov statistic: {round(d_max, 3)}')

    d = round(d_max * math.sqrt(len(data)), 3)
    critical_value = critical_value / math.sqrt(n)

    # Compare the statistic with the critical value and print the result
    if d < critical_value:
        print(f'Data looks normal (fail to reject H_0), critical_value = {critical_value},  d = {d}')
        print('d is less than the critical value')
    else:
        print(f'Data does not look normal (reject H_0), critical_value = {critical_value},  d = {d}')
        print('d is greater than the critical value')

    return d_max


def visualize_data(data: List[float]):
    """
    Visualize the given data using a histogram.

    Args:
        data (List[float]): The data to visualize.
    """
    # Plot a histogram of the data
    plt.hist(data, bins=20, edgecolor='black')
    plt.title('Histogram of Generated Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()


class TestNormalDistribution(unittest.TestCase):
    """A suite of unit tests for verifying the normal distribution analysis functions."""

    def test_generate_random_data(self):
        """Test the generate_random_data function for correct output length."""
        num_points = 100
        data = generate_random_data(num_points)
        self.assertEqual(len(data), num_points)

    def test_kolmogorov_statistic(self):
        """
        Test the kolmogorov_statistic function by comparing its output with the scipy implementation.
        """
        import numpy as np
        from scipy import stats

        data = generate_random_data(1000)

        # Calculate the Kolmogorov statistic for the generated data
        d_max = kolmogorov_statistic(data)

        # Use scipy's implementation of the Kolmogorov-Smirnov test for comparison
        scipy_ks_statistic = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))
        print(f'scipy_ks_statistic: {round(scipy_ks_statistic.statistic, 3)}')


        # Assert that the calculated Kolmogorov statistic is approximately equal to scipy's result
        self.assertAlmostEqual(round(d_max, 3), round(scipy_ks_statistic.statistic, 3), places=2, msg='Test failed')


if __name__ == "__main__":
    # Run the unittests when the script is executed
    unittest.main()
