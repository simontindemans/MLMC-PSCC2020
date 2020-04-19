"""
Helper class for storing univariate and bivariate summary statistics on the fly.
author: Simon Tindemans, s.h.tindemans@tudelft.nl
"""
# SPDX-License-Identifier: MIT


import numpy as np
import math

class SummaryStats:
    """
    SummaryStats computes and stores summary statistics (mean, variance, and related concepts) of data.

    Data elements are assumed to be of type numpy.ndarray, although this is not checked and related objects may work.
    """
    def __init__(self, shape=(1), data=None, exact_mean=None):
        """
        Initialise SummaryStats object, either as an empty object or using data.

        :param shape: Optional tuple specifying the shape of each data element (numpy.ndarray).
        :param data: Optional numpy.ndarray data array. The outermost dimension (axis=0) iterates over samples.
        :param exact_mean: Optional exact value for the mean, which will override data driven values for mean and SE.
        """
        if data is None:
            self.count = 0
            if shape == (1):
                self.data_mean = np.array(0.0)
                self.m2 = np.array(0.0)
            else:
                self.data_mean = np.zeros(shape=shape)
                self.m2 = np.zeros(shape=shape)
        else:
            self.count = data.shape[0]
            self.data_mean = data.mean(axis=0)
            self.m2 = data.var(axis=0, ddof=0) * self.count

        self.shape = shape
        self.exact_mean = exact_mean

    def __iadd__(self, other):
        """
        Overload += operator to combine the statistics of two datasets. This is also used to replace the + operator.

        :param other: SummaryStats object
        :return: SummaryStats object
        """

        if other.count != 0:
            self.m2 += other.m2 + (self.data_mean - other.data_mean) ** 2 * self.count * other.count / (self.count + other.count)
            self.data_mean = (self.count * self.data_mean + other.count * other.data_mean) / (self.count + other.count)
            self.count += other.count

        if other.exact_mean is not None:
            assert (self.exact_mean is None) or np.all((self.exact_mean == other.exact_mean))
            self.exact_mean = other.exact_mean

        return self

    def append(self, value):
        """
        Append a single value and update the summary statistics.

        Implemented using a robust online algorithm. See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

        :param value: scalar value to be added
        :return: None
        """
        self.count += 1
        delta = value - self.data_mean
        self.data_mean += delta / self.count
        self.m2 += delta * (value - self.data_mean)

    @property
    def mean(self):
        """Mean, based no exact mean where available, or data mean otherwise."""
        return self.exact_mean if (self.exact_mean is not None) else self.data_mean

    @property
    def stddev(self):
        """Standard deviation, using the 'N-1' factor (based on estimator of unknown population variance)."""
        return np.sqrt(self.m2/(self.count - 1))

    @property
    def variance(self):
        """Variance, using the 'N-1' factor (unbiased estimator of unknown population variance)."""
        return self.m2/(self.count - 1) if (self.count > 1) else (math.inf * self.m2)

    @property
    def stderr(self):
        """Standard error, using the 'N-1' factor (based on estimator of unknown population variance)."""
        if self.exact_mean is not None:
            return np.zeros(shape=self.shape)
        else:
            return np.sqrt(self.m2/(self.count * (self.count - 1)))

    @property
    def sum(self): return self.count * self.data_mean

    @property
    def contains_information(self):
        return True if (self.count > 0 or self.exact_mean is not None) else False

class SummaryBivariateStats:
    """
    SummaryBivariateStats computes and stores summary bivariate statistics (covariance, correlation) of data.

    Data elements are assumed to be of type numpy.ndarray, although this is not checked and related objects may work.

    """
    def __init__(self, shape=(1), data1=None, data2=None):
        """
        Initialise SummaryBivariateStats object, either as an empty object or using data.

        :param shape: Optional tuple specifying the shape of each data element (numpy.ndarray).
        :param data1: Optional numpy.ndarray data array. The outermost dimension (axis=0) iterates over samples.
        :param data2: Optional numpy.ndarray data array. The outermost dimension (axis=0) iterates over samples.
        """
        if data1 is None or data2 is None:
            # If no valid data is supplied, initialise empty object
            self.count = 0
            self._mean_1 = np.zeros(shape=shape)
            self._mean_2 = np.zeros(shape=shape)
            self.comoment = np.zeros(shape=shape)
            self._m2_1 = np.zeros(shape=shape)
            self._m2_2 = np.zeros(shape=shape)
        else:
            # If both data structure as valid, analyse data
            assert data1.shape == data2.shape
            self.count = data1.shape[0]
            self._mean_1 = data1.mean(axis=0)
            self._m2_1 = data1.var(axis=0, ddof=0) * self.count
            self._mean_2 = data2.mean(axis=0)
            self._m2_2 = data2.var(axis=0, ddof=0) * self.count
            # Compute the comoment by elementwise multiplication of differences from the mean, followed by summation
            # over all samples
            self.comoment = np.sum((data1 - self._mean_1) * (data2 - self._mean_2), axis=0)

    def __iadd__(self, other):
        """
        Overload += operator to combine the statistics of two datasets. This is also used to replace the + operator.

        :param other: SummaryBivariateStats object
        :return: SummaryBivariateStats object
        """
        self._m2_1 += other._m2_1 + (self._mean_1 - other._mean_1) ** 2 * self.count * other.count / (self.count + other.count)
        self._mean_1 = (self.count * self._mean_1 + other.count * other._mean_1) / (self.count + other.count)
        self._m2_2 += other._m2_2 + (self._mean_2 - other._mean_2) ** 2 * self.count * other.count / (self.count + other.count)
        self._mean_2 = (self.count * self._mean_2 + other.count * other._mean_2) / (self.count + other.count)
        self.count += other.count
        self.comoment += other.comoment + (self._mean_1 - other._mean_1) * (self._mean_2 - other._mean_2) * self.count * other.count / (self.count + other.count)
        return self

    def append(self, value1, value2):
        """
        Append a single value pair and update the summary statistics.

        Implemented using a robust online algorithm. See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

        :param value: scalar value to be added
        :return: None
        """
        self.count += 1
        delta1 = value1 - self._mean_1
        delta2 = value2 - self._mean_2
        self._mean_1 += delta1 / self.count
        self._mean_2 += delta2 / self.count
        self._m2_1 += delta1 * (value1 - self._mean_1)
        self._m2_2 += delta2 * (value2 - self._mean_2)
        # note the apparent asymmetry of the following statement, due to delta1 containing the 'old' mean and
        # the second term containing the updated mean. This is in fact identical to its mirror version. See
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        self.comoment += delta1 * (value2 - self._mean_2)

    @property
    def covariance(self): return self.comoment/(self.count - 1) if (self.count > 1) else math.inf

    @property
    def correlation(self):
        return np.vectorize(lambda comoment, m2_product: comoment/np.sqrt(m2_product) if m2_product > 0 else math.nan)\
            (self.comoment, self._m2_1 * self._m2_2)


