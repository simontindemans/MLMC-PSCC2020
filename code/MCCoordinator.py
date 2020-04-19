# -*- coding: utf-8 -*-
"""
Provides an extensible Multi-Level Monte Carlo simulator, with a class interface MCCoordinator.

Extensible Multi-Level Monte Carlo simulator, optionally making use of joblib for parallel evaluation. It performs
MLMC estimation of expectation values on the basis of the MLSampleFactory and MLSample interfaces exposed by the
simulator. Helper classes are provided for numerical functions.

Classes:
MCCoordinator           : Interface class for MLMC simulations
SummaryStats            : Helper class to store univariate statistics
SummaryBivariateStats   : Helper class to store bivariate statistics

Functions:
_sample_core(.)         : Core sample routine: called from MCCoordinator.run()
round_to_n(.)           : Rounds numbers to n significant digits for display

Target: python 3.7

@author: Simon Tindemans
Delft University of Technology
s.h.tindemans@tudelft.nl
"""
# SPDX-License-Identifier: MIT

import math
from collections import defaultdict
from time import time

import joblib
import numpy as np
import itertools

import MLSampleBase
from SummaryStats import SummaryStats, SummaryBivariateStats

class MCCoordinator:
    """
    Class interface for Multi-Level Monte Carlo simulator.

    Extensible Multi-Level Monte Carlo simulator, optionally making use of joblib for parallel evaluation. It performs
    MLMC estimation of expectation values on the basis of the MLSampleFactory and MLSample interfaces exposed by the
    simulator.

    Methods:
        __init__(.)                     : Constructor
        run(.)                          : Perform MLMC sampling
        summary_results(.)              : Print summary results
        verbose_results(.)              : Print verbose results, including MLMC analysis

    """

    def __init__(self, factory, ml_hierarchy=None, use_expectations=True,
                 use_joblib=False, joblib_n_jobs=-1, joblib_batch_size='auto'):
        """
        Initialises MCCoordinator object and links it to a MLSampleFactory.

        The parallel library joblib is used to distribute embarrassingly parallel Monte Carlo sampling tasks among
        multiple cores. It is recommended to assess its impact on performance. In general, it will be beneficial if
        the evaluation of individual samples is resource-intensive, but the overhead involved in dispatching tasks
        can be substantial if samples are evaluated rapidly. In addition, joblib can be disabled for special use cases,
        including profiling.

        NOTE: The default of joblib_n_jobs = -1 creates parallel jobs equal to the number of logical cores. In
        systems with HyperThreading (intel i7, some intel i5 CPUs), this may cause each process to run at approx. half
        the single-process speed. Although the total throughput is not affected, the measured time-per-sample is,
        because cores are waiting for each other.

        :param factory: MLSampleFactory-derived object that is used to generate samples
        :param ml_hierarchy: tuple or list of levels in decreasing accuracy
        :param use_joblib: boolean, enabling the use of parallel evaluation on multi-core systems
        :param joblib_n_jobs: integer, number of parallel jobs to launch. Use -1 for the number of logical cores.
        :param joblib_batch_size: default 'auto', but int can be supplied
        """
        assert isinstance(factory, MLSampleBase.MLSampleFactory)
        self.factory = factory

        self.use_expectations = use_expectations

        # Initialise joblib settings
        self.use_joblib = use_joblib
        self.joblib_n_jobs = joblib_n_jobs
        self.joblib_batch_size = joblib_batch_size

        # Get sample output information from the MLSampleFactory
        self.output_labels = factory.output_labels
        self.output_units = factory.output_units
        assert len(self.output_labels) == len(self.output_units)
        self.output_shape = (len(factory.output_labels),)

        # initialise feasible and relevant multilevel sets
        self.available_levels = factory.available_levels
        self.permissible_level_sets = factory.permissible_level_sets

        # Construct a hierarchy and coefficients
        if ml_hierarchy is None: ml_hierarchy = factory.suggested_hierarchy

        assert set(ml_hierarchy) <= set(self.available_levels)

        self.level_list = ml_hierarchy
        self.target_level = ml_hierarchy[0]

        #define level sets
        self.ml_set_list = [frozenset((ml_hierarchy[idx], ml_hierarchy[idx+1])) for idx in range(len(ml_hierarchy) - 1)]
        self.ml_set_list.append(frozenset([ml_hierarchy[-1]]))

        # for each level set, store coefficients to generate the telescopic sum in default order
        self.ml_set_coefficients = {frozenset((ml_hierarchy[idx], ml_hierarchy[idx+1])):
                                        {ml_hierarchy[idx]: 1, ml_hierarchy[idx+1]: -1}
                                    for idx in range(len(ml_hierarchy) - 1)}
        self.ml_set_coefficients[frozenset({ml_hierarchy[-1]})] = {ml_hierarchy[-1] : 1}

        # associate a set index 0, 1, ... for each level set
        self.ml_set_index = {frozenset((ml_hierarchy[idx], ml_hierarchy[idx+1])): len(ml_hierarchy) - 1 - idx
                             for idx in range(len(ml_hierarchy) - 1)}
        self.ml_set_index[frozenset({ml_hierarchy[-1]})] = 0


        # initialise empty summary statistics dictionaries
        # use defaultdict to generate 'zero' elements of the correct type as and when required
        self.level_stats = defaultdict(lambda: SummaryStats(shape=self.output_shape))
        self.level_exec_times = defaultdict(lambda: SummaryStats(shape=1))
        self.level_pair_stats = defaultdict(lambda: SummaryBivariateStats(shape=self.output_shape))
        self.set_exec_times = defaultdict(lambda: SummaryStats(shape=1))
        self.ml_stats = defaultdict(lambda: SummaryStats(shape=self.output_shape))
        self.ml_sample_values = defaultdict(lambda key: np.zeros(shape=(0, len(key), len(self.output_labels))))

        # initialise total wall clock time
        self.wall_clock_time = 0

    def explore(self, n_samples=1, time_seconds=None):

        if time_seconds is None:
            run_list = {frozenset(idx_set): n_samples for idx_set in self.ml_set_list}
            self.run(run_list, estimate_covariance=True, compute_level_contributions=True)

        else:
            # use time_seconds and run n_samples at a time.

            start_time = time()
            run_list = {frozenset(idx_set): n_samples for idx_set in self.ml_set_list}

            while True:
                self.run(run_list, estimate_covariance=True, compute_level_contributions=True)
                if time() - start_time >= time_seconds:
                    break


    def run_recommended(self,
                        n_samples=None,
                        time_seconds=None,
                        optimization_target=None,
                        minimum_variance_ratio=0.1,
                        verbose=False):
        """
        Generates and evaluates Multi-level samples.

        run(.) analyses a pre-specified number of samples per MC level. It collects summary statistics by default,
        but can also be tasked to store all generated outputs.

        :param run_counts: set of integers, indicating the number of samples to evaluate at each ML level
        :param store_sample_values: boolean, indicating whether all samples should be stored
        :return: None
        """

        # n_samples and time_seconds are mutually exclusive
        assert bool(n_samples) != bool(time_seconds), "Exactly one option (n_samples or time_seconds) must be used."

        if n_samples is not None:
            if hasattr(n_samples, '__getitem__') and hasattr(n_samples, '__len__'):
                assert len(n_samples) == len(self.ml_set_coefficients)
                # we can index it, assume one number per level
                run_list = {idx_set: n_samples[num] for num, idx_set in enumerate(self.ml_set_list)}
            else:
                run_list = {idx_set : n_samples for idx_set in self.ml_set_list}

        elif time_seconds is not None:

            # ensure we have relevant timing data
            assert all(level_set in self.set_exec_times for level_set in self.ml_set_list)

            # if no explicit optimization target is specified, use the first output variable
            if optimization_target is None:
                optimization_target = self.output_labels[0]
            optimization_target_ix = self.output_labels.index(optimization_target)
            if verbose:
                print('Optimizing for', optimization_target)

            relative_sample_number = {}

            maximum_variance = np.max([self.level_stats[level].variance[optimization_target_ix]
                                       for level in self.level_list])

            for level_set in self.ml_set_list:
                if (self.use_expectations is True) and all([self.factory.expectation_value_available(level) for level in level_set]):
                    relative_sample_number[level_set] = 0.0
                else:
                    estimated_variance = self.ml_stats[level_set].variance[optimization_target_ix]
                    minimum_variance = minimum_variance_ratio**self.ml_set_index[level_set] * maximum_variance
                    relative_sample_number[level_set] = np.sqrt(max(minimum_variance, estimated_variance) /
                                                                self.set_exec_times[level_set].mean)

            if np.sum([relative_sample_number[level_set] for level_set in self.ml_set_list]) < 1E-10:
                relative_sample_number = {level_set: 1.0 for level_set in self.ml_set_list}
            print("relative sample numbers: ",relative_sample_number)

            time_multiplier = sum([self.set_exec_times[level_set].mean * relative_sample_number[level_set]
                                   for level_set in self.ml_set_list])

            if time_multiplier <= 1E-10:
                run_list = {level_set: 1 for level_set in self.ml_set_list}
            else:
                number_multiplier = time_seconds / time_multiplier
                run_list = {level_set: max(1, int(number_multiplier * relative_sample_number[level_set]))
                            for level_set in self.ml_set_list}

        self.run(run_list, compute_level_contributions=True, estimate_covariance=True)

    def run(self, run_list,
            store_sample_values=False, compute_level_contributions=False,
            estimate_covariance=True):
        """bla"""


        start_wall_clock_time = time()

        assert isinstance(run_list, dict)

        print('Running:', run_list)

        # NOTE: this will require care with parallel random seeds
        with joblib.Parallel(n_jobs=self.joblib_n_jobs, verbose=5, batch_size=self.joblib_batch_size, backend='loky') as parallel:
            # iterate over ML sets
            for level_set, run_count in run_list.items():
                # Values are computed, and stored and analysed, iteratively for each index set
                # This is faster than element-wise analysis, and not as memory-hungry as a full storage/analysis separation.

                # verify that the requested combination of sample levels is supported by the sample generator
                assert any([level_set <= permissible_set for permissible_set in self.permissible_level_sets])

                # check if the level set can be evaluated using expectation values
                if (self.use_expectations is True) and all([self.factory.expectation_value_available(level) for level in level_set]):

                    expectation = {}
                    for level in level_set:
                        expectation[level] = self.factory.expectation_value(level)
                        self.level_stats[level] += SummaryStats(shape=(len(self.output_labels)), exact_mean=expectation[level])

                    if compute_level_contributions:
                        ml_value = sum([self.ml_set_coefficients[level_set][level] * expectation[level] for level in level_set])
                        self.ml_stats[level_set] += SummaryStats(shape=(len(self.output_labels)), exact_mean=ml_value)

                    # Marginal cost of sampling this set has become zero
                    self.set_exec_times[level_set] += SummaryStats(exact_mean=np.array(0.0))

                else:

                    # This loop operates on numeric indices instead of arbitrary level items
                    # Because a dict does not offer a stable numerical order, define bidirectional maps
                    number_idx_pairs = list(enumerate(level_set))
                    map_int_to_idx = [pair[1] for pair in number_idx_pairs]
                    map_idx_to_int = {pair[1]: pair[0] for pair in number_idx_pairs}

                    # Pre-allocate memory
                    new_sample_values = np.zeros(shape=(run_count, len(level_set), len(self.output_labels)))
                    new_set_times = np.zeros(shape=run_count)
                    new_level_times = np.zeros(shape=(run_count, len(level_set)))

                    if self.use_joblib:
                        # dispatch calls to _sample_core(.) via joblib
                        jl_results = parallel(
                            joblib.delayed(_sample_core)
                            (factory=self.factory, idx_set=level_set, idx_list=map_int_to_idx,
                             output_size=len(self.output_labels))
                            for sample in range(run_count)
                        )
                        # joblib collects the results in sample order. Unzip the results to separate into the
                        # three different output types.
                        values, sample_times, idx_times = zip(*jl_results)
                        # ...and store them in floating point numpy arrays [there may be a cleaner way to do this]
                        # for n_count in range(run_count):
                        #     new_sample_values[n_count,:,:] = values[n_count]
                        #     new_set_times[n_count] = sample_times[n_count]
                        #     new_level_times[n_count,:] = idx_times[n_count]
                        new_sample_values = np.asfarray(values)
                        new_set_times = np.asfarray(sample_times)
                        new_level_times = np.asfarray(idx_times)
                    else:
                        # without joblib, use direct assignment to numpy arrays
                        for n_count in range(run_count):
                            new_sample_values[n_count, :, :], new_set_times[n_count], new_level_times[n_count, :] \
                                = _sample_core(factory=self.factory, idx_set=level_set, idx_list=map_int_to_idx,
                                               output_size=len(self.output_labels))


                    # store summary statistics of values and execution times, split by ML-term and index
                    self.set_exec_times[level_set] += SummaryStats(data=new_set_times)
                    for level in level_set:
                        self.level_stats[level] += SummaryStats(data=new_sample_values[:, map_idx_to_int[level]])
                        self.level_exec_times[level] += SummaryStats(data=new_level_times[:, map_idx_to_int[level]])
                    if estimate_covariance:
                        for int_pair in itertools.combinations(range(len(level_set)), 2):
                            self.level_pair_stats[frozenset({map_int_to_idx[int_pair[0]], map_int_to_idx[int_pair[1]]})] += \
                                SummaryBivariateStats(data1=new_sample_values[:, int_pair[0]], data2=new_sample_values[:, int_pair[1]])

                    # optionally, compute linear combination of outputs to estimate telescopic sum terms
                    if compute_level_contributions:
                        new_ml_values = np.sum([self.ml_set_coefficients[level_set][idx] * new_sample_values[:,map_idx_to_int[idx]]
                                                for idx in level_set], axis=0)
                        self.ml_stats[level_set] += SummaryStats(data=new_ml_values)

                    # optionally, store raw results
                    if store_sample_values:
                        self.ml_sample_values[level_set].append(new_sample_values, axis=0)


        self.wall_clock_time += time() - start_wall_clock_time



    def summary_result(self):
        """
        Print summary results to screen.

        :return: None
        """
        # Suppress divide by zeros and nan errors; store old settings to restore on return.
        old_settings = np.seterr(divide='ignore', invalid='ignore')

        # Compute the total time as computed by index computations and sample computations
        total_time_level = sum([level_stats.sum for level_stats in self.level_exec_times.values()])
        total_time = sum([set_stats.sum for set_stats in self.set_exec_times.values()])

        # compare measures of time with wall clock time
        print('\nWall clock time:', round_to_n(self.wall_clock_time, 3), 's',
              '\tTotal sample generation:', round_to_n(total_time, 3), 's',
              '\tIndex specialisation:', round_to_n(total_time_level, 3), 's')

        # compute the overall MLMC estimate and its standard error, by quadratic summation of term errors
        mean_result = sum([ml_stats.mean for key, ml_stats in self.ml_stats.items()])
        stderr_result = np.sqrt(sum([ml_stats.stderr ** 2 for key, ml_stats in self.ml_stats.items()]))

        est_time_per_target_sample = min([self.set_exec_times[level_set].mean for level_set in self.set_exec_times.keys() if self.target_level in level_set])
        target_stats = self.level_stats[self.target_level]
        est_time_spent_for_target_result = target_stats.count * est_time_per_target_sample

        # report results, alongside an estimated 'computational speed'
        for result_index, result_label in enumerate(self.output_labels):
            print(result_label, ":", round_to_n(mean_result[result_index], 4), self.output_units[result_index],
                  "+-", round_to_n(abs(stderr_result[result_index]), 2), self.output_units[result_index],
                  "\tcomputational speed: ", 'n/a' if stderr_result[result_index] == 0 else
                  round_to_n(mean_result[result_index] ** 2 / (stderr_result[result_index] ** 2 * self.wall_clock_time), 3),
                  "\testimated speedup: ", "n/a" if stderr_result[result_index] == 0 else
                      round_to_n(((target_stats.stderr[result_index] ** 2) * est_time_spent_for_target_result)
                                 / ((stderr_result[result_index] ** 2) * self.wall_clock_time), 3)
                  )

        # restore div-zero and nan settings
        np.seterr(**old_settings)

    def verbose_result(self):
        """
        Print verbose results to screen.

        Results are broken down by ML index, then results by pair.

        :return: None
        """

        self.summary_result()
        # Suppress divide by zeros and nan errors; store old settings to restore on return.
        old_settings = np.seterr(divide='ignore', invalid='ignore')

        estimate_level_time = {}
        for level in self.available_levels:
            time_values = [self.set_exec_times[idx_set].mean for idx_set in self.set_exec_times.keys() if level in idx_set]
            if len(time_values) > 0:
                estimate_level_time[level] = min(time_values)

        print('')
        print('Index set and time estimates:')
        for level in estimate_level_time:
            print(level, ':', round_to_n(estimate_level_time[level],2), 's')
        print('')


        print('')
        print('RESULTS BY ML INDEX')
        for level, idx_stats in self.level_stats.items():
            print('index:', level,
                  "\tcount: ", idx_stats.count,
                  "\tavg specialisation time:", round_to_n(1000 * self.level_exec_times[level].mean, 3), "ms")
            idx_mean_result = idx_stats.mean
            idx_stderr_result = idx_stats.stderr
            for result_index, result_label in enumerate(self.output_labels):
                print("\t", result_label, ":", round_to_n(idx_mean_result[result_index], 4), self.output_units[result_index],
                      "\t+-", round_to_n(idx_stderr_result[result_index], 2), self.output_units[result_index],
                      )

        print('')
        print('RESULTS BY SET')
        for idx_set, set_stats in self.set_exec_times.items():
            # note: include 'set' to remove 'frozenset' bracket
            print(set(idx_set), "\tcount: ", set_stats.count,
                  "\tavg exec time: ", round_to_n(1000 * set_stats.mean, 3), "ms")

        print('')
        print('RESULTS BY TELESCOPIC TERM')
        for idx_set, ml_stats in self.ml_stats.items():
            print(set(idx_set))
            level_mean_result = ml_stats.mean
            level_stderr_result = ml_stats.stderr
            for result_index, result_label in enumerate(self.output_labels):
                print("\t", result_label, "contribution:", round_to_n(level_mean_result[result_index], 4), self.output_units[result_index],
                      "\t+-", round_to_n(level_stderr_result[result_index] , 2), self.output_units[result_index],
                      "\tsample correlation:",
                      "n/a" if (len(idx_set) != 2) else round_to_n(self.level_pair_stats[idx_set].correlation[result_index], 3))

        np.seterr(**old_settings)


def _sample_core(factory, idx_set, idx_list, output_size, random_seed=None):
    """
    Helper function for MCCoordinator.run(.) that generates and evaluates a single multi-level sample

    NOTE: The random_seed is used for parallel processing scenarios, to avoid synchronised random generators in
    different processes. It can also be used to store/recall seeds for re-runs.

    :param factory: reference to an MLSampleFactory object
    :param idx_set: set of MC levels to analyse; used as a key for sample generation
    :param idx_list: list of MC levels to analyse; used to assign numerical order to levels
    :param output_size: integer size of sample output (1xn numpy.ndarray)
    :param random_seed: optional specification of random seed
    :return: tuple of
        - numpy.ndarray: output values for each level (levels x outputs)
        - scalar: time taken to complete evaluation
        - numpy.ndarray:
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # initialise data structures
    sample_values = np.zeros(shape=(len(idx_list), output_size))
    idx_times = np.zeros(shape=(len(idx_list)))

    start_time_sample = time()

    # ask the factory to generate a sample. Depending on the implementation, this may be a very expensive computation
    # that evaluates the sample at all levels specified by idx_set, or a lightweight initialisation with lazy
    # evaluation as and when required.
    sample = factory.generate_sample(level_set=idx_set)

    # loop over all MC levels requested from the sample
    for n_idx, idx in enumerate(idx_list):
        # Start the clock for level 'idx'. Note that recorded times may be inaccurate and/or depend on
        # calling order if work is split between levels
        start_time_idx = time()

        # generate and store the result
        sample_values[n_idx] = sample[idx]

        # store the time taken
        idx_times[n_idx] = time() - start_time_idx

    # determine the total time taken for the sample (all levels)
    sample_time = time() - start_time_sample

    return sample_values, sample_time, idx_times


def round_to_n(x, n):
    """
    Round x to n significant digits.

    :param x: scalar to be converted
    :param n: number of significant digits
    :return: scalar
    """
    if np.isnan(x): return x
    if np.sign(x) == 0: return 0.0
    return np.round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))



