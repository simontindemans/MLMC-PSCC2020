# -*- coding: utf-8 -*-
"""
MLMC storage dispatch case study
@author: Simon Tindemans
Delft University of Technology
s.h.tindemans@tudelft.nl

Function optimal_n_store_generator() by Michael Evans, Imperial College London.

This code implements the composite system adequacy case study in the paper
"Accelerating System Adequacy Assessment using the Multilevel Monte Carlo Approach",
Simon Tindemans and Goran Strbac,
accepted for publication at PSCC 2020 and s special issue of EPSR.
A preprint is available at: arXiv:1910.13013

If you use (parts of) this code, please cite the preprint or published paper.
"""
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import quadprog

# import base class definitions for multi-level sampling
import gen_adequacy

import MLSampleBase

EFFECTIVE_ZERO = 1e-10

def define_storage(store_total_power, store_duration, store_units, store_dispersion):
    """
    Initialise contribution of storage units

    :param store_total_power: total power of storage units
    :param store_duration: mean duration (at max power) of storage units
    :param store_units: number of storage units
    :param store_dispersion: dispersion of duration
    :return: None
    """

    if store_units == 1:
        store_power_list = np.array([store_total_power])
        store_energy_list = np.array([store_duration * store_total_power])
    else:  # create diverse array of storage units
        store_power_list = np.full(shape=(store_units), fill_value=store_total_power / store_units)
        store_energy_list = np.linspace(
            (1 + store_dispersion) * store_total_power * store_duration / store_units,
            (1 - store_dispersion) * store_total_power * store_duration / store_units,
            num=store_units)

    return store_power_list, store_energy_list



class StorageSystem(MLSampleBase.MLSampleFactory):
    """
    Storage System class definition

    """

    def __init__(self, demand_samples, wind_samples,
                 store_power_list=None, store_energy_list=None, store_efficiency=1.0):
        """
        Initialise storage system

        :param demand_samples: dictionary of {year: time series} pairs
        :param wind_samples:  dictionary of {year: time series} pairs
        :param store_power_list: list of max power of storage units
        :param store_energy_list: list of max energy of storage units
        :param store_efficiency: efficiency of charging
        """

        if store_efficiency != 1.0:
            raise NotImplementedError()

        # Call superclass constructor, announcing available sample levels, accepted combinations and outputs.
        super(self.__class__, self).__init__(
            output_labels=('LOLE', 'EENS'),
            output_units=('h', 'MWh'),
            available_levels=('OptimalNStore', 'GreedyNStore', 'Greedy1Store', 'AvgStore', 'NoStore'),
            suggested_hierarchy=('GreedyNStore','Greedy1Store','AvgStore'),
            permissible_level_sets=[
                {'OptimalNStore', 'GreedyNStore', 'Greedy1Store', 'AvgStore', 'NoStore'},
                ]
        )

        self.demand_samples = demand_samples
        self.wind_samples = wind_samples

        # Generate net demand traces from all permutations of demand traces and wind power traces
        net_demand_sample_combinations = [demand_samples[demand_year] - wind_samples[wind_year]
                                          for demand_year in demand_samples
                                          for wind_year in wind_samples]
        # initialise a system that has 3 hours LOLE across all net demand sample traces
        self.rv_system = gen_adequacy.autogen_system(load_profile=np.concatenate(net_demand_sample_combinations),
                                                      wind_profile=None,
                                                      LOLH=3,
                                                      gen_set=[1200, 600, 600, 250, 250, 120, 60, 20, 20, 10, 10, 10],
                                                      MTBF=2000,
                                                      apply_load_offset=True,
                                                      resolution=10, gen_availability=0.90)

        # manually apply load offset to gross and net demand samples to hit 3 hours LOLE in the reference scenario
        self.demand_samples = {year: demand_samples[year] + self.rv_system.load_offset for year in self.demand_samples}

        # initialise storage units
        assert (store_power_list is None) == (store_energy_list is None), \
            "store_power_list and store_energy_list must both be specified or not at all."
        if store_power_list is not None:
            assert len(store_power_list) == len(store_energy_list), \
                "store_power_list and store_energy_list must have equal length"

        if store_power_list is None:
            self.store_power_list = np.array([0.0])
            self.store_energy_list = np.array([0.0])
        else:
            self.store_power_list = np.array(store_power_list)
            self.store_energy_list = np.array(store_energy_list)

        # compute the average storage response of a single unit
        mean_daily_demand = np.mean(
            [self.demand_samples[year].reshape((-1, 24)).mean(axis=0) for year in self.demand_samples],
            axis=0)

        mean_daily_demand_with_storage = self._load_flattener_periodic(mean_daily_demand,
                                                                 power_limit=np.sum(self.store_power_list),
                                                                 energy_limit=np.sum(self.store_energy_list))
        daily_storage_demand_offset = mean_daily_demand_with_storage - mean_daily_demand
        self.avg_storage_demand_offset_trace = np.tile(daily_storage_demand_offset, 365)

        return

    def generate_sample(self, level_set):
        """
        Implement base class function to generate sample objects
        :param level_set: index set to support
        :return: StorageSample object (derived from MLSample)
        """
        return StorageSample(level_set=level_set, power_system=self)

    def expectation_value(self, level):
        """
        Implement base class function to compute expectation values directly (without sampling)
        :param level: single index
        :return: expectation value
        """
        if level == 'OptimalNStore':
            raise NotImplementedError
        if level == 'GreedyNStore':
            raise NotImplementedError
        elif level == 'Greedy1Store':
            raise NotImplementedError
        elif level == 'AvgStore':
            return self._expectation_avg()
        elif level == 'NoStore':
            return self._expectation_no_store()
        else:
            raise NotImplementedError

    def expectation_value_available(self, level):
        """
        Implement base class function that specifies whether analytical expectation values are available
        :param level: single index
        :return: boolean
        """
        if level == 'OptimalNStore':
            return False
        if level == 'GreedyNStore':
            return False
        elif level == 'Greedy1Store':
            return False
        elif level == 'AvgStore':
            return True
        elif level == 'NoStore':
            return True
        else:
            raise NotImplementedError

    def _expectation_no_store(self, _cache={}):
        """
        Internal function to return expectation values for the NoStore reference
        :param _cache:
        :return: expectation value pair (LOLP, EENS)
        """
        # include dirty memoization hack to store previously computed result
        if len(_cache) == 0:
            _cache[0] = np.array((8760 * self.rv_system.lolp(), 8760 * self.rv_system.epns()))
        return _cache[0]

    def _expectation_avg(self, _cache={}):
        """
        Internal function to return expectation values for the AvgStore case
        :param _cache:
        :return: expectation value pair (LOLP, EENS)
        """
        # include dirty memoization hack to store previously computed result
        if len(_cache) == 0:



            # generate all net demand traces with inclusion of identical daily storage dispatch
            adjusted_net_demand_samples = [self.demand_samples[demand_year] + self.avg_storage_demand_offset_trace - self.wind_samples[wind_year]
                                           for demand_year in self.demand_samples
                                           for wind_year in self.wind_samples]
            # create a gen_adequacy.SingleNodeSystem and use internal tools to compute LOLP and EENS by convolution
            adjusted_rv_system = gen_adequacy.SingleNodeSystem(gen_list=self.rv_system.gen_list,
                                                                  load_profile=np.concatenate(adjusted_net_demand_samples),
                                                                  wind_profile=None,
                                                                  resolution=self.rv_system.resolution,
                                                                  load_offset=None
                                                                  )
            _cache[0] = np.array((8760 * adjusted_rv_system.lolp(), 8760 * adjusted_rv_system.epns()))
        return _cache[0]

    def _load_flattener_periodic(self, load_vec, power_limit, energy_limit):
        """
        Function that determines a peak-shaving/valley-filling dispatch pattern for the battery, with periodic boundaries.

        :param load_vec: load profile
        :param power_limit: max power
        :param energy_limit: max energy
        :return: flattened load profile (original + battery)

        I selected this optimiser after reading:
        https://scaron.info/blog/quadratic-programming-in-python.html

        The variable vector consists of 1 initial (and final) energy level and N power levels (for each time stamp)
        """
        batch_size = len(load_vec)

        G_mat = np.identity(batch_size + 1)
        G_mat[0, 0] = 0.000001  # NOTE: this should be zero, but the solver requires positive definite G

        if isinstance(load_vec, pd.Series):
            a_vec = -load_vec.values
        else:
            a_vec = -load_vec
        a_vec = np.insert(a_vec, 0, 0)

        equality_constraint = np.ones((1, batch_size + 1))
        equality_constraint[0, 0] = 0
        power_constraints = np.vstack((np.eye(batch_size, batch_size + 1, 1), -np.eye(batch_size, batch_size + 1, 1)))
        energy_constraints = np.vstack((np.tri(batch_size, batch_size + 1), -np.tri(batch_size, batch_size + 1)))

        C_mat = np.vstack((equality_constraint, power_constraints, energy_constraints)).transpose()
        b_vec = -np.concatenate(
            (np.zeros(1), np.ones(2 * batch_size) * power_limit, np.zeros(batch_size), np.ones(batch_size) * energy_limit))

        power_solution = quadprog.solve_qp(G=G_mat, a=a_vec, C=C_mat, b=b_vec, meq=1, factorized=False)[0][1:]

        return load_vec + power_solution



class StorageSample(MLSampleBase.MLSample):
    """
    Class for a single sample (i.e. realisation) of annual loss of load.

    This derives from MSSampleBase.MLSample for compatibility with the MLMC framework.
    """
    def __init__(self, level_set, power_system):
        """
        Initialise sample

        :param level_set: index set to support
        :param power_system: reference to the power system object
        """
        # Required function: formally initialise sample by instantiating random realisation.
        super(self.__class__, self).__init__(level_set=level_set)

        self.system = power_system

        # select a random demand year and its corresponding demand trace
        demand_trace = self.system.demand_samples[np.random.choice(list(power_system.demand_samples))]
        # select a random wind trace
        wind_trace = self.system.wind_samples[np.random.choice(list(power_system.wind_samples))]
        # generate a random thermal generation trace
        thermal_gen_trace = self.system.rv_system.generation_trace(num_steps=8760)
        # compute the resulting margin trace
        self.margin_trace = thermal_gen_trace + wind_trace - demand_trace


        return

    def generate_value(self, level):
        """
        Generate sample outputs at specified index level. [Required implementation for MLSample base class]

        :param level: specific index
        :return: (LOL, ENS) array
        """

        if level == 'OptimalNStore':
            return self.optimal_n_store_generator()
        if level == 'GreedyNStore':
            return self.greedy_n_store_generator()
        elif level == 'Greedy1Store':
            return self.greedy_1_store_generator()
        elif level == 'AvgStore':
            return self.avg_store_generator()
        elif level == 'NoStore':
            return self.no_store_generator()
        else:
            raise RuntimeError()

    def no_store_generator(self):
        """
        Compute (LOL, ENS) for sample for the NoStore model

        :return: (LOL, ENS) array
        """
        shortfalls = self.margin_trace < 0
        lol = np.sum(shortfalls)
        ue = -np.sum(self.margin_trace[shortfalls])
        return np.array([lol, ue])

    def avg_store_generator(self):
        """
        Compute (LOL, ENS) for sample in the AvgStore model
        :return: (LOL, ENS) array
        """

        adjusted_margin_trace = self.margin_trace - self.system.avg_storage_demand_offset_trace

        shortfalls = adjusted_margin_trace < 0
        lol = np.sum(shortfalls)
        ue = -np.sum(adjusted_margin_trace[shortfalls])
        return np.array([lol, ue])



    def greedy_1_store_generator(self):
        """
        Compute (LOL, ENS) for sample in the Greedy1Store model
        :return: (LOL, ENS) array
        """
        # efficiency measure: evaluate AvgStore policy. If it results in no impacts, skip further evaluation
        if (self['AvgStore'] < EFFECTIVE_ZERO).all():
            return np.array([0., 0.])

        # apply greedy storage to margin trace
        post_margin_trace = self._greedy_storage(pre_margin=self.margin_trace,
                                                 power_limit=np.sum(self.system.store_power_list),
                                                 energy_limit=np.sum(self.system.store_energy_list))

        shortfalls = post_margin_trace < 0
        lol = np.sum(shortfalls)
        ue = -np.sum(post_margin_trace[shortfalls])
        return np.array([lol, ue])

    def greedy_n_store_generator(self):
        """
        Compute (LOL, ENS) for sample in the GreedyNStore model
        :return: (LOL, ENS) array
        """

        # efficiency measure: evaluate NoStore policy. If it results in no impacts, skip further evaluation
        if (self['NoStore'] < EFFECTIVE_ZERO).all():
            return np.array([0., 0.])

        # apply greedy storage units to margin trace, one by one
        updated_margin_trace = self.margin_trace

        # generate a list of units and sort them by time to go
        store_list = [item for item in zip(self.system.store_power_list, self.system.store_energy_list)]
        store_list.sort(key=lambda x: x[1]/x[0], reverse=True)

        for unit_power, unit_energy in store_list:
            updated_margin_trace = self._greedy_storage(pre_margin=updated_margin_trace,
                                                        power_limit=unit_power,
                                                        energy_limit=unit_energy)

        shortfalls = updated_margin_trace < 0
        lol = np.sum(shortfalls)
        ue = -np.sum(updated_margin_trace[shortfalls])
        return np.array([lol, ue])


    def optimal_n_store_generator(self):
        """
        Compute (LOL, ENS) for sample in the OptimalNStore model
        :return: (LOL, ENS) array

        Implemented by Michael Evans, Imperial College London.
        """

        def optimal_policy(shortfall, state, power_limit, dt):
            """
            Greedy optimal dispatch policy for n devices; for a single time step.

            :param shortfall: current shortfall without storage
            :param initial_state: duration state vector across stores
            :param power_limit: max power vector across stores
            :param dt: time step (units of energy/power)
            :return: optimal dispatch
            """
            n = len(state)
            one = np.ones([n, 1])  # unity vector (length n)
            zero = np.zeros([n, 1])  # zero vector (length n)
            y = np.unique(np.sort(np.concatenate([state, np.max([state - dt * one, zero], axis=0)], axis=0)))
            y = y[::-1]  # decreasing order
            E_u = 0
            i = -1  # to allow for 0-indexing
            while True:
                i += 1
                E_l = E_u
                E_u = np.sum(
                    np.multiply(power_limit, np.max([np.min([state - y[i] * one, dt * one], axis=0), zero], axis=0)))
                if E_u >= shortfall * dt or i == len(y) - 1:
                    break
            if E_u <= shortfall * dt:
                z_hat = y[i]
            else:
                z_hat = y[i - 1] + (shortfall * dt - E_l) * (y[i] - y[i - 1]) / (E_u - E_l)
            u = np.multiply(power_limit, np.max([np.min([(state - z_hat * one) / dt, one], axis=0), zero], axis=0))
            return u

        def recharge(excess, state, duration_limit, power_limit_up, power_limit_down, eta, dt):
            """
            Greedy recharge policy for n devices; for a single time step.

            :param excess: current excess without storage
            :param initial_state: duration state vector across stores
            :param duration_limit: maximum duration limit across stores
            :param power_limit: max power vector across stores
            :param dt: time step (units of energy/power)
            :return: recharge dispatch
            """
            n = len(state)
            one = np.ones([n, 1])  # unity vector (length n)
            zero = np.zeros([n, 1])  # zero vector (length n)
            z_bar = np.min([state - eta * power_limit_up * dt / power_limit_down, duration_limit], axis=0)
            y = np.unique(np.sort(np.concatenate([state, z_bar], axis=0), axis=0))  # ascending order
            E_u = 0
            i = -1  # to allow for 0-indexing
            while True:
                i += 1
                E_l = E_u
                E_u = np.sum(np.multiply(power_limit_down,
                                         np.max([np.min([y[i] * one - state, z_bar - state, dt * one], axis=0), zero],
                                                axis=0)))
                if E_u >= excess * dt or i == len(y) - 1:
                    break
            if E_u <= excess * dt:
                z_hat = y[i]
            else:
                z_hat = y[i - 1] + (excess * dt - E_l) * (y[i] - y[i - 1]) / (E_u - E_l)
            u = np.multiply(power_limit_down,
                            np.min([np.max([(state - z_hat * one) / dt, (state - z_bar) / dt], axis=0), zero],
                                   axis=0)) / eta
            return u

        lol=0
        eu=0.0
        energy_limits=np.array([self.system.store_energy_list]).transpose()
        power_limits=np.array([self.system.store_power_list]).transpose()
        state = energy_limits/power_limits # assume batteries start full
        # self.E_mat=np.zeros([1,8761])
        # self.E_mat[0,[0]]=np.sum(energy_limits)
        # self.u_mat=np.zeros([1,8760])
        # self.x_mat=np.zeros([27,8761])
        # self.x_mat[:,[0]]=state
        for ts in range(0,len(self.margin_trace)):
            if self.margin_trace[ts]<=0:
                control_input=optimal_policy(-self.margin_trace[ts],state,power_limits,dt=1)
                ens=-self.margin_trace[ts]-np.sum(control_input)
                lol+=(ens>EFFECTIVE_ZERO)
                eu+=ens
            else:
                control_input=recharge(self.margin_trace[ts],state,energy_limits/power_limits,-power_limits,power_limits,
                                       eta=1, dt=1)
            state=state-control_input/power_limits
            # self.x_mat[:,[ts+1]]=state
            # self.E_mat[0,[ts+1]]=np.sum(state*power_limits)
            # self.u_mat[0,[ts]]=np.sum(control_input)
        return np.array([lol,eu])

    def _greedy_storage(self, pre_margin, power_limit, energy_limit, dt=1):
        """
        Execute greedy storage algorithm on a net margin time trace

        :param pre_margin: margin time series without storage
        :param power_limit: max power (charge and discharge assumed identical)
        :param energy_limit: max energy storage
        :param dt: time step (units of energy/power)
        :return: adjusted margin time series
        """
        assert power_limit > 0
        assert energy_limit > 0

        # assume start with a full store
        energy_stored = energy_limit

        post_margin = np.zeros(pre_margin.size)
        for t, margin in enumerate(pre_margin):
            store_power = max(min(margin, power_limit, (energy_limit - energy_stored) / dt), -power_limit,
                              -energy_stored / dt)
            energy_stored += store_power
            post_margin[t] = margin - store_power

        return post_margin


if __name__ == "__main__":

    import MCCoordinator


    def storage_system(**kwargs):
        """
        Instantiate StorageSystem object using GB system data

        :param wind_power: assumed wind power capacity (in MW)
        :param kwargs: additional arguments to be supplied to the StorageSystem constructor
        :return: StorageSystem object
        """

        data = pd.read_csv('../data/UKdata/20161213_uk_wind_solar_demand_temperature.csv',
                           parse_dates=['UTC Time', 'Local Time'], infer_datetime_format=True, dayfirst=True,
                           index_col=0)

        demand_data = data['demand_net'].dropna()['2006':'2015']
        wind_data = 10000 * data['wind_merra1'].dropna()

        demand_samples = {yeardata[0]: yeardata[1].values[:8760] for yeardata in
                          demand_data.groupby(demand_data.index.year)}
        wind_samples = {yeardata[0]: yeardata[1].values[:8760] for yeardata in wind_data.groupby(wind_data.index.year)}

        dataframe = pd.read_csv('../data/battery_data.csv', index_col=0)
        store_power_list = 3 * dataframe['Power (MW)']
        store_energy_list = 3 * dataframe['Energy (MWh)']

        return StorageSystem(demand_samples=demand_samples,
                                            wind_samples=wind_samples,
                                            store_power_list=store_power_list,
                                            store_energy_list=store_energy_list,
                                            **kwargs)


    system = storage_system()

    # set up MLMC coordinator with link to system
    mcc = MCCoordinator.MCCoordinator(factory=system,
                                      ml_hierarchy=['OptimalNStore', 'GreedyNStore', 'AvgStore'],
                                      use_expectations=True,
                                      use_joblib=False, joblib_n_jobs=-1, joblib_batch_size=5)

    mcc.explore(n_samples=5)

    for i in range(10):
        mcc.run_recommended(time_seconds=10, verbose=True, optimization_target='EENS')

    mcc.verbose_result()