# -*- coding: utf-8 -*-
"""
MLMC composite adequacy assessment case study
@author: Simon Tindemans
Delft University of Technology
s.h.tindemans@tudelft.nl

This code implements the composite system adequacy case study in the paper
"Accelerating System Adequacy Assessment using the Multilevel Monte Carlo Approach",
Simon Tindemans and Goran Strbac,
accepted for publication at PSCC 2020 and s special issue of EPSR.
A preprint is available at: arXiv:1910.13013

If you use (parts of) this code, please cite the preprint or published paper.
"""
# SPDX-License-Identifier: MIT


import numpy as np
from scipy import optimize
from collections import defaultdict

# import base class definitions for multi-level sampling
import MLSampleBase
import gen_adequacy

# define effective zero with room for plenty of rounding errors
EFFECTIVE_ZERO = 1E-7



class HLSystem(MLSampleBase.MLSampleFactory):
    """System definition for the HL1-HL2 sampler.
    
    Initialises the system and provides routines for generating random samples at HL1 and HL2 level.
    """

    def __init__(self,  nodes, lines, generators, relative_load_profile,
                 line_capacity_scaling=1.0, load_level_scaling=1.0, line_for_scaling=0.99):
        """
        HLSystem(...) creates a grid object.

        'nodes' is an Nx1 array, with N equal to the number of nodes.
        Each 'node' element is structured as
        node[0] = nominal load [power]

        'lines' is an Lx4 array, with L equal to the number of lines.
        Each 'line' element is structured as
        line[0] = originating node (1,...,N)
        line[1] = terminating node (1,...,N)
        line[2] = line reactance X
        line[3] = max capacity
        line[4] = permanent outage rate [events/year]
        line[5] = average outage duration [number of load steps (typically hours)]

        'generators' is an Gx3 array, with G equal to the number of generators
        Each 'generator' element is structured as
        generator[0] = Power
        generator[1] = connection node
        generator[2] = FOR

        It is assumed that the all nodes form a single cluster and that the
        total generation level equals the total load on the network.
        """

        # Call superclass constructor, announcing available sample levels, accepted combinations and outputs.
        super(self.__class__, self).__init__(
            available_levels=('HL2full', 'HL2gen', 'HL1'),
            suggested_hierarchy=('HL2full', 'HL2gen', 'HL1'),
            output_labels=('LOLP','EPNS'),
            output_units=('','MW'),
            permissible_level_sets=[{'HL2full', 'HL2gen', 'HL1'}]
        )

        self.line_availability = \
            np.array([1.0 - line_for_scaling*line[4]*line[5]/len(relative_load_profile) for line in lines])
        self.numNodes = len(nodes)

        # Initialise load nodes
        self.loadMax = load_level_scaling * np.array(nodes)
        self.load_profile = relative_load_profile

        # Initialise generators and generator-node links
        self.numGenerators = len(generators)
        self.genAvailability = np.array([1.0 - generator[2] for generator in generators])
        self.genCapacities = np.array([generator[0] for generator in generators])
        self.nodeGenMatrix = np.zeros([self.numNodes, self.numGenerators])
        for index, generator in enumerate(generators):
            self.nodeGenMatrix[generator[1] - 1, index] = generator[0]

        # Initialise lines and capacities
        self.lines = lines
        self.numLines = len(lines)
        self.lineCapacity = np.array([line_capacity_scaling * line[3] for line in lines])

        # Initialise connectivity and power flow matrix
        self.a_mat = np.zeros([self.numLines, self.numNodes])
        for index, line in enumerate(lines):
            from_node = line[0] - 1
            to_node = line[1] - 1
            self.a_mat[index, from_node] = 1.0
            self.a_mat[index, to_node] = -1.0
        self.lines_div_x = np.array([1.0 / line[2] for line in lines])
        d_mat = np.diag(self.lines_div_x)
        bp_mat = self.a_mat.T.dot(d_mat).dot(self.a_mat)
        offset_mat = np.full(shape=(self.numNodes, self.numNodes),
                             fill_value=float(np.mean(self.lines_div_x) / self.numNodes))
        b_hat_mat = bp_mat + offset_mat
        self.flowMat = d_mat.dot(self.a_mat).dot(np.linalg.inv(b_hat_mat))

        # initialise derivative structures for optimisation
        self.baseFlow = self.flowMat.dot(self.loadMax)
        wide_matrix = np.tile(self.flowMat, 2)
        self.optMatrix = np.vstack((wide_matrix, -wide_matrix))

        # cost coefficients for generation (0) and loss of load (1), respectively
        self.objective_scalars = np.concatenate((np.zeros(self.numNodes), np.ones(self.numNodes)), axis=0)

        self._HL1_expectation = None

    def generate_sample(self, level_set):
        return HLSample(level_set=level_set, system=self)

    def expectation_value_available(self, level):
        return True if level == 'HL1' else False

    def expectation_value(self, level):
        if level == 'HL1':
            if self._HL1_expectation is not None:
                return self._HL1_expectation
            else:
                gen_list = [gen_adequacy.Generator(unit_capacity=self.genCapacities[i],
                                                   unit_availability=self.genAvailability[i],
                                                   unit_mtbf=1.0,
                                                   unit_count=1)
                            for i in range(self.numGenerators)
                            ]

                rv_model = gen_adequacy.SingleNodeSystem(
                    gen_list=gen_list,
                    load_profile=np.sum(self.loadMax)*self.load_profile,
                    resolution=1
                )

                self._HL1_expectation = np.array((rv_model.lolp(), rv_model.epns()))
                return self._HL1_expectation
        else:
            raise NotImplementedError


class HLSample(MLSampleBase.MLSample):
    def __init__(self, level_set, system):
        # Required function: formally initialise sample by instantiating random realisation.
        super(self.__class__, self).__init__(level_set=level_set)

        self.system = system

        self.gen_status = (np.random.random(len(system.genAvailability)) < system.genAvailability).astype(int)
        self.load_factor = np.random.choice(system.load_profile)

    def generate_value(self, level):
        # Required function: generate sample outputs at various index levels.
        if level == 'HL1':
            return self.hl1_generator()
        elif level == 'HL2gen':
            return self.hl2gen_generator()
        elif level == 'HL2full':
            return self.hl2full_generator()
        else:
            raise RuntimeError()

    def hl1_generator(self):
        # Implement HL1 analysis
        total_power = self.system.genCapacities.dot(self.gen_status)
        total_load = self.load_factor * sum(self.system.loadMax)

        pns_hl1 = max(total_load - total_power, 0)
        if pns_hl1 > EFFECTIVE_ZERO:
            lol_hl1 = 1
        else:
            lol_hl1 = 0
        return np.array([lol_hl1, pns_hl1])

    def hl2gen_generator(self):
        # Implement HL2 analysis considering only generator outages
        node_power = self.system.nodeGenMatrix.dot(self.gen_status)
        node_load = self.load_factor * self.system.loadMax

        # Convention for optimisation: decision variables are generation levels and load shed

        decision_bounds = [(0, power) for power in node_power] + [(0, load) for load in node_load]
        # NOTE: the following modified version is required for numba(jit), which does not have list comprehension
        # decision_bounds = (len(node_power)+len(node_load))*[None]
        # for i in range(len(node_power)):
        #     decision_bounds[i] = (0, node_power[i])
        # for i in range(len(node_load)):
        #     decision_bounds[i + len(node_power)] = (0, node_load[i])

        upper_bounds = np.concatenate((self.system.lineCapacity + self.load_factor * self.system.baseFlow,
                                       self.system.lineCapacity - self.load_factor * self.system.baseFlow))

        opt_result = optimize.linprog(self.system.objective_scalars,
                                      A_ub=self.system.optMatrix, b_ub=upper_bounds,
                                      A_eq=np.ones((1, 2 * self.system.numNodes)), b_eq=np.sum(node_load),
                                      bounds=decision_bounds, method='revised simplex')

        assert opt_result.success

        pns_hl2gen = np.sum(opt_result.x[self.system.numNodes:])
        if pns_hl2gen > 1e-6:
            lol_hl2 = 1
        else:
            lol_hl2 = 0

        return np.array([lol_hl2, pns_hl2gen])

    def hl2full_generator(self):

        # sample line status, and revert to hl2gen level if all lines are up
        line_status = (np.random.random(len(self.system.line_availability)) < self.system.line_availability).astype(int)
        line_filter = np.nonzero(line_status)
        if line_status.all():
            return self.hl2gen_generator()

        islands = self.connected_islands(line_status=line_status)

        # if len(islands) > 1:
        #     log.info("processing line failure(s): {} island(s) created.".format(len(islands)))

        pns_hl2full = 0

        for island_nodes in islands:

            # Initialise connectivity and power flow matrix
            a_mat = self.system.a_mat[line_filter][:, island_nodes]
            lines_div_x = self.system.lines_div_x[line_filter]
            d_mat = np.diag(lines_div_x)
            bp_mat = a_mat.T.dot(d_mat).dot(a_mat)
            offset_mat = np.full(shape=(len(island_nodes), len(island_nodes)),
                                 fill_value=float(np.mean(lines_div_x) / len(island_nodes)))
            b_hat_mat = bp_mat + offset_mat
            flow_mat = d_mat.dot(a_mat).dot(np.linalg.inv(b_hat_mat))

            # initialise derivative structures for optimisation
            baseFlow = flow_mat.dot(self.system.loadMax[island_nodes])
            wide_matrix = np.tile(flow_mat, 2)
            optMatrix = np.vstack((wide_matrix, -wide_matrix))
            objective_scalars = np.concatenate((np.zeros(len(island_nodes)), np.ones(len(island_nodes))), axis=0)

            # Implement HL2 analysis
            node_power = (self.system.nodeGenMatrix.dot(self.gen_status))[island_nodes]
            node_load = (self.load_factor * self.system.loadMax)[island_nodes]

            # Convention for optimisation: decision variables are generation levels and load shed

            decision_bounds = [(0, power) for power in node_power] + [(0, load) for load in node_load]
            # NOTE: the following modified version is required for numba(jit), which does not have list comprehension
            # decision_bounds = (len(node_power)+len(node_load))*[None]
            # for i in range(len(node_power)):
            #     decision_bounds[i] = (0, node_power[i])
            # for i in range(len(node_load)):
            #     decision_bounds[i + len(node_power)] = (0, node_load[i])

            upper_bounds = np.concatenate(((self.system.lineCapacity[line_filter] + self.load_factor * baseFlow),
                                           (self.system.lineCapacity[line_filter] - self.load_factor * baseFlow)))

            opt_result = optimize.linprog(objective_scalars,
                                          A_ub=optMatrix, b_ub=upper_bounds,
                                          A_eq=np.ones((1, 2 * len(island_nodes))), b_eq=np.sum(node_load),
                                          bounds=decision_bounds, method='revised simplex')

            assert opt_result.success

            pns_hl2full += np.sum(opt_result.x[self.system.numNodes:])

        if pns_hl2full > 1e-6:
            lol_hl2b = 1
        else:
            lol_hl2b = 0

        return np.array([lol_hl2b, pns_hl2full])

    def connected_islands(self, line_status):

        neighbours = defaultdict(lambda: [])
        for line_idx, line_info in enumerate(self.system.lines):
            if line_status[line_idx] == 1:
                from_node = line_info[0] - 1
                to_node = line_info[1] - 1
                neighbours[from_node].append(to_node)
                neighbours[to_node].append(from_node)

        node_labels = np.array(range(self.system.numNodes))
        node_visited = np.full(shape=(self.system.numNodes,), fill_value=False)

        def tag_and_spread(node, value):
            if node_visited[node] == False:
                node_labels[node] = value
                node_visited[node] = True
                for next_node in neighbours[node]:
                    tag_and_spread(next_node, value)

        for node in range(self.system.numNodes):
            tag_and_spread(node=node, value=node_labels[node])

        # identify all clusters with identical node labels and return them
        node_numbers = np.array(range(self.system.numNodes))
        clusters = [node_numbers[node_labels == label] for label in np.unique(node_labels)]

        return clusters

