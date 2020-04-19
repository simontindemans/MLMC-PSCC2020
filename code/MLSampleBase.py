# -*- coding: utf-8 -*-
"""
Provides base classes that are used as scaffolding for an MLMC simulation.

Two classes are defined:
MLSampleFactory : Base class for sample-generating objects.
MLSample        : Base class for generated samples.

Together, the classes form an interface to the MCCoordinator class, which orchestrates Monte Carlo sampling. In an
implementation, the model representation (including parameters) will be embedded in an object that derives from
MLSampleFactory. This object is used to generate samples, which are derived from MLSample.

Example:
    In a reliability analysis application, the object deriving from MLSampleFactory contains the physical system
    representation and its reliability models (fault rates, etc). The overridden member function
    MLSampleFactory.generate_sample(idx_set) is used to generate an MLSample object. It represents the random system
    state, and can be queried at any 'level' (idx) that is part of idx_set, using MLSample[idx].

Target: python 3.7

@author: Simon Tindemans
Delft University of Technology
s.h.tindemans@tudelft.nl
"""
# SPDX-License-Identifier: MIT



from collections import defaultdict

class MLSampleFactory:
    """
    Base class for sample-generating objects.

    MLSampleFactory provides the core interface for the Monte Carlo coordinator (MCCoordinator). It should be used as
    a base class for the core simulation engine.

    Exposed interface for derived classes:
        __init__(.)                     : Initialise available MC levels and combinations - can be extended
        generate_sample()               : Generate a sample - must be overridden
        expectation_value(.)            : Compute expectation values without sampling - can be overridden
        expectation_value_available(.)       : Test availability of expectation value without sampling - can be overridden

    """
    def __init__(self, output_labels, output_units, available_levels, suggested_hierarchy, permissible_level_sets):
        """
        MLSampleFactory constructor initialises available MC levels and combinations, and expected outputs.

        NOTE: The constructor can be extended (but not overridden) in the derived class. If present, the derived class
        constructor *must* call this constructor.
           super(self.__class__, self).__init__( ..... )

        :param available_levels: list of all supported MC levels (can be of any type)
        :param target_level: label of target MC level, usually the highest resolution model
        :param output_labels: list of strings, describing the output values of the samples (i.e. properties of the sampled state)
        :param output_units: list of strings, containing the display units for the outputs
        :param permissible_level_sets: list of lists with permissible level combinations for sampling
        :param suggested_level_sets: optional list of lists with suggested level combinations for sampling

        """

        # Check that we are not overwriting something previously defined in the derived class
        assert not hasattr(self, 'available_levels')
        assert not hasattr(self, 'target_level')
        assert not hasattr(self, 'permissible_level_sets')
        assert not hasattr(self, 'suggested_level_sets')
        assert not hasattr(self, 'suggested_hierarchy')
        assert not hasattr(self, 'output_labels')
        assert not hasattr(self, 'output_units')
        # Check consistency of inputs
        assert set(suggested_hierarchy) <= set(available_levels)
        assert len(output_labels) == len(output_units)

        # convert via dictionary to ensure uniqueness (and maintain order in Python 3.7+)
        self.available_levels = list({idx: None for idx in available_levels})
        self.output_labels = output_labels
        self.output_units = output_units
        self.suggested_hierarchy = suggested_hierarchy
        self.permissible_level_sets = permissible_level_sets

    def generate_sample(self, level_set):
        """
        Generate and return a sample, derived from HLSample, that is valid for every element of the list idx_set.

        NOTE: This function must be overridden in the derived class

        :param level_set: list of valid MC level indices
        :return: HLSample object (usually subclassed)
        """
        raise NotImplementedError()

    def expectation_value(self, level):
        """
        Base class stub to compute the expectation value for MC level idx. Always returns the tuple (False, None).

        This function can be overridden in a derived class to supply efficient estimators of expectation values.

        :param level: label of target MC level (can be of any type, but should be an element of self.available_levels)
        :return: (False, None), indicating the lack of implementation
        """
        raise NotImplementedError

    def expectation_value_available(self, level):
        """
        Base class stub to compute the availability of expectation value for MC level idx. Always returns False.

        This function can be overridden in a derived class to supply efficient estimators of expectation values.

        :param level: label of target MC level (can be of any type, but should be an element of self.available_levels)
        :return: False, indicating the lack of implementation
        """
        return False


class MLSample:
    """
    Base class for correlated Monte Carlo samples, generated by an MLSampleFactory object.

    MLSample provides the core interface for storing and evaluating MC samples at various levels. It should be used as
    a base class for specialised samples.

    NOTE: implementation is currently unsafe: MCCoordinator checks suitability; sample does not.

    Exposed interface for derived classes:
        __init__(.)                     : Initialise available MC levels and combinations - can be extended
        generate_value(.)               : Generate output value corresponding to the MC level - must be overridden

    Properties:
        get_idx_set(.)                  : Return list of implemented MC levels

    Methods:
        __getitem__(.)                  : Overrides [] operator to access outputs at particular MC levels
    """

    def __init__(self, level_set):
        """
        MLSample constructor.

        NOTE: The constructor can be extended (but not overridden) in the derived class. If present, the derived class
        constructor *must* call this constructor, e.g. using
           super(self.__class__, self).__init__(idx_set)

        :param level_set: List of MC index levels (must be elements of MLSampleFactory.available_levels)
        """
        # Check that we are not overwriting something defined in the derived class
        assert not hasattr(self, '_realisations')

        self._requested_level_set = level_set
        # Create an empty output dictionary
        self._realisations = defaultdict(lambda: None)

    def generate_value(self, level):
        """
        Base class stub for sample evaluator function. Must be overridden in derived class.

        This function is only called from self.__getitem__(.).
        :param level: Label of MC level
        :return: numpy.ndarray object [but this stub does not return anything]
        """
        raise NotImplementedError()

    def __getitem__(self, level):
        """Implements the [] operator. Return a numpy.ndarray output at level 'idx', and generate its value if necessary.

        :param level: label of MC level
        :return: numpy.ndarray object with outputs
        """

        if self._realisations[level] is None:
            # Must generate realisation at this depth. This allows for 'lazy' construction of samples, provided that
            # the probability structure allows for this.
            self._realisations[level] = self.generate_value(level)

        return self._realisations[level]








