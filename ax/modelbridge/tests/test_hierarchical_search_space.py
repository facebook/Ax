#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from contextlib import ExitStack
from random import random

from ax.core.experiment import Experiment
from ax.core.objective import Objective
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.search_space import HierarchicalSearchSpace
from ax.core.trial import Trial
from ax.metrics.noisy_function import GenericNoisyFunctionMetric
from ax.modelbridge.cross_validation import cross_validate
from ax.modelbridge.registry import Models
from ax.runners.synthetic import SyntheticRunner
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import mock_botorch_optimize
from pyre_extensions import assert_is_instance, none_throws


class TestHierarchicalSearchSpace(TestCase):
    """Tests for various modelbridge functionality with commonly used transforms
    using hierarchical search spaces (HSS).
    """

    def setUp(self) -> None:
        super().setUp()
        int_range = RangeParameter(
            name="int_range",
            parameter_type=ParameterType.INT,
            lower=0,
            upper=10,
        )
        str_choice = ChoiceParameter(
            name="str_choice",
            parameter_type=ParameterType.STRING,
            values=["a", "b", "c"],
        )
        fixed_root = FixedParameter(
            name="root",
            parameter_type=ParameterType.STRING,
            value="root",
            dependents={"root": ["int_range", "str_choice"]},
        )
        # This HSS does not have a real hierarchy.
        self.non_hierarchical_hss = HierarchicalSearchSpace(
            parameters=[
                fixed_root,
                int_range,
                str_choice,
            ]
        )
        choice_root = ChoiceParameter(
            name="root",
            parameter_type=ParameterType.STRING,
            values=["range", "choice"],
            dependents={"range": ["int_range"], "choice": ["str_choice"]},
        )
        # This HSS has a simple hierarchy -- one parameter on each branch.
        self.simple_hss = HierarchicalSearchSpace(
            parameters=[choice_root, int_range, str_choice]
        )
        fixed_leaf = FixedParameter(
            name="fixed_leaf",
            parameter_type=ParameterType.STRING,
            value="leaf",
        )
        middle_choice = ChoiceParameter(
            name="middle_choice",
            parameter_type=ParameterType.INT,
            values=[0, 1],
            dependents={0: ["fixed_leaf"], 1: ["int_range", "str_choice"]},
        )
        int_choice = ChoiceParameter(
            name="int_choice",
            parameter_type=ParameterType.INT,
            values=[0, 1, 2, 3],
            is_ordered=False,
        )
        float_range = RangeParameter(
            name="float_range",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=5.0,
        )
        choice_root2 = ChoiceParameter(
            name="root2",
            parameter_type=ParameterType.BOOL,
            values=[True, False],
            dependents={True: ["middle_choice", "float_range"], False: ["int_choice"]},
        )
        # This HSS has a more complex, multi-level hierarchy.
        self.complex_hss = HierarchicalSearchSpace(
            parameters=[
                choice_root2,
                int_choice,
                middle_choice,
                float_range,
                fixed_leaf,
                int_range,
                str_choice,
            ]
        )

    @mock_botorch_optimize
    def _test_gen_base(
        self,
        hss: HierarchicalSearchSpace,
        expected_num_candidate_params: list[int],
        num_sobol_trials: int = 5,
        num_bo_trials: int = 5,
    ) -> Experiment:
        """Test Sobol & MBM candidate generation with HSS using default transforms.

        Args:
            hss: The hierarchical search space to test.
            expected_num_candidate_params: The expected number of parameters in each
                candidate. This list should include all possible values, since different
                branches of HSS may have different numbers of parameters.
            num_sobol_trials: The number of Sobol trials to run.
            num_bo_trials: The number of BO trials to run.

        Returns:
            The experiment with the generated candidates. This can be used to chain
            tests for other functionality that requires data.
        """
        experiment = Experiment(
            name="test_experiment",
            search_space=hss,
            optimization_config=OptimizationConfig(
                objective=Objective(
                    metric=GenericNoisyFunctionMetric(
                        name="random", f=lambda _: random()
                    ),
                    minimize=True,
                )
            ),
            runner=SyntheticRunner(),
        )

        sobol = Models.SOBOL(search_space=hss)
        for _ in range(num_sobol_trials):
            trial = experiment.new_trial(generator_run=sobol.gen(n=1))
            trial.run().mark_completed()

        for _ in range(num_bo_trials):
            mbm = Models.BOTORCH_MODULAR(
                experiment=experiment, data=experiment.fetch_data()
            )
            trial = experiment.new_trial(generator_run=mbm.gen(n=1))
            trial.run().mark_completed()

        for t in experiment.trials.values():
            trial = assert_is_instance(t, Trial)
            arm = none_throws(trial.arm)
            self.assertIn(len(arm.parameters), expected_num_candidate_params)
            # Check that the trials have the full parameterization recorded.
            full_parameterization = none_throws(
                trial._get_candidate_metadata(arm_name=arm.name)
            )[Keys.FULL_PARAMETERIZATION]
            self.assertEqual(full_parameterization.keys(), hss.parameters.keys())

        return experiment

    @mock_botorch_optimize
    def _base_test_predict_and_cv(
        self,
        experiment: Experiment,
        expect_errors_with_final_parameterization: bool = False,
    ) -> None:
        """Test predict and cross validation with a given experiment.
        The predict tests are done using the full parameterization, the
        final parameterization with the full parameterization recorded in
        metadata, and with the final parameterization only. When the final
        parameterization lacks some parameters, this may error out.
        `expect_errors_with_final_parameterization` arg is used to handle
        the `KeyError` that is expected (but should be fixed) in this setting.
        """
        mbm = Models.BOTORCH_MODULAR(
            experiment=experiment, data=experiment.fetch_data()
        )
        for t in experiment.trials.values():
            trial = assert_is_instance(t, Trial)
            arm = none_throws(trial.arm)
            final_parameterization = arm.parameters
            full_parameterization = none_throws(
                trial._get_candidate_metadata(arm_name=arm.name)
            )[Keys.FULL_PARAMETERIZATION]
            # Predict with full parameterization -- this should always work.
            mbm.predict([ObservationFeatures(parameters=full_parameterization)])
            # Predict with final parameterization -- this may error out when
            # ``inject_dummy_values_to_complete_flat_parameterization``  is False.
            # The new default is True, so it should not happen in these tests.
            with ExitStack() as es:
                if expect_errors_with_final_parameterization:
                    es.enter_context(self.assertRaises(KeyError))
                mbm.predict([ObservationFeatures(parameters=final_parameterization)])
            # Predict with final parameterization but include the full parameterization
            # in the metadata. This is similar to what happens inside cross_validate.
            mbm.predict(
                [
                    ObservationFeatures(
                        parameters=final_parameterization,
                        metadata={Keys.FULL_PARAMETERIZATION: full_parameterization},
                    )
                ]
            )
        cv_res = cross_validate(model=mbm)
        self.assertEqual(len(cv_res), len(experiment.trials))

    def test_with_non_hierarchical_hss(self) -> None:
        experiment = self._test_gen_base(
            hss=self.non_hierarchical_hss, expected_num_candidate_params=[3]
        )
        self._base_test_predict_and_cv(experiment=experiment)

    def test_with_simple_hss(self) -> None:
        experiment = self._test_gen_base(
            hss=self.simple_hss, expected_num_candidate_params=[2]
        )
        self._base_test_predict_and_cv(experiment=experiment)

    def test_with_complex_hss(self) -> None:
        experiment = self._test_gen_base(
            hss=self.complex_hss, expected_num_candidate_params=[2, 4, 5]
        )
        self._base_test_predict_and_cv(experiment=experiment)
