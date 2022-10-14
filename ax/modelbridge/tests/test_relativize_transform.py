# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from unittest.mock import Mock

import numpy as np
from ax.core.observation import (
    Observation,
    ObservationData,
    ObservationFeatures,
    observations_from_data,
    recombine_observations,
)
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types import ComparisonOp
from ax.metrics.branin import BraninMetric
from ax.modelbridge.registry import Models
from ax.modelbridge.transforms.relativize import Relativize
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import not_none
from ax.utils.stats.statstools import relativize_data
from ax.utils.testing.core_stubs import (
    get_branin_multi_objective_optimization_config,
    get_branin_optimization_config,
    get_branin_with_multi_task,
    get_search_space,
)
from hypothesis import assume, given, settings, strategies as st


class RelativizeDataTest(TestCase):
    def test_relativize_transform_requires_a_modelbridge(self) -> None:
        with self.assertRaisesRegex(ValueError, "modelbridge"):
            Relativize(
                search_space=None,
                observations=[],
            )

    def test_relativize_transform_requires_a_modelbridge_to_have_status_quo_data(
        self,
    ) -> None:
        sobol = Models.SOBOL(search_space=get_search_space())
        self.assertIsNone(sobol.status_quo)
        with self.assertRaisesRegex(ValueError, "status quo data"):
            Relativize(
                search_space=None,
                observations=[],
                modelbridge=sobol,
            ).transform_observations(
                observations=[
                    Observation(
                        data=ObservationData(
                            metric_names=["foo"],
                            means=np.array([2]),
                            covariance=np.array([[0.1]]),
                        ),
                        features=ObservationFeatures(parameters={"x": 1}),
                    )
                ],
            )

    def test_relativize_transform_observations(self) -> None:
        obs_data = [
            ObservationData(
                metric_names=["foobar", "foobaz"],
                means=np.array([2, 5]),
                covariance=np.array([[0.1, 0.0], [0.0, 0.2]]),
            ),
            ObservationData(
                metric_names=["foobar", "foobaz"],
                means=np.array([1.0, 10.0]),
                covariance=np.array([[0.3, 0.0], [0.0, 0.4]]),
            ),
        ]
        obs_features = [
            # pyre-fixme[6]: For 2nd param expected `Optional[int64]` but got `int`.
            ObservationFeatures(parameters={"x": 1}, trial_index=0),
            # pyre-fixme[6]: For 2nd param expected `Optional[int64]` but got `int`.
            ObservationFeatures(parameters={"x": 2}, trial_index=0),
        ]
        observations = recombine_observations(obs_features, obs_data)
        modelbridge = Mock(
            status_quo=Mock(
                data=obs_data[0],
                features=obs_features[0],
            )
        )
        tf = Relativize(
            search_space=None,
            observations=observations,
            modelbridge=modelbridge,
        )
        results = tf.transform_observations(observations)
        self.assertEqual(results[0].data.metric_names, ["foobar", "foobaz"])
        # status quo means must always be zero
        self.assertTrue(
            np.allclose(results[0].data.means, np.array([0.0, 0.0])),
            results[0].data.means,
        )
        # status quo covariances must always be zero
        self.assertTrue(
            np.allclose(results[0].data.covariance, np.array([[0.0, 0.0], [0.0, 0.0]])),
            results[0].data.covariance,
        )
        self.assertEqual(results[1].data.metric_names, ["foobar", "foobaz"])
        self.assertTrue(
            np.allclose(results[1].data.means, np.array([-51.25, 98.4])),
            results[1].data.means,
        )
        self.assertTrue(
            np.allclose(
                results[1].data.covariance, np.array([[812.5, 0.0], [0.0, 480.0]])
            ),
            results[1].data.covariance,
        )
        obsd_t = tf._untransform_observation_data(obs_data)
        self.assertEqual(obsd_t, obs_data)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `hypothesis.strategies.floats($parameter$min_value = - 10.000000,
    #  $parameter$max_value = 10.000000)` to decorator factory `hypothesis.given`.
    @given(
        st.floats(min_value=-10.0, max_value=10.0),
        st.floats(min_value=0, max_value=10.0),
        st.floats(min_value=-10.0, max_value=10.0),
        st.floats(min_value=0, max_value=10.0),
    )
    @settings(max_examples=1000, deadline=None)
    def test_transform_status_quos_always_zero(
        self,
        sq_mean: float,
        sq_sem: float,
        mean: float,
        sem: float,
    ) -> None:
        assume(abs(sq_mean) >= 1e-10)
        assume(abs(sq_mean) != sq_sem)

        obs_data = [
            ObservationData(
                metric_names=["foo"],
                means=np.array([sq_mean]),
                covariance=np.array([[sq_sem]]),
            ),
            ObservationData(
                metric_names=["foo"],
                means=np.array([mean]),
                covariance=np.array([[sem]]),
            ),
        ]
        obs_features = [
            # pyre-fixme[6]: For 2nd param expected `Optional[int64]` but got `int`.
            ObservationFeatures(parameters={"x": 1}, trial_index=0),
            # pyre-fixme[6]: For 2nd param expected `Optional[int64]` but got `int`.
            ObservationFeatures(parameters={"x": 2}, trial_index=0),
        ]
        modelbridge = Mock(
            status_quo=Mock(
                data=obs_data[0],
                features=obs_features[0],
            )
        )
        observations = recombine_observations(obs_features, obs_data)
        transform = Relativize(
            search_space=None,
            observations=observations,
            modelbridge=modelbridge,
        )
        relative_obs = transform.transform_observations(observations)
        self.assertEqual(relative_obs[0].data.metric_names, ["foo"])
        self.assertAlmostEqual(relative_obs[0].data.means[0], 0, places=4)
        self.assertAlmostEqual(relative_obs[0].data.covariance[0][0], 0, places=4)

    def test_multitask_data(self) -> None:
        experiment = get_branin_with_multi_task()
        data = experiment.fetch_data()

        observations = observations_from_data(
            experiment=experiment,
            data=data,
        )
        relative_observations = observations_from_data(
            experiment=experiment,
            data=relativize_data(
                data=data,
                status_quo_name="status_quo",
                as_percent=True,
                include_sq=True,
            ),
        )
        status_quo_row = data.df.loc[
            (data.df["arm_name"] == "status_quo") & (data.df["trial_index"] == 1)
        ]
        modelbridge = Mock(
            status_quo=Observation(
                data=ObservationData(
                    metric_names=status_quo_row["metric_name"].values,
                    means=status_quo_row["mean"].values,
                    covariance=np.array([status_quo_row["sem"].values ** 2]),
                ),
                features=ObservationFeatures(
                    parameters=not_none(experiment.status_quo).parameters
                ),
            )
        )

        transform = Relativize(
            search_space=None,
            observations=observations,
            modelbridge=modelbridge,
        )
        relative_obs_t = transform.transform_observations(observations)
        self.maxDiff = None
        # this assertion just checks that order is the same, which
        # is only important for the purposes of this test
        self.assertEqual(
            [datum.data.metric_names for datum in relative_obs_t],
            [datum.data.metric_names for datum in relative_observations],
        )
        means = [
            np.array([datum.data.means for datum in relative_obs_t]),
            np.array([datum.data.means for datum in relative_observations]),
        ]
        # `self.assertAlmostEqual(relative_obs_data, expected_obs_data)`
        # fails 1% of the time, so we check with numpy.
        self.assertTrue(
            all(np.isclose(means[0], means[1])),
            means,
        )
        covariances = [
            np.array([datum.data.covariance for datum in relative_observations]),
            np.array([datum.data.covariance for datum in relative_obs_t]),
        ]
        self.assertTrue(
            all(np.isclose(covariances[0], covariances[1])),
            covariances,
        )


class RelativizeDataOptConfigTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        search_space = get_search_space()
        gr = Models.SOBOL(search_space=search_space).gen(n=1)
        self.model = Mock(
            search_space=search_space,
            status_quo=Mock(
                features=ObservationFeatures(parameters=gr.arms[0].parameters)
            ),
        )

    def test_transform_optimization_config_without_constraints(self) -> None:
        relativize = Relativize(
            search_space=None,
            observations=[],
            modelbridge=self.model,
        )
        optimization_config = get_branin_optimization_config()
        new_config = relativize.transform_optimization_config(
            optimization_config=optimization_config,
            modelbridge=None,
            fixed_features=Mock(),
        )
        self.assertEqual(new_config.objective, optimization_config.objective)

    def test_transform_optimization_config_with_relative_constraints(self) -> None:
        relativize = Relativize(
            search_space=None,
            observations=[],
            modelbridge=self.model,
        )
        optimization_config = get_branin_optimization_config()
        optimization_config.outcome_constraints = [
            OutcomeConstraint(
                metric=BraninMetric("b2", ["x2", "x1"]),
                op=ComparisonOp.GEQ,
                bound=-200.0,
                relative=True,
            )
        ]
        new_config = relativize.transform_optimization_config(
            optimization_config=deepcopy(optimization_config),
            modelbridge=None,
            fixed_features=Mock(),
        )
        self.assertEqual(new_config.objective, optimization_config.objective)
        self.assertEqual(
            new_config.outcome_constraints[0].bound,
            optimization_config.outcome_constraints[0].bound,
        )
        self.assertFalse(new_config.outcome_constraints[0].relative)
        # Untransform the constraints
        cons = relativize.untransform_outcome_constraints(
            outcome_constraints=new_config.outcome_constraints, fixed_features=Mock()
        )
        self.assertEqual(cons, optimization_config.outcome_constraints)

    def test_transform_optimization_config_with_non_relative_constraints(self) -> None:
        relativize = Relativize(
            search_space=None,
            observations=[],
            modelbridge=self.model,
        )
        optimization_config = get_branin_optimization_config()
        optimization_config.outcome_constraints = [
            OutcomeConstraint(
                metric=BraninMetric("b2", ["x2", "x1"]),
                op=ComparisonOp.GEQ,
                bound=-200.0,
                relative=False,
            )
        ]
        with self.assertRaisesRegex(ValueError, "All constraints must be relative"):
            relativize.transform_optimization_config(
                optimization_config=optimization_config,
                modelbridge=None,
                fixed_features=Mock(),
            )

    def test_transform_optimization_config_with_relative_thresholds(self) -> None:
        relativize = Relativize(
            search_space=None,
            observations=[],
            modelbridge=self.model,
        )
        optimization_config = get_branin_multi_objective_optimization_config(
            has_objective_thresholds=True,
        )
        for threshold in optimization_config.objective_thresholds:
            threshold.relative = True

        new_config = relativize.transform_optimization_config(
            optimization_config=optimization_config,
            modelbridge=None,
            fixed_features=Mock(),
        )
        self.assertEqual(new_config.objective, optimization_config.objective)
        self.assertEqual(
            # pyre-fixme[16]: `OptimizationConfig` has no attribute
            #  `objective_thresholds`.
            new_config.objective_thresholds[0].bound,
            optimization_config.objective_thresholds[0].bound,
        )
        self.assertFalse(new_config.objective_thresholds[0].relative)
        self.assertEqual(
            new_config.objective_thresholds[1].bound,
            optimization_config.objective_thresholds[1].bound,
        )
        self.assertFalse(new_config.objective_thresholds[1].relative)

    def test_transform_optimization_config_with_non_relative_thresholds(self) -> None:
        relativize = Relativize(
            search_space=None,
            observations=[],
            modelbridge=self.model,
        )
        optimization_config = get_branin_multi_objective_optimization_config(
            has_objective_thresholds=True,
        )
        optimization_config.objective_thresholds[1].relative = False

        with self.assertRaisesRegex(
            ValueError, "All objective thresholds must be relative"
        ):
            relativize.transform_optimization_config(
                optimization_config=optimization_config,
                modelbridge=None,
                fixed_features=Mock(),
            )
