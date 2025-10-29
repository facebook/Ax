# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy
from itertools import product
from unittest.mock import Mock

import numpy as np
import numpy.typing as npt
from ax.adapter import Adapter
from ax.adapter.base import DataLoaderConfig
from ax.adapter.data_utils import extract_experiment_data
from ax.adapter.registry import Generators
from ax.adapter.transforms.base import Transform
from ax.adapter.transforms.relativize import (
    BaseRelativize,
    Relativize,
    relativize,
    RelativizeWithConstantControl,
)
from ax.core import BatchTrial
from ax.core.arm import Arm
from ax.core.experiment import Experiment
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.observation_utils import observations_from_data, recombine_observations
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types import ComparisonOp
from ax.exceptions.core import DataRequiredError
from ax.generators.base import Generator
from ax.metrics.branin import BraninMetric
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_data_batch,
    get_branin_experiment,
    get_branin_multi_objective_optimization_config,
    get_branin_optimization_config,
    get_branin_with_multi_task,
    get_search_space,
)
from pandas.testing import assert_frame_equal
from pyre_extensions import assert_is_instance, none_throws


class RelativizeDataTest(TestCase):
    relativize_classes: list[type[Transform]] = [
        Relativize,
        RelativizeWithConstantControl,
    ]
    cases: list[tuple[type[Transform], list[tuple[npt.NDArray, npt.NDArray]]]] = [
        (
            Relativize,
            [
                (np.array([0.0, 0.0]), np.array([[0.0, 0.0], [0.0, 0.0]])),
                (
                    np.array([-22.56, 98.01652893]),
                    np.array([[604.8, 0.0], [0.0, 512.39669421]]),
                ),
                (np.array([0.0, 0.0]), np.array([[0.0, 0.0], [0.0, 0.0]])),
                (np.array([-51.25, 98.4]), np.array([[812.5, 0.0], [0.0, 480.0]])),
            ],
        ),
        (
            RelativizeWithConstantControl,
            [
                (np.array([0.0, 0.0]), np.array([[0.0, 0.0], [0.0, 0.0]])),
                (
                    np.array([-20.0, 100.0]),
                    np.array([[400.0, 0.0], [0.0, 115.70247934]]),
                ),
                (np.array([0.0, 0.0]), np.array([[0.0, 0.0], [0.0, 0.0]])),
                (np.array([-50.0, 100.0]), np.array([[750.0, 0.0], [0.0, 160.0]])),
            ],
        ),
    ]

    def test_relativize_transform_requires_a_adapter(self) -> None:
        for relativize_cls in self.relativize_classes:
            with self.assertRaisesRegex(
                AssertionError,
                f"{relativize_cls.__name__} transform requires an adapter",
            ):
                relativize_cls(search_space=None)

    def test_relativize_transform_requires_a_adapter_to_have_status_quo_data(
        self,
    ) -> None:
        for relativize_cls in self.relativize_classes:
            # adapter has no status quo
            sobol = Generators.SOBOL(experiment=get_branin_experiment())
            self.assertIsNone(sobol.status_quo)
            with self.assertRaisesRegex(
                DataRequiredError,
                f"{relativize_cls.__name__} requires status quo data.",
            ):
                relativize_cls(search_space=None, adapter=sobol).transform_observations(
                    observations=[
                        Observation(
                            data=ObservationData(
                                metric_signatures=["foo"],
                                means=np.array([2]),
                                covariance=np.array([[0.1]]),
                            ),
                            features=ObservationFeatures(parameters={"x": 1}),
                            arm_name="0_0",
                        )
                    ],
                )

            # adapter has status quo
            exp = get_branin_experiment(with_batch=True, with_status_quo=True)
            # making status_quo out of design
            none_throws(exp._status_quo)._parameters["x1"] = 10000.0
            for t in exp.trials.values():
                t.mark_running(no_runner_required=True)
                exp.attach_data(
                    get_branin_data_batch(batch=assert_is_instance(t, BatchTrial))
                )
                t.mark_completed()
            data = exp.fetch_data()
            adapter = Adapter(
                experiment=exp, generator=Generator(), transforms=[relativize_cls]
            )
            mean_in_data = data.df.query(
                f"arm_name == '{none_throws(exp.status_quo).name}'"
            )["mean"].item()
            # adapter.status_quo_data_by_trial is accurate
            self.assertEqual(
                mean_in_data,
                none_throws(adapter.status_quo_data_by_trial)[0].means[0],
            )

            # create a new experiment
            new_exp = get_branin_experiment(
                with_batch=True,
                with_status_quo=True,
            )
            for t in new_exp.trials.values():
                t.mark_running(no_runner_required=True)
                new_exp.attach_data(
                    get_branin_data_batch(batch=assert_is_instance(t, BatchTrial))
                )
                t.mark_completed()
            new_data = new_exp.fetch_data()
            # Construct adapter with the new data.
            adapter = Adapter(
                experiment=new_exp, generator=Generator(), transforms=[relativize_cls]
            )
            # The new data is different from the original data
            self.assertNotEqual(data, new_data)
            self.assertFalse(data.df.equals(new_data.df))
            mean_in_data = new_data.df.query(
                f"arm_name == '{none_throws(new_exp.status_quo).name}'"
            )["mean"].item()
            # adapter.status_quo_data_by_trial remains accurate
            self.assertEqual(
                mean_in_data,
                none_throws(adapter.status_quo_data_by_trial)[0].means[0],
            )
            # test transform edge cases
            observations = observations_from_data(
                experiment=exp,
                data=data,
            )
            tf = relativize_cls(search_space=None, adapter=adapter)
            # making observation coming from trial_index not in adapter
            observations[0].features.trial_index = 999
            self.assertRaises(ValueError, tf.transform_observations, observations)

    def test_relativize_transform_observations(self) -> None:
        def _check_transform_observations(
            tf: Transform,
            observations: list[Observation],
            expected_mean_and_covar: list[tuple[npt.NDArray, npt.NDArray]],
        ) -> None:
            results = tf.transform_observations(observations)
            for i, tsfm_obs in enumerate(results):
                expected_mean, expected_covar = expected_mean_and_covar[i]
                self.assertEqual(tsfm_obs.data.metric_signatures, metric_signatures)
                # status quo means must always be zero
                self.assertTrue(
                    np.allclose(tsfm_obs.data.means, expected_mean),
                    tsfm_obs.data.means,
                )
                # status quo covariances must always be zero
                self.assertTrue(
                    np.allclose(tsfm_obs.data.covariance, expected_covar),
                    tsfm_obs.data.covariance,
                )
            # Check untransform
            untsfm_results = tf.untransform_observations(results)
            j = 0
            for untsfm_obs in untsfm_results:
                obs = observations[j]
                # skip status quo for the non-target trial since that
                # is removed when transforming observations
                if untsfm_obs.arm_name != obs.arm_name:
                    j += 1
                    obs = observations[j]
                self.assertTrue(np.allclose(untsfm_obs.data.means, obs.data.means))
                self.assertTrue(
                    np.allclose(untsfm_obs.data.covariance, obs.data.covariance)
                )
                j += 1

        metric_signatures: list[str] = ["foobar", "foobaz"]
        arm_names = ["status_quo", "0_0", "status_quo", "1_0"]
        obs_data = [
            ObservationData(
                metric_signatures=metric_signatures,
                means=np.array([2.5, 5.5]),
                covariance=np.array([[0.2, 0.0], [0.0, 0.3]]),
            ),
            ObservationData(
                metric_signatures=metric_signatures,
                means=np.array([2.0, 11.0]),
                covariance=np.array([[0.25, 0.0], [0.0, 0.35]]),
            ),
            ObservationData(
                metric_signatures=metric_signatures,
                means=np.array([2.0, 5.0]),
                covariance=np.array([[0.1, 0.0], [0.0, 0.2]]),
            ),
            ObservationData(
                metric_signatures=metric_signatures,
                means=np.array([1.0, 10.0]),
                covariance=np.array([[0.3, 0.0], [0.0, 0.4]]),
            ),
        ]
        obs_features = [
            ObservationFeatures(parameters={"x": 1}, trial_index=0),
            ObservationFeatures(parameters={"x": 2}, trial_index=0),
            ObservationFeatures(parameters={"x": 1}, trial_index=1),
            ObservationFeatures(parameters={"x": 3}, trial_index=1),
        ]

        observations = recombine_observations(obs_features, obs_data, arm_names)
        adapter = Mock(
            status_quo=Mock(
                data=obs_data[2], features=obs_features[2], arm_name=arm_names[2]
            ),
            status_quo_name=arm_names[2],
            status_quo_data_by_trial={0: obs_data[0], 1: obs_data[2]},
            _experiment=None,
        )

        for relativize_cls, expected_mean_and_covar in self.cases:
            tf = relativize_cls(
                search_space=None,
                adapter=adapter,
            )
            # check transform and untransform on observations
            _check_transform_observations(
                tf=tf,
                observations=observations,
                expected_mean_and_covar=expected_mean_and_covar,
            )
            # transform should still work when trial_index is None
            if relativize_cls in [RelativizeWithConstantControl, Relativize]:
                adapter = Mock(
                    status_quo=Mock(
                        data=obs_data[2],
                        features=obs_features[2],
                        arm_name=arm_names[2],
                    ),
                    status_quo_data_by_trial={0: obs_data[0], 1: obs_data[2]},
                )
                tf = relativize_cls(
                    search_space=None,
                    adapter=adapter,
                )
                observations2 = deepcopy(observations)
                for obs in observations2:
                    obs.features.trial_index = None
                _check_transform_observations(
                    tf=tf,
                    observations=observations2[2:4],
                    expected_mean_and_covar=expected_mean_and_covar[2:4],
                )

    def test_bad_relativize(self) -> None:
        # Check instantiation and subclassing of BaseRelativize
        class BadRelativize(BaseRelativize):
            pass

        for abstract_cls in [BaseRelativize, BadRelativize]:
            with self.assertRaisesRegex(TypeError, "Can't instantiate abstract class"):
                abstract_cls(search_space=None, adapter=None)

    def test_transform_status_quos_always_zero(self) -> None:
        for _ in range(1000):
            sq_mean = np.random.uniform(-10.0, 10.0)
            sq_sem = np.random.uniform(0, 10.0)
            mean = np.random.uniform(-10.0, 10.0)
            sem = np.random.uniform(0, 10.0)
            if abs(sq_mean) < 1e-10 or abs(sq_mean) == sq_sem:
                continue

            arm_names = ["status_quo", "0_0"]
            obs_data = [
                ObservationData(
                    metric_signatures=["foo"],
                    means=np.array([sq_mean]),
                    covariance=np.array([[sq_sem]]),
                ),
                ObservationData(
                    metric_signatures=["foo"],
                    means=np.array([mean]),
                    covariance=np.array([[sem]]),
                ),
            ]
            obs_features = [
                ObservationFeatures(parameters={"x": 1}, trial_index=0),
                ObservationFeatures(parameters={"x": 2}, trial_index=0),
            ]
            adapter = Mock(
                status_quo=Mock(
                    data=obs_data[0], features=obs_features[0], arm_name=arm_names[0]
                ),
                status_quo_data_by_trial={0: obs_data[0]},
            )
            observations = recombine_observations(obs_features, obs_data, arm_names)
            for relativize_cls in [Relativize, RelativizeWithConstantControl]:
                transform = relativize_cls(search_space=None, adapter=adapter)
                relative_obs = transform.transform_observations(observations)
                self.assertEqual(relative_obs[0].data.metric_signatures, ["foo"])
                self.assertAlmostEqual(relative_obs[0].data.means[0], 0, places=4)
                self.assertAlmostEqual(
                    relative_obs[0].data.covariance[0][0], 0, places=4
                )

    def test_multitask_data(self) -> None:
        experiment = get_branin_with_multi_task()
        data = experiment.fetch_data()

        observations = observations_from_data(
            experiment=experiment,
            data=data,
        )
        relative_observations = observations_from_data(
            experiment=experiment,
            data=data.relativize(
                status_quo_name="status_quo",
                as_percent=True,
                include_sq=True,
            ),
        )

        sq_obs_data = []
        for i in data.df["trial_index"].unique():
            status_quo_data = data.df.loc[
                (data.df["arm_name"] == "status_quo") & (data.df["trial_index"] == i)
            ]
            sq_obs_data.append(
                ObservationData(
                    metric_signatures=status_quo_data["metric_signature"].to_list(),
                    means=status_quo_data["mean"].to_numpy(),
                    covariance=status_quo_data["sem"].to_numpy()[np.newaxis, :] ** 2,
                )
            )

        adapter = Mock(
            status_quo=Observation(
                data=sq_obs_data[0],
                features=ObservationFeatures(
                    parameters=none_throws(experiment.status_quo).parameters
                ),
                arm_name="status_quo",
            ),
            status_quo_data_by_trial={
                i: sq_obs_data[i] for i in range(len(sq_obs_data))
            },
        )

        # not checking RelativizeWithConstantControl here
        # because relativize_data uses delta method
        transform = Relativize(search_space=None, adapter=adapter)

        relative_obs_t = transform.transform_observations(observations)
        self.maxDiff = None
        # this assertion just checks that order is the same, which
        # is only important for the purposes of this test
        self.assertEqual(
            [datum.data.metric_signatures for datum in relative_obs_t],
            [datum.data.metric_signatures for datum in relative_observations],
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

    def test_transform_experiment_data(self) -> None:
        experiment = get_branin_with_multi_task()
        experiment.fetch_data()
        experiment_data = extract_experiment_data(
            experiment=experiment, data_loader_config=DataLoaderConfig()
        )
        adapter = Adapter(experiment=experiment, generator=Generator())
        for relativize_cls in [Relativize, RelativizeWithConstantControl]:
            t = relativize_cls(search_space=experiment.search_space, adapter=adapter)
            relativized_data = t.transform_experiment_data(
                experiment_data=deepcopy(experiment_data)
            )
            self.assertNotEqual(experiment_data, relativized_data)
            # Check that arm data hasn't changed.
            assert_frame_equal(relativized_data.arm_data, experiment_data.arm_data)
            # Check that observation data was relativized correctly.
            expected_mean, expected_sem = [], []
            for (  # pyre-ignore [23]: Pyre doesn't know about the index structure.
                trial_index,
                arm_name,
            ), row in experiment_data.observation_data.iterrows():
                sq_row = experiment_data.observation_data.loc[
                    (trial_index, "status_quo")
                ]
                if arm_name == "status_quo":
                    mean, sem = 0, 0
                else:
                    mean, sem = relativize(
                        means_t=row["mean", "branin"],
                        sems_t=row["sem", "branin"],
                        mean_c=sq_row["mean", "branin"],
                        sem_c=sq_row["sem", "branin"],
                        as_percent=True,
                        control_as_constant=t.control_as_constant,
                    )
                expected_mean.append(mean)
                expected_sem.append(sem)
            self.assertEqual(
                relativized_data.observation_data[("mean", "branin")].tolist(),
                expected_mean,
            )
            self.assertEqual(
                relativized_data.observation_data[("sem", "branin")].tolist(),
                expected_sem,
            )


class RelativizeDataOptConfigTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        search_space = get_search_space()
        gr = Generators.SOBOL(experiment=Experiment(search_space=search_space)).gen(n=1)
        self.model = Mock(
            search_space=search_space,
            status_quo=Mock(
                features=ObservationFeatures(parameters=gr.arms[0].parameters)
            ),
        )
        self.model.status_quo_data_by_trial = {0: None}

    def test_transform_optimization_config_without_constraints(self) -> None:
        for relativize_cls, pruning_target in product(
            [Relativize, RelativizeWithConstantControl],
            (None, Arm(parameters={"x0": 0.0, "x1": 0.0})),
        ):
            relativize = relativize_cls(search_space=None, adapter=self.model)
            optimization_config = get_branin_optimization_config()
            optimization_config.pruning_target_parameterization = pruning_target
            new_config = relativize.transform_optimization_config(
                optimization_config=optimization_config,
                adapter=None,
                fixed_features=Mock(),
            )
            self.assertEqual(new_config, optimization_config)

    def test_transform_optimization_config_with_relative_constraints(self) -> None:
        for relativize_cls, pruning_target in product(
            [Relativize, RelativizeWithConstantControl],
            (None, Arm(parameters={"x0": 0.0, "x1": 0.0})),
        ):
            relativize = relativize_cls(search_space=None, adapter=self.model)
            optimization_config = get_branin_optimization_config()
            optimization_config.pruning_target_parameterization = pruning_target
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
                adapter=None,
                fixed_features=Mock(),
            )
            self.assertEqual(new_config.objective, optimization_config.objective)
            self.assertEqual(
                new_config.pruning_target_parameterization,
                optimization_config.pruning_target_parameterization,
            )
            self.assertEqual(
                new_config.outcome_constraints[0].bound,
                optimization_config.outcome_constraints[0].bound,
            )
            self.assertFalse(new_config.outcome_constraints[0].relative)
            # Untransform the constraints
            cons = relativize.untransform_outcome_constraints(
                outcome_constraints=new_config.outcome_constraints,
                fixed_features=Mock(),
            )
            self.assertEqual(cons, optimization_config.outcome_constraints)

    def test_transform_optimization_config_with_non_relative_constraints(self) -> None:
        for relativize_cls in [Relativize, RelativizeWithConstantControl]:
            relativize = relativize_cls(search_space=None, adapter=self.model)
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
                    adapter=None,
                    fixed_features=Mock(),
                )

    def test_transform_optimization_config_with_relative_thresholds(self) -> None:
        for relativize_cls in [Relativize, RelativizeWithConstantControl]:
            relativize = relativize_cls(
                search_space=None,
                adapter=self.model,
            )
            optimization_config = get_branin_multi_objective_optimization_config(
                has_objective_thresholds=True,
            )
            for threshold in optimization_config.objective_thresholds:
                threshold.relative = True

            new_config = relativize.transform_optimization_config(
                optimization_config=optimization_config,
                adapter=None,
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
        for relativize_cls in [Relativize, RelativizeWithConstantControl]:
            relativize = relativize_cls(
                search_space=None,
                adapter=self.model,
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
                    adapter=None,
                    fixed_features=Mock(),
                )
