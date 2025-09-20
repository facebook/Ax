#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy
from math import isnan, nan
from typing import cast

import numpy as np
from ax.adapter import Adapter
from ax.adapter.base import DataLoaderConfig
from ax.adapter.cross_validation import cross_validate
from ax.adapter.data_utils import extract_experiment_data
from ax.adapter.registry import Generators, MBM_X_trans, Y_trans
from ax.adapter.torch import TorchAdapter
from ax.adapter.transforms.map_key_to_float import MapKeyToFloat
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from ax.api.utils.generation_strategy_dispatch import _get_sobol_node
from ax.core.map_data import MAP_KEY
from ax.core.observation import (
    Observation,
    ObservationData,
    ObservationFeatures,
    observations_from_data,
)
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.trial import Trial
from ax.core.trial_status import TrialStatus
from ax.early_stopping.strategies import PercentileEarlyStoppingStrategy
from ax.generation_strategy.center_generation_node import CenterGenerationNode
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generators.base import Generator
from ax.generators.torch.botorch_modular.generator import BoTorchGenerator
from ax.generators.torch.botorch_modular.surrogate import ModelConfig, SurrogateSpec
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment_with_timestamp_map_metric,
    get_hierarchical_search_space_experiment,
)
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.gp_regression import SingleTaskGP
from pandas.testing import assert_frame_equal
from pyre_extensions import assert_is_instance, none_throws


class ClientTest(TestCase):
    def setUp(self) -> None:
        super().setUp()

        seed = 8888
        range_parameters = [
            RangeParameterConfig(name="width", parameter_type="float", bounds=(1, 20)),
            RangeParameterConfig(name="height", parameter_type="float", bounds=(1, 20)),
        ]

        self.initialization_budget = 10
        self.max_steps = 15
        self.metric_name = "test_loss"
        self.num_parameters = len(range_parameters)

        objective = f"-{self.metric_name}"

        surrogate_spec = SurrogateSpec(
            model_configs=[ModelConfig(botorch_model_class=SingleTaskGP)]
        )
        generator_spec = GeneratorSpec(
            generator_enum=Generators.BOTORCH_MODULAR,
            model_kwargs={
                "surrogate_spec": surrogate_spec,
                "botorch_acqf_class": qLogExpectedImprovement,
                "transforms": MBM_X_trans + Y_trans,
                "data_loader_config": DataLoaderConfig(
                    fit_only_completed_map_metrics=False,
                    latest_rows_per_group=1,
                ),
            },
        )

        generation_strategy = self._construct_generation_strategy(
            generator_spec=generator_spec,
            initialization_budget=self.initialization_budget,
            seed=seed,
        )
        early_stopping_strategy = PercentileEarlyStoppingStrategy(
            min_progression=3,
            min_curves=3,
        )

        self.client = Client()
        self.client.configure_experiment(parameters=range_parameters)
        self.client.configure_optimization(objective=objective)
        self.client.set_generation_strategy(generation_strategy=generation_strategy)
        self.client.set_early_stopping_strategy(
            early_stopping_strategy=early_stopping_strategy
        )

    @staticmethod
    def _construct_generation_strategy(
        generator_spec: GeneratorSpec,
        initialization_budget: int,
        seed: int,
    ) -> GenerationStrategy:
        """Constructs a Center + Sobol + Modular BoTorch `GenerationStrategy`
        using the provided `generator_spec` for the Modular BoTorch node.
        """
        sobol_node = _get_sobol_node(
            initialization_budget=initialization_budget,
            min_observed_initialization_trials=None,
            initialize_with_center=True,
            use_existing_trials_for_initialization=True,
            allow_exceeding_initialization_budget=False,
            initialization_random_seed=seed,
        )
        center_node = CenterGenerationNode(next_node_name=sobol_node.node_name)
        botorch_node = GenerationNode(
            node_name="MBM",
            generator_specs=[generator_spec],
            should_deduplicate=True,
        )
        return GenerationStrategy(
            name=f"Center+Sobol+{botorch_node.node_name}",
            nodes=[center_node, sobol_node, botorch_node],
        )

    @staticmethod
    def _loss_fn(step: int, width: float, height: float) -> float:
        return 100.0 / (10.0 + width * step) + 0.1 * height

    def _simulate(
        self,
        with_early_stopping: bool = True,
        attach_with_progression: bool = True,
        complete_with_progression: bool = True,
    ) -> None:
        """Simulate typical usage of Client API."""
        for _ in range(self.initialization_budget):
            (trial_data,) = self.client.get_next_trials(max_trials=1).items()
            trial_index, parameters = trial_data
            kwargs: dict[str, float] = {k: float(v) for k, v in parameters.items()}

            stopped: bool = False
            if with_early_stopping:
                for i in range(self.max_steps - 1):
                    step = i + 1
                    result = self._loss_fn(step=step, **kwargs)
                    self.client.attach_data(
                        trial_index=trial_index,
                        raw_data={self.metric_name: result},
                        progression=step if attach_with_progression else None,
                    )
                    if stopped := self.client.should_stop_trial_early(
                        trial_index=trial_index
                    ):
                        self.client.mark_trial_early_stopped(trial_index=trial_index)
                        break

            if not stopped:
                result = self._loss_fn(step=self.max_steps, **kwargs)
                self.client.complete_trial(
                    trial_index=trial_index,
                    raw_data={self.metric_name: result},
                    progression=self.max_steps if complete_with_progression else None,
                )

    def _test_no_early_stopping(self, with_progression: bool) -> None:
        self._simulate(
            with_early_stopping=False,
            attach_with_progression=with_progression,
            complete_with_progression=with_progression,
        )

        # ensure there are no early-stopped trials for the purposes of this test
        self.assertEqual(
            len(self.client._experiment.trials_by_status[TrialStatus.EARLY_STOPPED]),
            0,
        )

        self.client.get_next_trials(max_trials=1)

        adapter = assert_is_instance(
            self.client._generation_strategy.adapter, TorchAdapter
        )
        generator = assert_is_instance(adapter.generator, BoTorchGenerator)
        surrogate = generator.surrogate
        (dataset,) = surrogate.training_data

        # the transform behaves as a no-op (list of parameters to add to
        # search space is empty)
        self.assertListEqual(
            assert_is_instance(
                adapter.transforms["MapKeyToFloat"], MapKeyToFloat
            )._parameter_list,
            [],
        )

        # progression information is omitted from data propagated to the model
        self.assertListEqual(dataset.feature_names, ["width", "height"])

        # Check that cross validation works.
        cross_validate(model=adapter)

    def _test_early_stopping(self, complete_with_progression: bool) -> None:
        self._simulate(
            with_early_stopping=True,
            attach_with_progression=True,
            complete_with_progression=complete_with_progression,
        )

        # ensure there are early-stopped trials for the purposes of this test
        self.assertGreater(
            len(self.client._experiment.trials_by_status[TrialStatus.EARLY_STOPPED]),
            0,
        )

        (trial_index,) = self.client.get_next_trials(max_trials=1)

        adapter = assert_is_instance(
            self.client._generation_strategy.adapter, TorchAdapter
        )
        generator = assert_is_instance(adapter.generator, BoTorchGenerator)
        surrogate = generator.surrogate
        (dataset,) = surrogate.training_data

        # check that the data being fed includes all trials, including
        # early-stopped trials
        self.assertEqual(
            dataset.X.shape,
            (self.initialization_budget, self.num_parameters + 1),
        )

        # check that the data being fed to the model is properly
        # contextualized with progression information
        self.assertListEqual(dataset.feature_names, ["width", "height", "step"])

        trial = cast(Trial, self.client._experiment.trials[trial_index])
        generator_run = none_throws(trial.generator_run)
        candidate_metadata_by_arm_signature = none_throws(
            generator_run.candidate_metadata_by_arm_signature
        )
        signature = none_throws(trial.arm).signature
        candidate_metadata = none_throws(candidate_metadata_by_arm_signature[signature])

        # check that candidate is generated at the target progression
        self.assertEqual(int(candidate_metadata["step"]), self.max_steps)

        # Check that cross validation works.
        cross_validate(model=adapter)

    def test_no_early_stopping_with_progression(self) -> None:
        self._test_no_early_stopping(with_progression=True)

    def test_no_early_stopping_no_progression(self) -> None:
        self._test_no_early_stopping(with_progression=False)

    def test_early_stopping_with_final_progression(self) -> None:
        self._test_early_stopping(complete_with_progression=True)


class MapKeyToFloatTransformTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.experiment = get_branin_experiment_with_timestamp_map_metric(
            with_trials_and_data=True,
        )
        self.search_space = self.experiment.search_space
        self.adapter = Adapter(experiment=self.experiment, generator=Generator())
        self.observations = observations_from_data(
            experiment=self.experiment,
            data=self.experiment.fetch_data(),
            latest_rows_per_group=None,
        )
        self.map_key: str = MAP_KEY
        # Does not require explicitly specifying `config`.
        self.t = MapKeyToFloat(observations=self.observations, adapter=self.adapter)

        # Set up a hierarchical search space experiment
        self.hss_experiment = get_hierarchical_search_space_experiment(
            num_observations=10, use_map_data=True
        )
        self.hss_observations = observations_from_data(
            experiment=self.hss_experiment, data=self.hss_experiment.lookup_data()
        )
        self.hss_experiment_data = extract_experiment_data(
            experiment=self.hss_experiment,
            data_loader_config=DataLoaderConfig(),
        )
        self.hierarchical_search_space = self.hss_experiment.search_space

        self.hss_t = MapKeyToFloat(
            search_space=self.hierarchical_search_space,
            experiment_data=self.hss_experiment_data,
            # We didn't construct an adapter. So we pass `config` manually.
            config={"parameters": {self.map_key: {}}},
        )

    def test_Init(self) -> None:
        # Check for error if adapter & parameters are not provided.
        with self.assertWarnsRegex(Warning, "optimization config"):
            MapKeyToFloat(observations=self.observations)

        experiment_data = extract_experiment_data(
            experiment=self.experiment,
            data_loader_config=DataLoaderConfig(fit_only_completed_map_metrics=False),
        )
        t2 = MapKeyToFloat(experiment_data=experiment_data, adapter=self.adapter)
        for t in (self.t, t2):
            # Check for default initialization
            self.assertEqual(len(t._parameter_list), 1)
            (p,) = t._parameter_list
            self.assertEqual(p.name, self.map_key)
            self.assertEqual(p.parameter_type, ParameterType.FLOAT)
            self.assertEqual(p.lower, 0.0)
            self.assertEqual(p.upper, 4.0)
            self.assertFalse(p.log_scale)

        # specifying a parameter name that is not in the observation features' metadata
        with self.assertRaisesRegex(KeyError, "'baz'"):
            MapKeyToFloat(
                observations=self.observations,
                config={"parameters": {"baz": {}}},
            )

        # test that one is able to override default config
        with self.subTest(msg="override default config"):
            t = MapKeyToFloat(
                observations=self.observations,
                config={"parameters": {self.map_key: {"lower": 0.1}}},
            )
            self.assertDictEqual(t.parameters, {self.map_key: {"lower": 0.1}})
            self.assertEqual(len(t._parameter_list), 1)

            p = t._parameter_list[0]

            self.assertEqual(p.name, self.map_key)
            self.assertEqual(p.parameter_type, ParameterType.FLOAT)
            self.assertEqual(p.lower, 0.1)
            self.assertEqual(p.upper, 4.0)
            self.assertFalse(p.log_scale)

    def test_TransformSearchSpace(self) -> None:
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)

        self.assertSetEqual(set(ss2.parameters), {"x1", "x2", self.map_key})

        p = assert_is_instance(ss2.parameters[self.map_key], RangeParameter)
        self.assertEqual(p.name, self.map_key)
        self.assertEqual(p.parameter_type, ParameterType.FLOAT)
        self.assertEqual(p.lower, 0.0)
        self.assertEqual(p.upper, 4.0)
        self.assertFalse(p.log_scale)

    def test_TransformObservationFeatures(self) -> None:
        # NaN progressions get filled in with the upper bound,
        # and then upon untransforming, the value remains non-NaN and won't
        # match its original value.
        with self.subTest("Non-NaN observation features"):
            keep_indices = [
                i
                for i, obs in enumerate(self.observations)
                if not isnan(obs.features.metadata["step"])
            ]
            observation_features = [self.observations[i].features for i in keep_indices]
            obs_ft2 = deepcopy(observation_features)
            obs_ft2 = self.t.transform_observation_features(obs_ft2)

            expected = []
            for i in keep_indices:
                obs = self.observations[i]
                obsf = obs.features.clone()
                obsf.parameters[self.map_key] = obsf.metadata.pop(self.map_key)
                expected.append(obsf)

            self.assertEqual(obs_ft2, expected)
            obs_ft2 = self.t.untransform_observation_features(obs_ft2)
            self.assertEqual(obs_ft2, observation_features)

        with self.subTest("NaN observation features"):
            keep_indices = [
                i
                for i, obs in enumerate(self.observations)
                if isnan(obs.features.metadata["step"])
            ]
            observation_features = [self.observations[i].features for i in keep_indices]
            obs_ft2 = deepcopy(observation_features)
            obs_ft2 = self.t.transform_observation_features(obs_ft2)

            # upper bound
            expected = []
            for i in keep_indices:
                obs = self.observations[i]
                obsf = obs.features.clone()
                obsf.parameters[self.map_key] = 4.0
                obsf.metadata = {}
                expected.append(obsf)

            self.assertEqual(obs_ft2, expected)
            untransformed = self.t.untransform_observation_features(obs_ft2)
            expected = observation_features
            for obs in expected:
                obs.metadata["step"] = 4

            self.assertEqual(untransformed, observation_features)

    def test_mixed_nan_progression(self) -> None:
        observations = [
            Observation(
                data=ObservationData(
                    metric_names=["metric1", "metric2"],
                    means=np.array([1.0, 2.0]),
                    covariance=np.diag([1.0, 2.0]),
                ),
                features=ObservationFeatures(
                    parameters={"x1": 5.0, "x2": 2.0},
                    metadata={self.map_key: 10.0},
                ),
            ),
            Observation(
                data=ObservationData(
                    metric_names=["metric3"],
                    means=np.array([30.0]),
                    covariance=np.array([[30.0]]),
                ),
                features=ObservationFeatures(
                    parameters={"x1": 5.0, "x2": 2.0},
                    metadata={self.map_key: np.nan},
                ),
            ),
            Observation(
                data=ObservationData(
                    metric_names=["x1", "x2"],
                    means=np.array([1.5, 2.5]),
                    covariance=np.diag([1.0, 2.0]),
                ),
                features=ObservationFeatures(
                    parameters={"x1": 5.0, "x2": 2.0},
                    metadata={self.map_key: 20.0},
                ),
            ),
        ]
        observation_features = [obs.features for obs in observations]

        t = MapKeyToFloat(observations=observations, adapter=self.adapter)

        self.assertEqual(len(t._parameter_list), 1)
        (p,) = t._parameter_list
        self.assertEqual(p.name, self.map_key)
        self.assertEqual(p.parameter_type, ParameterType.FLOAT)
        self.assertEqual(p.lower, 10.0)
        self.assertEqual(p.upper, 20.0)

        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = t.transform_observation_features(obs_ft2)

        self.assertEqual(
            obs_ft2,
            [
                ObservationFeatures(
                    parameters={"x1": 5.0, "x2": 2.0, self.map_key: 10.0},
                    metadata={},
                ),
                ObservationFeatures(
                    parameters={"x1": 5.0, "x2": 2.0, self.map_key: 20.0},
                    metadata={},
                ),
                ObservationFeatures(
                    parameters={"x1": 5.0, "x2": 2.0, self.map_key: 20.0},
                    metadata={},
                ),
            ],
        )

    def test_TransformObservationFeaturesKeyNotInMetadata(self) -> None:
        observation_features = [obs.features for obs in self.observations]
        obs_ft2 = deepcopy(observation_features)
        # remove the key from metadata dicts
        for obsf in obs_ft2:
            obsf.metadata.pop(self.map_key)
            # To avoid this being treated as empty metadata.
            # In typical experiment, trial completion timestamp would be here.
            obsf.metadata["dummy"] = 1.0
        # should be exactly one parameter
        (p,) = self.t._parameter_list
        # Transform fills missing values with the upper bound.
        tf_obs_ft = self.t.transform_observation_features(obs_ft2)
        for obs in tf_obs_ft:
            self.assertEqual(obs.parameters[p.name], p.upper)
            self.assertEqual(obs.metadata, {"dummy": 1.0})

    def test_constant_progression(self) -> None:
        for constant in (23, np.nan):
            with self.subTest(msg=f"{constant=}"):
                observations = deepcopy(self.observations)
                for obs in observations:
                    obs.features.metadata[self.map_key] = constant
                observation_features = [obs.features for obs in observations]
                t = MapKeyToFloat(observations=observations, adapter=self.adapter)

                # forward and reverse transforms are identity/no-ops
                obs_ft2 = deepcopy(observation_features)
                obs_ft2 = t.transform_observation_features(obs_ft2)
                self.assertEqual(obs_ft2, observation_features)
                obs_ft2 = t.untransform_observation_features(obs_ft2)
                self.assertEqual(obs_ft2, observation_features)

    def test_TransformObservationFeaturesWithEmptyMetadata(self) -> None:
        # undefined metadata
        obsf = ObservationFeatures(
            trial_index=42,
            parameters={"x1": 1.0, "x2": 2.0},
            metadata=None,
        )
        self.t.transform_observation_features([obsf])
        self.assertEqual(
            obsf,
            ObservationFeatures(
                trial_index=42,
                parameters={"x1": 1.0, "x2": 2.0, self.map_key: 4.0},
                metadata={},
            ),
        )
        # empty metadata
        obsf = ObservationFeatures(
            trial_index=42,
            parameters={"x1": 1.0, "x2": 2.0},
            metadata={},
        )
        self.t.transform_observation_features([obsf])
        self.assertEqual(
            obsf,
            ObservationFeatures(
                trial_index=42,
                parameters={"x1": 1.0, "x2": 2.0, self.map_key: 4.0},
                metadata={},
            ),
        )

    def test_TransformObservationFeaturesWithEmptyParameters(self) -> None:
        obsf = ObservationFeatures(parameters={})
        self.t.transform_observation_features([obsf])

        p = self.t._parameter_list[0]
        self.assertEqual(
            obsf,
            ObservationFeatures(parameters={self.map_key: p.upper}),
        )

    def test_with_different_map_key(self) -> None:
        observations = [
            Observation(
                features=ObservationFeatures(
                    trial_index=0,
                    parameters={"x1": width, "x2": height},
                    metadata={"map_key": timestamp},
                ),
                data=ObservationData(
                    metric_names=[], means=np.array([]), covariance=np.empty((0, 0))
                ),
            )
            for width, height, timestamp in (
                (0.0, 1.0, 12345.0),
                (0.1, 0.9, 12346.0),
            )
        ]
        t = MapKeyToFloat(
            observations=observations,
            config={"parameters": {"map_key": {"log_scale": False}}},
        )
        self.assertEqual(t.parameters, {"map_key": {"log_scale": False}})
        self.assertEqual(len(t._parameter_list), 1)
        tf_obs_ft = t.transform_observation_features(
            [obs.features for obs in observations]
        )
        self.assertEqual(
            tf_obs_ft[0].parameters, {"x1": 0.0, "x2": 1.0, "map_key": 12345.0}
        )
        self.assertEqual(
            tf_obs_ft[1].parameters, {"x1": 0.1, "x2": 0.9, "map_key": 12346.0}
        )

    def test_transform_experiment_data(self) -> None:
        # First, set up a case with no NaNs.
        data = self.experiment.lookup_data()
        data.map_df["step"] = data.map_df["step"].fillna(0)
        experiment_data = extract_experiment_data(
            experiment=self.experiment,
            data=data,
            data_loader_config=DataLoaderConfig(fit_only_completed_map_metrics=False),
        )
        transformed_data = self.t.transform_experiment_data(
            experiment_data=deepcopy(experiment_data)
        )
        # Check that arm_data is not modified.
        assert_frame_equal(experiment_data.arm_data, transformed_data.arm_data)

        # Since there are no NaNs, observation data is also not modified.
        assert_frame_equal(
            experiment_data.observation_data, transformed_data.observation_data
        )
        # Test with NaNs.
        experiment_data = extract_experiment_data(
            experiment=self.experiment,
            data_loader_config=DataLoaderConfig(fit_only_completed_map_metrics=False),
        )
        # The timestamp is always NaN for 'branin'
        actual = experiment_data.observation_data.index.get_level_values(
            "step"
        ).to_numpy()
        expected_timestamp = np.array([nan, 4.0, nan, 2.0, nan, 0.0])
        self.assertTrue(np.array_equal(actual, expected_timestamp, equal_nan=True))
        # Transform and check that the index is filled with the upper bound.
        transformed_data = self.t.transform_experiment_data(
            experiment_data=deepcopy(experiment_data)
        )
        transformed_timestamp = (
            transformed_data.observation_data.index.get_level_values("step").tolist()
        )
        upper = self.t._parameter_list[0].upper
        expected_trans_timestamp = [upper, 4.0, upper, 2.0, upper, 0.0]
        self.assertEqual(transformed_timestamp, expected_trans_timestamp)
