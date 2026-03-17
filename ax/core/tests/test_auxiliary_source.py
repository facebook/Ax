# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import unittest

import pandas as pd
from ax.adapter.base import DataLoaderConfig
from ax.adapter.data_utils import extract_experiment_data
from ax.core.auxiliary_source import _check_parameter_compatibility, AuxiliarySource
from ax.core.data import Data
from ax.core.metric import Metric
from ax.core.observation import Observation, ObservationFeatures
from ax.core.observation_utils import observations_from_data
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.search_space import SearchSpace
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.utils.common.constants import Keys
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_search_space,
    get_hierarchical_search_space_experiment,
    get_test_map_data_experiment,
)
from pyre_extensions import assert_is_instance, none_throws


class ParameterCompatibilityTest(unittest.TestCase):
    def test_mismatch_type(self) -> None:
        p1 = FixedParameter(name="p1", parameter_type=ParameterType.STRING, value="a")
        p2 = ChoiceParameter(
            name="p2", parameter_type=ParameterType.STRING, values=["1", "2"]
        )
        p3 = ChoiceParameter(name="p3", parameter_type=ParameterType.INT, values=[1, 2])
        with self.assertRaisesRegex(
            ValueError,
            r"p2 \(<class 'ax.core.parameter.ChoiceParameter'>\) is not compatible "
            r"with p1 \(<class 'ax.core.parameter.FixedParameter'>\).",
        ):
            _check_parameter_compatibility(p1, p2)
        with self.assertRaisesRegex(ValueError, r"STRING"):
            _check_parameter_compatibility(p2, p3)

    def test_range_params(self) -> None:
        p1 = RangeParameter(
            name="p1", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
        )
        p2 = RangeParameter(
            name="p2", parameter_type=ParameterType.FLOAT, lower=-1.0, upper=2.0
        )
        p3 = RangeParameter(
            name="p3", parameter_type=ParameterType.INT, lower=2.0, upper=5.0
        )
        p4 = RangeParameter(
            name="p4", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
        )
        with self.assertRaisesRegex(ValueError, r"FLOAT"):
            _check_parameter_compatibility(p1, p3, strict=False)
        with self.assertRaisesRegex(ValueError, r"Range mismatch"):
            _check_parameter_compatibility(p1, p2, strict=True)
        _check_parameter_compatibility(p1, p2, strict=False)
        _check_parameter_compatibility(p1, p4, strict=True)

    def test_choice_params(self) -> None:
        p1 = ChoiceParameter(
            name="p1", parameter_type=ParameterType.STRING, values=["a", "b", "c"]
        )
        p2 = ChoiceParameter(
            name="p2", parameter_type=ParameterType.STRING, values=["a", "b"]
        )
        p3 = ChoiceParameter(
            name="p3", parameter_type=ParameterType.STRING, values=["a", "b", "c"]
        )
        _check_parameter_compatibility(p2, p1, strict=False)
        _check_parameter_compatibility(p1, p2, strict=False)
        with self.assertRaisesRegex(ValueError, r"Values mismatch"):
            _check_parameter_compatibility(p2, p1, strict=True)
        _check_parameter_compatibility(p1, p3, strict=True)

    def test_choice_params_numerical(self) -> None:
        p1_int = ChoiceParameter(
            name="p1_int", parameter_type=ParameterType.INT, values=[1, 2, 3]
        )
        p2_int = ChoiceParameter(
            name="p2_int", parameter_type=ParameterType.INT, values=[1, 2]
        )
        p3_int = ChoiceParameter(
            name="p3_int", parameter_type=ParameterType.INT, values=[4, 5, 6]
        )
        p1_float = ChoiceParameter(
            name="p1_float",
            parameter_type=ParameterType.FLOAT,
            values=[1.0, 2.0, 3.0],
        )
        p2_float = ChoiceParameter(
            name="p2_float", parameter_type=ParameterType.FLOAT, values=[4.0, 5.0]
        )

        # For numerical ChoiceParameters in non-strict mode, should pass
        # regardless of values
        _check_parameter_compatibility(p1_int, p2_int, strict=False)
        _check_parameter_compatibility(p2_int, p1_int, strict=False)
        _check_parameter_compatibility(p1_int, p3_int, strict=False)
        _check_parameter_compatibility(p1_float, p2_float, strict=False)

        # In strict mode, should still require exact value matching
        _check_parameter_compatibility(p1_int, p1_int, strict=True)
        with self.assertRaisesRegex(ValueError, r"Values mismatch"):
            _check_parameter_compatibility(p1_int, p2_int, strict=True)
        with self.assertRaisesRegex(ValueError, r"Values mismatch"):
            _check_parameter_compatibility(p1_int, p3_int, strict=True)

    def test_fixed_params(self) -> None:
        p1 = FixedParameter(name="p1", parameter_type=ParameterType.STRING, value="a")
        p2 = FixedParameter(name="p2", parameter_type=ParameterType.STRING, value="b")
        p3 = FixedParameter(name="p3", parameter_type=ParameterType.STRING, value="a")
        _check_parameter_compatibility(p1, p3)
        with self.assertRaisesRegex(ValueError, r"Value mismatch"):
            _check_parameter_compatibility(p1, p2, strict=False)


class AuxiliarySourceTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        base_params = list(get_branin_search_space().parameters.values())
        fp1 = FixedParameter(name="fp1", parameter_type=ParameterType.STRING, value="a")
        fp2 = FixedParameter(name="fp2", parameter_type=ParameterType.STRING, value="b")
        x3 = RangeParameter(
            name="x3", parameter_type=ParameterType.FLOAT, lower=0, upper=1
        )
        rp1 = RangeParameter(
            name="rp1", parameter_type=ParameterType.FLOAT, lower=-10, upper=20
        )
        self.target_ss = SearchSpace(base_params + [x3, fp1])
        source_ss = SearchSpace(base_params + [rp1, fp2])
        source_ss2 = SearchSpace(base_params + [x3, fp1, rp1])
        source_ss3 = SearchSpace(base_params)
        transfer_param_config = {"x3": "rp1"}

        source_exp1 = get_branin_experiment(
            with_completed_trial=True, search_space=self.target_ss.clone()
        )
        source_exp2 = get_branin_experiment(
            with_completed_trial=True, search_space=source_ss
        )
        source_exp3 = get_branin_experiment(
            with_completed_trial=True, search_space=source_ss2
        )
        source_exp4 = get_branin_experiment(
            with_completed_trial=True, search_space=source_ss3
        )
        # Data for trial 0, arm "0_0", metric_name "branin"
        base_data = source_exp2.fetch_data()

        # We already have one trial; add another with the same arm
        arm = source_exp2.trials[0].arms[0]
        source_exp2.attach_trial(
            parameterizations=[arm.parameters], arm_names=[arm.name]
        )
        new_data = Data(
            df=pd.DataFrame(
                {
                    "arm_name": [arm.name] * 2,
                    "metric_name": ["new_metric", "branin"],
                    "mean": [5.0, 4.0],
                    "sem": [6.0, 1.0],
                    "trial_index": [0, 1],
                    "metric_signature": ["new_metric", "branin"],
                }
            )
        )
        all_data = Data.from_multiple_data([base_data, new_data])
        source_exp2.attach_data(all_data)
        source_exp2.add_tracking_metric(Metric(name="new_metric"))
        transfer_metric_config = {
            "newtarget": {"branin", "new_metric"},
            "broken": {"asefase"},
        }

        self.auxsrc1 = AuxiliarySource(
            experiment=source_exp1, update_fixed_params=False
        )
        self.auxsrc2 = AuxiliarySource(
            experiment=source_exp2,
            transfer_param_config=transfer_param_config,
            transfer_metric_config=transfer_metric_config,
        )
        self.auxsrc3 = AuxiliarySource(
            experiment=source_exp2,
            transfer_param_config=transfer_param_config,
            update_fixed_params=False,
        )
        self.auxsrc4 = AuxiliarySource(
            experiment=source_exp3,
        )
        self.auxsrc5 = AuxiliarySource(
            experiment=source_exp2,
            transfer_param_config=transfer_param_config,
            transfer_metric_config=transfer_metric_config,
            metric_names=["branin"],
            trial_indices=[0],
        )
        self.auxsrc6 = AuxiliarySource(
            experiment=source_exp4,
        )

    def test_init(self) -> None:
        self.assertEqual(len(self.auxsrc1.data.df), 1)
        # metrics "new_metric" and "branin" for trial 0 and "branin" for trial 1
        self.assertEqual(len(self.auxsrc2.data.df), 3)

    def test_search_space_compat(self) -> None:
        # Full SS match
        self.auxsrc1.check_search_space_compatibility(
            target_search_space=self.target_ss
        )
        # Non-matching FPs, and not updating
        with self.assertRaisesRegex(
            UserInputError, r"Parameter fp2 is not in target search space."
        ):
            self.auxsrc3.check_search_space_compatibility(
                target_search_space=self.target_ss
            )
        # Non-matching FPs, but FPs updated
        self.auxsrc2.check_search_space_compatibility(
            target_search_space=self.target_ss
        )
        # Missing range param in source ss
        with self.assertRaisesRegex(
            UserInputError, r"Source experiment is missing parameter x3."
        ):
            self.auxsrc6.check_search_space_compatibility(
                target_search_space=self.target_ss
            )
        # With filled params
        self.auxsrc6.check_search_space_compatibility(
            target_search_space=self.target_ss, filled_params=["x3"]
        )

        # Extra params in source ss
        with self.assertRaisesRegex(UserInputError, r"rp1 is not in"):
            self.auxsrc4.check_search_space_compatibility(
                target_search_space=self.target_ss
            )

    def test_map_observations_and_experiment_data(self) -> None:
        expanded_target_ss = self.target_ss.clone()
        expanded_target_ss.add_parameter(
            RangeParameter(
                name="x4", parameter_type=ParameterType.FLOAT, lower=0, upper=1
            )
        )
        for target_ss in (self.target_ss, expanded_target_ss):
            observations = observations_from_data(
                experiment=self.auxsrc1.experiment, data=self.auxsrc1.data
            )
            new_obs = self.auxsrc1.map_observations(
                observations=observations, target_search_space=target_ss
            )
            self.assertEqual(len(new_obs), 1)
            self.assertEqual(
                set(new_obs[0].features.parameters.keys()), {"x1", "x2", "x3", "fp1"}
            )
            # Repeat with experiment_data and compare the output.
            experiment_data = extract_experiment_data(
                experiment=self.auxsrc1.experiment,
                data_loader_config=DataLoaderConfig(),
            )
            new_experiment_data = self.auxsrc1.map_experiment_data(
                experiment_data=experiment_data, target_search_space=target_ss
            )
            self.assertEqual(
                new_obs, new_experiment_data.convert_to_list_of_observations()
            )

            observations = observations_from_data(
                experiment=self.auxsrc2.experiment, data=self.auxsrc2.data
            )
            # Add OOD observation that should be dropped
            observations.append(
                Observation(
                    features=ObservationFeatures(
                        {"x1": 0.0, "x2": 0.0, "rp1": 0.0, "fp2": "wrong"}
                    ),
                    data=None,  # pyre-ignore [6]
                    arm_name="ood",
                )
            )
            rval = float(observations[0].features.parameters["rp1"])
            self.assertEqual(
                set(observations[0].features.parameters.keys()),
                {"x1", "x2", "rp1", "fp2"},
            )
            new_obs = self.auxsrc2.map_observations(
                observations=observations, target_search_space=target_ss
            )
            # There are 2 in-design observations and one out-of-design observation
            # Keep the 2 in-design observations (which have the same arm)
            self.assertEqual(len(new_obs), 2)
            for ob in new_obs:
                self.assertEqual(
                    set(ob.features.parameters.keys()), {"x1", "x2", "x3", "fp1"}
                )
                self.assertEqual(ob.features.parameters["x3"], rval)
                self.assertEqual(ob.features.parameters["fp1"], "a")
            # Repeat with experiment_data and compare the output.
            experiment_data = extract_experiment_data(
                experiment=self.auxsrc2.experiment,
                data_loader_config=DataLoaderConfig(),
            )
            new_experiment_data = self.auxsrc2.map_experiment_data(
                experiment_data=experiment_data, target_search_space=target_ss
            )
            self.assertEqual(
                new_obs, new_experiment_data.convert_to_list_of_observations()
            )

            # Check error if all OOD.
            with self.assertRaisesRegex(UserInputError, "No observations were mapped"):
                self.auxsrc2.map_observations(
                    observations=observations[-1:], target_search_space=target_ss
                )

    def test_map_observations_and_experiment_data_hierarchical(self) -> None:
        # Construct a hierarchical experiment with some data.
        experiment = get_hierarchical_search_space_experiment(num_observations=3)
        with self.assertRaisesRegex(UnsupportedError, "hierarchical search spaces"):
            AuxiliarySource(experiment=experiment)
        aux_src = AuxiliarySource(
            experiment=experiment,
            transfer_param_config={"lr": "learning_rate"},
            update_fixed_params=False,
        )
        # Construct the target search space by renaming learning_rate -> lr.
        target_ss = experiment.search_space.clone()
        none_throws(
            assert_is_instance(
                target_ss._parameters["model"], ChoiceParameter
            )._dependents
        )["Linear"][0] = "lr"
        new_param = target_ss._parameters.pop("learning_rate")
        new_param._name = "lr"
        target_ss._parameters["lr"] = new_param
        # Check that the trials have full parameterization with learning_rate.
        observations = observations_from_data(
            experiment=aux_src.experiment, data=aux_src.data
        )
        for obs in observations:
            self.assertIn(
                "learning_rate",
                none_throws(obs.features.metadata)[Keys.FULL_PARAMETERIZATION],
            )
        # Map the observations to the target search space.
        new_obs = aux_src.map_observations(
            observations=observations, target_search_space=target_ss
        )
        # Check that the trials have full parameterization with lr.
        for obs in new_obs:
            self.assertIn(
                "lr", none_throws(obs.features.metadata)[Keys.FULL_PARAMETERIZATION]
            )
            self.assertNotIn(
                "learning_rate",
                none_throws(obs.features.metadata)[Keys.FULL_PARAMETERIZATION],
            )

        # Repeat with experiment_data and compare the output.
        experiment_data = extract_experiment_data(
            experiment=aux_src.experiment, data_loader_config=DataLoaderConfig()
        )
        new_experiment_data = aux_src.map_experiment_data(
            experiment_data=experiment_data, target_search_space=target_ss
        )
        self.assertEqual(new_obs, new_experiment_data.convert_to_list_of_observations())

    def test_transfer_metrics(self) -> None:
        self.assertEqual(
            self.auxsrc2.get_metrics_to_transfer_from("branin"), ["branin"]
        )
        self.assertEqual(
            set(self.auxsrc2.get_metrics_to_transfer_from("newtarget")),
            {"branin", "new_metric"},
        )
        with self.assertRaisesRegex(ValueError, r"notthere"):
            self.auxsrc2.get_metrics_to_transfer_from("notthere")
        with self.assertRaisesRegex(ValueError, r"asefase"):
            self.auxsrc2.get_metrics_to_transfer_from("broken")

    def test_transfer_data(self) -> None:
        data = none_throws(
            self.auxsrc2.get_data_to_transfer_from(target_metric="branin")
        )
        # There are 2 "branin" observations and one "new_metric" observation
        # Transfer from "branin"
        self.assertEqual(len(data.df), 2)
        self.assertEqual(data.df["metric_name"].unique(), ["branin"])

        # Transfer from both "branin" and "new_metric"
        data = none_throws(
            self.auxsrc2.get_data_to_transfer_from(target_metric="newtarget")
        )
        self.assertEqual(len(data.df), 3)
        self.assertEqual(
            list(data.df["metric_name"].unique()), ["branin", "new_metric"]
        )

        self.auxsrc2.transfer_metric_config["non-mapped-target"] = set()
        self.assertIsNone(
            self.auxsrc2.get_data_to_transfer_from(target_metric="non-mapped-target")
        )

    def test_validate_offline_metrics_in_data_missing_metrics(self) -> None:
        experiment = get_test_map_data_experiment(
            num_trials=5, num_fetches=3, num_complete=4
        )
        transfer_metric_config = {
            "target_metric_1": {"missing_auxiliary_metric_1"},
            "target_metric_2": {"branin", "missing_auxiliary_metric_2"},
        }

        with self.assertLogs(
            "ax.core.auxiliary_source",
            level=logging.WARNING,
        ) as cm:
            _ = AuxiliarySource(
                experiment=experiment, transfer_metric_config=transfer_metric_config
            )
        self.assertIn("Metrics not in data", cm.output[0])
        self.assertIn("missing_auxiliary_metric_1", cm.output[0])
        self.assertIn("missing_auxiliary_metric_2", cm.output[0])

    def test_specifying_trial_indices(self) -> None:
        exp = get_branin_experiment(
            with_batch=True, search_space=self.target_ss.clone(), num_batch_trial=3
        )
        data = Data(
            df=pd.DataFrame(
                {
                    "arm_name": ["0_0", "0_0"],
                    "metric_name": ["new_metric", "branin"],
                    "mean": [5.0, 4.0],
                    "sem": [6.0, 1.0],
                    "trial_index": [0, 1],
                    "metric_signature": ["new_metric", "branin"],
                }
            )
        )
        exp.attach_data(data)
        aux_src = AuxiliarySource(experiment=exp, trial_indices=[1])
        self.assertEqual(aux_src.experiment.trials[0].arms, exp.trials[1].arms)
        self.assertEqual(len(aux_src.experiment.trials), 1)
        self.assertEqual(list(aux_src.data.df["trial_index"].unique()), [1])

    def test_trial_type_not_cloned(self) -> None:
        exp = get_branin_experiment(
            with_batch=True, search_space=self.target_ss.clone(), num_batch_trial=1
        )
        exp.trials[0]._trial_type = "foo"
        # this would raise an exception if trial_type is retained
        aux_src = AuxiliarySource(experiment=exp)
        self.assertIsNone(aux_src.experiment.trials[0].trial_type)

    def test_data_contains_only_specified_metrics(self) -> None:
        metrics = list(self.auxsrc5.data.df["metric_name"].unique())
        self.assertEqual(metrics, ["branin"])
        trial_indices = list(self.auxsrc5.data.df["trial_index"].unique())
        self.assertEqual(trial_indices, [0])

    def test_experiment_name_required(self) -> None:
        exp = get_branin_experiment(
            with_batch=True, search_space=self.target_ss.clone(), num_batch_trial=1
        )
        exp._name = None
        with self.assertRaisesRegex(ValueError, "Experiment's name is None."):
            AuxiliarySource(experiment=exp)
