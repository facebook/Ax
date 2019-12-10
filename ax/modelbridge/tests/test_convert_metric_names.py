#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import copy

from ax.core.observation import observations_from_data
from ax.modelbridge.transforms.convert_metric_names import (
    ConvertMetricNames,
    convert_mt_observations,
    tconfig_from_mt_experiment,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_multi_type_experiment


class ConvertMetricNamesTest(TestCase):
    def setUp(self):
        self.experiment = get_multi_type_experiment(add_trials=True)
        self.data = self.experiment.fetch_data()
        self.observations = observations_from_data(self.experiment, self.data)
        self.observation_data = [o.data for o in self.observations]
        self.observation_features = [o.features for o in self.observations]
        self.tconfig = tconfig_from_mt_experiment(self.experiment)

    def testConvertMetricNames(self):
        transform = ConvertMetricNames(
            None, self.observation_features, self.observation_data, config=self.tconfig
        )

        transformed_observations = convert_mt_observations(
            self.observations, self.experiment
        )
        transformed_observation_data = [o.data for o in transformed_observations]
        transformed_observation_features = [
            o.features for o in transformed_observations
        ]

        # All trials should have canonical name "m1"
        for obsd in transformed_observation_data:
            self.assertEqual(obsd.metric_names[0], "m1")

        # By default untransform does nothing
        untransformed_observation_data = transform.untransform_observation_data(
            transformed_observation_data, transformed_observation_features
        )
        self.assertEqual(transformed_observation_data, untransformed_observation_data)

        transform.perform_untransform = True
        untransformed_observation_data = transform.untransform_observation_data(
            transformed_observation_data, transformed_observation_features
        )

        # Should have original metric_name
        for i in range(len(self.observations)):
            metric_name = (
                "m1" if self.observation_features[i].trial_index == 0 else "m2"
            )
            self.assertEqual(
                untransformed_observation_data[i].metric_names[0], metric_name
            )

    def testBadInputs(self):
        with self.assertRaises(ValueError):
            ConvertMetricNames(
                None, self.observation_features, self.observation_data, config=None
            )

        with self.assertRaises(ValueError):
            tconfig_copy = dict(self.tconfig)
            tconfig_copy.pop("metric_name_map")
            ConvertMetricNames(
                None,
                self.observation_features,
                self.observation_data,
                config=tconfig_copy,
            )

        with self.assertRaises(ValueError):
            tconfig_copy = dict(self.tconfig)
            tconfig_copy.pop("trial_index_to_type")
            ConvertMetricNames(
                None,
                self.observation_features,
                self.observation_data,
                config=tconfig_copy,
            )

        with self.assertRaises(ValueError):
            tconfig_copy = dict(self.tconfig)
            tconfig_copy.pop("metric_name_to_trial_type")
            ConvertMetricNames(
                None,
                self.observation_features,
                self.observation_data,
                config=tconfig_copy,
            )

        with self.assertRaises(ValueError):
            tconfig_copy = dict(self.tconfig)
            tconfig_copy["trial_index_to_type"].pop(0)
            ConvertMetricNames(
                None,
                self.observation_features,
                self.observation_data,
                config=tconfig_copy,
            )

    def testMultipleMetrics(self):
        # Create copy of online metric for offline
        online_metric = copy(self.experiment.metrics["m1"])
        online_metric._name = "m3"
        self.experiment.add_tracking_metric(online_metric, "type2", "m4")
        tconfig = tconfig_from_mt_experiment(self.experiment)
        ConvertMetricNames(
            None, self.observation_features, self.observation_data, config=tconfig
        )
