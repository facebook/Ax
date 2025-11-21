#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from copy import deepcopy
from functools import partial
from itertools import product

import numpy as np

from ax.adapter.base import Adapter, DataLoaderConfig
from ax.adapter.data_utils import extract_experiment_data
from ax.adapter.transforms.bilog_y import bilog_transform, BilogY, inv_bilog_transform
from ax.adapter.transforms.log_y import match_ci_width
from ax.core.observation_utils import observations_from_data
from ax.generators.base import Generator
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from pandas.testing import assert_frame_equal, assert_series_equal


class BilogYTest(TestCase):
    def setUp(self) -> None:
        self.exp = get_branin_experiment(
            with_status_quo=True,
            with_completed_batch=True,
            with_absolute_constraint=True,
            with_relative_constraint=True,
        )
        self.data = self.exp.fetch_data()
        self.bound = self.exp.optimization_config.outcome_constraints[1].bound

    def get_adapter(self) -> Adapter:
        return Adapter(
            search_space=self.exp.search_space,
            generator=Generator(),
            experiment=self.exp,
            data=self.exp.lookup_data(),
        )

    def test_Init(self) -> None:
        # With adapter.
        t = BilogY(
            search_space=self.exp.search_space,
            adapter=self.get_adapter(),
        )
        self.assertEqual(t.metric_to_bound, {"branin_e": self.bound})

        with self.subTest("With no adapter"):
            t = BilogY(
                search_space=self.exp.search_space,
                adapter=None,
            )
            self.assertEqual(t.metric_to_bound, {})

    def test_Bilog(self) -> None:
        self.assertAlmostEqual(
            float(bilog_transform(y=np.array(7.3), bound=3)), 4.667706820558076
        )
        self.assertAlmostEqual(
            float(bilog_transform(y=np.array(7.3), bound=10)), 8.691667180349821
        )
        self.assertAlmostEqual(
            float(inv_bilog_transform(y=np.array(4.3), bound=3)), 5.669296667619244
        )
        self.assertAlmostEqual(
            float(inv_bilog_transform(y=np.array(2.3), bound=3)), 1.9862472925295231
        )
        self.assertAlmostEqual(
            float(
                inv_bilog_transform(bilog_transform(y=np.array(0.3), bound=3), bound=3)
            ),
            0.3,
        )
        self.assertAlmostEqual(
            float(
                bilog_transform(inv_bilog_transform(y=np.array(0.3), bound=3), bound=3)
            ),
            0.3,
        )

    def test_TransformUntransform(self) -> None:
        bound = self.exp.optimization_config.outcome_constraints[0].bound
        observations = observations_from_data(
            experiment=self.exp, data=self.exp.lookup_data()
        )
        t = BilogY(
            search_space=self.exp.search_space,
            adapter=self.get_adapter(),
        )

        # Transform
        transformed_data = t._transform_observation_data(
            deepcopy([obs.data for obs in observations])
        )
        for obs, transform_obs in zip(observations, transformed_data):
            # Non-constraints should be the same
            self.assertEqual(
                transform_obs.metric_signatures, ["branin", "branin_d", "branin_e"]
            )
            self.assertTrue((transform_obs.means[0:2] == obs.data.means[0:2]).all())
            self.assertTrue(
                (
                    transform_obs.covariance[0:2, 0:2] == obs.data.covariance[0:2, 0:2]
                ).all()
            )
            # Make sure the bilog transform with ci width matching was applied
            self.assertAlmostEqual(
                transform_obs.means[2], bilog_transform(obs.data.means[2], bound=bound)
            )
            self.assertTrue(  # The transformed variance should be smaller
                transform_obs.covariance[2, 2] < obs.data.covariance[2, 2]
            )

        # Untransform
        untransformed_data = t._untransform_observation_data(deepcopy(transformed_data))
        for obs, untransform_obs, transform_obs in zip(
            observations, untransformed_data, transformed_data
        ):
            # Non-constraints should be the same
            self.assertEqual(
                untransform_obs.metric_signatures, ["branin", "branin_d", "branin_e"]
            )
            self.assertTrue((untransform_obs.means[0:2] == obs.data.means[0:2]).all())
            self.assertTrue(
                (
                    untransform_obs.covariance[0:2, 0:2]
                    == obs.data.covariance[0:2, 0:2]
                ).all()
            )
            # Make sure the inverse bilog transform with ci width matching was applied
            self.assertAlmostEqual(
                untransform_obs.means[2],
                inv_bilog_transform(transform_obs.means[2], bound=bound),
            )
            # And that we are back where we started as invf(f(x)) = x
            self.assertAlmostEqual(
                obs.data.means[2],
                inv_bilog_transform(transform_obs.means[2], bound=bound),
            )
            self.assertAlmostEqual(
                untransform_obs.covariance[2, 2], obs.data.covariance[2, 2]
            )

    def test_TransformOptimizationConfig(self) -> None:
        t = BilogY(
            search_space=self.exp.search_space,
            adapter=self.get_adapter(),
        )
        oc = self.exp.optimization_config
        # This should be a no-op
        new_oc = t.transform_optimization_config(optimization_config=oc)
        self.assertEqual(new_oc, oc)

    def test_TransformSearchSpace(self) -> None:
        t = BilogY(search_space=self.exp.search_space, adapter=self.get_adapter())
        # This should be a no-op
        new_ss = t.transform_search_space(self.exp.search_space)
        self.assertEqual(new_ss, self.exp.search_space)

    def test_transform_experiment_data(self) -> None:
        t = BilogY(search_space=self.exp.search_space, adapter=self.get_adapter())
        experiment_data = extract_experiment_data(
            experiment=self.exp, data_loader_config=DataLoaderConfig()
        )
        transformed_data = t.transform_experiment_data(
            experiment_data=deepcopy(experiment_data)
        )

        # Check that arm data is identical.
        assert_frame_equal(transformed_data.arm_data, experiment_data.arm_data)

        # Check that non-constraint metrics are unchanged.
        cols = list(product(("mean", "sem"), ("branin", "branin_d")))
        assert_frame_equal(
            transformed_data.observation_data[cols],
            experiment_data.observation_data[cols],
        )

        # Check that `branin_e` has been transformed correctly.
        assert_series_equal(
            transformed_data.observation_data[("mean", "branin_e")],
            bilog_transform(
                experiment_data.observation_data[("mean", "branin_e")], bound=self.bound
            ),
        )
        # Sem is smaller than before.
        self.assertTrue(
            (
                transformed_data.observation_data[("sem", "branin_e")]
                < experiment_data.observation_data[("sem", "branin_e")]
            ).all()
        )
        # Compare against transforming the old way.
        mean, var = match_ci_width(
            mean=experiment_data.observation_data[("mean", "branin_e")],
            sem=None,
            variance=experiment_data.observation_data[("sem", "branin_e")] ** 2,
            transform=partial(bilog_transform, bound=self.bound),
        )
        assert_series_equal(
            transformed_data.observation_data[("mean", "branin_e")], mean
        )
        # Can't use assert_series_equal since the metadata is destroyed in var.
        self.assertTrue(
            transformed_data.observation_data[("sem", "branin_e")].equals(var**0.5)
        )
