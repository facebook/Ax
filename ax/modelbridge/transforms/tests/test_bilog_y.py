#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from copy import deepcopy

from ax.core.observation import observations_from_data
from ax.exceptions.core import DataRequiredError
from ax.modelbridge.base import Adapter
from ax.modelbridge.transforms.bilog_y import (
    bilog_transform,
    BilogY,
    inv_bilog_transform,
)
from ax.models.base import Generator
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment


class BilogYTest(TestCase):
    def setUp(self) -> None:
        self.exp = get_branin_experiment(
            with_status_quo=True,
            with_completed_batch=True,
            with_absolute_constraint=True,
            with_relative_constraint=True,
        )
        self.data = self.exp.fetch_data()

    def get_mb(self) -> Adapter:
        return Adapter(
            search_space=self.exp.search_space,
            model=Generator(),
            experiment=self.exp,
            data=self.exp.lookup_data(),
        )

    def test_Init(self) -> None:
        observations = observations_from_data(
            experiment=self.exp, data=self.exp.lookup_data()
        )
        t = BilogY(
            search_space=self.exp.search_space,
            observations=observations,
            modelbridge=self.get_mb(),
        )
        self.assertEqual(t.metric_to_bound, {"branin_e": -0.25})

    def test_Bilog(self) -> None:
        self.assertAlmostEqual(
            float(bilog_transform(y=7.3, bound=3)), 4.667706820558076
        )
        self.assertAlmostEqual(
            float(bilog_transform(y=7.3, bound=10)), 8.691667180349821
        )
        self.assertAlmostEqual(
            float(inv_bilog_transform(y=4.3, bound=3)), 5.669296667619244
        )
        self.assertAlmostEqual(
            float(inv_bilog_transform(y=2.3, bound=3)), 1.9862472925295231
        )
        self.assertAlmostEqual(
            float(inv_bilog_transform(bilog_transform(y=0.3, bound=3), bound=3)),
            0.3,
        )
        self.assertAlmostEqual(
            float(bilog_transform(inv_bilog_transform(y=0.3, bound=3), bound=3)),
            0.3,
        )

    def test_TransformUntransform(self) -> None:
        bound = self.exp.optimization_config.outcome_constraints[0].bound
        observations = observations_from_data(
            experiment=self.exp, data=self.exp.lookup_data()
        )
        t = BilogY(
            search_space=self.exp.search_space,
            observations=observations,
            modelbridge=self.get_mb(),
        )

        # Transform
        transformed_data = t._transform_observation_data(
            deepcopy([obs.data for obs in observations])
        )
        for obs, transform_obs in zip(observations, transformed_data):
            # Non-constraints should be the same
            self.assertEqual(
                transform_obs.metric_names, ["branin", "branin_d", "branin_e"]
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
                untransform_obs.metric_names, ["branin", "branin_d", "branin_e"]
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
            observations=observations_from_data(
                experiment=self.exp, data=self.exp.lookup_data()
            ),
            modelbridge=self.get_mb(),
        )
        oc = self.exp.optimization_config
        # This should be a no-op
        new_oc = t.transform_optimization_config(optimization_config=oc)
        self.assertEqual(new_oc, oc)

    def test_TransformSearchSpace(self) -> None:
        t = BilogY(
            search_space=self.exp.search_space,
            observations=observations_from_data(
                experiment=self.exp, data=self.exp.lookup_data()
            ),
            modelbridge=self.get_mb(),
        )
        # This should be a no-op
        new_ss = t.transform_search_space(self.exp.search_space)
        self.assertEqual(new_ss, self.exp.search_space)

    def test_ModelBridgeIsNone(self) -> None:
        t = BilogY(
            search_space=self.exp.search_space,
            observations=observations_from_data(
                experiment=self.exp, data=self.exp.lookup_data()
            ),
            modelbridge=None,
        )
        self.assertEqual(t.metric_to_bound, {})

    def test_Raises(self) -> None:
        exp = get_branin_experiment(with_status_quo=True, with_batch=True)
        with self.assertRaisesRegex(DataRequiredError, "BilogY requires observations."):
            BilogY(
                search_space=exp.search_space,
                observations=observations_from_data(
                    experiment=exp, data=exp.lookup_data()
                ),
                modelbridge=None,
            )
        # Relative constraints should raise
        exp = get_branin_experiment(
            with_status_quo=True,
            with_completed_batch=True,
            with_relative_constraint=True,
        )
