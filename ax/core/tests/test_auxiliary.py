# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.core.auxiliary import AuxiliaryExperiment
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment, get_experiment_with_data


class AuxiliaryExperimentTest(TestCase):
    def test_AuxiliaryExperiment(self) -> None:
        for get_exp_func in [get_experiment, get_experiment_with_data]:
            exp = get_exp_func()
            data = exp.lookup_data()

            # Test init
            aux_exp = AuxiliaryExperiment(experiment=exp)
            self.assertEqual(aux_exp.experiment, exp)
            self.assertEqual(aux_exp.data, data)

            another_aux_exp = AuxiliaryExperiment(
                experiment=exp, data=exp.lookup_data()
            )
            self.assertEqual(another_aux_exp.experiment, exp)
            self.assertEqual(another_aux_exp.data, data)
