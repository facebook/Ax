# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import unique

from ax.core.auxiliary import AuxiliaryExperiment, AuxiliaryExperimentType
from ax.core.experiment import Experiment
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_experiment,
    get_experiment_with_data,
    get_search_space,
)


class AuxiliaryExperimentTest(TestCase):
    def test_AuxiliaryExperiment(self) -> None:
        @unique
        class TestAuxiliaryExperimentType(AuxiliaryExperimentType):
            MyAuxExptType = "my_auxiliary_experiment_type"
            MyOtherAuxExpType = "my_other_auxiliary_experiment_type"

        for get_exp_func in [get_experiment, get_experiment_with_data]:
            exp = get_exp_func()
            data = exp.lookup_data()

            # Test init
            aux_exp = AuxiliaryExperiment(experiment=exp)
            self.assertEqual(aux_exp.experiment, exp)
            self.assertEqual(aux_exp.data, data)

            aux_exp = AuxiliaryExperiment(experiment=exp, data=exp.lookup_data())
            self.assertEqual(aux_exp.experiment, exp)
            self.assertEqual(aux_exp.data, data)

            # init exp with auxiliary_experiments
            with self.assertRaisesRegex(
                TypeError,
                "auxiliary_experiments_by_type must be a dict "
                "of AuxiliaryExperimentType",
            ):
                exp_w_aux_exp = Experiment(
                    name="test",
                    search_space=get_search_space(),
                    auxiliary_experiments_by_type={  # pyre-ignore
                        "some_use_case": [aux_exp],
                    },
                )
            exp_w_aux_exp = Experiment(
                name="test",
                search_space=get_search_space(),
                auxiliary_experiments_by_type={
                    TestAuxiliaryExperimentType.MyAuxExptType: [aux_exp],
                },
            )

            # modifying auxiliary_experiments_by_type
            with self.assertRaisesRegex(
                TypeError,
                "'mappingproxy' object does not support item assignment",
            ):
                exp_w_aux_exp.auxiliary_experiments_by_type[  # pyre-ignore
                    TestAuxiliaryExperimentType.MyOtherAuxExpType
                ] = [aux_exp]

            exp_w_aux_exp.update_auxiliary_experiments(
                auxiliary_experiment_type=TestAuxiliaryExperimentType.MyOtherAuxExpType,
                auxiliary_experiments=[aux_exp],
            )

            self.assertEqual(
                exp_w_aux_exp.auxiliary_experiments_by_type,
                {
                    TestAuxiliaryExperimentType.MyAuxExptType: [aux_exp],
                    TestAuxiliaryExperimentType.MyOtherAuxExpType: [aux_exp],
                },
            )
