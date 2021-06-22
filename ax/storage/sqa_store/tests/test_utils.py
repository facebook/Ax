# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.storage.sqa_store.db import (
    init_test_engine_and_session_factory,
)
from ax.storage.sqa_store.load import (
    load_experiment,
)
from ax.storage.sqa_store.save import (
    save_experiment,
)
from ax.storage.sqa_store.utils import copy_db_ids
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_experiment_with_batch_trial,
    get_experiment_with_data,
)


class SQAStoreUtilsTest(TestCase):
    def setUp(self):
        init_test_engine_and_session_factory(force_init=True)

    def testCopyDBIDsBatchTrialExp(self):
        exp1 = get_experiment_with_batch_trial()
        save_experiment(exp1)
        exp2 = load_experiment(exp1.name)
        self.assertEqual(exp1, exp2)

        # empty some of exp2 db_ids
        exp2.trials[0].db_id = None
        exp2.trials[0].generator_runs[0].arms[0].db_id = None

        # copy db_ids from exp1 to exp2
        copy_db_ids(exp1, exp2)
        self.assertEqual(exp1, exp2)

    def testCopyDBIDsDataExp(self):
        exp1 = get_experiment_with_data()
        save_experiment(exp1)
        exp2 = load_experiment(exp1.name)
        self.assertEqual(exp1, exp2)

        # empty some of exp2 db_ids
        data, _ = exp2.lookup_data_for_trial(0)
        data.db_id = None

        # copy db_ids from exp1 to exp2
        copy_db_ids(exp1, exp2)
        self.assertEqual(exp1, exp2)

    def testCopyDBIDsRepeatedArms(self):
        exp = get_experiment_with_batch_trial()
        exp.trials[0]
        save_experiment(exp)

        exp.new_batch_trial().add_arms_and_weights(exp.trials[0].arms)
        save_experiment(exp)

        self.assertNotEqual(exp.trials[0].arms[0].db_id, exp.trials[1].arms[0].db_id)
