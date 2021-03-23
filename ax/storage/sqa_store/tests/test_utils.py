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
        exp2 = get_experiment_with_batch_trial()
        self.assertEqual(exp1, exp2)

        save_experiment(exp1)
        # exp1 has db_ids now, but exp2 does not
        self.assertNotEqual(exp1, exp2)

        # copy db_ids from exp1 to exp2
        copy_db_ids(exp1, exp2)
        self.assertEqual(exp1, exp2)

        loaded_exp1 = load_experiment(exp1.name)
        self.assertEqual(loaded_exp1, exp2)

    def testCopyDBIDsDataExp(self):
        exp1 = get_experiment_with_data()
        exp2 = get_experiment_with_data()
        # need to copy this over, otherwise timestamps won't be equal
        exp2._data_by_trial = dict(exp1._data_by_trial)
        self.assertEqual(exp1, exp2)

        save_experiment(exp1)
        self.assertNotEqual(exp1, exp2)

        copy_db_ids(exp1, exp2)
        self.assertEqual(exp1, exp2)

        data, _ = exp2.lookup_data_for_trial(0)
        self.assertIsNotNone(data._db_id)

        loaded_exp1 = load_experiment(exp1.name)
        self.assertEqual(loaded_exp1, exp2)
