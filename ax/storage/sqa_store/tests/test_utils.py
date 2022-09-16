# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.storage.sqa_store.db import init_test_engine_and_session_factory
from ax.storage.sqa_store.load import load_experiment
from ax.storage.sqa_store.save import save_experiment
from ax.storage.sqa_store.utils import copy_db_ids
from ax.utils.common.base import Base
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_experiment_with_batch_trial,
    get_experiment_with_data,
)


class DummyClassWithBaseline(Base):
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, baseline_workflow_inputs, db_id):
        # pyre-fixme[4]: Attribute must be annotated.
        self.baseline_workflow_inputs = baseline_workflow_inputs
        # pyre-fixme[4]: Attribute must be annotated.
        self.dummy_db_id = db_id


class SQAStoreUtilsTest(TestCase):
    def setUp(self) -> None:
        init_test_engine_and_session_factory(force_init=True)

    def testCopyDBIDsBatchTrialExp(self) -> None:
        exp1 = get_experiment_with_batch_trial()
        save_experiment(exp1)
        exp2 = load_experiment(exp1.name)
        self.assertEqual(exp1, exp2)

        # empty some of exp2 db_ids
        # pyre-fixme[8]: Attribute has type `int`; used as `None`.
        exp2.trials[0].db_id = None
        # pyre-fixme[8]: Attribute has type `int`; used as `None`.
        exp2.trials[0].generator_runs[0].arms[0].db_id = None

        # copy db_ids from exp1 to exp2
        copy_db_ids(exp1, exp2)
        self.assertEqual(exp1, exp2)

    def testCopyDBIDsDataExp(self) -> None:
        exp1 = get_experiment_with_data()
        save_experiment(exp1)
        exp2 = load_experiment(exp1.name)
        self.assertEqual(exp1, exp2)

        # empty some of exp2 db_ids
        data, _ = exp2.lookup_data_for_trial(0)
        # pyre-fixme[8]: Attribute has type `int`; used as `None`.
        data.db_id = None

        # copy db_ids from exp1 to exp2
        copy_db_ids(exp1, exp2)
        self.assertEqual(exp1, exp2)

    def testCopyDBIDsRepeatedArms(self) -> None:
        exp = get_experiment_with_batch_trial()
        exp.trials[0]
        save_experiment(exp)

        exp.new_batch_trial().add_arms_and_weights(exp.trials[0].arms)
        save_experiment(exp)

        self.assertNotEqual(exp.trials[0].arms[0].db_id, exp.trials[1].arms[0].db_id)

    def test_copy_db_ids_none_search_space(self) -> None:
        exp1 = get_experiment_with_batch_trial()
        save_experiment(exp1)
        exp2 = load_experiment(exp1.name)
        self.assertEqual(exp1, exp2)

        # empty search_space of exp1
        # pyre-fixme[8]: Attribute has type `SearchSpace`; used as `None`.
        exp1._search_space = None

        # empty some of exp2 db_ids
        # pyre-fixme[8]: Attribute has type `int`; used as `None`.
        exp2.trials[0].db_id = None
        # pyre-fixme[8]: Attribute has type `int`; used as `None`.
        exp2.trials[0].generator_runs[0].arms[0].db_id = None

        with self.assertWarnsRegex(
            Warning,
            "Encountered two objects of different types",
        ):
            # copy db_ids from exp1 to exp2
            copy_db_ids(exp1, exp2)

        # empty search space of exp2 for comparison
        # pyre-fixme[8]: Attribute has type `SearchSpace`; used as `None`.
        exp2._search_space = None
        self.assertEqual(exp1, exp2)

    def test_json_copy_db_ids(self) -> None:
        target_obj = DummyClassWithBaseline(
            baseline_workflow_inputs=[{1: 2, 3: 4}, {"a": "b", "c": "d"}],
            db_id=None,
        )
        source_obj = DummyClassWithBaseline(
            baseline_workflow_inputs=[{1: 2, 3: 4}, {"a": "b", "c": "d"}],
            db_id=1,
        )
        copy_db_ids(source_obj, target_obj)
        self.assertEqual(source_obj, target_obj)
