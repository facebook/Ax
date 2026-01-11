# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import tempfile
from pathlib import Path
from unittest import TestCase

from ax.api.client import Client
from ax.api.configs import RangeParameterConfig


class DummyRunner:
    def run_trial(self, trial) -> dict[str, int]:
        return {"trial_index": trial.index}


class ClientJsonSnapshotRunnerStrippingTest(TestCase):
    def test_runner_is_stripped_in_snapshot_and_restored_in_memory(self) -> None:
        client = Client()

        client.configure_experiment(
            name="test_exp",
            parameters=[
                RangeParameterConfig(
                    name="x",
                    parameter_type="float",
                    bounds=(0.0, 1.0),
                )
            ],
        )

        client.configure_optimization(objective="m1")

        runner = DummyRunner()
        client.configure_runner(runner=runner)

        # Sanity: runner attached before save
        self.assertIsNotNone(client._experiment.runner)
        self.assertIsInstance(client._experiment.runner, DummyRunner)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "client_state.json"

            client.save_to_json_file(str(path))

            # After save, runner should be restored.
            self.assertIsNotNone(client._experiment.runner)
            self.assertIsInstance(client._experiment.runner, DummyRunner)

            # Snapshot should record that runner was stripped.
            snapshot = json.loads(path.read_text())
            self.assertTrue(snapshot.get("runner_was_stripped", False))

            # Loading should produce a client with no runner attached.
            loaded = Client.load_from_json_file(str(path))
            self.assertIsNone(loaded._experiment.runner)
