#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass, field
from unittest import mock

from ax.core.base_trial import BaseTrial
from ax.core.runner import Runner, RunnerConfig
from ax.core.search_space import SearchSpace
from ax.utils.common.sentinel import Unset, UNSET
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_batch_trial, get_trial
from pyre_extensions import override


class DummyRunner(Runner):
    def run(self, trial: BaseTrial) -> dict[str, str]:
        return {"metadatum": f"value_for_trial_{trial.index}"}


class DummyRunnerConfig(RunnerConfig):
    @dataclass(frozen=True)
    class RunnerUpdateArguments(RunnerConfig.RunnerUpdateArguments):
        name: str | None | Unset = UNSET
        count: int | None | Unset = UNSET
        tag: str | None | Unset = field(default=UNSET, metadata={"attr": "_tag"})

    @dataclass(frozen=True)
    class SearchSpaceUpdateArguments(RunnerConfig.SearchSpaceUpdateArguments):
        label: str | None | Unset = UNSET


class DummyUpdatableRunner(Runner):
    config_type = DummyRunnerConfig

    def __init__(self) -> None:
        self.name: str | None = "original"
        self.count: int | None = 5
        self.label: str | None = "default_label"
        self._tag: str | None = "old_tag"
        self.events: list[str] = []

    @override
    def _validate_on_search_space_update(
        self,
        search_space: SearchSpace,
        arguments: RunnerConfig.SearchSpaceUpdateArguments | None = None,
    ) -> None:
        self.events.append(f"_validate_on_search_space_update:label={self.label}")

    def run(self, trial: BaseTrial) -> dict[str, str]:
        return {}


class RunnerTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.dummy_runner = DummyRunner()
        self.trials = [get_trial(), get_batch_trial()]

    def test_base_runner_staging_required(self) -> None:
        self.assertFalse(self.dummy_runner.staging_required)

    def test_base_runner_stop(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.dummy_runner.stop(trial=mock.Mock(), reason="")

    def test_base_runner_clone(self) -> None:
        runner_clone = self.dummy_runner.clone()
        self.assertIsInstance(runner_clone, DummyRunner)
        self.assertEqual(runner_clone, self.dummy_runner)

    def test_base_runner_run_multiple(self) -> None:
        metadata = self.dummy_runner.run_multiple(trials=self.trials)
        self.assertEqual(
            metadata,
            {t.index: {"metadatum": f"value_for_trial_{t.index}"} for t in self.trials},
        )
        self.assertEqual({}, self.dummy_runner.run_multiple(trials=[]))

    def test_base_runner_poll_trial_status(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.dummy_runner.poll_trial_status(trials=self.trials)

    def test_base_runner_poll_exception(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.dummy_runner.poll_exception(trial=self.trials[0])

    def test_poll_available_capacity(self) -> None:
        self.assertEqual(self.dummy_runner.poll_available_capacity(), -1)

    def test_on_search_space_update_default_is_noop(self) -> None:
        self.assertIsNone(
            self.dummy_runner.on_search_space_update(search_space=mock.Mock())
        )

    def test_run_metadata_report_keys(self) -> None:
        self.assertEqual(self.dummy_runner.run_metadata_report_keys, [])

    def test_set_attributes(self) -> None:
        """_set_attributes applies non-UNSET fields, skips UNSET ones,
        and respects 'attr' metadata for private attribute names."""
        runner = DummyUpdatableRunner()
        with self.subTest("applies non-UNSET, skips UNSET"):
            runner._set_attributes(
                DummyRunnerConfig.RunnerUpdateArguments(name="new_name")
            )
            self.assertEqual(runner.name, "new_name")
            self.assertEqual(runner.count, 5)
        with self.subTest("respects attr metadata"):
            runner._set_attributes(
                DummyRunnerConfig.RunnerUpdateArguments(tag="new_tag")
            )
            self.assertEqual(runner._tag, "new_tag")

    def test_on_search_space_update(self) -> None:
        """on_search_space_update validates before applying, applies fields,
        rejects wrong argument types, and is a no-op without arguments."""
        runner = DummyUpdatableRunner()
        ss = mock.Mock()
        with self.subTest("no-op without arguments"):
            runner.on_search_space_update(search_space=ss)
            self.assertEqual(runner.label, "default_label")
        with self.subTest("validates before applying"):
            runner.on_search_space_update(
                search_space=ss,
                arguments=DummyRunnerConfig.SearchSpaceUpdateArguments(label="updated"),
            )
            self.assertEqual(
                runner.events,
                [
                    "_validate_on_search_space_update:label=default_label",
                    "_validate_on_search_space_update:label=default_label",
                ],
            )
            self.assertEqual(runner.label, "updated")
        with self.subTest("rejects wrong argument type"):
            with self.assertRaisesRegex(TypeError, "Expected"):
                runner.on_search_space_update(
                    search_space=ss,
                    # pyre-ignore[6]: Intentionally passing wrong type.
                    arguments=RunnerConfig.RunnerUpdateArguments(),
                )
