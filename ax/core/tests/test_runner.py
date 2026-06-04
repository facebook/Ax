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

    def _validate_update(
        self,
        arguments: RunnerConfig.RunnerUpdateArguments,
        search_space: SearchSpace | None = None,
    ) -> None:
        self.events.append(f"_validate_update:name={self.name}")

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

    def test_update_rejects_wrong_argument_type(self) -> None:
        runner = DummyUpdatableRunner()
        with self.assertRaisesRegex(TypeError, "DummyUpdatableRunner expects"):
            runner.update(RunnerConfig.RunnerUpdateArguments())

    def test_update_applies_non_unset_fields(self) -> None:
        runner = DummyUpdatableRunner()
        runner.update(
            DummyRunnerConfig.RunnerUpdateArguments(name="new_name", count=10)
        )
        self.assertEqual(runner.name, "new_name")
        self.assertEqual(runner.count, 10)

    def test_update_skips_unset_fields(self) -> None:
        runner = DummyUpdatableRunner()
        runner.update(DummyRunnerConfig.RunnerUpdateArguments(name="changed"))
        self.assertEqual(runner.name, "changed")
        self.assertEqual(runner.count, 5)  # unchanged

    def test_update_allows_setting_to_none(self) -> None:
        runner = DummyUpdatableRunner()
        runner.update(DummyRunnerConfig.RunnerUpdateArguments(name=None))
        self.assertIsNone(runner.name)
        self.assertEqual(runner.count, 5)  # unchanged

    def test_update_uses_attr_metadata_for_private_attributes(self) -> None:
        runner = DummyUpdatableRunner()
        runner.update(DummyRunnerConfig.RunnerUpdateArguments(tag="new_tag"))
        self.assertEqual(runner._tag, "new_tag")

    def test_update_validates_before_applying(self) -> None:
        """Pre-validate runs before fields are applied (sees old values)."""
        runner = DummyUpdatableRunner()
        runner.update(DummyRunnerConfig.RunnerUpdateArguments(name="updated"))
        self.assertEqual(runner.events, ["_validate_update:name=original"])
        self.assertEqual(runner.name, "updated")

    def test_set_attributes_applies_non_unset_fields(self) -> None:
        """_set_attributes applies non-UNSET fields and skips UNSET ones."""
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

    def test_update_passes_search_space_to_validate(self) -> None:
        """update() forwards the search_space kwarg to _validate_update."""
        runner = DummyUpdatableRunner()
        ss_mock = mock.Mock()
        with mock.patch.object(
            DummyUpdatableRunner, "_validate_update"
        ) as mock_validate:
            runner.update(
                DummyRunnerConfig.RunnerUpdateArguments(name="new"),
                search_space=ss_mock,
            )
            mock_validate.assert_called_once_with(
                DummyRunnerConfig.RunnerUpdateArguments(name="new"),
                search_space=ss_mock,
            )
