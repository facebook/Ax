#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from unittest import mock

import numpy as np
import pandas as pd
from ax.core.arm import Arm
from ax.core.generator_run import GeneratorRun
from ax.metrics.chemistry import ChemistryMetric, ChemistryProblemType
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_trial


class DummyEnum(Enum):
    DUMMY: str = "dummy"


class ChemistryMetricTest(TestCase):
    def testChemistryMetric(self):
        # basic test
        read_csv = pd.read_csv
        for problem_type in (
            ChemistryProblemType.DIRECT_ARYLATION,
            ChemistryProblemType.SUZUKI,
        ):
            with mock.patch(
                "ax.metrics.chemistry.pd.read_csv",
                wraps=lambda filename, index_col: read_csv(
                    filename, index_col=index_col, nrows=1
                ),
            ) as mock_read_csv:
                metric = ChemistryMetric(name="test_metric", problem_type=problem_type)
                self.assertFalse(metric.noiseless)
                self.assertIs(metric.problem_type, problem_type)
                self.assertFalse(metric.lower_is_better)
                if problem_type is ChemistryProblemType.DIRECT_ARYLATION:
                    param_names = [
                        "Base_SMILES",
                        "Concentration",
                        "Ligand_SMILES",
                        "Solvent_SMILES",
                        "Temp_C",
                    ]
                    param_values = (
                        "O=C([O-])C.[K+]",
                        0.1,
                        (
                            "CC(C)C1=CC(C(C)C)=C(C(C(C)C)=C1)C2=C(P(C3CCCCC3)"
                            "C4CCCCC4)C(OC)=CC=C2OC"
                        ),
                        "CC(N(C)C)=O",
                        105,
                    )
                    obj = 5.47
                else:
                    param_names = [
                        "Base_SMILES",
                        "Electrophile_SMILES",
                        "Ligand_SMILES",
                        "Nucleophile_SMILES",
                        "Solvent_SMILES",
                    ]
                    param_values = (
                        "[Na+].[OH-]",
                        "ClC1=CC=C(N=CC=C2)C2=C1",
                        "CC(P(C(C)(C)C)C(C)(C)C)(C)C",
                        "CC1=CC=C(N(C2CCCCO2)N=C3)C3=C1B(O)O",
                        "N#CC",
                    )
                    obj = 4.76

                params = dict(zip(param_names, param_values))
                trial = get_trial()
                trial._generator_run = GeneratorRun(
                    arms=[Arm(name="0_0", parameters=params)]
                )
                df = metric.fetch_trial_data(trial).df
                self.assertEqual(mock_read_csv.call_count, 1)
                self.assertEqual(df["mean"].values[0], obj)
                self.assertTrue(np.isnan(df["sem"].values[0]))
                # test caching
                metric.fetch_trial_data(trial)
                self.assertEqual(mock_read_csv.call_count, 1)

                # test noiseless
                metric = ChemistryMetric(
                    name="test_metric", problem_type=problem_type, noiseless=True
                )
                df = metric.fetch_trial_data(trial).df
                self.assertEqual(df["sem"].values[0], 0.0)
