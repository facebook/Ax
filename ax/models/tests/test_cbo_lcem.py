#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from ax.models.torch.cbo_lcem import LCEMBO
from ax.utils.common.testutils import TestCase
from botorch.models.contextual_multioutput import LCEMGP, FixedNoiseLCEMGP
from botorch.models.model_list_gp_regression import ModelListGP


class LCEMBOTest(TestCase):
    def testLCEMBO(self):
        d = 1
        train_x = torch.rand(10, d)
        train_y = torch.cos(train_x)
        task_indices = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        train_x = torch.cat([train_x, task_indices.unsqueeze(-1)], axis=1)

        # Test setting attributes
        m = LCEMBO()
        self.assertIsNone(m.context_cat_feature)
        self.assertIsNone(m.context_emb_feature)
        self.assertIsNone(m.embs_dim_list)

        # Test get_and_fit_model
        train_yvar = np.nan * torch.ones(train_y.shape)
        gp = m.get_and_fit_model(
            Xs=[train_x],
            Ys=[train_y],
            Yvars=[train_yvar],
            task_features=[d],
            fidelity_features=[],
            metric_names=[],
        )
        self.assertIsInstance(gp, ModelListGP)
        self.assertIsInstance(gp.models[0], LCEMGP)

        train_yvar = 0.05 * torch.ones(train_y.shape)
        gp = m.get_and_fit_model(
            Xs=[train_x],
            Ys=[train_y],
            Yvars=[train_yvar],
            task_features=[d],
            fidelity_features=[],
            metric_names=[],
        )
        self.assertIsInstance(gp, ModelListGP)
        self.assertIsInstance(gp.models[0], FixedNoiseLCEMGP)

        # Verify errors are raised in get_and_fit_model
        train_yvar = np.nan * torch.ones(train_y.shape)
        with self.assertRaises(NotImplementedError):
            gp = m.get_and_fit_model(
                Xs=[train_x],
                Ys=[train_y],
                Yvars=[train_yvar],
                task_features=[d, 2],
                fidelity_features=[],
                metric_names=[],
            )
        with self.assertRaises(ValueError):
            gp = m.get_and_fit_model(
                Xs=[train_x],
                Ys=[train_y],
                Yvars=[train_yvar],
                task_features=[],
                fidelity_features=[],
                metric_names=[],
            )
