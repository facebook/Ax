#!/usr/bin/env python3

from unittest import mock

import torch
from ae.lazarus.ae.models.torch import botorch_utils as utils
from ae.lazarus.ae.utils.common.testutils import TestCase


def objective(Y):
    return Y


class TestGetAcquisitionFunction(TestCase):
    def setUp(self):
        self.model = mock.MagicMock()
        self.X_observed = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        self.X_pending = torch.tensor([[1.0, 3.0, 4.0]])

    @mock.patch(f"{utils.__name__}.acquisition.batch_modules.qExpectedImprovement")
    def testGetQEI(self, mock_acquisition):
        acquisition_function = utils.get_acquisition_function(
            "qEI",
            self.model,
            self.X_observed,
            objective=objective,
            constraints=None,
            X_pending=self.X_pending,
            acquisition_function_args=None,
        )
        self.assertTrue(acquisition_function == mock_acquisition.return_value)
        mock_acquisition.assert_called_once_with(
            self.model,
            best_f=self.model.posterior(self.X_observed).mean.max().item(),
            objective=objective,
            constraints=None,
            X_pending=self.X_pending,
        )

    @mock.patch(f"{utils.__name__}.acquisition.batch_modules.qProbabilityOfImprovement")
    def testGetQPI(self, mock_acquisition):
        acquisition_function = utils.get_acquisition_function(
            "qPI",
            self.model,
            self.X_observed,
            objective=objective,
            constraints=None,
            X_pending=self.X_pending,
            acquisition_function_args=None,
        )
        self.assertTrue(acquisition_function == mock_acquisition.return_value)
        mock_acquisition.assert_called_once_with(
            self.model,
            best_f=self.model.posterior(self.X_observed).mean.max().item(),
            objective=objective,
            constraints=None,
            X_pending=self.X_pending,
        )

    @mock.patch(f"{utils.__name__}.acquisition.batch_modules.qNoisyExpectedImprovement")
    def testGetQNEI(self, mock_acquisition):
        acquisition_function = utils.get_acquisition_function(
            "qNEI",
            self.model,
            self.X_observed,
            objective=objective,
            constraints=None,
            X_pending=self.X_pending,
            acquisition_function_args=None,
        )
        self.assertTrue(acquisition_function == mock_acquisition.return_value)
        mock_acquisition.assert_called_once_with(
            self.model,
            self.X_observed,
            objective=objective,
            constraints=None,
            X_pending=self.X_pending,
        )

    @mock.patch(f"{utils.__name__}.acquisition.batch_modules.qUpperConfidenceBound")
    def testGetQUCB(self, mock_acquisition):
        acquisition_function = utils.get_acquisition_function(
            "qUCB",
            self.model,
            self.X_observed,
            objective=objective,
            constraints=None,
            X_pending=self.X_pending,
            acquisition_function_args={"beta": 2.0},
        )
        self.assertTrue(acquisition_function == mock_acquisition.return_value)
        mock_acquisition.assert_called_once_with(
            self.model, beta=2.0, X_pending=self.X_pending
        )

    def testGetQUCBNoBeta(self):
        self.assertRaises(
            ValueError,
            utils.get_acquisition_function,
            "qUCB",
            self.model,
            self.X_observed,
            objective=objective,
            constraints=None,
            X_pending=self.X_pending,
            acquisition_function_args={},
        )

    def testGetAcquisitionNotImplemented(self):
        self.assertRaises(
            NotImplementedError,
            utils.get_acquisition_function,
            "qKG",
            self.model,
            self.X_observed,
            objective=objective,
            constraints=None,
            X_pending=self.X_pending,
        )
