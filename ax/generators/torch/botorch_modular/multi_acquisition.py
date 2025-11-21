#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from ax.generators.torch.botorch_modular.acquisition import Acquisition
from botorch.acquisition.multioutput_acquisition import (
    MultiOutputAcquisitionFunctionWrapper,
)


class MultiAcquisition(Acquisition):
    """
    A MultiAcquisition class for generating points by optimizing multiple
    acquisition functions jointly.
    """

    def _instantiate_acquisition(
        self,
    ) -> None:
        """Constructs the acquisition function based on the provided AF clases."""
        self.acq_function_sequence = None
        if len(self.botorch_acqf_classes_with_options) > 1:
            acqfs = [
                self._construct_botorch_acquisition(
                    botorch_acqf_class=botorch_acqf_class,
                    botorch_acqf_options=botorch_acqf_options,
                    model=self._model,
                )
                for botorch_acqf_class, botorch_acqf_options in (
                    self.botorch_acqf_classes_with_options
                )
            ]
            self.acqf = MultiOutputAcquisitionFunctionWrapper(acqfs=acqfs)
            self.models_used = [self.surrogate.model_name_by_metric]
        else:
            # Using one acqf with multiple models.
            botorch_acqf_class, botorch_acqf_options = (
                self.botorch_acqf_classes_with_options[0]
            )
            # Default acqf is the surrogate default.
            self.acqf = self._construct_botorch_acquisition(
                botorch_acqf_class=botorch_acqf_class,
                botorch_acqf_options=botorch_acqf_options,
                model=self._model,
            )
            self.models_used = [self.surrogate.model_name_by_metric]
            if self.n is not None and self.n > 1:
                # Using multiple models
                models_used, models = self.surrogate.models_for_gen(n=self.n)
                # Either 1 or n models will be returned. If 1, we do nothing.
                if len(models) == self.n:
                    self.acq_function_sequence = []
                    for model in models:
                        model, _, _, _ = self._subset_model(
                            model=model, objective_weights=self._full_objective_weights
                        )
                        acqf = self._construct_botorch_acquisition(
                            botorch_acqf_class=botorch_acqf_class,
                            botorch_acqf_options=botorch_acqf_options,
                            model=model,
                        )
                        self.acq_function_sequence.append(acqf)
                    self.models_used = models_used
