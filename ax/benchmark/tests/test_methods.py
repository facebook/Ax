# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.benchmark.methods.modular_botorch import get_sobol_botorch_modular_acquisition
from ax.modelbridge.registry import Models
from ax.utils.common.testutils import TestCase
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient


class TestMethods(TestCase):
    def test_mbm_acquisition(self) -> None:
        method = get_sobol_botorch_modular_acquisition(
            acquisition_cls=qKnowledgeGradient,
            acquisition_options={"num_fantasies": 16},
        )
        self.assertEqual(method.name, "SOBOL+BOTORCH_MODULAR::qKnowledgeGradient")
        gs = method.generation_strategy
        sobol, kg = gs._steps
        self.assertEqual(kg.model, Models.BOTORCH_MODULAR)
        model_kwargs = kg.model_kwargs
        # pyre-fixme[16]: Optional type has no attribute `__getitem__`.
        self.assertEqual(model_kwargs["botorch_acqf_class"], qKnowledgeGradient)
        self.assertEqual(model_kwargs["acquisition_options"], {"num_fantasies": 16})
