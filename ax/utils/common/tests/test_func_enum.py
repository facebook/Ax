#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.utils.common.func_enum import FuncEnum
from ax.utils.common.testutils import TestCase


def morph_into_salamander(how_soon: int) -> bool:
    if how_soon <= 0:  # Now is the time!
        return True
    else:  # Not morphing yet
        return False


class AnimalAbilities(FuncEnum):
    # ꒰(˶• ᴗ •˶)꒱
    AXOLOTL_MORPH = "morph_into_salamander"


class EqualityTest(TestCase):
    def test_basic(self) -> None:
        self.assertEqual(  # Check underlying function correctness.
            AnimalAbilities.AXOLOTL_MORPH._get_function_for_value(),
            morph_into_salamander,
        )

    def test_call(self) -> None:
        # Should be too early to morph...
        self.assertFalse(AnimalAbilities.AXOLOTL_MORPH(how_soon=1))
        # Should've morphed yesterday!
        self.assertTrue(AnimalAbilities.AXOLOTL_MORPH(how_soon=-1))
