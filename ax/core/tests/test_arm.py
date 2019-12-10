#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.core.arm import Arm
from ax.utils.common.testutils import TestCase


class ArmTest(TestCase):
    def setUp(self):
        pass

    def testInit(self):
        arm = Arm(parameters={"y": 0.25, "x": 0.75, "z": 75})
        self.assertEqual(str(arm), "Arm(parameters={'y': 0.25, 'x': 0.75, 'z': 75})")

        arm = Arm(parameters={"y": 0.25, "x": 0.75, "z": 75}, name="status_quo")
        self.assertEqual(
            str(arm),
            "Arm(name='status_quo', parameters={'y': 0.25, 'x': 0.75, 'z': 75})",
        )

    def testNameValidation(self):
        arm = Arm(parameters={"y": 0.25, "x": 0.75, "z": 75})
        self.assertFalse(arm.has_name)
        with self.assertRaises(ValueError):
            arm.name
        arm.name = "0_0"
        with self.assertRaises(ValueError):
            arm.name = "1_0"

    def testNameOrShortSignature(self):
        arm = Arm(parameters={"y": 0.25, "x": 0.75, "z": 75}, name="0_0")
        self.assertEqual(arm.name_or_short_signature, "0_0")

        arm = Arm(parameters={"y": 0.25, "x": 0.75, "z": 75})
        self.assertEqual(arm.name_or_short_signature, arm.signature[-4:])

    def testEq(self):
        arm1 = Arm(parameters={"y": 0.25, "x": 0.75, "z": 75})
        arm2 = Arm(parameters={"z": 75, "x": 0.75, "y": 0.25})
        self.assertEqual(arm1, arm2)

        arm3 = Arm(parameters={"z": 5, "x": 0.75, "y": 0.25})
        self.assertNotEqual(arm1, arm3)

        arm4 = Arm(name="0_0", parameters={"y": 0.25, "x": 0.75, "z": 75})
        arm5 = Arm(name="0_0", parameters={"y": 0.25, "x": 0.75, "z": 75})
        self.assertEqual(arm4, arm5)

        arm6 = Arm(name="0_1", parameters={"y": 0.25, "x": 0.75, "z": 75})
        self.assertNotEqual(arm4, arm6)

    def testClone(self):
        arm1 = Arm(parameters={"y": 0.25, "x": 0.75, "z": 75})
        arm2 = arm1.clone()
        self.assertFalse(arm1 is arm2)
        self.assertEqual(arm1, arm2)
        self.assertFalse(arm1.parameters is arm2.parameters)
