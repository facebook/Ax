#!/usr/bin/env python3

from ae.lazarus.ae.core.arm import Arm
from ae.lazarus.ae.utils.common.testutils import TestCase


class ArmTest(TestCase):
    def setUp(self):
        pass

    def testInit(self):
        arm = Arm(params={"y": 0.25, "x": 0.75, "z": 75})
        self.assertEqual(str(arm), "Arm(params={'y': 0.25, 'x': 0.75, 'z': 75})")

        arm = Arm(params={"y": 0.25, "x": 0.75, "z": 75}, name="status_quo")
        self.assertEqual(
            str(arm), "Arm(name=status_quo, params={'y': 0.25, 'x': 0.75, 'z': 75})"
        )

    def testNameValidation(self):
        arm = Arm(params={"y": 0.25, "x": 0.75, "z": 75})
        self.assertFalse(arm.has_name)
        with self.assertRaises(ValueError):
            getattr(arm, "name")
        arm.name = "0_0"
        with self.assertRaises(ValueError):
            arm.name = "1_0"

    def testNameOrShortSignature(self):
        arm = Arm(params={"y": 0.25, "x": 0.75, "z": 75}, name="0_0")
        self.assertEqual(arm.name_or_short_signature, "0_0")

        arm = Arm(params={"y": 0.25, "x": 0.75, "z": 75})
        self.assertEqual(arm.name_or_short_signature, arm.signature[-4:])

    def testEq(self):
        arm1 = Arm(params={"y": 0.25, "x": 0.75, "z": 75})
        arm2 = Arm(params={"z": 75, "x": 0.75, "y": 0.25})
        self.assertEqual(arm1, arm2)

        arm3 = Arm(params={"z": 5, "x": 0.75, "y": 0.25})
        self.assertNotEqual(arm1, arm3)

        arm4 = Arm(name="0_0", params={"y": 0.25, "x": 0.75, "z": 75})
        arm5 = Arm(name="0_0", params={"y": 0.25, "x": 0.75, "z": 75})
        self.assertEqual(arm4, arm5)

        arm6 = Arm(name="0_1", params={"y": 0.25, "x": 0.75, "z": 75})
        self.assertNotEqual(arm4, arm6)

    def testClone(self):
        arm1 = Arm(params={"y": 0.25, "x": 0.75, "z": 75})
        arm2 = arm1.clone()
        self.assertFalse(arm1 is arm2)
        self.assertEqual(arm1, arm2)
        self.assertFalse(arm1.params is arm2.params)
