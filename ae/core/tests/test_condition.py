#!/usr/bin/env python3

from ae.lazarus.ae.core.condition import Condition
from ae.lazarus.ae.utils.common.testutils import TestCase


class ConditionTest(TestCase):
    def setUp(self):
        pass

    def testInit(self):
        condition = Condition(params={"y": 0.25, "x": 0.75, "z": 75})
        self.assertEqual(
            str(condition), "Condition(params={'y': 0.25, 'x': 0.75, 'z': 75})"
        )

        condition = Condition(params={"y": 0.25, "x": 0.75, "z": 75}, name="status_quo")
        self.assertEqual(
            str(condition),
            "Condition(name=status_quo, params={'y': 0.25, 'x': 0.75, 'z': 75})",
        )

    def testNameValidation(self):
        condition = Condition(params={"y": 0.25, "x": 0.75, "z": 75})
        self.assertFalse(condition.has_name)
        with self.assertRaises(ValueError):
            getattr(condition, "name")
        condition.name = "0_0"
        with self.assertRaises(ValueError):
            condition.name = "1_0"

    def testNameOrShortSignature(self):
        condition = Condition(params={"y": 0.25, "x": 0.75, "z": 75}, name="0_0")
        self.assertEqual(condition.name_or_short_signature, "0_0")

        condition = Condition(params={"y": 0.25, "x": 0.75, "z": 75})
        self.assertEqual(condition.name_or_short_signature, condition.signature[-4:])

    def testEq(self):
        condition1 = Condition(params={"y": 0.25, "x": 0.75, "z": 75})
        condition2 = Condition(params={"z": 75, "x": 0.75, "y": 0.25})
        self.assertEqual(condition1, condition2)

        condition3 = Condition(params={"z": 5, "x": 0.75, "y": 0.25})
        self.assertNotEqual(condition1, condition3)

        condition4 = Condition(name="0_0", params={"y": 0.25, "x": 0.75, "z": 75})
        condition5 = Condition(name="0_0", params={"y": 0.25, "x": 0.75, "z": 75})
        self.assertEqual(condition4, condition5)

        condition6 = Condition(name="0_1", params={"y": 0.25, "x": 0.75, "z": 75})
        self.assertNotEqual(condition4, condition6)

    def testClone(self):
        condition1 = Condition(params={"y": 0.25, "x": 0.75, "z": 75})
        condition2 = condition1.clone()
        self.assertFalse(condition1 is condition2)
        self.assertEqual(condition1, condition2)
        self.assertFalse(condition1.params is condition2.params)
