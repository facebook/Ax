#!/usr/bin/env python3

import numpy as np
from ae.lazarus.ae.core.parameter import ParameterType, RangeParameter
from ae.lazarus.ae.core.search_space import SearchSpace
from ae.lazarus.ae.generator.numpy import NumpyGenerator
from ae.lazarus.ae.generator.transforms.int_to_float import IntToFloat
from ae.lazarus.ae.generator.transforms.log import Log
from ae.lazarus.ae.generator.transforms.unit_x import UnitX
from ae.lazarus.ae.models.numpy.gpy import GPyGP
from ae.lazarus.ae.tests.fake import get_branin_experiment
from ae.lazarus.ae.utils.common.testutils import TestCase


class TransformCallbackTest(TestCase):
    def test_transform_callback_int(self):
        exp = get_branin_experiment()
        parameters = [
            RangeParameter(
                name="x1", parameter_type=ParameterType.INT, lower=1, upper=10
            ),
            RangeParameter(
                name="x2", parameter_type=ParameterType.INT, lower=5, upper=15
            ),
        ]
        gpei = NumpyGenerator(
            experiment=exp,
            search_space=SearchSpace(parameters=parameters),
            model=GPyGP(),
            transforms=[IntToFloat],
        )
        transformed = gpei._transform_callback([5.4, 7.6])
        self.assertTrue(np.allclose(transformed, [5, 8]))

    def test_transform_callback_log(self):
        exp = get_branin_experiment()
        parameters = [
            RangeParameter(
                name="x1",
                parameter_type=ParameterType.FLOAT,
                lower=1,
                upper=3,
                log_scale=True,
            ),
            RangeParameter(
                name="x2",
                parameter_type=ParameterType.FLOAT,
                lower=1,
                upper=3,
                log_scale=True,
            ),
        ]
        gpei = NumpyGenerator(
            experiment=exp,
            search_space=SearchSpace(parameters=parameters),
            model=GPyGP(),
            transforms=[Log],
        )
        transformed = gpei._transform_callback([1.2, 2.5])
        self.assertTrue(np.allclose(transformed, [1.2, 2.5]))

    def test_transform_callback_unitx(self):
        exp = get_branin_experiment()
        parameters = [
            RangeParameter(
                name="x1", parameter_type=ParameterType.FLOAT, lower=0, upper=10
            ),
            RangeParameter(
                name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=100
            ),
        ]
        gpei = NumpyGenerator(
            experiment=exp,
            search_space=SearchSpace(parameters=parameters),
            model=GPyGP(),
            transforms=[UnitX],
        )
        transformed = gpei._transform_callback([0.75, 0.35])
        self.assertTrue(np.allclose(transformed, [0.75, 0.35]))

    def test_transform_callback_int_log(self):
        exp = get_branin_experiment()
        parameters = [
            RangeParameter(
                name="x1",
                parameter_type=ParameterType.INT,
                lower=1,
                upper=100,
                log_scale=True,
            ),
            RangeParameter(
                name="x2",
                parameter_type=ParameterType.INT,
                lower=1,
                upper=100,
                log_scale=True,
            ),
        ]
        gpei = NumpyGenerator(
            experiment=exp,
            search_space=SearchSpace(parameters=parameters),
            model=GPyGP(),
            transforms=[IntToFloat, Log],
        )
        transformed = gpei._transform_callback([0.5, 1.5])
        self.assertTrue(np.allclose(transformed, [0.47712, 1.50515]))

    def test_transform_callback_int_unitx(self):
        exp = get_branin_experiment()
        parameters = [
            RangeParameter(
                name="x1", parameter_type=ParameterType.INT, lower=0, upper=10
            ),
            RangeParameter(
                name="x2", parameter_type=ParameterType.INT, lower=0, upper=100
            ),
        ]
        gpei = NumpyGenerator(
            experiment=exp,
            search_space=SearchSpace(parameters=parameters),
            model=GPyGP(),
            transforms=[IntToFloat, UnitX],
        )
        transformed = gpei._transform_callback([0.75, 0.35])
        self.assertTrue(np.allclose(transformed, [0.8, 0.35]))
