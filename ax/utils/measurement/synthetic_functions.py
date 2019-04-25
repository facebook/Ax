#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from ax.utils.common.docutils import copy_doc
from ax.utils.common.typeutils import checked_cast, not_none


def informative_failure_on_none(func: Callable) -> Any:
    def function_wrapper(*args, **kwargs) -> Any:
        res = func(*args, **kwargs)
        if res is None:
            raise NotImplementedError(
                f"{args[0].__class__.__name__} does not specify property "
                f'"{func.__name__}".'
            )
        return not_none(res)

    return function_wrapper


class SyntheticFunction(ABC):

    _required_dimensionality = None
    _domain = None
    _minimums = None
    _maximums = None
    _fmin = None
    _fmax = None

    @property
    @informative_failure_on_none
    def name(self) -> str:
        return f"{self.__class__.__name__}"

    def __call__(
        self,
        *args: Union[int, float, np.ndarray],
        **kwargs: Union[int, float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Simplified way to call the synthetic function and pass the argument
        numbers directly, e.g. `branin(2.0, 3.0)`.
        """
        if kwargs:
            if self.required_dimensionality:
                assert (
                    len(kwargs) == self.required_dimensionality or len(kwargs) == 1
                ), (
                    f"Function {self.name} expected either "
                    f"{self.required_dimensionality} arguments "
                    "or a single numpy array argument."
                )
            assert not args, (
                f"Function {self.name} expected either all anonymous "
                "arguments or all keyword arguments."
            )
            args = list(kwargs.values())  # pyre-ignore[9]
        for x in args:
            if isinstance(x, np.ndarray):
                return self.f(X=x)
            assert isinstance(
                x, (int, float, np.int64, np.float64)
            ), f"Expected numerical arguments or numpy arrays, got {type(x)}."
        return checked_cast(float, self.f(np.array(args)))

    def f(self, X: np.ndarray) -> Union[float, np.ndarray]:
        """Synthetic function implementation.

        Args:
            X (numpy.ndarray): an n by d array, where n represents the number
                of observations and d is the dimensionality of the inputs.

        Returns:
            numpy.ndarray: an n-dimensional array.
        """
        assert isinstance(X, np.ndarray), "X must be a numpy (nd)array."
        if self.required_dimensionality:
            if len(X.shape) == 1:
                input_dim = X.shape[0]
            elif len(X.shape) == 2:
                input_dim = X.shape[1]
            else:
                raise ValueError(
                    "Synthetic function call expects input of either 1-d array or "
                    "n by d array, where n is number of observations and d is "
                    "dimensionality of the input."
                )
            assert input_dim == self.required_dimensionality, (
                f"Input violates required dimensionality of {self.name}: "
                f"{self.required_dimensionality}."
            )
        if len(X.shape) == 1:
            return self._f(X=X)
        else:
            return np.array([self._f(X=x) for x in X])

    @property
    @informative_failure_on_none
    def required_dimensionality(self) -> Optional[int]:
        """Required dimensionality of input to this function."""
        return self._required_dimensionality

    @property
    @informative_failure_on_none
    def domain(self) -> List[Tuple[float, ...]]:
        """Domain on which function is evaluated.

        The list is of the same length as the dimensionality of the inputs,
        where each element of the list is a tuple corresponding to the min
        and max of the domain for that dimension.
        """
        return self._domain  # pyre-ignore[7]: will not be None b/c decorator

    @property
    @informative_failure_on_none
    def minimums(self) -> List[Tuple[float, ...]]:
        """List of global minimums.

        Each element of the list is a d-tuple, where d is the dimensionality
        of the inputs. There may be more than one global minimums.
        """
        return self._minimums  # pyre-ignore[7]: will not be None b/c decorator

    @property
    @informative_failure_on_none
    def maximums(self) -> List[Tuple[float, ...]]:
        """List of global minimums.

        Each element of the list is a d-tuple, where d is the dimensionality
        of the inputs. There may be more than one global minimums.
        """
        return self._maximums  # pyre-ignore[7]: will not be None b/c decorator

    @property
    @informative_failure_on_none
    def fmin(self) -> float:
        """Value at global minimum(s)."""
        return self._fmin  # pyre-ignore[7]: will not be None b/c decorator

    @property
    @informative_failure_on_none
    def fmax(self) -> float:
        """Value at global minimum(s)."""
        return self._fmax  # pyre-ignore[7]: will not be None b/c decorator

    @classmethod
    @abstractmethod
    def _f(self, X: np.ndarray) -> float:
        """Implementation of the synthetic function. Must be implemented in subclass.

        Args:
            X (numpy.ndarray): an n by d array, where n represents the number
                of observations and d is the dimensionality of the inputs.

        Returns:
            numpy.ndarray: an n-dimensional array.
        """
        ...  # pragma: no cover


class Hartmann6(SyntheticFunction):
    """Hartmann6 function (6-dimensional with 1 global minimum)."""

    _required_dimensionality = 6
    _domain = [(0, 1) for i in range(6)]
    _minimums = [(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)]
    _fmin = -3.32237
    _fmax = 0.0

    @copy_doc(SyntheticFunction._f)
    def _f(self, X: np.ndarray) -> float:
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array(
            [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
        )
        P = 10 ** (-4) * np.array(
            [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        )
        y = 0.0
        for j, alpha_j in enumerate(alpha):
            t = 0
            for k in range(6):
                t += A[j, k] * ((X[k] - P[j, k]) ** 2)
            y -= alpha_j * np.exp(-t)
        return float(y)


class Branin(SyntheticFunction):
    """Branin function (2-dimensional with 3 global minima)."""

    _required_dimensionality = 2
    _domain = [(-5, 10), (0, 15)]
    _minimums = [(-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)]
    _fmin = 0.397887
    _fmax = 294.0

    @copy_doc(SyntheticFunction._f)
    def _f(self, X: np.ndarray) -> float:
        x_1 = X[0]
        x_2 = X[1]
        return float(
            (x_2 - 5.1 / (4 * np.pi ** 2) * x_1 ** 2 + 5.0 / np.pi * x_1 - 6.0) ** 2
            + 10 * (1 - 1.0 / (8 * np.pi)) * np.cos(x_1)
            + 10
        )


hartmann6 = Hartmann6()
branin = Branin()
