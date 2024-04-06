#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from logging import Logger
from typing import Any

from ax.utils.common.logger import get_logger

logger: Logger = get_logger(__name__)


class DeprecatedTransformMixin:
    """
    Mixin class for deprecated transforms.

    This class is used to log warnings when a deprecated transform is used,
    and will construct the new transform that should be used instead.

    The deprecated transform should inherit as follows:

    class DeprecatedTransform(DeprecatedTransformMixin, NewTransform):
        ...

    :meta private:
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Log a warning that the transform is deprecated, and construct the
        new transform.
        """
        warning_msg = self.warn_deprecated_message(
            self.__class__.__name__, type(self).__bases__[1].__name__
        )
        logger.warning(warning_msg)

        super().__init__(*args, **kwargs)

    @staticmethod
    def warn_deprecated_message(
        deprecated_transform_name: str, new_transform_name: str
    ) -> str:
        """
        Constructs the warning message.
        """
        return (
            f"`{deprecated_transform_name}` transform has been deprecated "
            "and will be removed in a future release. "
            f"Using `{new_transform_name}` instead."
        )
