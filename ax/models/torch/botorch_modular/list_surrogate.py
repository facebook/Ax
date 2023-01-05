# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


raise DeprecationWarning(  # pragma: no cover
    "`ListSurrogate` has been deprecated. Please use Surrogate instead, which may now "
    "utilize individual BoTorch models per outcome that will get wrapped into a "
    "ModelList as in ListSurrogate."
)


class ListSurrogate:  # pragma: no cover
    pass
