#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

from logging import Logger
from typing import Dict, List

from ax.utils.common.logger import get_logger


logger: Logger = get_logger(__name__)


def _get_suffix(input_str: str, delim: str = ".", n_chunks: int = 1) -> str:
    return delim.join(input_str.split(delim)[-n_chunks:])


def _get_shortest_unique_suffix_dict(
    input_str_list: List[str], delim: str = "."
) -> Dict[str, str]:
    """Maps a list of strings to their shortest unique suffixes

    Maps all original strings to the smallest number of chunks, as specified by
    delim, that are not a suffix of any other original string. If the original
    string was a suffix of another string, map it to its unaltered self.

    Args:
        input_str_list: a list of strings to create the suffix mapping for
        delim: the delimiter used to split up the strings into meaningful chunks

    Returns:
        dict: A dict with the original strings as keys and their abbreviations as
            values
    """

    # all input strings must be unique
    assert len(input_str_list) == len(set(input_str_list))
    if delim == "":
        raise ValueError("delim must be a non-empty string.")
    suffix_dict = defaultdict(list)
    # initialize suffix_dict with last chunk
    for istr in input_str_list:
        suffix_dict[_get_suffix(istr, delim=delim, n_chunks=1)].append(istr)
    max_chunks = max(len(istr.split(delim)) for istr in input_str_list)
    if max_chunks == 1:
        return {istr: istr for istr in input_str_list}
    # the upper range of this loop is `max_chunks + 2` because:
    #     - `i` needs to take the value of `max_chunks`, hence one +1
    #     - the contents of the loop are run one more time to check if `all_unique`,
    #           hence the other +1
    for i in range(2, max_chunks + 2):
        new_dict = defaultdict(list)
        all_unique = True
        for suffix, suffix_str_list in suffix_dict.items():
            if len(suffix_str_list) > 1:
                all_unique = False
                for istr in suffix_str_list:
                    new_dict[_get_suffix(istr, delim=delim, n_chunks=i)].append(istr)
            else:
                new_dict[suffix] = suffix_str_list
        if all_unique:
            if len(set(input_str_list)) != len(suffix_dict.keys()):
                break
            return {
                suffix_str_list[0]: suffix
                for suffix, suffix_str_list in suffix_dict.items()
            }
        suffix_dict = new_dict
    # If this function has not yet exited, some input strings still share a suffix.
    # This is not expected, but in this case, the function will return the identity
    # mapping, i.e., a dict with the original strings as both keys and values.
    logger.warning(
        "Something went wrong. Returning dictionary with original strings as keys and "
        "values."
    )
    return {istr: istr for istr in input_str_list}
