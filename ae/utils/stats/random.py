#!/usr/bin/env python3

import hashlib


def salt_to_seed(salt):
    return int(hashlib.md5(salt.encode("utf-8")).hexdigest()[:-7], 16) % 1000001
