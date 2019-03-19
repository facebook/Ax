#!/usr/bin/env python3


def is_foreign_key_field(field: str) -> bool:
    """Return true if field name is a foreign key field, i.e. ends in `_id`."""
    return len(field) > 3 and field[-3:] == "_id"
