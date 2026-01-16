from __future__ import annotations

from typing import List

from .sinq import Sinq

"""
Factory / builder functions for constructing Sinq objects.

These functions intentionally live outside the Sinq class to keep
construction logic separate from data behavior.
"""


def build_composite_sinq(
    name: str,
    sources: list[Sinq],
    *,
    overwrite: bool = True,
    fresh: bool = True,
) -> Sinq:
    """
    Build a composite Sinq from source Sinqs.
    """
    sinq = Sinq(sinqname=name, fresh=fresh)

    for src in sources:
        sinq.merge_sinq(src, overwrite=overwrite)

    return sinq