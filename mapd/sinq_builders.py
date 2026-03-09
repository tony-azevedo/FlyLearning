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


def subset_sinq(
    source: Sinq,
    *,
    rows: list[str],
    name: str,
    fresh: bool = True,
) -> Sinq:
    """
    Create a new Sinq containing only a subset of rows from a source Sinq.
    """
    sinq = Sinq(sinqname=name, fresh=fresh)

    # Copy only selected rows
    sinq.df = source.df.loc[rows].copy()

    # Optional provenance
    sinq._sources = [source]
    sinq._subset_rows = rows

    sinq.save()
    return sinq