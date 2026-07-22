"""Overlay ABC + registry for the Trial Browser.

An Overlay renders supplementary artists (firing rate, movement bouts, …)
on top of the probe/ephys axes the browser already owns. The browser
calls ``clear()`` on every overlay before every trial switch and then
``draw(trial, axes)`` on overlays whose checkbox is active.

Overlays are registered by decorator:

    @register_overlay
    class FiringRateOverlay(Overlay):
        name = "Firing rate"
        def draw(self, trial, axes): ...

``axes`` is a dict ``{"probe": ax_probe, "ephys": ax_ephys, "fig": figure}``.
Overlays are responsible for tracking every artist they create in
``self._artists`` so ``clear()`` can remove them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..trial import Trial


_OVERLAY_REGISTRY: "dict[str, type[Overlay]]" = {}


def register_overlay(cls: "type[Overlay]") -> "type[Overlay]":
    """Register an Overlay subclass by its ``name`` class attribute."""
    if not getattr(cls, "name", None):
        raise ValueError(
            f"{cls.__name__} must define a non-empty class attribute `name`"
        )
    if cls.name in _OVERLAY_REGISTRY and _OVERLAY_REGISTRY[cls.name] is not cls:
        # Allow re-registration on module reload (IPython autoreload), but
        # surface genuine name collisions between different classes.
        existing = _OVERLAY_REGISTRY[cls.name]
        if existing.__module__ != cls.__module__:
            raise ValueError(
                f"Overlay name {cls.name!r} already registered by {existing!r}"
            )
    _OVERLAY_REGISTRY[cls.name] = cls
    return cls


def available_overlays() -> "dict[str, type[Overlay]]":
    """Shallow copy of the registry (name -> subclass)."""
    return dict(_OVERLAY_REGISTRY)


class Overlay(ABC):
    """Base class for browser overlays.

    Subclasses set ``name`` (used as the UI label and registry key) and
    implement ``draw(trial, axes)``. ``clear()`` is provided and removes
    every artist that was appended to ``self._artists``.
    """

    name: str = ""  # subclasses override

    def __init__(self):
        self._artists: list = []

    @abstractmethod
    def draw(self, trial: "Trial", axes: dict) -> None:
        """Render overlay artists for ``trial`` into the provided axes.

        Implementations should append every created artist to
        ``self._artists`` so ``clear()`` can later remove them.
        """

    def clear(self) -> None:
        """Remove all artists created by the last ``draw()`` call."""
        for art in self._artists:
            try:
                art.remove()
            except (ValueError, NotImplementedError, AttributeError):
                # Artist no longer attached — nothing to do.
                pass
        self._artists.clear()
