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

    #: Drawing parameters a subclass understands, with their defaults. The
    #: browser pushes shared settings (e.g. the smoothing kernel) to every
    #: overlay through ``set_params`` before each draw; keys an overlay does not
    #: declare here are simply stored and ignored, so one control can address
    #: several overlays without knowing which of them care.
    DEFAULT_PARAMS: dict = {}

    def __init__(self):
        self._artists: list = []
        self.params: dict = dict(self.DEFAULT_PARAMS)

    def set_params(self, **params) -> None:
        """Merge ``params`` into ``self.params``. Takes effect on the next draw."""
        self.params.update(params)

    @abstractmethod
    def draw(self, trial: "Trial", axes: dict) -> None:
        """Render overlay artists for ``trial`` into the provided axes.

        Implementations should append every created artist to
        ``self._artists`` so ``clear()`` can later remove them.
        """

    def shade_kernel_edges(self, trial, axes, sigma_s, n_sigma=3.0):
        """Shade where a kernel of sd ``sigma_s`` lacks full support.

        There are no spikes before the recording starts, so any smoothed rate
        necessarily climbs out of zero over the first few sigma and falls back at
        the end — at sigma = 25 ms the first sample can read half the true rate.
        The analysis pipeline trims these samples away; the browser shows the
        whole trial, so here they get marked instead.

        Only the first overlay to call this in a given draw actually shades, via a
        flag left in the ``axes`` dict — the browser rebuilds that dict for every
        draw, so the flag resets on its own and two overlays asking for shading
        cannot double the alpha.
        """
        if axes.get("_kernel_edges_shaded") or not sigma_s:
            return
        axes["_kernel_edges_shaded"] = True
        import numpy as np

        t = np.asarray(trial.time).ravel()
        if t.size < 2:
            return
        edge = float(n_sigma) * float(sigma_s)
        for ax in (axes.get("probe"), axes.get("ephys")):
            if ax is None:
                continue
            for a, b in ((t[0], t[0] + edge), (t[-1] - edge, t[-1])):
                span = ax.axvspan(a, b, color="#d62728", alpha=0.07, lw=0,
                                  zorder=0)
                self._artists.append(span)

    def clear(self) -> None:
        """Remove all artists created by the last ``draw()`` call."""
        for art in self._artists:
            try:
                art.remove()
            except (ValueError, NotImplementedError, AttributeError):
                # Artist no longer attached — nothing to do.
                pass
        self._artists.clear()
