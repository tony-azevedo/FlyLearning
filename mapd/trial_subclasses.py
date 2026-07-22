"""Per-protocol Trial subclasses.

Each subclass opts in to the protocol registry via ``@Trial.register_protocol``.
The base ``Trial`` implementation already handles probe + ephys rendering in
``draw_in_browser``; subclasses override ``draw_in_browser`` only when they
need protocol-specific overlays (e.g. LED onset markers, piezo cue shading).

To extend: define a new subclass in this module, decorate with
``@Trial.register_protocol("ProtocolName", "OlderNameForSameProtocol")``,
and override ``draw_in_browser`` if needed. ``Trial.for_path(path)`` will
return the subclass for any trial file matching the registered protocol.
"""
from __future__ import annotations

from .trial import Trial


@Trial.register_protocol(
    "LEDFlashTriggerPiezoControl",
    "LEDFlashWithPiezoCueControl",
)
class LEDFlashPiezoTrial(Trial):
    """Trial with an LED-flash cue and a piezo-controlled probe.

    Covers both the current protocol name (``LEDFlashTriggerPiezoControl``)
    and the earlier variant (``LEDFlashWithPiezoCueControl``). The two
    were the same experiment under different naming conventions.
    """
    # Phase-1 identity subclass: inherits draw_in_browser from Trial.
    # Override this method to add LED-onset markers, cue-window shading,
    # etc. — remember to clean up any extra artists when the trial changes
    # (the base hook only mutates the artists the browser provides).
