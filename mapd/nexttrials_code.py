from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Type, Any, Optional
import numpy as np

# ---------- Base Class & Registry ----------

@dataclass
class Trial:
    data: np.ndarray                 # timeseries (T x C or T,)
    meta: Dict[str, Any]             # metadata from HDF5 / elsewhere
    sampling_rate_hz: float

    # Class-wide registry of subclasses by protocol name
    _REGISTRY: Dict[str, Type["Trial"]] = {}

    PROTOCOL: Optional[str] = None   # subclasses set this

    # ---- Registry utilities ----
    @classmethod
    def register(cls, subclass: Type["Trial"]) -> Type["Trial"]:
        """Decorator to register a subclass by its PROTOCOL name."""
        if not subclass.PROTOCOL:
            raise ValueError(f"{subclass.__name__} must define PROTOCOL")
        cls._REGISTRY[subclass.PROTOCOL] = subclass
        return subclass

    # ---- Factory / dispatch ----
    @classmethod
    def from_record(cls, data: np.ndarray, meta: Dict[str, Any]) -> "Trial":
        """
        Factory: look at meta['protocol'] (or similar) and dispatch to the right subclass.
        """
        protocol = meta.get("protocol")
        if protocol in cls._REGISTRY:
            return cls._REGISTRY[protocol].from_record(data, meta)  # late-binding to subclass
        # Fallback: if unknown, construct base (or raise)
        return cls(data=data, meta=meta, sampling_rate_hz=cls.default_sampling_rate(meta))

    # ---- Alternative constructors ----
    @classmethod
    def from_h5(cls, h5_reader) -> "Trial":
        """
        Example: read data/meta from an HDF5 reader you control.
        Subclasses can override to read custom fields but still call super().
        """
        data = h5_reader.read_timeseries()     # shape (T,) or (T, C)
        meta = h5_reader.read_meta()           # dict
        # Dispatch to specific subclass:
        return cls.from_record(data, meta)

    # ---- Class-specific defaults (late-binding via cls) ----
    @classmethod
    def default_sampling_rate(cls, meta: Dict[str, Any]) -> float:
        """
        Late-binding: subclasses can override to change default SR based on protocol.
        """
        return float(meta.get("samplingRateHz", 10_000.0))

    # ---- Instance methods: operate on data ----
    def duration_s(self) -> float:
        return self.data.shape[0] / self.sampling_rate_hz

    def is_rest(self) -> bool:
        return bool(self.meta.get("ndf", 0) == 0)

    # ---- Static helpers: pure utilities (no self/cls) ----
    @staticmethod
    def validate_meta_keys(meta: Dict[str, Any], required: set[str]):
        missing = required - set(meta)
        if missing:
            raise ValueError(f"Missing meta keys: {sorted(missing)}")

    @staticmethod
    def detrend(x: np.ndarray) -> np.ndarray:
        t = np.arange(x.shape[0])
        # simple linear detrend; replace with scipy if you want
        A = np.vstack([t, np.ones_like(t)]).T
        m, b = np.linalg.lstsq(A, x, rcond=None)[0]
        return x - (m * t + b)


# ---------- Subclass for a specific protocol ----------

@Trial.register
@dataclass
class LEDFlashPiezoCueTrial(Trial):
    PROTOCOL: str = "LEDFlashWithPiezoCue"

    @classmethod
    def from_record(cls, data: np.ndarray, meta: Dict[str, Any]) -> "LEDFlashPiezoCueTrial":
        # Subclass-specific validation before constructing
        cls.validate_meta_keys(meta, {"protocol", "ledOnsetMs", "piezoWidth"})
        sr = cls.default_sampling_rate(meta)
        return cls(data=data, meta=meta, sampling_rate_hz=sr)

    @classmethod
    def default_sampling_rate(cls, meta: Dict[str, Any]) -> float:
        # Maybe this protocol is always recorded at 20 kHz unless overridden
        return float(meta.get("samplingRateHz", 20_000.0))

    # Protocol-specific feature
    def led_onset_samples(self) -> int:
        onset_ms = float(self.meta["ledOnsetMs"])
        return int(round(onset_ms * 1e-3 * self.sampling_rate_hz))

    # Protocol-specific decision rule
    def is_probe(self) -> bool:
        # e.g., controlToggle==0 means probe; fall back to False
        return bool(self.meta.get("controlToggle", 1) == 0)

    # Static utility specific to this protocol
    @staticmethod
    def window_around_onset(x: np.ndarray, onset_idx: int, pre: int, post: int) -> np.ndarray:
        lo = max(0, onset_idx - pre)
        hi = min(x.shape[0], onset_idx + post)
        return x[lo:hi]
