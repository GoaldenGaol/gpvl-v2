"""Volitional state representations: V(t), Ξ(t), and entity snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from volition.constants import (
    DIM4_BAND_HIGH_TFR,
    DIM4_BAND_MID_TFR,
    DIM4_IRREVERSIBLE,
    DIM4_NO_RETURN,
)
from volition.hazard import hazard_of_collapse
from volition.regime import Regime, classify_regime

EntityKind = Literal["country", "firm"]


@dataclass
class VolitionalVector:
    """
    7-dimensional volitional state vector V(t) = [dim1..dim7].

    Extracted from narrative-data embeddings (frozen Nov 2023).
    dim4 is the active latent scalar for fertility and cooperation prediction.
    """

    dim1: float = 0.0
    dim2: float = 0.0
    dim3: float = 0.0
    dim4: float = 0.0
    dim5: float = 0.0
    dim6: float = 0.0
    dim7: float = 0.0

    @property
    def as_array(self) -> NDArray[np.float64]:
        return np.array(
            [self.dim1, self.dim2, self.dim3, self.dim4,
             self.dim5, self.dim6, self.dim7],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> VolitionalVector:
        if arr.shape != (7,):
            raise ValueError(f"Expected 7-D vector, got shape {arr.shape}")
        return cls(*arr.tolist())

    def __repr__(self) -> str:
        vals = ", ".join(f"{v:.3f}" for v in self.as_array)
        return f"VolitionalVector([{vals}])"


@dataclass
class FullState:
    """
    Full volitional system state Ξ(t) = (P, C, B, X, Ω, I).

    Phase 2 provides the structural container; component dynamics
    are wired in Phase 3 (equations) and Phase 4 (VPDE).
    """

    property_aggregate: float = 0.0
    competence_mean: float = 0.0
    belief_entropy: float = 0.0
    action_set_size: int = 0
    cooperation_density: float = 0.0
    institutional_drift: float = 0.0
    volitional_vector: VolitionalVector = field(default_factory=VolitionalVector)

    @property
    def dim4(self) -> float:
        return self.volitional_vector.dim4


@dataclass(frozen=True)
class StateSummary:
    """Structured summary of a volitional entity at a point in time."""

    name: str
    year: int
    dim4: float
    regime: Regime
    hazard: float
    tfr_band: str
    entity_kind: EntityKind
    tfr: float | None = None
    sector: str | None = None
    region: str | None = None
    collapsed: bool | None = None

    @property
    def is_irreversible(self) -> bool:
        return self.dim4 > DIM4_IRREVERSIBLE

    @property
    def violates_no_return(self) -> bool:
        return self.is_irreversible and self.dim4 < DIM4_NO_RETURN

    def format(self, *, verbose: bool = False) -> str:
        parts = [
            f"{self.name} ({self.year})",
            f"dim4={self.dim4:.2f}",
            f"regime={self.regime.name}",
            f"band={self.tfr_band}",
            f"hazard={self.hazard:.3f}",
        ]
        if self.tfr is not None:
            parts.insert(2, f"TFR={self.tfr:.2f}")
        if verbose:
            parts.append(f"kind={self.entity_kind}")
            if self.sector:
                parts.append(f"sector={self.sector}")
            if self.region:
                parts.append(f"region={self.region}")
            if self.collapsed is not None:
                parts.append(f"collapsed={self.collapsed}")
            parts.append(f"irreversible={self.is_irreversible}")
            if self.violates_no_return:
                parts.append("INVARIANT_VIOLATION")
        return " — ".join(parts)


def tfr_band(dim4: float) -> str:
    """Classify dim4 into TFR regime band (UNIFIED-THEORY L2)."""
    if dim4 <= DIM4_BAND_HIGH_TFR:
        return "HIGH_TFR"
    if dim4 <= DIM4_BAND_MID_TFR:
        return "MID_TFR"
    return "LOW_TFR"


def build_summary(
    name: str,
    year: int,
    dim4: float,
    *,
    tfr: float | None = None,
    entity_kind: EntityKind = "country",
    sector: str | None = None,
    region: str | None = None,
    collapsed: bool | None = None,
) -> StateSummary:
    """Build a structured StateSummary for any volitional entity."""
    return StateSummary(
        name=name,
        year=year,
        dim4=dim4,
        regime=classify_regime(dim4),
        hazard=hazard_of_collapse(dim4),
        tfr_band=tfr_band(dim4),
        entity_kind=entity_kind,
        tfr=tfr,
        sector=sector,
        region=region,
        collapsed=collapsed,
    )


@dataclass
class VolitionalState:
    """Country/year volitional state with optional TFR and full vector."""

    name: str
    year: int
    dim4: float
    tfr: float | None = None
    vector: VolitionalVector | None = field(default=None, repr=False)

    def summary(self) -> StateSummary:
        return build_summary(
            self.name, self.year, self.dim4, tfr=self.tfr, entity_kind="country"
        )

    def summarize(self, *, verbose: bool = False) -> str:
        return self.summary().format(verbose=verbose)


@dataclass
class FirmVolitionalState:
    """Firm-level volitional state for fatal-pivot / collapse modeling."""

    name: str
    year: int
    dim4: float
    sector: str
    region: str
    collapsed: bool
    notes: str = ""

    def summary(self) -> StateSummary:
        return build_summary(
            self.name,
            self.year,
            self.dim4,
            entity_kind="firm",
            sector=self.sector,
            region=self.region,
            collapsed=self.collapsed,
        )

    def summarize(self, *, verbose: bool = False) -> str:
        return self.summary().format(verbose=verbose)

    @classmethod
    def from_row(cls, row: object) -> FirmVolitionalState:
        """Construct from a pandas Series or dict-like row."""
        return cls(
            name=str(row["firm"]),  # type: ignore[index]
            year=int(row["year"]),  # type: ignore[index]
            dim4=float(row["dim4_firm"]),  # type: ignore[index]
            sector=str(row["sector"]),  # type: ignore[index]
            region=str(row["region"]),  # type: ignore[index]
            collapsed=bool(int(row["y_pivot_5yr"])),  # type: ignore[index]
            notes=str(row.get("notes", "") if hasattr(row, "get") else row["notes"]),  # type: ignore[index]
        )