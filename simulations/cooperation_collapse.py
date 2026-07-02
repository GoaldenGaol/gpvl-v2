"""Simulate cooperation collapse at dim4 > 0.92 threshold."""

from volition.constants import DIM4_COOP_COLLAPSE
from volition.hazard import hazard_of_collapse
from volition.regime import Regime, classify_regime

THRESHOLD_ROUNDS = 40


def simulate_rounds(dim4: float, n_rounds: int = 100) -> list[Regime]:
    """Return regime at each round (static dim4 model for Phase 0)."""
    regime = classify_regime(dim4)
    return [regime] * n_rounds


def main() -> None:
    test_values = [0.80, 0.93, 1.05, 1.41]
    print(f"Cooperation collapse threshold: dim4 > {DIM4_COOP_COLLAPSE}")
    print(f"Collapse expected in < {THRESHOLD_ROUNDS} rounds above threshold\n")

    for dim4 in test_values:
        regime = classify_regime(dim4)
        hazard = hazard_of_collapse(dim4)
        collapsed = dim4 > DIM4_COOP_COLLAPSE
        print(
            f"dim4={dim4:.2f}  regime={regime.name:14s}  "
            f"hazard={hazard:.3f}  collapsed={collapsed}"
        )


if __name__ == "__main__":
    main()