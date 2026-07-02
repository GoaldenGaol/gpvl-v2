"""Regime transition simulation using VPDE dim4 dynamics."""

import matplotlib.pyplot as plt

from volition.regime import classify_regime
from volition.vpde import solve_vpde


def main() -> None:
    initial_values = [0.75, 0.90, 0.98, 1.10]
    fig, ax = plt.subplots(figsize=(8, 5))

    for v0 in initial_values:
        t, trajectory = solve_vpde(v0)
        ax.plot(t, trajectory, label=f"dim4₀={v0:.2f}")

    ax.axhline(0.92, color="orange", linestyle="--", label="coop collapse (+0.92)")
    ax.axhline(1.00, color="red", linestyle="--", label="irreversible (+1.00)")
    ax.set_xlabel("Time t")
    ax.set_ylabel("dim4(t)")
    ax.set_title("VPDE Regime Transition (1D)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = "simulations/output"
    import os
    os.makedirs(out, exist_ok=True)
    fig.savefig(f"{out}/regime_transition.png", dpi=150)
    print(f"Saved plot to {out}/regime_transition.png")

    for v0 in initial_values:
        final_regime = classify_regime(float(solve_vpde(v0)[1][-1]))
        print(f"dim4₀={v0:.2f} → final regime: {final_regime.name}")


if __name__ == "__main__":
    main()