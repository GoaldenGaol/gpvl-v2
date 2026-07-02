"""Regime transition simulation: VPDE dim4 dynamics + Markov Θ."""

import os

import matplotlib.pyplot as plt

from volition.regime import Regime, RegimeTransitionMatrix, classify_regime
from volition.vpde import solve_vpde


def main() -> None:
    initial_values = [0.75, 0.90, 0.98, 1.10]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: VPDE continuous dim4 trajectories
    ax = axes[0]
    for v0 in initial_values:
        t, trajectory = solve_vpde(v0)
        ax.plot(t, trajectory, label=f"dim4₀={v0:.2f}")

    ax.axhline(0.92, color="orange", linestyle="--", label="coop collapse (+0.92)")
    ax.axhline(1.00, color="red", linestyle="--", label="irreversible (+1.00)")
    ax.set_xlabel("Time t")
    ax.set_ylabel("dim4(t)")
    ax.set_title("VPDE dim4 Dynamics (1D)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Markov Θ regime chain
    theta = RegimeTransitionMatrix.default()
    ax2 = axes[1]
    regime_map = {Regime.COOPERATIVE: 0, Regime.PRE_COLLAPSE: 1, Regime.IRREVERSIBLE: 2}
    for start_dim4 in initial_values:
        start_regime = classify_regime(start_dim4)
        chain = theta.simulate(start_regime, n_steps=30, seed=int(start_dim4 * 100))
        y = [regime_map[r] for r in chain]
        ax2.plot(y, label=f"start={start_regime.name[:4]}")

    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["COOP", "PRE", "IRREV"])
    ax2.set_xlabel("Step")
    ax2.set_title("Markov Θ Regime Chain")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    out = "simulations/output"
    os.makedirs(out, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{out}/regime_transition.png", dpi=150)
    print(f"Saved plot to {out}/regime_transition.png")
    print(f"Stationary π = {theta.stationary_distribution()}")

    for v0 in initial_values:
        final_regime = classify_regime(float(solve_vpde(v0)[1][-1]))
        print(f"dim4₀={v0:.2f} → VPDE final regime: {final_regime.name}")


if __name__ == "__main__":
    main()