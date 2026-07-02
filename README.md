# GPVL v2 — General Physics of Volition

**Frozen 2023 model + axiomatic theory + VPDE dynamics**  
License: CC-BY-4.0 | Python ≥ 3.11

An open, reproducible Python framework for cooperative equilibria, dim4 regime detection, and volitional PDE modeling. Built on the [Science of Volition](https://github.com/GoaldenGaol/science-of-volition) framework.

## Abstract

The volitional state vector **V(t) = [dim1…dim7]** is extracted from narrative-data embeddings. The fourth component **dim4** predicts cross-national fertility 18 years later (Pearson *r* ≈ −0.93) and identifies cooperation-collapse thresholds at **+0.92** and irreversibility at **+1.00**. GPVL v2 implements these mappings as testable, extensible Python modules with symbolic equations, MCMC inference, and octonion/G2 geometry hooks.

## Quick Start

```bash
git clone https://github.com/GoaldenGaol/gpvl-v2.git
cd gpvl-v2
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"
pytest tests/ -v
```

## CLI

After install, the `gpvl` command is available:

```bash
gpvl info                          # project + dataset summary
gpvl validate                      # dim4 → TFR validation stats
gpvl validate --json               # machine-readable output
gpvl export-latex                  # full arXiv bundle → docs/arxiv/
gpvl export-latex -o build/arxiv   # custom output directory
gpvl export-latex --equations-only # equations.tex + thresholds.tex only
gpvl invariants                    # check empirical invariants A–E
```

### Symbolic equations and Φ transition map

```python
from volition.equations import (
    phi_transition_equations,
    evaluate_phi,
    all_axioms,
    check_all_invariants,
)
import numpy as np

# 7-D transition map V(t+1) = Φ(V(t), i_t, u(t))
print(phi_transition_equations()[3])  # dim4 component

state = np.array([0, 0, 0, 0.85, 0, 0, 0])
print(evaluate_phi(state, rho_plunder=0.03))

print(list(all_axioms().keys()))  # axiom_1 … axiom_8
```

Equivalent module invocation:

```bash
python -m volition.cli export-latex --bundle
```

## Usage Examples

### Regime classification

```python
from volition import classify_regime, hazard_of_collapse, VolitionalState

state = VolitionalState(name="South Korea", year=2023, dim4=1.41, tfr=0.72)
print(state.summarize())
print(state.summarize(verbose=True))  # includes band, kind, irreversible flag

print(classify_regime(0.95))       # Regime.PRE_COLLAPSE
print(hazard_of_collapse(1.41))    # ~0.73
```

### Regime transition matrix Θ

```python
from volition import Regime, RegimeTransitionMatrix

theta = RegimeTransitionMatrix.default()
chain = theta.simulate(Regime.COOPERATIVE, n_steps=20, seed=42)
print(chain)  # Markov regime path
print(theta.stationary_distribution())
```

### Firm-level fatal pivot

```python
from volition.data import load_firm_states, firm_validation_stats, classify_firm_risk

firms = load_firm_states()
stats = firm_validation_stats()
print(f"ROC-AUC = {stats['roc_auc']:.3f}")
print(classify_firm_risk(1.20))  # CRITICAL
```

### dim4 data validation

```python
from volition.data import load_countries, validation_stats

df = load_countries()
stats = validation_stats(df)
print(f"Pearson r = {stats['pearson_r']:.3f}, R² = {stats['r_squared']:.3f}")
```

### Goalden universal threshold

```python
from volition.goalden import GoaldenThreshold, EquilibriumPrediction

gt = GoaldenThreshold(rho_plunder=0.0312, c_mean=1.28, r_mean=0.91)
print(f"threshold = {gt.threshold:.5f}")  # 0.11648
assert gt.predict() == EquilibriumPrediction.COOPERATIVE
```

### VPDE dynamics

```python
from volition.vpde import solve_vpde

t, dim4_traj = solve_vpde(dim4_initial=0.98)
```

### MCMC fertility model fit

```python
from volition.mcmc import fit_fertility_model

params = fit_fertility_model(n_steps=500, burn_in=100)
print(f"alpha={params.alpha:.3f}, beta={params.beta:.3f}, r={params.pearson_r:.3f}")
```

### LaTeX export

```python
from volition.export.latex import write_arxiv_bundle

result = write_arxiv_bundle("docs/arxiv")
print(f"Main document: {result.main_tex}")
# Compile: pdflatex main.tex  (from docs/arxiv/)
```

Or via CLI:

```bash
gpvl export-latex
```

## Simulations

```bash
python simulations/cooperation_collapse.py
python simulations/fertility_forecast.py
python simulations/regime_transition.py   # saves plot to simulations/output/
```

## Project Structure

```
gpvl-v2/
├── src/volition/          # Core library
│   ├── equations.py       # sympy symbolic core
│   ├── vpde.py            # 1D VPDE solver
│   ├── regime.py          # dim4 regime classification
│   ├── goalden.py         # ρ_plunder threshold
│   ├── mcmc/              # emcee parameter inference
│   ├── geometry/          # octonion/G2 stubs (v2.1 calibration)
│   └── export/            # LaTeX fragments
├── data/                  # Frozen 2023 dim4 CSVs
├── docs/                  # CORE-FRAMEWORK.md, UNIFIED-THEORY.md
├── tests/                 # pytest suite
└── simulations/           # Runnable demos
```

## dim4 Quick Reference

| Threshold | Meaning | Reversals observed |
|-----------|---------|-------------------|
| > +0.92 | Cooperation collapses (<40 rounds) | None |
| > +1.00 | Irreversible demographic band | Zero (1950–2025) |

2023 values: South Korea +1.41 · Japan +1.33 · Italy +1.22 · USA crosses +1.00 in 2029±1

## arXiv Submission Block

```latex
\title{The Science of Volition: A Testable Framework for Cooperative Equilibria}
\author{GoaldenGaol}
\begin{abstract}
We present an open framework in which a seven-dimensional volitional state
vector $V(t)$, extracted from narrative embeddings, predicts cross-national
fertility and cooperation collapse through a single latent scalar $\mathrm{dim4}$.
Empirical validation on 187 countries (1950--2023) yields Pearson
$r = -0.934$ for $\mathrm{TFR}(t{+}18) = f(\mathrm{dim4}(t))$.
Thresholds at $\mathrm{dim4} > 0.92$ (cooperation collapse) and
$\mathrm{dim4} > 1.00$ (irreversibility) show zero historical reversals.
\end{abstract}
```

Generate full arXiv manuscript fragments:

```bash
gpvl export-latex
# writes docs/arxiv/main.tex, equations.tex, thresholds.tex, validation.tex, ...
```

## Core Documents

- [CORE-FRAMEWORK.md](docs/CORE-FRAMEWORK.md) — complete axiomatic packet
- [UNIFIED-THEORY.md](docs/UNIFIED-THEORY.md) — dim4 systems architecture
- [empirical-validation.md](docs/empirical-validation.md) — validation statistics

## Citation

```bibtex
@software{gpvl_v2_2026,
  author  = {GoaldenGaol},
  title   = {GPVL v2: General Physics of Volition},
  year    = {2026},
  url     = {https://github.com/GoaldenGaol/gpvl-v2},
  license = {CC-BY-4.0}
}
```

## License

CC-BY-4.0 — see [LICENSE](LICENSE). Framework docs vendored from [science-of-volition](https://github.com/GoaldenGaol/science-of-volition).