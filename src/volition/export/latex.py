"""LaTeX export for equations, thresholds, and arXiv-ready fragments."""

from __future__ import annotations

from pathlib import Path

from volition.constants import DIM4_COOP_COLLAPSE, DIM4_IRREVERSIBLE, DIM4_NO_RETURN
from volition.equations import export_equations_latex


def export_threshold_table() -> str:
    """Return LaTeX tabular for dim4 thresholds."""
    return r"""\begin{table}[h]
\centering
\begin{tabular}{lll}
\toprule
Threshold & Meaning & Reversals observed \\
\midrule
$> +""" + f"{DIM4_COOP_COLLAPSE:.2f}" + r"""$ & Cooperation collapses ($<40$ rounds) & None \\
$> +""" + f"{DIM4_IRREVERSIBLE:.2f}" + r"""$ & Irreversible demographic band & Zero (1950--2025) \\
$< +""" + f"{DIM4_NO_RETURN:.2f}" + r"""$ after $>+1.0$ & No-return invariant & Zero \\
\bottomrule
\end{tabular}
\caption{dim4 empirical thresholds (frozen 2023).}
\end{table}"""


def write_equations_tex(output_dir: Path | str = "docs/arxiv") -> Path:
    """Write core equation LaTeX fragments to output directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    equations = export_equations_latex()
    content = r"""\section{Core Equations}
\begin{align}
\text{Fertility (linear):} \quad & """ + equations["fertility_linear"] + r""" \\
\text{Fertility (quadratic):} \quad & """ + equations["fertility_quadratic"] + r""" \\
\text{Goalden threshold:} \quad & """ + equations["goalden_threshold"] + r""" \\
\text{Transition map:} \quad & """ + equations["transition_map"] + r"""
\end{align}
"""

    path = out / "equations.tex"
    path.write_text(content, encoding="utf-8")

    threshold_path = out / "thresholds.tex"
    threshold_path.write_text(export_threshold_table(), encoding="utf-8")

    return path