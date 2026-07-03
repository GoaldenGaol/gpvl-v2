"""LaTeX export for equations, thresholds, and arXiv-ready fragments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from volition.constants import DIM4_COOP_COLLAPSE, DIM4_IRREVERSIBLE, DIM4_NO_RETURN
from volition.data.dim4 import validation_stats
from volition.equations import export_equations_latex
from volition.paths import default_arxiv_dir


@dataclass
class ArxivExportResult:
    """Paths written by the arXiv LaTeX export bundle."""

    output_dir: Path
    main_tex: Path
    equations_tex: Path
    thresholds_tex: Path
    validation_tex: Path
    abstract_tex: Path
    preamble_tex: Path


def export_threshold_table() -> str:
    """Return LaTeX tabular for dim4 thresholds."""
    return rf"""\begin{{table}}[h]
\centering
\begin{{tabular}}{{lll}}
\toprule
Threshold & Meaning & Reversals observed \\
\midrule
$> +{DIM4_COOP_COLLAPSE:.2f}$ & Cooperation collapses ($<40$ rounds) & None \\
$> +{DIM4_IRREVERSIBLE:.2f}$ & Irreversible demographic band & Zero (1950--2025) \\
$< +{DIM4_NO_RETURN:.2f}$ after $>+1.0$ & No-return invariant & Zero \\
\bottomrule
\end{{tabular}}
\caption{{dim4 empirical thresholds (frozen 2023).}}
\label{{tab:dim4-thresholds}}
\end{{table}}"""


def export_validation_table() -> str:
    """Return LaTeX table of dim4 vs TFR validation statistics."""
    stats = validation_stats()
    return rf"""\begin{{table}}[h]
\centering
\begin{{tabular}}{{lr}}
\toprule
Statistic & Value \\
\midrule
Pearson $r$ & ${stats['pearson_r']:.3f}$ \\
$R^2$ & ${stats['r_squared']:.3f}$ \\
Countries ($N$) & {int(stats['n_countries'])} \\
Above irreversible ($\mathrm{{dim4}} > 1.0$) & {int(stats['n_above_irreversible'])} \\
No-return violations & {int(stats['no_return_violations'])} \\
\bottomrule
\end{{tabular}}
\caption{{dim4 $\rightarrow$ TFR validation on frozen 2023 cross-section.}}
\label{{tab:validation}}
\end{{table}}"""


def export_preamble() -> str:
    """Return shared LaTeX preamble for arXiv manuscript fragments."""
    return r"""\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
"""


def export_abstract() -> str:
    """Return arXiv abstract block."""
    stats = validation_stats()
    return rf"""\begin{{abstract}}
We present an open framework in which a seven-dimensional volitional state
vector $V(t)$, extracted from narrative embeddings, predicts cross-national
fertility and cooperation collapse through a single latent scalar $\mathrm{{dim4}}$.
Empirical validation on {int(stats['n_countries'])} countries (1950--2023) yields Pearson
$r = {stats['pearson_r']:.3f}$ for $\mathrm{{TFR}}(t{{+}}18) = f(\mathrm{{dim4}}(t))$.
Thresholds at $\mathrm{{dim4}} > {DIM4_COOP_COLLAPSE}$ (cooperation collapse) and
$\mathrm{{dim4}} > {DIM4_IRREVERSIBLE}$ (irreversibility) show zero historical reversals.
The GPVL v2 replication package provides symbolic equations, VPDE dynamics,
MCMC inference, and extensible octonion/$G_2$ geometry hooks.
\end{{abstract}}"""


def export_equations_content() -> str:
    """Return LaTeX align block for core equations, Φ, dynamics, and axioms."""
    eq = export_equations_latex()
    return (
        r"""\section{Core Equations}
\subsection{Fertility Mappings}
\begin{align}
\mathrm{TFR}_{\text{linear}} &= """
        + eq["fertility_linear"].split("=", 1)[-1].strip()
        + r""" \\
\mathrm{TFR}_{\text{quadratic}} &= """
        + eq["fertility_quadratic"].split("=", 1)[-1].strip()
        + r""" \\
\frac{\partial \mathrm{TFR}}{\partial \mathrm{dim4}} &< 0
\end{align}

\subsection{Goalden Threshold}
\begin{equation}
"""
        + eq["goalden_threshold"]
        + r"""
\end{equation}

\subsection{Transition Map $\Phi$}
\begin{equation}
"""
        + eq["phi_dim4"]
        + r"""
\end{equation}

\subsection{VPDE (continuous)}
\begin{equation}
"""
        + eq["vpde"]
        + r"""
\end{equation}

\subsection{System Dynamics $S(t)$}
\begin{align}
"""
        + eq["dynamics"].replace(r"\\", r" \nonumber \\")
        + r"""
\end{align}

\subsection{Regime Transition Matrix $\Theta$}
\begin{equation}
\Theta = """
        + eq["regime_theta"]
        + r"""
\end{equation}

\subsection{Axioms (excerpt)}
\begin{align}
\text{Axiom 6:} \quad & """
        + eq["axiom_6"]
        + r""" \\
\text{Axiom 8:} \quad & """
        + eq["axiom_8"]
        + r"""
\end{align}
"""
    )


def export_main_tex() -> str:
    """Return master LaTeX document including all fragments."""
    return r"""\documentclass[11pt]{article}
\input{preamble.tex}

\title{The Science of Volition: A Testable Framework for Cooperative Equilibria}
\author{GoaldenGaol}
\date{\today}

\begin{document}
\maketitle
\input{abstract.tex}

\input{equations.tex}
\input{thresholds.tex}
\input{validation.tex}

\end{document}
"""


def write_equations_tex(output_dir: Path | str | None = None) -> Path:
    """Write core equation LaTeX fragments to output directory."""
    out = Path(output_dir) if output_dir is not None else default_arxiv_dir()
    out.mkdir(parents=True, exist_ok=True)

    path = out / "equations.tex"
    path.write_text(export_equations_content(), encoding="utf-8")

    threshold_path = out / "thresholds.tex"
    threshold_path.write_text(export_threshold_table(), encoding="utf-8")

    return path


def write_arxiv_bundle(output_dir: Path | str | None = None) -> ArxivExportResult:
    """
    Write complete arXiv-ready LaTeX fragment bundle.

    Generates: main.tex, preamble.tex, abstract.tex, equations.tex,
    thresholds.tex, validation.tex
    """
    out = Path(output_dir) if output_dir is not None else default_arxiv_dir()
    out.mkdir(parents=True, exist_ok=True)

    files = {
        "preamble.tex": export_preamble(),
        "abstract.tex": export_abstract(),
        "equations.tex": export_equations_content(),
        "thresholds.tex": export_threshold_table(),
        "validation.tex": export_validation_table(),
        "main.tex": export_main_tex(),
    }

    paths: dict[str, Path] = {}
    for name, content in files.items():
        path = out / name
        path.write_text(content, encoding="utf-8")
        paths[name] = path

    return ArxivExportResult(
        output_dir=out,
        main_tex=paths["main.tex"],
        equations_tex=paths["equations.tex"],
        thresholds_tex=paths["thresholds.tex"],
        validation_tex=paths["validation.tex"],
        abstract_tex=paths["abstract.tex"],
        preamble_tex=paths["preamble.tex"],
    )