"""Full arXiv manuscript section generators with live replication data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from volition.constants import (
    DIM4_COOP_COLLAPSE,
    DIM4_IRREVERSIBLE,
    FRAMEWORK_PEARSON_R,
    FRAMEWORK_R_SQUARED,
)
from volition.data.dim4 import load_countries_full, validation_stats
from volition.data.firms import firm_validation_stats
from volition.geometry import calibrate_geometry, multiplication_table_latex
from volition.mcmc import ModelKind, run_mcmc
from volition.paths import default_arxiv_dir
from volition.vpde import calibrate_tau
from volition.vpde.calibration import USA_TARGET


@dataclass
class ManuscriptExportResult:
    """Paths written by the full arXiv manuscript export."""

    output_dir: Path
    main_tex: Path
    files: dict[str, Path] = field(default_factory=dict)


def _gather_stats() -> dict:
    """Collect live statistics for manuscript sections."""
    country_stats = validation_stats(load_countries_full())
    firm_stats = firm_validation_stats()
    mcmc = run_mcmc(kind=ModelKind.LINEAR, use_full_dataset=True, n_steps=300, burn_in=80)
    vpde = calibrate_tau(USA_TARGET)
    _, geom = calibrate_geometry()
    return {
        "country": country_stats,
        "firm": firm_stats,
        "mcmc": mcmc,
        "vpde": vpde,
        "geom": geom,
    }


def export_introduction() -> str:
    return r"""\section{Introduction}
\label{sec:introduction}

The Science of Volition proposes that cooperative equilibria in human societies
are governed by a low-dimensional volitional state vector $V(t)$ extracted from
narrative corpora. The fourth latent component $\mathrm{dim4}$ predicts
cross-national fertility 18 years later, cooperation-collapse thresholds, and
firm-level fatal-pivot risk.

This manuscript accompanies the GPVL~v2 replication package: a reproducible
Python implementation with symbolic equations, Phi-coupled VPDE dynamics,
MCMC inference, and octonion/$G_2$ geometry hooks. All thresholds and mappings
are falsifiable against frozen 2023 data.
"""


def export_methods(stats: dict) -> str:
    vpde = stats["vpde"]
    geom = stats["geom"]
    return rf"""\section{{Methods}}
\label{{sec:methods}}

\subsection{{Volitional State Vector}}
The state $V(t) = [\mathrm{{dim1}},\ldots,\mathrm{{dim7}}]$ is extracted from
narrative embeddings (frozen November 2023). The active scalar $\mathrm{{dim4}}$
maps to total fertility rate via $ \mathrm{{TFR}}(t{{+}}18) = f(\mathrm{{dim4}}(t))$.

\subsection{{Transition Map and VPDE}}
Discrete dynamics follow $V(t{{+}}1) = \Phi(V(t), i_t, u_t)$. The continuous
extension solves
\[
  \frac{{dV}}{{dt}} = \frac{{\Phi(V, i_t, u_t) - V}}{{\tau}} + D\,\varepsilon(t).
\]
We calibrate $\tau = {vpde.tau:.3f}$ so $\mathrm{{dim4}} = 0.94$ crosses the
irreversibility threshold at $t \approx {vpde.achieved_crossing_time:.1f}$ years
(USA forecast anchor).

\subsection{{MCMC Inference}}
We fit the linear model $\mathrm{{TFR}} = \alpha\cdot\mathrm{{dim4}} + \beta$
via affine-invariant ensemble MCMC (emcee), reporting posterior medians and
84\% credible intervals.

\subsection{{Octonion / $G_2$ Embedding}}
The 7-D latent vector embeds into the imaginary octonion units $\mathbb{{O}}$;
$G_2$ calibration uses the empirical $\mathrm{{dim4}}$ distribution
($\mu = {geom.dim4_mean:.3f}$, $\sigma = {geom.dim4_std:.3f}$, $N = {geom.n_samples}$).
"""


def export_results(stats: dict) -> str:
    cs = stats["country"]
    fs = stats["firm"]
    mcmc = stats["mcmc"]
    vpde = stats["vpde"]
    s = mcmc.summary
    return rf"""\section{{Results}}
\label{{sec:results}}

\subsection{{Country-Level Fertility Mapping}}
On the frozen 2023 cross-section ($N = {int(cs['n_countries'])}$ countries),
Pearson $r = {cs['pearson_r']:.3f}$ and $R^2 = {cs['r_squared']:.3f}$.
The framework panel target (1950--2023) reports $r = {FRAMEWORK_PEARSON_R}$ and
$R^2 = {FRAMEWORK_R_SQUARED}$.

\subsection{{MCMC Posterior (Linear Model)}}
\begin{{table}}[h]
\centering
\begin{{tabular}}{{lrr}}
\toprule
Parameter & Median & 84\% CI \\
\midrule
$\alpha$ & {mcmc.alpha:.4f} & [{s.p16[0]:.4f}, {s.p84[0]:.4f}] \\
$\beta$  & {mcmc.beta:.4f}  & [{s.p16[1]:.4f}, {s.p84[1]:.4f}] \\
$\sigma$ & {mcmc.sigma:.4f} & --- \\
\bottomrule
\end{{tabular}}
\caption{{MCMC posterior on 195-country full dataset.}}
\label{{tab:mcmc}}
\end{{table}}

\subsection{{Firm-Level Fatal Pivot}}
Firm-level $\mathrm{{dim4}}$ predicts 5--7 year collapse with ROC-AUC $= {fs['roc_auc']:.3f}$
($N = {int(fs['n_firms'])}$ firms).

\subsection{{Threshold Invariants}}
\begin{{itemize}}
  \item $\mathrm{{dim4}} > {DIM4_COOP_COLLAPSE}$: cooperation collapse ($<40$ rounds)
  \item $\mathrm{{dim4}} > {DIM4_IRREVERSIBLE}$: irreversible band ({int(cs['n_above_irreversible'])} countries above)
  \item VPDE crossing time: $t = {vpde.achieved_crossing_time:.2f}$ yr (residual ${vpde.residual:.3f}$)
\end{{itemize}}
"""


def export_discussion() -> str:
    return r"""\section{Discussion}
\label{sec:discussion}

The GPVL~v2 package demonstrates that the frozen 2023 $\mathrm{dim4}$ scalar
replicates negative fertility correlation, firm-level collapse ranking, and
threshold invariants on publicly available data. The VPDE time-scale $\tau$
bridges discrete $\Phi$ transitions to continuous forecasts (e.g.\ USA
irreversibility crossing).

Limitations: the public replication uses 2023 cross-sections; full 7-D latent
extraction and panel forecasts require the narrative embedding pipeline.
Octonion/$G_2$ calibration is anchored on $\mathrm{dim4}$ until dim1--dim7
latents are released.

Future work: independent panel replication, pronatal-policy null-effect tests,
and $G_2$ automorphism calibration on full $V(t)$.
"""


def export_appendix() -> str:
    oct_table = multiplication_table_latex()
    return (
        r"""\appendix
\section{Octonion Multiplication Rules}
\label{app:octonion}
"""
        + oct_table
        + r"""

\section{Replication Commands}
\label{app:replication}
\begin{verbatim}
pip install -e ".[dev]"
gpvl validate
gpvl invariants
gpvl mcmc --full
gpvl calibrate-vpde
gpvl calibrate-geometry
gpvl export-latex --manuscript
\end{verbatim}
"""
    )


def export_references() -> str:
    return r"""\begin{thebibliography}{9}

\bibitem{core2023}
GoaldenGaol.
\textit{The Science of Volition: Core Framework (Frozen 2023)}.
CC-BY-4.0, 2025.

\bibitem{gpvlv2}
GoaldenGaol.
\textit{GPVL v2: General Physics of Volition Replication Package}.
Software, 2026. \url{https://github.com/GoaldenGaol/gpvl-v2}

\bibitem{baeZ2002}
J.~C. Baez.
\textit{The Octonions}.
Bull.\ Amer.\ Math.\ Soc.\ \textbf{39} (2002), 145--205.

\bibitem{foremanmackey2013}
D.~Foreman-Mackey et al.
\textit{emcee: The MCMC Hammer}.
PASP \textbf{125} (2013), 306--312.

\end{thebibliography}
"""


def export_arxiv_readme(output_dir: Path) -> str:
    return f"""arXiv Upload Guide — GPVL v2 Manuscript
========================================

Primary file: main.tex
Compiler: pdfLaTeX

Files in this bundle:
  main.tex, preamble.tex, abstract.tex
  introduction.tex, methods.tex, results.tex, discussion.tex
  equations.tex, thresholds.tex, validation.tex
  appendix.tex, references.tex

Upload all .tex files to arXiv.

Build locally:
  cd {output_dir.name}
  pdflatex main.tex
  pdflatex main.tex

Generated by: gpvl export-latex --manuscript
License: CC-BY-4.0
"""


def export_main_manuscript() -> str:
    return r"""\documentclass[11pt,preprint]{article}
\input{preamble.tex}

\title{The Science of Volition: A Testable Framework\\for Cooperative Equilibria}
\author{GoaldenGaol\\\texttt{shlewi88@gmail.com}}
\date{\today}

\begin{document}
\maketitle
\input{abstract.tex}

\input{introduction.tex}
\input{methods.tex}
\input{results.tex}
\input{discussion.tex}

\input{equations.tex}
\input{thresholds.tex}
\input{validation.tex}

\input{references.tex}
\input{appendix.tex}

\end{document}
"""


def write_full_manuscript(output_dir: Path | str | None = None) -> ManuscriptExportResult:
    """Write complete arXiv manuscript with all sections and live data."""
    from volition.export.latex import (
        export_abstract,
        export_equations_content,
        export_preamble,
        export_threshold_table,
        export_validation_table,
    )

    out = Path(output_dir) if output_dir is not None else default_arxiv_dir()
    out.mkdir(parents=True, exist_ok=True)

    stats = _gather_stats()

    contents = {
        "preamble.tex": export_preamble(),
        "abstract.tex": export_abstract(),
        "introduction.tex": export_introduction(),
        "methods.tex": export_methods(stats),
        "results.tex": export_results(stats),
        "discussion.tex": export_discussion(),
        "equations.tex": export_equations_content(),
        "thresholds.tex": export_threshold_table(),
        "validation.tex": export_validation_table(),
        "appendix.tex": export_appendix(),
        "references.tex": export_references(),
        "main.tex": export_main_manuscript(),
        "00README.txt": export_arxiv_readme(out),
    }

    paths: dict[str, Path] = {}
    for name, content in contents.items():
        path = out / name
        path.write_text(content, encoding="utf-8")
        paths[name] = path

    return ManuscriptExportResult(
        output_dir=out,
        main_tex=paths["main.tex"],
        files=paths,
    )