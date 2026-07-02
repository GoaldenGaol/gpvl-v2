"""GPVL v2 command-line interface."""

from __future__ import annotations

import argparse
import json
import sys

from volition import __version__
from volition.data.dim4 import load_countries_full, validation_stats
from volition.data.firms import firm_validation_stats, load_firm_states
from volition.export.latex import write_arxiv_bundle, write_equations_tex
from volition.paths import default_arxiv_dir


def _cmd_version(_: argparse.Namespace) -> int:
    print(f"gpvl {__version__}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    stats = validation_stats()
    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print("dim4 → TFR Validation (frozen 2023)")
        print(f"  Pearson r     : {stats['pearson_r']:.4f}")
        print(f"  R²            : {stats['r_squared']:.4f}")
        print(f"  Countries (N) : {int(stats['n_countries'])}")
        print(f"  Above +1.0    : {int(stats['n_above_irreversible'])}")
        print(f"  No-return violations: {int(stats['no_return_violations'])}")
    return 0


def _cmd_export_latex(args: argparse.Namespace) -> int:
    output = args.output or default_arxiv_dir()

    if args.bundle:
        result = write_arxiv_bundle(output)
        print(f"Wrote arXiv bundle to {result.output_dir}/")
        print(f"  main.tex       : {result.main_tex.name}")
        print(f"  equations.tex  : {result.equations_tex.name}")
        print(f"  thresholds.tex : {result.thresholds_tex.name}")
        print(f"  validation.tex : {result.validation_tex.name}")
        print(f"  abstract.tex   : {result.abstract_tex.name}")
        print(f"  preamble.tex   : {result.preamble_tex.name}")
    else:
        path = write_equations_tex(output)
        print(f"Wrote equations to {path}")
        print(f"Wrote thresholds to {path.parent / 'thresholds.tex'}")

    return 0


def _cmd_info(_: argparse.Namespace) -> int:
    countries = load_countries_full()
    country_stats = validation_stats()
    firm_stats = firm_validation_stats()
    firms = load_firm_states()
    print("GPVL v2 — Science of Volition")
    print(f"  Version          : {__version__}")
    print(f"  Countries (full) : {len(countries)}")
    print(f"  Firms            : {len(firms)}")
    print(f"  Country r        : {country_stats['pearson_r']:.4f}")
    print(f"  Firm ROC-AUC     : {firm_stats['roc_auc']:.4f}")
    print(f"  LaTeX output     : {default_arxiv_dir()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpvl",
        description="GPVL v2 — General Physics of Volition CLI",
    )
    parser.add_argument("--version", action="version", version=f"gpvl {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="Print package version").set_defaults(func=_cmd_version)

    info = sub.add_parser("info", help="Show project and dataset summary")
    info.set_defaults(func=_cmd_info)

    validate = sub.add_parser("validate", help="Run dim4 validation statistics")
    validate.add_argument("--json", action="store_true", help="Output as JSON")
    validate.set_defaults(func=_cmd_validate)

    export = sub.add_parser("export-latex", help="Generate LaTeX fragments for arXiv")
    export.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory (default: docs/arxiv)",
    )
    export.add_argument(
        "--bundle",
        action="store_true",
        default=True,
        help="Write full arXiv bundle including main.tex (default: on)",
    )
    export.add_argument(
        "--equations-only",
        action="store_true",
        help="Write only equations.tex and thresholds.tex",
    )
    export.set_defaults(func=_cmd_export_latex)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "export-latex" and args.equations_only:
        args.bundle = False

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())