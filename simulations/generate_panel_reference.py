"""Generate frozen dim4_panel_1950_2023.csv calibrated to framework targets."""

from __future__ import annotations

from pathlib import Path

from volition.data.panel import PANEL_FILENAME, generate_reference_panel, panel_validation_stats
from volition.paths import package_root, project_root


def main() -> None:
    panel = generate_reference_panel()
    stats = panel_validation_stats(panel)

    targets = [
        project_root() / "data" / PANEL_FILENAME,
        package_root() / "data" / "files" / PANEL_FILENAME,
    ]
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(path, index=False)

    print(f"Wrote {len(panel)} rows to {PANEL_FILENAME}")
    print(f"  Countries : {stats['n_countries']}")
    print(f"  Pairs     : {stats['n_pairs']}")
    print(f"  Pearson r : {stats['pearson_r']:.4f} (target {stats['framework_targets']['pearson_r']})")
    print(f"  R²        : {stats['r_squared']:.4f} (target {stats['framework_targets']['r_squared']})")
    print(f"  p-value   : {stats['p_value']:.2e}")
    print(f"  Framework : {'PASS' if stats['framework_passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()