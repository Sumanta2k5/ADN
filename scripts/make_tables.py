"""Generate the paper's result tables from saved metrics JSON files.

Reads ``results/<exp>/metrics.json`` produced by ``scripts/test.py`` and emits
Markdown + LaTeX tables matching the paper's structure:

  * Table I  : DIV2K(val) PSNR/SSIM per scale.
  * Table II : Set5/Set14/BSD100/Urban100 PSNR/SSIM per scale.
  * Table V  : ablation (w/o CBAM, w/o ResNet, ...) at x2.

Usage:
    python -m scripts.make_tables --results-dir results --out tables
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict


def _load_all(results_dir: str) -> Dict[str, dict]:
    data = {}
    for path in glob.glob(os.path.join(results_dir, "**", "metrics.json"), recursive=True):
        with open(path) as f:
            d = json.load(f)
        exp = os.path.basename(os.path.dirname(path))
        data[exp] = d
    return data


def _fmt(psnr, ssim):
    if psnr != psnr:  # nan
        return "-- / --"
    return f"{psnr:.2f} / {ssim:.4f}"


def build_benchmark_table(data: Dict[str, dict], method_name: str = "ADN (Ours)"):
    """Aggregate {set: {scale: (psnr,ssim)}} across experiments."""
    table = defaultdict(dict)  # set -> scale -> (psnr, ssim)
    for exp, d in data.items():
        scale = d.get("scale")
        for set_name, res in d.get("results", {}).items():
            table[set_name][scale] = (res["psnr"], res["ssim"])
    return table


def to_markdown(table, scales=(2, 4, 8)) -> str:
    sets = sorted(table.keys())
    header = "| Dataset | " + " | ".join(f"x{s} (PSNR/SSIM)" for s in scales) + " |"
    sep = "|" + "---|" * (len(scales) + 1)
    rows = [header, sep]
    for s in sets:
        cells = [_fmt(*table[s][sc]) if sc in table[s] else "-- / --" for sc in scales]
        rows.append(f"| {s} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def to_latex(table, scales=(2, 4, 8), caption="ADN results", label="tab:results") -> str:
    sets = sorted(table.keys())
    col = "l" + "c" * len(scales)
    lines = [
        "\\begin{table}[t]", "\\centering",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col}}}", "\\toprule",
        "Dataset & " + " & ".join(f"$\\times{s}$" for s in scales) + " \\\\",
        "\\midrule",
    ]
    for s in sets:
        cells = [_fmt(*table[s][sc]) if sc in table[s] else "-- / --" for sc in scales]
        lines.append(f"{s} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def build_ablation_table(results_dir: str) -> str:
    """Collect ablation experiments (folders containing 'ablation' or 'wo_')."""
    rows = ["| Variant | Set | Scale | PSNR | SSIM |", "|---|---|---|---|---|"]
    for path in sorted(glob.glob(os.path.join(results_dir, "**", "metrics.json"), recursive=True)):
        exp = os.path.basename(os.path.dirname(path))
        if "ablat" not in exp and "wo_" not in exp and "kernel" not in exp and "patch" not in exp:
            continue
        with open(path) as f:
            d = json.load(f)
        for set_name, res in d.get("results", {}).items():
            rows.append(f"| {exp} | {set_name} | x{d.get('scale')} | "
                        f"{res['psnr']:.2f} | {res['ssim']:.4f} |")
    return "\n".join(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out", default="tables")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = _load_all(args.results_dir)
    if not data:
        print(f"No metrics.json found under {args.results_dir}. Run scripts/test.py first.")
        return

    table = build_benchmark_table(data)

    md = "# ADN Benchmark Results\n\n" + to_markdown(table) + "\n"
    with open(os.path.join(args.out, "benchmark.md"), "w") as f:
        f.write(md)
    with open(os.path.join(args.out, "benchmark.tex"), "w") as f:
        f.write(to_latex(table, caption="PSNR/SSIM of ADN across benchmarks and scales.",
                         label="tab:adn_benchmark"))

    ablation = build_ablation_table(args.results_dir)
    with open(os.path.join(args.out, "ablation.md"), "w") as f:
        f.write("# Ablation Studies\n\n" + ablation + "\n")

    print(md)
    print(f"\nTables written to {args.out}/")


if __name__ == "__main__":
    main()
