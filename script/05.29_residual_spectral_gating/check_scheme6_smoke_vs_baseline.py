#!/usr/bin/env python3
"""Gate Scheme 6 smoke results against the 5.26 no-TTA baseline."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: str, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid numeric value for {label}: {value!r}") from exc


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("model", ""), row.get("dataset", ""), row.get("horizon", ""))


def index_rows(rows: list[dict[str, str]], variant: str) -> dict[tuple[str, str, str], dict[str, str]]:
    indexed = {}
    for row in rows:
        if row.get("variant", "") != variant:
            continue
        indexed[row_key(row)] = row
    return indexed


def split_arg(values: list[str]) -> list[str]:
    out = []
    for value in values:
        out.extend(item for item in value.split() if item)
    return out


def fmt_pct(value: float) -> str:
    return f"{value:+.3f}%"


def seed_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme6-averaged", default="Spectral-master/results/summary/scheme6_smoke_averaged.csv")
    parser.add_argument(
        "--reference-averaged",
        default="result/5.26_paper8_experiment/final_tables/paper8_full_comparison_averaged.csv",
    )
    parser.add_argument("--variant", default="adapter_residual_only")
    parser.add_argument("--reference-variant", default="baseline_no_tta")
    parser.add_argument("--models", nargs="+", default=["DLinear"])
    parser.add_argument("--datasets", nargs="+", default=["ETTh2", "ETTm2", "exchange_rate", "electricity"])
    parser.add_argument("--horizons", nargs="+", default=["96"])
    parser.add_argument("--expected-seeds", nargs="+", default=["0", "1", "2"])
    parser.add_argument("--min-baseline-wins", type=int, default=1)
    parser.add_argument("--min-avg-improvement-pct", type=float, default=0.0)
    args = parser.parse_args()

    models = split_arg(args.models)
    datasets = split_arg(args.datasets)
    horizons = split_arg(args.horizons)
    expected_seeds = set(split_arg(args.expected_seeds))
    expected_keys = list(itertools.product(models, datasets, horizons))

    scheme6_rows = index_rows(read_csv(Path(args.scheme6_averaged)), args.variant)
    reference_rows = index_rows(read_csv(Path(args.reference_averaged)), args.reference_variant)

    missing = []
    incomplete = []
    comparisons = []
    for key in expected_keys:
        scheme6 = scheme6_rows.get(key)
        reference = reference_rows.get(key)
        if scheme6 is None or reference is None:
            missing.append((key, scheme6 is not None, reference is not None))
            continue

        seeds = set(scheme6.get("seeds", "").split())
        missing_seeds = sorted(expected_seeds - seeds, key=seed_sort_key)
        if missing_seeds or scheme6.get("missing_seeds", "").strip():
            incomplete.append((key, " ".join(missing_seeds) or scheme6.get("missing_seeds", "").strip()))

        scheme6_mse = parse_float(scheme6.get("test_mse_mean", ""), f"scheme6 {key}")
        baseline_mse = parse_float(reference.get("test_mse_mean", ""), f"baseline {key}")
        if baseline_mse == 0:
            raise SystemExit(f"Baseline MSE is zero for {key}; cannot compute improvement")
        improvement = (baseline_mse - scheme6_mse) / baseline_mse * 100.0
        comparisons.append((key, scheme6_mse, baseline_mse, improvement))

    if missing:
        print("Missing smoke comparison rows:")
        for key, has_scheme6, has_reference in missing:
            print(f"  {key}: scheme6={has_scheme6} baseline={has_reference}")

    if incomplete:
        print("Incomplete smoke seed rows:")
        for key, missing_seed_text in incomplete:
            print(f"  {key}: missing seeds {missing_seed_text}")

    if comparisons:
        print("Scheme6 smoke vs baseline_no_tta")
        print("model        dataset        horizon  scheme6_mse  baseline_mse  mse_improvement")
        for key, scheme6_mse, baseline_mse, improvement in comparisons:
            model, dataset, horizon = key
            print(
                f"{model:<12} {dataset:<14} {horizon:>7}  "
                f"{scheme6_mse:>11.6f}  {baseline_mse:>12.6f}  {fmt_pct(improvement):>15}"
            )

    wins = sum(1 for _, _, _, improvement in comparisons if improvement > 0.0)
    avg_improvement = sum(item[3] for item in comparisons) / len(comparisons) if comparisons else float("-inf")
    print(
        f"summary: rows={len(comparisons)}/{len(expected_keys)}, wins={wins}, "
        f"avg_mse_improvement={fmt_pct(avg_improvement)}"
    )

    failed = bool(missing or incomplete)
    if wins < args.min_baseline_wins:
        print(f"FAIL: baseline wins {wins} < required {args.min_baseline_wins}")
        failed = True
    if avg_improvement < args.min_avg_improvement_pct:
        print(
            f"FAIL: average improvement {fmt_pct(avg_improvement)} "
            f"< required {fmt_pct(args.min_avg_improvement_pct)}"
        )
        failed = True

    if failed:
        print("DO_NOT_RUN_FULL_GRID")
        raise SystemExit(1)

    print("PASS: smoke gate cleared")


if __name__ == "__main__":
    main()
