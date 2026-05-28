#!/usr/bin/env python3
"""Collect baseline/COSA results into a flat CSV table."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def parse_json_from_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    decoder = json.JSONDecoder()
    objects = []
    idx = 0
    while idx < len(text):
        start = text.find("{", idx)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(obj, dict):
            objects.append(obj)
        idx = start + end

    for obj in reversed(objects):
        if "final_results" in obj:
            return obj
    if objects:
        return objects[-1]
    raise ValueError(f"No JSON object found in {path}")


def infer_fields(path: Path, root: Path) -> dict[str, str]:
    rel = path.relative_to(root)
    parts = rel.parts
    fields = {
        "family": "",
        "variant": "",
        "model": "",
        "dataset": "",
        "horizon": "",
        "seed": "",
        "log_path": str(path),
    }

    if "SIMPLE" in parts:
        idx = parts.index("SIMPLE")
        fields["family"] = "SIMPLE"
        if len(parts) > idx + 4:
            fields["variant"] = parts[idx + 1]
            fields["model"] = parts[idx + 2]
            fields["dataset"] = parts[idx + 3]
        else:
            fields["variant"] = "COSA"
            fields["model"] = parts[idx + 1] if len(parts) > idx + 1 else ""
            fields["dataset"] = parts[idx + 2] if len(parts) > idx + 2 else ""
    else:
        fields["family"] = parts[0] if parts else ""
        fields["variant"] = parts[-4] if len(parts) >= 4 else path.stem

    for token in path.stem.split("_"):
        if token.startswith("pred"):
            fields["horizon"] = token.replace("pred", "")
        elif token.startswith("seed"):
            fields["seed"] = token.replace("seed", "")
    return fields


def parse_metric_text(text: str) -> dict[str, float]:
    metrics = {}
    for key in ("test_mse", "test_mae", "train_mse", "train_mae"):
        match = re.search(rf"{key}\s*:\s*([-+0-9.eE]+)", text)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def infer_baseline_fields(path: Path, baseline_root: Path) -> dict[str, str]:
    rel = path.relative_to(baseline_root)
    parts = rel.parts
    fields = {
        "family": "BASELINE",
        "variant": "baseline_no_tta",
        "model": "",
        "dataset": "",
        "horizon": "",
        "seed": "",
        "log_path": str(path),
    }
    if len(parts) >= 4 and parts[0].startswith("seed"):
        fields["seed"] = parts[0].replace("seed", "")
        fields["model"] = parts[1]
        dataset_horizon = parts[2]
    elif len(parts) >= 3:
        fields["model"] = parts[0]
        dataset_horizon = parts[1]
    else:
        dataset_horizon = ""

    if dataset_horizon:
        if "_" in dataset_horizon:
            dataset, horizon = dataset_horizon.rsplit("_", 1)
            fields["dataset"] = dataset
            fields["horizon"] = horizon
        else:
            fields["dataset"] = dataset_horizon
    return fields


def parse_list(values: list[str] | None) -> set[str]:
    return set(values or [])


def keep_row(row: dict, args: argparse.Namespace) -> bool:
    filters = {
        "family": parse_list(args.families),
        "dataset": parse_list(args.datasets),
        "model": parse_list(args.models),
        "variant": parse_list(args.variants),
        "horizon": parse_list(args.horizons),
    }
    for key, allowed in filters.items():
        if allowed and row.get(key, "") not in allowed:
            return False
    return True


def get_nested(data: dict, *keys):
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key, "")
    return current


def collect_json_logs(root: Path) -> list[dict]:
    rows = []
    log_paths = sorted(set(root.rglob("*.log")) | set(root.rglob("*.txt")))
    for log_path in log_paths:
        try:
            result = parse_json_from_log(log_path)
        except ValueError:
            continue

        row = infer_fields(log_path, root)
        final = result.get("final_results", {})
        spectral = result.get("spectral_statistics", {})
        overall = result.get("time_statistics", {}).get("overall_stats", {})
        row.update({
            "adapter": result.get("model", ""),
            "test_mse": final.get("test_mse", ""),
            "test_mae": final.get("test_mae", ""),
            "adaptation_count": final.get("adaptation_count", ""),
            "total_time_seconds": overall.get("total_time_seconds", ""),
            "throughput_samples_per_sec": overall.get("throughput_samples_per_sec", ""),
            "spectral_mode": spectral.get("spectral_mode", ""),
            "spectral_k": spectral.get("spectral_k", ""),
            "context_dim": spectral.get("context_dim", ""),
            "spectral_usage": spectral.get("spectral_usage", ""),
            "spectral_repr": spectral.get("spectral_repr", ""),
            "raw_context_dim": spectral.get("raw_context_dim", ""),
            "effective_context_dim": spectral.get("effective_context_dim", ""),
            "context_projection_dim": spectral.get("context_projection_dim", ""),
            "max_context_dim_for_full_adapter": spectral.get("max_context_dim_for_full_adapter", ""),
            "use_context_projection": spectral.get("use_context_projection", ""),
            "spectral_power_ratio_mean": spectral.get("spectral_power_ratio_mean", ""),
            "estimated_context_param_increase": spectral.get("estimated_context_param_increase", ""),
            "estimated_context_param_increase_ratio": spectral.get("estimated_context_param_increase_ratio", ""),
            "context_budget_exceeded": spectral.get("context_budget_exceeded", ""),
            "full_adapter_context_limit_exceeded": spectral.get("full_adapter_context_limit_exceeded", ""),
            "context_projection_params": spectral.get(
                "context_projection_params",
                get_nested(result, "parameters", "context_projection_params"),
            ),
        })
        rows.append(row)
    return rows


def collect_baselines(baseline_root: Path | None) -> list[dict]:
    if baseline_root is None or not baseline_root.exists():
        return []

    rows = []
    for result_path in sorted(baseline_root.rglob("test.txt")):
        metrics = parse_metric_text(result_path.read_text(encoding="utf-8", errors="replace"))
        if "test_mse" not in metrics:
            continue

        row = infer_baseline_fields(result_path, baseline_root)
        row.update({
            "adapter": "BaselineNoTTA",
            "test_mse": metrics.get("test_mse", ""),
            "test_mae": metrics.get("test_mae", ""),
            "train_mse": metrics.get("train_mse", ""),
            "train_mae": metrics.get("train_mae", ""),
            "adaptation_count": 0,
            "total_time_seconds": "",
            "throughput_samples_per_sec": "",
            "spectral_mode": "",
            "spectral_k": "",
            "context_dim": "",
            "spectral_usage": "",
            "spectral_repr": "",
            "raw_context_dim": "",
            "effective_context_dim": "",
            "context_projection_dim": "",
            "max_context_dim_for_full_adapter": "",
            "use_context_projection": "",
            "spectral_power_ratio_mean": "",
            "estimated_context_param_increase": "",
            "estimated_context_param_increase_ratio": "",
            "context_budget_exceeded": "",
            "full_adapter_context_limit_exceeded": "",
            "context_projection_params": "",
        })
        rows.append(row)
    return rows


def collect(root: Path, baseline_root: Path | None = None) -> list[dict]:
    return collect_json_logs(root) + collect_baselines(baseline_root)


def to_float(value):
    if value == "" or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row.get("family", ""),
            row.get("variant", ""),
            row.get("model", ""),
            row.get("dataset", ""),
            row.get("horizon", ""),
        )
        grouped[key].append(row)

    averaged = []
    metric_names = ("test_mse", "test_mae", "train_mse", "train_mae")
    for key, group in sorted(grouped.items()):
        family, variant, model, dataset, horizon = key
        seeds = sorted(
            {row.get("seed", "") for row in group if row.get("seed", "") != ""},
            key=lambda item: (0, int(item)) if item.isdigit() else (1, item),
        )
        out = {
            "family": family,
            "variant": variant,
            "model": model,
            "dataset": dataset,
            "horizon": horizon,
            "seed_count": len(seeds),
            "seeds": " ".join(seeds),
        }
        for metric in metric_names:
            values = [to_float(row.get(metric, "")) for row in group]
            values = [value for value in values if value is not None]
            out[f"{metric}_mean"] = statistics.mean(values) if values else ""
            out[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else (0.0 if values else "")
        averaged.append(out)
    return averaged


def seed_sort_key(item: str) -> tuple[int, int | str]:
    return (0, int(item)) if item.isdigit() else (1, item)


def family_for_variant(variant: str) -> str:
    if variant == "baseline_no_tta":
        return "BASELINE"
    if variant.startswith("cosa_"):
        return "SIMPLE"
    return ""


def expected_groups(args: argparse.Namespace) -> set[tuple[str, str, str, str, str]]:
    if not (args.datasets and args.models and args.variants and args.horizons):
        return set()

    allowed_families = parse_list(args.families)
    groups = set()
    for variant in args.variants:
        family = family_for_variant(variant)
        if not family:
            continue
        if allowed_families and family not in allowed_families:
            continue
        for model in args.models:
            for dataset in args.datasets:
                for horizon in args.horizons:
                    groups.add((family, variant, model, dataset, horizon))
    return groups


def validate_complete_grid(rows: list[dict], args: argparse.Namespace) -> None:
    expected_seeds = parse_list(args.expected_seeds)
    if not args.require_complete or not expected_seeds:
        return

    grouped = defaultdict(set)
    for row in rows:
        key = (
            row.get("family", ""),
            row.get("variant", ""),
            row.get("model", ""),
            row.get("dataset", ""),
            row.get("horizon", ""),
        )
        seed = row.get("seed", "")
        if seed:
            grouped[key].add(seed)

    groups_to_check = expected_groups(args) or set(grouped)
    problems = []
    for key in sorted(groups_to_check):
        seeds = grouped.get(key, set())
        missing = sorted(expected_seeds - seeds, key=seed_sort_key)
        if missing:
            family, variant, model, dataset, horizon = key
            problems.append(
                f"{family}/{variant}/{model}/{dataset}/pred{horizon}: missing seeds {' '.join(missing)}"
            )

    if problems:
        sample = "\n".join(problems[:20])
        if len(problems) > 20:
            sample += f"\n... and {len(problems) - 20} more"
        raise SystemExit(f"Incomplete result grid for required seeds:\n{sample}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/summary", help="summary log root")
    parser.add_argument("--baseline-root", default=None, help="optional baseline result root")
    parser.add_argument("--output", default="results/summary/collected_results.csv")
    parser.add_argument("--average-output", default=None, help="optional averaged CSV output")
    parser.add_argument("--families", nargs="*", default=None, help="optional result-family filter")
    parser.add_argument("--datasets", nargs="*", default=None, help="optional dataset filter")
    parser.add_argument("--models", nargs="*", default=None, help="optional model filter")
    parser.add_argument("--horizons", nargs="*", default=None, help="optional prediction horizon filter")
    parser.add_argument("--variants", nargs="*", default=None, help="optional variant filter")
    parser.add_argument("--expected-seeds", nargs="*", default=None, help="seeds required per averaged row")
    parser.add_argument("--require-complete", action="store_true", help="fail if any expected seed is missing")
    args = parser.parse_args()

    root = Path(args.root)
    baseline_root = Path(args.baseline_root) if args.baseline_root else None
    rows = [row for row in collect(root, baseline_root) if keep_row(row, args)]
    validate_complete_grid(rows, args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "family",
        "variant",
        "model",
        "dataset",
        "horizon",
        "seed",
        "adapter",
        "test_mse",
        "test_mae",
        "train_mse",
        "train_mae",
        "adaptation_count",
        "total_time_seconds",
        "throughput_samples_per_sec",
        "spectral_mode",
        "spectral_k",
        "context_dim",
        "spectral_usage",
        "spectral_repr",
        "raw_context_dim",
        "effective_context_dim",
        "context_projection_dim",
        "max_context_dim_for_full_adapter",
        "use_context_projection",
        "spectral_power_ratio_mean",
        "estimated_context_param_increase",
        "estimated_context_param_increase_ratio",
        "context_budget_exceeded",
        "full_adapter_context_limit_exceeded",
        "context_projection_params",
        "log_path",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} ({len(rows)} rows)")

    if args.average_output:
        average_output = Path(args.average_output)
        average_output.parent.mkdir(parents=True, exist_ok=True)
        averaged_rows = aggregate_rows(rows)
        average_fieldnames = [
            "family",
            "variant",
            "model",
            "dataset",
            "horizon",
            "seed_count",
            "seeds",
            "test_mse_mean",
            "test_mse_std",
            "test_mae_mean",
            "test_mae_std",
            "train_mse_mean",
            "train_mse_std",
            "train_mae_mean",
            "train_mae_std",
        ]
        with average_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=average_fieldnames)
            writer.writeheader()
            writer.writerows(averaged_rows)
        print(f"wrote {average_output} ({len(averaged_rows)} rows)")


if __name__ == "__main__":
    main()
