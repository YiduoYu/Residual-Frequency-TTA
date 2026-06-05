#!/usr/bin/env python3
"""Collect Scheme 6 logs, average seeds, and compare against 5.26 results."""

from __future__ import annotations

import argparse
import csv
import json
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

    for family in ("COSA_SCHEME6", "COSA_SELECT", "SIMPLE"):
        if family in parts:
            idx = parts.index(family)
            fields["family"] = family
            if family == "SIMPLE":
                if len(parts) > idx + 4:
                    fields["variant"] = parts[idx + 1]
                    fields["model"] = parts[idx + 2]
                    fields["dataset"] = parts[idx + 3]
                else:
                    fields["variant"] = "COSA"
                    fields["model"] = parts[idx + 1] if len(parts) > idx + 1 else ""
                    fields["dataset"] = parts[idx + 2] if len(parts) > idx + 2 else ""
            else:
                fields["variant"] = parts[idx + 1] if len(parts) > idx + 1 else ""
                fields["model"] = parts[idx + 2] if len(parts) > idx + 2 else ""
                fields["dataset"] = parts[idx + 3] if len(parts) > idx + 3 else ""
            break
    else:
        fields["family"] = parts[0] if parts else ""
        fields["variant"] = parts[-4] if len(parts) >= 4 else path.stem

    for token in path.stem.split("_"):
        if token.startswith("pred"):
            fields["horizon"] = token.replace("pred", "")
        elif token.startswith("seed"):
            fields["seed"] = token.replace("seed", "")
    if not fields["horizon"] and path.stem.isdigit():
        fields["horizon"] = path.stem
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


def collect(root: Path) -> list[dict]:
    rows = []
    log_paths = sorted(set(root.rglob("*.log")) | set(root.rglob("*.txt")))
    for log_path in log_paths:
        try:
            result = parse_json_from_log(log_path)
        except ValueError:
            continue

        row = infer_fields(log_path, root)
        final = result.get("final_results", {})
        selective = result.get("selective_statistics", {})
        scheme6 = result.get("scheme6_statistics", {})
        overall = result.get("time_statistics", {}).get("overall_stats", {})
        row.update({
            "adapter": result.get("model", ""),
            "test_mse": final.get("test_mse", ""),
            "test_mae": final.get("test_mae", ""),
            "adaptation_count": final.get("adaptation_count", ""),
            "total_time_seconds": overall.get("total_time_seconds", ""),
            "throughput_samples_per_sec": overall.get("throughput_samples_per_sec", ""),
            "trigger_rate": selective.get("trigger_rate", ""),
            "skipped_update_ratio": selective.get("skipped_update_ratio", ""),
            "spectral_mode": selective.get("spectral_mode", scheme6.get("spectral_mode", "")),
            "spectral_k": selective.get("spectral_k", scheme6.get("spectral_k", "")),
            "context_dim": selective.get("context_dim", scheme6.get("context_dim", "")),
            "spectral_usage": selective.get("spectral_usage", scheme6.get("spectral_usage", "")),
            "spectral_repr": selective.get("spectral_repr", scheme6.get("spectral_repr", "")),
            "spectral_power_ratio_mean": selective.get("spectral_power_ratio_mean", ""),
            "direct_spectral_enabled": scheme6.get("direct_spectral_enabled", 1 if scheme6 else ""),
            "scheme6_correction_source": scheme6.get("correction_source", ""),
            "scheme6_gate_mode": scheme6.get("spectral_gate_mode", ""),
            "scheme6_window_type": scheme6.get("window_type", ""),
            "scheme6_out_proj": scheme6.get("spectral_out_proj", ""),
            "scheme6_fft_precision": scheme6.get("fft_precision", ""),
            "scheme6_fft_length_mode": scheme6.get("fft_length_mode", ""),
            "scheme6_use_global_gate": scheme6.get("use_global_gate", ""),
            "scheme6_global_gate_floor": scheme6.get("global_gate_floor", ""),
            "scheme6_constant_global_gate": scheme6.get("constant_global_gate", ""),
            "scheme6_complex_mod_freq_cutoff_ratio": scheme6.get("complex_mod_freq_cutoff_ratio", ""),
            "scheme6_complex_mod_freq_mask_mean": scheme6.get("complex_mod_freq_mask_mean", ""),
            "scheme6_global_gate_lr_multiplier": scheme6.get("global_gate_lr_multiplier", ""),
            "scheme6_reference_content_injection": scheme6.get("reference_content_injection", ""),
            "scheme6_global_gate_mean": scheme6.get("global_gate_mean", ""),
            "scheme6_raw_global_gate_mean": scheme6.get("raw_global_gate_mean", ""),
            "scheme6_local_gate_mean": scheme6.get("local_gate_mean", ""),
            "scheme6_low_freq_ratio_mean": scheme6.get("low_freq_ratio_mean", ""),
            "scheme6_spectral_l2_mean": scheme6.get("spectral_l2_mean", ""),
            "scheme6_direct_delta_norm_mean": scheme6.get("direct_delta_norm_mean", ""),
            "scheme6_reference_content_active_ratio": scheme6.get("reference_content_active_ratio", ""),
            "scheme6_complex_mod_magnitude_mean": scheme6.get("complex_mod_magnitude_mean", ""),
            "scheme6_complex_mod_phase_abs_mean": scheme6.get("complex_mod_phase_abs_mean", ""),
            "context_projection_params": selective.get(
                "context_projection_params",
                scheme6.get(
                    "context_projection_params",
                    get_nested(result, "parameters", "context_projection_params"),
                ),
            ),
        })
        rows.append(row)
    return rows


PER_SEED_FIELDS = [
    "family",
    "variant",
    "model",
    "dataset",
    "horizon",
    "seed",
    "adapter",
    "test_mse",
    "test_mae",
    "adaptation_count",
    "total_time_seconds",
    "throughput_samples_per_sec",
    "trigger_rate",
    "skipped_update_ratio",
    "spectral_mode",
    "spectral_k",
    "context_dim",
    "spectral_usage",
    "spectral_repr",
    "spectral_power_ratio_mean",
    "direct_spectral_enabled",
    "scheme6_correction_source",
    "scheme6_gate_mode",
    "scheme6_window_type",
    "scheme6_out_proj",
    "scheme6_fft_precision",
    "scheme6_fft_length_mode",
    "scheme6_use_global_gate",
    "scheme6_global_gate_floor",
    "scheme6_constant_global_gate",
    "scheme6_complex_mod_freq_cutoff_ratio",
    "scheme6_complex_mod_freq_mask_mean",
    "scheme6_global_gate_lr_multiplier",
    "scheme6_reference_content_injection",
    "scheme6_global_gate_mean",
    "scheme6_raw_global_gate_mean",
    "scheme6_local_gate_mean",
    "scheme6_low_freq_ratio_mean",
    "scheme6_spectral_l2_mean",
    "scheme6_direct_delta_norm_mean",
    "scheme6_reference_content_active_ratio",
    "scheme6_complex_mod_magnitude_mean",
    "scheme6_complex_mod_phase_abs_mean",
    "context_projection_params",
    "log_path",
]

AVERAGE_FIELDS = [
    "family",
    "variant",
    "model",
    "dataset",
    "horizon",
    "seed_count",
    "seeds",
    "missing_seeds",
    "test_mse_mean",
    "test_mse_std",
    "test_mae_mean",
    "test_mae_std",
    "adaptation_count_mean",
    "total_time_seconds_mean",
    "throughput_samples_per_sec_mean",
]

IMPROVEMENT_FIELDS = [
    "variant",
    "model",
    "dataset",
    "horizon",
    "seed_count",
    "seeds",
    "missing_seeds",
    "scheme6_mse_mean",
    "scheme6_mse_std",
    "scheme6_mae_mean",
    "scheme6_mae_std",
    "reference_family",
    "reference_variant",
    "reference_seed_count",
    "reference_seeds",
    "reference_mse_mean",
    "reference_mae_mean",
    "mse_delta",
    "mae_delta",
    "mse_improvement_pct",
    "mae_improvement_pct",
]


def to_float(value):
    if value == "" or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def seed_sort_key(item: str) -> tuple[int, int | str]:
    return (0, int(item)) if item.isdigit() else (1, item)


def sorted_seeds(values) -> list[str]:
    return sorted({str(value) for value in values if str(value) != ""}, key=seed_sort_key)


def mean_or_blank(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else ""


def std_or_blank(values):
    values = [value for value in values if value is not None]
    if not values:
        return ""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_rows(rows: list[dict], expected_seeds: set[str]) -> list[dict]:
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
    for key, group in sorted(grouped.items()):
        family, variant, model, dataset, horizon = key
        seeds = sorted_seeds(row.get("seed", "") for row in group)
        missing = sorted(expected_seeds - set(seeds), key=seed_sort_key) if expected_seeds else []
        out = {
            "family": family,
            "variant": variant,
            "model": model,
            "dataset": dataset,
            "horizon": horizon,
            "seed_count": len(seeds),
            "seeds": " ".join(seeds),
            "missing_seeds": " ".join(missing),
        }
        for metric in ("test_mse", "test_mae"):
            values = [to_float(row.get(metric, "")) for row in group]
            out[f"{metric}_mean"] = mean_or_blank(values)
            out[f"{metric}_std"] = std_or_blank(values)
        out["adaptation_count_mean"] = mean_or_blank([to_float(row.get("adaptation_count", "")) for row in group])
        out["total_time_seconds_mean"] = mean_or_blank([to_float(row.get("total_time_seconds", "")) for row in group])
        out["throughput_samples_per_sec_mean"] = mean_or_blank(
            [to_float(row.get("throughput_samples_per_sec", "")) for row in group]
        )
        averaged.append(out)
    return averaged


def ref_key(row: dict) -> tuple[str, str, str]:
    return (row.get("model", ""), row.get("dataset", ""), row.get("horizon", ""))


def load_reference_rows(path: Path, variants: set[str]) -> dict[tuple[str, str, str], list[dict]]:
    if not path.exists():
        return {}
    refs = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if variants and row.get("variant", "") not in variants:
                continue
            refs[ref_key(row)].append(row)
    return refs


def improvement_pct(reference, candidate):
    if reference in ("", None) or candidate in ("", None):
        return ""
    reference = float(reference)
    candidate = float(candidate)
    if reference == 0:
        return ""
    return (reference - candidate) / reference * 100.0


def build_improvement_rows(averaged: list[dict], reference_rows: dict[tuple[str, str, str], list[dict]]) -> list[dict]:
    out = []
    for row in averaged:
        if row.get("family") != "COSA_SCHEME6":
            continue
        for ref in reference_rows.get(ref_key(row), []):
            scheme6_mse = to_float(row.get("test_mse_mean", ""))
            scheme6_mae = to_float(row.get("test_mae_mean", ""))
            ref_mse = to_float(ref.get("test_mse_mean", ""))
            ref_mae = to_float(ref.get("test_mae_mean", ""))
            out.append({
                "variant": row.get("variant", ""),
                "model": row.get("model", ""),
                "dataset": row.get("dataset", ""),
                "horizon": row.get("horizon", ""),
                "seed_count": row.get("seed_count", ""),
                "seeds": row.get("seeds", ""),
                "missing_seeds": row.get("missing_seeds", ""),
                "scheme6_mse_mean": row.get("test_mse_mean", ""),
                "scheme6_mse_std": row.get("test_mse_std", ""),
                "scheme6_mae_mean": row.get("test_mae_mean", ""),
                "scheme6_mae_std": row.get("test_mae_std", ""),
                "reference_family": ref.get("family", ""),
                "reference_variant": ref.get("variant", ""),
                "reference_seed_count": ref.get("seed_count", ""),
                "reference_seeds": ref.get("seeds", ""),
                "reference_mse_mean": ref.get("test_mse_mean", ""),
                "reference_mae_mean": ref.get("test_mae_mean", ""),
                "mse_delta": (scheme6_mse - ref_mse) if scheme6_mse is not None and ref_mse is not None else "",
                "mae_delta": (scheme6_mae - ref_mae) if scheme6_mae is not None and ref_mae is not None else "",
                "mse_improvement_pct": improvement_pct(ref_mse, scheme6_mse),
                "mae_improvement_pct": improvement_pct(ref_mae, scheme6_mae),
            })
    return out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="Spectral-master/results/summary", help="summary log root")
    parser.add_argument("--output", default="Spectral-master/results/summary/scheme6_collected_results.csv")
    parser.add_argument(
        "--average-output",
        default="Spectral-master/results/summary/scheme6_collected_results_averaged.csv",
    )
    parser.add_argument(
        "--improvement-output",
        default="Spectral-master/results/summary/scheme6_improvement_vs_526.csv",
    )
    parser.add_argument(
        "--reference-averaged",
        default="result/5.26_paper8_experiment/final_tables/paper8_full_comparison_averaged.csv",
    )
    parser.add_argument(
        "--reference-variants",
        nargs="*",
        default=[
            "baseline_no_tta",
            "cosa_paper8_fixed",
            "cosa_paper8_paas",
            "spectral_paper8_residual_profile_real_imag_hist8_projTrue_dim8",
        ],
    )
    parser.add_argument("--families", nargs="*", default=None, help="optional result-family filter")
    parser.add_argument("--datasets", nargs="*", default=None, help="optional dataset filter")
    parser.add_argument("--models", nargs="*", default=None, help="optional model filter")
    parser.add_argument("--horizons", nargs="*", default=None, help="optional prediction horizon filter")
    parser.add_argument("--variants", nargs="*", default=None, help="optional variant filter")
    parser.add_argument("--expected-seeds", nargs="*", default=["0", "1", "2"], help="seeds expected per averaged row")
    parser.add_argument("--require-complete", action="store_true", help="fail if an averaged Scheme6 row misses a seed")
    args = parser.parse_args()

    root = Path(args.root)
    rows = [row for row in collect(root) if keep_row(row, args)]
    expected_seeds = parse_list(args.expected_seeds)
    averaged_rows = aggregate_rows(rows, expected_seeds)

    if args.require_complete:
        incomplete = [
            row for row in averaged_rows
            if row.get("family") == "COSA_SCHEME6" and row.get("missing_seeds", "")
        ]
        if incomplete:
            sample = "\n".join(
                f"{row['variant']}/{row['model']}/{row['dataset']}/pred{row['horizon']}: missing {row['missing_seeds']}"
                for row in incomplete[:20]
            )
            raise SystemExit(f"Incomplete Scheme6 seed grid:\n{sample}")

    reference_rows = load_reference_rows(Path(args.reference_averaged), parse_list(args.reference_variants))
    improvement_rows = build_improvement_rows(averaged_rows, reference_rows)

    write_csv(Path(args.output), rows, PER_SEED_FIELDS)
    write_csv(Path(args.average_output), averaged_rows, AVERAGE_FIELDS)
    write_csv(Path(args.improvement_output), improvement_rows, IMPROVEMENT_FIELDS)


if __name__ == "__main__":
    main()
