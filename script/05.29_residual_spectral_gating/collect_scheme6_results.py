#!/usr/bin/env python3
"""Collect Scheme 6/COSA JSON logs into a flat CSV table."""

from __future__ import annotations

import argparse
import csv
import json
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
    return fields


def parse_list(values: list[str] | None) -> set[str]:
    return set(values or [])


def keep_row(row: dict, args: argparse.Namespace) -> bool:
    filters = {
        "family": parse_list(args.families),
        "dataset": parse_list(args.datasets),
        "model": parse_list(args.models),
        "variant": parse_list(args.variants),
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
            "spectral_mode": selective.get("spectral_mode", ""),
            "spectral_k": selective.get("spectral_k", ""),
            "context_dim": selective.get("context_dim", ""),
            "spectral_usage": selective.get("spectral_usage", ""),
            "spectral_repr": selective.get("spectral_repr", ""),
            "spectral_power_ratio_mean": selective.get("spectral_power_ratio_mean", ""),
            "direct_spectral_enabled": scheme6.get("direct_spectral_enabled", ""),
            "scheme6_gate_mode": scheme6.get("spectral_gate_mode", ""),
            "scheme6_window_type": scheme6.get("window_type", ""),
            "scheme6_out_proj": scheme6.get("spectral_out_proj", ""),
            "scheme6_fft_precision": scheme6.get("fft_precision", ""),
            "scheme6_fft_length_mode": scheme6.get("fft_length_mode", ""),
            "scheme6_use_global_gate": scheme6.get("use_global_gate", ""),
            "scheme6_reference_content_injection": scheme6.get("reference_content_injection", ""),
            "scheme6_global_gate_mean": scheme6.get("global_gate_mean", ""),
            "scheme6_local_gate_mean": scheme6.get("local_gate_mean", ""),
            "scheme6_low_freq_ratio_mean": scheme6.get("low_freq_ratio_mean", ""),
            "scheme6_spectral_l2_mean": scheme6.get("spectral_l2_mean", ""),
            "scheme6_direct_delta_norm_mean": scheme6.get("direct_delta_norm_mean", ""),
            "scheme6_reference_content_active_ratio": scheme6.get("reference_content_active_ratio", ""),
            "context_projection_params": selective.get(
                "context_projection_params",
                get_nested(result, "parameters", "context_projection_params"),
            ),
        })
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="Spectral-master/results/summary", help="summary log root")
    parser.add_argument("--output", default="Spectral-master/results/summary/scheme6_collected_results.csv")
    parser.add_argument("--families", nargs="*", default=None, help="optional result-family filter")
    parser.add_argument("--datasets", nargs="*", default=None, help="optional dataset filter")
    parser.add_argument("--models", nargs="*", default=None, help="optional model filter")
    parser.add_argument("--variants", nargs="*", default=None, help="optional variant filter")
    args = parser.parse_args()

    root = Path(args.root)
    rows = [row for row in collect(root) if keep_row(row, args)]
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
        "scheme6_gate_mode",
        "scheme6_window_type",
        "scheme6_out_proj",
        "scheme6_fft_precision",
        "scheme6_fft_length_mode",
        "scheme6_use_global_gate",
        "scheme6_reference_content_injection",
        "scheme6_global_gate_mean",
        "scheme6_local_gate_mean",
        "scheme6_low_freq_ratio_mean",
        "scheme6_spectral_l2_mean",
        "scheme6_direct_delta_norm_mean",
        "scheme6_reference_content_active_ratio",
        "context_projection_params",
        "log_path",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
