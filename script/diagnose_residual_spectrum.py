#!/usr/bin/env python3
"""Diagnose spectral structure in frozen base-model residuals."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

SPECTRAL_ROOT = Path(__file__).resolve().parents[1] / "Spectral-master"
if str(SPECTRAL_ROOT) not in sys.path:
    sys.path.insert(0, str(SPECTRAL_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute low-frequency power diagnostics for base-model residual profiles."
    )
    parser.add_argument("--cfg", dest="cfg_file", default=None, help="optional config file")
    parser.add_argument("--max-batches", type=int, default=20, help="maximum test batches to inspect")
    parser.add_argument("--spectral-k", type=int, default=8, help="number of non-DC low-frequency bins")
    parser.add_argument(
        "--output",
        default="results/diagnostics/residual_spectrum.json",
        help="JSON output path",
    )
    parser.add_argument("--csv-output", default="", help="optional per-variable CSV output path")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="execution device; cpu is the default for diagnostics",
    )
    args, cfg_opts = parser.parse_known_args()
    return args, cfg_opts


def normalize_dataset_name(name: str) -> str:
    aliases = {
        "weather": "weather",
        "exchange": "exchange_rate",
        "exchange-rate": "exchange_rate",
        "exchange_rate": "exchange_rate",
        "exchange rate": "exchange_rate",
        "etth1": "ETTh1",
        "etth2": "ETTh2",
        "ettm1": "ETTm1",
        "ettm2": "ETTm2",
    }
    return aliases.get(str(name).strip().lower(), name)


def spectral_summary(profile, spectral_k: int, eps: float = 1e-12):
    import torch

    profile_centered = profile - profile.mean(dim=0, keepdim=True)
    coeff = torch.fft.rfft(profile_centered, dim=0)
    non_dc = coeff[1:]
    if non_dc.numel() == 0:
        n_vars = profile.shape[1]
        zeros = torch.zeros(n_vars, dtype=torch.float32)
        return zeros, zeros, zeros, torch.zeros(n_vars, dtype=torch.long), zeros

    power = torch.abs(non_dc) ** 2
    total_power = power.sum(dim=0).clamp_min(eps)
    k = min(int(spectral_k), power.shape[0])
    top_k_power = power[:k].sum(dim=0)
    power_ratio = top_k_power / total_power
    dominant_power, dominant_idx = power.max(dim=0)
    dominant_bin = dominant_idx + 1
    dominant_power_ratio = dominant_power / total_power
    return total_power, top_k_power, power_ratio, dominant_bin, dominant_power_ratio


def write_csv(path: Path, variables: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variable_idx",
        "total_power",
        "top_k_power",
        "power_ratio",
        "dominant_bin",
        "dominant_power_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(variables)


def main() -> None:
    args, cfg_opts = parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch

    from config import get_cfg_defaults, get_norm_module_cfg
    from datasets.build import update_cfg_from_dataset
    from datasets.loader import get_test_dataloader
    from models.build import build_model, build_norm_module, load_best_model
    from models.forecast import forecast
    from utils.misc import set_devices, set_seeds

    cfg = get_cfg_defaults()
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(cfg_opts)
    cfg.DATA.NAME = normalize_dataset_name(cfg.DATA.NAME)
    update_cfg_from_dataset(cfg, cfg.DATA.NAME)
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True
    cfg.TTA.ENABLE = False
    if args.device == "cuda":
        set_devices(cfg.VISIBLE_DEVICES)

    set_seeds(cfg.SEED)
    model = build_model(cfg)
    norm_module = build_norm_module(cfg) if cfg.NORM_MODULE.ENABLE else None
    model = load_best_model(cfg, model)
    if cfg.NORM_MODULE.ENABLE:
        norm_module = load_best_model(get_norm_module_cfg(cfg), norm_module)
    model.eval()
    if norm_module is not None:
        norm_module.eval()

    test_loader = get_test_dataloader(cfg)
    profiles = []
    batch_count = 0
    sample_count = 0
    with torch.no_grad():
        for inputs in test_loader:
            pred, targets = forecast(cfg, inputs, model, norm_module)
            residual = targets - pred
            profiles.append(residual.mean(dim=0).detach().float().cpu())
            batch_count += 1
            sample_count += int(residual.shape[0])
            if args.max_batches and batch_count >= args.max_batches:
                break

    if not profiles:
        raise RuntimeError("No residual profiles were collected; check dataset and TEST.BATCH_SIZE settings.")

    profile = torch.stack(profiles, dim=0).mean(dim=0)
    total_power, top_k_power, power_ratio, dominant_bin, dominant_power_ratio = spectral_summary(
        profile,
        args.spectral_k,
    )

    variables = []
    for idx in range(profile.shape[1]):
        variables.append({
            "variable_idx": int(idx),
            "total_power": float(total_power[idx]),
            "top_k_power": float(top_k_power[idx]),
            "power_ratio": float(power_ratio[idx]),
            "dominant_bin": int(dominant_bin[idx]),
            "dominant_power_ratio": float(dominant_power_ratio[idx]),
        })

    mean_power_ratio = float(power_ratio.mean())
    mean_dominant_power_ratio = float(dominant_power_ratio.mean())
    result = {
        "dataset": cfg.DATA.NAME,
        "model": cfg.MODEL.NAME,
        "pred_len": int(cfg.DATA.PRED_LEN),
        "spectral_k": int(args.spectral_k),
        "batches": int(batch_count),
        "samples": int(sample_count),
        "diagnostic_threshold_for_direct": float(
            getattr(cfg.TTA.COSA.SELECT, "SPECTRAL_MIN_POWER_RATIO_FOR_DIRECT", 0.20)
        ),
        "mean_power_ratio": mean_power_ratio,
        "mean_dominant_power_ratio": mean_dominant_power_ratio,
        "direct_correction_recommendation": (
            "do_not_use_without_more_evidence"
            if mean_power_ratio < getattr(cfg.TTA.COSA.SELECT, "SPECTRAL_MIN_POWER_RATIO_FOR_DIRECT", 0.20)
            else "spectral_structure_present_but_still_requires_separate_validation"
        ),
        "variables": variables,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.csv_output:
        write_csv(Path(args.csv_output), variables)
    print(json.dumps({
        "output": str(output),
        "mean_power_ratio": mean_power_ratio,
        "mean_dominant_power_ratio": mean_dominant_power_ratio,
        "recommendation": result["direct_correction_recommendation"],
    }, indent=2))


if __name__ == "__main__":
    main()
