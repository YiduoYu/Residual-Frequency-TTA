#!/usr/bin/env python3
"""CPU-only component checks for Selective-Spectral COSA."""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1] / "Spectral-master"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tta.cosa import SimpleOutputAdapter
from tta.cosa_select import SelectiveCOSAAdapter, SpectralContextBuilder, validate_spectral_usage


def _profile(pred_len=8, n_vars=3, sign=1.0, phase=0.0):
    t = torch.arange(pred_len, dtype=torch.float32)
    waves = []
    for var_idx in range(n_vars):
        waves.append(sign * torch.sin(2.0 * torch.pi * (var_idx + 1) * t / pred_len + phase))
    return torch.stack(waves, dim=-1)


def _history(profile):
    return deque([{
        "target_mean": 1.0,
        "target_mean_var": torch.arange(profile.shape[1], dtype=torch.float32),
        "residual_profile": profile,
    }])


def check_validate_spectral_usage() -> None:
    validate_spectral_usage(SimpleNamespace(SPECTRAL_USAGE="context_only", ENABLE_DIRECT_SPECTRAL_CORRECTION=False))
    try:
        validate_spectral_usage(SimpleNamespace(SPECTRAL_USAGE="direct", ENABLE_DIRECT_SPECTRAL_CORRECTION=False))
    except ValueError:
        pass
    else:
        raise AssertionError("SPECTRAL_USAGE != context_only should be rejected")

    try:
        validate_spectral_usage(SimpleNamespace(SPECTRAL_USAGE="context_only", ENABLE_DIRECT_SPECTRAL_CORRECTION=True))
    except ValueError:
        pass
    else:
        raise AssertionError("direct spectral correction should be rejected")


def check_simple_output_adapter_context_shapes() -> None:
    y = torch.randn(2, 5, 3)
    for var_wise in (False, True):
        adapter = SimpleOutputAdapter(pred_len=5, buffer_context_size=4, n_vars=3, var_wise_gating=var_wise)
        out_2d = adapter(y, torch.randn(2, 4))
        out_3d = adapter(y, torch.randn(2, 3, 4))
        assert tuple(out_2d.shape) == tuple(y.shape)
        assert tuple(out_3d.shape) == tuple(y.shape)


def check_spectral_context_shapes() -> None:
    sample_history = _history(_profile())
    expected = {
        "mean": (2, 4),
        "spectral": (2, 3, 9),
        "concat": (2, 3, 13),
    }
    for mode, shape in expected.items():
        builder = SpectralContextBuilder(
            buffer_context_size=4,
            spectral_k=3,
            mode=mode,
            device="cpu",
            pred_len=8,
            n_vars=3,
            include_power_ratio=True,
            include_residual_scale=True,
        )
        context = builder.build(sample_history, batch_size=2)
        assert tuple(context.shape) == shape, (mode, tuple(context.shape), shape)


def check_spectral_context_uses_residual_profile() -> None:
    builder = SpectralContextBuilder(
        buffer_context_size=4,
        spectral_k=3,
        mode="spectral",
        device="cpu",
        pred_len=8,
        n_vars=3,
    )
    scalar_only_history = deque([{"target_mean": 1.0, "residual": 999.0}])
    context = builder.build(scalar_only_history, batch_size=2)
    assert torch.count_nonzero(context).item() == 0, "scalar residual history must not drive spectral context"


def check_phase_and_sign_sensitivity() -> None:
    builder = SpectralContextBuilder(
        buffer_context_size=4,
        spectral_k=3,
        mode="spectral",
        device="cpu",
        pred_len=8,
        n_vars=3,
        spectral_repr="real_imag",
    )
    context_a = builder.build(_history(_profile(sign=1.0)), batch_size=1)
    context_b = builder.build(_history(_profile(sign=-1.0)), batch_size=1)
    assert not torch.allclose(context_a, context_b), "signed residual profile should affect spectral context"


def check_no_current_ground_truth_leakage() -> None:
    builder = SpectralContextBuilder(
        buffer_context_size=4,
        spectral_k=3,
        mode="concat",
        device="cpu",
        pred_len=8,
        n_vars=3,
    )
    history = _history(_profile(sign=1.0))
    before = builder.build(history, batch_size=1).clone()
    repeated = builder.build(history, batch_size=1)
    assert torch.equal(before, repeated)

    history.append({
        "target_mean": 9.0,
        "target_mean_var": torch.full((3,), 9.0),
        "residual_profile": _profile(sign=1.0, phase=1.0),
    })
    after = builder.build(history, batch_size=1)
    assert not torch.equal(before, after)


def check_context_projection_and_optimizer_params() -> None:
    raw_context = torch.randn(2, 3, 13)
    projector = torch.nn.Linear(13, 5)
    probe = object.__new__(SelectiveCOSAAdapter)
    probe.device = torch.device("cpu")
    object.__setattr__(probe, "context_projector", projector)
    probe.raw_context_dim = 13
    probe.effective_context_dim = 5
    projected = SelectiveCOSAAdapter._project_context(probe, raw_context)
    assert tuple(projected.shape) == (2, 3, 5)
    assert projected.shape[-1] <= 5

    adapter = SimpleOutputAdapter(pred_len=5, buffer_context_size=5, n_vars=3, var_wise_gating=True)
    params = list(adapter.parameters()) + list(projector.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    optimizer_param_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
    assert all(id(param) in optimizer_param_ids for param in projector.parameters())


def check_parameter_budget_diagnostics() -> None:
    probe = object.__new__(SelectiveCOSAAdapter)
    probe.cfg = SimpleNamespace(DATA=SimpleNamespace(PRED_LEN=5, N_VAR=3))
    probe.adapter_layers = 1
    probe.var_wise_gating = True
    probe.hidden_dim = 8
    probe.buffer_context_size = 4
    probe.raw_context_dim = 13
    probe.effective_context_dim = 5
    probe.context_projection_params = 70
    probe.context_projection_dim = 5
    probe.max_context_dim_for_full_adapter = 4
    probe.param_budget_multiplier = 0.1
    probe.context_builder = SimpleNamespace(mode="concat")
    probe.use_context_projection = False
    stats = SelectiveCOSAAdapter._estimate_context_param_increase(probe)
    required = {
        "raw_context_dim",
        "effective_context_dim",
        "context_projection_dim",
        "max_context_dim_for_full_adapter",
        "approx_context_param_increase",
        "effective_vs_mean_ratio",
        "full_adapter_context_limit_exceeded",
        "budget_exceeded",
    }
    assert required.issubset(stats.keys())
    assert stats["raw_vs_mean_ratio"] > 0.0
    assert stats["effective_vs_mean_ratio"] > 0.0
    assert stats["full_adapter_context_limit_exceeded"] is True
    assert stats["budget_exceeded"] is True


def main() -> None:
    checks = [
        check_validate_spectral_usage,
        check_simple_output_adapter_context_shapes,
        check_spectral_context_shapes,
        check_spectral_context_uses_residual_profile,
        check_phase_and_sign_sensitivity,
        check_no_current_ground_truth_leakage,
        check_context_projection_and_optimizer_params,
        check_parameter_budget_diagnostics,
    ]
    for check in checks:
        check()
        print(f"OK {check.__name__}")


if __name__ == "__main__":
    main()
