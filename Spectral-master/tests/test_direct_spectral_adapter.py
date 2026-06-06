from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tta.direct_spectral import (  # noqa: E402
    DirectSpectralContextBuilder,
    DirectSpectralCorrection,
)


def test_context_shape():
    torch.manual_seed(7)
    batch_size, pred_len, n_vars, k = 4, 16, 3, 8
    builder = DirectSpectralContextBuilder(
        pred_len=pred_len,
        n_vars=n_vars,
        spectral_k=k,
        history_size=8,
        use_power_ratio=True,
        use_residual_stats=True,
    )
    history = [
        {"residual_profile": torch.randn(pred_len, n_vars)}
        for _ in range(3)
    ]

    context = builder(history, batch_size=batch_size)

    assert builder.raw_context_dim == 19
    assert context.shape == (batch_size, n_vars, 19)
    assert torch.isfinite(context).all()


def test_missing_history_returns_zero_context_with_shape():
    builder = DirectSpectralContextBuilder(
        pred_len=16,
        n_vars=3,
        spectral_k=8,
        history_size=8,
        use_power_ratio=True,
        use_residual_stats=True,
    )

    context = builder([], batch_size=2)

    assert context.shape == (2, 3, 19)
    assert torch.equal(context, torch.zeros_like(context))


def test_adapter_forward_shape():
    torch.manual_seed(11)
    batch_size, pred_len, n_vars, raw_context_dim = 2, 16, 3, 19
    correction = DirectSpectralCorrection(
        pred_len=pred_len,
        n_vars=n_vars,
        raw_context_dim=raw_context_dim,
        context_projection_dim=8,
        var_wise_gating=True,
    )
    pred = torch.randn(batch_size, pred_len, n_vars)
    raw_context = torch.randn(batch_size, n_vars, raw_context_dim)

    output = correction(pred, raw_context)

    assert output.shape == pred.shape
    assert torch.isfinite(output).all()


def test_trainable_params_nonzero_and_include_output_adapter_and_projection():
    correction = DirectSpectralCorrection(
        pred_len=16,
        n_vars=3,
        raw_context_dim=19,
        context_projection_dim=8,
        var_wise_gating=True,
    )
    trainable = {
        name: param
        for name, param in correction.named_parameters()
        if param.requires_grad
    }

    assert sum(param.numel() for param in trainable.values()) > 0
    assert any(name.startswith("output_adapter.") for name in trainable)
    assert any(name.startswith("context_projector.") for name in trainable)
    assert correction.context_projection_params > 0


def test_forbidden_backend_params_absent():
    correction = DirectSpectralCorrection(
        pred_len=16,
        n_vars=3,
        raw_context_dim=19,
        context_projection_dim=8,
        var_wise_gating=True,
    )
    forbidden = [
        "film",
        "ia3",
        "lora",
        "backend_gate",
        "backend_types",
        "backend_mixture",
        "last_backend_weights",
        "mixture_mode",
        "identity",
        "post_bias",
        "srdc",
    ]
    names = [name for name, _ in correction.named_parameters()]
    attrs = list(vars(correction).keys()) + list(vars(correction.output_adapter).keys())

    for term in forbidden:
        assert not any(term in name.lower() for name in names)
        assert not any(term in attr.lower() for attr in attrs)

    assert any(name == "output_adapter.gate" for name in names)


def test_no_balanced_backend_import_dependency():
    source = (ROOT / "tta" / "direct_spectral.py").read_text()

    assert "SpectralTuningBackend" not in source
    assert "MultiWindowSpectralEncoder" not in source
