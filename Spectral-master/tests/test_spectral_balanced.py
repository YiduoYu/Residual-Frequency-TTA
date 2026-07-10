from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tta.spectral_balanced import MultiWindowSpectralEncoder, SpectralBalancedCOSAAdapter, SpectralTuningBackend


def _assert_gate_sums_to_one(tensor, dim=-1, atol=1e-6):
    summed = tensor.sum(dim=dim)
    expected = torch.ones_like(summed)
    assert torch.allclose(summed, expected, atol=atol), summed


def _has_grad(module):
    grads = [param.grad for param in module.parameters() if param.requires_grad]
    return any(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in grads)


def test_balanced_encoder_and_backend_sanity():
    torch.manual_seed(7)
    batch_size, pred_len, n_vars, condition_dim = 4, 16, 3, 8
    z = torch.randn(pred_len, n_vars, requires_grad=True)
    h = torch.randn(batch_size, pred_len, n_vars)
    target = torch.randn_like(h)

    encoder = MultiWindowSpectralEncoder(
        input_length=pred_len,
        n_vars=n_vars,
        condition_dim=condition_dim,
        encoder_type="multi_window",
        num_windows=3,
        window_types="hann,hamming,rect",
        feature_type="real_imag",
        pooling="low_mid_high",
        fusion="softmax_gate",
        num_frequency_bins=8,
    )
    condition = encoder(z, batch_size=batch_size)
    assert condition.shape == (batch_size, n_vars, condition_dim)
    assert encoder.last_window_weight_tensor.shape == (batch_size, n_vars, 3)
    _assert_gate_sums_to_one(encoder.last_window_weight_tensor)

    backend = SpectralTuningBackend(
        pred_len=pred_len,
        n_vars=n_vars,
        condition_dim=condition_dim,
        backend_types="film,ia3,adapter,lora",
        mixture_mode="residual",
        lora_rank=4,
        lora_alpha=8,
    )
    adapted = backend(h, condition)
    assert adapted.shape == h.shape
    assert backend.last_backend_weight_tensor.shape == (batch_size, n_vars, 4)
    _assert_gate_sums_to_one(backend.last_backend_weight_tensor)
    assert (adapted - h).abs().max().item() < 1e-2

    loss = F.mse_loss(adapted, target) + 0.01 * condition.pow(2).mean()
    loss.backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
    assert _has_grad(encoder.pool_projection)
    assert _has_grad(encoder.fusion.gate)
    assert _has_grad(backend.backend_gate)
    assert _has_grad(backend.film_generator)
    assert _has_grad(backend.ia3_generator)
    assert _has_grad(backend.adapter)
    assert _has_grad(backend.adapter_gate)
    assert _has_grad(backend.lora_down)
    assert _has_grad(backend.lora_up)
    assert _has_grad(backend.lora_gate)


def test_backend_ablation_modes():
    torch.manual_seed(11)
    batch_size, pred_len, n_vars, condition_dim = 2, 12, 2, 8
    h = torch.randn(batch_size, pred_len, n_vars)
    condition = torch.randn(batch_size, n_vars, condition_dim)
    backend_sets = [
        "film",
        "ia3",
        "adapter",
        "lora",
        "film,ia3",
        "ia3,lora",
        "film,ia3,adapter,lora",
    ]

    for backend_types in backend_sets:
        backend = SpectralTuningBackend(
            pred_len=pred_len,
            n_vars=n_vars,
            condition_dim=condition_dim,
            backend_types=backend_types,
            mixture_mode="residual",
        )
        adapted = backend(h, condition)
        assert adapted.shape == h.shape
        _assert_gate_sums_to_one(backend.last_backend_weight_tensor)


def test_spectral_encoder_ablation_modes():
    torch.manual_seed(13)
    pred_len, n_vars, condition_dim = 10, 2, 6
    z = torch.randn(pred_len, n_vars)
    cases = [
        ("single_window", 1, "rect", "magnitude", "mean"),
        ("multi_window", 3, "hann,hamming,rect", "real_imag", "concat_projection"),
    ]

    for encoder_type, num_windows, window_types, feature_type, fusion in cases:
        encoder = MultiWindowSpectralEncoder(
            input_length=pred_len,
            n_vars=n_vars,
            condition_dim=condition_dim,
            encoder_type=encoder_type,
            num_windows=num_windows,
            window_types=window_types,
            feature_type=feature_type,
            pooling="low_mid_high",
            fusion=fusion,
            num_frequency_bins=4,
        )
        condition = encoder(z, batch_size=3)
        assert condition.shape == (3, n_vars, condition_dim)
        _assert_gate_sums_to_one(encoder.last_window_weight_tensor)


def test_log_summary_skips_empty_warmup_rows():
    adapter = SpectralBalancedCOSAAdapter.__new__(SpectralBalancedCOSAAdapter)
    adapter.batch_logs = [
        {"window_fusion_weights": "[]", "backend_mixture_weights": "{}"},
        {
            "window_fusion_weights": "[0.2, 0.3, 0.5]",
            "backend_mixture_weights": '{"film": 0.4, "ia3": 0.6}',
        },
    ]
    adapter.output_adapter = type("OutputAdapterStub", (), {"backend_types": ["film", "ia3"]})()

    assert adapter._mean_json_list_from_logs("window_fusion_weights") == [0.2, 0.3, 0.5]
    assert adapter._mean_backend_weights_from_logs() == {"film": 0.4, "ia3": 0.6}


def test_backend_config_parsing_is_stable():
    backend = SpectralTuningBackend(
        pred_len=8,
        n_vars=1,
        condition_dim=4,
        backend_types="film,lora",
        lora_enabled="False",
        mixture_mode="RESIDUAL",
        backend_gate="SOFTMAX",
    )
    assert backend.backend_types == ["film"]
    assert backend.mixture_mode == "residual"
    assert backend.backend_gate_type == "softmax"


if __name__ == "__main__":
    test_balanced_encoder_and_backend_sanity()
    test_backend_ablation_modes()
    test_spectral_encoder_ablation_modes()
    test_log_summary_skips_empty_warmup_rows()
    test_backend_config_parsing_is_stable()
