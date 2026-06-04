import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.optimizer import get_optimizer
from tta.cosa import SimpleAdapter, SimpleOutputAdapter
from tta.cosa_spectral import SpectralContextBuilder
from utils.misc import prepare_inputs

try:
    from models.forecast import forecast
except ModuleNotFoundError:
    forecast = None


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "y")
    return bool(value)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def validate_scheme6_usage(spectral_cfg):
    usage = str(getattr(spectral_cfg, "USAGE", "direct")).lower()
    direct_enabled = _as_bool(getattr(spectral_cfg, "ENABLE_DIRECT_CORRECTION", False))
    supported = ("direct", "context_plus_direct")
    if usage not in supported:
        raise ValueError(f"COSA_SCHEME6 supports SPECTRAL.USAGE in {supported}; got {usage!r}.")
    if not direct_enabled:
        raise ValueError("COSA_SCHEME6 requires TTA.COSA.SPECTRAL.ENABLE_DIRECT_CORRECTION=True.")
    mode = str(getattr(spectral_cfg, "MODE", "concat")).lower()
    if usage == "context_plus_direct" and mode not in ("spectral", "concat"):
        raise ValueError("context_plus_direct requires TTA.COSA.SPECTRAL.MODE='spectral' or 'concat'.")


class Scheme6DirectSpectralCorrector(nn.Module):
    def __init__(
        self,
        pred_len,
        n_vars,
        spectral_k=8,
        history_size=8,
        correction_source="residual",
        gate_mode="unidirectional",
        use_global_gate=True,
        global_gate_features="hs,l2,low_freq_ratio",
        global_gate_init_bias=-6.0,
        window_type="hann",
        out_proj="diagonal",
        min_power_ratio=0.20,
        enable_reference_content_injection=False,
        fft_precision="fp32",
        fft_length_mode="exact",
        enable_revin_on_spectral_path=False,
        eps=1e-6,
    ):
        super().__init__()
        self.pred_len = int(pred_len)
        self.n_vars = int(n_vars)
        self.n_freq = self.pred_len // 2 + 1
        self.spectral_k = int(spectral_k)
        self.history_size = int(history_size)
        self.correction_source = str(correction_source).lower()
        self.gate_mode = str(gate_mode).lower()
        self.use_global_gate = bool(use_global_gate)
        self.global_gate_features = _as_list(global_gate_features) or ["hs", "l2", "low_freq_ratio"]
        self.global_gate_init_bias = float(global_gate_init_bias)
        self.window_type = str(window_type).lower()
        self.out_proj = str(out_proj).lower()
        self.min_power_ratio = float(min_power_ratio)
        self.enable_reference_content_injection = bool(enable_reference_content_injection)
        self.fft_precision = str(fft_precision).lower()
        self.fft_length_mode = str(fft_length_mode).lower()
        self.enable_revin_on_spectral_path = bool(enable_revin_on_spectral_path)
        self.eps = float(eps)

        if self.correction_source not in ("prediction", "residual", "both"):
            raise ValueError(f"Unsupported correction source: {self.correction_source}")
        if self.gate_mode not in ("fy_only", "unidirectional", "bidirectional"):
            raise ValueError(f"Unsupported gate mode: {self.gate_mode}")
        if self.window_type not in ("none", "rect", "rectangular", "hann", "hamming"):
            raise ValueError(f"Unsupported window type: {self.window_type}")
        if self.out_proj not in ("none", "diagonal", "depthwise", "dense"):
            raise ValueError(f"Unsupported output projection: {self.out_proj}")
        if self.fft_length_mode != "exact":
            raise ValueError("FFT_LENGTH_MODE='exact' is the supported Scheme 6 path.")

        self.gy_gain = nn.Parameter(torch.zeros(self.n_freq, self.n_vars))
        self.gy_bias = nn.Parameter(torch.zeros(self.n_freq, self.n_vars))
        if self.correction_source in ("residual", "both") or self.gate_mode == "bidirectional":
            self.gr_gain = nn.Parameter(torch.zeros(self.n_freq, self.n_vars))
            self.gr_bias = nn.Parameter(torch.zeros(self.n_freq, self.n_vars))

        if self.out_proj in ("diagonal", "depthwise"):
            self.out_gain = nn.Parameter(torch.zeros(self.n_freq, self.n_vars))
        elif self.out_proj == "dense":
            self.real_proj = nn.Linear(self.n_vars, self.n_vars, bias=False)
            self.imag_proj = nn.Linear(self.n_vars, self.n_vars, bias=False)
            nn.init.eye_(self.real_proj.weight)
            nn.init.eye_(self.imag_proj.weight)

        if self.use_global_gate:
            input_dim = len(self.global_gate_features)
            hidden_dim = max(4, min(16, input_dim * 4))
            self.global_gate = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            for module in self.global_gate:
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.weight)
                    nn.init.zeros_(module.bias)
            nn.init.constant_(self.global_gate[-1].bias, self.global_gate_init_bias)
        else:
            self.global_gate = None

        self.register_buffer("window", self._make_window(), persistent=False)
        self.last_stats = self._empty_stats()

    def _empty_stats(self):
        return {
            "direct_spectral_enabled": 1,
            "correction_source": self.correction_source,
            "spectral_gate_mode": self.gate_mode,
            "window_type": self.window_type,
            "spectral_out_proj": self.out_proj,
            "fft_precision": self.fft_precision,
            "fft_length_mode": self.fft_length_mode,
            "use_global_gate": int(self.use_global_gate),
            "enable_reference_content_injection": int(self.enable_reference_content_injection),
            "global_gate_mean": 0.0,
            "local_gate_mean": 0.0,
            "reference_gate_mean": 0.0,
            "low_freq_ratio_mean": 0.0,
            "spectral_l2_mean": 0.0,
            "direct_delta_norm": 0.0,
            "reference_content_active_ratio": 0.0,
        }

    def _make_window(self):
        if self.window_type in ("none", "rect", "rectangular"):
            return torch.ones(self.pred_len, dtype=torch.float32)
        if self.window_type == "hann":
            return torch.hann_window(self.pred_len, periodic=False, dtype=torch.float32)
        if self.window_type == "hamming":
            return torch.hamming_window(self.pred_len, periodic=False, dtype=torch.float32)
        raise ValueError(f"Unsupported window type: {self.window_type}")

    def _reference_profile(self, sample_history, batch_size, device):
        profiles = []
        for entry in reversed(sample_history):
            profile = entry.get("residual_profile")
            if profile is not None:
                profiles.append(torch.as_tensor(profile, dtype=torch.float32, device=device))
            if len(profiles) >= self.history_size:
                break

        if not profiles:
            profile = torch.zeros(self.pred_len, self.n_vars, dtype=torch.float32, device=device)
        else:
            profile = torch.stack(profiles, dim=0).mean(dim=0)
            if profile.shape != (self.pred_len, self.n_vars):
                raise ValueError(
                    f"residual_profile has shape {tuple(profile.shape)}, expected {(self.pred_len, self.n_vars)}"
                )
        return profile.unsqueeze(0).expand(int(batch_size), -1, -1)

    def _low_freq_ratio(self, coeff):
        non_dc = coeff[:, 1:, :]
        total_power = torch.sum(torch.abs(non_dc).pow(2), dim=1).clamp_min(self.eps)
        selected = coeff[:, 1 : self.spectral_k + 1, :]
        selected_power = torch.sum(torch.abs(selected).pow(2), dim=1)
        return (selected_power / total_power).clamp(0.0, 1.0)

    def _l2_energy(self, coeff):
        non_dc = coeff[:, 1:, :]
        return torch.sqrt(torch.mean(torch.abs(non_dc).pow(2), dim=1).clamp_min(self.eps))

    def _global_gate_features(self, y, low_freq_ratio, spectral_l2):
        values = []
        for feature in self.global_gate_features:
            key = feature.lower()
            if key in ("hs", "mean", "y_mean"):
                values.append(y.mean(dim=1))
            elif key in ("l2", "spectral_l2", "energy"):
                values.append(spectral_l2)
            elif key in ("low_freq_ratio", "rho_lf", "power_ratio"):
                values.append(low_freq_ratio)
            else:
                raise ValueError(f"Unsupported global gate feature: {feature}")
        return torch.stack(values, dim=-1)

    def _apply_out_projection(self, z):
        if self.out_proj == "none":
            return z
        if self.out_proj in ("diagonal", "depthwise"):
            scale = 1.0 + 0.1 * torch.tanh(self.out_gain).unsqueeze(0)
            return z * scale
        real = self.real_proj(z.real)
        imag = self.imag_proj(z.imag)
        return torch.complex(real, imag)

    def forward(self, y0, sample_history):
        if y0.dim() != 3:
            raise ValueError(f"Scheme 6 expects [B, L, V], got {tuple(y0.shape)}")
        batch_size, pred_len, n_vars = y0.shape
        if pred_len != self.pred_len or n_vars != self.n_vars:
            raise ValueError(f"Scheme 6 got {(pred_len, n_vars)}, expected {(self.pred_len, self.n_vars)}")

        original_dtype = y0.dtype
        y_base = y0.float()
        if self.enable_revin_on_spectral_path:
            y_mean = y_base.mean(dim=1, keepdim=True)
            y_scale = y_base.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
            y = (y_base - y_mean) / y_scale
        else:
            y_mean = None
            y_scale = None
            y = y_base

        ref = self._reference_profile(sample_history, batch_size, y.device)
        if self.enable_revin_on_spectral_path:
            ref = ref / y_scale
        ref = ref - ref.mean(dim=1, keepdim=True)
        ref = ref * self.window.to(device=y.device, dtype=y.dtype)[None, :, None]

        fy = torch.fft.rfft(y.float(), n=self.pred_len, dim=1)
        fr = torch.fft.rfft(ref.float(), n=self.pred_len, dim=1)

        if self.gate_mode == "fy_only":
            gy = torch.ones_like(fy.real)
        else:
            gy = torch.sigmoid(torch.abs(fr) * self.gy_gain.unsqueeze(0) + self.gy_bias.unsqueeze(0))
        if hasattr(self, "gr_gain"):
            gr = torch.sigmoid(torch.abs(fy) * self.gr_gain.unsqueeze(0) + self.gr_bias.unsqueeze(0))
        else:
            gr = torch.zeros_like(fy.real)

        z = torch.zeros_like(fy)
        if self.correction_source in ("prediction", "both"):
            z = z + gy * fy

        low_freq_ratio = self._low_freq_ratio(fr)
        spectral_l2 = self._l2_energy(fr)
        reference_active = torch.zeros_like(low_freq_ratio)
        if self.correction_source in ("residual", "both"):
            reference_active = torch.ones_like(low_freq_ratio, dtype=fy.real.dtype)
            z = z + gr * fr
        elif self.gate_mode == "bidirectional" and self.enable_reference_content_injection:
            reference_active = (low_freq_ratio >= self.min_power_ratio).to(fy.real.dtype)
            z = z + reference_active.unsqueeze(1) * gr * fr
        gr_mean = float(gr.detach().mean().cpu())

        z = self._apply_out_projection(z)
        delta = torch.fft.irfft(z, n=self.pred_len, dim=1)

        if self.global_gate is None:
            global_gate = torch.ones(batch_size, self.n_vars, dtype=delta.dtype, device=delta.device)
        else:
            gate_features = self._global_gate_features(y, low_freq_ratio, spectral_l2)
            global_gate = torch.sigmoid(self.global_gate(gate_features).squeeze(-1))

        y_hat = y + global_gate.unsqueeze(1) * delta
        if self.enable_revin_on_spectral_path:
            y_hat = y_hat * y_scale + y_mean

        denom = max(float(y_base.numel()), 1.0)
        self.last_stats = {
            "direct_spectral_enabled": 1,
            "correction_source": self.correction_source,
            "spectral_gate_mode": self.gate_mode,
            "window_type": self.window_type,
            "spectral_out_proj": self.out_proj,
            "fft_precision": self.fft_precision,
            "fft_length_mode": self.fft_length_mode,
            "use_global_gate": int(self.use_global_gate),
            "enable_reference_content_injection": int(self.enable_reference_content_injection),
            "global_gate_mean": float(global_gate.detach().mean().cpu()),
            "local_gate_mean": float(gy.detach().mean().cpu()),
            "reference_gate_mean": float(gr_mean),
            "low_freq_ratio_mean": float(low_freq_ratio.detach().mean().cpu()),
            "spectral_l2_mean": float(spectral_l2.detach().mean().cpu()),
            "direct_delta_norm": float(torch.norm((y_hat - y_base).detach()).cpu() / denom),
            "reference_content_active_ratio": float(reference_active.detach().mean().cpu()),
        }
        return y_hat.to(dtype=original_dtype)


class Scheme6ResidualSpectralAdapter(SimpleAdapter):
    def __init__(self, cfg, model, norm_module=None):
        super().__init__(cfg, model, norm_module)

        spectral_cfg = cfg.TTA.COSA.SPECTRAL
        validate_scheme6_usage(spectral_cfg)
        self.spectral_usage = str(getattr(spectral_cfg, "USAGE", "direct")).lower()
        self.spectral_repr = getattr(spectral_cfg, "REPR", "real_imag")
        self.context_builder = None
        self.context_projector = None
        self.raw_context_dim = 0
        self.effective_context_dim = 0
        self.use_context_projection = False

        if self._uses_context_adapter():
            self.context_builder = SpectralContextBuilder(
                buffer_context_size=self.buffer_context_size,
                spectral_k=getattr(spectral_cfg, "K", 8),
                mode=getattr(spectral_cfg, "MODE", "concat"),
                device=self.device,
                pred_len=cfg.DATA.PRED_LEN,
                n_vars=cfg.DATA.N_VAR,
                feature_source=getattr(spectral_cfg, "SOURCE", "residual_profile"),
                spectral_repr=self.spectral_repr,
                spectral_history_size=getattr(spectral_cfg, "HISTORY_SIZE", 8),
                include_power_ratio=_as_bool(getattr(spectral_cfg, "INCLUDE_POWER_RATIO", True)),
                include_residual_scale=_as_bool(getattr(spectral_cfg, "INCLUDE_RESIDUAL_STATS", True)),
            )
            self.raw_context_dim = int(self.context_builder.context_dim)
            self.effective_context_dim = self.raw_context_dim
            self._configure_context_projector(spectral_cfg)
            self.output_adapter = SimpleOutputAdapter(
                pred_len=cfg.DATA.PRED_LEN,
                buffer_context_size=self.effective_context_dim,
                n_vars=cfg.DATA.N_VAR,
                var_wise_gating=self.var_wise_gating,
                num_layers=self.adapter_layers,
                hidden_dim=self.hidden_dim,
            ).to(self.device)

        self.direct_corrector = Scheme6DirectSpectralCorrector(
            pred_len=cfg.DATA.PRED_LEN,
            n_vars=cfg.DATA.N_VAR,
            spectral_k=getattr(spectral_cfg, "K", 8),
            history_size=getattr(spectral_cfg, "HISTORY_SIZE", 8),
            correction_source=getattr(spectral_cfg, "CORRECTION_SOURCE", "residual"),
            gate_mode=getattr(spectral_cfg, "GATE_MODE", "unidirectional"),
            use_global_gate=_as_bool(getattr(spectral_cfg, "USE_GLOBAL_GATE", True)),
            global_gate_features=getattr(spectral_cfg, "GLOBAL_GATE_FEATURES", "hs,l2,low_freq_ratio"),
            global_gate_init_bias=getattr(spectral_cfg, "GLOBAL_GATE_INIT_BIAS", -6.0),
            window_type=getattr(spectral_cfg, "WINDOW_TYPE", "hann"),
            out_proj=getattr(spectral_cfg, "OUT_PROJ", "diagonal"),
            min_power_ratio=getattr(spectral_cfg, "MIN_POWER_RATIO_FOR_DIRECT", 0.20),
            enable_reference_content_injection=_as_bool(
                getattr(spectral_cfg, "ENABLE_REFERENCE_CONTENT_INJECTION", False)
            ),
            fft_precision=getattr(spectral_cfg, "FFT_PRECISION", "fp32"),
            fft_length_mode=getattr(spectral_cfg, "FFT_LENGTH_MODE", "exact"),
            enable_revin_on_spectral_path=_as_bool(
                getattr(spectral_cfg, "ENABLE_REVIN_ON_SPECTRAL_PATH", False)
            ),
        ).to(self.device)

        self.batch_logs = []
        self.scheme6_log_csv_path = getattr(spectral_cfg, "LOG_CSV_PATH", "")
        self._freeze_all_model_params()
        self._unfreeze_adapter_params()
        self._adapter_params = self._adapter_trainable_parameters()
        self.optimizer = get_optimizer(self._adapter_params, cfg.TTA)
        self.model_state, self.optimizer_state = self._copy_model_and_optimizer()

    def _uses_context_adapter(self):
        return getattr(self, "spectral_usage", "direct") == "context_plus_direct"

    def _adapter_trainable_parameters(self):
        params = []
        if self._uses_context_adapter():
            params.extend(self.output_adapter.parameters())
            if self.context_projector is not None:
                params.extend(self.context_projector.parameters())
        if getattr(self, "direct_corrector", None) is not None:
            params.extend(self.direct_corrector.parameters())
        return params

    def _unfreeze_adapter_params(self):
        if hasattr(self, "output_adapter"):
            for param in self.output_adapter.parameters():
                param.requires_grad_(self._uses_context_adapter())
        context_projector = getattr(self, "context_projector", None)
        if context_projector is not None:
            for param in context_projector.parameters():
                param.requires_grad_(self._uses_context_adapter())
        direct_corrector = getattr(self, "direct_corrector", None)
        if direct_corrector is not None:
            for param in direct_corrector.parameters():
                param.requires_grad_(True)

    def _get_all_models(self):
        models = [self.model, self.output_adapter]
        if self.context_projector is not None:
            models.append(self.context_projector)
        if self.direct_corrector is not None:
            models.append(self.direct_corrector)
        if self.norm_module is not None:
            models.append(self.norm_module)
        return models

    def _configure_context_projector(self, spectral_cfg):
        requested = _as_bool(getattr(spectral_cfg, "USE_CONTEXT_PROJECTION", True))
        projection_dim = int(getattr(spectral_cfg, "CONTEXT_PROJECTION_DIM", 8))
        if requested and self.raw_context_dim > projection_dim:
            self.effective_context_dim = projection_dim
            self.use_context_projection = True
            self.context_projector = nn.Linear(self.raw_context_dim, self.effective_context_dim).to(self.device)
            nn.init.xavier_uniform_(self.context_projector.weight, gain=0.1)
            nn.init.zeros_(self.context_projector.bias)

    def _project_context(self, context_data):
        context_data = context_data.to(self.device)
        if self.context_projector is None:
            return context_data
        batch_size, n_vars, raw_dim = context_data.shape
        if raw_dim != self.raw_context_dim:
            raise ValueError(f"context dim {raw_dim} does not match raw context dim {self.raw_context_dim}")
        flat = context_data.reshape(batch_size * n_vars, raw_dim)
        projected = self.context_projector(flat)
        return projected.reshape(batch_size, n_vars, self.effective_context_dim)

    def _get_individual_context_for_batch(self, batch_size, current_batch_idx):
        if self.context_builder is None:
            return None
        return self.context_builder.build(self.sample_history, batch_size).to(self.device)

    def _apply_prediction_adapter(self, prediction, raw_context=None):
        adapted = prediction
        if self._uses_context_adapter():
            if raw_context is None:
                raise ValueError("context_plus_direct requires context data")
            adapted = self.output_adapter(adapted, self._project_context(raw_context))
        return self.direct_corrector(adapted, self.sample_history)

    def _direct_stats(self):
        return self.direct_corrector.last_stats

    def update_memory_buffer(self, targets, prediction):
        batch_size = targets.shape[0]
        signed_residual = targets - prediction
        seq_mse = F.mse_loss(prediction, targets).item()
        batch_info = {
            "time_idx": self.current_time_idx,
            "target_mean": targets.mean().item(),
            "target_mean_var": targets.mean(dim=(0, 1)).detach().float().cpu(),
            "target_std_var": targets.std(dim=(0, 1), unbiased=False).detach().float().cpu(),
            "residual": seq_mse,
            "residual_profile": signed_residual.mean(dim=0).detach().float().cpu(),
            "residual_mean_var": signed_residual.mean(dim=(0, 1)).detach().float().cpu(),
            "residual_std_var": signed_residual.std(dim=(0, 1), unbiased=False).detach().float().cpu(),
        }
        self.sample_history.append(batch_info)
        self.current_time_idx += batch_size
        self.step_count += 1
        if not self.adapters_enabled:
            self.adapters_enabled = True
        return seq_mse

    def switch_model_to_train(self):
        self.model.eval()
        if self.norm_module is not None:
            self.norm_module.eval()
        self.output_adapter.train(self._uses_context_adapter())
        if self.context_projector is not None:
            self.context_projector.train(self._uses_context_adapter())
        self.direct_corrector.train()

    def switch_model_to_eval(self):
        self.model.eval()
        if self.norm_module is not None:
            self.norm_module.eval()
        self.output_adapter.eval()
        if self.context_projector is not None:
            self.context_projector.eval()
        self.direct_corrector.eval()

    @torch.enable_grad()
    def adapt_simple(self):
        if forecast is None:
            raise ImportError("forecast dependencies are required to run Scheme6ResidualSpectralAdapter")
        batch_start = 0
        batch_end = 0
        batch_idx = 0
        total_start_time = time.time()
        self.switch_model_to_eval()

        for _, inputs in enumerate(self.test_loader):
            enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all = prepare_inputs(inputs)

            while batch_end < len(enc_window_all):
                batch_wall_start = time.time()
                calculated_batch_size = None
                period = None
                dominant_freq = None

                if self.paas_enabled:
                    period_start = time.time()
                    enc_window_first = enc_window_all[batch_start]
                    period, calculated_batch_size, dominant_freq = self._calculate_period_and_batch_size(enc_window_first)
                    batch_size = calculated_batch_size
                    self.time_stats["paas_period_calculation"] += time.time() - period_start
                    self.time_counts["paas_period_calculation"] += 1
                else:
                    batch_size = getattr(self.cfg.TTA.COSA, "BATCH_SIZE", 64)
                    calculated_batch_size = batch_size

                batch_end = min(batch_start + batch_size, len(enc_window_all))
                batch_size = batch_end - batch_start

                if self.save_paas_csv:
                    self._save_paas_batch_info(
                        batch_idx=batch_idx,
                        batch_start=batch_start,
                        calculated_batch_size=calculated_batch_size,
                        actual_batch_size=batch_size,
                        period=period,
                        fft_dominant_freq=dominant_freq,
                    )

                self.cur_step += batch_size
                batch_inputs = (
                    enc_window_all[batch_start:batch_end],
                    enc_window_stamp_all[batch_start:batch_end],
                    dec_window_all[batch_start:batch_end],
                    dec_window_stamp_all[batch_start:batch_end],
                )

                pred_start = time.time()
                pred, ground_truth = forecast(self.cfg, batch_inputs, self.model, self.norm_module)
                original_pred = pred.clone()
                self.time_stats["base_prediction"] += time.time() - pred_start
                self.time_counts["base_prediction"] += 1

                raw_context = None
                if self.adapters_enabled:
                    if self._uses_context_adapter():
                        context_start = time.time()
                        raw_context = self._get_individual_context_for_batch(batch_size, batch_idx)
                        self.time_stats["context_generation"] += time.time() - context_start
                        self.time_counts["context_generation"] += 1

                    final_start = time.time()
                    with torch.no_grad():
                        pred = self._apply_prediction_adapter(pred, raw_context)
                    self.time_stats["final_prediction"] += time.time() - final_start
                    self.time_counts["final_prediction"] += 1

                if self.save_csv:
                    self._save_batch_predictions_to_csv(original_pred, pred, ground_truth, batch_idx)

                metric_start = time.time()
                mse = F.mse_loss(pred, ground_truth, reduction="none").mean(dim=(-2, -1)).detach().cpu().numpy()
                mae = F.l1_loss(pred, ground_truth, reduction="none").mean(dim=(-2, -1)).detach().cpu().numpy()
                self.time_stats["metric_computation"] += time.time() - metric_start
                self.time_counts["metric_computation"] += 1
                self.mse_all.append(mse)
                self.mae_all.append(mae)

                gradient_steps = 0
                if self.adapters_enabled:
                    effective_steps = min(self.adapt_steps, 5) if self.fast_adaptation else self.adapt_steps
                    adapt_start = time.time()
                    for step in range(effective_steps):
                        step_start = time.time()
                        self.n_adapt += 1
                        gradient_steps += 1
                        self.switch_model_to_train()

                        forward_start = time.time()
                        adapted_pred = self._apply_prediction_adapter(original_pred, raw_context)
                        self.time_stats["adapter_forward"] += time.time() - forward_start
                        self.time_counts["adapter_forward"] += 1

                        loss_start = time.time()
                        loss = F.mse_loss(adapted_pred, ground_truth)
                        l2_reg = sum(param.pow(2).sum() for param in self._adapter_params if param.requires_grad)
                        loss = loss + 1e-4 * l2_reg
                        self.time_stats["loss_computation"] += time.time() - loss_start
                        self.time_counts["loss_computation"] += 1

                        if self.adaptive_lr and self.fast_adaptation:
                            current_lr = self._adaptive_learning_rate(loss.item(), step, batch_idx)
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = current_lr

                        backward_start = time.time()
                        self.optimizer.zero_grad()
                        loss.backward()
                        max_norm = max(0.05, min(0.5, loss.item())) if self.fast_adaptation else 0.1
                        torch.nn.utils.clip_grad_norm_(self._adapter_params, max_norm=max_norm)
                        self.optimizer.step()
                        self.time_stats["backward_update"] += time.time() - backward_start
                        self.time_counts["backward_update"] += 1

                        self.switch_model_to_eval()
                        if (
                            self.fast_adaptation
                            and step > 2
                            and len(self.loss_history) >= 2
                            and abs(self.loss_history[-1] - self.loss_history[-2]) < self.convergence_threshold
                        ):
                            break

                        self.time_stats["per_adaptation_step"] += time.time() - step_start
                        self.time_counts["per_adaptation_step"] += 1

                    self.time_stats["total_adaptation"] += time.time() - adapt_start
                    self.time_counts["total_adaptation"] += 1

                residual = F.mse_loss(original_pred, ground_truth).item()
                buffer_start = time.time()
                self.update_memory_buffer(ground_truth, original_pred)
                self.time_stats["buffer_update"] += time.time() - buffer_start
                self.time_counts["buffer_update"] += 1

                direct_stats = self._direct_stats()
                self.batch_logs.append({
                    "batch_idx": batch_idx,
                    "batch_size": batch_size,
                    "spectral_usage": self.spectral_usage,
                    "spectral_repr": self.spectral_repr,
                    "spectral_mode": getattr(self.context_builder, "mode", ""),
                    "spectral_k": int(getattr(self.direct_corrector, "spectral_k", 0)),
                    "raw_context_dim": int(self.raw_context_dim),
                    "effective_context_dim": int(self.effective_context_dim),
                    "use_context_projection": int(self.use_context_projection),
                    "direct_spectral_enabled": int(direct_stats["direct_spectral_enabled"]),
                    "direct_correction_source": direct_stats["correction_source"],
                    "direct_spectral_gate_mode": direct_stats["spectral_gate_mode"],
                    "direct_window_type": direct_stats["window_type"],
                    "direct_out_proj": direct_stats["spectral_out_proj"],
                    "direct_fft_precision": direct_stats["fft_precision"],
                    "direct_fft_length_mode": direct_stats["fft_length_mode"],
                    "direct_use_global_gate": int(direct_stats["use_global_gate"]),
                    "direct_reference_content_injection": int(direct_stats["enable_reference_content_injection"]),
                    "direct_global_gate_mean": float(direct_stats["global_gate_mean"]),
                    "direct_local_gate_mean": float(direct_stats["local_gate_mean"]),
                    "direct_reference_gate_mean": float(direct_stats["reference_gate_mean"]),
                    "direct_low_freq_ratio_mean": float(direct_stats["low_freq_ratio_mean"]),
                    "direct_spectral_l2_mean": float(direct_stats["spectral_l2_mean"]),
                    "direct_delta_norm": float(direct_stats["direct_delta_norm"]),
                    "direct_reference_content_active_ratio": float(direct_stats["reference_content_active_ratio"]),
                    "residual": residual,
                    "gradient_steps": gradient_steps,
                    "mse": float(np.mean(mse)),
                    "mae": float(np.mean(mae)),
                    "wall_clock_time_ms": round((time.time() - batch_wall_start) * 1000.0, 3),
                })

                batch_start = batch_end
                batch_idx += 1

        self.time_stats["total_time"] = time.time() - total_start_time
        assert self.cur_step == len(self.test_data) - self.cfg.DATA.PRED_LEN - 1

        self.mse_all = np.concatenate(self.mse_all)
        self.mae_all = np.concatenate(self.mae_all)
        assert len(self.mse_all) == len(self.test_loader.dataset)

        if self.save_csv:
            self._export_predictions_to_csv()
        if self.save_paas_csv:
            self._export_paas_info_to_csv()
        self._export_scheme6_log()
        self._print_combined_results()

    def _export_scheme6_log(self):
        if not self.batch_logs:
            return
        if self.scheme6_log_csv_path:
            csv_path = self.scheme6_log_csv_path
        else:
            csv_dir = os.path.join(self.cfg.RESULT_DIR, "csv_predictions", "COSA_SCHEME6")
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = os.path.join(
                csv_dir,
                f"{self.cfg.MODEL.NAME}_{self.cfg.DATA.NAME}_pred{self.cfg.DATA.PRED_LEN}_batches.csv",
            )
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.batch_logs[0].keys()))
            writer.writeheader()
            writer.writerows(self.batch_logs)

    def _print_combined_results(self):
        total_params = sum(int(param.numel()) for _, param in self.named_parameters() if param.requires_grad)
        avg_times = {}
        for key in self.time_stats:
            if key != "total_time":
                count = self.time_counts[key] if self.time_counts[key] > 0 else 1
                avg_times[key] = self.time_stats[key] / count

        direct_rows = [row for row in self.batch_logs if row.get("direct_spectral_enabled")]

        def mean_direct(key):
            if not direct_rows:
                return 0.0
            return float(np.mean([float(row.get(key, 0.0)) for row in direct_rows]))

        adaptation_count = max(self.time_counts.get("total_adaptation", 1), 1)
        context_projection_params = (
            sum(int(param.numel()) for param in self.context_projector.parameters())
            if self.context_projector is not None
            else 0
        )
        combined_results = {
            "model": "Scheme6ResidualSpectralAdapter",
            "scheme6_statistics": {
                "spectral_usage": self.spectral_usage,
                "spectral_mode": getattr(self.context_builder, "mode", ""),
                "spectral_repr": self.spectral_repr,
                "spectral_k": int(getattr(self.direct_corrector, "spectral_k", 0)),
                "context_dim": int(self.effective_context_dim),
                "context_projection_params": int(context_projection_params),
                "direct_spectral_enabled": True,
                "correction_source": self._direct_stats()["correction_source"],
                "spectral_gate_mode": self._direct_stats()["spectral_gate_mode"],
                "window_type": self._direct_stats()["window_type"],
                "spectral_out_proj": self._direct_stats()["spectral_out_proj"],
                "fft_precision": self._direct_stats()["fft_precision"],
                "fft_length_mode": self._direct_stats()["fft_length_mode"],
                "use_global_gate": bool(self._direct_stats()["use_global_gate"]),
                "reference_content_injection": bool(self._direct_stats()["enable_reference_content_injection"]),
                "global_gate_mean": mean_direct("direct_global_gate_mean"),
                "local_gate_mean": mean_direct("direct_local_gate_mean"),
                "low_freq_ratio_mean": mean_direct("direct_low_freq_ratio_mean"),
                "spectral_l2_mean": mean_direct("direct_spectral_l2_mean"),
                "direct_delta_norm_mean": mean_direct("direct_delta_norm"),
                "reference_content_active_ratio": mean_direct("direct_reference_content_active_ratio"),
            },
            "time_statistics": {
                "adapter_operations": {
                    "context_generation_ms": round(avg_times.get("context_generation", 0) * 1000, 3),
                    "adapter_forward_pass_ms": round(avg_times.get("adapter_forward", 0) * 1000, 3),
                    "final_prediction_ms": round(avg_times.get("final_prediction", 0) * 1000, 3),
                },
                "adaptation_training": {
                    "loss_computation_ms": round(avg_times.get("loss_computation", 0) * 1000, 3),
                    "backward_update_ms": round(avg_times.get("backward_update", 0) * 1000, 3),
                    "per_adaptation_step_ms": round(avg_times.get("per_adaptation_step", 0) * 1000, 3),
                    "total_per_batch_ms": round(avg_times.get("total_adaptation", 0) * 1000, 3),
                    "adaptation_steps": self.adapt_steps,
                },
                "overall_stats": {
                    "total_time_seconds": round(self.time_stats["total_time"], 2),
                    "total_adaptations": int(self.n_adapt),
                    "avg_time_per_adaptation_ms": round(
                        self.time_stats.get("total_adaptation", 0) / adaptation_count * 1000,
                        3,
                    ),
                    "throughput_samples_per_sec": round(len(self.test_loader.dataset) / self.time_stats["total_time"], 1),
                },
            },
            "final_results": {
                "adaptation_count": int(self.n_adapt),
                "test_mse": float(self.mse_all.mean()),
                "test_mae": float(self.mae_all.mean()),
            },
            "parameters": {
                "total_params": total_params,
            },
        }
        print(json.dumps(combined_results, indent=2))


def build_adapter(cfg, model, norm_module=None):
    return Scheme6ResidualSpectralAdapter(cfg, model, norm_module)
