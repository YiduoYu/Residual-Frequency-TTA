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


def _init_near_zero(linear, std=1e-4):
    nn.init.normal_(linear.weight, mean=0.0, std=std)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class SpectralFusion(nn.Module):
    def __init__(self, condition_dim, num_windows, mode="softmax_gate"):
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.num_windows = int(num_windows)
        self.mode = mode

        if self.mode == "softmax_gate":
            self.gate = nn.Linear(self.condition_dim, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        elif self.mode == "concat_projection":
            self.projection = nn.Linear(self.condition_dim * self.num_windows, self.condition_dim)
            nn.init.zeros_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
            with torch.no_grad():
                for idx in range(self.num_windows):
                    start = idx * self.condition_dim
                    end = start + self.condition_dim
                    self.projection.weight[:, start:end] += torch.eye(self.condition_dim) / self.num_windows
        elif self.mode != "mean":
            raise ValueError(f"Unsupported spectral fusion mode: {self.mode}")

        self.last_weights = None

    def forward(self, window_features):
        if window_features.dim() != 4:
            raise ValueError(f"window_features must have shape [B, K, V, C], got {tuple(window_features.shape)}")
        batch_size, num_windows, n_vars, condition_dim = window_features.shape
        if num_windows != self.num_windows:
            raise ValueError(f"expected {self.num_windows} windows, got {num_windows}")
        if condition_dim != self.condition_dim:
            raise ValueError(f"expected condition dim {self.condition_dim}, got {condition_dim}")

        if self.mode == "mean":
            fused = window_features.mean(dim=1)
            weights = torch.full(
                (batch_size, n_vars, num_windows),
                1.0 / num_windows,
                dtype=window_features.dtype,
                device=window_features.device,
            )
        elif self.mode == "softmax_gate":
            scores = self.gate(window_features).squeeze(-1).permute(0, 2, 1)
            weights = torch.softmax(scores, dim=-1)
            fused = torch.sum(weights.permute(0, 2, 1).unsqueeze(-1) * window_features, dim=1)
        else:
            flattened = window_features.permute(0, 2, 1, 3).reshape(batch_size, n_vars, num_windows * condition_dim)
            fused = self.projection(flattened)
            weights = torch.full(
                (batch_size, n_vars, num_windows),
                1.0 / num_windows,
                dtype=window_features.dtype,
                device=window_features.device,
            )

        self.last_weights = weights.detach()
        return fused


class MultiWindowSpectralEncoder(nn.Module):
    def __init__(
        self,
        input_length,
        n_vars,
        condition_dim=8,
        encoder_type="multi_window",
        num_windows=3,
        window_types=None,
        feature_type="real_imag",
        pooling="low_mid_high",
        fusion="softmax_gate",
        num_frequency_bins=8,
        eps=1e-6,
    ):
        super().__init__()
        self.input_length = int(input_length)
        self.n_vars = int(n_vars)
        self.condition_dim = int(condition_dim)
        self.encoder_type = str(encoder_type).lower()
        self.feature_type = str(feature_type).lower()
        self.pooling = str(pooling).lower()
        self.num_frequency_bins = int(num_frequency_bins)
        self.eps = float(eps)

        windows = self._build_windows(window_types, num_windows)
        self.register_buffer("windows", windows)
        self.num_windows = int(windows.shape[0])

        component_count = {"magnitude": 1, "real_imag": 2, "mag_phase": 2}.get(self.feature_type)
        if component_count is None:
            raise ValueError(f"Unsupported spectral feature type: {self.feature_type}")
        if self.pooling != "low_mid_high":
            raise ValueError(f"Unsupported spectral pooling mode: {self.pooling}")

        self.pool_projection = nn.Linear(component_count * 3, self.condition_dim)
        nn.init.xavier_uniform_(self.pool_projection.weight, gain=0.1)
        nn.init.zeros_(self.pool_projection.bias)
        self.fusion = SpectralFusion(self.condition_dim, self.num_windows, mode=fusion)

        self.last_window_weights = []
        self.last_window_weight_tensor = None
        self.last_energy_bands = [0.0, 0.0, 0.0]

    def _build_windows(self, window_types, num_windows):
        if self.encoder_type == "multitaper":
            raise NotImplementedError("multitaper spectral encoding is reserved for a future DPSS window implementation")

        if self.encoder_type == "single_window":
            requested_types = [_as_list(window_types)[0] if _as_list(window_types) else "rect"]
        elif self.encoder_type == "multi_window":
            requested_types = _as_list(window_types) or ["hann", "hamming", "rect"]
            target_count = int(num_windows)
            if target_count <= 0:
                raise ValueError("num_windows must be positive")
            requested_types = [requested_types[idx % len(requested_types)] for idx in range(target_count)]
        else:
            raise ValueError(f"Unsupported spectral encoder type: {self.encoder_type}")

        bank = [self._make_window(name) for name in requested_types]
        return torch.stack(bank, dim=0)

    def _make_window(self, name):
        key = str(name).lower()
        if key in ("rect", "rectangular", "boxcar", "ones"):
            return torch.ones(self.input_length, dtype=torch.float32)
        if key == "hann":
            return torch.hann_window(self.input_length, periodic=False, dtype=torch.float32)
        if key == "hamming":
            return torch.hamming_window(self.input_length, periodic=False, dtype=torch.float32)
        if key == "blackman":
            return torch.blackman_window(self.input_length, periodic=False, dtype=torch.float32)
        raise ValueError(f"Unsupported spectral window type: {name}")

    def _prepare_input(self, z):
        if z.dim() == 1:
            z = z[:, None]
        if z.dim() == 2:
            z = z.unsqueeze(0)
        if z.dim() != 3:
            raise ValueError(f"spectral input must have shape [P, V] or [B, P, V], got {tuple(z.shape)}")

        if z.shape[-1] == 1 and self.n_vars > 1:
            z = z.expand(*z.shape[:-1], self.n_vars)
        if z.shape[-1] != self.n_vars:
            raise ValueError(f"spectral input has {z.shape[-1]} variables, expected {self.n_vars}")

        length = z.shape[1]
        if length < self.input_length:
            pad = torch.zeros(
                z.shape[0],
                self.input_length - length,
                self.n_vars,
                dtype=z.dtype,
                device=z.device,
            )
            z = torch.cat([z, pad], dim=1)
        elif length > self.input_length:
            z = z[:, -self.input_length :, :]
        return z

    def _select_frequency_bins(self, coeff):
        selected = coeff[:, :, 1 : self.num_frequency_bins + 1, :]
        if selected.shape[2] < self.num_frequency_bins:
            pad = torch.zeros(
                selected.shape[0],
                selected.shape[1],
                self.num_frequency_bins - selected.shape[2],
                selected.shape[3],
                dtype=selected.dtype,
                device=selected.device,
            )
            selected = torch.cat([selected, pad], dim=2)
        return selected

    def _pool_low_mid_high(self, features):
        pooled = []
        freq_len = features.shape[-2]
        for start, end in self._band_slices(freq_len):
            if end <= start:
                pooled.append(torch.zeros_like(features[..., 0, :]))
            else:
                pooled.append(features[..., start:end, :].mean(dim=-2))
        return torch.cat(pooled, dim=-1)

    def _band_slices(self, freq_len):
        cuts = torch.linspace(0, freq_len, steps=4).round().long().tolist()
        slices = []
        for idx in range(3):
            start = int(cuts[idx])
            end = int(cuts[idx + 1])
            if end <= start and freq_len > 0:
                start = min(idx, freq_len - 1)
                end = start + 1
            slices.append((start, end))
        return slices

    def _update_energy_bands(self, selected):
        with torch.no_grad():
            power = torch.abs(selected).pow(2)
            total = power.sum().clamp_min(self.eps)
            bands = []
            for start, end in self._band_slices(power.shape[2]):
                if end <= start:
                    bands.append(0.0)
                else:
                    bands.append(float((power[:, :, start:end, :].sum() / total).detach().cpu()))
            self.last_energy_bands = bands

    def forward(self, z, batch_size=None):
        z = self._prepare_input(z)
        z = z.to(device=self.windows.device, dtype=self.windows.dtype)
        z_centered = z - z.mean(dim=1, keepdim=True)
        scale = z_centered.std(dim=1, unbiased=False).clamp_min(self.eps)

        windowed = z_centered.unsqueeze(1) * self.windows[None, :, :, None]
        coeff = torch.fft.rfft(windowed, dim=2)
        selected = self._select_frequency_bins(coeff)
        selected = selected / scale[:, None, None, :]
        self._update_energy_bands(selected)

        if self.feature_type == "magnitude":
            features = torch.log1p(torch.abs(selected)).unsqueeze(-1)
        elif self.feature_type == "real_imag":
            features = torch.stack([selected.real, selected.imag], dim=-1)
        else:
            features = torch.stack([torch.log1p(torch.abs(selected)), torch.angle(selected)], dim=-1)

        features = torch.nan_to_num(features.permute(0, 1, 3, 2, 4), nan=0.0, posinf=0.0, neginf=0.0)
        pooled = self._pool_low_mid_high(features)
        window_features = self.pool_projection(pooled)
        fused = self.fusion(window_features)

        if batch_size is not None and fused.shape[0] == 1 and batch_size != 1:
            fused = fused.expand(int(batch_size), -1, -1)
            if self.fusion.last_weights is not None:
                self.fusion.last_weights = self.fusion.last_weights.expand(int(batch_size), -1, -1)

        self.last_window_weight_tensor = self.fusion.last_weights
        if self.last_window_weight_tensor is not None:
            self.last_window_weights = (
                self.last_window_weight_tensor.detach().mean(dim=(0, 1)).cpu().tolist()
            )
        return fused


class SpectralTuningBackend(nn.Module):
    SUPPORTED_BACKENDS = ("film", "ia3", "adapter", "lora")

    def __init__(
        self,
        pred_len,
        n_vars,
        condition_dim,
        backend_types="film,ia3,adapter,lora",
        var_wise_gating=False,
        adapter_layers=1,
        adapter_hidden_dim=64,
        backend_gate="softmax",
        mixture_mode="residual",
        film_eps=0.1,
        ia3_eps=0.1,
        adapter_conditioning="residual_gate",
        lora_enabled=True,
        lora_rank=4,
        lora_alpha=8,
        lora_targets=None,
    ):
        super().__init__()
        self.pred_len = int(pred_len)
        self.n_vars = int(n_vars)
        self.condition_dim = int(condition_dim)
        self.backend_types = [item.lower() for item in _as_list(backend_types)]
        self.backend_gate_type = str(backend_gate).lower()
        self.mixture_mode = str(mixture_mode).lower()
        self.film_eps = float(film_eps)
        self.ia3_eps = float(ia3_eps)
        self.adapter_conditioning = adapter_conditioning
        self.lora_enabled = _as_bool(lora_enabled)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_targets = _as_list(lora_targets)

        if not self.backend_types:
            raise ValueError("At least one balanced tuning backend must be enabled")
        unknown = sorted(set(self.backend_types) - set(self.SUPPORTED_BACKENDS))
        if unknown:
            raise ValueError(f"Unsupported balanced tuning backend(s): {unknown}")
        if "lora" in self.backend_types and not self.lora_enabled:
            self.backend_types.remove("lora")
        if not self.backend_types:
            raise ValueError("No backend remains enabled after applying lora_enabled")
        if self.mixture_mode not in ("residual", "direct"):
            raise ValueError(f"Unsupported mixture mode: {self.mixture_mode}")
        if self.backend_gate_type != "softmax":
            raise ValueError(f"Unsupported backend gate: {self.backend_gate_type}")
        if self.adapter_conditioning != "residual_gate":
            raise ValueError(f"Unsupported adapter conditioning: {self.adapter_conditioning}")

        if "film" in self.backend_types:
            self.film_generator = nn.Linear(self.condition_dim, 2)
            _init_near_zero(self.film_generator)
        if "ia3" in self.backend_types:
            self.ia3_generator = nn.Linear(self.condition_dim, 1)
            _init_near_zero(self.ia3_generator)
        if "adapter" in self.backend_types:
            self.adapter = SimpleOutputAdapter(
                pred_len=self.pred_len,
                buffer_context_size=self.condition_dim,
                n_vars=self.n_vars,
                var_wise_gating=var_wise_gating,
                num_layers=adapter_layers,
                hidden_dim=adapter_hidden_dim,
            )
            with torch.no_grad():
                self.adapter.gate.fill_(1e-4)
            self.adapter_gate = nn.Linear(self.condition_dim, 1)
            _init_near_zero(self.adapter_gate)
        if "lora" in self.backend_types:
            self.lora_down = nn.Linear(self.pred_len, self.lora_rank, bias=False)
            self.lora_up = nn.Linear(self.lora_rank, self.pred_len, bias=False)
            nn.init.xavier_uniform_(self.lora_down.weight, gain=0.1)
            nn.init.xavier_uniform_(self.lora_up.weight, gain=1e-4)
            self.lora_gate = nn.Linear(self.condition_dim, 1)
            _init_near_zero(self.lora_gate)

        self.backend_gate = nn.Linear(self.condition_dim, len(self.backend_types))
        nn.init.zeros_(self.backend_gate.weight)
        nn.init.zeros_(self.backend_gate.bias)

        self.last_backend_weights = {}
        self.last_backend_weight_tensor = None
        self.last_adaptation_norm = 0.0

    def _prepare_condition(self, h, s):
        if h.dim() != 3:
            raise ValueError(f"balanced tuning expects h with shape [B, L, V], got {tuple(h.shape)}")
        if s.dim() == 2:
            s = s.unsqueeze(1).expand(-1, h.shape[-1], -1)
        if s.dim() != 3:
            raise ValueError(f"spectral condition must have shape [B, V, C] or [B, C], got {tuple(s.shape)}")
        if s.shape[0] != h.shape[0]:
            raise ValueError(f"condition batch size {s.shape[0]} does not match h batch size {h.shape[0]}")
        if s.shape[1] == 1 and h.shape[-1] > 1:
            s = s.expand(-1, h.shape[-1], -1)
        if s.shape[1] != h.shape[-1]:
            raise ValueError(f"condition has {s.shape[1]} variables, h has {h.shape[-1]}")
        if s.shape[-1] != self.condition_dim:
            raise ValueError(f"condition dim {s.shape[-1]} does not match {self.condition_dim}")
        return s.to(device=h.device, dtype=h.dtype)

    def _film(self, h, s):
        raw = self.film_generator(s)
        gamma = 1.0 + self.film_eps * torch.tanh(raw[..., 0]).unsqueeze(1)
        beta = self.film_eps * torch.tanh(raw[..., 1]).unsqueeze(1)
        return gamma * h + beta

    def _ia3(self, h, s):
        raw = self.ia3_generator(s).squeeze(-1)
        gate = 1.0 + self.ia3_eps * torch.tanh(raw).unsqueeze(1)
        return gate * h

    def _adapter(self, h, s):
        adapter_out = self.adapter(h, s)
        residual = adapter_out - h
        gate = torch.sigmoid(self.adapter_gate(s).squeeze(-1)).unsqueeze(1)
        return h + gate * residual

    def _lora(self, h, s):
        batch_size, pred_len, n_vars = h.shape
        flat = h.transpose(1, 2).reshape(batch_size * n_vars, pred_len)
        residual = self.lora_up(self.lora_down(flat)) * (self.lora_alpha / max(self.lora_rank, 1))
        residual = residual.reshape(batch_size, n_vars, pred_len).transpose(1, 2)
        gate = torch.sigmoid(self.lora_gate(s).squeeze(-1)).unsqueeze(1)
        return h + gate * residual

    def forward(self, h, s, **kwargs):
        s = self._prepare_condition(h, s)
        outputs = []
        for backend_type in self.backend_types:
            if backend_type == "film":
                outputs.append(self._film(h, s))
            elif backend_type == "ia3":
                outputs.append(self._ia3(h, s))
            elif backend_type == "adapter":
                outputs.append(self._adapter(h, s))
            elif backend_type == "lora":
                outputs.append(self._lora(h, s))

        stacked = torch.stack(outputs, dim=-1)
        scores = self.backend_gate(s)
        weights = torch.softmax(scores, dim=-1)
        mix_weights = weights.unsqueeze(1)

        if self.mixture_mode == "residual":
            adapted = h + torch.sum(mix_weights * (stacked - h.unsqueeze(-1)), dim=-1)
        else:
            adapted = torch.sum(mix_weights * stacked, dim=-1)

        self.last_backend_weight_tensor = weights.detach()
        mean_weights = self.last_backend_weight_tensor.mean(dim=(0, 1)).cpu().tolist()
        self.last_backend_weights = {
            backend_type: float(mean_weights[idx]) for idx, backend_type in enumerate(self.backend_types)
        }
        self.last_adaptation_norm = float(torch.norm((adapted - h).detach()).cpu() / max(h.numel(), 1))
        return adapted


class SpectralBalancedCOSAAdapter(SimpleAdapter):
    def __init__(self, cfg, model, norm_module=None):
        super().__init__(cfg, model, norm_module)
        spectral_cfg = cfg.TTA.COSA.SPECTRAL
        balanced_cfg = cfg.TTA.COSA.BALANCED

        condition_dim = int(getattr(spectral_cfg, "CONDITION_DIM", 8))
        feature_type = getattr(spectral_cfg, "FEATURE_TYPE", getattr(spectral_cfg, "REPR", "real_imag"))
        self.spectral_usage = "balanced_tuning"
        self.spectral_encoder = MultiWindowSpectralEncoder(
            input_length=cfg.DATA.PRED_LEN,
            n_vars=cfg.DATA.N_VAR,
            condition_dim=condition_dim,
            encoder_type=getattr(spectral_cfg, "ENCODER_TYPE", "multi_window"),
            num_windows=getattr(spectral_cfg, "NUM_WINDOWS", 3),
            window_types=getattr(spectral_cfg, "WINDOW_TYPES", "hann,hamming,rect"),
            feature_type=feature_type,
            pooling=getattr(spectral_cfg, "POOLING", "low_mid_high"),
            fusion=getattr(spectral_cfg, "FUSION", "softmax_gate"),
            num_frequency_bins=getattr(spectral_cfg, "K", 8),
        ).to(self.device)

        self.output_adapter = SpectralTuningBackend(
            pred_len=cfg.DATA.PRED_LEN,
            n_vars=cfg.DATA.N_VAR,
            condition_dim=condition_dim,
            backend_types=getattr(balanced_cfg, "BACKENDS", "film,ia3,adapter,lora"),
            var_wise_gating=self.var_wise_gating,
            adapter_layers=self.adapter_layers,
            adapter_hidden_dim=self.hidden_dim,
            backend_gate=getattr(balanced_cfg, "BACKEND_GATE", "softmax"),
            mixture_mode=getattr(balanced_cfg, "MIXTURE_MODE", "residual"),
            film_eps=getattr(balanced_cfg, "FILM_EPS", 0.1),
            ia3_eps=getattr(balanced_cfg, "IA3_EPS", 0.1),
            adapter_conditioning=getattr(balanced_cfg, "ADAPTER_CONDITIONING", "residual_gate"),
            lora_enabled=_as_bool(getattr(balanced_cfg, "LORA_ENABLED", True)),
            lora_rank=getattr(balanced_cfg, "LORA_RANK", 4),
            lora_alpha=getattr(balanced_cfg, "LORA_ALPHA", 8),
            lora_targets=getattr(balanced_cfg, "LORA_TARGETS", "query_projection,value_projection"),
        ).to(self.device)

        self._adapter_params = list(self.spectral_encoder.parameters()) + list(self.output_adapter.parameters())
        self.optimizer = get_optimizer(self._adapter_params, cfg.TTA)
        self.model_state, self.optimizer_state = self._copy_model_and_optimizer()

        self.batch_logs = []
        self.log_gates = _as_bool(getattr(balanced_cfg, "LOG_GATES", True))
        self.spectral_log_csv_path = getattr(spectral_cfg, "LOG_CSV_PATH", "")
        self.condition_dim = condition_dim

    def _unfreeze_adapter_params(self):
        output_adapter = getattr(self, "output_adapter", None)
        if output_adapter is not None:
            for param in output_adapter.parameters():
                param.requires_grad_(True)
        spectral_encoder = getattr(self, "spectral_encoder", None)
        if spectral_encoder is not None:
            for param in spectral_encoder.parameters():
                param.requires_grad_(True)

    def switch_model_to_train(self):
        self.model.eval()
        if self.norm_module is not None:
            self.norm_module.eval()
        self.output_adapter.train()
        self.spectral_encoder.train()

    def switch_model_to_eval(self):
        self.model.eval()
        if self.norm_module is not None:
            self.norm_module.eval()
        self.output_adapter.eval()
        self.spectral_encoder.eval()

    def _spectral_profile_from_history(self):
        profiles = []
        history_size = int(getattr(self.cfg.TTA.COSA.SPECTRAL, "HISTORY_SIZE", 8))
        for entry in reversed(self.sample_history):
            profile = entry.get("residual_profile")
            if profile is not None:
                profiles.append(torch.as_tensor(profile, dtype=torch.float32, device=self.device))
            if len(profiles) >= history_size:
                break

        if not profiles:
            return torch.zeros(self.cfg.DATA.PRED_LEN, self.cfg.DATA.N_VAR, dtype=torch.float32, device=self.device)

        profile = torch.stack(profiles, dim=0).mean(dim=0)
        expected_shape = (self.cfg.DATA.PRED_LEN, self.cfg.DATA.N_VAR)
        if tuple(profile.shape) != expected_shape:
            raise ValueError(f"residual_profile has shape {tuple(profile.shape)}, expected {expected_shape}")
        return profile

    def _get_spectral_condition_for_batch(self, batch_size):
        profile = self._spectral_profile_from_history()
        return self.spectral_encoder(profile, batch_size=batch_size)

    def update_memory_buffer(self, targets, prediction):
        batch_size = targets.shape[0]
        signed_residual = targets - prediction
        residual_profile = signed_residual.mean(dim=0)
        residual_mean_var = signed_residual.mean(dim=(0, 1))
        residual_std_var = signed_residual.std(dim=(0, 1), unbiased=False)
        target_mean_var = targets.mean(dim=(0, 1))
        target_std_var = targets.std(dim=(0, 1), unbiased=False)
        seq_mse = F.mse_loss(prediction, targets).item()

        batch_info = {
            "time_idx": self.current_time_idx,
            "target_mean": targets.mean().item(),
            "target_mean_var": target_mean_var.detach().float().cpu(),
            "target_std_var": target_std_var.detach().float().cpu(),
            "residual": seq_mse,
            "residual_profile": residual_profile.detach().float().cpu(),
            "residual_mean_var": residual_mean_var.detach().float().cpu(),
            "residual_std_var": residual_std_var.detach().float().cpu(),
        }
        self.sample_history.append(batch_info)
        self.current_time_idx += batch_size
        self.step_count += 1
        if not self.adapters_enabled:
            self.adapters_enabled = True
        return seq_mse

    @torch.enable_grad()
    def adapt_simple(self):
        if forecast is None:
            raise ImportError("forecast dependencies are required to run SpectralBalancedCOSAAdapter.adapt_simple")
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

                spectral_condition = None
                if self.adapters_enabled:
                    context_start = time.time()
                    with torch.no_grad():
                        spectral_condition = self._get_spectral_condition_for_batch(batch_size)
                    self.time_stats["context_generation"] += time.time() - context_start
                    self.time_counts["context_generation"] += 1

                    final_start = time.time()
                    with torch.no_grad():
                        pred = self.output_adapter(pred, spectral_condition)
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

                residual = F.mse_loss(original_pred, ground_truth).item()
                gradient_steps = 0
                if self.adapters_enabled:
                    effective_steps = self.adapt_steps
                    if self.fast_adaptation:
                        effective_steps = min(self.adapt_steps, 5)

                    adapt_start = time.time()
                    for step in range(effective_steps):
                        step_start = time.time()
                        self.n_adapt += 1
                        gradient_steps += 1
                        self.switch_model_to_train()

                        forward_start = time.time()
                        condition_for_training = self._get_spectral_condition_for_batch(batch_size)
                        adapted_pred = self.output_adapter(original_pred, condition_for_training)
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
                        if self.fast_adaptation:
                            max_norm = max(0.05, min(0.5, loss.item()))
                            torch.nn.utils.clip_grad_norm_(self._adapter_params, max_norm=max_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(self._adapter_params, max_norm=0.1)
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

                buffer_start = time.time()
                self.update_memory_buffer(ground_truth, original_pred)
                self.time_stats["buffer_update"] += time.time() - buffer_start
                self.time_counts["buffer_update"] += 1

                if self.log_gates:
                    energy = self.spectral_encoder.last_energy_bands
                    self.batch_logs.append({
                        "batch_idx": batch_idx,
                        "batch_size": batch_size,
                        "spectral_usage": self.spectral_usage,
                        "spectral_encoder_type": self.spectral_encoder.encoder_type,
                        "spectral_feature_type": self.spectral_encoder.feature_type,
                        "spectral_pooling": self.spectral_encoder.pooling,
                        "spectral_fusion": self.spectral_encoder.fusion.mode,
                        "spectral_k": int(self.spectral_encoder.num_frequency_bins),
                        "num_windows": int(self.spectral_encoder.num_windows),
                        "condition_dim": int(self.condition_dim),
                        "window_fusion_weights": json.dumps(self.spectral_encoder.last_window_weights),
                        "backend_mixture_weights": json.dumps(self.output_adapter.last_backend_weights),
                        "adaptation_delta_norm": float(self.output_adapter.last_adaptation_norm),
                        "spectral_energy_low": float(energy[0]),
                        "spectral_energy_mid": float(energy[1]),
                        "spectral_energy_high": float(energy[2]),
                        "backends": ",".join(self.output_adapter.backend_types),
                        "mixture_mode": self.output_adapter.mixture_mode,
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
        self._export_balanced_log()
        self._print_combined_results()

    def _export_balanced_log(self):
        if not self.batch_logs:
            return
        if self.spectral_log_csv_path:
            csv_path = self.spectral_log_csv_path
        else:
            csv_dir = os.path.join(self.cfg.RESULT_DIR, "csv_predictions", "COSA_BALANCED")
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = os.path.join(
                csv_dir,
                f"{self.cfg.MODEL.NAME}_{self.cfg.DATA.NAME}_pred{self.cfg.DATA.PRED_LEN}_batches.csv",
            )
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.batch_logs[0].keys()))
            writer.writeheader()
            writer.writerows(self.batch_logs)

    def _mean_json_list_from_logs(self, key):
        values = []
        for row in self.batch_logs:
            raw = row.get(key)
            if not raw:
                continue
            parsed = json.loads(raw)
            if not parsed:
                continue
            values.append(parsed)
        if not values:
            return []
        arr = np.asarray(values, dtype=np.float64)
        return arr.mean(axis=0).tolist()

    def _mean_backend_weights_from_logs(self):
        values = []
        for row in self.batch_logs:
            raw = row.get("backend_mixture_weights")
            if not raw:
                continue
            parsed = json.loads(raw)
            if parsed:
                values.append(parsed)
        if not values:
            return {}
        backends = self.output_adapter.backend_types
        return {
            backend: float(np.mean([item.get(backend, 0.0) for item in values]))
            for backend in backends
        }

    def _print_combined_results(self):
        total_params = sum(int(param.numel()) for _, param in self.named_parameters() if param.requires_grad)
        avg_times = {}
        for key in self.time_stats:
            if key != "total_time":
                count = self.time_counts[key] if self.time_counts[key] > 0 else 1
                avg_times[key] = self.time_stats[key] / count

        adaptation_count = max(self.time_counts.get("total_adaptation", 1), 1)
        energy = self.spectral_encoder.last_energy_bands
        delta_norms = [row["adaptation_delta_norm"] for row in self.batch_logs] if self.batch_logs else []

        combined_results = {
            "model": "SpectralBalancedCOSAAdapter",
            "spectral_statistics": {
                "spectral_usage": self.spectral_usage,
                "spectral_mode": self.spectral_encoder.encoder_type,
                "spectral_k": int(self.spectral_encoder.num_frequency_bins),
                "context_dim": int(self.condition_dim),
                "spectral_repr": self.spectral_encoder.feature_type,
                "spectral_fusion": self.spectral_encoder.fusion.mode,
                "num_windows": int(self.spectral_encoder.num_windows),
                "spectral_energy_low": float(energy[0]),
                "spectral_energy_mid": float(energy[1]),
                "spectral_energy_high": float(energy[2]),
                "window_fusion_weights_mean": self._mean_json_list_from_logs("window_fusion_weights"),
            },
            "balanced_statistics": {
                "backends": self.output_adapter.backend_types,
                "backend_gate": self.output_adapter.backend_gate_type,
                "mixture_mode": self.output_adapter.mixture_mode,
                "backend_weights_mean": self._mean_backend_weights_from_logs(),
                "adaptation_delta_norm_mean": float(np.mean(delta_norms)) if delta_norms else 0.0,
                "log_gates": bool(self.log_gates),
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

    def adapt(self):
        self.adapt_simple()


def build_adapter(cfg, model, norm_module=None):
    return SpectralBalancedCOSAAdapter(cfg, model, norm_module)
