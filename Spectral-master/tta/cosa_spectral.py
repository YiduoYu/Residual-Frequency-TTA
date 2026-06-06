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


def validate_spectral_usage(spectral_cfg):
    spectral_usage = getattr(spectral_cfg, "USAGE", "context_only")
    direct_enabled = _as_bool(getattr(spectral_cfg, "ENABLE_DIRECT_CORRECTION", False))
    if spectral_usage != "context_only":
        raise ValueError(
            "Spectral COSA only supports USAGE='context_only'; "
            f"got {spectral_usage!r}."
        )
    if direct_enabled:
        raise ValueError(
            "ENABLE_DIRECT_CORRECTION must remain False. "
            "This implementation feeds spectral residual features only through COSA context."
        )


class SpectralContextBuilder:
    def __init__(
        self,
        buffer_context_size: int,
        spectral_k: int,
        mode: str,
        device,
        pred_len: int,
        n_vars: int,
        feature_source: str = "residual_profile",
        spectral_repr: str = "real_imag",
        spectral_history_size: int = 8,
        include_power_ratio: bool = True,
        include_residual_scale: bool = True,
        eps: float = 1e-6,
    ):
        if mode not in ("spectral", "concat"):
            raise ValueError(f"Unsupported spectral context mode: {mode}")
        if feature_source != "residual_profile":
            raise ValueError(f"Unsupported spectral feature source: {feature_source}")
        if spectral_repr not in ("real_imag", "mag_phase"):
            raise ValueError(f"Unsupported spectral representation: {spectral_repr}")

        self.buffer_context_size = int(buffer_context_size)
        self.spectral_k = int(spectral_k)
        self.mode = mode
        self.device = torch.device(device)
        self.pred_len = int(pred_len)
        self.n_vars = int(n_vars)
        self.feature_source = feature_source
        self.spectral_repr = spectral_repr
        self.spectral_history_size = int(spectral_history_size)
        self.include_power_ratio = bool(include_power_ratio)
        self.include_residual_scale = bool(include_residual_scale)
        self.eps = float(eps)
        self.last_power_ratio_mean = 0.0

        repr_width = 2 if self.spectral_repr == "real_imag" else 3
        extra_width = int(self.include_power_ratio) + 2 * int(self.include_residual_scale)
        self.spectral_dim = self.spectral_k * repr_width + extra_width
        if self.mode == "spectral":
            self.context_dim = self.spectral_dim
        else:
            self.context_dim = self.buffer_context_size + self.spectral_dim

    def build(self, sample_history, batch_size):
        spectral_context = self._spectral_features(sample_history)
        spectral_context = spectral_context.unsqueeze(0).expand(batch_size, -1, -1)
        if self.mode == "spectral":
            return spectral_context

        mean_context = self._variable_mean_context(sample_history, batch_size)
        return torch.cat([mean_context, spectral_context], dim=-1)

    def _variable_mean_context(self, sample_history, batch_size):
        if len(sample_history) == 0:
            context = torch.zeros(self.n_vars, self.buffer_context_size, dtype=torch.float32, device=self.device)
            return context.unsqueeze(0).expand(batch_size, -1, -1)

        vectors = []
        history_size = min(self.buffer_context_size, len(sample_history))
        for i in range(history_size):
            entry = sample_history[-(i + 1)]
            value = entry.get("target_mean_var")
            if value is None:
                scalar = float(entry.get("target_mean", 0.0))
                value = torch.full((self.n_vars,), scalar, dtype=torch.float32)
            vectors.append(torch.as_tensor(value, dtype=torch.float32, device=self.device).flatten())

        if len(vectors) < self.buffer_context_size:
            pad_value = vectors[-1] if vectors else torch.zeros(self.n_vars, dtype=torch.float32, device=self.device)
            vectors.extend([pad_value] * (self.buffer_context_size - len(vectors)))

        context = torch.stack(vectors, dim=-1)
        if context.shape[0] != self.n_vars:
            raise ValueError(f"target_mean_var has {context.shape[0]} variables, expected {self.n_vars}")
        return context.unsqueeze(0).expand(batch_size, -1, -1)

    def _spectral_features(self, sample_history):
        profiles = []
        for entry in reversed(sample_history):
            profile = entry.get("residual_profile")
            if profile is not None:
                profiles.append(torch.as_tensor(profile, dtype=torch.float32, device=self.device))
            if len(profiles) >= self.spectral_history_size:
                break

        if len(profiles) == 0:
            self.last_power_ratio_mean = 0.0
            return torch.zeros(self.n_vars, self.spectral_dim, dtype=torch.float32, device=self.device)

        profile = torch.stack(profiles, dim=0).mean(dim=0)
        if profile.shape != (self.pred_len, self.n_vars):
            raise ValueError(f"residual_profile has shape {tuple(profile.shape)}, expected {(self.pred_len, self.n_vars)}")

        profile_centered = profile - profile.mean(dim=0, keepdim=True)
        coeff = torch.fft.rfft(profile_centered, dim=0)
        selected = coeff[1:self.spectral_k + 1]
        if selected.shape[0] < self.spectral_k:
            pad = torch.zeros(
                self.spectral_k - selected.shape[0],
                self.n_vars,
                dtype=selected.dtype,
                device=self.device,
            )
            selected = torch.cat([selected, pad], dim=0)

        residual_scale = profile_centered.std(dim=0, unbiased=False).clamp_min(self.eps)
        features = []
        if self.spectral_repr == "real_imag":
            features.append(selected.real.T / residual_scale[:, None])
            features.append(selected.imag.T / residual_scale[:, None])
        else:
            magnitude = torch.log1p(torch.abs(selected)).T
            phase = torch.angle(selected).T
            features.extend([magnitude, torch.cos(phase), torch.sin(phase)])

        if self.include_power_ratio:
            non_dc = coeff[1:]
            total_power = torch.sum(torch.abs(non_dc) ** 2, dim=0).clamp_min(self.eps)
            selected_power = torch.sum(torch.abs(selected) ** 2, dim=0)
            power_ratio = (selected_power / total_power).clamp_min(0.0)
            self.last_power_ratio_mean = float(power_ratio.mean().detach().cpu())
            features.append(power_ratio[:, None])
        else:
            self.last_power_ratio_mean = 0.0

        if self.include_residual_scale:
            residual_mean = profile.mean(dim=0)
            residual_std = profile.std(dim=0, unbiased=False)
            features.extend([residual_mean[:, None], residual_std[:, None]])

        context = torch.cat(features, dim=-1)
        return torch.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)


class SpectralCOSAAdapter(SimpleAdapter):
    def __init__(self, cfg, model, norm_module=None):
        validate_spectral_usage(cfg.TTA.COSA.SPECTRAL)
        super().__init__(cfg, model, norm_module)

        spectral_cfg = cfg.TTA.COSA.SPECTRAL
        self.spectral_usage = getattr(spectral_cfg, "USAGE", "context_only")
        self.spectral_source = getattr(spectral_cfg, "SOURCE", "residual_profile")
        self.spectral_repr = getattr(spectral_cfg, "REPR", "real_imag")
        self.use_context_projection = False
        self.context_projector = None

        self.context_builder = SpectralContextBuilder(
            buffer_context_size=self.buffer_context_size,
            spectral_k=getattr(spectral_cfg, "K", 8),
            mode=getattr(spectral_cfg, "MODE", "concat"),
            device=self.device,
            pred_len=cfg.DATA.PRED_LEN,
            n_vars=cfg.DATA.N_VAR,
            feature_source=self.spectral_source,
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
        self._unfreeze_adapter_params()
        self._adapter_params = self._adapter_trainable_parameters()
        self.optimizer = get_optimizer(self._adapter_params, cfg.TTA)
        self.model_state, self.optimizer_state = self._copy_model_and_optimizer()

        self.batch_logs = []
        self.spectral_log_csv_path = getattr(spectral_cfg, "LOG_CSV_PATH", "")
        self.context_projection_params = self._context_projection_param_count()
        self.param_budget_multiplier = float(getattr(spectral_cfg, "PARAM_BUDGET_MULTIPLIER", 1.25))
        self.context_projection_dim = int(getattr(spectral_cfg, "CONTEXT_PROJECTION_DIM", self.effective_context_dim))
        self.max_context_dim_for_full_adapter = int(
            getattr(spectral_cfg, "MAX_CONTEXT_DIM_FOR_FULL_ADAPTER", self.effective_context_dim)
        )
        self.estimated_context_param_increase = self._estimate_context_param_increase()
        self._warn_if_context_budget_exceeded()

    def _configure_context_projector(self, spectral_cfg):
        requested = _as_bool(getattr(spectral_cfg, "USE_CONTEXT_PROJECTION", True))
        projection_dim = int(getattr(spectral_cfg, "CONTEXT_PROJECTION_DIM", 8))
        if requested and self.raw_context_dim > projection_dim:
            self.effective_context_dim = projection_dim
            self.use_context_projection = True
            self.context_projector = nn.Linear(self.raw_context_dim, self.effective_context_dim).to(self.device)
            nn.init.xavier_uniform_(self.context_projector.weight, gain=0.1)
            nn.init.zeros_(self.context_projector.bias)

    def _adapter_trainable_parameters(self):
        params = list(self.output_adapter.parameters())
        if self.context_projector is not None:
            params.extend(self.context_projector.parameters())
        return params

    def _context_projection_param_count(self):
        if self.context_projector is None:
            return 0
        return sum(int(param.numel()) for param in self.context_projector.parameters())

    def _adapter_param_count_for_context(self, context_dim):
        input_dim = self.cfg.DATA.PRED_LEN + int(context_dim)
        output_dim = self.cfg.DATA.PRED_LEN
        if self.adapter_layers == 1:
            per_head = input_dim * output_dim + output_dim
        else:
            per_head = input_dim * self.hidden_dim + self.hidden_dim + self.hidden_dim * output_dim + output_dim
        if self.var_wise_gating:
            return self.cfg.DATA.N_VAR * per_head + self.cfg.DATA.N_VAR
        return per_head + 1

    def _estimate_context_param_increase(self):
        baseline = max(self._adapter_param_count_for_context(self.buffer_context_size), 1)
        raw = self._adapter_param_count_for_context(self.raw_context_dim)
        effective = self._adapter_param_count_for_context(self.effective_context_dim) + self.context_projection_params
        context_delta = max(0, self.effective_context_dim - self.buffer_context_size)
        if self.var_wise_gating:
            approx_context_param_increase = self.cfg.DATA.N_VAR * self.cfg.DATA.PRED_LEN * context_delta
        else:
            approx_context_param_increase = self.cfg.DATA.PRED_LEN * context_delta
        budget_limit = self.param_budget_multiplier * baseline
        full_adapter_context_limit_exceeded = (
            not self.use_context_projection
            and self.effective_context_dim > self.max_context_dim_for_full_adapter
        )
        return {
            "raw_context_dim": int(self.raw_context_dim),
            "effective_context_dim": int(self.effective_context_dim),
            "context_projection_dim": int(self.context_projection_dim),
            "max_context_dim_for_full_adapter": int(self.max_context_dim_for_full_adapter),
            "context_projection_params": int(self.context_projection_params),
            "approx_context_param_increase": int(approx_context_param_increase),
            "raw_vs_mean_ratio": float(raw / baseline),
            "effective_vs_mean_ratio": float(effective / baseline),
            "budget_multiplier": float(self.param_budget_multiplier),
            "budget_limit_params": float(budget_limit),
            "full_adapter_context_limit_exceeded": bool(full_adapter_context_limit_exceeded),
            "budget_exceeded": bool(effective > budget_limit or full_adapter_context_limit_exceeded),
        }

    def _warn_if_context_budget_exceeded(self):
        stats = self.estimated_context_param_increase
        if stats["budget_exceeded"]:
            print(
                "[COSA_SPECTRAL warning] projected spectral adapter exceeds parameter budget: "
                f"raw_context_dim={self.raw_context_dim}, "
                f"effective_context_dim={self.effective_context_dim}, "
                f"context_projection_dim={self.context_projection_dim}, "
                f"max_context_dim_for_full_adapter={self.max_context_dim_for_full_adapter}, "
                f"effective_vs_mean_ratio={stats['effective_vs_mean_ratio']:.3f}, "
                f"budget_multiplier={stats['budget_multiplier']:.3f}"
            )

    def _project_context(self, context_data):
        context_data = context_data.to(self.device)
        if self.context_projector is None:
            return context_data
        if context_data.dim() != 3:
            raise ValueError("context projection expects variable-wise [B, V, C] context")
        batch_size, n_vars, raw_dim = context_data.shape
        if raw_dim != self.raw_context_dim:
            raise ValueError(f"context dim {raw_dim} does not match raw context dim {self.raw_context_dim}")
        flat = context_data.reshape(batch_size * n_vars, raw_dim)
        projected = self.context_projector(flat)
        return projected.reshape(batch_size, n_vars, self.effective_context_dim)

    def _get_individual_context_for_batch(self, batch_size, current_batch_idx):
        raw_context = self.context_builder.build(self.sample_history, batch_size)
        return raw_context.to(self.device)

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

    def _unfreeze_adapter_params(self):
        for param in self.output_adapter.parameters():
            param.requires_grad_(True)
        context_projector = getattr(self, "context_projector", None)
        if context_projector is not None:
            for param in context_projector.parameters():
                param.requires_grad_(True)

    def _get_all_models(self):
        models = [self.model, self.output_adapter]
        if self.context_projector is not None:
            models.append(self.context_projector)
        if self.norm_module is not None:
            models.append(self.norm_module)
        return models

    def switch_model_to_train(self):
        self.model.eval()
        if self.norm_module is not None:
            self.norm_module.eval()
        self.output_adapter.train()
        if self.context_projector is not None:
            self.context_projector.train()

    def switch_model_to_eval(self):
        self.model.eval()
        if self.norm_module is not None:
            self.norm_module.eval()
        self.output_adapter.eval()
        if self.context_projector is not None:
            self.context_projector.eval()

    @torch.enable_grad()
    def adapt_simple(self):
        if forecast is None:
            raise ImportError("forecast dependencies are required to run SpectralCOSAAdapter.adapt_simple")
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
                    context_start = time.time()
                    raw_context = self._get_individual_context_for_batch(batch_size, batch_idx)
                    self.time_stats["context_generation"] += time.time() - context_start
                    self.time_counts["context_generation"] += 1

                    final_start = time.time()
                    with torch.no_grad():
                        pred = self.output_adapter(pred, self._project_context(raw_context))
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
                        context_for_training = self._project_context(raw_context)
                        adapted_pred = self.output_adapter(original_pred, context_for_training)
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

                self.batch_logs.append({
                    "batch_idx": batch_idx,
                    "batch_size": batch_size,
                    "spectral_usage": self.spectral_usage,
                    "spectral_repr": self.spectral_repr,
                    "spectral_mode": self.context_builder.mode,
                    "spectral_k": int(self.context_builder.spectral_k),
                    "raw_context_dim": int(self.raw_context_dim),
                    "effective_context_dim": int(self.effective_context_dim),
                    "context_projection_dim": int(self.context_projection_dim),
                    "max_context_dim_for_full_adapter": int(self.max_context_dim_for_full_adapter),
                    "use_context_projection": int(self.use_context_projection),
                    "spectral_power_ratio_mean": float(self.context_builder.last_power_ratio_mean),
                    "estimated_context_param_increase": self.estimated_context_param_increase["approx_context_param_increase"],
                    "estimated_context_param_increase_ratio": self.estimated_context_param_increase["effective_vs_mean_ratio"],
                    "context_budget_exceeded": int(self.estimated_context_param_increase["budget_exceeded"]),
                    "full_adapter_context_limit_exceeded": int(
                        self.estimated_context_param_increase["full_adapter_context_limit_exceeded"]
                    ),
                    "context_projection_params": int(self.context_projection_params),
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
        self._export_spectral_log()
        self._print_combined_results()

    def _export_spectral_log(self):
        if not self.batch_logs:
            return
        if self.spectral_log_csv_path:
            csv_path = self.spectral_log_csv_path
        else:
            csv_dir = os.path.join(self.cfg.RESULT_DIR, "csv_predictions", "COSA_SPECTRAL")
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

        adaptation_count = max(self.time_counts.get("total_adaptation", 1), 1)
        total_batches = max(len(self.batch_logs), 1)

        combined_results = {
            "model": "SpectralCOSAAdapter",
            "spectral_statistics": {
                "spectral_usage": self.spectral_usage,
                "spectral_repr": self.spectral_repr,
                "spectral_mode": self.context_builder.mode,
                "spectral_k": int(self.context_builder.spectral_k),
                "context_dim": int(self.effective_context_dim),
                "raw_context_dim": int(self.raw_context_dim),
                "effective_context_dim": int(self.effective_context_dim),
                "context_projection_dim": int(self.context_projection_dim),
                "max_context_dim_for_full_adapter": int(self.max_context_dim_for_full_adapter),
                "use_context_projection": bool(self.use_context_projection),
                "spectral_power_ratio_mean": float(self.context_builder.last_power_ratio_mean),
                "estimated_context_param_increase": self.estimated_context_param_increase["approx_context_param_increase"],
                "estimated_context_param_increase_ratio": self.estimated_context_param_increase["effective_vs_mean_ratio"],
                "context_budget_exceeded": bool(self.estimated_context_param_increase["budget_exceeded"]),
                "full_adapter_context_limit_exceeded": bool(
                    self.estimated_context_param_increase["full_adapter_context_limit_exceeded"]
                ),
                "context_projection_params": int(self.context_projection_params),
                "total_batches": int(total_batches),
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
                "context_projection_params": int(self.context_projection_params),
            },
        }
        print(json.dumps(combined_results, indent=2))


def build_adapter(cfg, model, norm_module=None):
    return SpectralCOSAAdapter(cfg, model, norm_module)
