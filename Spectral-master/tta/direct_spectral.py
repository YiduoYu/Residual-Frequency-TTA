import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.optimizer import get_optimizer
from tta.direct_adapter_base import DirectAdapterBase, DirectOutputAdapter, forecast, prepare_inputs


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _count_params(module, trainable_only=False):
    return sum(
        int(param.numel())
        for param in module.parameters()
        if (param.requires_grad or not trainable_only)
    )


class DirectSpectralContextBuilder:
    def __init__(
        self,
        pred_len,
        n_vars,
        spectral_k=8,
        history_size=8,
        use_power_ratio=True,
        use_residual_stats=True,
        include_mean_context=False,
        mean_context_size=5,
        eps=1e-6,
    ):
        self.pred_len = int(pred_len)
        self.n_vars = int(n_vars)
        self.spectral_k = int(spectral_k)
        self.history_size = int(history_size)
        self.use_power_ratio = _as_bool(use_power_ratio)
        self.use_residual_stats = _as_bool(use_residual_stats)
        self.include_mean_context = _as_bool(include_mean_context)
        self.mean_context_size = int(mean_context_size)
        self.eps = float(eps)
        self.spectral_context_dim = (
            2 * self.spectral_k
            + int(self.use_power_ratio)
            + 2 * int(self.use_residual_stats)
        )
        self.mean_context_dim = self.mean_context_size if self.include_mean_context else 0
        self.raw_context_dim = self.spectral_context_dim + self.mean_context_dim

    def _zero_context(self, batch_size, device, dtype):
        return torch.zeros(
            int(batch_size),
            self.n_vars,
            self.raw_context_dim,
            device=device,
            dtype=dtype,
        )

    def _zero_spectral_context(self, device, dtype):
        return torch.zeros(self.n_vars, self.spectral_context_dim, device=device, dtype=dtype)

    def _coerce_profile(self, profile, device, dtype):
        profile = torch.as_tensor(profile, device=device, dtype=dtype).detach()
        if profile.dim() != 2:
            raise ValueError(f"residual_profile must have shape [P, V], got {tuple(profile.shape)}")
        if profile.shape[0] != self.pred_len:
            raise ValueError(f"residual_profile length {profile.shape[0]} does not match pred_len {self.pred_len}")
        if profile.shape[1] == 1 and self.n_vars > 1:
            profile = profile.expand(self.pred_len, self.n_vars)
        if profile.shape[1] != self.n_vars:
            raise ValueError(f"residual_profile has {profile.shape[1]} variables, expected {self.n_vars}")
        return profile

    def _coerce_var_vector(self, value, device, dtype, name):
        vector = torch.as_tensor(value, device=device, dtype=dtype).detach()
        if vector.dim() == 0:
            vector = vector.view(1)
        if vector.dim() != 1:
            vector = vector.reshape(-1)
        if vector.shape[0] == 1 and self.n_vars > 1:
            vector = vector.expand(self.n_vars)
        if vector.shape[0] != self.n_vars:
            raise ValueError(f"{name} has {vector.shape[0]} variables, expected {self.n_vars}")
        return vector

    def _build_mean_context(self, sample_history, device, dtype):
        if not self.include_mean_context or self.mean_context_size <= 0:
            return None

        values = []
        for entry in reversed(list(sample_history)):
            if not isinstance(entry, dict):
                continue
            if entry.get("target_mean_var") is not None:
                values.append(self._coerce_var_vector(entry["target_mean_var"], device, dtype, "target_mean_var"))
            elif entry.get("target_mean") is not None:
                values.append(self._coerce_var_vector(entry["target_mean"], device, dtype, "target_mean"))
            if len(values) >= self.mean_context_size:
                break

        if not values:
            return torch.zeros(self.n_vars, self.mean_context_size, device=device, dtype=dtype)

        while len(values) < self.mean_context_size:
            values.append(values[-1])
        return torch.stack(values[:self.mean_context_size], dim=-1)

    def _build_spectral_context(self, profiles, device, dtype):
        if not profiles:
            return self._zero_spectral_context(device, dtype)

        profile = torch.stack(profiles, dim=0).mean(dim=0)
        residual_mean = profile.mean(dim=0)
        profile_centered = profile - residual_mean.view(1, self.n_vars)
        residual_scale = profile_centered.std(dim=0, unbiased=False).clamp_min(self.eps)

        coeff = torch.fft.rfft(profile_centered, dim=0)
        non_dc = coeff[1:]
        selected = coeff[1:self.spectral_k + 1]
        if selected.shape[0] < self.spectral_k:
            pad = torch.zeros(
                self.spectral_k - selected.shape[0],
                self.n_vars,
                device=device,
                dtype=selected.dtype,
            )
            selected = torch.cat([selected, pad], dim=0)

        selected_raw = selected
        selected = selected / residual_scale.view(1, self.n_vars)
        context_parts = [selected.real.T, selected.imag.T]

        if self.use_power_ratio:
            if non_dc.numel() == 0:
                ratio = torch.zeros(self.n_vars, device=device, dtype=dtype)
            else:
                selected_power = selected_raw.abs().pow(2).sum(dim=0).real
                total_power = non_dc.abs().pow(2).sum(dim=0).real.clamp_min(self.eps)
                ratio = selected_power / total_power
            context_parts.append(ratio.unsqueeze(-1).to(dtype=dtype))

        if self.use_residual_stats:
            context_parts.append(residual_mean.unsqueeze(-1))
            context_parts.append(residual_scale.unsqueeze(-1))

        return torch.cat(context_parts, dim=-1).to(dtype=dtype)

    def build(self, sample_history, batch_size, device=None, dtype=torch.float32):
        device = device if device is not None else torch.device("cpu")
        history_entries = list(sample_history)[-max(self.history_size, 0):]
        profiles = []
        for entry in history_entries:
            if not isinstance(entry, dict) or entry.get("residual_profile") is None:
                continue
            profiles.append(self._coerce_profile(entry["residual_profile"], device, dtype))

        context_parts = [self._build_spectral_context(profiles, device, dtype)]
        mean_context = self._build_mean_context(history_entries, device, dtype)
        if mean_context is not None:
            context_parts.append(mean_context)
        context = torch.cat(context_parts, dim=-1).to(dtype=dtype)
        return context.unsqueeze(0).expand(int(batch_size), -1, -1).contiguous()

    def __call__(self, sample_history, batch_size, device=None, dtype=torch.float32):
        return self.build(sample_history, batch_size, device=device, dtype=dtype)


class DirectSpectralCorrection(nn.Module):
    def __init__(
        self,
        pred_len,
        n_vars,
        raw_context_dim,
        context_projection_dim=8,
        var_wise_gating=False,
        adapter_layers=1,
        hidden_dim=64,
    ):
        super().__init__()
        self.pred_len = int(pred_len)
        self.n_vars = int(n_vars)
        self.raw_context_dim = int(raw_context_dim)
        self.context_projection_dim = int(context_projection_dim)
        self.use_context_projection = (
            self.context_projection_dim > 0
            and self.context_projection_dim < self.raw_context_dim
        )
        self.effective_context_dim = (
            self.context_projection_dim if self.use_context_projection else self.raw_context_dim
        )

        if self.use_context_projection:
            self.context_projector = nn.Linear(self.raw_context_dim, self.effective_context_dim)
            nn.init.xavier_uniform_(self.context_projector.weight, gain=0.05)
            nn.init.zeros_(self.context_projector.bias)
        else:
            self.context_projector = None

        self.output_adapter = DirectOutputAdapter(
            pred_len=self.pred_len,
            buffer_context_size=self.effective_context_dim,
            n_vars=self.n_vars,
            var_wise_gating=var_wise_gating,
            num_layers=adapter_layers,
            hidden_dim=hidden_dim,
        )

    @property
    def context_projection_params(self):
        if self.context_projector is None:
            return 0
        return _count_params(self.context_projector)

    def project_context(self, raw_context):
        if raw_context.dim() != 3:
            raise ValueError(f"raw_context must have shape [B, V, C], got {tuple(raw_context.shape)}")
        if raw_context.shape[1] != self.n_vars:
            raise ValueError(f"raw_context has {raw_context.shape[1]} variables, expected {self.n_vars}")
        if raw_context.shape[-1] != self.raw_context_dim:
            raise ValueError(f"raw_context dim {raw_context.shape[-1]} does not match {self.raw_context_dim}")
        if self.context_projector is None:
            return raw_context
        return self.context_projector(raw_context)

    def forward(self, pred, raw_context):
        raw_context = raw_context.to(device=pred.device, dtype=pred.dtype)
        context = self.project_context(raw_context)
        return self.output_adapter(pred, context)


class DirectSpectralAdapter(DirectAdapterBase):
    def __init__(self, cfg, model, norm_module=None):
        super().__init__(cfg, model, norm_module=norm_module)
        direct_cfg = cfg.TTA.DIRECT_SPECTRAL

        self.spectral_k = int(getattr(direct_cfg, "K", 8))
        self.history_size = int(getattr(direct_cfg, "HISTORY_SIZE", 8))
        self.context_projection_dim = int(getattr(direct_cfg, "CONTEXT_PROJECTION_DIM", 8))
        self.use_residual_stats = _as_bool(getattr(direct_cfg, "USE_RESIDUAL_STATS", True))
        self.use_power_ratio = _as_bool(getattr(direct_cfg, "USE_POWER_RATIO", True))
        self.include_mean_context = _as_bool(getattr(direct_cfg, "INCLUDE_MEAN_CONTEXT", False))
        self.mean_context_size = int(getattr(direct_cfg, "MEAN_CONTEXT_SIZE", 5))
        self.param_budget_multiplier = float(getattr(direct_cfg, "PARAM_BUDGET_MULTIPLIER", 1.25))
        self.log_csv_path = getattr(direct_cfg, "LOG_CSV_PATH", "")

        if hasattr(self, "output_adapter"):
            del self.output_adapter

        self.context_builder = DirectSpectralContextBuilder(
            pred_len=cfg.DATA.PRED_LEN,
            n_vars=cfg.DATA.N_VAR,
            spectral_k=self.spectral_k,
            history_size=self.history_size,
            use_power_ratio=self.use_power_ratio,
            use_residual_stats=self.use_residual_stats,
            include_mean_context=self.include_mean_context,
            mean_context_size=self.mean_context_size,
        )
        self.correction = DirectSpectralCorrection(
            pred_len=cfg.DATA.PRED_LEN,
            n_vars=cfg.DATA.N_VAR,
            raw_context_dim=self.context_builder.raw_context_dim,
            context_projection_dim=self.context_projection_dim,
            var_wise_gating=self.var_wise_gating,
            adapter_layers=self.adapter_layers,
            hidden_dim=self.hidden_dim,
        ).to(self.device)

        self._freeze_all_model_params()
        self._adapter_params = list(self.correction.parameters())
        for param in self._adapter_params:
            param.requires_grad_(True)
        self.optimizer = get_optimizer(self._adapter_params, cfg.TTA)
        self.model_state, self.optimizer_state = self._copy_model_and_optimizer()
        self.batch_logs = []
        self.parameter_budget = self._parameter_budget_info()
        self._warn_if_over_budget()

    @property
    def raw_context_dim(self):
        return self.context_builder.raw_context_dim

    @property
    def effective_context_dim(self):
        return self.correction.effective_context_dim

    def _estimate_output_adapter_params(self, context_dim):
        adapter = DirectOutputAdapter(
            pred_len=self.cfg.DATA.PRED_LEN,
            buffer_context_size=int(context_dim),
            n_vars=self.cfg.DATA.N_VAR,
            var_wise_gating=self.var_wise_gating,
            num_layers=self.adapter_layers,
            hidden_dim=self.hidden_dim,
        )
        return _count_params(adapter)

    def _parameter_budget_info(self):
        mean_context_adapter_params = self._estimate_output_adapter_params(1)
        raw_context_adapter_params = self._estimate_output_adapter_params(self.raw_context_dim)
        effective_context_adapter_params = self._estimate_output_adapter_params(self.effective_context_dim)
        context_projection_params = self.correction.context_projection_params
        effective_total = effective_context_adapter_params + context_projection_params
        return {
            "mean_context_adapter_params": int(mean_context_adapter_params),
            "raw_context_adapter_params": int(raw_context_adapter_params),
            "effective_context_adapter_params": int(effective_context_adapter_params),
            "context_projection_params": int(context_projection_params),
            "effective_vs_mean_ratio": float(effective_total / max(mean_context_adapter_params, 1)),
            "budget_multiplier": float(self.param_budget_multiplier),
        }

    def _warn_if_over_budget(self):
        ratio = self.parameter_budget["effective_vs_mean_ratio"]
        if ratio > self.param_budget_multiplier:
            print(
                "warning: DirectSpectralAdapter effective parameter ratio "
                f"{ratio:.3f} exceeds budget multiplier {self.param_budget_multiplier:.3f}"
            )

    def _get_all_models(self):
        models = [self.model, self.correction]
        if self.norm_module is not None:
            models.append(self.norm_module)
        return models

    def switch_model_to_train(self):
        self.model.eval()
        if self.norm_module is not None:
            self.norm_module.eval()
        self.correction.train()

    def switch_model_to_eval(self):
        self.model.eval()
        if self.norm_module is not None:
            self.norm_module.eval()
        self.correction.eval()

    def _get_raw_context_for_batch(self, batch_size, reference):
        return self.context_builder(
            self.sample_history,
            batch_size=batch_size,
            device=reference.device,
            dtype=reference.dtype,
        )

    def update_memory_buffer(self, targets, prediction):
        batch_size = targets.shape[0]
        signed_residual = (targets - prediction).detach()
        residual_profile = signed_residual.mean(dim=0)
        seq_mse = F.mse_loss(prediction, targets).item()
        batch_info = {
            "time_idx": self.current_time_idx,
            "residual_profile": residual_profile.detach().cpu(),
            "residual_mean_var": signed_residual.mean(dim=(0, 1)).detach().cpu(),
            "residual_std_var": signed_residual.std(dim=(0, 1), unbiased=False).detach().cpu(),
            "target_mean_var": targets.detach().mean(dim=(0, 1)).cpu(),
            "target_std_var": targets.detach().std(dim=(0, 1), unbiased=False).cpu(),
            "seq_mse": float(seq_mse),
        }
        self.sample_history.append(batch_info)
        self.current_time_idx += batch_size
        self.step_count += 1

        if not self.adapters_enabled:
            self.adapters_enabled = True

        return seq_mse

    def _record_batch_log(self, batch_idx, mse, mae, raw_context=None):
        row = {
            "batch_idx": int(batch_idx),
            "mse": float(np.mean(mse)),
            "mae": float(np.mean(mae)),
            "adapters_enabled": bool(self.adapters_enabled),
            "raw_context_dim": int(self.raw_context_dim),
            "effective_context_dim": int(self.effective_context_dim),
        }
        if raw_context is not None:
            row["raw_context_norm"] = float(torch.norm(raw_context.detach()).cpu())
        self.batch_logs.append(row)

    def _export_batch_logs(self):
        if not self.log_csv_path or not self.batch_logs:
            return
        os.makedirs(os.path.dirname(self.log_csv_path), exist_ok=True)
        fieldnames = sorted({key for row in self.batch_logs for key in row})
        with open(self.log_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.batch_logs)

    @torch.enable_grad()
    def adapt_simple(self):
        if forecast is None:
            raise ImportError("forecast dependencies are required to run DirectSpectralAdapter.adapt_simple")

        batch_start = 0
        batch_end = 0
        batch_idx = 0
        total_start_time = time.time()

        self.switch_model_to_eval()

        for _, inputs in enumerate(self.test_loader):
            enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all = prepare_inputs(inputs)

            while batch_end < len(enc_window_all):
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
                    batch_size = getattr(self.adapter_cfg, "BATCH_SIZE", 64)
                    calculated_batch_size = batch_size

                batch_end = batch_start + batch_size
                if batch_end > len(enc_window_all):
                    batch_end = len(enc_window_all)
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
                original_pred = pred.detach()
                ground_truth = ground_truth.detach()
                pred = original_pred
                self.time_stats["base_prediction"] += time.time() - pred_start
                self.time_counts["base_prediction"] += 1

                raw_context = None
                if self.adapters_enabled:
                    context_start = time.time()
                    raw_context = self._get_raw_context_for_batch(batch_size, pred)
                    self.time_stats["context_generation"] += time.time() - context_start
                    self.time_counts["context_generation"] += 1

                    final_start = time.time()
                    with torch.no_grad():
                        pred = self.correction(pred, raw_context)
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
                self._record_batch_log(batch_idx, mse, mae, raw_context=raw_context)

                buffer_start = time.time()
                self.update_memory_buffer(ground_truth, original_pred)
                self.time_stats["buffer_update"] += time.time() - buffer_start
                self.time_counts["buffer_update"] += 1

                if self.adapters_enabled:
                    context_start = time.time()
                    train_raw_context = self._get_raw_context_for_batch(batch_size, original_pred)
                    self.time_stats["context_generation"] += time.time() - context_start
                    self.time_counts["context_generation"] += 1

                    effective_steps = self.adapt_steps
                    if self.fast_adaptation:
                        effective_steps = min(self.adapt_steps, 5)

                    adapt_start = time.time()
                    for step in range(effective_steps):
                        step_start = time.time()
                        self.n_adapt += 1
                        self.switch_model_to_train()

                        forward_start = time.time()
                        adapted_pred = self.correction(original_pred, train_raw_context)
                        self.time_stats["adapter_forward"] += time.time() - forward_start
                        self.time_counts["adapter_forward"] += 1

                        loss_start = time.time()
                        loss = F.mse_loss(adapted_pred, ground_truth)
                        l2_reg = sum(p.pow(2).sum() for p in self._adapter_params if p.requires_grad)
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

                        clip_params = [p for p in self._adapter_params if p.requires_grad]
                        if self.fast_adaptation:
                            max_norm = max(0.05, min(0.5, loss.item()))
                            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=max_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=0.1)

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

        self._export_batch_logs()
        self._print_combined_results()

    def _time_statistics(self):
        avg_times = {}
        for key in self.time_stats:
            if key != "total_time":
                count = self.time_counts[key] if self.time_counts[key] > 0 else 1
                avg_times[key] = self.time_stats[key] / count

        adapter_total = (
            avg_times.get("context_generation", 0)
            + avg_times.get("adapter_forward", 0)
            + avg_times.get("final_prediction", 0)
        )
        adaptation_count = max(self.time_counts.get("total_adaptation", 1), 1)
        total_time = max(self.time_stats.get("total_time", 0.0), 1e-12)

        return {
            "adapter_operations": {
                "context_generation_ms": round(avg_times.get("context_generation", 0) * 1000, 3),
                "adapter_forward_pass_ms": round(avg_times.get("adapter_forward", 0) * 1000, 3),
                "final_prediction_ms": round(avg_times.get("final_prediction", 0) * 1000, 3),
                "total_adapter_operation_ms": round(adapter_total * 1000, 3),
            },
            "adaptation_training": {
                "loss_computation_ms": round(avg_times.get("loss_computation", 0) * 1000, 3),
                "backward_update_ms": round(avg_times.get("backward_update", 0) * 1000, 3),
                "per_adaptation_step_ms": round(avg_times.get("per_adaptation_step", 0) * 1000, 3),
                "total_per_batch_ms": round(
                    self.time_stats.get("total_adaptation", 0) / adaptation_count * 1000,
                    3,
                ),
                "adaptation_steps": int(self.adapt_steps),
            },
            "overall_stats": {
                "total_time_seconds": round(self.time_stats.get("total_time", 0.0), 2),
                "total_adaptations": int(self.n_adapt),
                "throughput_samples_per_sec": round(len(self.test_loader.dataset) / total_time, 1),
            },
        }

    def _summary_dict(self):
        trainable_parameter_count = _count_params(self.correction, trainable_only=True)
        return {
            "model": "DirectSpectralAdapter",
            "raw_context_dim": int(self.raw_context_dim),
            "effective_context_dim": int(self.effective_context_dim),
            "context_projection_dim": int(self.context_projection_dim),
            "use_context_projection": bool(self.correction.use_context_projection),
            "use_power_ratio": bool(self.use_power_ratio),
            "use_residual_stats": bool(self.use_residual_stats),
            "include_mean_context": bool(self.include_mean_context),
            "mean_context_size": int(self.mean_context_size),
            "spectral_context_dim": int(self.context_builder.spectral_context_dim),
            "spectral_k": int(self.spectral_k),
            "history_size": int(self.history_size),
            "trainable_parameter_count": int(trainable_parameter_count),
            "context_projection_parameter_count": int(self.correction.context_projection_params),
            "test_mse": float(self.mse_all.mean()) if len(self.mse_all) else None,
            "test_mae": float(self.mae_all.mean()) if len(self.mae_all) else None,
            "total_adaptation_count": int(self.n_adapt),
            "parameter_budget": self.parameter_budget,
            "time_statistics": self._time_statistics(),
        }

    def _print_combined_results(self):
        print(json.dumps(self._summary_dict(), indent=2))

    def count_parameters(self):
        summary = self._summary_dict()
        parameter_rows = []
        for name, param in self.named_parameters():
            parameter_rows.append({
                "name": name,
                "requires_grad": bool(param.requires_grad),
                "numel": int(param.numel()),
                "shape": list(param.shape),
            })
        summary["parameters"] = parameter_rows
        print(json.dumps({"model": "DirectSpectralAdapter", "parameters": parameter_rows}, indent=2))
        return summary

    def adapt(self):
        self.adapt_simple()


def build_adapter(cfg, model, norm_module=None):
    return DirectSpectralAdapter(cfg, model, norm_module=norm_module)
