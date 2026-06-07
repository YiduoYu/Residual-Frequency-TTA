from copy import deepcopy
from collections import deque, defaultdict
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.optimizer import get_optimizer
from utils.misc import prepare_inputs

try:
    from models.forecast import forecast
except ModuleNotFoundError:
    forecast = None

try:
    from config import get_norm_method
except ModuleNotFoundError:
    def get_norm_method(cfg):
        return getattr(cfg, "NORMALIZE", "NST")

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

try:
    from datasets.loader import get_test_dataloader
except ModuleNotFoundError:
    get_test_dataloader = None


def _default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DirectOutputAdapter(nn.Module):
    def __init__(
        self,
        pred_len: int,
        buffer_context_size: int = 5,
        n_vars: int = 1,
        var_wise_gating: bool = False,
        num_layers: int = 1,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.buffer_context_size = buffer_context_size
        self.n_vars = n_vars
        self.var_wise = var_wise_gating
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        input_dim = self.pred_len + self.buffer_context_size
        output_dim = self.pred_len

        if self.var_wise:
            if self.num_layers == 1:
                self.fc_layers = nn.ModuleList([
                    nn.Linear(input_dim, output_dim) for _ in range(n_vars)
                ])
            else:
                self.fc_layers = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.Tanh(),
                        nn.Dropout(0.1),
                        nn.Linear(hidden_dim, output_dim),
                    ) for _ in range(n_vars)
                ])
            self.gate = nn.Parameter(torch.zeros(n_vars))
        else:
            if self.num_layers == 1:
                self.fc = nn.Linear(input_dim, output_dim)
            else:
                self.fc = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, output_dim),
                )
            self.gate = nn.Parameter(torch.zeros(1))
        self._initialize_parameters()

    def _initialize_parameters(self):
        if self.var_wise:
            for fc_module in self.fc_layers:
                if self.num_layers == 1:
                    nn.init.xavier_uniform_(fc_module.weight, gain=0.1)
                    nn.init.zeros_(fc_module.bias)
                else:
                    for layer in fc_module:
                        if isinstance(layer, nn.Linear):
                            nn.init.xavier_uniform_(layer.weight, gain=0.1)
                            nn.init.zeros_(layer.bias)
        else:
            if self.num_layers == 1:
                nn.init.xavier_uniform_(self.fc.weight, gain=0.1)
                nn.init.zeros_(self.fc.bias)
            else:
                for layer in self.fc:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, gain=0.1)
                        nn.init.zeros_(layer.bias)

    def forward(self, y: torch.Tensor, context_data: torch.Tensor = None):
        if context_data is None:
            return y

        batch_size, pred_len, n_vars = y.shape
        if context_data.dim() not in (2, 3):
            raise ValueError(f"context_data must have shape [B, C] or [B, V, C], got {tuple(context_data.shape)}")
        if context_data.shape[0] != batch_size:
            raise ValueError(f"context batch size {context_data.shape[0]} does not match prediction batch size {batch_size}")
        if context_data.dim() == 3 and context_data.shape[1] != n_vars:
            raise ValueError(f"context variable size {context_data.shape[1]} does not match prediction variable size {n_vars}")
        if context_data.shape[-1] != self.buffer_context_size:
            raise ValueError(
                f"context dim {context_data.shape[-1]} does not match adapter context dim {self.buffer_context_size}"
            )
        context_data = context_data.to(device=y.device, dtype=y.dtype)

        if self.var_wise:
            if n_vars > len(self.fc_layers):
                raise ValueError(f"prediction has {n_vars} variables but adapter has {len(self.fc_layers)} variable heads")
            corrections = []
            for var_idx in range(n_vars):
                y_var = y[:, :, var_idx]
                if context_data.dim() == 3:
                    context_var = context_data[:, var_idx, :]
                else:
                    context_var = context_data
                combined_input = torch.cat([y_var, context_var], dim=-1)
                correction_var = self.fc_layers[var_idx](combined_input)
                corrections.append(correction_var.unsqueeze(-1))

            correction = torch.cat(corrections, dim=-1)
            gating_factor = torch.tanh(self.gate).unsqueeze(0).unsqueeze(0)
        else:
            y_flattened = y.transpose(1, 2).contiguous().view(batch_size * n_vars, pred_len)
            if context_data.dim() == 3:
                context_repeated = context_data.contiguous().view(batch_size * n_vars, -1)
            else:
                context_repeated = context_data.unsqueeze(1).repeat(1, n_vars, 1).view(batch_size * n_vars, -1)
            combined_input = torch.cat([y_flattened, context_repeated], dim=-1)
            correction = self.fc(combined_input)
            correction = correction.view(batch_size, n_vars, pred_len).transpose(1, 2)
            gating_factor = torch.tanh(self.gate)

        return y + gating_factor * correction


class DirectAdapterBase(nn.Module):
    def __init__(self, cfg, model: nn.Module, norm_module=None):
        super().__init__()
        self.cfg = cfg
        self.model_cfg = cfg.MODEL
        self.model = model
        self.device = _default_device()
        self.norm_method = get_norm_method(cfg)
        self.norm_module = norm_module
        self.adapter_cfg = cfg.TTA.DIRECT_SPECTRAL.ADAPTER
        if get_test_dataloader is None:
            raise ImportError("datasets dependencies are required to instantiate DirectAdapterBase")
        self.test_loader = get_test_dataloader(cfg)
        self.test_data = self.test_loader.dataset.test

        self.buffer_context_size = getattr(self.adapter_cfg, "BUFFER_CONTEXT_SIZE", 5)
        self.adapt_steps = getattr(self.adapter_cfg, "STEPS", 20)

        self.paas_enabled = getattr(self.adapter_cfg, "PAAS", False)
        self.period_n = getattr(self.adapter_cfg, "PERIOD_N", 1)

        self.fast_adaptation = getattr(self.adapter_cfg, "FAST_ADAPTATION", False)
        self.adaptive_lr = getattr(self.adapter_cfg, "ADAPTIVE_LR", False)
        self.max_lr = getattr(self.adapter_cfg, "MAX_LR", 0.005)
        self.min_lr = getattr(self.adapter_cfg, "MIN_LR", 0.0001)
        self.convergence_threshold = getattr(self.adapter_cfg, "CONVERGENCE_THRESHOLD", 1e-4)
        self.var_wise_gating = getattr(self.adapter_cfg, "VAR_WISE_GATING", False)

        self.save_csv = getattr(self.adapter_cfg, "SAVE_CSV", False)
        self.csv_predictions = []

        self.save_paas_csv = getattr(self.adapter_cfg, "SAVE_PAAS_CSV", False)
        self.paas_batch_info = []

        self.adapter_layers = getattr(self.adapter_cfg, "ADAPTER_LAYERS", 1)
        self.hidden_dim = getattr(self.adapter_cfg, "HIDDEN_DIM", 64)
        self.per_batch_lr_reset = getattr(self.adapter_cfg, "PER_BATCH_LR_RESET", True)

        self.loss_history = deque(maxlen=5)
        self.current_lr = getattr(cfg.TTA.SOLVER, "BASE_LR", 0.001)

        self.sample_history = deque(maxlen=200)
        self.current_time_idx = 0

        self.time_stats = defaultdict(float)
        self.time_counts = defaultdict(int)

        # Keep the same temporary initialization side effect as the original base
        # adapter so seeded DSA correction weights do not shift.
        self.output_adapter = DirectOutputAdapter(
            pred_len=cfg.DATA.PRED_LEN,
            buffer_context_size=self.buffer_context_size,
            n_vars=cfg.DATA.N_VAR,
            var_wise_gating=self.var_wise_gating,
            num_layers=self.adapter_layers,
            hidden_dim=self.hidden_dim,
        ).to(self.device)

        self.adapters_enabled = False
        self.step_count = 0

        self._freeze_all_model_params()
        self._unfreeze_adapter_params()

        self.optimizer = get_optimizer(self.output_adapter.parameters(), cfg.TTA)
        self.model_state, self.optimizer_state = self._copy_model_and_optimizer()

        cfg.TEST.BATCH_SIZE = len(self.test_loader.dataset)
        self.test_loader = get_test_dataloader(cfg)
        self.cur_step = cfg.DATA.SEQ_LEN - 2
        self.n_adapt = 0

        self.mse_all = []
        self.mae_all = []

    def forward(self, enc_window, enc_window_stamp, dec_window, dec_window_stamp):
        raise NotImplementedError

    def reset(self):
        self._load_model_and_optimizer()
        self.adapters_enabled = False
        self.step_count = 0
        self.loss_history.clear()
        self.current_lr = getattr(self.cfg.TTA.SOLVER, "BASE_LR", 0.001)

    def _copy_model_and_optimizer(self):
        model_state = deepcopy(self.model.state_dict())
        optimizer_state = deepcopy(self.optimizer.state_dict())
        return model_state, optimizer_state

    def _load_model_and_optimizer(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    def _get_all_models(self):
        models = [self.model, self.output_adapter]
        if self.norm_module is not None:
            models.append(self.norm_module)
        return models

    def _freeze_all_model_params(self):
        for param in self.model.parameters():
            param.requires_grad_(False)
        if self.norm_module is not None:
            for param in self.norm_module.parameters():
                param.requires_grad_(False)

    def _unfreeze_adapter_params(self):
        for param in self.output_adapter.parameters():
            param.requires_grad_(True)

    def switch_model_to_train(self):
        self.model.eval()
        if self.norm_module is not None:
            self.norm_module.eval()
        self.output_adapter.train()

    def switch_model_to_eval(self):
        self.model.eval()
        if self.norm_module is not None:
            self.norm_module.eval()
        self.output_adapter.eval()

    def update_memory_buffer(self, targets: torch.Tensor, prediction: torch.Tensor):
        batch_size = targets.shape[0]
        seq_mse = F.mse_loss(prediction, targets).item()

        batch_info = {
            "time_idx": self.current_time_idx,
            "target_mean": targets.mean().item(),
        }
        self.sample_history.append(batch_info)

        self.current_time_idx += batch_size
        self.step_count += 1

        if not self.adapters_enabled:
            self.adapters_enabled = True

        return seq_mse

    def _calculate_period_and_batch_size(self, enc_window_first):
        fft_result = torch.fft.rfft(enc_window_first - enc_window_first.mean(dim=0), dim=0)
        amplitude = torch.abs(fft_result)
        power = torch.mean(amplitude ** 2, dim=0)

        dominant_freq = None
        try:
            dominant_freq_idx = torch.argmax(amplitude[:, power.argmax()]).item()
            dominant_freq = float(dominant_freq_idx / enc_window_first.shape[0])
            period = enc_window_first.shape[0] // dominant_freq_idx
        except Exception:
            period = 24
            dominant_freq = 1.0 / 24

        period *= self.period_n
        batch_size = period + 1
        return period, batch_size, dominant_freq

    def _adaptive_learning_rate(self, current_loss: float, step: int, batch_idx: int = 0) -> float:
        if step == 0 and self.per_batch_lr_reset:
            self.current_lr = getattr(self.cfg.TTA.SOLVER, "BASE_LR", 0.001)
            self.loss_history.append(current_loss)
            return self.current_lr

        self.loss_history.append(current_loss)
        recent_losses = list(self.loss_history)[-3:]

        if len(recent_losses) < 2:
            return self.current_lr

        loss_trend = recent_losses[-1] - recent_losses[0]
        loss_variance = torch.tensor(recent_losses).var().item()

        if loss_trend > 0 and loss_variance < 1e-6:
            self.current_lr = min(self.current_lr * 1.2, self.max_lr)
        elif loss_trend < -0.01:
            self.current_lr = min(self.current_lr * 1.05, self.max_lr)
        elif abs(loss_trend) < 1e-6:
            self.current_lr = max(self.current_lr * 0.8, self.min_lr)

        if step >= 1:
            cosine_factor = 0.5 * (1 + torch.cos(torch.tensor(step * 3.14159 / self.adapt_steps)))
            self.current_lr = self.min_lr + (self.current_lr - self.min_lr) * cosine_factor

        return self.current_lr

    def _save_batch_predictions_to_csv(
        self,
        original_pred: torch.Tensor,
        tta_pred: torch.Tensor,
        ground_truth: torch.Tensor,
        batch_idx: int,
    ):
        batch_size, pred_len, n_vars = tta_pred.shape

        original_pred_np = original_pred.detach().cpu().numpy()
        tta_pred_np = tta_pred.detach().cpu().numpy()
        ground_truth_np = ground_truth.detach().cpu().numpy()

        for sample_idx in range(batch_size):
            for time_idx in range(pred_len):
                for var_idx in range(n_vars):
                    orig_val = float(original_pred_np[sample_idx, time_idx, var_idx])
                    tta_val = float(tta_pred_np[sample_idx, time_idx, var_idx])
                    gt_val = float(ground_truth_np[sample_idx, time_idx, var_idx])

                    row_data = {
                        "batch_idx": batch_idx,
                        "sample_idx": sample_idx,
                        "global_sample_idx": batch_idx * batch_size + sample_idx,
                        "timestep": time_idx + 1,
                        "variable_idx": var_idx,
                        "original_prediction": orig_val,
                        "tta_prediction": tta_val,
                        "ground_truth": gt_val,
                        "tta_improvement": tta_val - orig_val,
                        "original_absolute_error": float(abs(orig_val - gt_val)),
                        "tta_absolute_error": float(abs(tta_val - gt_val)),
                        "original_squared_error": float((orig_val - gt_val) ** 2),
                        "tta_squared_error": float((tta_val - gt_val) ** 2),
                        "error_improvement": float(abs(orig_val - gt_val) - abs(tta_val - gt_val)),
                    }
                    self.csv_predictions.append(row_data)

    def _export_predictions_to_csv(self):
        if not self.csv_predictions:
            return
        if pd is None:
            raise ImportError("pandas is required when TTA.DIRECT_SPECTRAL.ADAPTER.SAVE_CSV is enabled")

        df = pd.DataFrame(self.csv_predictions)

        model_name = self.cfg.MODEL.NAME
        dataset_name = self.cfg.DATA.NAME
        pred_len = self.cfg.DATA.PRED_LEN

        csv_dir = os.path.join(self.cfg.RESULT_DIR, "csv_predictions", "DIRECT_SPECTRAL")
        os.makedirs(csv_dir, exist_ok=True)

        csv_filename = f"{model_name}_{dataset_name}_pred{pred_len}_predictions.csv"
        csv_path = os.path.join(csv_dir, csv_filename)

        df.to_csv(csv_path, index=False)

    def _save_paas_batch_info(
        self,
        batch_idx: int,
        batch_start: int,
        calculated_batch_size: int,
        actual_batch_size: int,
        period: int = None,
        fft_dominant_freq: float = None,
    ):
        batch_info = {
            "batch_idx": batch_idx,
            "batch_start_idx": batch_start,
            "calculated_batch_size": calculated_batch_size,
            "actual_batch_size": actual_batch_size,
            "period": period if period is not None else "N/A",
            "fft_dominant_frequency": fft_dominant_freq if fft_dominant_freq is not None else "N/A",
            "paas_enabled": self.paas_enabled,
            "period_n_multiplier": self.period_n,
        }
        self.paas_batch_info.append(batch_info)

    def _export_paas_info_to_csv(self):
        if not self.paas_batch_info:
            return
        if pd is None:
            raise ImportError("pandas is required when TTA.DIRECT_SPECTRAL.ADAPTER.SAVE_PAAS_CSV is enabled")

        df = pd.DataFrame(self.paas_batch_info)

        model_name = self.cfg.MODEL.NAME
        dataset_name = self.cfg.DATA.NAME
        pred_len = self.cfg.DATA.PRED_LEN

        csv_dir = os.path.join(self.cfg.RESULT_DIR, "csv_predictions", "DIRECT_SPECTRAL")
        os.makedirs(csv_dir, exist_ok=True)

        csv_filename = f"{model_name}_{dataset_name}_pred{pred_len}_paas_batch_sizes.csv"
        csv_path = os.path.join(csv_dir, csv_filename)

        df.to_csv(csv_path, index=False)
