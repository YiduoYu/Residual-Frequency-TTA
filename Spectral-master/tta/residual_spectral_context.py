import torch


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


