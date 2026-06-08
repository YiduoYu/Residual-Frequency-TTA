# ReSpect-TTA Diagram Full Explanation

This note explains every component in the ReSpect-TTA diagram, including what it is, why it is added, what effect it has, where it is implemented, and which parameters activate it.

Current presentation name: **ReSpect-TTA**.

Code name in the repository: many files still use `ReSpect-TTA` / `RESPECT_TTA` because this method was developed as the sixth experimental scheme before the final name was chosen.

Core code:

- `Spectral-master/tta/respect_tta.py`
- Main class: `ReSpectTTAAdapter`
- Core correction module: `ReSpectDirectSpectralCorrector`
- Experiment script: `script/respect_tta/run_respect_tta.sh`

## 1. One-Sentence Story

ReSpect-TTA keeps the pretrained forecasting backbone frozen, records recent prediction residuals, converts both the base prediction and residual memory into the frequency domain, learns a residual-conditioned complex spectral correction, and adds the gated correction back to the original prediction during test-time adaptation.

## 2. Main Formula

The core prediction formula in the diagram is:

```text
F_y = rFFT(Y_t^(0))
F_r = rFFT(R_buf)
Delta Y_t = iFFT(M(F_r) odot F_y)
Y_hat_t = Y_t^(0) + g_t odot Delta Y_t
```

Meaning:

- `Y_t^(0)` is the frozen backbone's original prediction.
- `R_buf` is the recent residual memory from previous test batches.
- `F_y` is the prediction spectrum.
- `F_r` is the residual spectrum.
- `M(F_r)` is a learned complex modulator generated from the residual spectrum.
- `odot` means element-wise multiplication.
- `Delta Y_t` is the time-domain correction.
- `g_t` is a learned correction gate.
- `Y_hat_t` is the final adapted forecast.

## 3. Full Runtime Workflow

1. The test input window `X_t` is sent into a pretrained forecasting model.
2. The forecasting model produces a base prediction `Y_t^(0)`.
3. The backbone is frozen; only the ReSpect-TTA adapter is trainable.
4. If no residual memory exists yet, the first batch mostly behaves like the frozen backbone.
5. After ground truth becomes available, the code computes the signed residual:

```text
R_t = Y_t - Y_t^(0)
```

6. The residual is stored in `sample_history` as `residual_profile`.
7. For later batches, recent residual profiles are averaged into `R_buf`.
8. `Y_t^(0)` is transformed by `rFFT` into `F_y`.
9. `R_buf` is centered, windowed, and transformed by `rFFT` into `F_r`.
10. `F_r` is used to generate a complex modulator `M(F_r)`.
11. `M(F_r)` directly modulates the prediction spectrum `F_y`.
12. `iFFT` converts the modulated spectrum back into a time-domain correction `Delta Y_t`.
13. A learned global gate `g_t` controls how much correction is applied.
14. The final output is:

```text
Y_hat_t = Y_t^(0) + g_t odot Delta Y_t
```

15. When ground truth for the current batch is available, the adapter is updated using MSE loss.
16. The new residual is added to the memory buffer for future batches.

Important defense point: **current prediction only uses past residuals. Ground truth from the current batch is used after prediction, so there is no future leakage.**

## 4. Block-by-Block Explanation

## 4.1 Input Historical Window `X_t`

What it is:

- The historical time-series window used by the forecasting model.
- Shape is conceptually `[batch, seq_len, variables]`.
- In experiments, `SEQ_LEN=96`.

Why it is needed:

- This is the normal input to the original forecasting model.
- ReSpect-TTA does not change the input format of the backbone.

What it does:

- Provides historical observations for the frozen model to forecast the next horizon.

How it is passed:

In `run_respect_tta.sh`, the script passes:

```text
DATA.SEQ_LEN = 96
DATA.LABEL_LEN = 48
DATA.PRED_LEN = 96 / 192 / 336 / 720
MODEL.seq_len = 96
MODEL.label_len = 48
MODEL.pred_len = pred_len
```

Code location:

- Experiment parameter passing: `script/respect_tta/run_respect_tta.sh`
- Forecast call: `Spectral-master/tta/respect_tta.py`, inside `adapt_simple`

How to say it in presentation:

> The input is unchanged from the original forecasting setup. We do not redesign the backbone input; our method only adds an online residual correction after the backbone prediction.

## 4.2 Frozen Backbone

What it is:

- The pretrained forecasting model, such as DLinear, FreTS, iTransformer, MICN, OLS, or PatchTST.

Why it is added / kept frozen:

- The goal is test-time adaptation with a lightweight adapter.
- Freezing the backbone avoids unstable full-model updates during testing.
- It also reduces computation and prevents catastrophic changes to the pretrained forecasting ability.

What it does:

- Produces the base prediction `Y_t^(0)`.
- Serves as a stable forecasting prior.

How it is implemented:

The script disables training and loads a checkpoint:

```text
TRAIN.ENABLE False
TRAIN.CHECKPOINT_DIR ./checkpoints/seed{seed}/{model}/{dataset}_{pred_len}/
TEST.ENABLE False
TTA.ENABLE True
```

The adapter code freezes the model:

```text
self._freeze_all_model_params()
self._unfreeze_adapter_params()
```

Code location:

- `Spectral-master/tta/respect_tta.py`, `ReSpectTTAAdapter.__init__`

How to say it:

> The backbone remains frozen. We only update the small spectral adapter, so the method is lightweight and safer for online test-time adaptation.

## 4.3 Base Prediction `Y_t^(0)`

What it is:

- The original prediction from the frozen backbone before ReSpect-TTA correction.

Why it is needed:

- It is the main forecast that we want to correct.
- ReSpect-TTA is designed as a residual correction method, not a replacement forecasting model.

What it does:

- Provides both the final baseline term and the spectrum to be corrected.

How it is implemented:

Inside `adapt_simple`:

```text
pred, ground_truth = forecast(...)
original_pred = pred.clone()
```

Then `pred` can be adapted, while `original_pred` is preserved for residual computation and adapter training.

Code location:

- `Spectral-master/tta/respect_tta.py`, `adapt_simple`

How to say it:

> The base prediction is the frozen model output. ReSpect-TTA learns only a correction term on top of it.

## 4.4 Prediction Spectrum `F_y`

What it is:

- The real FFT of the base prediction:

```text
F_y = rFFT(Y_t^(0))
```

Why it is added:

- Many time-series errors appear as frequency patterns, such as trend mismatch, seasonal shift, or oscillation amplitude error.
- Working in frequency space lets the adapter correct different frequency components separately.

What it does:

- Converts the base prediction from time domain to frequency domain.
- Gives the correction module access to amplitude and phase information.

How it is implemented:

Inside `ReSpectDirectSpectralCorrector.forward`:

```text
fy = torch.fft.rfft(y.float(), n=self.pred_len, dim=1)
```

Shape:

```text
y:  [B, pred_len, V]
F_y: [B, pred_len//2 + 1, V]
```

Parameters:

```text
FFT_PRECISION = fp32
FFT_LENGTH_MODE = exact
```

Why `rFFT`:

- The prediction is real-valued, so `rFFT` is more efficient than full complex FFT.
- `rFFT` returns only the non-redundant frequency bins.

How to say it:

> I transform the prediction into the frequency domain so the correction can target spectral components instead of changing every time step independently.

## 4.5 Recent Residual Memory `R_buf`

What it is:

- A memory buffer of recent signed prediction errors.
- The residual is:

```text
R_t = Y_t - Y_t^(0)
```

Why it is added:

- At test time, the data distribution may shift.
- Recent residuals reveal how the frozen model is currently wrong.
- If the model repeatedly underestimates or misses periodic components, the residual memory captures that pattern.

What it does:

- Provides online feedback from recent test batches.
- Acts as the source of adaptation signal.

How it is implemented:

After ground truth arrives:

```text
signed_residual = targets - prediction
residual_profile = signed_residual.mean(dim=0)
self.sample_history.append(batch_info)
```

Important:

- The residual is computed against `original_pred`, the frozen backbone prediction.
- This keeps the memory focused on the backbone's raw error pattern.

Parameters:

```text
TTA.RESPECT.BUFFER_CONTEXT_SIZE = 10
TTA.RESPECT.SPECTRAL.HISTORY_SIZE = 8
```

Difference between the two:

- `BUFFER_CONTEXT_SIZE=10` is used by the context adapter path.
- `HISTORY_SIZE=8` is used by the direct spectral corrector to average recent residual profiles.

Code location:

- `Spectral-master/tta/respect_tta.py`, `update_memory_buffer`
- `Spectral-master/tta/respect_tta.py`, `_reference_profile`

How to say it:

> The residual memory stores how the model has recently failed. It gives the adapter an online signal about the current test distribution.

## 4.6 Residual Spectrum `F_r`

What it is:

- The frequency-domain representation of recent residual memory:

```text
F_r = rFFT(R_buf)
```

Why it is added:

- A raw residual in time domain tells us where errors happen.
- A residual spectrum tells us what type of error pattern exists: low-frequency trend error, high-frequency noise, seasonal mismatch, or phase shift.

What it does:

- Converts recent error behavior into spectral features.
- Provides the conditioning signal for the complex modulator.

How it is implemented:

The code first builds a reference residual profile:

```text
profile = average(last HISTORY_SIZE residual_profile entries)
```

Then it centers and windows the residual:

```text
ref = ref - ref.mean(dim=1, keepdim=True)
ref = ref * HannWindow
```

Then it applies `rFFT`:

```text
fr = torch.fft.rfft(ref.float(), n=self.pred_len, dim=1)
```

Parameters:

```text
HISTORY_SIZE = 8
WINDOW_TYPE = hann
FFT_LENGTH_MODE = exact
FFT_PRECISION = fp32
```

Why centering:

- Removing the mean prevents the DC component from dominating the residual spectrum.

Why Hann window:

- It reduces boundary artifacts when applying FFT to a finite-length residual profile.

How to say it:

> The residual spectrum summarizes recent forecasting errors by frequency. Instead of treating residuals as just another context vector, I use them to decide how to correct the prediction spectrum.

## 4.7 Complex Modulator `M(F_r)`

What it is:

- A trainable module that maps the residual spectrum into a complex multiplier.

Formula:

```text
M(F_r) = W [Re(F_r), Im(F_r), |F_r|, freq] + b
```

Why it is added:

- Simple residual addition can only add a residual-shaped signal.
- Complex modulation can change both amplitude and phase of the prediction spectrum.
- This is more expressive for time-series forecasting errors where the prediction may have the right general pattern but wrong strength or timing.

What it does:

- Generates a frequency-wise, variable-wise complex correction operator.
- The output has real and imaginary parts, so it can influence amplitude and phase.

How it is implemented:

Trainable parameters:

```text
complex_mod_weight: [n_freq, n_vars, 4, 2]
complex_mod_bias:   [n_freq, n_vars, 2]
```

Input features:

```text
fr_norm.real
fr_norm.imag
abs(fr_norm)
freq_coord
```

Code:

```text
features = torch.stack([fr_norm.real, fr_norm.imag, torch.abs(fr_norm), freq_coord], dim=-1)
raw = torch.einsum("bfvc,fvco->bfvo", features, self.complex_mod_weight)
raw = raw + self.complex_mod_bias.unsqueeze(0)
modulation = torch.complex(raw[..., 0], raw[..., 1])
```

Shape:

```text
F_r:       [B, n_freq, V]
features:  [B, n_freq, V, 4]
M(F_r):    [B, n_freq, V]
```

Parameters:

```text
CORRECTION_SOURCE = complex_modulation
COMPLEX_MOD_FREQ_CUTOFF_RATIO = 1.0
```

Why normalize `F_r`:

- The code divides by residual energy to avoid the modulator being dominated by scale.
- This makes the learned mapping focus more on residual pattern than raw magnitude.

Why initialize to zero:

- `complex_mod_weight` and `complex_mod_bias` start at zero.
- Therefore the initial correction is nearly zero.
- This makes the initial model equivalent to the frozen backbone and avoids harmful corrections before adaptation.

How to say it:

> The complex modulator learns how recent residual frequencies should modify the prediction frequencies. Because it is complex-valued, it can correct both amplitude and phase.

## 4.8 Element-Wise Spectral Modulation `M(F_r) odot F_y`

What it is:

- The residual-conditioned correction is applied by multiplying the modulator with the prediction spectrum:

```text
Z = M(F_r) odot F_y
```

Why it is added:

- This is the key difference from using residual spectrum only as context.
- The residual spectrum is not just sent into an MLP; it directly forms an operator acting on the prediction spectrum.

What it does:

- Scales and shifts the frequency behavior of the base prediction.
- Applies a different correction for each frequency and variable.

How it is implemented:

```text
modulation = self._complex_modulation(fr)
modulation = modulation * self.complex_mod_freq_mask
z = z + modulation * fy
```

Parameters:

```text
COMPLEX_MOD_FREQ_CUTOFF_RATIO = 1.0
```

Meaning of `COMPLEX_MOD_FREQ_CUTOFF_RATIO`:

- `1.0` means all frequency bins are allowed.
- Smaller values, such as `0.05`, `0.10`, or `0.20`, are low-pass ablations that only allow lower frequencies.

Why multiply instead of directly add:

- Addition says: "add this residual-shaped signal."
- Multiplication says: "use the residual spectrum to transform the prediction spectrum."
- Multiplication is more like a learned correction operator.

How to say it:

> This is why I say the residual spectrum acts as a correction operator. It directly modulates the prediction spectrum, instead of only being used as side information.

## 4.9 iFFT and Spectral Correction `Delta Y_t`

What it is:

- The inverse FFT of the modulated spectrum:

```text
Delta Y_t = iFFT(M(F_r) odot F_y)
```

Why it is added:

- The final forecast is in time domain.
- After learning a spectral correction, the method must convert it back into a time-domain signal.

What it does:

- Produces the additive correction term `Delta Y_t`.

How it is implemented:

```text
delta = torch.fft.irfft(z, n=self.pred_len, dim=1)
```

Shape:

```text
z:       [B, n_freq, V]
Delta Y: [B, pred_len, V]
```

How to say it:

> After spectral modulation, I transform the correction back to the time domain and add it to the original prediction.

## 4.10 Learned Correction Gate `g_t`

What it is:

- A learned gate controlling the correction strength.

Final formula:

```text
Y_hat_t = Y_t^(0) + g_t odot Delta Y_t
```

Why it is added:

- Spectral correction can help when the residual pattern is reliable.
- But if residual memory is noisy or weak, an aggressive correction can hurt.
- The gate lets the model decide how much correction to apply.

What it does:

- Controls correction magnitude for each variable.
- Prevents over-correction.
- Makes adaptation smoother and safer.

How it is implemented:

Gate features:

```text
hs              = mean of current prediction
l2              = residual spectral energy
low_freq_ratio  = ratio of low-frequency residual power
```

Code:

```text
gate_features = self._global_gate_features(y, low_freq_ratio, spectral_l2)
raw_global_gate = sigmoid(self.global_gate(gate_features).squeeze(-1))
global_gate = raw_global_gate
y_hat = y + global_gate.unsqueeze(1) * delta
```

Gate network:

```text
Linear(input_dim, hidden_dim)
ReLU
Linear(hidden_dim, 1)
Sigmoid
```

Default gate feature list:

```text
GLOBAL_GATE_FEATURES = hs,l2,low_freq_ratio
```

Main experiment gate parameters:

```text
USE_GLOBAL_GATE = True
GLOBAL_GATE_INIT_BIAS = -2.0
GLOBAL_GATE_LR_MULTIPLIER = 10.0
GLOBAL_GATE_FLOOR = 0.0
CONSTANT_GLOBAL_GATE = -1.0
```

Meaning:

- `GLOBAL_GATE_INIT_BIAS=-2.0` initializes the sigmoid gate around `sigmoid(-2) = 0.119`.
- So the correction starts small, but not completely closed.
- `GLOBAL_GATE_LR_MULTIPLIER=10.0` lets the gate adapt faster than the other adapter parameters.
- `GLOBAL_GATE_FLOOR=0.0` means there is no forced minimum correction.
- `CONSTANT_GLOBAL_GATE=-1.0` disables the constant gate and uses the learned gate.

Why not set the gate to 1:

- A full-strength correction can easily hurt early in adaptation.
- The gate makes the correction conservative at the beginning.

How to say it:

> The gate is a safety mechanism. It decides whether the spectral correction should be trusted for the current batch and variable.

## 4.11 Final Add / Fuse

What it is:

- The original prediction and gated correction are fused:

```text
Y_hat_t = Y_t^(0) + g_t odot Delta Y_t
```

Why it is added:

- The method is residual learning.
- Instead of generating a completely new forecast, it adjusts the base prediction.

What it does:

- Preserves the backbone's forecasting ability.
- Adds only the learned correction needed for current test-time conditions.

Code:

```text
y_hat = y + global_gate.unsqueeze(1) * delta
```

How to say it:

> The final forecast is still anchored on the frozen backbone. ReSpect-TTA only adds a gated spectral residual correction.

## 4.12 Online Update After Ground Truth Arrives

What it is:

- The adapter is updated online after the true target becomes available.

Why it is added:

- Test-time adaptation needs feedback from the test stream.
- The method learns from recent forecasting errors without retraining the full backbone.

What it does:

- Updates only adapter parameters.
- Updates residual memory for the next batch.

How it is implemented:

Prediction phase:

```text
with torch.no_grad():
    pred = self._apply_prediction_adapter(pred, raw_context)
```

Adapter update phase:

```text
adapted_pred = self._apply_prediction_adapter(original_pred, raw_context)
loss = MSE(adapted_pred, ground_truth)
loss = loss + 1e-4 * l2_reg
loss.backward()
clip_grad_norm_(adapter_params, max_norm=0.1)
optimizer.step()
```

Memory update:

```text
self.update_memory_buffer(ground_truth, original_pred)
```

Parameters:

```text
TTA.RESPECT.STEPS = 3
TTA.RESPECT.BATCH_SIZE = 48
```

Why update memory using `original_pred`:

- The residual memory should describe the frozen backbone error.
- If the code used the already-corrected prediction, the memory could mix model error and adapter behavior.

How to say it:

> After the label arrives, I train only the adapter using MSE and then update the residual memory. The backbone remains frozen throughout the process.

## 5. Optional Context Path in the Code

The simplified diagram mainly shows the direct spectral correction path. The main experiment also uses:

```text
SPECTRAL.USAGE = context_plus_direct
```

This means there are two adapter components:

1. A residual-context context adapter.
2. The direct ReSpect-TTA spectral correction module.

How it works:

```text
adapted = output_adapter(base_prediction, projected_context)
final   = direct_corrector(adapted, sample_history)
```

Where the context comes from:

- Recent target mean context.
- Residual spectral features from the previous residual buffer.
- Top `K=8` non-DC spectral bins.
- `real_imag` representation.

Parameters:

```text
SPECTRAL.USAGE = context_plus_direct
SPECTRAL.MODE = concat
SPECTRAL.K = 8
SPECTRAL.HISTORY_SIZE = 8
SPECTRAL.REPR = real_imag
SPECTRAL.SOURCE = residual_profile
USE_CONTEXT_PROJECTION = True
CONTEXT_PROJECTION_DIM = 8
```

Why it is not drawn in detail:

- The slide focuses on the new ReSpect-TTA contribution: direct residual-conditioned spectral correction.
- The context adapter is inherited from spectral context reference and is less central to the new story.

How to answer if asked:

> In the full implementation, I keep the spectral context reference context path and add the new direct correction path after it. The slide simplifies the context path because the main novelty is that the residual spectrum directly modulates the prediction spectrum.

## 6. Why Each Added Component Exists

| Added component | Why add it | Effect |
|---|---|---|
| Residual memory `R_buf` | Captures recent online forecasting errors | Gives the adapter test-time feedback |
| Residual spectrum `F_r` | Converts recent errors into frequency patterns | Identifies trend, periodic, and phase-related errors |
| Prediction spectrum `F_y` | Represents the forecast in frequency domain | Allows frequency-wise correction |
| Complex modulator `M(F_r)` | Uses residual spectrum to generate a correction operator | Can adjust amplitude and phase |
| Element-wise multiplication `M(F_r) odot F_y` | Applies residual-conditioned correction directly to prediction frequencies | Makes residual spectrum an operator, not just context |
| iFFT | Converts spectral correction back to time domain | Produces `Delta Y_t` |
| Learned gate `g_t` | Controls correction strength | Reduces over-correction risk |
| Hann window | Reduces FFT boundary artifacts | Makes residual spectrum smoother |
| Diagonal output projection | Adds a small frequency-variable scaling after modulation | Gives extra flexibility with low parameter cost |
| High gate LR | Lets the correction gate adapt quickly | Helps the model open/close correction online |
| Frozen backbone | Prevents unstable full-model updates | Keeps adaptation lightweight and conservative |

## 7. Exact Main Variant Parameters

The main ReSpect-TTA variant corresponds to:

```text
context_plus_complex_modulation_high_gate_lr
```

Experiment tag:

```text
adapter_context_plus_complex_modulation_high_gate_lr
```

Key parameters passed by the script:

| Parameter | Value | Meaning |
|---|---:|---|
| `TTA.ENABLE` | `True` | Enable test-time adaptation |
| `TRAIN.ENABLE` | `False` | Do not train the backbone |
| `TTA.RESPECT.BATCH_SIZE` | `48` | Online adaptation batch size |
| `TTA.RESPECT.STEPS` | `3` | Adapter gradient steps per batch |
| `TTA.RESPECT.BUFFER_CONTEXT_SIZE` | `10` | Context memory length |
| `TTA.RESPECT.SPECTRAL.USAGE` | `context_plus_direct` | Use context adapter plus direct correction |
| `TTA.RESPECT.SPECTRAL.MODE` | `concat` | Concatenate mean context and spectral context |
| `TTA.RESPECT.SPECTRAL.K` | `8` | Use first 8 non-DC spectral bins for context/gate statistics |
| `TTA.RESPECT.SPECTRAL.HISTORY_SIZE` | `8` | Average up to 8 recent residual profiles |
| `TTA.RESPECT.SPECTRAL.ENABLE_DIRECT_CORRECTION` | `True` | Activate direct spectral correction |
| `TTA.RESPECT.SPECTRAL.CORRECTION_SOURCE` | `complex_modulation` | Use residual-conditioned complex modulation |
| `TTA.RESPECT.SPECTRAL.GATE_MODE` | `unidirectional` | Gate uses residual-to-prediction direction |
| `TTA.RESPECT.SPECTRAL.USE_GLOBAL_GATE` | `True` | Use learned correction gate |
| `TTA.RESPECT.SPECTRAL.GLOBAL_GATE_INIT_BIAS` | `-2.0` | Start with a small open gate |
| `TTA.RESPECT.SPECTRAL.GLOBAL_GATE_LR_MULTIPLIER` | `10.0` | Train gate faster |
| `TTA.RESPECT.SPECTRAL.GLOBAL_GATE_FLOOR` | `0.0` | No forced minimum correction |
| `TTA.RESPECT.SPECTRAL.CONSTANT_GLOBAL_GATE` | `-1.0` | Disable constant gate |
| `TTA.RESPECT.SPECTRAL.COMPLEX_MOD_FREQ_CUTOFF_RATIO` | `1.0` | Allow all frequencies |
| `TTA.RESPECT.SPECTRAL.WINDOW_TYPE` | `hann` | Window residual before FFT |
| `TTA.RESPECT.SPECTRAL.OUT_PROJ` | `diagonal` | Use low-cost frequency-wise output scaling |
| `TTA.RESPECT.SPECTRAL.FFT_PRECISION` | `fp32` | FFT uses float32 path |
| `TTA.RESPECT.SPECTRAL.FFT_LENGTH_MODE` | `exact` | FFT length equals prediction length |

Datasets and models:

```text
Models:   DLinear, FreTS, iTransformer, MICN, OLS, PatchTST
Datasets: ETTh1, ETTh2, ETTm1, ETTm2, exchange_rate, weather, electricity, traffic
Pred lens: 96, 192, 336, 720
Seeds:    0, 1, 2
```

## 8. Shapes and Data Flow

Let:

```text
B = batch size
L = prediction length
V = number of variables
F = L // 2 + 1
```

Then:

| Symbol | Shape | Meaning |
|---|---|---|
| `X_t` | `[B, seq_len, V]` | Historical input window |
| `Y_t^(0)` | `[B, L, V]` | Frozen backbone prediction |
| `R_t` | `[B, L, V]` | Signed residual |
| `residual_profile` | `[L, V]` | Batch-averaged residual |
| `R_buf` | `[B, L, V]` | Averaged recent residual profile expanded to batch |
| `F_y` | `[B, F, V]` complex | Prediction spectrum |
| `F_r` | `[B, F, V]` complex | Residual spectrum |
| `M(F_r)` | `[B, F, V]` complex | Complex spectral modulator |
| `Z` | `[B, F, V]` complex | Modulated correction spectrum |
| `Delta Y_t` | `[B, L, V]` | Time-domain correction |
| `g_t` | `[B, V]` | Correction gate |
| `Y_hat_t` | `[B, L, V]` | Final forecast |

## 9. Why This Is Different From spectral context reference

Short version:

> spectral context reference uses residual spectrum as context; ReSpect-TTA uses it as a direct correction operator on the prediction spectrum.

More complete version:

- spectral context reference builds spectral context features from residual history and feeds them into an output adapter.
- ReSpect-TTA still can keep that context path, but adds a direct spectral operation:

```text
Delta Y_t = iFFT(M(F_r) odot F_y)
```

- This makes the residual spectrum actively transform the prediction spectrum.

## 10. Why Use Residual Spectrum as an Operator

The phrase "residual spectrum acts as a correction operator" means:

- The residual spectrum does not only describe the recent error.
- It is used to generate `M(F_r)`.
- `M(F_r)` is multiplied with the prediction spectrum `F_y`.
- Therefore the residual pattern directly changes the prediction's frequency components.

In simple words:

> The residual tells the model not only what went wrong, but also how to modify the next prediction.

## 11. What the Professor May Ask

### Q1. Why use residuals instead of only the input `X_t`?

Because residuals directly describe the current model error under test-time distribution shift. The input tells us the signal, but the residual tells us how the frozen model is failing.

### Q2. Why FFT?

Many forecasting errors are frequency-structured. FFT lets the adapter separately correct low-frequency trend errors, seasonal components, and high-frequency variations.

### Q3. Why complex modulation?

A real-valued scale mostly changes amplitude. A complex multiplier can change both amplitude and phase, which is important when the prediction has timing or phase mismatch.

### Q4. Does this use future information?

No. The prediction for the current batch uses only residual memory from previous batches. The current ground truth is used only after prediction to update the adapter and memory for future batches.

### Q5. Why freeze the backbone?

To keep adaptation stable and lightweight. Updating the full model online can overfit or destabilize the pretrained forecasting model.

### Q6. Why need the gate `g_t`?

The residual memory may sometimes be noisy. The gate controls correction strength and reduces the risk of over-correction.

### Q7. Why initialize the complex modulator to zero?

So the initial correction is zero and the model initially behaves like the frozen backbone. This makes online adaptation safer.

### Q8. Why use `GLOBAL_GATE_INIT_BIAS=-2.0`?

Because `sigmoid(-2.0)` is about `0.119`, so the correction starts small but not completely closed. This allows cautious early correction.

### Q9. Why use `GLOBAL_GATE_LR_MULTIPLIER=10.0`?

The gate needs to adapt quickly online because it decides whether the correction should be trusted. A higher learning rate helps the gate open or close faster.

### Q10. Why use Hann window?

FFT assumes periodic continuation of the sequence. A Hann window reduces boundary discontinuities in the residual profile and makes the residual spectrum less noisy.

### Q11. Why multiply `M(F_r)` with `F_y` instead of directly adding `F_r`?

Adding `F_r` would simply inject a residual-shaped signal. Multiplying `F_y` makes the residual spectrum a transformation of the prediction spectrum, which is more flexible.

### Q12. What exactly is trained?

Only adapter parameters are trained:

- context adapter, if `context_plus_direct` is enabled
- context projector, if enabled
- complex modulator weights and biases
- correction gate
- output projection parameters

The backbone remains frozen.

### Q13. What is the loss?

The adapter is updated with:

```text
MSE(adapted_prediction, ground_truth) + 1e-4 * L2(adapter_parameters)
```

### Q14. What is the main limitation?

The method depends on recent residuals being informative. If the residual memory is too noisy, too short, or the distribution changes abruptly, the correction may become unreliable.

## 12. Minimal Slide Script

Use this if you only have one slide:

> This slide shows ReSpect-TTA, our residual-conditioned spectral correction method. The frozen backbone first produces a base prediction `Y_t^(0)`. We keep the backbone fixed and store recent signed residuals after ground truth becomes available. Then we transform both the base prediction and the residual memory into the frequency domain. The key idea is that the residual spectrum is not only used as context. It generates a complex modulator `M(F_r)` that directly corrects the prediction spectrum `F_y`. After inverse FFT, we obtain a time-domain correction `Delta Y_t`, and a learned gate `g_t` controls how much of this correction should be added. So the final forecast is the base prediction plus a gated spectral residual correction.

## 13. Shortest Possible Explanation

```text
Backbone gives Y_t^(0).
Past residuals give F_r.
Prediction gives F_y.
ReSpect-TTA learns M(F_r).
Correction is iFFT(M(F_r) odot F_y).
Final output is Y_t^(0) + gated correction.
```

## 14. Code Location Cheat Sheet

| Concept | File / function |
|---|---|
| Main dispatch to ReSpect-TTA adapter | `Spectral-master/main.py` |
| Build adapter when `RESULT_DIR` contains `RESPECT_TTA` | `Spectral-master/main.py` |
| Main adapter class | `Spectral-master/tta/respect_tta.py`, `ReSpectTTAAdapter` |
| Direct spectral correction module | `Spectral-master/tta/respect_tta.py`, `ReSpectDirectSpectralCorrector` |
| Residual memory update | `Spectral-master/tta/respect_tta.py`, `update_memory_buffer` |
| Build residual reference profile | `Spectral-master/tta/respect_tta.py`, `_reference_profile` |
| Prediction and residual FFT | `Spectral-master/tta/respect_tta.py`, `forward` |
| Complex modulator | `Spectral-master/tta/respect_tta.py`, `_complex_modulation` |
| Gate features | `Spectral-master/tta/respect_tta.py`, `_global_gate_features` |
| Final formula | `Spectral-master/tta/respect_tta.py`, `forward` |
| Online adapter training | `Spectral-master/tta/respect_tta.py`, `adapt_simple` |
| residual-context spectral context | `Spectral-master/tta/residual_spectral_context.py`, `SpectralContextBuilder` |
| Experiment parameters | `script/respect_tta/run_respect_tta.sh` |

## 15. Things Not to Overclaim

Do not say:

- This is a completely new forecasting backbone.
- This always improves every dataset and horizon.
- This uses the current ground truth before prediction.
- This replaces spectral context reference completely.

Better to say:

- It is a lightweight test-time residual correction module.
- It is added on top of a frozen pretrained backbone.
- It uses recent residual spectra to adapt online.
- The main novelty is direct residual-conditioned spectral modulation.
- The results need to be analyzed by dataset, horizon, and model because online correction can help or hurt depending on residual stability.

## 16. Clean Answer If Asked "What Did You Add?"

> Starting from the spectral/context adapter reference baseline, I added a direct residual-conditioned spectral correction path. It stores recent residuals, transforms them with rFFT, uses the residual spectrum to generate a complex modulator, applies that modulator to the prediction spectrum, converts the result back with iFFT, and adds it to the base forecast through a learned gate.

## 17. Clean Answer If Asked "Why Is This Reasonable?"

> Under test-time distribution shift, the frozen model's recent residuals reveal systematic errors. If those errors have spectral structure, such as trend or seasonality mismatch, then correcting the prediction in the frequency domain can be more targeted than only adapting in the time domain.

## 18. Clean Answer If Asked "What Are the Parameters?"

> The main setting uses `context_plus_direct`, `complex_modulation`, `K=8`, `HISTORY_SIZE=8`, `hann` window, `diagonal` output projection, learned global gate, gate initialization bias `-2.0`, gate learning-rate multiplier `10.0`, `TTA_STEPS=3`, and batch size `48`.

