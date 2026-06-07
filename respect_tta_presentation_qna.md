# ReSpect-TTA Presentation Q&A

This file lists likely professor questions and short answers for presenting the
ReSpect-TTA slide. The goal is to answer clearly without overclaiming.

## 0. One-Sentence Summary

**Q: Can you summarize your method in one sentence?**

A: ReSpect-TTA keeps the forecasting backbone frozen, uses recent prediction
residuals to build a residual spectrum, and uses that spectrum to generate a
complex modulator that directly corrects the current prediction spectrum.

Chinese version:

> ReSpect-TTA 不改原来的预测模型，而是用最近预测误差的频谱生成一个复数调制器，直接修正当前预测的频谱。

## 1. Motivation

**Q: What problem are you solving?**

A: The problem is distribution shift in time-series forecasting. A model trained
on historical data may become less accurate after deployment because the future
stream can have different trends, variance, periodicity, or cross-variable
relationships.

**Q: Why use test-time adaptation here?**

A: Forecasting is suitable for online test-time adaptation because the ground
truth becomes available after a delay. Once the true value is observed, we can
compute the previous prediction error and update a lightweight adapter without
retraining the full backbone.

**Q: Why not just retrain the whole model?**

A: Full retraining is expensive and may not be practical during deployment.
ReSpect-TTA only updates a small adapter, so it is cheaper, faster, and less
likely to destroy the pretrained backbone behavior.

## 2. Relation to COSA-Spectral

**Q: What is the difference between COSA-Spectral and ReSpect-TTA?**

A: COSA-Spectral uses the residual spectrum as context features for a
time-domain adapter. ReSpect-TTA uses the residual spectrum to generate a
complex operator that directly modifies the prediction spectrum.

Short version:

> COSA-Spectral uses spectrum as context; ReSpect-TTA uses spectrum as a correction operator.

**Q: What does "operator" mean here?**

A: It means the residual spectrum is not only extra information. It is used to
produce \(M(F_r)\), and \(M(F_r)\) directly acts on the prediction spectrum:

\[
\Delta Y_t = \mathrm{iFFT}(M(F_r) \odot F_y).
\]

So the residual spectrum changes the prediction spectrum through multiplication.

**Q: Are you replacing COSA-Spectral?**

A: Not exactly. The main variant keeps the useful COSA-Spectral context path and
adds a direct spectral correction path. So it is better described as an
extension of COSA-Spectral.

## 3. Diagram and Formula Questions

**Q: What is \(Y_t^{(0)}\)?**

A: It is the original prediction from the frozen backbone:

\[
Y_t^{(0)} = f_\theta(X_t).
\]

**Q: What is \(R_{buf}\)?**

A: \(R_{buf}\) is the recent residual memory. It stores recent prediction errors
computed after the true target becomes available:

\[
R_t = Y_t - Y_t^{(0)}.
\]

The method averages recent residual profiles to form the residual reference.

**Q: What is \(F_y\)?**

A: \(F_y\) is the frequency representation of the current base prediction:

\[
F_y = \mathrm{rFFT}(Y_t^{(0)}).
\]

**Q: What is \(F_r\)?**

A: \(F_r\) is the frequency representation of recent residual memory:

\[
F_r = \mathrm{rFFT}(R_{buf}).
\]

**Q: What is \(M(F_r)\)?**

A: \(M(F_r)\) is a learned complex modulator generated from the residual
spectrum. It is a trainable mapping from residual spectral features to a complex
multiplier.

In code, the features include:

\[
[\Re(F_r), \Im(F_r), |F_r|, \omega],
\]

where \(\omega\) is the normalized frequency coordinate.

**Q: Why is \(M(F_r)\) complex?**

A: Because FFT coefficients are complex. A complex modulator can adjust both
amplitude and phase of the prediction spectrum. Amplitude controls how strong a
frequency component is, and phase controls where the periodic pattern is shifted.

**Q: What does \(M(F_r) \odot F_y\) mean?**

A: It is element-wise spectral modulation. Each frequency component of the
prediction spectrum is adjusted by the residual-conditioned complex modulator.

**Q: Why multiply instead of adding the residual spectrum directly?**

A: Directly adding residual spectra can be unstable. Multiplication makes the
correction depend on the current prediction spectrum, so the correction is
relative to what the backbone already predicts. This is more controlled than
injecting residual content directly.

**Q: What is \(\Delta Y_t\)?**

A: \(\Delta Y_t\) is the time-domain correction reconstructed from the modulated
spectrum:

\[
\Delta Y_t = \mathrm{iFFT}(M(F_r) \odot F_y).
\]

**Q: What is \(g_t\)? Where does it come from?**

A: \(g_t\) is a learned correction gate. It controls how much of the spectral
correction should be added to the base prediction:

\[
\hat{Y}_t = Y_t^{(0)} + g_t \odot \Delta Y_t.
\]

The gate is produced by a small neural module using features such as prediction
summary, residual spectral energy, and low-frequency residual ratio.

Short answer:

> \(g_t\) is a learned gate that prevents over-correction.

**Q: Why do you need the gate?**

A: The residual spectrum may not always predict future error correctly. The gate
acts as a safety control. If the correction is reliable, it can open more; if
the residual signal is noisy or mismatched, it can reduce the correction.

## 4. Online Adaptation

**Q: Do you use future labels when making the current prediction?**

A: No. The current prediction only uses residuals from previous completed
predictions. The true \(Y_t\) is used only after it becomes available, to update
the adapter for future predictions.

**Q: Is there data leakage?**

A: The intended protocol is leakage-free because the residual buffer contains
only past observed errors. The ground truth of the current prediction is not used
before generating \(\hat{Y}_t\).

**Q: What parameters are updated?**

A: The backbone is frozen. Only the lightweight adapter parameters are updated:
the context adapter, the complex modulation parameters, and the gate parameters.

**Q: What loss do you use for adaptation?**

A: The online adaptation loss is MSE between the adapted prediction and the
ground truth after the ground truth becomes available:

\[
\mathcal{L}_{adapt} = \mathrm{MSE}(\hat{Y}_t, Y_t).
\]

**Q: What happens at the beginning when there is no residual history?**

A: The residual reference is initialized as zeros. Also, the complex modulation
parameters are zero-initialized, so the method initially behaves close to the
frozen backbone and learns corrections only after residual evidence accumulates.

## 5. Why Frequency Domain?

**Q: Why use FFT at all?**

A: Time-series errors often contain periodic or long-range structure. FFT makes
these patterns easier to represent because periodic errors can concentrate in
frequency components.

**Q: Why not just use a time-domain residual adapter?**

A: A time-domain adapter can correct local point-wise errors, but it may not
explicitly capture global periodic or phase-related error patterns. The spectral
path directly targets frequency structure.

**Q: Why use rFFT instead of FFT?**

A: The input time series is real-valued. rFFT is the efficient real-valued FFT
that keeps only the non-redundant frequency coefficients.

**Q: What does the residual spectrum tell us?**

A: It tells us where the model has recently made structured errors in the
frequency domain. For example, the model may consistently underfit a periodic
component or shift a pattern in phase.

## 6. Experimental Result Questions

**Q: What are your main baselines?**

A: The main baselines are no-TTA, COSA fixed, COSA PAAS, and COSA-Spectral. The
most important comparison is COSA-Spectral because it already uses residual FFT
features as context.

**Q: What is the main result?**

A: On 171 collected model-dataset-horizon conditions averaged over 3 seeds,
ReSpect-TTA wins over no-TTA on 161/171 conditions with +14.37% average MSE
improvement. It wins over COSA-Spectral on 162/171 conditions with +3.07%
average MSE improvement.

**Q: Why is the improvement over COSA-Spectral smaller than over no-TTA?**

A: COSA-Spectral is already a strong baseline because it uses spectral residual
features as context. The +3.07% improvement is meaningful because it shows extra
value beyond an already spectral-aware method.

**Q: Which model benefits most?**

A: MICN benefits most in the collected results. Against COSA-Spectral, MICN wins
32/32 conditions with about +4.86% mean MSE improvement.

**Q: Which datasets are strongest?**

A: The strongest average improvements appear on exchange_rate, weather,
electricity, traffic, and several MICN settings. Against COSA-Spectral, all
collected exchange_rate, weather, electricity, traffic, and ETTh1 conditions
improve.

**Q: Where does it fail?**

A: The main negative-transfer cases are concentrated on ETTm1 and ETTm2 at
horizon 720, especially for OLS, FreTS, DLinear, and PatchTST.

**Q: Why does it fail on ETTm1/ETTm2 horizon 720?**

A: The likely reason is that recent residual spectra may not extrapolate cleanly
to very long horizons. The gate may also open too much or adapt too slowly,
causing over-correction.

**Q: Do you have statistical significance tests?**

A: Not yet. The current project reports averaged results over 3 seeds and
condition-level wins. A future improvement is to add paired seed-level tests,
bootstrap confidence intervals, or Wilcoxon signed-rank tests.

**Q: Why only 171 conditions instead of 192?**

A: The theoretical grid is 6 models x 8 datasets x 4 horizons = 192. The current
collected grid has 171 because some checkpoints or valid runs were unavailable,
especially traffic for several backbones and electricity for some PatchTST/OLS
conditions. The report only claims results on the 171 comparable conditions.

## 7. Ablation and Design Questions

**Q: Did direct spectral correction alone work?**

A: Not enough. Direct complex modulation alone improved over no-TTA in some
settings, but it was weaker than COSA-Spectral, especially for long horizons.

**Q: What made the final method work?**

A: The final method combines two paths: the retained Spectral-COSA context path
and the direct complex modulation path. The context path stabilizes adaptation,
and the direct path adds frequency-native correction.

**Q: What did the gate diagnostics show?**

A: Gate policy is important. A constant gate of 0.10 performed strongly on the
horizon-720 diagnostic slice, winning 40/42 against COSA-Spectral with +3.15%
average MSE improvement. This suggests gate stabilization is a key future target.

**Q: Why not always use constant gate 0.10?**

A: Constant gate 0.10 is strong for long-horizon diagnostics, but it may not be
universally optimal. MICN diagnostics show that model-specific adaptivity still
matters, so a learned horizon-aware gate is likely better.

**Q: What is the most important future ablation?**

A: The most important future ablation is separating the contribution of:

- Spectral-COSA context path
- direct complex modulation path
- learned gate policy
- gate learning-rate multiplier
- constant/floor gate variants

## 8. Code Questions

**Q: Where is the core code?**

A: The core adapter is implemented in:

```text
Spectral-master/tta/cosa_scheme6.py
```

The final presentation name is ReSpect-TTA, but the historical code file still
uses the `scheme6` name.

**Q: Which class implements the spectral correction?**

A: `Scheme6DirectSpectralCorrector` implements the direct spectral correction:
rFFT, residual spectrum, complex modulation, iFFT, and gate.

**Q: Which class implements the full online adapter?**

A: `Scheme6ResidualSpectralAdapter` wraps the frozen backbone, optional
Spectral-COSA context adapter, direct spectral corrector, online adaptation
loop, residual buffer update, and result logging.

**Q: Where is the experiment script?**

A: The main experiment script is:

```text
script/05.29_residual_spectral_gating/run_residual_spectral_gating.sh
```

The final main variant is:

```text
adapter_context_plus_complex_modulation_high_gate_lr
```

**Q: Where are the main result files?**

A: The main result files are:

```text
result/06.05_scheme6_experiment_records/spectral_master_results/summary/scheme6_full_context_plus_complex_averaged.csv
result/06.05_scheme6_experiment_records/spectral_master_results/summary/scheme6_full_context_plus_complex_improvement.csv
```

## 9. Potential Criticism and Safe Answers

**Q: Is this a completely new architecture?**

A: No. It is not a new forecasting backbone. It is a lightweight test-time
adaptation adapter placed after existing backbones.

**Q: Are you claiming it always improves?**

A: No. It improves most collected conditions, but there are negative-transfer
cases, mainly ETTm1/ETTm2 at horizon 720.

**Q: What is the novelty then?**

A: The novelty is using the residual spectrum to generate a complex modulation
operator that directly corrects the prediction spectrum. COSA-Spectral uses
spectral information as context; ReSpect-TTA uses it as a correction operator.

**Q: Could the improvement come only from extra parameters?**

A: That is possible and should be tested more carefully. The current evidence
shows consistent empirical gains, but a future version should report parameter
counts and ablations that isolate each component.

**Q: Could the improvement come from the retained COSA-Spectral context path?**

A: Partly, yes. That is why the claim is framed as an extension rather than a
replacement. The final method combines context and direct correction; future
ablation should isolate their individual contributions.

**Q: Why should we trust the result if some grid points are missing?**

A: The report is transparent and only claims the 171 collected comparable
conditions. It does not claim the full 192-condition grid. Missing combinations
should be filled in future work.

## 10. Short Answers to Memorize

**What is ReSpect-TTA?**

> A lightweight online TTA adapter that uses recent residual spectra to directly
> correct the current prediction spectrum.

**Main difference from COSA-Spectral?**

> COSA-Spectral uses residual spectrum as context; ReSpect-TTA uses it as a
> correction operator.

**Why frequency domain?**

> Because time-series errors often have periodic structure, and FFT exposes
> amplitude and phase patterns.

**What is \(g_t\)?**

> A learned gate that controls the strength of the spectral correction.

**What is updated during test time?**

> Only the adapter, complex modulator, and gate; the backbone remains frozen.

**Main result?**

> +14.37% average MSE improvement over no-TTA and +3.07% over COSA-Spectral on
> 171 collected conditions.

**Main limitation?**

> Negative transfer remains on ETTm1/ETTm2 at horizon 720, so gate stabilization
> is the next improvement target.

