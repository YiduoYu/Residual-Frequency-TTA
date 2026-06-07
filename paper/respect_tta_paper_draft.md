# ReSpect-TTA: Residual-Conditioned Complex Frequency Modulation for Online Test-Time Adaptation in Time-Series Forecasting

**Author(s):** To be completed  
**Affiliation(s):** To be completed  
**Corresponding Author:** To be completed  
**Draft date:** 2026-06-05  
**Paper type:** Empirical machine learning conference paper draft  
**Citation style:** APA 7 draft style  

## English Abstract

Modern time-series forecasters often degrade after deployment because the statistical structure of the observed stream changes over time. Test-time adaptation (TTA) addresses this problem by updating a small set of parameters during inference after ground truth becomes available, but existing output-space adapters usually correct predictions in the time domain. This paper presents ReSpect-TTA, a residual-conditioned complex frequency modulation method for online TTA in long-term time-series forecasting. ReSpect-TTA keeps the base forecasting model frozen, maintains COSA-style online residual memory, and uses the recent residual spectrum to generate a complex modulator applied directly to the current prediction spectrum. The corrected forecast is reconstructed through inverse FFT and controlled by a learned global gate. We evaluate the current main variant, `adapter_context_plus_complex_modulation_high_gate_lr`, on a collected grid of 171 model-dataset-horizon conditions, each averaged over three seeds, covering six forecasting backbones, eight benchmark datasets, and four prediction horizons where checkpoints and valid runs are available. Against the no-TTA baseline, ReSpect-TTA wins on 161/171 conditions with an average MSE improvement of 14.37%. Against COSA-Spectral, it wins on 162/171 conditions with an average MSE improvement of 3.07% and an average MAE improvement of 1.14%. The strongest relative gains appear on MICN, exchange-rate, weather, electricity, and several long-horizon settings, while negative transfer remains concentrated on ETTm1/ETTm2 at horizon 720 for several backbones. These results indicate that residual-conditioned complex spectral correction is a promising extension of output-space TTA, but its gate policy requires further study before the method can be treated as a final architecture.

**Keywords:** time-series forecasting, test-time adaptation, frequency-domain learning, residual correction, complex modulation, COSA, online adaptation

## 中文摘要（zh-TW）

長期時間序列預測模型在部署後常面臨非定態與分布漂移，使預訓練模型的誤差隨時間累積。本文提出 ReSpect-TTA，一種用於線上 test-time adaptation 的 residual-conditioned complex frequency modulation 方法。此方法保留 frozen backbone 與 COSA 式 residual buffer，但不再只以時域 adapter 修正輸出，而是將近期 residual profile 轉換為頻域參考，產生複數調制器 \(M(F_r)\)，直接作用於目前預測頻譜 \(F_y\)，再經 inverse FFT 回到時域，並以 global gate 控制修正幅度。本文根據目前已完成的 171 個 model-dataset-horizon 條件（每個條件三個 seeds）進行分析。結果顯示，ReSpect-TTA 相較 no-TTA baseline 在 161/171 個條件上取得較低 MSE，平均 MSE 改善 14.37%；相較 COSA-Spectral，則在 162/171 個條件上勝出，平均 MSE 改善 3.07%、平均 MAE 改善 1.14%。然而，ETTm1/ETTm2 的 720-step horizon 仍出現少數負遷移，顯示 gate policy 與長期穩定性仍需進一步消融。整體而言，ReSpect-TTA 支持以 residual spectrum 參數化頻域修正算子的研究方向，但目前更適合被定位為具實證潛力的主候選，而非已完成的最終方法。

**關鍵詞：** 時間序列預測、測試時間適應、頻域學習、殘差修正、複數調制、COSA、線上適應

## 1. Introduction

Long-term time-series forecasting models are typically trained on historical data and then evaluated on future streams whose statistical properties may differ from the training distribution. This mismatch is not incidental. Mean, variance, periodic structure, and cross-variate relationships often change over time, and prior work has treated this non-stationarity as a central obstacle for accurate forecasting (Kim et al., 2022; Liu et al., 2022). Static forecasters can therefore lose accuracy after deployment even when their original validation performance is strong.

Test-time adaptation (TTA) offers a practical response to this problem. In forecasting, unlike many vision tasks, the ground truth for a prediction becomes observable after a delay. This makes it possible to update lightweight adaptation modules online using a direct regression loss, while keeping the base model frozen. COSA follows this idea by correcting the frozen model's output through a context-aware output-space adapter. Its reported gains show that residual correction can be effective without retraining the backbone (Im & Kwon, 2026). Recent parameter-efficient TTA work has also emphasized small calibration modules, dynamic gating, and frequency-domain losses as tools for adaptation under non-stationarity (Medeiros et al., 2025).

This paper studies a frequency-native extension of that line of work. ReSpect-TTA is motivated by a simple observation: if the recent residuals contain structured error modes, then their frequency representation may be a better conditioning signal than a purely time-domain summary. Frequency-domain forecasting models such as FEDformer and FreTS argue that Fourier representations can expose global structure, energy concentration, and long-range dependencies that are difficult to capture with point-wise mappings alone (Yi et al., 2023; Zhou et al., 2022). ReSpect-TTA applies this intuition not to the base forecaster itself, but to online residual correction.

The current implementation uses the recent residual profile to generate a complex modulator. Given a frozen base prediction \(Y_t^{(0)}\), ReSpect-TTA computes its spectrum \(F_y\). It also builds a residual reference buffer \(R_{buf}\) from recent observed errors and computes \(F_r\). A trainable residual-conditioned operator \(M_\phi(F_r)\) then produces a complex modulation applied to \(F_y\). The resulting correction is reconstructed by inverse FFT and mixed back into the prediction through a global gate:

\[
\Delta Y_t = \mathrm{iFFT}(M_\phi(F_r) \odot F_y), \quad
\hat{Y}_t = Y_t^{(0)} + g_t \odot \Delta Y_t.
\]

The key distinction from COSA-Spectral is the correction path. COSA-Spectral uses residual FFT features as context for a time-domain output adapter. ReSpect-TTA uses residual FFT features to parameterize a complex operator that acts directly on the prediction spectrum.

This draft addresses three research questions:

**RQ1.** Does residual-conditioned complex frequency modulation improve online TTA performance over no-TTA and COSA baselines?

**RQ2.** Does ReSpect-TTA add value beyond COSA-Spectral, where spectral information is used only as time-domain adapter context?

**RQ3.** Which diagnostics suggest the next stabilization target for ReSpect-TTA?

The current evidence supports ReSpect-TTA as a promising main candidate. Across 171 collected averaged conditions, it improves over no-TTA, fixed COSA, PAAS COSA, and COSA-Spectral in most comparable cases. At the same time, the results do not justify an overclaim of universal improvement. The remaining negative-transfer cases are concentrated in long-horizon ETTm1/ETTm2 conditions, and gate diagnostics indicate that long-horizon stability is still a live design problem.

## 2. Related Work

### 2.1 Test-Time Adaptation for Forecasting

TTA for time-series forecasting differs from vision TTA because labels become available sequentially after prediction. COSA exploits this property through a context-aware output-space adapter that directly corrects a frozen model's prediction. It uses recent ground-truth statistics and residual signals to guide lightweight online updates, with only adapter parameters updated at test time (Im & Kwon, 2026). ReSpect-TTA keeps this deployment model: the base forecasting model remains frozen, online residual memory is retained, and adaptation uses observed labels after prediction.

PETSA provides a complementary perspective. It argues that parameter-efficient TTA can adapt forecasting models by updating small calibration modules rather than the full model, and it includes a frequency-domain term to preserve periodicity (Medeiros et al., 2025). ReSpect-TTA is different in architecture and loss design, but it shares the same premise that adaptation should be small, gated, and compatible with frozen backbones.

### 2.2 Frequency-Domain Forecasting

Frequency-domain modeling has become a recurring strategy in long-term forecasting. FEDformer combines seasonal-trend decomposition with frequency-enhanced Transformer blocks and reports reduced prediction error on benchmark datasets (Zhou et al., 2022). FreTS builds MLP forecasting modules directly in the complex frequency domain and attributes part of its effectiveness to the global view and energy compaction properties of the spectrum (Yi et al., 2023). These findings support the basic design assumption behind ReSpect-TTA: residual errors may contain structured frequency information that is useful for long-horizon correction.

However, ReSpect-TTA is not a new base forecaster. It is an adaptation layer placed after existing backbones such as DLinear, FreTS, iTransformer, MICN, OLS, and PatchTST. This makes the contribution narrower but also more modular. If a residual-conditioned spectral corrector improves multiple frozen backbones, it can be treated as an adapter-level improvement rather than a backbone-specific architectural change.

### 2.3 Non-Stationarity, Normalization, and Backbone Diversity

Non-stationarity remains a major theme in time-series forecasting. RevIN removes and restores instance-level statistics to address distribution shift (Kim et al., 2022), while Non-stationary Transformers combine stationarization with de-stationary attention to avoid losing informative non-stationary structure (Liu et al., 2022). These works frame a tension relevant to ReSpect-TTA: reducing distribution shift can improve predictability, but overly aggressive correction can remove useful signal or induce negative transfer.

Backbone choice also matters. DLinear showed that simple linear models can be strong baselines for long-term forecasting (Zeng et al., 2023). PatchTST uses patching and channel independence to improve Transformer forecasting (Nie et al., 2023). iTransformer inverts the usual Transformer tokenization so attention operates over variate tokens rather than temporal tokens (Liu et al., 2024). MICN combines local and global context with multi-scale convolutional branches (Wang et al., 2023). ReSpect-TTA is evaluated across these heterogeneous families to test whether the adapter improves output correction rather than exploiting one backbone's weakness.

## 3. Method

### 3.1 Problem Setting

Let \(f_\theta\) denote a pretrained forecasting model. At test time, the model receives an input window \(X_t\) and produces a prediction \(Y_t^{(0)} = f_\theta(X_t)\) for a horizon of length \(L\). After a delay, the true target \(Y_t\) becomes available. The online adaptation protocol updates only adapter parameters and leaves \(\theta\) frozen. The residual for a completed prediction is

\[
R_t = Y_t - Y_t^{(0)}.
\]

ReSpect-TTA keeps a recent residual history and summarizes it as a residual profile \(R_{buf}\). The adapter uses \(R_{buf}\) to condition the spectral correction applied to the next predictions.

### 3.2 Residual-Conditioned Complex Modulation

The active ReSpect-TTA candidate is `adapter_context_plus_complex_modulation_high_gate_lr`. It combines the original Spectral-COSA context path with a direct complex modulation path. The direct path first computes the real FFT of the current prediction and residual reference:

\[
F_y = \mathrm{rFFT}(Y_t^{(0)}, n=L), \quad
F_r = \mathrm{rFFT}(w \odot R_{buf}, n=L),
\]

where \(w\) is a Hann window applied to the residual reference. The residual spectrum is then converted into per-frequency, per-variable features:

\[
\psi(F_r) = [\Re(F_r), \Im(F_r), |F_r|, \omega],
\]

where \(\omega\) is a normalized frequency coordinate. A zero-initialized trainable projection maps these features to a complex-valued modulation:

\[
M_\phi(F_r) = W_\phi \psi(F_r) + b_\phi.
\]

The spectral correction is

\[
Z = M_\phi(F_r) \odot F_y,
\quad
\Delta Y_t = \mathrm{irFFT}(Z, n=L).
\]

Zero initialization is important. At the start of adaptation, \(M_\phi(F_r)\) is zero, so the direct correction path is identity-safe: it does not perturb the frozen base prediction until online updates learn useful correction structure. This reduces the risk that the frequency path harms predictions before residual evidence accumulates.

### 3.3 Global Gate and Context Path

The direct correction is controlled by a global gate:

\[
\hat{Y}_t = Y_t^{(0)} + g_t \odot \Delta Y_t.
\]

For the main candidate, \(g_t\) is learned and receives a high learning-rate multiplier. Its features include a prediction summary, residual-spectrum energy, and low-frequency residual ratio. The diagnostic variants modify this gate policy. `constant_gate_010` fixes the correction strength at 0.10 for long-horizon diagnosis. `high_gate_lr_floor_005` and `high_gate_lr_floor_010` retain the learned gate but impose a minimum correction strength:

\[
g_t = \alpha + (1-\alpha)\sigma(g_\eta(s_t)).
\]

The current main candidate also keeps the Spectral-COSA context adapter active. Thus the final architecture has two information channels: a time-domain context adapter that receives spectral residual features, and a direct spectral correction path that applies a residual-conditioned complex operator to the prediction spectrum.

### 3.4 Experimental Design

The evaluation follows the existing 5.26-style protocol used in the project. The benchmark grid covers six forecasting backbones: DLinear, FreTS, iTransformer, MICN, OLS, and PatchTST. It covers eight datasets: ETTh1, ETTh2, ETTm1, ETTm2, exchange_rate, weather, electricity, and traffic. Prediction horizons are 96, 192, 336, and 720. For each valid averaged condition, results are computed over seeds 0, 1, and 2.

The theoretical grid has \(6 \times 8 \times 4 = 192\) model-dataset-horizon conditions. The current collected ReSpect-TTA main-candidate result contains 171 conditions, each with three seeds. Twenty-one conditions are absent because valid checkpoints or runs were not available in the collected result set: FreTS-traffic at all four horizons, iTransformer-traffic at all four horizons, OLS-traffic at all four horizons, OLS-electricity at horizon 720, PatchTST-electricity at all four horizons, and PatchTST-traffic at all four horizons. The main analysis therefore reports comparable results on the 171-condition collected grid rather than claiming a complete 192-condition grid.

The main metrics are test MSE and test MAE. Comparisons are made against four reference families where available: no-TTA baseline, fixed COSA, PAAS COSA, and COSA-Spectral with residual-profile real-imag spectral context, history size 8, and projected context dimension 8.

## 4. Results

### 4.1 Main Aggregated Performance

Table 1 summarizes the main candidate against each reference family.

**Table 1**  
**Aggregate performance of ReSpect-TTA main candidate**

| Reference | Comparable conditions | MSE wins | Mean MSE improvement | Median MSE improvement | Worst MSE improvement | Mean MAE improvement |
|---|---:|---:|---:|---:|---:|---:|
| No-TTA baseline | 171 | 161/171 | +14.37% | +11.13% | -3.48% | +6.81% |
| COSA-Spectral | 171 | 162/171 | +3.07% | +2.72% | -2.54% | +1.14% |
| COSA fixed | 171 | 168/171 | +8.52% | +6.37% | -1.67% | +3.22% |
| COSA PAAS | 171 | 164/171 | +10.34% | +5.85% | -0.84% | +4.55% |

The result is strongest against no-TTA and the original COSA variants. The comparison against COSA-Spectral is more conservative but still positive: ReSpect-TTA wins on 162 of 171 comparable conditions and improves average MSE by 3.07%. Since COSA-Spectral already uses residual FFT features as context, this comparison is the most relevant test of whether direct frequency-domain correction adds value beyond spectral feature augmentation.

### 4.2 Performance by Model, Horizon, and Dataset

Against COSA-Spectral, ReSpect-TTA improves every collected MICN condition and every collected iTransformer condition. The average MSE improvements by model are:

**Table 2**  
**ReSpect-TTA vs. COSA-Spectral by backbone**

| Backbone | Conditions | MSE wins | Mean MSE improvement |
|---|---:|---:|---:|
| DLinear | 32 | 30/32 | +2.70% |
| FreTS | 28 | 25/28 | +2.65% |
| MICN | 32 | 32/32 | +4.86% |
| OLS | 27 | 25/27 | +2.70% |
| PatchTST | 24 | 22/24 | +2.41% |
| iTransformer | 28 | 28/28 | +2.78% |

Horizon-wise results show that the gains are not limited to short horizons. ReSpect-TTA wins on 42/43 conditions at horizon 96, 43/43 at horizon 192, 42/43 at horizon 336, and 35/42 at horizon 720. The mean MSE improvements against COSA-Spectral are +3.05%, +3.30%, +3.25%, and +2.68%, respectively.

The dataset-level pattern is also informative. ReSpect-TTA wins all comparable ETTh1, exchange_rate, traffic, weather, and electricity conditions. The weaker cases are concentrated in ETTm1 and ETTm2. Against COSA-Spectral, ETTm1 has 20/24 wins with a mean MSE improvement of +1.54%, while ETTm2 has 20/24 wins with a mean MSE improvement of +2.07%. These two datasets contain most of the observed negative-transfer cases.

### 4.3 Negative-Transfer Cases

The worst COSA-Spectral comparisons are concentrated at horizon 720:

**Table 3**  
**Worst ReSpect-TTA comparisons against COSA-Spectral**

| Backbone | Dataset | Horizon | MSE improvement |
|---|---|---:|---:|
| OLS | ETTm2 | 720 | -2.54% |
| OLS | ETTm1 | 720 | -2.06% |
| FreTS | ETTm1 | 720 | -1.84% |
| DLinear | ETTm2 | 720 | -1.79% |
| DLinear | ETTm1 | 720 | -1.58% |
| PatchTST | ETTm1 | 720 | -1.05% |

These cases suggest that ReSpect-TTA's correction can become too aggressive or misaligned when the residual spectrum from recent history does not extrapolate cleanly to the long-horizon target. The issue is not a global failure of the method, because the same horizon also contains strong gains for MICN, exchange_rate, weather, electricity, and several ETTh conditions. The failure mode is localized enough to guide the next ablations.

### 4.4 Gate Diagnostics

The most important diagnostic result is the constant-gate experiment at horizon 720. `constant_gate_010` fixes the global gate at 0.10 while keeping the context adapter and complex modulation trainable. On the 42 collected pred720 conditions, it wins 42/42 against no-TTA and 40/42 against COSA-Spectral, with an average MSE improvement of +28.78% over no-TTA and +3.15% over COSA-Spectral. This is slightly higher than the main learned high-gate variant on the same long-horizon slice.

The learned gate floor diagnostic is less decisive. `high_gate_lr_floor_005` wins 35/42 against COSA-Spectral at pred720, with average MSE improvement of +2.74%. This is useful but not superior to the main high-gate candidate. The current result archive therefore treats `floor_005` as a diagnostic rather than a promoted main variant.

The MICN-specific diagnostics are notable. At pred720, MICN benefits consistently from ReSpect-TTA variants. `constant_gate_020` improves over COSA-Spectral on all eight MICN datasets, with gains ranging from +3.41% to +8.25% against COSA-Spectral. However, it does not consistently beat the learned high-gate variant. This suggests that MICN's strong ReSpect-TTA behavior is not simply caused by needing a larger fixed gate; model-specific adaptivity remains relevant.

## 5. Discussion

### 5.1 Interpretation

The results support the central design claim: residual spectra contain useful correction information that can be applied directly to the prediction spectrum during online TTA. Compared with COSA-Spectral, which supplies spectral residual features to a time-domain adapter, ReSpect-TTA makes the spectral path operational. The residual spectrum does not merely describe the context. It parameterizes a complex operator that changes the amplitude and phase of the forecast spectrum before reconstruction.

The gains are modest but meaningful against COSA-Spectral. A 3.07% average MSE improvement over a spectral context baseline is not a dramatic replacement of COSA-Spectral, but it is enough to justify ReSpect-TTA as an architectural extension. The larger improvements against no-TTA and original COSA confirm that the method is not only matching a weak baseline. It improves over multiple adaptation settings.

Frequency-domain literature helps explain why this works. FEDformer and FreTS both argue that frequency representations expose global temporal structure and compact signal energy (Yi et al., 2023; Zhou et al., 2022). ReSpect-TTA adapts that argument to residual correction. The base prediction spectrum \(F_y\) contains the model's current forecast structure; the residual spectrum \(F_r\) contains recent systematic error. Their interaction through \(M(F_r) \odot F_y\) is a natural way to learn corrections that are proportional to the forecast's own spectral content.

### 5.2 Why the Gate Matters

The global gate is not only a scaling detail. It is the main safety mechanism between a useful spectral correction and harmful overcorrection. The old residual-only ReSpect-TTA path already showed that directly injecting residual spectral content can be unstable. ReSpect-TTA avoids that failure by using residuals to modulate the prediction spectrum and by controlling the correction magnitude with a gate.

The constant-gate result raises an important question. If a fixed gate of 0.10 improves pred720 stability, then the learned gate may sometimes under-open, over-open, or adapt slowly for long horizons. However, the MICN diagnostics also show that a fixed gate is not universally better than the learned high-gate policy. The next design step should therefore not be to replace the learned gate with a constant gate. A better direction is to make the learned gate horizon-aware or dataset-state-aware while retaining adaptivity.

### 5.3 Contribution

This study makes three contributions at the current draft stage.

First, it reframes output-space TTA for forecasting as a frequency-native correction problem. Prior COSA-style adaptation corrects predictions after the backbone through lightweight time-domain modules. ReSpect-TTA keeps the same deployment advantages but changes the correction operator.

Second, it introduces a residual-conditioned complex modulation mechanism. Rather than injecting residual spectra directly, the method generates a complex modulator from residual history and applies it to the current prediction spectrum. This design can learn both amplitude and phase changes while preserving identity-safe initialization.

Third, it provides a multi-backbone empirical probe. The current result set spans 171 averaged conditions over six model families, eight datasets, and four horizons where valid runs are available. The method improves most comparisons against no-TTA, COSA, PAAS COSA, and COSA-Spectral, while also revealing a clear long-horizon failure pattern.

## 6. Limitations and Future Work

The current draft has several limitations.

First, the collected main grid is incomplete. The intended grid has 192 model-dataset-horizon conditions, but the current analysis covers 171 conditions. Missing conditions are concentrated in traffic for FreTS, iTransformer, OLS, and PatchTST, plus PatchTST-electricity and OLS-electricity at horizon 720. These gaps must be either filled or explicitly excluded in any final paper.

Second, the current evidence is mostly aggregate. The draft reports wins, average improvements, and diagnostic summaries, but does not yet include statistical tests, confidence intervals, or paired seed-level significance analysis. Because each condition has three seeds, a future revision should add paired tests or bootstrap confidence intervals for the main comparisons.

Third, the method needs sharper ablation. The current main candidate combines the Spectral-COSA context adapter with direct complex modulation. The next paper revision should separate the contribution of the context path, the complex modulation path, the high-gate learning-rate multiplier, the diagonal output projection, the frequency cutoff, and the gate policy.

Fourth, the negative-transfer cases require targeted explanation. The ETTm1/ETTm2 h720 failures may be caused by residual non-stationarity, inappropriate gate opening, spectral leakage, or mismatch between recent residual frequency and long-horizon future error. The present result set identifies the pattern but does not yet isolate the cause.

Future work should prioritize three experiments: a pred720 gate-policy sweep, a direct-path-only versus context-plus-direct ablation, and a low-frequency cutoff analysis for ETTm1/ETTm2. If the results remain stable, ReSpect-TTA can then be moved from "main candidate" to "final method."

## 7. Conclusion

ReSpect-TTA is a promising extension of online output-space TTA for time-series forecasting. It preserves the practical structure of COSA: frozen backbone, residual memory, lightweight online updates, and leakage-free use of delayed ground truth. Its main change is to move residual correction into the frequency domain, where recent residual spectra generate a complex modulator applied to the current prediction spectrum.

The current empirical evidence is positive but not final. On 171 collected averaged conditions, ReSpect-TTA improves over no-TTA, fixed COSA, PAAS COSA, and COSA-Spectral in most cases. Its strongest claim is not that it universally dominates all baselines, but that frequency-native residual-conditioned correction adds measurable value beyond spectral context features alone. The remaining long-horizon failures make the next research step clear: stabilize the gate policy and isolate the exact contribution of complex modulation.

## Data Availability Statement

The experiment summaries used in this draft are available in the local project archive under `result/06.05_scheme6_experiment_records/spectral_master_results/summary/`. The main files used are `scheme6_full_context_plus_complex_results.csv`, `scheme6_full_context_plus_complex_averaged.csv`, `scheme6_full_context_plus_complex_improvement.csv`, `scheme6_context_plus_constant010_pred720_improvement.csv`, and `scheme6_gate_floor005_pred720_improvement.csv`. Public release status of the code and raw logs should be confirmed before submission.

## Ethics Declaration

This study uses public benchmark time-series datasets and offline model evaluation logs. No human-subject data collection, intervention, or personally identifiable information is involved in the current analysis.

## Author Contributions

Using CRediT terminology, the current draft assumes the following roles: conceptualization, methodology, software, investigation, data curation, formal analysis, writing-original draft, and writing-review and editing. Specific author names should be assigned before submission.

## Conflict of Interest Statement

The author(s) declare no known competing interests. This statement should be revised if institutional, financial, or project-related conflicts exist.

## Funding Acknowledgment

No external funding is declared in the current draft. This statement should be revised if the project received course, laboratory, institutional, or grant support.

## AI Disclosure

This manuscript draft was prepared with assistance from AI-based research and writing tools. The author(s) are responsible for verifying the accuracy of all claims, citations, equations, experimental results, and interpretations before submission.

## References

Im, J., & Kwon, H.-Y. (2026). *COSA: Context-aware output-space adapter for test-time adaptation in time series forecasting*. International Conference on Learning Representations. https://openreview.net/pdf/4c0fc8945c9af3f17fe1e61c42c6f6d449c76944.pdf

Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., & Choo, J. (2022). *Reversible instance normalization for accurate time-series forecasting against distribution shift*. International Conference on Learning Representations. https://openreview.net/forum?id=cGDAkQo1C0p

Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., & Long, M. (2024). *iTransformer: Inverted transformers are effective for time series forecasting*. International Conference on Learning Representations. https://openreview.net/forum?id=JePfAI8fah

Liu, Y., Wu, H., Wang, J., & Long, M. (2022). *Non-stationary transformers: Exploring the stationarity in time series forecasting*. Advances in Neural Information Processing Systems, 35. https://proceedings.neurips.cc/paper_files/paper/2022/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html

Medeiros, H. R., Sharifi-Noghabi, H., Oliveira, G. L., & Irandoust, S. (2025). *Accurate parameter-efficient test-time adaptation for time series forecasting*. arXiv. https://doi.org/10.48550/arXiv.2506.23424

Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). *A time series is worth 64 words: Long-term forecasting with transformers*. International Conference on Learning Representations. https://doi.org/10.48550/arXiv.2211.14730

Wang, H., Peng, J., Huang, F., Wang, J., Chen, J., & Xiao, Y. (2023). *MICN: Multi-scale local and global context modeling for long-term series forecasting*. International Conference on Learning Representations. https://openreview.net/forum?id=zt53IDUR1U

Yi, K., Zhang, Q., Fan, W., Wang, S., Wang, P., He, H., Lian, D., An, N., Cao, L., & Niu, Z. (2023). *Frequency-domain MLPs are more effective learners in time series forecasting*. Advances in Neural Information Processing Systems, 36. https://proceedings.neurips.cc/paper/2023/hash/f1d16af76939f476b5f040fd1398c0a3-Abstract.html

Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). *Are transformers effective for time series forecasting?* Proceedings of the AAAI Conference on Artificial Intelligence. https://doi.org/10.48550/arXiv.2205.13504

Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L., & Jin, R. (2022). *FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting*. International Conference on Machine Learning. https://doi.org/10.48550/arXiv.2201.12740
