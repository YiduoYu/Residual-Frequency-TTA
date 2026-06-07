# Peer Review Package: ReSpect-TTA Paper Draft

**Manuscript:** ReSpect-TTA: Residual-Conditioned Complex Frequency Modulation for Online Test-Time Adaptation in Time-Series Forecasting  
**Reviewed file:** `paper/respect_tta_paper_draft.md`  
**Review date:** 2026-06-05  
**Review round:** Round 1 pre-submission review  
**Mode:** Full academic-paper-reviewer panel  

## Reviewer Configuration Card

| Reviewer | Role | Expertise | Review Focus |
|---|---|---|---|
| EIC | Machine learning conference area chair | Time-series forecasting, TTA, empirical ML paper standards | Contribution strength, venue fit, maturity, decision |
| R1 | Methodology/statistics reviewer | Benchmark design, ablation, statistical reporting | Experimental validity, missing grid, significance, reproducibility |
| R2 | Domain reviewer | Long-term forecasting, spectral forecasting, online adaptation | Literature positioning, novelty, relation to COSA/Spectral/FreTS/FEDformer |
| R3 | Perspective reviewer | Deployment-oriented forecasting and applied ML | Practical value, runtime, risk, usability of the adapter |
| DA | Devil's Advocate | Argument stress testing and counter-evidence | Strongest objections, overclaiming, alternative explanations |

## Editorial Decision

**Decision: Major Revision**

The manuscript has a plausible and interesting contribution, but it is not yet ready for submission as a full empirical ML paper. The core idea is clear: ReSpect-TTA moves COSA-style residual correction from a time-domain context adapter toward a residual-conditioned complex frequency operator. The early result set is promising, especially the consistent gains over no-TTA and the positive average improvement over COSA-Spectral. However, the current paper is closer to a well-organized project report than a defensible conference submission. The main blockers are incomplete experimental coverage, insufficient ablation isolation, absence of statistical uncertainty analysis, and a contribution claim that depends on a combined variant whose components are not yet separated.

The decision is Major Revision rather than Reject because the method is technically coherent, the reported comparisons are transparent about the 171/192 grid limitation, and the negative-transfer analysis already points toward actionable follow-up experiments. The decision is not Minor Revision because the missing experiments and ablations affect the central causal claim of the paper.

## Reviewer Summary

| Reviewer | Recommendation | Confidence |
|---|---|---:|
| EIC | Major Revision | 4/5 |
| R1 Methodology | Major Revision | 5/5 |
| R2 Domain | Major Revision | 4/5 |
| R3 Perspective | Minor-to-Major Revision | 3/5 |
| Devil's Advocate | Major Revision, with one critical argument issue | 4/5 |

## EIC Review

### Summary Assessment

The manuscript presents ReSpect-TTA, an online TTA method that uses recent residual spectra to generate a complex frequency-domain modulation of the current prediction spectrum. The paper is readable, technically motivated, and conservative in several important places, especially where it states that the current result is a "main candidate" rather than a final method. The reported gains over no-TTA, COSA, PAAS COSA, and COSA-Spectral suggest the direction is worth developing. The paper's main weakness is maturity. The central empirical claim is based on a combined variant, `adapter_context_plus_complex_modulation_high_gate_lr`, but the paper does not yet isolate the incremental value of the direct complex modulation path from the retained Spectral-COSA context path or the high-gate learning-rate policy. For a conference submission, this is a major gap because readers need to know what component caused the improvement.

### Strengths

**S1: Clear mechanism statement.** The Introduction and Method sections consistently describe the correction as \(M(F_r) \odot F_y\), which helps distinguish ReSpect-TTA from residual injection and from COSA-Spectral.

**S2: Conservative result framing.** The paper states that the current grid has 171 collected conditions rather than the full 192-condition grid. This transparency reduces the risk of overstating the evidence.

**S3: Useful negative-transfer discussion.** Table 3 identifies the worst cases, mostly ETTm1/ETTm2 at horizon 720. This makes the paper more credible than a purely promotional report.

### Weaknesses

**W1: The contribution is not isolated.**  
**Problem:** The main variant combines Spectral-COSA context, complex modulation, high-gate LR, diagonal projection, and learned global gating.  
**Why it matters:** Without ablations, the paper cannot show that "residual-conditioned complex modulation" is the actual cause of the improvement over COSA-Spectral.  
**Suggestion:** Add a core ablation table: Spectral-COSA only, context-plus-direct without complex modulation, complex modulation only, context-plus-complex with normal gate LR, context-plus-complex high-gate LR, and constant/floor gate variants.  
**Severity:** Critical/Major.

**W2: Missing grid weakens venue fit.**  
**Problem:** The draft reports 171/192 conditions and lists missing combinations, but does not explain whether missingness is random, checkpoint-related, model failure, or protocol failure.  
**Why it matters:** Missing conditions are concentrated in traffic/electricity for specific backbones, which may bias aggregate conclusions.  
**Suggestion:** Add a missing-run audit table with reason codes and decide whether to fill the missing conditions or define a principled comparable subset.  
**Severity:** Major.

**W3: The paper lacks a figure.**  
**Problem:** The architecture is described in equations but no figure is included.  
**Why it matters:** A frequency-domain adapter paper needs a visual pipeline to make the COSA-Spectral vs ReSpect-TTA distinction immediately legible.  
**Suggestion:** Include one architecture figure and one aggregate result figure.  
**Severity:** Minor/Major depending on target venue.

### EIC Questions

1. What is the minimal ReSpect-TTA component that still beats COSA-Spectral?
2. Are the 21 missing grid points caused by missing checkpoints, runtime failures, or incompatible model-dataset combinations?
3. Is the target venue a course paper, workshop paper, or main-track ML conference? The evidence threshold differs substantially.

### EIC Scores

| Dimension | Score | Descriptor | Notes |
|---|---:|---|---|
| Originality | 76 | Strong | Clear adapter-level idea, but related to existing spectral/TTA trends. |
| Methodological Rigor | 58 | Weak/Adequate | Promising protocol, insufficient ablation/statistics. |
| Evidence Sufficiency | 60 | Adequate | Large-ish result set, but incomplete and mostly aggregate. |
| Argument Coherence | 78 | Strong | Well structured and appropriately cautious. |
| Writing Quality | 82 | Strong | Clear, readable, low fluff. |

## Reviewer 1: Methodology and Statistical Review

### Summary Assessment

The empirical section is organized, but the statistical reporting is not yet strong enough for a quantitative ML paper. The paper reports means, wins, and percentage improvements, which are useful first summaries. It does not report variance across seeds for the main comparison tables, confidence intervals, paired statistical tests, or robustness to missing grid conditions. The three-seed design is acceptable as an initial benchmark, but the paper needs uncertainty estimates because many improvements over COSA-Spectral are small, especially the central +3.07% mean MSE improvement. Without uncertainty analysis, it is unclear how many gains are stable across seeds versus driven by particular conditions.

### Strengths

**S1: Transparent denominator.** The paper reports 171 comparable conditions and does not hide the missing 21 conditions.

**S2: Multiple baselines.** The comparison includes no-TTA, fixed COSA, PAAS COSA, and COSA-Spectral. This is appropriate because COSA-Spectral is the strongest conceptual baseline.

**S3: Both MSE and MAE are mentioned.** Reporting both metrics is useful because a frequency correction could reduce one error type while worsening another.

### Weaknesses

**W1: No seed-level uncertainty or significance.**  
**Problem:** Tables report aggregate percentage improvements but not confidence intervals, standard deviations, or paired tests.  
**Why it matters:** The key COSA-Spectral improvement is modest. Without uncertainty, the paper cannot distinguish stable improvement from noise.  
**Suggestion:** Add per-condition seed-level paired deltas, bootstrap CIs over model-dataset-horizon conditions, and a sign test or Wilcoxon signed-rank test for ReSpect-TTA vs COSA-Spectral.  
**Severity:** Critical/Major.

**W2: Aggregate averaging may overweight easy conditions.**  
**Problem:** The paper averages percentage improvements across heterogeneous datasets, horizons, and models.  
**Why it matters:** A simple mean can be dominated by datasets with large baseline errors or by uneven missingness.  
**Suggestion:** Report macro averages by model, dataset, and horizon, plus a condition-level median. Consider a hierarchical summary: per-dataset average first, then overall average.  
**Severity:** Major.

**W3: Missing conditions are not analyzed for bias.**  
**Problem:** The missing grid is concentrated in traffic and electricity for several backbones.  
**Why it matters:** Since traffic/electricity are high-dimensional datasets where TTA behavior may differ, removing them can inflate or deflate the perceived method performance.  
**Suggestion:** Add a sensitivity analysis excluding traffic/electricity from all methods or using only the common complete subset.  
**Severity:** Major.

**W4: Runtime reporting is absent from the paper.**  
**Problem:** TTA methods impose online cost, but the draft does not report adaptation time, throughput, or parameter counts.  
**Why it matters:** The claimed practical value of lightweight adaptation depends on computational overhead.  
**Suggestion:** Add parameter count and throughput table for COSA, COSA-Spectral, and ReSpect-TTA.  
**Severity:** Major.

### Methodology Questions

1. Are the improvements computed from seed-matched comparisons for every baseline, or are some baselines missing seeds?
2. Does the online adaptation update after every batch with the same protocol across all baselines?
3. What is the parameter count of the direct complex modulation path as a function of horizon and variable count?
4. Are standard deviations in the raw CSV available for the main ReSpect-TTA and reference rows?

### Methodology Scores

| Dimension | Score | Descriptor | Notes |
|---|---:|---|---|
| Originality | 72 | Adequate/Strong | Method is interesting but not yet isolated. |
| Methodological Rigor | 50 | Weak | Missing tests, ablations, and missingness audit. |
| Evidence Sufficiency | 52 | Weak/Adequate | Enough for internal report, not for submission. |
| Argument Coherence | 74 | Strong | Claims are mostly aligned with evidence. |
| Writing Quality | 78 | Strong | Clear methods prose. |

## Reviewer 2: Domain Review

### Summary Assessment

The paper is well positioned around COSA, COSA-Spectral, frequency-domain forecasting, and non-stationarity. It correctly treats ReSpect-TTA as an adapter-level contribution rather than a new base forecaster. The most valuable conceptual distinction is between spectral features as context and spectral operators as correction. That said, the related work is too thin for a paper claiming a new spectral TTA mechanism. It cites appropriate anchor papers, but it does not yet develop a tight taxonomy of output-space TTA, spectral forecasting, and residual/error modeling. It also does not discuss whether similar residual correction ideas exist in classical forecasting or signal processing.

### Strengths

**S1: Correct positioning against COSA-Spectral.** The paper identifies COSA-Spectral as the most relevant baseline rather than relying only on no-TTA.

**S2: Good distinction between adapter and backbone.** The paper states that ReSpect-TTA is not a new base forecaster, which prevents scope confusion.

**S3: Frequency-domain motivation is plausible.** The use of FEDformer and FreTS as background is appropriate because both support the value of frequency-domain representations.

### Weaknesses

**W1: Literature review lacks residual/error-correction context.**  
**Problem:** The paper jumps from COSA and frequency forecasting to ReSpect-TTA without reviewing residual correction, error compensation, or hybrid residual modeling traditions.  
**Why it matters:** Reviewers may see the idea as obvious if residual-error correction literature is not acknowledged.  
**Suggestion:** Add a subsection on residual correction/error compensation in forecasting, including classical residual modeling and recent neural residual correction if applicable.  
**Severity:** Major.

**W2: COSA-Spectral is described but not formally defined.**  
**Problem:** The paper states what COSA-Spectral does, but does not provide enough implementation detail for readers to understand the baseline.  
**Why it matters:** The contribution is relative to COSA-Spectral, so the baseline must be technically transparent.  
**Suggestion:** Add a short baseline definition with context feature construction, real/imag representation, history size, projection dimension, and disabled direct correction.  
**Severity:** Major.

**W3: Novelty claim needs sharper wording.**  
**Problem:** The current contribution statements are plausible but broad.  
**Why it matters:** "Frequency-native residual-conditioned correction" may sound novel, but the exact novelty must be stated in relation to existing spectral forecasting and TTA papers.  
**Suggestion:** State the novelty as: "a post-hoc online output-space adapter in which recent residual spectra parameterize a complex operator applied to the frozen prediction spectrum."  
**Severity:** Minor/Major.

### Domain Questions

1. Are there existing spectral residual correction or frequency compensation methods that should be compared conceptually?
2. How does ReSpect-TTA relate to frequency compensation layers such as FCP-style models, if relevant?
3. Why is complex modulation preferable to separate real-valued amplitude gating plus phase correction?

### Domain Scores

| Dimension | Score | Descriptor | Notes |
|---|---:|---|---|
| Originality | 74 | Strong | Good idea, needs tighter novelty boundary. |
| Methodological Rigor | 58 | Weak/Adequate | Domain baseline needs more formal description. |
| Evidence Sufficiency | 62 | Adequate | Enough to motivate, not enough to finalize. |
| Argument Coherence | 76 | Strong | Clear conceptual chain. |
| Writing Quality | 80 | Strong | Readable and disciplined. |

## Reviewer 3: Practical and Cross-Disciplinary Review

### Summary Assessment

The manuscript has practical appeal because it targets a realistic deployment problem: forecasters drift after deployment, and a lightweight online adapter is easier to use than retraining full backbones. The paper also correctly avoids claiming that ReSpect-TTA replaces all base models. The missing practical analysis is cost and operational behavior. TTA methods must be evaluated not only by MSE/MAE but also by online latency, memory, update frequency, and failure modes when residuals are noisy or delayed. The current paper would be more convincing if it included deployment-oriented diagnostics.

### Strengths

**S1: Deployment-compatible framing.** Frozen backbones and adapter-only updates are practical for systems where full retraining is expensive.

**S2: Gate mechanism addresses safety.** The paper recognizes that unbounded spectral correction can harm forecasts and treats the global gate as a safety mechanism.

**S3: Negative-transfer section is practically useful.** Identifying ETTm1/ETTm2 h720 failures helps future users know where to be cautious.

### Weaknesses

**W1: No online cost analysis.**  
**Problem:** The paper does not report overhead relative to COSA-Spectral.  
**Why it matters:** A method that improves MSE by roughly 3% over COSA-Spectral may not be worth deploying if it substantially increases latency.  
**Suggestion:** Report throughput, adaptation time, memory footprint, and trainable parameter count.  
**Severity:** Major.

**W2: No operational failure scenario.**  
**Problem:** The paper assumes ground truth arrives reliably and residuals are usable.  
**Why it matters:** Online adaptation can fail when labels are delayed, sparse, noisy, or affected by regime shifts.  
**Suggestion:** Add a short subsection on deployment assumptions and stress tests.  
**Severity:** Minor/Major.

**W3: The gate diagnostics are useful but under-visualized.**  
**Problem:** The paper reports gate variants in prose only.  
**Why it matters:** Gate behavior is the most practical safety issue.  
**Suggestion:** Add a figure or table comparing learned gate, constant gate, and floor gate on pred720 by dataset.  
**Severity:** Minor.

### Practical Questions

1. What is the latency overhead of FFT/iFFT plus complex modulation relative to COSA-Spectral?
2. How many batches are required before the residual buffer becomes useful?
3. Does ReSpect-TTA degrade gracefully when the residual buffer is empty or stale?

### Practical Scores

| Dimension | Score | Descriptor | Notes |
|---|---:|---|---|
| Originality | 70 | Adequate/Strong | Practical extension, not a full deployment study. |
| Methodological Rigor | 56 | Weak/Adequate | Missing runtime and stress tests. |
| Evidence Sufficiency | 58 | Weak/Adequate | Accuracy evidence only. |
| Argument Coherence | 78 | Strong | Practical story is clear. |
| Writing Quality | 80 | Strong | Clear and restrained. |

## Devil's Advocate Review

### Strongest Counter-Argument

The strongest objection is that the paper may be attributing improvements to "residual-conditioned complex frequency modulation" without proving that this component is responsible. The main variant is a bundled system: it retains the Spectral-COSA context adapter, adds the complex modulation path, changes the global-gate learning-rate behavior, keeps diagonal projection, and evaluates a partially missing grid. The observed +3.07% average MSE improvement over COSA-Spectral could therefore come from high-gate LR, extra trainable capacity, implementation details, different missing-condition composition, or favorable averaging rather than from complex modulation. The paper argues that residual spectra parameterize a useful complex operator, but the current evidence does not eliminate simpler explanations. A reviewer could reasonably say: "You added parameters and tuning on top of COSA-Spectral, then compared aggregate improvements without uncertainty tests. Show me the isolated causal effect." Until that is resolved, the central contribution remains plausible but not established.

### Issue List

**CRITICAL: Causal attribution is not demonstrated.**  
**Location:** Method, Results, Discussion.  
**Problem:** The paper claims value for complex modulation but does not isolate it from context path and gate policy.  
**Fix:** Add component ablations and a direct comparison to a capacity-matched real-valued spectral adapter.

**MAJOR: Missingness may bias the result.**  
**Location:** Section 3.4 and Results.  
**Problem:** 21/192 conditions are missing, concentrated in certain backbones/datasets.  
**Fix:** Add missingness audit and complete-common-subset analysis.

**MAJOR: Small gains lack uncertainty.**  
**Location:** Table 1 and Section 4.  
**Problem:** The key +3.07% gain over COSA-Spectral is not accompanied by confidence intervals or paired tests.  
**Fix:** Add bootstrap CI, sign test, and per-seed paired deltas.

**MAJOR: Baseline tuning fairness is unclear.**  
**Location:** Experimental Design.  
**Problem:** ReSpect-TTA has gate diagnostics and high-gate LR, but the baseline tuning budget for COSA-Spectral is not discussed.  
**Fix:** State whether COSA-Spectral was equally tuned or inherited from prior best settings.

**MINOR: The title may overstate readiness.**  
**Location:** Title.  
**Problem:** Title reads like a finished method, while the paper says "current main candidate."  
**Fix:** For internal/course version, acceptable. For submission, make method final or title it as an empirical study/probe.

### Ignored Alternative Explanations

1. Gains may come from extra trainable parameters rather than spectral structure.
2. Gains may come from high-gate LR or fixed gate magnitude rather than complex modulation.
3. Gains may come from favorable missing-grid composition.
4. Gains over no-TTA may mostly replicate COSA's residual-adapter benefit, not ReSpect-TTA's new contribution.
5. ETTm1/ETTm2 failures may indicate the residual spectrum is not a reliable predictor of future long-horizon errors under certain regimes.

### Devil's Advocate Verdict

Major Revision. The paper has a viable idea, but the causal claim is not yet sufficiently defended.

## Consensus Analysis

### Points of Agreement

**CONSENSUS-5:** All reviewers agree the core idea is promising and technically coherent.

**CONSENSUS-5:** All reviewers agree the main evidence gap is ablation: the paper must isolate complex modulation from context adapter, gate LR, and gate policy.

**CONSENSUS-4:** EIC, R1, R2, and DA agree that missing-grid handling is a major issue for submission.

**CONSENSUS-4:** EIC, R1, R3, and DA agree that gate diagnostics should become a formal experiment section rather than mostly prose.

**CONSENSUS-3:** R1, R3, and DA agree runtime/parameter overhead is necessary because the paper is about online TTA.

### Disagreements

**Disagreement 1: Is this paper rejectable now?**  
R1 and DA see a critical causal-attribution gap, while EIC and R2 view it as fixable. Editor resolution: Major Revision, not Reject. The draft is transparent and has enough positive evidence to justify revision.

**Disagreement 2: How severe is the practical-runtime gap?**  
R3 treats runtime as central; R2 treats it as secondary to novelty and ablation. Editor resolution: runtime is a required major revision because online TTA methods must justify overhead.

## Required Revisions

| # | Revision Item | Source | Severity | Section | Estimated Effort |
|---|---|---|---|---:|
| R1 | Add component ablations isolating complex modulation | EIC, R1, R2, DA | Critical | Method/Results | 3-7 days, plus run time |
| R2 | Add uncertainty analysis for main comparisons | R1, DA | Major | Results | 1-2 days |
| R3 | Audit missing 21 grid conditions and add sensitivity analysis | EIC, R1, DA | Major | Experiments/Results | 1-2 days |
| R4 | Add runtime, parameter count, and online overhead reporting | R1, R3 | Major | Experiments/Results | 1-2 days |
| R5 | Formalize COSA-Spectral baseline and tuning fairness | R2, DA | Major | Related Work/Experiments | 1 day |
| R6 | Expand related work on residual/error correction | R2 | Major | Related Work | 1-2 days |

### Required Item Details

**R1: Component Ablation**
- **Problem:** The current main variant is bundled.
- **Requirement:** At minimum compare COSA-Spectral, complex modulation only, context-plus-direct without high-gate LR, context-plus-complex with normal gate LR, context-plus-complex high-gate LR, constant gate 0.10, and floor gate 0.05/0.10 on a common subset.
- **Acceptance criteria:** A table clearly shows which component accounts for the improvement over COSA-Spectral.

**R2: Statistical Uncertainty**
- **Problem:** Mean percentage improvements are not enough for the key claim.
- **Requirement:** Add seed-level paired deltas, bootstrap 95% CI over conditions, and sign/Wilcoxon tests for ReSpect-TTA vs COSA-Spectral.
- **Acceptance criteria:** The paper can state whether the +3.07% improvement is statistically and practically stable.

**R3: Missingness Audit**
- **Problem:** 21/192 missing conditions may bias aggregate results.
- **Requirement:** Add a table with missing combinations and reason codes. Add a sensitivity result on the complete comparable subset.
- **Acceptance criteria:** A reader can judge whether missingness changes the main conclusion.

**R4: Runtime and Parameter Overhead**
- **Problem:** Online adaptation cost is not reported.
- **Requirement:** Report trainable parameter count, mean adaptation time, throughput, and overhead versus COSA-Spectral.
- **Acceptance criteria:** The paper can defend ReSpect-TTA as a practical online adapter.

**R5: Baseline Formalization**
- **Problem:** COSA-Spectral is the key baseline but is underspecified.
- **Requirement:** Define its residual FFT context, representation, history size, projection dimension, direct-correction disabled state, and tuning budget.
- **Acceptance criteria:** The experimental comparison is reproducible and fair.

**R6: Related Work Expansion**
- **Problem:** The draft omits residual/error-correction literature.
- **Requirement:** Add a concise subsection covering residual correction/error compensation in forecasting and position ReSpect-TTA against it.
- **Acceptance criteria:** Novelty claim is bounded and defensible.

## Suggested Revisions

| # | Suggestion | Source | Priority | Expected Improvement |
|---|---|---|---|---|
| S1 | Add architecture figure contrasting COSA, COSA-Spectral, and ReSpect-TTA | EIC, R3 | P2 | Improves readability |
| S2 | Add gate diagnostic figure for pred720 | R3, DA | P2 | Clarifies stability story |
| S3 | Add table of worst/best cases by dataset and horizon | R1 | P2 | Strengthens error analysis |
| S4 | Tighten novelty sentence in Introduction | R2 | P2 | Reduces overclaim risk |
| S5 | Add exact implementation details for \(M(F_r)\) dimensions and initialization | R1/R2 | P2 | Improves reproducibility |
| S6 | Convert references to fully consistent APA or target conference format | EIC | P3 | Formatting readiness |

## Revision Roadmap

### Priority 1: Make the Central Claim Defensible

- [ ] Run or collect component ablations on a common subset.
- [ ] Separate complex modulation from context adapter and gate LR effects.
- [ ] Add a capacity-matched or parameter-count-controlled baseline if feasible.
- [ ] Rewrite contribution claims after ablation results.

### Priority 2: Strengthen Statistical and Experimental Reporting

- [ ] Add seed-level paired deltas and confidence intervals.
- [ ] Add sign/Wilcoxon test for ReSpect-TTA vs COSA-Spectral.
- [ ] Add missing-run audit with reason codes.
- [ ] Add complete-subset sensitivity analysis.
- [ ] Add runtime and parameter-count table.

### Priority 3: Improve Paper Framing

- [ ] Add a formal COSA-Spectral baseline subsection.
- [ ] Expand related work on residual correction and spectral compensation.
- [ ] Add architecture and gate diagnostic figures.
- [ ] Tighten abstract and conclusion after new experiments.

### Total Estimated Effort

**Major revision:** 1-3 weeks if existing logs already contain most ablations; 3-6 weeks if new full-grid runs are needed.

## Final Editorial Note

This is a strong internal draft and a promising research direction. For a course project report, it may already be usable after adding figures and a small ablation table. For a workshop or conference submission, the central empirical claim needs stronger evidence. The most important revision is not prose polishing. It is proving that complex residual-conditioned spectral modulation, rather than a bundled combination of extra capacity and gate tuning, is responsible for the observed gain.
