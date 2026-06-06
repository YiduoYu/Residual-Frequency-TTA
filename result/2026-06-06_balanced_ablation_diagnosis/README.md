# 2026-06-06 Balanced Spectral-PEFT Ablation Diagnosis

Branch: `c/2026-05-29-balanced-spectral-peft`

Remote GPU workspace: `/home/normal/3830_FinalProject_COSA_latest_plan2_1/Spectral-master`

Hard subset scope: models `DLinear FreTS iTransformer`, datasets `weather ETTm1`, prediction lengths `336 720`. Most final comparisons use seeds `0 1 2`; entries marked `seed0` are diagnostic-only runs.

## Branch Objective

This branch studies Balanced Spectral-PEFT for time-series test-time adaptation. The research goal is to move old Spectral's COSA-based spectral context toward a more standalone, lightweight spectral residual adapter.

The goal is not simple hyperparameter tuning or immediately beating old Spectral. The goal is to understand whether a COSA-independent spectral adapter/backend can preserve useful residual spectral information without depending on the original `SimpleOutputAdapter`-centered COSA framework.

## Main Conclusion

Balanced Spectral-PEFT improves over the weaker baseline/COSA-style settings, but it does not beat old Spectral on the hard subset.

The strongest historical reference remains:

- old Spectral mean MSE: `0.348721`
- best Balanced Spectral-PEFT no-LoRA mean MSE: `0.382168`

All later diagnostic variants failed to stably improve over the no-LoRA balanced baseline. Several variants showed small seed0 improvements, but those did not survive seed012 validation.

## Ablation Summary

| Variant | Scope | Mean MSE | Diagnosis |
| --- | --- | ---: | --- |
| old Spectral | hard subset seed012 | 0.348721 | Best overall reference. |
| default balanced | hard subset seed012 | 0.395532 | Balanced default is worse than old Spectral. |
| no_lora | hard subset seed012 | 0.382168 | Best Balanced Spectral-PEFT baseline. |
| legacy_raw full8 no_lora | hard subset seed012 | 0.382832 | Raw context full8 nearly matches no_lora, but does not improve it. |
| legacy_raw full19 | hard subset seed012 | 0.396917 | Expanding raw context hurts. |
| legacy_raw full19 adapter_only | hard subset seed012 | 0.406067 | Adapter-only is worse than mixture. |
| SRDC alpha=0.25 keepDC | hard subset seed0 | 0.421128 | Direct residual correction gives only weak bias signal. |
| SRDC DC-only | hard subset seed0 | 0.419783 | DC-only is better than full SRDC, so signal is mostly bias/DC. |
| POST_BIAS alpha=0.50 | hard subset seed012 | 0.385031 | Seed0 improvement does not survive seed012. |
| identity + cap=0.10 | hard subset seed0 | 0.413663 | Identity branch plus cap over-constrains adaptation. |
| cap-only | hard subset seed0 | 0.410040 | Backend delta cap alone damages learned no_lora backend. |
| identity-only | hard subset seed0 | 0.403380 | Identity in softmax mixture dilutes useful backend adaptation. |
| film + adapter | hard subset seed012 | 0.385869 | Best pruning variant, but still worse than no_lora. |

## Diagnosis

LoRA is negative but not the root cause. Disabling LoRA gives the best Balanced Spectral-PEFT result, but the no-LoRA path still remains far behind old Spectral.

Expanding raw spectral context is unstable. The legacy_raw full8 diagnostic is close to no_lora, but full19 degrades substantially. This suggests that simply restoring more old-Spectral-style raw FFT features does not fix the current backend.

Direct spectral residual shape correction is ineffective. SRDC shows that keeping DC/bias helps slightly, while centered no-DC frequency-shape correction loses the benefit. The useful signal is residual bias/DC, not non-DC spectral shape in this direct correction form.

Post-bias calibration does not survive seed012. It improved seed0 slightly, but seed012 mean MSE remains worse than historical no_lora.

Identity and backend delta caps over-constrain learned adaptation. The split diagnostics showed both mechanisms are harmful: cap-only degraded to `0.410040`, and identity-only degraded to `0.403380`.

Film+adapter pruning is the only pruning variant with a useful seed0 signal, but seed012 validation gives `0.385869`, still worse than no_lora `0.382168`. IA3 appears low-value or negative in this branch, but removing it is not enough to create a new main method.

Overall, the current multi-backend PEFT mixture is over-composed. It is less effective than old Spectral's direct residual spectral context path, even after removing LoRA, pruning backends, adding residual bias calibration, or trying conservative gates.

## Final Recommendation

Do not continue this branch as the main method.

Keep `no_lora` as the best Balanced Spectral-PEFT baseline for this branch.

Start a new branch for a standalone direct spectral adapter that preserves old Spectral's direct residual-context path while removing unnecessary multi-backend PEFT mixture. The next design should avoid softmax mixing among many adaptation backends and should focus on a direct, stable residual-context correction path.

## Result Paths

Checkpoints and datasets are not stored in GitHub. All experiments reused existing checkpoints on the remote GPU workspace:

`/home/normal/3830_FinalProject_COSA_latest_plan2_1/Spectral-master/checkpoints`

The main remote result root is:

`/home/normal/3830_FinalProject_COSA_latest_plan2_1/Spectral-master/results/summary`

Available exact result paths:

| Variant | Path |
| --- | --- |
| old Spectral reference table | `result/5.26_paper8_experiment/final_tables/paper8_full_comparison_averaged.csv` |
| default/no_lora early ablation summaries | `result/2026-06-05_balanced_ablation_diagnosis/` |
| legacy_raw full19 | `/home/normal/3830_FinalProject_COSA_latest_plan2_1/Spectral-master/results/summary/COSA_BALANCED/balanced_spectral_peft_legacy_raw_real_imag_win3_softmax_gate_frozen_encoder_k8_cond19_backendsfilm_ia3_adapter_lora_off/` |
| legacy_raw full19 adapter_only | `/home/normal/3830_FinalProject_COSA_latest_plan2_1/Spectral-master/results/summary/COSA_BALANCED/balanced_spectral_peft_legacy_raw_real_imag_win3_softmax_gate_frozen_encoder_k8_cond19_backendsadapter_lora_off/` |
| SRDC alpha=0.25 keepDC | `/home/normal/3830_FinalProject_COSA_latest_plan2_1/Spectral-master/results/summary/COSA_BALANCED/balanced_spectral_peft_multi_window_real_imag_win3_softmax_gate_frozen_encoder_k8_cond8_backendssrdc_lora_on_srdc_alpha0.25_k8_keepdcTrue_centerFalse_maxdelta0.10/` |
| SRDC DC-only | `/home/normal/3830_FinalProject_COSA_latest_plan2_1/Spectral-master/results/summary/COSA_BALANCED/balanced_spectral_peft_multi_window_real_imag_win3_softmax_gate_frozen_encoder_k8_cond8_backendssrdc_lora_on_srdc_alpha0.25_k0_keepdcTrue_centerFalse_maxdelta0.10/` |
| POST_BIAS alpha=0.50 seed012 | `/home/normal/3830_FinalProject_COSA_latest_plan2_1/Spectral-master/results/summary/COSA_BALANCED/balanced_spectral_peft_multi_window_real_imag_win3_softmax_gate_frozen_encoder_k8_cond8_backendsfilm_ia3_adapter_lora_off_postbias_on_pbalpha0.50_pbbeta0.90_pbhist8_pbmaxratio0.05/` |
| identity + cap=0.10 seed0 | `/home/normal/3830_FinalProject_COSA_latest_plan2_1/Spectral-master/results/summary/COSA_BALANCED/bsp_multi_window_win3_frozen_encoder_k8_cond8_backendsidentity_film_ia3_adapter_identity_on_identityinitTrue_ilogit2.0_ologit-2.0_backendcap_on_maxdelta0.10_lora_off_postbias_off_pba0.10_pbb0.90_pbh8_pbm0.05/` |
| cap-only seed0 | `/home/normal/3830_FinalProject_COSA_latest_plan2_1/Spectral-master/results/summary/COSA_BALANCED/bsp_multi_window_win3_frozen_encoder_k8_cond8_backendsfilm_ia3_adapter_identity_off_identityinitTrue_ilogit2.0_ologit-2.0_backendcap_on_maxdelta0.10_lora_off_postbias_off_pba0.10_pbb0.90_pbh8_pbm0.05/` |
| identity-only seed0 | `/home/normal/3830_FinalProject_COSA_latest_plan2_1/Spectral-master/results/summary/COSA_BALANCED/bsp_multi_window_win3_frozen_encoder_k8_cond8_backendsidentity_film_ia3_adapter_identity_on_identityinitTrue_ilogit2.0_ologit-2.0_backendcap_off_maxdelta0.05_lora_off_postbias_off_pba0.10_pbb0.90_pbh8_pbm0.05/` |
| film + adapter seed012 | `/home/normal/3830_FinalProject_COSA_latest_plan2_1/Spectral-master/results/summary/COSA_BALANCED/bsp_multi_window_win3_frozen_encoder_k8_cond8_backendsfilm_adapter_identity_off_identityinitTrue_ilogit2.0_ologit-2.0_backendcap_off_maxdelta0.05_lora_off_postbias_off_pba0.10_pbb0.90_pbh8_pbm0.05/` |

Large result logs, checkpoints, datasets, and CSV outputs are intentionally not copied into GitHub.
