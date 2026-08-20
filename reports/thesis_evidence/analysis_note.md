# Thesis evidence audit

## Headline interpretation rule

The headline robustness metric is the per-seed minimum of FGSM and PGD conditional robust accuracy at each epsilon, followed by a cumulative minimum over increasing epsilon. The raw PGD values remain in `attack_curve_seed.csv`. A high-budget rebound is treated as an attack-optimisation warning, not recovered robustness.

## Descriptive findings

- **Chest X-ray:** parameter count versus robustness AUC has Spearman rho=0.20; the highest mean audited AUC is observed for ResNet-152 (0.050).
- **Malaria:** parameter count versus robustness AUC has Spearman rho=0.00; the highest mean audited AUC is observed for ResNet-152 (0.274).
- **OCT:** parameter count versus robustness AUC has Spearman rho=0.10; the highest mean audited AUC is observed for ResNet-152 (0.068).

- The strict audit adjusted 10 seed-budget observations across 8/45 core runs; the largest raw PGD rebound was 0.430.
- These descriptive results support a dataset- and budget-dependent account; they do not support a universal monotonic relationship between model size and robustness.

## Defence reporting

- **PGD-AT:** 19/45 evaluated seeds met the non-collapse criterion in the planned table cells.
- **TRADES:** 27/27 evaluated seeds met the non-collapse criterion in the planned table cells.
- **MART:** 26/27 evaluated seeds met the non-collapse criterion in the planned table cells.

Success-only defence means are diagnostic supplements. The primary table retains every seed and reports `n_success/3` and collapse status beside any success-only value.

## AutoAttack audit

Audited cells: 37 (33 in the defence tables, 4 on case-study backbones reported separately).

| Regime | Cells | AutoAttack lower | Largest reduction | Mean RA under AutoAttack |
|---|---|---|---|---|
| success | 23 | 6 | 0.027 | 0.622 |
| collapsed | 10 | 3 | 0.309 | 0.001 |

2 audited cell(s) carry no published PGD-50 figure. The audit script reads that field from `results/<dataset>/<model>/defense_<method>/seed<N>/defense_results*.json`; where no such run was ever stored the field is left empty rather than filled from the audit's own 256-image subset, which has a different denominator. Affected cells:
- `chest_xray_pneumonia/convnext_tiny/TRADES/seed42`: no stored defence evaluation under the published protocol.
- `chest_xray_pneumonia/deit_small/TRADES/seed42`: no stored defence evaluation under the published protocol.

## Scope controls

- Cross-dataset claims use only the complete three-dataset ResNet matrix.
- DeiT-S, ConvNeXt-T, AutoAttack, Square Attack, CW, and DeepFool are labelled Chest X-ray case studies.
- Grad-CAM and 2D projections are explanatory observations only.