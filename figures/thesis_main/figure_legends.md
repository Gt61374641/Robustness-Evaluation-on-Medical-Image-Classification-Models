# Main figure legends

## Figure 3

Representative classes from the Chest X-ray, Malaria, and OCT2017 evaluation datasets. This descriptive overview establishes modality and label-space differences and is not used as quantitative evidence.

## Figure 4

Unified evaluation workflow. The cross-dataset core comprises three imaging datasets, a five-depth ResNet ladder, matched FGSM/PGD budgets, audited robustness metrics, and three adversarial-training methods. DeiT-S, ConvNeXt-T, and the multi-attack suite are Chest X-ray case studies.

## Figure 5

Construction of the audited attack-set envelope, illustrated on Chest X-ray ResNet-50 at seed 42. Panel a shows the raw FGSM and PGD-20 curves with their per-budget minimum, which coincides with PGD at every budget in this run. Panel b applies the cumulative minimum over increasing budget, which is the step that removes the high-budget rebound. This run is an illustrative pathological example chosen because it carries the largest raw PGD rebound of the core sweep, not a representative one.

## Figure 6

Clean test accuracy of the five ResNet depths across datasets. Points show the three-seed mean and error bars show one standard deviation.

## Figure 7

Representative clean images, scaled perturbations, and PGD adversarial images for the three datasets. Perturbations are amplified for visibility; quantitative conclusions rely on the attack sweeps rather than visual salience.

## Figure 8

Observed FGSM conditional robust accuracy across perturbation budgets. Lines show three-seed means and shaded bands show descriptive 95% confidence intervals.

## Figure 9

Conservative attack-ensemble robustness curves. For each seed and budget, the lower of FGSM and PGD conditional robust accuracy is retained, followed by a cumulative minimum over increasing budgets. This prevents high-budget PGD optimisation failures from being interpreted as recovered robustness.

## Figure 10

Model parameters versus normalised log-epsilon robustness AUC and the critical perturbation budget at which audited conditional robust accuracy first falls to 0.5. Spearman correlations are descriptive; upward triangles denote right-censoring above 16/255.

## Figure 11

Chest X-ray cross-architecture extension. Clean accuracy and audited robustness AUC are compared for the ResNet ladder, DeiT-S, and ConvNeXt-T. This single-dataset extension is a case study and is not pooled with the cross-dataset ResNet analysis.

## Figure 12

Chest X-ray multi-attack audit for seed 42. Bounded-attack robust accuracy and CW/DeepFool minimal L2 perturbations are shown in separate panels because the metrics are not directly comparable.

## Figure 13

Chest X-ray defence comparison for representative ResNet depths. All-seed estimates retain collapsed runs; successful-seed summaries must be accompanied by the success count.

## Figure 14

Malaria defence comparison for representative ResNet depths under the same reporting rule.

## Figure 15

OCT defence comparison for representative ResNet depths under the same reporting rule.

## Figure 16

Interpretive visualisations. Grad-CAM illustrates representative attention shifts, while two-dimensional penultimate-layer feature projections are overlaid with surrogate decision regions. Neither component is used to prove quantitative superiority or to claim recovery of the true high-dimensional decision boundary.
