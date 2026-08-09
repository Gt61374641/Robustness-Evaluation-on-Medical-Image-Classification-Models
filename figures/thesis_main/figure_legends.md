# Main figure legends

## Figure 1

Representative classes from the Chest X-ray, Malaria, and OCT2017 evaluation datasets. This descriptive overview establishes modality and label-space differences and is not used as quantitative evidence.

## Figure 2

Unified evaluation workflow. The cross-dataset core comprises three imaging datasets, a five-depth ResNet ladder, matched FGSM/PGD budgets, audited robustness metrics, and three adversarial-training methods. DeiT-S, ConvNeXt-T, and the multi-attack suite are Chest X-ray case studies.

## Figure 3

Construction of the audited attack-set envelope, illustrated on Chest X-ray ResNet-50 at seed 42. Panel a shows the raw FGSM and PGD-20 curves with their per-budget minimum, which coincides with PGD at every budget in this run. Panel b applies the cumulative minimum over increasing budget, which is the step that removes the high-budget rebound. This run is an illustrative pathological example chosen because it carries the largest raw PGD rebound of the core sweep, not a representative one.

## Figure 4

Clean test accuracy of the five ResNet depths across datasets. Points show the three-seed mean and error bars show one standard deviation.

## Figure 5

Representative clean images, scaled perturbations, and PGD adversarial images for the three datasets. Perturbations are amplified for visibility; quantitative conclusions rely on the attack sweeps rather than visual salience.

## Figure 6

Observed FGSM conditional robust accuracy across perturbation budgets. Lines show three-seed means and shaded bands show descriptive 95% confidence intervals.

## Figure 7

Conservative attack-ensemble robustness curves. For each seed and budget, the lower of FGSM and PGD conditional robust accuracy is retained, followed by a cumulative minimum over increasing budgets. This prevents high-budget PGD optimisation failures from being interpreted as recovered robustness.

## Figure 8

Model parameters versus normalised log-epsilon robustness AUC and the critical perturbation budget at which audited conditional robust accuracy first falls to 0.5. Spearman correlations are descriptive; upward triangles denote right-censoring above 16/255.

## Figure 9

Chest X-ray cross-architecture extension. Clean accuracy and audited robustness AUC are compared for the ResNet ladder, DeiT-S, and ConvNeXt-T. This single-dataset extension is a case study and is not pooled with the cross-dataset ResNet analysis.

## Figure 10

Chest X-ray multi-attack audit for seed 42. Bounded-attack robust accuracy and CW/DeepFool minimal L2 perturbations are shown in separate panels because the metrics are not directly comparable.

## Figure 11

Chest X-ray defence comparison for representative ResNet depths. Bars show the estimate conditional on convergence; a cross marks a cell in which no seed converged, and a k/n annotation marks a bar resting on fewer than three seeds. All-seed estimates are given in Table 4. The Standard baseline is the audited envelope value, 0.000 at 8/255.

## Figure 12

Malaria defence comparison for representative ResNet depths under the same reporting rule.

## Figure 13

OCT defence comparison for representative ResNet depths under the same reporting rule.

## Figure 14

Interpretive visualisations. Grad-CAM illustrates representative attention shifts, while two-dimensional penultimate-layer feature projections are overlaid with surrogate decision regions. Neither component is used to prove quantitative superiority or to claim recovery of the true high-dimensional decision boundary.
