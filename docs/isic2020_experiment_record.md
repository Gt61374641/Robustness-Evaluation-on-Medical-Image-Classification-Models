# ISIC 2020 Experiment Record

Date recorded: 2026-05-06  
Dataset: ISIC 2020  
Model: DenseNet121  
Seed: 42  
Config: `configs/isic2020_densenet121.yaml`  
Main checkpoint: `checkpoints/isic2020_densenet121_seed42.pth`

## 1. Current Status

The ISIC 2020 pipeline has been successfully run end-to-end for the current DenseNet121 baseline:

- Clean training and clean evaluation completed.
- Main robustness attacks completed with `--max-samples 1024`.
- Extended attacks completed with `--max-samples 1024`.
- Main adversarial-training defenses completed: PGD-AT and TRADES.
- SCI-style attack/defense figures generated.
- Paper tables generated after patching the table script to support ISIC class names.
- Weekly-report PPT generated from the current ISIC results.

Generated result groups:

- `results/isic2020/densenet121/train/seed42/history.json`
- `results/isic2020/densenet121/clean/seed42/clean_results.json`
- `results/isic2020/densenet121/robustness/seed42/robustness_attacks_main.json`
- `results/isic2020/densenet121/robustness/seed42/robustness_attacks_extended.json`
- `results/isic2020/densenet121/defense_PGD-AT/seed42/defense_results.json`
- `results/isic2020/densenet121/defense_TRADES/seed42/defense_results.json`
- `figures/sci/isic2020/densenet121/`
- `figures/sci_clean/isic2020/densenet121/`
- `figures/sci_defense/isic2020/densenet121/`
- `figures/paper_tables/isic2020/densenet121/`

## 2. Clean Performance

Overall clean performance looks high, but it is misleading because the model fails the minority malignant class.

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9783 |
| Balanced accuracy | 0.5000 |
| ROC AUC | 0.8727 |
| ECE | 0.0029 |
| Test samples | 3314 |

Per-class clean performance:

| Class | Support | Class accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| benign | 3242 | 1.0000 | 0.9783 | 1.0000 | 0.9890 |
| malignant | 72 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Important finding:

The model appears accurate because ISIC 2020 is extremely imbalanced. In the full metadata, benign samples are about 98.24% and malignant samples are about 1.76%. The current model effectively learns a majority-class solution: it predicts benign very well but does not correctly detect malignant cases.

This means the current ISIC result is useful as a completed pipeline run, but it should not be treated as a clinically meaningful final model.

## 3. Main Attack Findings

Representative attack results at 8/255 or default DeepFool:

| Attack | Clean accuracy | Robust accuracy | ASR | Accuracy drop | ECE adv |
| --- | ---: | ---: | ---: | ---: | ---: |
| FGSM 8/255 | 0.9824 | 1.0000 | 0.0000 | 0.0000 | 0.0175 |
| PGD 8/255 | 0.9824 | 0.1889 | 0.8111 | 0.7793 | 0.4648 |
| AutoPGD 8/255 | 0.9824 | 0.0000 | 1.0000 | 0.9648 | 0.9441 |
| DeepFool | 0.9824 | 0.2962 | 0.7038 | 0.6758 | 0.1934 |

Interpretation:

- FGSM has almost no visible effect in this run.
- PGD sharply reduces robust accuracy.
- AutoPGD is the strongest current attack and collapses robust accuracy to 0 on the 1024-sample evaluation subset.
- DeepFool also exposes substantial fragility.
- Therefore, FGSM should be reported as a weak one-step baseline, while PGD, AutoPGD and DeepFool provide the more meaningful robustness evidence.

## 4. FGSM-Specific Observation

FGSM was evaluated across epsilon values from 1/255 to 16/255:

| FGSM epsilon | Robust accuracy | ASR | Linf max | L2 mean |
| --- | ---: | ---: | ---: | ---: |
| 1/255 | 0.9990 | 0.0010 | 0.003921 | 1.5173 |
| 2/255 | 0.9990 | 0.0010 | 0.007843 | 3.0309 |
| 4/255 | 1.0000 | 0.0000 | 0.015686 | 6.0474 |
| 8/255 | 1.0000 | 0.0000 | 0.031372 | 12.0405 |
| 16/255 | 1.0000 | 0.0000 | 0.062745 | 23.8614 |

Code-level explanation:

- FGSM is implemented using ART `FastGradientMethod` with only `eps`.
- It is a single-step attack: it computes one gradient direction and applies one bounded perturbation.
- PGD is implemented as a multi-step attack with `max_iter=20`, `eps_step=eps/10`, and `num_random_init=1`.
- Evaluation metrics compute robust accuracy only on samples that were originally correct on clean data.
- In the 1024-sample attack subset, nearly all originally-correct samples are benign. Malignant samples have clean accuracy 0, so they do not meaningfully contribute to robust accuracy.

Therefore, FGSM does not prove that the model is robust. It mainly shows that a one-step perturbation does not cross the current majority-class benign decision boundary.

## 5. Defense Findings

PGD 8/255 defense comparison:

| Method | Category | Clean accuracy | Robust accuracy | ASR | Accuracy drop |
| --- | --- | ---: | ---: | ---: | ---: |
| Standard | Standard model | 0.9783 | 0.1889 | 0.8111 | 0.7793 |
| PGD-AT | Adversarial training | 0.9824 | 0.9940 | 0.0060 | 0.0059 |
| TRADES | Adversarial training | 0.9824 | 1.0000 | 0.0000 | 0.0000 |

DeepFool defense comparison:

| Method | Category | Clean accuracy | Robust accuracy | ASR | Accuracy drop |
| --- | --- | ---: | ---: | ---: | ---: |
| Standard | Standard model | 0.9783 | 0.2962 | 0.7038 | 0.6758 |
| PGD-AT | Adversarial training | 0.9824 | 0.4960 | 0.5040 | 0.4824 |
| TRADES | Adversarial training | 0.9824 | 0.4702 | 0.5298 | 0.5088 |

Interpretation:

- PGD-AT and TRADES look highly effective against PGD 8/255 on the 1024-sample subset.
- The same defenses provide only moderate gains against DeepFool.
- Defense effectiveness is attack-dependent.
- These defense results should be treated as preliminary until the clean model handles the malignant class and the evaluation is repeated.

## 6. Issues Encountered and Fixes Already Made

### Missing dependency

Initial run failed with:

`ModuleNotFoundError: No module named 'timm'`

Resolution: install missing dependencies in the conda environment.

### Wrong working directory

Initial command was run from `C:\Users\13070`, so Python could not find `scripts/train.py`.

Resolution: run commands from `C:\Users\13070\Desktop\UCL\毕设\code1`.

### CPU/GPU memory pressure during robustness evaluation

Full robustness evaluation attempted to collect all test samples into memory and some attacks allocated large arrays.

Resolution: use `--max-samples 1024` for expensive attacks and defense evaluation.

### Defense OOM bug

`evaluate_defense.py` initially collected full training data for adversarial training even when `--max-samples` was provided.

Resolution: patched `run_adversarial_training` so `max_samples` is passed into the training-data collection step.

### Paper table class-name bug

`generate_paper_tables.py` originally expected class names like `Normal` and `Pneumonia`, causing ISIC table generation to fail.

Resolution: patched table generation to dynamically detect class columns such as `asr_benign` and `asr_malignant`.

## 7. Current Limitations

The most important limitation is class imbalance:

- Overall accuracy is dominated by benign samples.
- Malignant recall is 0.
- Robust accuracy and ASR mainly reflect behavior on benign samples.
- Defense results may look over-optimistic because they also operate on a benign-dominated evaluation subset.

Secondary limitations:

- Main attack and defense results were run with `--max-samples 1024`, not the full test set.
- Current result is seed 42 only.
- Preprocessing baseline defenses have not been run for ISIC.
- Confidence intervals or repeated subsets/seeds have not yet been reported.

## 8. Next Experiment Improvements

Priority 1: fix the clean ISIC classifier before drawing strong robustness conclusions.

Recommended changes:

1. Add class imbalance handling during training.
   - Try weighted cross-entropy.
   - Try focal loss.
   - Try balanced sampler or malignant oversampling.

2. Track better clean metrics.
   - Balanced accuracy.
   - Malignant recall.
   - Malignant precision.
   - Macro F1.
   - ROC AUC and PR AUC.

3. Repeat robustness evaluation after clean model correction.
   - Main attacks: FGSM, PGD, DeepFool.
   - Extended attacks: AutoPGD and SquareAttack.
   - Keep the same epsilon sweep for comparability.

4. Repeat defense evaluation after clean model correction.
   - PGD-AT.
   - TRADES.
   - Optional preprocessing baselines only after main conclusions are stable.

5. Improve reporting.
   - Report per-class ASR.
   - Report robust accuracy for malignant separately.
   - Add repeated seed or repeated subset evaluation.
   - Clearly distinguish full test-set clean metrics from `--max-samples 1024` robustness metrics.

## 9. Suggested Immediate Next Run

Before scaling to another dataset, run a corrected ISIC baseline with class imbalance handling.

Target:

- Maintain high clean accuracy.
- Increase malignant recall from 0 to a meaningful value.
- Re-run PGD/AutoPGD/DeepFool after the model no longer collapses to the benign class.

Only after this correction should the ISIC attack/defense results be treated as thesis-grade evidence.

## 10. Thesis Narrative So Far

A useful current framing:

The first ISIC experiment successfully validates the end-to-end robustness pipeline, including training, clean evaluation, adversarial attacks, defenses, SCI-style figures and paper tables. However, the results reveal a clinically important failure mode: high overall clean accuracy hides complete failure on the malignant minority class. Strong iterative attacks such as PGD and AutoPGD expose major fragility, while FGSM is too weak and too dominated by majority-class behavior to be informative. The next stage should correct class imbalance and repeat the robustness analysis before scaling the same pipeline to additional datasets and models.
