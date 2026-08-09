# Thesis main tables

- `table01_datasets.csv`: effective partitions used by the experiments.
- `table02_protocol.csv`: model, attack, audit, and defence settings.
- `table03_core_metrics.csv`: three-seed clean and audited robustness summary.
- `table04_defenses.csv`: all-seed and success-only defence summaries.

`robust_accuracy_8_255_all_mean` is full adversarial accuracy, so it is directly
comparable to clean accuracy. `n_success_over_3` and `collapse_status` must always
be shown beside any success-only value. Critical epsilon is right-censored when
the audited conditional robust accuracy does not fall to 0.5 by 16/255.
