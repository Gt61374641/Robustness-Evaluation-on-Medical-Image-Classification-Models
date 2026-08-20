# Thesis main tables

The file number and the dissertation table number are not the same, because
dissertation Table 4 is the AutoAttack audit and is not rendered here.

| File | Dissertation table | Contents |
|---|---|---|
| `table01_datasets.csv` | Table 1 | Effective partitions used by the experiments |
| `table02_protocol.csv` | Table 2 | Model, attack, audit, and defence settings |
| `table03_core_metrics.csv` | Table 3 | Three-seed clean and audited robustness summary |
| `table04_defenses.csv` | Table 5 | All-seed and success-only defence summaries |
| — | Table 4 | Grouped from `reports/thesis_evidence/autoattack_audit.csv`; see the "AutoAttack audit" section of `analysis_note.md` |

`robust_accuracy_8_255_all_mean` is full adversarial accuracy, so it is directly
comparable to clean accuracy. `n_success_over_3` and `collapse_status` must always
be shown beside any success-only value. Critical epsilon is right-censored when
the audited conditional robust accuracy does not fall to 0.5 by 16/255.
