# 实验记录：医学图像复杂度-鲁棒性评估 + 对抗训练

> 毕设题目：*Robustness Evaluation on Medical Image Classification Models*
> 借鉴：Rodriguez et al. 2022 (BMC, *On the role of deep learning model complexity in adversarial robustness for medical images*)
> 最后更新：2026-06-20

---

## 1. 项目目标与研究假设

研究**模型复杂度**如何影响**对抗鲁棒性**,以及**对抗训练(AT)**如何改变这一关系。
固定同一架构家族(ResNet),只变复杂度,标准训练 → ε 扫描画 accuracy-vs-ε 曲线 → PGD 对抗训练重评 → 可解释性分析。

**研究假设(待验证,非预设):**
- **H1**：标准训练下,复杂度较低的模型是否更鲁棒?
- **H2**：对抗训练后,鲁棒性排序是否改变(如最复杂/clean 最高的模型变最鲁棒)?

---

## 2. 实验设置

### 2.1 模型(ResNet 复杂度阶梯,timm,ImageNet 预训练微调)

| 模型 | 参数量 | 角色 |
|---|---|---|
| ResNet-18 | 11.7 M | 最小 |
| ResNet-34 | 21.8 M | |
| ResNet-50 | 25.6 M | 中 |
| ResNet-101 | 44.5 M | |
| ResNet-152 | 60.2 M | 最大 |

标准训练 5 个全做;PGD-AT 做 3 个(18/50/152)。

### 2.2 数据集

| 数据集 | 规模 | 类别 | 划分 | 角色 |
|---|---|---|---|---|
| **Chest X-ray pneumonia** | train 5148 / test 624 | 2(NORMAL/PNEUMONIA) | 官方 train/test,val 从 train 切 10% | **主**(已完成) |
| OCT2017 | train 83484 / test 968 | 4(CNV/DME/DRUSEN/NORMAL) | 官方 train/test,val 切 10% | 次(数据就位,未跑) |
| Malaria (NIH) | 27558(70/10/20) | 2(Parasitized/Uninfected) | **患者级 GroupShuffleSplit**(0 泄漏) | 次(数据就位,未跑) |

> ISIC2020 已弃用(极端不平衡)并删除,归档于 `_archive/`。

### 2.3 训练配置

- 优化器 Adam,lr 1e-4,weight_decay 1e-4,cosine 调度,AMP(fp16)。
- epochs:chest_xray 20 / OCT 10 / malaria 15。
- 类别平衡:chest_xray、OCT 用 `weighted_cross_entropy`;malaria `none`(≈50/50)。
- 模型选择按 **val balanced accuracy**(防止塌缩到多数类)。
- 环境:conda env `(medimg-robust)`,torch + timm + ART(adversarial-robustness-toolbox)+ sklearn,单卡 CUDA。

---

## 3. 方法学

### 3.1 攻击

- **核心**:FGSM(单步)+ PGD(L∞,标准模型 PGD-20、随机起点)。
- **核心 ε 网格**(`attacks_main`):`{1,2,4,8,16}/255`。
- **细 ε 探针**(`attacks_fine`):`{0.05,0.1,0.15,0.2,0.25,0.5,1}/255` —— **PGD 真正的可区分区间在此**(见结果)。
- **stress**(`attacks_stress`):`{32,64}/255`(单列)。
- **扩展**(`attacks_extended`,可选):AutoPGD、SquareAttack。
- 规范:模型 eval、参数冻结、ART `clip_values=(0,1)`、白盒;`NormalizedModel` 在边界内做 ImageNet 归一化,攻击作用于 [0,1] 像素。
- **公平性**:所有模型/攻击共用同一固定测试集(chest_xray test=624<1024,直接用全测试集)。

### 3.2 对抗训练(AT)

- **PGD-AT(Madry)**为核心:内层 PGD-7,eps=8/255,eps_step=2/255,nb_epochs 对齐标准训练,**完整训练集**。
- TRADES 作为可选第二方法(本轮未跑)。
- **防御模型强评估**(避免高估):**PGD-50 + 5 随机重启**;`defense_eval` 段 eps={1,2,4,8,16}/255。
- **AutoAttack**:仅 ≥3 类启用(二分类下其 DLR 损失无定义会崩,见 §6)。

### 3.3 评估指标

- **Full robust accuracy** = (clean 正确 ∧ 攻击后仍正确) / **全部测试样本**。← accuracy-vs-ε 主曲线
- **Conditional robust accuracy** = 同分子 / **clean 正确样本数**。
- **ASR** = 1 − conditional。
- **bootstrap CI**:对每点重采样测试集给 95% CI(单 seed 内);多 seed 时图用**跨 seed mean±std**。
- 另:accuracy drop、per-class robustness、ECE 校准、扰动幅度(L2/L∞ 核验预算)。
- **Clean 基线**:accuracy、balanced accuracy、ROC-AUC、per-class/macro/weighted P-R-F1、混淆矩阵。

---

## 4. 实验流程与指令

### 4.1 准备

```powershell
# 数据(chest_xray/oct2017 已有;仅 malaria 需下)
python scripts/download_data.py --dataset malaria
# 生成 15 份配置(3 数据集 × 5 模型,ε网格/训练块逐字节一致)
python scripts/make_configs.py
```

### 4.2 单 (数据集,模型) 标准流程

```powershell
python scripts/train.py --config configs/chest_xray_pneumonia_resnet18.yaml [--seed 43]
python scripts/evaluate_clean.py --config ... --checkpoint checkpoints/..._seed42.pth [--seed 43]
python scripts/evaluate_robustness.py --config ... --checkpoint ... --max-samples 1024 [--seed 43]
python scripts/evaluate_robustness.py --config ... --checkpoint ... --attacks-section attacks_fine --max-samples 1024 [--seed 43]
```

### 4.3 对抗训练 + 强评估

```powershell
# 训练 + 强评估(完整训练集;--max-samples 只限评估子集)
python scripts/evaluate_defense.py --config configs/chest_xray_pneumonia_resnet18.yaml --defense PGD-AT --max-samples 1024
# 只重评(加 --checkpoint 跳过训练;断点续跑只补缺攻击)
python scripts/evaluate_defense.py --config ... --defense PGD-AT --checkpoint checkpoints/..._pgd_at.pth --max-samples 1024
```

### 4.4 可解释性 + 出图

```powershell
python scripts/generate_gradcam_figures.py --config ... --checkpoint ... --attack PGD --eps 0.031373 --num-samples 8 [--out-dir ..._at]
python scripts/generate_decision_boundary_figures.py --dataset chest_xray_pneumonia --models resnet18 resnet152 [--checkpoint-suffix _pgd_at]
python scripts/generate_complexity_figures.py --dataset chest_xray_pneumonia --seeds seed42 seed43 seed44
```

### 4.5 批处理

- `run_pipeline.bat <dataset> <model>` —— 单模型 train→clean→鲁棒(main+fine)→GradCAM。
- `run_all_models.bat` —— 三数据集 × 5 模型 + AT + 出图。
- `finish_chest_xray.bat` —— chest_xray 收尾(补缺 + seed43/44 多 seed + AT + 决策边界 + 出图)。

---

## 5. 结果(chest_xray,已完成)

### 5.1 Clean 基线(seed42,test=624)

| 模型 | accuracy | balanced acc | ROC-AUC | macro F1 |
|---|---|---|---|---|
| ResNet-18 | 0.829 | 0.772 | 0.952 | 0.792 |
| ResNet-34 | 0.848 | 0.799 | 0.960 | 0.819 |
| ResNet-50 | 0.846 | 0.798 | **0.970** | 0.818 |
| ResNet-101 | 0.841 | 0.792 | 0.960 | 0.812 |
| ResNet-152 | 0.827 | 0.770 | 0.965 | 0.790 |

> clean 大致可比(0.83-0.85);NORMAL recall 偏低(~0.55,Kermany 测试集已知分布漂移)。

### 5.2 标准训练:复杂度-鲁棒性(3 seed:42/43/44,mean±std)

**PGD@0.1/255(full robust accuracy)—— 核心招牌结果:**

| 模型 | 参数M | clean | **PGD@0.1/255** | FGSM@0.1/255 | 分组 |
|---|---|---|---|---|---|
| ResNet-18 | 11.7 | 0.813±0.011 | **0.296 ± 0.037** | 0.569 | 🟢 鲁棒 |
| ResNet-34 | 21.8 | 0.833±0.013 | **0.308 ± 0.101** | 0.614 | 🟢 鲁棒 |
| ResNet-50 | 25.6 | 0.869±0.021 | **0.021 ± 0.014** | 0.573 | 🔴 崩溃 |
| ResNet-101 | 44.5 | 0.827±0.010 | **0.051 ± 0.027** | 0.592 | 🔴 崩溃 |
| ResNet-152 | 60.2 | 0.839±0.010 | **0.316 ± 0.071** | 0.622 | 🟢 鲁棒 |

**发现(H1):非单调,3-seed 稳定复现。** {R18,R34,R152}≈0.30 vs {R50,R101}≈0.03,两组误差带不重叠 → 真信号,非噪声。
- **范文「越简单越鲁棒」的单调结论在本设置(预训练微调 + 可靠 PGD)下不复现。**
- 形态:中上等容量(R50/R101)最脆弱,两端 + R34 鲁棒。
- 混淆:R50 clean 最高(0.869)却最脆弱(准确率-鲁棒性权衡的体现),但 R152 高 clean 又鲁棒 → 非单一规律。
- PGD 仅在 sub-0.25/255 有区分度;≥1/255 全塌缩(见 §6 大 ε 退化)。FGSM 全程平缓且趋势相反。

### 5.3 对抗训练(PGD-50+5重启强评估,full robust accuracy)

| 模型 | AT-clean | PGD@1 | PGD@2 | PGD@4 | **PGD@8** | PGD@16 |
|---|---|---|---|---|---|---|
| ResNet-18 | 0.742 | 0.724 | 0.708 | 0.678 | **0.622** | 0.442 |
| ResNet-50 | 0.785 | 0.763 | 0.736 | 0.699 | **0.627** | 0.268 |
| ResNet-152 | 0.798 | 0.779 | 0.748 | 0.718 | **0.625** | 0.228 |

**发现(H2):**
1. **AT 巨幅提升鲁棒性**:标准 PGD@8/255 ≈ 0 → AT 后 ≈ **0.62**(三者一致)。
2. **AT-clean 随复杂度单调上升**(0.742→0.785→0.798)→ 大模型 AT 后 clean 代价更小。
3. **小-中 ε(1-4/255):大模型更鲁棒**(R152>R50>R18)→ **印证范文 AT 结论**(clean 最高/最复杂者最鲁棒)。
4. **8/255 三者≈相等**(AT 抹平复杂度差异);**16/255 反转**(R18 0.44 > R152 0.23)。

### 5.4 可解释性

- **Grad-CAM**:标准 + AT 模型的 clean vs adversarial 显著图(`figures/gradcam/chest_xray_pneumonia/{model}{,_at}`)。
- **决策边界 t-SNE+KNN**:R18 vs R152,clean(o)按类聚簇、adversarial(×)被推过边界(`figures/decision_boundary/chest_xray_pneumonia/`)。

### 5.5 一句话总结

> 标准训练下复杂度-鲁棒性**非单调**(范文单调结论不成立);可靠 PGD 揭示中上容量模型最脆弱。**AT 大幅提升鲁棒性且使复杂度有利**(大模型 clean 代价更小、小中 ε 更鲁棒,印证范文 AT 结论),但极大 ε 反转。**FGSM 与 PGD 趋势相反**,印证单步评估不可靠。

---

## 6. 关键技术发现 / 坑

1. **大 ε 退化伪影**:标准模型在 ≥1/255 下 PGD 把预测塌缩到多数类(PNEUMONIA),使 full robust acc 虚高(如 R50@8/255 假 0.25)。→ **PGD 标准曲线裁到 ≤0.3/255**;FGSM 保留全程;AT 对比柱的 R50 标准值需脚注说明。
2. **AutoAttack 对二分类不适用**:ART 的 APGD-DLR 损失需 ≥3 类,二分类报 `index -3 is out of bounds for dimension 1 with size 2`,且极慢(APGD-CE 单模型 ~33min)。→ `create_defense_eval_attacks` 已改为 **nb_classes<3 自动跳过**;二分类强评估=PGD-50+5重启。OCT(4 类)将启用。需 `pip install multiprocess`。
3. **图脚本曾有取整 bug**:`_parse_eps_255` 把 eps*255 取整,导致 fine 的 0.05/0.1/0.15 全 round 成 0 互相覆盖 → 已改浮点 + x 轴 log。
4. **AT 训练量误限**:`evaluate_defense` 原 `--max-samples` 同时限制了训练数据 → 已解耦,正式 AT 用完整训练集,`--max-samples` 只限评估子集。
5. **指标定义**:统一为 full / conditional 双口径,ASR=1−conditional。

---

## 7. 产出文件

- **结果 JSON**:`results/chest_xray_pneumonia/{model}/{clean,robustness,defense_PGD-AT}/seed{N}/*.json`
- **图**:`figures/complexity/chest_xray_pneumonia/`(FGSM/PGD 曲线带 3-seed 带、AT 曲线、AT 对比柱、`complexity_summary_table.csv`、`at_comparison_table.csv`)、`figures/gradcam/`、`figures/decision_boundary/`
- **checkpoints**:`checkpoints/chest_xray_pneumonia_{model}_seed{42,43,44}.pth`(标准)+ `..._seed42_pgd_at.pth`(AT:18/50/152)
- **配置**:`configs/{dataset}_base.yaml` + 生成的 `configs/{dataset}_{model}.yaml`(15 份)

---

## 8. 状态与待办

**已完成:** chest_xray 全流程(标准 3-seed + AT + 强评估 + 可解释性 + 全套图表)。

**待办:**
- [ ] OCT2017 全量(标准 5 模型 + ε 扫描;AT 可选;AutoAttack 启用但慢)——验证非单调/AT 结论是否跨数据集。
- [ ] Malaria 全量(同 chest_xray,二分类跳过 AutoAttack)。
- [ ] TRADES 第二方法(可选,算力不足只在 R50 消融)。
- [ ] 多 seed:OCT/malaria 建议单 seed(次数据集);仅 chest_xray 做了 3-seed。
- [ ] 论文 Results/Discussion 写作。

**全实验预估:** chest_xray 已耗 ~13h;OCT+malaria 全量约 ~15h(OCT 训练 83k 是大头,可子采样加速)。

---

## 9. 复现要点

- 种子 42(主)/43/44(多 seed);确定性划分;每次跑 `save_config_snapshot`。
- 结果路径 `results/{dataset}/{model}/{experiment}/seed{N}/`;checkpoint `checkpoints/{dataset}_{model}_seed{N}[_pgd_at].pth`。
- 配置由 `make_configs.py` 从 base 生成,保证跨模型 ε 网格/训练块一致。
- 计划文档:`~/.claude/plans/detection-focus-on-robustness-evaluatio-golden-flask.md`;总体计划 `../MEDICAL_ROBUSTNESS_PLAN.md`。
