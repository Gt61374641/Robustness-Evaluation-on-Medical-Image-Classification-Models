# 实验记录：医学图像复杂度-鲁棒性评估 + 对抗训练

> 毕设题目：*Robustness Evaluation on Medical Image Classification Models*
> 借鉴：Rodriguez et al. 2022 (BMC, *On the role of deep learning model complexity in adversarial robustness for medical images*)
> 最后更新：2026-07-10（扩展批次代码就绪：新架构 DeiT-S/ConvNeXt-T、四种新攻击、MART、5 模型 AT 补全 → 见 §9 + `run_extension.sh`，待云端跑）

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

标准训练 5 个全做;PGD-AT 原做 3 个(18/50/152),**扩展批次补齐 R34/R101(三数据集)→ 5 模型全覆盖**。

**架构对比(扩展批次,chest 主数据集)**:与 ResNet-50(25.6M)参数配对的两个非 ResNet 架构,
预训练权重统一固定为 **ImageNet-1k-only 有监督**(隔离"架构"变量,排除预训练数据差异):

| 模型 | timm 权重 | 参数量 | 家族 |
|---|---|---|---|
| DeiT-S(= ViT-S/16 架构) | `deit_small_patch16_224.fb_in1k` | 22.1 M | Transformer |
| ConvNeXt-T | `convnext_tiny.fb_in1k` | 28.6 M | 现代 CNN |

> 注:不用 timm 默认 `vit_small_patch16_224`(其权重是 IN21k 预训练→IN1k 微调,与 ResNet 不公平);
> DeiT-S 就是 ViT-S/16 架构、IN1k-only 训练,论文中可写 "ViT-S (DeiT-S weights)"。
> chest 上全流程(标准 3-seed + ε 扫描 + attacks_extra + PGD-AT);Grad-CAM 仅 ConvNeXt
> (conv-hook 式 Grad-CAM 不适用 ViT token 图,写为局限)。

### 2.2 数据集

| 数据集 | 规模 | 类别 | 划分 | 角色 |
|---|---|---|---|---|
| **Chest X-ray pneumonia** | train 5148 / test 624 | 2(NORMAL/PNEUMONIA) | 官方 train/test,val 从 train 切 10% | **主**(已完成,含 TRADES) |
| OCT2017 | train 83484 / test 968 | 4(CNV/DME/DRUSEN/NORMAL) | 官方 train/test,val 切 10% | 次(已完成:标准 3-seed + AT) |
| Malaria (NIH) | 27558(70/10/20) | 2(Parasitized/Uninfected) | **患者级 GroupShuffleSplit**(0 泄漏) | 次(已完成:标准 3-seed + AT) |

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
- **攻击方法对比(`attacks_extra`,扩展批次,仅 chest,7 个标准模型)**:
  - **CW**(L2 优化型白盒,max_iter 100 × 10 binary-search;无 ε 网格 → 看扰动幅度 L2/L∞,不能直接和有界攻击比 robust acc);
  - **DeepFool**(最小扰动,直接度量到决策边界的距离);
  - **AutoAttack**(强集成;**二分类修复**:ART 默认集成含 APGD-DLR,二分类崩 → 工厂自动改用
    自定义列表 APGD-CE + DeepFool + Square,AutoAttack 自身会拒绝超出 ε 的候选,故 DeepFool 成员不越界);
  - **SquareAttack**(黑盒免梯度,5000 查询标准预算,排除梯度混淆嫌疑)。
  - ε ∈ {2, 8}/255(有界攻击)。共 6 种攻击范式:单步/迭代白盒、优化型、最小扰动、集成、黑盒。
- 规范:模型 eval、参数冻结、ART `clip_values=(0,1)`、白盒;`NormalizedModel` 在边界内做 ImageNet 归一化,攻击作用于 [0,1] 像素。
- **公平性**:所有模型/攻击共用同一固定测试集(chest_xray test=624<1024,直接用全测试集)。

### 3.2 对抗训练(AT)

- **PGD-AT(Madry)**为核心:内层 PGD-7,eps=8/255,eps_step=2/255,nb_epochs 对齐标准训练,**完整训练集**。
- **TRADES 第二方法消融(已完成,chest R18/R50/R152)**:走 ART 的 TRADES 训练器(β=6),**不带** PGD-AT 自定义循环的 eps/lr warmup;强评估同为 PGD-50+5重启。见 §5.3c。
- **MART 第三方法(扩展批次,chest R18/R50/R152)**:misclassification-aware(Wang et al. 2020,
  boosted CE + (1−p_clean)-加权 KL,β=6);**与 PGD-AT 共用自定义循环** → 同 warmup 稳定器、同
  内层 PGD-7、同鲁棒-val 选点,与 PGD-AT 严格可比(TRADES 走 ART 训练器无 warmup,对比时注意)。
  类别不平衡权重只作用于 CE 项(与 PGD-AT/标准训练一致)。
- **预处理防御(SpatialSmoothing/JPEG/FeatureSqueezing)明确不作主防御**:靠梯度混淆,自适应攻击
  (BPDA)可击穿,只保留为代码内 baseline,不进论文主结论。
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

### 5.3 对抗训练(统一 warmup 协议;PGD-50+5重启强评估,full robust accuracy)

> **最终验证版**:2026-06-30/07-01 在 AutoDL 用**统一 warmup 协议**(eps_warmup=5,lr_warmup=3,
> nb_epochs=20,自定义 PGD-AT 循环对齐标准训练:weighted CE + cosine + AMP + 鲁棒-val 选点)重跑并核验。
> 下表 chest(R50/R152 多 seed)。**注:此表替换了旧的非统一循环单值表**(旧值 R18 0.742/R50 0.785/R152 0.798 @8≈0.62)。
>
> ✅ **协议一致性(chest R50,已解决 2026-07-01)**:R50(seed42/43)的 AT 已用当前含 warmup 的 config
> (eps_warmup=5/lr_warmup=3/nb_epochs=20)**从 ImageNet 重训重评**(权重 mtime 2026-07-01,运行时 config 快照含
> warmup 字段),与 R18/R152 严格 apples-to-apples。与旧无-warmup 值(@8 0.527/0.522)几乎一致(@8 0.506/0.514)
> → **结论稳健,warmup 未翻案**;旧值已备份为 `..._max1024.nowarmup.bak.json`。malaria/oct 数据集内部协议本就一致。

| 模型 | seed | AT-clean | PGD@1 | PGD@2 | PGD@4 | **PGD@8** | PGD@16 | 状态 |
|---|---|---|---|---|---|---|---|---|
| ResNet-18 | 42 | 0.614 | 0.000 | 0.000 | 0.000 | **0.000** | 0.000 | ❌ 塌缩 |
| ResNet-18 | 43 | 0.625 | 0.300 | 0.051 | 0.000 | **0.000** | 0.000 | ❌ 塌缩 |
| ResNet-50 | 42 | 0.779 | 0.742 | 0.715 | 0.670 | **0.506** | 0.242 | ✅ (warmup) |
| ResNet-50 | 43 | 0.804 | 0.774 | 0.747 | 0.681 | **0.514** | 0.260 | ✅ (warmup) |
| ResNet-152 | 42 | 0.819 | 0.812 | 0.787 | 0.745 | **0.649** | 0.290 | ✅ 最强 |

**发现(H2,统一协议下):**
1. **AT 巨幅提升鲁棒性**:标准 PGD@8/255 强评估 ≈ 0 → AT 后 R50≈0.52、R152≈0.65。
2. **复杂度单调有利**:clean(0.80→0.82)与 robust@8(0.527→0.649)都随复杂度上升,**R152 最强** → **印证范文 AT 结论**(最复杂者 AT 后最鲁棒)。
3. **最小模型 R18 塌缩**:统一 warmup 协议下两 seed 均塌(clean≈0.62、@8=0)。旧的非统一循环下 R18 曾训成(0.622@8),说明 **R18 处于 AT 可训性边缘、对协议高度敏感** → 作为发现写入(与"复杂度有利"一致)。

### 5.3b 跨数据集 AT 汇总(@8/255,full robust acc;最终验证版)

| 模型 | chest clean/@8 | malaria clean/@8 | oct clean/@8 | 跨数据集 |
|---|---|---|---|---|
| ResNet-18 | 0.62 / **0.00** ❌ | 0.46 / **0.00** ❌ | 0.41 / **0.16** ⚠️弱 | 最小模型普遍难 AT |
| ResNet-50 | 0.80 / **0.53** ✅ | 0.96 / **0.90** ✅ | 0.74 / **0.64** ✅ | 三处都成功 |
| ResNet-152 | 0.82 / **0.65** ✅ | 0.96 / **0.91** ✅ | 0.25 / **0.03** ❌塌 | chest/malaria 最强;OCT 塌 |

> H2 由 **chest(R50+R152)+ malaria(R50+R152)** 稳稳支撑。**R18 三数据集一致塌缩**、**oct R152 塌缩**(最难 4 类任务 + 最深网络发散到平凡解,train loss 全程钉在 ln4=1.386)→ 均作为诚实发现写入,而非 bug(已用运行时 config 快照 + 逐 epoch 日志确认 warmup 已启用仍塌)。

### 5.3c 第二方法消融:TRADES vs PGD-AT(chest,seed42,PGD-50+5重启,full robust acc)

| 模型 | PGD-AT clean/@8 | **TRADES clean/@8** | 观察 |
|---|---|---|---|
| ResNet-18  | 0.614 / **0.000** ❌塌缩 | 0.771 / **0.619** ✅ | **TRADES 救回了 PGD-AT 塌缩的 R18** |
| ResNet-50  | 0.779 / **0.506** ✅ | 0.809 / **0.641** ✅ | TRADES 在 clean 与 robust 上均更优 |
| ResNet-152 | 0.819 / **0.649** ✅ | 0.782 / **0.657** ✅ | 两法接近(AT clean 略高,TRADES robust 略高) |

**发现(H2 稳健性 / 第二方法):**
1. **两种 AT 方法都大幅提升鲁棒性**,复杂度有利的趋势一致(R50→R152 robust 上升)→ H2 不依赖单一方法,**跨方法复现**。
2. **R18 的塌缩是 PGD-AT 特有的优化不稳定,而非"R18 本质上无法 AT"**:同样**不带 warmup** 的 TRADES(ART 训练器)把 R18 训成了 0.619@8。→ 修正 §5.3「R18 处于 AT 可训性边缘」的措辞:**边缘性对方法/优化高度敏感**,换方法即可跨过。
3. TRADES(β=6)整体给出更好的 clean-robust 权衡(R18/R50 clean 与 robust 双赢)。
> 图/表:`figures/sci_defense/chest_xray_pneumonia/{model}/sci_defense_pgd8_bars.*`(Standard/PGD-AT/TRADES 三柱)、`figures/paper_tables/chest_xray_pneumonia/resnet50/table5_defense_pgd8_comparison.*`。

### 5.4 可解释性

- **Grad-CAM(已扩到三数据集)**:标准 + AT 模型的 clean vs adversarial 显著图。
  - chest:`figures/gradcam/chest_xray_pneumonia/{model}{,_at}`(5 标准模型)。
  - malaria:标准 R18/R152 + AT R50/R152(`_at`);oct:标准 R18/R152 + AT R50(`_at`)。**仅收敛的 AT 模型出图**(塌缩模型的显著图无意义,故略)。
  - **汇总拼图(新)**:`figures/gradcam/{chest_xray_pneumonia,malaria,oct2017}_gradcam_summary.{png,pdf}` —— 行=模型、列=样本,一张图对比复杂度与标准/AT 的注意力迁移(脚本 `scripts/generate_gradcam_summary.py`,只读已生成的 PNG,无需 torch)。
- **决策边界 t-SNE+KNN**:R18 vs R152,clean(o)按类聚簇、adversarial(×)被推过边界(`figures/decision_boundary/chest_xray_pneumonia/`,目前仅 chest)。
- **H1 跨数据集误差带(新)**:`figures/combined/H1_*.{png,pdf}` 现在**三数据集(chest/malaria/oct)均为 seed42/43/44 三-seed mean±std**,误差带完整(此前 malaria/oct 仅 seed42、无带)。

### 5.5 一句话总结

> 标准训练下复杂度-鲁棒性**非单调**(范文单调结论不成立);可靠 PGD 揭示中上容量模型最脆弱。**AT 大幅提升鲁棒性且使复杂度有利**(大模型 clean 代价更小、小中 ε 更鲁棒,印证范文 AT 结论),但极大 ε 反转。**FGSM 与 PGD 趋势相反**,印证单步评估不可靠。

---

## 6. 关键技术发现 / 坑

1. **大 ε 退化伪影**:标准模型在 ≥1/255 下 PGD 把预测塌缩到多数类(PNEUMONIA),使 full robust acc 虚高(如 R50@8/255 假 0.25)。→ **PGD 标准曲线裁到 ≤0.3/255**;FGSM 保留全程;AT 对比柱的 R50 标准值需脚注说明。
2. **AutoAttack 对二分类不适用**:ART 的 APGD-DLR 损失需 ≥3 类,二分类报 `index -3 is out of bounds for dimension 1 with size 2`,且极慢(APGD-CE 单模型 ~33min)。→ `create_defense_eval_attacks` 已改为 **nb_classes<3 自动跳过**;二分类强评估=PGD-50+5重启。OCT(4 类)将启用。需 `pip install multiprocess`。
3. **图脚本曾有取整 bug**:`_parse_eps_255` 把 eps*255 取整,导致 fine 的 0.05/0.1/0.15 全 round 成 0 互相覆盖 → 已改浮点 + x 轴 log。
4. **AT 训练量误限**:`evaluate_defense` 原 `--max-samples` 同时限制了训练数据 → 已解耦,正式 AT 用完整训练集,`--max-samples` 只限评估子集。
5. **指标定义**:统一为 full / conditional 双口径,ASR=1−conditional。
6. **defense 图/表攻击名不匹配 bug(2026-07-01 修)**:标准模型鲁棒 JSON 的攻击键是 `PGD_eps=...`,而防御(PGD-AT/TRADES)评估键是 `PGD50-5restart_eps=...`(强评估)。`generate_defense_sci_figures.py` 按 `attack=="PGD"` 过滤 → 防御行全被漏掉,**「Defense comparison at PGD 8/255」柱状图与 table5/6/7 只剩 Standard**。已加 `_canonical_attack()` 把 PGD 变体归并到 `PGD` 家族;修复后三方法(Standard/PGD-AT/TRADES)正常出现。
7. **纯绘图脚本 torch 解耦**:`src/utils/__init__.py` 急切 `import torch` 会让无 torch 的机器上纯绘图脚本(如经 `plot_style`)导入即崩;已把 torch 依赖导入包 try/except 容错 → 出图流程可在无 GPU/torch 的本地跑。

---

## 7. 产出文件

- **结果 JSON**:`results/{dataset}/{model}/{clean,robustness,defense_PGD-AT,defense_TRADES}/seed{N}/*.json`(三数据集;chest 含 TRADES)
- **图**:
  - `figures/complexity/chest_xray_pneumonia/`(FGSM/PGD 曲线带 3-seed 带、AT 曲线、AT 对比柱、`complexity_summary_table.csv`、`at_comparison_table.csv`)
  - `figures/combined/H1_*.{png,pdf}`、`H2_*`、`paper_style_fig1`(**H1 三数据集均 3-seed 误差带**)
  - `figures/sci_defense/chest_xray_pneumonia/{resnet18,50,152}/`(Standard/PGD-AT/TRADES 对比,含 `sci_defense_summary_metrics.csv`)
  - `figures/paper_tables/chest_xray_pneumonia/resnet50/table1~7`(table5/6/7 已含 TRADES)
  - `figures/gradcam/`(三数据集 + `{dataset}_gradcam_summary.{png,pdf}` 汇总)、`figures/decision_boundary/`
- **checkpoints**:`checkpoints/chest_xray_pneumonia_{model}_seed{42,43,44}.pth`(标准)+ `..._seed42_pgd_at.pth`(AT:18/50/152)
- **配置**:`configs/{dataset}_base.yaml` + 生成的 `configs/{dataset}_{model}.yaml`(15 份)

---

## 8. 状态与待办(更新于 2026-07-01,extras 收尾后)

**已完成:**
- chest_xray / malaria / oct2017 三数据集**标准训练 + ε 扫描**,**三数据集现均 3-seed(42/43/44)** → H1 误差带完整。
- **AT 代表模型 R18/R50/R152 + 强评估**(三数据集),统一 warmup 协议:
  - ✅ **chest R152** 全 ε 评估(clean 0.819 / @8 0.649,最强);**chest R50 warmup 重跑完成**(seed42/43,@8 0.506/0.514,与旧值一致→结论稳健)。
  - ❌ 确认塌缩(warmup 仍救不回,已写成发现):**R18×三数据集(PGD-AT)**、**oct R152**。
- ✅ **TRADES 第二方法消融(chest R18/R50/R152)**:H2 跨方法复现;**TRADES 救回 PGD-AT 塌缩的 R18**(见 §5.3c)。
- ✅ **可解释性扩到三数据集**:malaria/oct Grad-CAM(标准 + 收敛的 AT)+ 三数据集汇总拼图。
- ✅ **图/表全部重生成**:H1/H2 combined(3-seed 带)、sci_defense(含 TRADES)、paper_tables(table5/6/7 含 TRADES)、gradcam 汇总。修复 defense 图/表攻击名不匹配 bug(§6.6)。

**待办(下次再做):**
- [ ] (可选)再救一次 **oct R152**:更强稳定化(eps_warmup=8 / lr_warmup=5 / nb_epochs=30 / LR 减半 / 梯度裁剪),独立一次性重跑。
- [ ] (可选)malaria/oct 的**决策边界**图 + `generate_complexity_figures.py` 三数据集刷新(complexity 曲线/CSV)。
- [ ] (可选)malaria/oct 的 TRADES 消融、chest AT 多 seed(R152 seed43)。
- [ ] (可选)`generate_clean_sci_figures.py` 刷新其余模型的 clean 诊断图/表(需 torch + checkpoint)。
- [ ] 论文 **Results / Discussion 写作**。

**复现要点(本轮新增):** AT 权重在 `checkpoints/{dataset}_{model}_seed42[_43]_pgd_at.pth`(共 11 个,已从云端下载到本地);
塌缩判定依据 = 运行时 `results/.../defense_PGD-AT/seedN/config.yaml` 快照(确认 warmup 已启用)+ `evaluate_defense.log`

---

## 9. 扩展批次(2026-07-10 定稿,代码就绪,待云端跑:`run_extension.sh`)

**范围决定:chest(主)全做;malaria/oct 只补 R34/R101 的 PGD-AT。**

| # | 内容 | 明细 | 状态 |
|---|---|---|---|
| 1 | 新架构标准训练 | chest × {deit_small, convnext_tiny} × seed{42,43,44},train→clean→鲁棒(main+fine) | ⬜ |
| 2 | 攻击方法对比 | chest × 7 模型 × `attacks_extra`(CW/DeepFool/AutoAttack/Square),seed42 | ⬜ |
| 3 | AT 补全 | {chest, malaria, oct} × {R34, R101} × PGD-AT + 强评估 → **5 模型 AT 全覆盖** | ⬜ |
| 4 | 新架构 AT | chest × {deit_small, convnext_tiny} × PGD-AT | ⬜ |
| 5 | MART | chest × {R18, R50, R152}(与 TRADES 三元组对齐)→ 三方法对比 | ⬜ |
| 6 | Grad-CAM | convnext_tiny 标准+AT(deit 跳过,ViT CAM 局限) | ⬜ |

**本批次代码改动(已完成并本地校验语法/配置):**
- `src/models/model_factory.py`:+`deit_small`(ViT-S/16, IN1k-only)、+`convnext_tiny`。
- `src/attacks/attack_factory.py`:CW/DeepFool 补 `batch_size`(ART 默认 1 慢到不可用);
  **AutoAttack 二分类修复**(nb_classes<3 → 自定义集成 APGD-CE+DeepFool+Square,去 DLR)。
- `scripts/evaluate_defense.py`:`_run_pgd_at` 泛化为 `_run_custom_at`(PGD-AT/MART 共用),
  新增 `_mart_loss`(忠实官方实现);`MAIN_DEFENSES` += MART。
- `configs/chest_xray_pneumonia_base.yaml`:+`attacks_extra` 段、+MART 防御块(batch 8,双前向)。
- `scripts/make_configs.py`:按数据集区分模型清单(chest 17 份配置中含 2 个新架构)。
- `scripts/evaluate_robustness.py`:`--attacks-section` += `attacks_extra`。

**协议决定(重要):防御(defended-model)强评估协议保持不变**(PGD-50+5重启;二分类跳过
AutoAttack)。二分类可用的 AutoAttack 只用于**标准模型**的攻击对比(attacks_extra)——若给新
AT 模型加 AutoAttack 而旧结果没有,同一张表里会混两种协议;要加必须先全量回填(见
`create_defense_eval_attacks` 注释)。

**跑完后的出图/表(下一步):** 架构对比曲线(ResNet 阶梯 + 2 新架构同图)、六攻击对比表
(有界攻击 robust acc @2,8/255;CW/DeepFool 报扰动 L2/L∞)、三防御方法柱状图
(Standard/PGD-AT/TRADES/MART)、5 点 H2 复杂度曲线 —— 需要小幅扩展
`generate_complexity_figures.py`/`generate_defense_sci_figures.py` 的模型清单参数。
逐 epoch(train loss 是否钉在 ln(类数):二分类 0.693 / 四分类 1.386)。

---

## 9. 复现要点

- 种子 42(主)/43/44(多 seed);确定性划分;每次跑 `save_config_snapshot`。
- 结果路径 `results/{dataset}/{model}/{experiment}/seed{N}/`;checkpoint `checkpoints/{dataset}_{model}_seed{N}[_pgd_at].pth`。
- 配置由 `make_configs.py` 从 base 生成,保证跨模型 ε 网格/训练块一致。
- 计划文档:`~/.claude/plans/detection-focus-on-robustness-evaluatio-golden-flask.md`;总体计划 `../MEDICAL_ROBUSTNESS_PLAN.md`。
