#!/usr/bin/env bash
# ============================================================================
# 建议1:只重跑「塌缩 / 评估被截断」的 4 个 AT,全部用盘上已带 warmup 的配置
# (eps_warmup=5, lr_warmup=3, nb_epochs=20)。健康的 R50 / malaria 等一律不碰。
#
# 用法(AutoDL Linux 实例,过夜无人值守、可关笔记本):
#   cd /root/autodl-tmp/Robustness-Evaluation-on-Medical-Image-Classification-Models
#   git pull                       # 或用 WinSCP/JupyterLab 直接把本文件传到项目根目录
#   nohup bash run_fix_at.sh > fix_at.log 2>&1 &
#   tail -f fix_at.log             # Ctrl-C 只停看日志,任务在后台继续
#
# 收敛自检(跑的过程中就能看):
#   grep -E "AT epoch|train eps|Best val" fix_at.log
#   # train loss 要跌破 ln(类数)(二分类<0.693 / 四分类<1.386),val robust 要上升
# ============================================================================
set -uo pipefail

# ---- AutoDL 加速:学术网络 + HF 镜像(timm 预训练权重 / git) ----
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
[ -f /etc/network_turbo ] && source /etc/network_turbo || true

MAX=1024

# 自检:GPU 是否可用(不可用就提前报警,别白跑)
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "[ERROR] CUDA 不可用 —— 先 bash setup_cloud.sh 再来。"; exit 1; }

run () {  # 参数:dataset model seed
  local d="$1" m="$2" s="$3" cfg="configs/$1_$2.yaml"
  if [ ! -f "$cfg" ]; then echo "[skip] 缺配置 $cfg"; return; fi
  echo ""
  echo "############### FIX-AT  ${d} / ${m} / seed${s} ###############"
  date
  python scripts/evaluate_defense.py --config "$cfg" \
    --defense PGD-AT --max-samples "$MAX" --seed "$s" \
    || echo "[fail] ${d}/${m}/seed${s} —— 跳过,继续下一个。"
}

echo "################ FIX-AT 重跑开始 ################"; date

# chest_xray:R18 两个 seed 塌缩(pre-warmup);R152 评估被截断到 1/255 → warmup 重训 + 全 ε 评估
run chest_xray_pneumonia resnet18  42
run chest_xray_pneumonia resnet18  43
run chest_xray_pneumonia resnet152 42

# oct2017:R152 保存的是无 warmup / 10 epoch 的旧结果 → warmup + 20 epoch 重训
# (兄弟 R50 用同一配方已从全 0 恢复到 clean 0.74,故 R152 恢复概率高)
run oct2017 resnet152 42

echo ""
echo "################ FIX-AT 全部完成 ################"; date
echo "下一步:tar czf results.tgz results/ checkpoints/*_pgd_at.pth  然后下载,再「关机」省 GPU。"
