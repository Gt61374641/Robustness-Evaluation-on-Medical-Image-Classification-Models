"""Decision-boundary visualisation via t-SNE + KNN (paper Fig 5/6).

For two model complexities, extract penultimate (pre-logit) features for a fixed
stratified set of test samples and their PGD-adversarial versions, project both to
2D with t-SNE, fit a KNN on the clean projection to draw decision regions, and
overlay clean (o) vs adversarial (x) points. Tighter clusters / adversarial points
sitting across the boundary indicate a more input-sensitive (less robust) model.

Usage:
    python scripts/generate_decision_boundary_figures.py --dataset chest_xray_pneumonia \
        --models resnet18 resnet152 --eps 0.031373 --max-samples 300
    # adversarially trained models (Fig 6): add --checkpoint-suffix _pgd_at
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from art.estimators.classification import PyTorchClassifier
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier

from src.utils.reproducibility import set_seed, load_config, get_results_dir
from src.datasets import get_dataloaders
from src.models import create_model
from src.models.model_factory import load_checkpoint
from src.attacks.attack_factory import create_attack
from src.evaluation.subset import get_attack_subset
from src.utils.plot_style import apply_publication_style, add_panel_label, finalize_figure, class_color
from scripts.evaluate_robustness import collect_test_data

DISPLAY = {"resnet18": "ResNet-18", "resnet34": "ResNet-34", "resnet50": "ResNet-50",
           "resnet101": "ResNet-101", "resnet152": "ResNet-152"}


@torch.no_grad()
def extract_features(model, x_np, device, batch=64):
    """Pre-logit (penultimate, pooled) features for inputs in [0,1]."""
    model.eval()
    feats = []
    for i in range(0, len(x_np), batch):
        xb = torch.from_numpy(x_np[i:i + batch]).to(device)
        f = model.forward_features(xb)
        emb = model.forward_head(f, pre_logits=True)
        feats.append(emb.cpu().numpy())
    return np.concatenate(feats, axis=0)


def panel(ax, emb2d, labels, n_clean, class_names, title, letter):
    """KNN decision regions on the clean projection; clean (o) vs adv (x) overlay."""
    clean_xy, adv_xy = emb2d[:n_clean], emb2d[n_clean:]
    clean_y, adv_y = labels[:n_clean], labels[n_clean:]

    knn = KNeighborsClassifier(n_neighbors=15).fit(clean_xy, clean_y)
    pad = (emb2d.max(0) - emb2d.min(0)) * 0.05 + 1e-6
    x0, y0 = emb2d.min(0) - pad
    x1, y1 = emb2d.max(0) + pad
    xx, yy = np.meshgrid(np.linspace(x0, x1, 250), np.linspace(y0, y1, 250))
    zz = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    classes = sorted(set(labels.tolist()))
    colors = [class_color(class_names[c] if c < len(class_names) else str(c), c) for c in classes]
    cmap = matplotlib.colors.ListedColormap(colors)
    ax.contourf(xx, yy, zz, alpha=0.18, cmap=cmap, levels=np.arange(-0.5, len(classes), 1))

    for c, col in zip(classes, colors):
        cn = class_names[c] if c < len(class_names) else str(c)
        m = clean_y == c
        ax.scatter(clean_xy[m, 0], clean_xy[m, 1], s=8, c=[col], marker="o",
                   edgecolors="none", label=f"{cn} (clean)")
        ma = adv_y == c
        ax.scatter(adv_xy[ma, 0], adv_xy[ma, 1], s=14, c=[col], marker="x", linewidths=0.8,
                   label=f"{cn} (adv)")
    ax.set_xticks([]); ax.set_yticks([])
    add_panel_label(ax, letter)
    ax.set_title(title)


def model_embedding(cfg, ckpt, device, x_clean, eps, max_iter):
    data_classes = cfg["_num_classes"]
    model = create_model(cfg["model"]["name"], data_classes, pretrained=False)
    model = load_checkpoint(model, ckpt, device=device).to(device).eval()

    classifier = PyTorchClassifier(
        model=model, loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        input_shape=(3, cfg["data"]["img_size"], cfg["data"]["img_size"]),
        nb_classes=data_classes, clip_values=(0.0, 1.0),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )
    attack = create_attack(classifier, {"name": "PGD", "max_iter": max_iter, "num_random_init": 1}, eps=eps)
    x_adv = attack.generate(x=x_clean)

    f_clean = extract_features(model, x_clean, device)
    f_adv = extract_features(model, x_adv, device)
    feats = np.concatenate([f_clean, f_adv], axis=0)
    emb2d = TSNE(n_components=2, init="pca", perplexity=30, random_state=cfg.get("seed", 42)).fit_transform(feats)
    return emb2d


def main():
    parser = argparse.ArgumentParser(description="t-SNE + KNN decision boundary figures")
    parser.add_argument("--dataset", default="chest_xray_pneumonia")
    parser.add_argument("--models", nargs="+", default=["resnet18", "resnet152"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=0.031373, help="PGD eps (default 8/255)")
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--checkpoint-suffix", default="", help="e.g. _pgd_at for AT models")
    parser.add_argument("--output-dir", type=Path, default=Path("figures") / "decision_boundary")
    args = parser.parse_args()

    set_seed(args.seed)
    apply_publication_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Shared fixed stratified subset of the test set (same images for both models).
    base_cfg = load_config(f"configs/{args.dataset}_{args.models[0]}.yaml")
    data = get_dataloaders(base_cfg)
    class_names = data.get("class_names", [])
    x_full, y_full = collect_test_data(data["test"], max_samples=None)
    if args.max_samples < len(x_full):
        cache = Path("results") / args.dataset / f"db_subset_seed{args.seed}_n{args.max_samples}.json"
        idx = get_attack_subset(y_full, args.max_samples, args.seed, cache)
        x_clean, y = x_full[idx], y_full[idx]
    else:
        x_clean, y = x_full, y_full
    labels = np.concatenate([y, y])  # clean then adv share true labels

    fig, axes = plt.subplots(1, len(args.models), figsize=(3.6 * len(args.models), 3.2))
    if len(args.models) == 1:
        axes = [axes]
    for ax, model_name, letter in zip(axes, args.models, "abcd"):
        ckpt = f"checkpoints/{args.dataset}_{model_name}_seed{args.seed}{args.checkpoint_suffix}.pth"
        if not Path(ckpt).exists():
            ax.set_title(f"{DISPLAY.get(model_name, model_name)}\n(missing checkpoint)")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        cfg = load_config(f"configs/{args.dataset}_{model_name}.yaml")
        cfg["_num_classes"] = data["num_classes"]
        emb2d = model_embedding(cfg, ckpt, device, x_clean, args.eps, args.max_iter)
        panel(ax, emb2d, labels, len(x_clean), class_names,
              DISPLAY.get(model_name, model_name), letter)

    axes[0].legend(fontsize=5, loc="best", framealpha=0.7)
    out_dir = args.output_dir / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.checkpoint_suffix.strip("_") or "standard"
    paths = finalize_figure(fig, out_dir / f"decision_boundary_{tag}", pad=1.0)
    print("Generated:")
    for p in paths:
        print(" ", p)


if __name__ == "__main__":
    main()
