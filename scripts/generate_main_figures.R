# Main manuscript figures (R backend) -- ggplot2 + patchwork, nature-figure style.
#
# R counterpart of scripts/generate_main_figures.py. Reads the SAME shared data
# in figures/data/ so numbers match the Python backend exactly. Outputs carry an
# "_r" suffix so both backends coexist for side-by-side comparison.
#
#   H1_pgd_across_datasets_r   PGD robust acc vs eps, 5-model ladder, 3 datasets
#   H1_attack_budget_r         FGSM vs PGD vs eps (rows) x datasets (cols)
#   H1_complexity_ushape_r     robust acc @0.1/255 vs capacity (U-shape)
#   defense_methods_r          Standard/PGD-AT/TRADES/MART, chest R18/R50/R152
#   attack_methods_r           CW/DeepFool L2 + AutoAttack/Square robust@8
#
# Run:  Rscript scripts/generate_main_figures.R
# Needs: jsonlite, ggplot2, patchwork, svglite, ragg

suppressPackageStartupMessages({
  library(jsonlite); library(ggplot2); library(patchwork)
})

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
proj_root <- if (length(script_path) == 1 && nzchar(script_path))
  normalizePath(file.path(dirname(script_path), ".."), mustWork = FALSE) else getwd()
if (!dir.exists(file.path(proj_root, "figures"))) proj_root <- getwd()
DATA <- file.path(proj_root, "figures", "data")
OUT  <- file.path(proj_root, "figures", "main"); dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

theme_set(
  theme_classic(base_size = 6.5, base_family = "Arial") +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks = element_line(linewidth = 0.35, colour = "black"),
      legend.title = element_blank(),
      legend.text = element_text(size = 5.6),
      legend.key.size = unit(2.6, "mm"),
      strip.text = element_text(size = 7, face = "bold"),
      strip.background = element_blank(),
      plot.tag = element_text(size = 10, face = "bold"),
      panel.grid = element_blank()
    )
)

LADDER_COLORS <- c("ResNet-18" = "#9FC4E8", "ResNet-34" = "#5B8FD6",
                   "ResNet-50" = "#0F4D92", "ResNet-101" = "#6A3D9A",
                   "ResNet-152" = "#9A4D8E")
DATASET_LEVELS <- c("Chest X-ray", "Malaria", "OCT")
METHOD_COLORS <- c("Standard" = "#CFCECE", "PGD-AT" = "#0F4D92",
                   "TRADES" = "#42949E", "MART" = "#9A4D8E")
COLLAPSE_RED <- "#B64342"
DISP <- c(resnet18 = "ResNet-18", resnet34 = "ResNet-34", resnet50 = "ResNet-50",
          resnet101 = "ResNet-101", resnet152 = "ResNet-152",
          deit_small = "DeiT-S", convnext_tiny = "ConvNeXt-T")

save_pub <- function(plot, name, w_mm, h_mm) {
  w <- w_mm / 25.4; h <- h_mm / 25.4; base <- file.path(OUT, name)
  svglite::svglite(paste0(base, ".svg"), width = w, height = h); print(plot); dev.off()
  grDevices::cairo_pdf(paste0(base, ".pdf"), width = w, height = h, family = "Arial"); print(plot); dev.off()
  ragg::agg_png(paste0(base, ".png"), width = w, height = h, units = "in", res = 600); print(plot); dev.off()
  cat("wrote", paste0("figures/main/", name), "\n")
}

# ── build long data frames from the shared JSON ─────────────────────────────
curves_long <- function() {
  d <- fromJSON(file.path(DATA, "h1_pgd_curves.json"), simplifyVector = FALSE)
  rows <- list()
  for (dd in d$datasets) {
    for (m in names(dd$models)) {
      for (atk in c("FGSM", "PGD")) {
        s <- dd$models[[m]][[atk]]
        if (length(s) == 0) next
        eps <- vapply(s, function(p) p$eps, numeric(1))
        mn  <- vapply(s, function(p) p$mean, numeric(1))
        sd_ <- vapply(s, function(p) p$std, numeric(1))
        rows[[length(rows) + 1]] <- data.frame(
          dataset = dd$display, model = DISP[[m]], attack = atk,
          eps = eps, mean = mn * 100, std = sd_ * 100, stringsAsFactors = FALSE)
      }
    }
  }
  df <- do.call(rbind, rows)
  df$dataset <- factor(df$dataset, levels = DATASET_LEVELS)
  df$model <- factor(df$model, levels = names(LADDER_COLORS))
  df
}

# ── H1 PGD across datasets ──────────────────────────────────────────────────
fig_h1_pgd <- function() {
  df <- subset(curves_long(), attack == "PGD" & eps <= 0.3)
  p <- ggplot(df, aes(eps, mean, colour = model, fill = model)) +
    geom_ribbon(aes(ymin = mean - std, ymax = mean + std), alpha = 0.13, colour = NA) +
    geom_line(linewidth = 0.5) + geom_point(size = 0.8) +
    facet_wrap(~dataset, nrow = 1) +
    scale_x_log10() +
    scale_colour_manual(values = LADDER_COLORS) +
    scale_fill_manual(values = LADDER_COLORS, guide = "none") +
    coord_cartesian(ylim = c(-3, 100)) +
    labs(x = expression(epsilon ~ "(/255, log)"),
         y = "PGD robust accuracy (%)", tag = "a") +
    theme(legend.position = c(0.99, 0.98), legend.justification = c(1, 1))
  save_pub(p, "H1_pgd_across_datasets_r", 183, 66)
}

# ── H1 FGSM vs PGD budget grid ──────────────────────────────────────────────
fig_h1_budget <- function() {
  df <- curves_long()
  df$attack <- factor(df$attack, levels = c("FGSM", "PGD"))
  p <- ggplot(df, aes(eps, mean, colour = model, fill = model)) +
    geom_ribbon(aes(ymin = mean - std, ymax = mean + std), alpha = 0.12, colour = NA) +
    geom_line(linewidth = 0.45) + geom_point(size = 0.6) +
    facet_grid(attack ~ dataset) +
    scale_x_log10() +
    scale_colour_manual(values = LADDER_COLORS) +
    scale_fill_manual(values = LADDER_COLORS, guide = "none") +
    coord_cartesian(ylim = c(-3, 100)) +
    labs(x = expression(epsilon ~ "(/255, log)"), y = "Robust accuracy (%)", tag = "a") +
    theme(legend.position = c(0.99, 0.99), legend.justification = c(1, 1))
  save_pub(p, "H1_attack_budget_r", 183, 112)
}

# ── H1 U-shape ──────────────────────────────────────────────────────────────
fig_h1_ushape <- function() {
  d <- fromJSON(file.path(DATA, "h1_complexity_fixedeps.json"), simplifyVector = FALSE)
  ladder <- unlist(d$ladder)
  rows <- list()
  for (dd in d$datasets) {
    for (i in seq_along(ladder)) {
      m <- ladder[i]; rec <- dd$models[[m]]
      grp <- if (i %in% c(1, 5)) "ends robust" else if (i %in% c(3, 4)) "middle fragile" else "other"
      rows[[length(rows) + 1]] <- data.frame(
        dataset = dd$display, xpos = i,
        model = sub("ResNet-", "R", DISP[[m]]),
        mean = ifelse(is.null(rec$mean), NA, rec$mean) * 100,
        std = ifelse(is.null(rec$std), 0, rec$std) * 100,
        grp = grp, stringsAsFactors = FALSE)
    }
  }
  df <- do.call(rbind, rows)
  df$dataset <- factor(df$dataset, levels = DATASET_LEVELS)
  df$model <- factor(df$model, levels = c("R18", "R34", "R50", "R101", "R152"))
  grp_cols <- c("ends robust" = "#2E7D32", "middle fragile" = COLLAPSE_RED, "other" = "#767676")
  p <- ggplot(df, aes(xpos, mean)) +
    geom_line(colour = "#767676", linewidth = 0.5) +
    geom_errorbar(aes(ymin = mean - std, ymax = mean + std, colour = grp),
                  width = 0.15, linewidth = 0.4) +
    geom_point(aes(colour = grp), size = 1.6) +
    facet_wrap(~dataset, nrow = 1, scales = "free_y") +
    scale_y_continuous(expand = expansion(mult = c(0.06, 0.16))) +
    scale_colour_manual(values = grp_cols, breaks = c("ends robust", "middle fragile")) +
    scale_x_continuous(breaks = 1:5, labels = c("R18", "R34", "R50", "R101", "R152"),
                       expand = expansion(add = 0.4)) +
    labs(x = "model capacity →",
         y = expression("Robust accuracy (%) @ " * epsilon * "=0.1/255"), tag = "a") +
    theme(legend.position = c(0.99, 0.99), legend.justification = c(1, 1))
  save_pub(p, "H1_complexity_ushape_r", 183, 66)
}

# ── Defense methods ─────────────────────────────────────────────────────────
fig_defense <- function() {
  d <- fromJSON(file.path(DATA, "defense_methods.json"), simplifyVector = FALSE)
  rows <- lapply(d$rows, function(r) data.frame(
    model = sub("ResNet-", "R", DISP[[r$model]]), method = r$method,
    clean = ifelse(is.null(r$clean), NA, r$clean) * 100,
    rob8  = ifelse(is.null(r$rob8), NA, r$rob8) * 100,
    collapsed = isTRUE(r$collapsed), stringsAsFactors = FALSE))
  df <- do.call(rbind, rows)
  df$model <- factor(df$model, levels = c("R18", "R50", "R152"))
  df$method <- factor(df$method, levels = names(METHOD_COLORS))
  dodge <- position_dodge(width = 0.8)
  mk <- function(yvar, ylab, mark) {
    g <- ggplot(df, aes(model, .data[[yvar]], fill = method)) +
      geom_col(position = dodge, width = 0.75, colour = "black", linewidth = 0.2) +
      scale_fill_manual(values = METHOD_COLORS) +
      coord_cartesian(ylim = c(0, 100)) +
      labs(x = NULL, y = ylab)
    if (mark) g <- g + geom_point(
      data = subset(df, collapsed), aes(y = 2.2, group = method),
      position = dodge, shape = 4, colour = COLLAPSE_RED, size = 1.1,
      stroke = 0.6, show.legend = FALSE)
    g
  }
  pa <- mk("rob8", "Robust accuracy (%) @ 8/255", TRUE) + labs(tag = "a") +
    theme(legend.position = c(0.02, 0.98), legend.justification = c(0, 1))
  pb <- mk("clean", "Clean accuracy (%)", FALSE) + labs(tag = "b") +
    theme(legend.position = "none")
  save_pub(pa + pb, "defense_methods_r", 183, 76)
}

# ── Attack methods ──────────────────────────────────────────────────────────
fig_attack <- function() {
  d <- fromJSON(file.path(DATA, "attack_methods.json"), simplifyVector = FALSE)
  getv <- function(r, k) ifelse(is.null(r[[k]]), NA, r[[k]])
  rows <- lapply(d$rows, function(r) data.frame(
    model = DISP[[r$model]],
    CW = getv(r, "CW_l2"), DeepFool = getv(r, "DeepFool_l2"),
    AutoAttack = getv(r, "AutoAttack8") * 100, Square = getv(r, "Square8") * 100,
    stringsAsFactors = FALSE))
  df <- do.call(rbind, rows)
  df$model <- factor(df$model, levels = DISP[c("resnet18", "resnet34", "resnet50",
                     "resnet101", "resnet152", "deit_small", "convnext_tiny")])
  long_l2 <- rbind(
    data.frame(model = df$model, attack = "CW", val = df$CW),
    data.frame(model = df$model, attack = "DeepFool", val = df$DeepFool))
  long_ra <- rbind(
    data.frame(model = df$model, attack = "AutoAttack", val = df$AutoAttack),
    data.frame(model = df$model, attack = "Square", val = df$Square))
  dodge <- position_dodge(width = 0.75)
  pa <- ggplot(long_l2, aes(model, val, fill = attack)) +
    geom_col(position = dodge, width = 0.7, colour = "black", linewidth = 0.2) +
    scale_fill_manual(values = c("CW" = "#0F4D92", "DeepFool" = "#42949E")) +
    labs(x = NULL, y = expression("Mean " * L[2] * " perturbation to fool"), tag = "a") +
    theme(axis.text.x = element_text(angle = 35, hjust = 1),
          legend.position = c(0.02, 0.98), legend.justification = c(0, 1))
  ymax <- max(long_ra$val, na.rm = TRUE) * 1.3; if (ymax < 20) ymax <- 20
  pb <- ggplot(long_ra, aes(model, val, fill = attack)) +
    geom_col(position = dodge, width = 0.7, colour = "black", linewidth = 0.2) +
    scale_fill_manual(values = c("AutoAttack" = "#B64342", "Square" = "#E28E2C")) +
    annotate("text", x = 4, y = ymax * 0.92, label = "AutoAttack drives every model to ≈0",
             size = 2, colour = COLLAPSE_RED, fontface = "italic") +
    coord_cartesian(ylim = c(0, ymax)) +
    labs(x = NULL, y = "Conditional robust accuracy (%) @ 8/255", tag = "b") +
    theme(axis.text.x = element_text(angle = 35, hjust = 1),
          legend.position = c(0.99, 0.98), legend.justification = c(1, 1))
  save_pub(pa + pb, "attack_methods_r", 186, 76)
}

fig_h1_pgd(); fig_h1_budget(); fig_h1_ushape(); fig_defense(); fig_attack()
cat("done -> figures/main/ (_r)\n")
