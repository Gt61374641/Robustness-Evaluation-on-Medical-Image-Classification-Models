# PGD-AT rescue / optimisation-stability figure -- R backend.
#
# ggplot2 + patchwork twin panels, same data & scientific claim as the Python
# version (scripts/generate_rescue_figure.py). Reads the shared data file:
#   figures/data/at_rescue.json   (written by scripts/extract_figure_data.py)
#
#   (a) Clean accuracy  : original PGD-AT vs rescue, per collapsed point
#   (b) PGD-8/255 robust: same; recovered points annotated with the gain
# Collapsed runs are drawn hollow (white fill + coloured outline) with a red x
# at the bar base, matching the ladder figure's visual grammar.
#
# Run:  Rscript scripts/generate_rescue_figure.R
# Needs: jsonlite, ggplot2, patchwork, svglite, ragg

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
  library(patchwork)
})

# Resolve project root from the script's own path (Rscript) or fall back to cwd.
script_path <- sub("^--file=", "",
                   grep("^--file=", commandArgs(FALSE), value = TRUE))
if (length(script_path) == 1 && nzchar(script_path)) {
  proj_root <- normalizePath(file.path(dirname(script_path), ".."), mustWork = FALSE)
} else {
  proj_root <- getwd()
}
if (!dir.exists(file.path(proj_root, "figures"))) proj_root <- getwd()

# ── Nature-style theme (shared with the ladder figure) ──────────────────────
theme_set(
  theme_classic(base_size = 6.5, base_family = "Arial") +
    theme(
      axis.line   = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks  = element_line(linewidth = 0.35, colour = "black"),
      legend.title = element_blank(),
      legend.text = element_text(size = 5.8),
      legend.key.size = unit(3, "mm"),
      legend.position = "bottom",
      plot.tag    = element_text(size = 10, face = "bold"),
      panel.grid  = element_blank(),
      axis.text.x = element_text(size = 5.8)
    )
)

ORIG_COLOR   <- "#9AA6B2"  # under-stabilised original run
RESCUE_COLOR <- "#0F4D92"  # stronger-stabilisation rescue run
COLLAPSE_RED <- "#B64342"
PROTO_COLORS <- c("Original PGD-AT" = ORIG_COLOR,
                  "PGD-AT rescue"   = RESCUE_COLOR)
DISPLAY <- c(resnet18 = "ResNet-18", resnet34 = "ResNet-34", resnet50 = "ResNet-50",
             resnet101 = "ResNet-101", resnet152 = "ResNet-152")

# ── Data: reshape orig/rescue x clean/robust8 into a long frame ─────────────
rows <- fromJSON(file.path(proj_root, "figures", "data", "at_rescue.json"),
                 simplifyDataFrame = TRUE)$rows
# story-first order: recovered point (rescue not collapsed) first
rows <- rows[order(rows$rescue_collapsed, rows$dataset), ]
rows$point <- paste0(rows$dataset_display, "\n", DISPLAY[rows$model])
point_levels <- rows$point

long <- do.call(rbind, lapply(seq_len(nrow(rows)), function(i) {
  r <- rows[i, ]
  data.frame(
    point     = rep(r$point, 4),
    protocol  = c("Original PGD-AT", "PGD-AT rescue"),
    metric    = rep(c("clean", "robust8"), each = 2),
    value     = c(r$orig_clean, r$rescue_clean, r$orig_robust8, r$rescue_robust8),
    collapsed = c(r$orig_collapsed, r$rescue_collapsed,
                  r$orig_collapsed, r$rescue_collapsed),
    stringsAsFactors = FALSE
  )
}))
long$point    <- factor(long$point, levels = point_levels)
long$protocol <- factor(long$protocol, levels = names(PROTO_COLORS))

# recovery annotation: rescue succeeded where original had collapsed (robust panel)
ann <- do.call(rbind, lapply(seq_len(nrow(rows)), function(i) {
  r <- rows[i, ]
  if (!r$rescue_collapsed && r$orig_collapsed) {
    data.frame(point = r$point,
               value = r$rescue_robust8,
               label = sprintf("+%.0f pts", (r$rescue_robust8 - r$orig_robust8) * 100),
               stringsAsFactors = FALSE)
  } else NULL
}))

dodge <- position_dodge(width = 0.7)

panel <- function(metric_key, ylab, annotate = FALSE) {
  d <- subset(long, metric == metric_key)
  d$point <- factor(d$point, levels = point_levels)
  p <- ggplot(d, aes(x = point, y = value, fill = protocol)) +
    geom_col(aes(colour = protocol), position = dodge, width = 0.62,
             linewidth = 0.35) +
    # collapsed bars: overpaint white so only the coloured outline shows
    geom_col(data = subset(d, collapsed), fill = "white",
             aes(colour = protocol), position = dodge, width = 0.62,
             linewidth = 0.35, show.legend = FALSE) +
    # red x at the base of every collapsed bar (mapped to shape -> legend entry)
    geom_point(data = subset(d, collapsed),
               aes(y = 0.02, group = protocol,
                   shape = "collapsed (trivial classifier)"),
               position = dodge, colour = COLLAPSE_RED,
               size = 1.3, stroke = 0.65) +
    scale_fill_manual(values = PROTO_COLORS) +
    scale_colour_manual(values = PROTO_COLORS, guide = "none") +
    scale_shape_manual(name = NULL, values = c("collapsed (trivial classifier)" = 4)) +
    guides(shape = guide_legend(override.aes = list(colour = COLLAPSE_RED))) +
    coord_cartesian(ylim = c(0, 1.03)) +
    scale_y_continuous(breaks = seq(0, 1, 0.2)) +
    labs(x = NULL, y = ylab)
  if (annotate && !is.null(ann) && nrow(ann) > 0) {
    ad <- ann; ad$point <- factor(ad$point, levels = point_levels)
    p <- p + geom_text(data = ad,
                       aes(x = point, y = value, label = label),
                       inherit.aes = FALSE, colour = RESCUE_COLOR,
                       fontface = "bold", size = 2.1, vjust = -0.6,
                       nudge_x = 0.17)
  }
  p
}

pa <- panel("clean", "Clean accuracy") + labs(tag = "a") +
  theme(legend.position = "none")
pb <- panel("robust8", "PGD-8/255 robust accuracy (full)", annotate = TRUE) +
  labs(tag = "b")

fig <- pa + pb + plot_layout(ncol = 2, guides = "collect") &
  theme(legend.position = "bottom")

# ── Export (svglite / cairo_pdf / ragg, per skill R quick-start) ────────────
out_dir <- file.path(proj_root, "figures", "at_ladder")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
base <- file.path(out_dir, "H2b_rescue_stability_r")
w <- 160 / 25.4; h <- 82 / 25.4

svglite::svglite(paste0(base, ".svg"), width = w, height = h)
print(fig); dev.off()
grDevices::cairo_pdf(paste0(base, ".pdf"), width = w, height = h, family = "Arial")
print(fig); dev.off()
ragg::agg_png(paste0(base, ".png"), width = w, height = h, units = "in", res = 600)
print(fig); dev.off()

cat("saved", paste0(base, ".svg / .pdf / .png"), "\n")
