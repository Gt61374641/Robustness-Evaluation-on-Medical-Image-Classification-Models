# H2 AT-ladder figure (full 5-model complexity ladder, 3 datasets) -- R backend.
#
# ggplot2 + patchwork twin panels, same data & scientific claim as the Python
# version (scripts/generate_at_ladder_figure.py). Reads the shared data file:
#   figures/data/at_ladder_h2.json
#
#   (a) trend : PGD@8 full robust accuracy vs model capacity (params, log10 x),
#       one line per dataset; collapsed points drawn hollow with a red x.
#   (b) bars  : same metric, model x dataset grouped bars; collapsed marked
#       with a red x at the bar base (visible even at ~0 height).
#
# Run:  Rscript scripts/generate_at_ladder_figure.R
# Needs: jsonlite, ggplot2, patchwork, svglite, ragg  (see install note below).
#   install.packages(c("jsonlite","ggplot2","patchwork","svglite","ragg"))

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

# ── Nature-style theme (from nature-figure skill R quick-start) ─────────────
theme_set(
  theme_classic(base_size = 6.5, base_family = "Arial") +
    theme(
      axis.line   = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks  = element_line(linewidth = 0.35, colour = "black"),
      legend.title = element_blank(),
      legend.text = element_text(size = 5.8),
      legend.key.size = unit(3, "mm"),
      legend.position = c(0.02, 0.98),
      legend.justification = c(0, 1),
      plot.title  = element_text(size = 7, face = "bold"),
      plot.tag    = element_text(size = 10, face = "bold"),
      panel.grid  = element_blank()
    )
)

DATASET_COLORS <- c("Chest X-ray" = "#0F4D92", "Malaria" = "#42949E", "OCT" = "#9A4D8E")
COLLAPSE_RED <- "#B64342"
DISPLAY <- c(resnet18 = "ResNet-18", resnet34 = "ResNet-34", resnet50 = "ResNet-50",
             resnet101 = "ResNet-101", resnet152 = "ResNet-152")

# ── Data ────────────────────────────────────────────────────────────────────
data <- fromJSON(file.path(proj_root, "figures", "data", "at_ladder_h2.json"))
df <- data$rows
df$dataset_display <- factor(df$dataset_display, levels = names(DATASET_COLORS))
df$model <- factor(df$model, levels = names(DISPLAY))
df$model_label <- factor(DISPLAY[as.character(df$model)], levels = DISPLAY)
df$state <- ifelse(df$collapsed, "collapsed", "trained")
# multi-seed error bars: only on trained points with >1 seed and non-zero spread.
# Kept on the FULL df (NA elsewhere) so position_dodge sees every dataset at every
# model and the bars stay aligned with the columns.
has_err <- !df$collapsed & df$n_seeds > 1 & df$robust8_std > 0
df$err_lo <- ifelse(has_err, pmax(0, df$robust8 - df$robust8_std), NA_real_)
df$err_hi <- ifelse(has_err, df$robust8 + df$robust8_std, NA_real_)

# ── Panel a: capacity trend ─────────────────────────────────────────────────
pa <- ggplot(df, aes(x = params_m, y = robust8, colour = dataset_display)) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = -0.03, ymax = 0.02,
           fill = COLLAPSE_RED, alpha = 0.06) +
  geom_line(aes(group = dataset_display), linewidth = 0.6) +
  # trained points: filled; collapsed: hollow (white fill) + red x overlay
  geom_point(data = subset(df, !collapsed), size = 1.8) +
  geom_point(data = subset(df, collapsed), size = 1.8, shape = 21, fill = "white",
             stroke = 0.5) +
  geom_point(data = subset(df, collapsed), shape = 4, colour = COLLAPSE_RED,
             size = 1.3, stroke = 0.6) +
  scale_colour_manual(values = DATASET_COLORS) +
  scale_x_log10(breaks = sort(unique(df$params_m)),
                labels = sprintf("%.0f", sort(unique(df$params_m)))) +
  coord_cartesian(ylim = c(-0.03, 1.0)) +
  labs(x = "Model capacity (parameters, M)",
       y = "PGD-8/255 robust accuracy (full)", tag = "a")

# ── Panel b: grouped bars ───────────────────────────────────────────────────
dodge <- position_dodge(width = 0.8)
pb <- ggplot(df, aes(x = model_label, y = robust8, fill = dataset_display)) +
  geom_col(aes(colour = dataset_display), position = dodge, width = 0.75,
           linewidth = 0.3) +
  # collapsed bars: overpaint white fill so only the coloured outline shows
  geom_col(data = subset(df, collapsed), fill = "white",
           aes(colour = dataset_display), position = dodge, width = 0.75,
           linewidth = 0.3) +
  # multi-seed error bars on trained points (NA rows drop out silently)
  geom_errorbar(aes(ymin = err_lo, ymax = err_hi, group = dataset_display),
                position = dodge, width = 0.35, linewidth = 0.35,
                colour = "grey20", na.rm = TRUE, show.legend = FALSE) +
  # red x at the base of every collapsed bar
  geom_point(data = subset(df, collapsed), aes(y = 0.022),
             position = dodge, shape = 4, colour = COLLAPSE_RED,
             size = 1.2, stroke = 0.6, show.legend = FALSE) +
  scale_fill_manual(values = DATASET_COLORS) +
  scale_colour_manual(values = DATASET_COLORS, guide = "none") +
  coord_cartesian(ylim = c(0, 1.0)) +
  labs(x = NULL, y = "PGD-8/255 robust accuracy (full)", tag = "b") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

fig <- pa + pb + plot_layout(ncol = 2)

# ── Export (svglite / cairo_pdf / ragg, per skill R quick-start) ────────────
out_dir <- file.path(proj_root, "figures", "at_ladder")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
base <- file.path(out_dir, "H2_at_ladder_r")
w <- 183 / 25.4; h <- 76 / 25.4   # double-column width, ~76 mm tall

svglite::svglite(paste0(base, ".svg"), width = w, height = h)
print(fig); dev.off()
grDevices::cairo_pdf(paste0(base, ".pdf"), width = w, height = h, family = "Arial")
print(fig); dev.off()
ragg::agg_png(paste0(base, ".png"), width = w, height = h, units = "in", res = 600)
print(fig); dev.off()

cat("saved", paste0(base, ".svg / .pdf / .png"), "\n")
