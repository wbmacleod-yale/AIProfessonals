#!/usr/bin/env Rscript
# ==============================================================================
# AI Decisions Experiment: Visualization of Results
#
# Generates publication-ready segmented bar graphs for AI vehicle classification
# and H.L.A. Hart mentions across 4 experimental conditions.
#
# Outputs:
#   1. A multi-page PDF document in results/figures/ containing 8 figures
#      alternating between Decision and Mentions Hart across 4 conditions.
#   2. Individual high-resolution PNG image files for each of the 8 figures.
#
# Usage:
#   Rscript plot_figures.R
#   Rscript plot_figures.R results/trials_20261003_131733.csv
# ==============================================================================

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(scales)
})

# ------------------------------------------------------------------------------
# 1. Configuration & Robust Path Resolution
# ------------------------------------------------------------------------------
# Automatically locate the AI_Decisions directory whether run via Rscript,
# command line, or interactively via source(...)
find_base_dir <- function() {
  # 1. When sourced interactively via source("path/to/plot_figures.R")
  for (frame in rev(sys.frames())) {
    if (!is.null(frame$ofile)) {
      candidate <- dirname(normalizePath(frame$ofile, mustWork = FALSE))
      if (dir.exists(file.path(candidate, "results"))) return(candidate)
    }
  }
  # 2. When run via Rscript
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd_args, value = TRUE)
  if (length(file_arg) > 0) {
    candidate <- dirname(normalizePath(sub("^--file=", "", file_arg[1]), mustWork = FALSE))
    if (dir.exists(file.path(candidate, "results"))) return(candidate)
  }
  # 3. Check known workspace candidates relative to current working directory
  candidates <- c(
    getwd(),
    file.path(getwd(), "RAs", "Sun", "AI_Decisions"),
    "/Users/williamsun/Developer/econ3304/values_economics_law/RAs/Sun/AI_Decisions"
  )
  for (cand in candidates) {
    if (dir.exists(file.path(cand, "results"))) return(cand)
  }
  return(getwd())
}

base_dir <- find_base_dir()
results_dir <- file.path(base_dir, "results")
output_dir <- file.path(results_dir, "figures")

args <- commandArgs(trailingOnly = TRUE)
default_csv <- file.path(results_dir, "trials_20261003_131733.csv")

if (length(args) >= 1) {
  if (file.exists(args[1])) {
    csv_file <- args[1]
  } else if (file.exists(file.path(results_dir, args[1]))) {
    csv_file <- file.path(results_dir, args[1])
  } else {
    csv_file <- args[1]
  }
} else if (file.exists(default_csv)) {
  csv_file <- default_csv
} else {
  # Auto-detect latest CSV in results/
  csv_files <- list.files(results_dir, pattern = "^trials_.*\\.csv$", full.names = TRUE)
  if (length(csv_files) > 0) {
    csv_file <- sort(csv_files, decreasing = TRUE)[1]
  } else {
    stop(sprintf("No trial CSV file found in '%s' directory.", results_dir))
  }
}

cat("=================================================================\n")
cat(" AI Decisions: Generating Figures\n")
cat(" Base Dir  :", base_dir, "\n")
cat(" Input CSV :", csv_file, "\n")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}
cat(" Output Dir:", output_dir, "\n")
cat("=================================================================\n\n")

# Canonical order of objects (Title Cased for display)
CANONICAL_OBJECTS <- c("Airplane", "Bicycle", "Car", "Scooter", "Stroller")

# Canonical factor levels for Decision
DECISION_LEVELS <- c("Vehicle (+1)", "Inconclusive (0)", "Not a Vehicle (-1)")

# Color Palettes
# Decision: Forest Green (+1), Amber Gold (0), Crimson Red (-1)
DECISION_COLORS <- c(
  "Vehicle (+1)"       = "#2E7D32",
  "Inconclusive (0)"   = "#F57F17",
  "Not a Vehicle (-1)" = "#C62828"
)

# Canonical factor levels for Mentions Hart
HART_LEVELS <- c("Mentions Hart", "Does Not Mention Hart")

# Mentions Hart: Royal Blue (True), Muted Silver/Slate (False)
HART_COLORS <- c(
  "Mentions Hart"          = "#1565C0",
  "Does Not Mention Hart"  = "#CFD8DC"
)

# ------------------------------------------------------------------------------
# 2. Data Loading & Sanitization
# ------------------------------------------------------------------------------
raw_data <- suppressMessages(read_csv(csv_file, show_col_types = FALSE))
cat("Total rows read in CSV:", nrow(raw_data), "\n")

# Filter out empty or unclassified rows (e.g. API quota errors or in-progress)
valid_data <- raw_data %>%
  filter(!is.na(decision), decision != "") %>%
  mutate(
    judge_role = as.logical(judge_role),
    rule_given = as.logical(rule_given),
    mentions_hart = as.logical(mentions_hart),
    decision_val = as.numeric(decision),
    object_clean = tools::toTitleCase(tolower(trimws(as.character(object))))
  )

cat("Valid completed trials :", nrow(valid_data), "\n")

# Determine models present in the data
available_models <- sort(unique(raw_data$model[!is.na(raw_data$model) & raw_data$model != ""]))
if (length(available_models) == 0) {
  available_models <- c("gemini-3.5-flash", "gemini-3.7-flash")
}
cat("Models detected (", length(available_models), "):", paste(available_models, collapse = ", "), "\n\n")

# Categorize decision and mentions_hart
valid_data <- valid_data %>%
  mutate(
    decision_cat = factor(
      case_when(
        decision_val == 1  ~ "Vehicle (+1)",
        decision_val == -1 ~ "Not a Vehicle (-1)",
        TRUE               ~ "Inconclusive (0)"
      ),
      levels = DECISION_LEVELS
    ),
    hart_cat = factor(
      ifelse(mentions_hart, "Mentions Hart", "Does Not Mention Hart"),
      levels = HART_LEVELS
    ),
    object_fct = factor(object_clean, levels = CANONICAL_OBJECTS),
    model_fct = factor(model, levels = available_models)
  )

# Base grid to ensure every figure has the exact same x-axis and facets
full_grid <- expand_grid(
  object_fct = factor(CANONICAL_OBJECTS, levels = CANONICAL_OBJECTS),
  model_fct = factor(available_models, levels = available_models)
)

# ------------------------------------------------------------------------------
# 3. Figure Definition & Builder Function
# ------------------------------------------------------------------------------

# Define the 4 experimental conditions
CONDITIONS <- list(
  list(
    id = 1,
    slug = "no_judge_no_rule",
    name = "No Judge Role, No Rule Given",
    judge = FALSE,
    rule = FALSE,
    prompt_text = 'Prompt: "Is a {object} a vehicle?"'
  ),
  list(
    id = 2,
    slug = "no_judge_rule",
    name = "No Judge Role, Rule Given",
    judge = FALSE,
    rule = TRUE,
    prompt_text = 'Prompt: "There is a rule that says: no vehicles in the park. Is a {object} a vehicle?"'
  ),
  list(
    id = 3,
    slug = "judge_no_rule",
    name = "Judge Role, No Rule Given",
    judge = TRUE,
    rule = FALSE,
    prompt_text = 'Prompt: "Imagine that you are a judge. Is a {object} a vehicle?"'
  ),
  list(
    id = 4,
    slug = "judge_rule",
    name = "Judge Role, Rule Given",
    judge = TRUE,
    rule = TRUE,
    prompt_text = 'Prompt: "Imagine that you are a judge. There is a rule that says: no vehicles in the park. Is a {object} a vehicle?"'
  )
)

create_segmented_barplot <- function(cond, var_type = c("decision", "mentions_hart")) {
  var_type <- match.arg(var_type)
  
  # Filter data for this condition
  cond_data <- valid_data %>%
    filter(judge_role == cond$judge, rule_given == cond$rule)
  
  totals <- cond_data %>%
    count(object_fct, model_fct, name = "total_n")
  
  # Text labels for sample sizes (N=...)
  n_labels <- full_grid %>%
    left_join(totals, by = c("object_fct", "model_fct")) %>%
    mutate(
      total_n = replace_na(total_n, 0),
      label_text = ifelse(total_n > 0, paste0("N=", total_n), "N=0")
    )
  
  if (var_type == "decision") {
    counts <- cond_data %>%
      count(object_fct, model_fct, decision_cat, name = "n")
    
    # Complete grid across all categories so legend key is fully populated
    plot_df <- full_grid %>%
      left_join(totals, by = c("object_fct", "model_fct")) %>%
      mutate(total_n = replace_na(total_n, 0)) %>%
      tidyr::crossing(decision_cat = factor(DECISION_LEVELS, levels = DECISION_LEVELS)) %>%
      left_join(counts, by = c("object_fct", "model_fct", "decision_cat")) %>%
      mutate(
        n = replace_na(n, 0),
        prop = ifelse(total_n > 0, n / total_n, 0)
      )
    
    fill_scale <- scale_fill_manual(
      name = "Decision Key:",
      values = DECISION_COLORS,
      limits = DECISION_LEVELS,
      drop = FALSE
    )
    
    var_title <- "Decision"
    y_label <- "Percentage of Decisions"
    
  } else {
    counts <- cond_data %>%
      count(object_fct, model_fct, hart_cat, name = "n")
    
    # Complete grid across all categories so legend key is fully populated
    plot_df <- full_grid %>%
      left_join(totals, by = c("object_fct", "model_fct")) %>%
      mutate(total_n = replace_na(total_n, 0)) %>%
      tidyr::crossing(hart_cat = factor(HART_LEVELS, levels = HART_LEVELS)) %>%
      left_join(counts, by = c("object_fct", "model_fct", "hart_cat")) %>%
      mutate(
        n = replace_na(n, 0),
        prop = ifelse(total_n > 0, n / total_n, 0)
      )
    
    fill_scale <- scale_fill_manual(
      name = "Mentions Hart Key:",
      values = HART_COLORS,
      limits = HART_LEVELS,
      drop = FALSE
    )
    
    var_title <- "Mentions of H.L.A. Hart"
    y_label <- "Percentage of Responses"
  }
  
  # Build ggplot
  p <- ggplot()
  
  # Add stacked bars
  if (var_type == "decision") {
    p <- p + geom_col(
      data = plot_df,
      aes(x = model_fct, y = prop, fill = decision_cat),
      position = position_stack(reverse = FALSE),
      width = 0.65,
      color = "black",
      linewidth = 0.35
    )
  } else {
    p <- p + geom_col(
      data = plot_df,
      aes(x = model_fct, y = prop, fill = hart_cat),
      position = position_stack(reverse = FALSE),
      width = 0.65,
      color = "black",
      linewidth = 0.35
    )
  }
  
  # Add sample size (N=...) annotations above each bar position
  p <- p + geom_text(
    data = n_labels,
    aes(x = model_fct, y = 1.04, label = label_text),
    size = 2.8,
    color = "grey40",
    fontface = "bold"
  )
  
  # Facet by Object so bars of the same object are grouped together
  p <- p +
    facet_grid(~ object_fct, drop = FALSE) +
    scale_x_discrete(drop = FALSE) +
    scale_y_continuous(
      labels = scales::percent_format(accuracy = 1),
      breaks = seq(0, 1, 0.25),
      limits = c(0, 1.10),
      expand = c(0, 0)
    ) +
    fill_scale +
    labs(
      title = paste0("Possibility ", cond$id, ": ", cond$name),
      subtitle = paste0(var_title, " by Model & Object  |  ", cond$prompt_text),
      x = "Evaluated Object & Model",
      y = y_label
    ) +
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 13, margin = margin(b = 4)),
      plot.subtitle = element_text(size = 9.5, color = "grey30", margin = margin(b = 10)),
      axis.title.x = element_text(face = "bold", size = 10.5, margin = margin(t = 8)),
      axis.title.y = element_text(face = "bold", size = 10.5, margin = margin(r = 8)),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 8.5, color = "black"),
      axis.text.y = element_text(size = 8.5, color = "black"),
      strip.background = element_rect(fill = "#ECEFF1", color = "grey70", linewidth = 0.6),
      strip.text = element_text(face = "bold", size = 11, color = "#263238"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_line(color = "grey88", linetype = "dashed"),
      legend.position = "bottom",
      legend.box = "horizontal",
      legend.title = element_text(face = "bold", size = 9.5),
      legend.text = element_text(size = 9),
      legend.key = element_rect(color = "black", linewidth = 0.4),
      legend.key.size = unit(0.45, "cm"),
      plot.margin = margin(t = 12, r = 16, b = 10, l = 14)
    )
  
  return(p)
}

# ------------------------------------------------------------------------------
# 4. Generate Figures Alternating Decision and Mentions Hart
# ------------------------------------------------------------------------------
figure_list <- list()
file_names <- character()

# Build the 8 figures across the 4 possibilities
fig_idx <- 1
for (cond in CONDITIONS) {
  # 1. Decision Figure
  p_dec <- create_segmented_barplot(cond, var_type = "decision")
  figure_list[[fig_idx]] <- p_dec
  file_names[fig_idx] <- sprintf("%02d_%s_decision.png", fig_idx, cond$slug)
  fig_idx <- fig_idx + 1
  
  # 2. Mentions Hart Figure
  p_hart <- create_segmented_barplot(cond, var_type = "mentions_hart")
  figure_list[[fig_idx]] <- p_hart
  file_names[fig_idx] <- sprintf("%02d_%s_mentions_hart.png", fig_idx, cond$slug)
  fig_idx <- fig_idx + 1
}

# ------------------------------------------------------------------------------
# 5. Save Outputs: Multi-Page PDF Document & Individual PNG Images
# ------------------------------------------------------------------------------

# 5.1 Multi-Page PDF Document
pdf_path <- file.path(output_dir, "ai_decisions_figures.pdf")
cat("Saving multi-page PDF document to:\n  ->", pdf_path, "\n")

pdf(pdf_path, width = 10.5, height = 6.0, onefile = TRUE)
for (i in seq_along(figure_list)) {
  print(figure_list[[i]])
}
invisible(dev.off())

# 5.2 Individual High-Resolution PNG Images
cat("\nSaving individual PNG images (300 DPI):\n")
for (i in seq_along(figure_list)) {
  img_path <- file.path(output_dir, file_names[i])
  ggsave(
    filename = img_path,
    plot = figure_list[[i]],
    width = 10.5,
    height = 6.0,
    dpi = 300
  )
  cat(sprintf("  [%d/8] Saved %s\n", i, file_names[i]))
}

cat("\n=================================================================\n")
cat(" Successfully generated all 8 figures!\n")
cat(" Multi-page PDF : ", pdf_path, "\n")
cat(" Images folder  : ", output_dir, "\n")
cat("=================================================================\n")
