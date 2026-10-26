#!/usr/bin/env Rscript
# ==============================================================================
# AI Decisions Experiment: Paid vs. Free (PvF) Visualization of Results
#
# Generates publication-ready comparative bar graphs for Gemini 3.5 Flash
# comparing Free vs. Paid accounts across 4 experimental conditions for:
#   1. Classification Decisions (Vehicle +1, Inconclusive 0, Not a Vehicle -1)
#   2. Mentions of H.L.A. Hart (Mentions Hart vs. Does Not Mention Hart)
#   3. Input Token Spend (Mean Input Tokens with SEM error bars)
#   4. Output Token Spend (Mean Output Tokens with SEM error bars)
#   5. Response Latency (Mean Latency in seconds with SEM error bars)
#
# Statistical Significance:
#   Includes significance stars (* p < 0.10, ** p < 0.05, *** p < 0.01) based on
#   an independent two-sample t-test of means between Free and Paid versions
#   within the header of each pair of bar graphs, accompanied by a significance
#   legend at the bottom right of each page.
#
# Outputs:
#   1. A multi-page PDF document in results/figures/ai_figures_pvf.pdf
#      containing 20 figures (5 metrics × 4 conditions).
#   2. Individual high-resolution PNG image files (300 DPI) for each figure.
#
# Usage:
#   1. Run with default dataset (pvf_merged_20261026.csv):
#        Rscript plot_figures_pvf.R
#
#   2. Specify a custom dataset via command line argument (Recommended):
#        Rscript plot_figures_pvf.R results/pvf_merged_20261026.csv
#        Rscript plot_figures_pvf.R pvf_merged_20261026.csv
#        Rscript plot_figures_pvf.R /absolute/path/to/dataset.csv
#
#   3. Or permanently change the default file by editing `default_csv` below.
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
find_base_dir <- function() {
  # 1. When sourced interactively via source("path/to/plot_figures_pvf.R")
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
    file.path(getwd(), "SunW", "AI_Decisions"),
    file.path(getwd(), "AI_Decisions"),
    "/Users/williamsun/Developer/ai_exp/AIProfessionals/SunW/AI_Decisions"
  )
  for (cand in candidates) {
    if (dir.exists(file.path(cand, "results"))) return(cand)
  }
  return(getwd())
}

base_dir <- find_base_dir()
results_dir <- file.path(base_dir, "results")
output_dir <- file.path(results_dir, "figures")

# ------------------------------------------------------------------------------
# Input Dataset Selection
# ------------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
default_csv <- file.path(results_dir, "pvf_merged_20261026.csv")

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
  # Auto-detect latest CSV matching pvf_*.csv or trials_*.csv in results/
  csv_files <- list.files(results_dir, pattern = "^(pvf_|trials_).*\\.csv$", full.names = TRUE)
  # Exclude non-experiment log files
  csv_files <- csv_files[!grepl("dpo", csv_files, ignore.case = TRUE)]
  if (length(csv_files) > 0) {
    csv_file <- sort(csv_files, decreasing = TRUE)[1]
  } else {
    stop(sprintf("No suitable CSV file found in '%s' directory.", results_dir))
  }
}

cat("=================================================================\n")
cat(" AI Decisions: Generating Paid vs. Free (PvF) Figures\n")
cat(" Base Dir  :", base_dir, "\n")
cat(" Input CSV :", csv_file, "\n")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}
cat(" Output Dir:", output_dir, "\n")
cat("=================================================================\n\n")

# Canonical order of objects (Title Cased for display)
CANONICAL_OBJECTS <- c("Airplane", "Bicycle", "Car", "Scooter", "Stroller")

# Canonical factor levels for Plan
PLAN_LEVELS <- c("Free", "Paid")

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

# Plan Colors for Token Spend Plots
# Input Tokens: Medium Teal (Free) vs. Deep Teal (Paid)
INPUT_TOKEN_COLORS <- c(
  "Free" = "#4DB6AC",
  "Paid" = "#00695C"
)

# Output Tokens: Medium Orchid (Free) vs. Deep Violet/Purple (Paid)
OUTPUT_TOKEN_COLORS <- c(
  "Free" = "#BA68C8",
  "Paid" = "#6A1B9A"
)

# Latency: Warm Amber (Free) vs. Deep Burnt Orange (Paid)
LATENCY_COLORS <- c(
  "Free" = "#FFB74D",
  "Paid" = "#E65100"
)

# Significance threshold helper
get_significance_stars <- function(p_val) {
  if (is.na(p_val)) return("")
  if (p_val < 0.01) return("***")
  if (p_val < 0.05) return("**")
  if (p_val < 0.10) return("*")
  return("")
}

# Robust two-sample independent t-test p-value calculator
calc_ttest_p <- function(x, y) {
  x <- x[!is.na(x)]
  y <- y[!is.na(y)]
  if (length(x) < 2 || length(y) < 2) return(NA_real_)
  if (var(x) == 0 && var(y) == 0) {
    if (mean(x) == mean(y)) return(1.0)
    return(NA_real_)
  }
  tryCatch({
    tt <- t.test(x, y)
    tt$p.value
  }, error = function(e) {
    NA_real_
  })
}

# ------------------------------------------------------------------------------
# 2. Data Loading & Sanitization
# ------------------------------------------------------------------------------
raw_data <- suppressMessages(read_csv(csv_file, show_col_types = FALSE))
cat("Total rows read in CSV:", nrow(raw_data), "\n")

# Filter for valid rows and target model: gemini-3.5-flash
valid_data <- raw_data %>%
  filter(!is.na(decision), decision != "") %>%
  filter(tolower(trimws(as.character(model))) == "gemini-3.5-flash") %>%
  mutate(
    judge_role = as.logical(judge_role),
    rule_given = as.logical(rule_given),
    mentions_hart = as.logical(mentions_hart),
    decision_val = as.numeric(decision),
    object_clean = tools::toTitleCase(tolower(trimws(as.character(object)))),
    plan_clean = tools::toTitleCase(tolower(trimws(as.character(plan)))),
    input_tokens = as.numeric(input_tokens),
    output_tokens = as.numeric(output_tokens),
    latency_seconds = as.numeric(latency_seconds)
  )

cat("Valid Gemini 3.5 Flash trials:", nrow(valid_data), "\n")
cat("Breakdown by Plan:\n")
print(table(valid_data$plan_clean))
cat("\n")

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
    plan_fct = factor(plan_clean, levels = PLAN_LEVELS)
  )

# Base grid to ensure every figure has the exact same x-axis and facets
full_grid <- expand_grid(
  object_fct = factor(CANONICAL_OBJECTS, levels = CANONICAL_OBJECTS),
  plan_fct = factor(PLAN_LEVELS, levels = PLAN_LEVELS)
)

# ------------------------------------------------------------------------------
# 3. Figure Definition & Builder Functions
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

# Helper function to compute t-test significance per object in a condition
compute_condition_significance <- function(cond_data, var_name) {
  sig_map <- list()
  for (obj in CANONICAL_OBJECTS) {
    x <- cond_data %>% filter(object_clean == obj, plan_clean == "Free") %>% pull(.data[[var_name]])
    y <- cond_data %>% filter(object_clean == obj, plan_clean == "Paid") %>% pull(.data[[var_name]])
    p_val <- calc_ttest_p(x, y)
    stars <- get_significance_stars(p_val)
    sig_map[[obj]] <- list(p_val = p_val, stars = stars)
  }
  return(sig_map)
}

# Unified Plot Generator for Decision & Mentions Hart (Categorical Proportions)
create_segmented_pvf_barplot <- function(cond, var_type = c("decision", "mentions_hart")) {
  var_type <- match.arg(var_type)

  # Filter data for this condition
  cond_data <- valid_data %>%
    filter(judge_role == cond$judge, rule_given == cond$rule)

  # Statistical significance per object
  var_numeric_name <- if (var_type == "decision") "decision_val" else "mentions_hart"
  sig_info <- compute_condition_significance(cond_data, var_numeric_name)

  # Construct facet header labels incorporating significance stars
  header_labels <- setNames(
    sapply(CANONICAL_OBJECTS, function(obj) {
      st <- sig_info[[obj]]$stars
      if (st != "") paste0(obj, " ", st) else obj
    }),
    CANONICAL_OBJECTS
  )

  totals <- cond_data %>%
    count(object_fct, plan_fct, name = "total_n")

  # Sample size text labels (N=...)
  n_labels <- full_grid %>%
    left_join(totals, by = c("object_fct", "plan_fct")) %>%
    mutate(
      total_n = replace_na(total_n, 0),
      label_text = ifelse(total_n > 0, paste0("N=", total_n), "N=0"),
      facet_header = factor(header_labels[as.character(object_fct)], levels = unname(header_labels))
    )

  # Dataframe for significance stars annotation above pair of bars
  star_annotations <- tibble(
    object_fct = factor(CANONICAL_OBJECTS, levels = CANONICAL_OBJECTS),
    facet_header = factor(unname(header_labels), levels = unname(header_labels)),
    stars = sapply(CANONICAL_OBJECTS, function(obj) sig_info[[obj]]$stars),
    x_pos = 1.5,
    y_pos = 1.09
  ) %>% filter(stars != "")

  if (var_type == "decision") {
    counts <- cond_data %>%
      count(object_fct, plan_fct, decision_cat, name = "n")

    plot_df <- full_grid %>%
      left_join(totals, by = c("object_fct", "plan_fct")) %>%
      mutate(total_n = replace_na(total_n, 0)) %>%
      tidyr::crossing(decision_cat = factor(DECISION_LEVELS, levels = DECISION_LEVELS)) %>%
      left_join(counts, by = c("object_fct", "plan_fct", "decision_cat")) %>%
      mutate(
        n = replace_na(n, 0),
        prop = ifelse(total_n > 0, n / total_n, 0),
        facet_header = factor(header_labels[as.character(object_fct)], levels = unname(header_labels))
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
      count(object_fct, plan_fct, hart_cat, name = "n")

    plot_df <- full_grid %>%
      left_join(totals, by = c("object_fct", "plan_fct")) %>%
      mutate(total_n = replace_na(total_n, 0)) %>%
      tidyr::crossing(hart_cat = factor(HART_LEVELS, levels = HART_LEVELS)) %>%
      left_join(counts, by = c("object_fct", "plan_fct", "hart_cat")) %>%
      mutate(
        n = replace_na(n, 0),
        prop = ifelse(total_n > 0, n / total_n, 0),
        facet_header = factor(header_labels[as.character(object_fct)], levels = unname(header_labels))
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

  p <- ggplot()

  # Stacked bars
  if (var_type == "decision") {
    p <- p + geom_col(
      data = plot_df,
      aes(x = plan_fct, y = prop, fill = decision_cat),
      position = position_stack(reverse = FALSE),
      width = 0.65,
      color = "black",
      linewidth = 0.35
    )
  } else {
    p <- p + geom_col(
      data = plot_df,
      aes(x = plan_fct, y = prop, fill = hart_cat),
      position = position_stack(reverse = FALSE),
      width = 0.65,
      color = "black",
      linewidth = 0.35
    )
  }

  # Sample size (N=...) annotations
  p <- p + geom_text(
    data = n_labels,
    aes(x = plan_fct, y = 1.04, label = label_text),
    size = 2.8,
    color = "grey40",
    fontface = "bold"
  )

  # Significance stars centered above pair of bars
  if (nrow(star_annotations) > 0) {
    p <- p + geom_text(
      data = star_annotations,
      aes(x = x_pos, y = y_pos, label = stars),
      size = 4.2,
      color = "#C62828",
      fontface = "bold"
    )
  }

  # Facet and theme
  p <- p +
    facet_grid(~ facet_header, drop = FALSE) +
    scale_x_discrete(drop = FALSE) +
    scale_y_continuous(
      labels = scales::percent_format(accuracy = 1),
      breaks = seq(0, 1, 0.25),
      limits = c(0, 1.15),
      expand = c(0, 0)
    ) +
    fill_scale +
    labs(
      title = paste0("Possibility ", cond$id, ": ", cond$name),
      subtitle = paste0(var_title, " by Plan (Gemini 3.5 Flash) | ", cond$prompt_text),
      x = "Evaluated Object & Account Plan",
      y = y_label,
      caption = "Significance (t-test):  * p < 0.10   ** p < 0.05   *** p < 0.01"
    ) +
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 13, margin = margin(b = 4)),
      plot.subtitle = element_text(size = 9.2, color = "grey30", margin = margin(b = 10)),
      axis.title.x = element_text(face = "bold", size = 10.5, margin = margin(t = 8)),
      axis.title.y = element_text(face = "bold", size = 10.5, margin = margin(r = 8)),
      axis.text.x = element_text(angle = 0, hjust = 0.5, size = 9.0, face = "bold", color = "black"),
      axis.text.y = element_text(size = 8.5, color = "black"),
      strip.background = element_rect(fill = "#ECEFF1", color = "grey70", linewidth = 0.6),
      strip.text = element_text(face = "bold", size = 10.5, color = "#263238"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_line(color = "grey88", linetype = "dashed"),
      legend.position = "bottom",
      legend.box = "horizontal",
      legend.title = element_text(face = "bold", size = 9.5),
      legend.text = element_text(size = 9),
      legend.key = element_rect(color = "black", linewidth = 0.4),
      legend.key.size = unit(0.45, "cm"),
      plot.caption = element_text(hjust = 1, size = 9.0, face = "bold", color = "grey25", margin = margin(t = 6)),
      plot.margin = margin(t = 12, r = 16, b = 10, l = 14)
    )

  return(p)
}

# Unified Plot Generator for Continuous Numerical Metrics (Input/Output Tokens, Latency Seconds with SEM)
create_continuous_metric_barplot <- function(cond, metric_type = c("input_tokens", "output_tokens", "latency_seconds")) {
  metric_type <- match.arg(metric_type)

  # Filter data for this condition
  cond_data <- valid_data %>%
    filter(judge_role == cond$judge, rule_given == cond$rule)

  # Statistical significance per object
  sig_info <- compute_condition_significance(cond_data, metric_type)

  # Construct facet header labels incorporating significance stars
  header_labels <- setNames(
    sapply(CANONICAL_OBJECTS, function(obj) {
      st <- sig_info[[obj]]$stars
      if (st != "") paste0(obj, " ", st) else obj
    }),
    CANONICAL_OBJECTS
  )

  # Calculate means, standard errors, and sample sizes per object and plan
  stats_df <- cond_data %>%
    filter(!is.na(.data[[metric_type]])) %>%
    group_by(object_fct, plan_fct) %>%
    summarise(
      n = n(),
      mean_val = mean(.data[[metric_type]], na.rm = TRUE),
      sd_val = sd(.data[[metric_type]], na.rm = TRUE),
      se_val = ifelse(n > 1, sd_val / sqrt(n), 0),
      .groups = "drop"
    )

  plot_df <- full_grid %>%
    left_join(stats_df, by = c("object_fct", "plan_fct")) %>%
    mutate(
      n = replace_na(n, 0),
      mean_val = replace_na(mean_val, 0),
      se_val = replace_na(se_val, 0),
      label_text = ifelse(n > 0, paste0("N=", n), "N=0"),
      facet_header = factor(header_labels[as.character(object_fct)], levels = unname(header_labels))
    )

  # Upper limit for y-axis
  max_obs <- max(plot_df$mean_val + plot_df$se_val, na.rm = TRUE)
  default_max <- if (metric_type == "latency_seconds") 15 else 100
  y_max <- if (is.finite(max_obs) && max_obs > 0) max_obs * 1.25 else default_max

  # Dataframe for significance stars annotation above pair of bars
  star_annotations <- tibble(
    object_fct = factor(CANONICAL_OBJECTS, levels = CANONICAL_OBJECTS),
    facet_header = factor(unname(header_labels), levels = unname(header_labels)),
    stars = sapply(CANONICAL_OBJECTS, function(obj) sig_info[[obj]]$stars),
    x_pos = 1.5,
    y_pos = y_max * 0.94
  ) %>% filter(stars != "")

  # Plot properties
  if (metric_type == "input_tokens") {
    var_title <- "Input Token Spend"
    y_label <- "Mean Input Tokens"
    fill_palette <- INPUT_TOKEN_COLORS
    key_title <- "Input Tokens (Plan):"
    val_formatter <- function(x) ifelse(x > 0, sprintf("%.1f", x), "")
  } else if (metric_type == "output_tokens") {
    var_title <- "Output Token Spend"
    y_label <- "Mean Output Tokens"
    fill_palette <- OUTPUT_TOKEN_COLORS
    key_title <- "Output Tokens (Plan):"
    val_formatter <- function(x) ifelse(x > 0, sprintf("%.1f", x), "")
  } else {
    var_title <- "Response Latency"
    y_label <- "Mean Latency (Seconds)"
    fill_palette <- LATENCY_COLORS
    key_title <- "Latency (Plan):"
    val_formatter <- function(x) ifelse(x > 0, sprintf("%.2fs", x), "")
  }

  p <- ggplot(plot_df, aes(x = plan_fct, y = mean_val, fill = plan_fct)) +
    geom_col(
      width = 0.65,
      color = "black",
      linewidth = 0.35
    ) +
    geom_errorbar(
      aes(ymin = pmax(0, mean_val - se_val), ymax = mean_val + se_val),
      width = 0.22,
      linewidth = 0.45,
      color = "grey25"
    ) +
    # Value label on top of bar
    geom_text(
      aes(
        y = mean_val + se_val,
        label = val_formatter(mean_val)
      ),
      vjust = -0.6,
      size = 2.7,
      fontface = "bold",
      color = "black"
    ) +
    # Sample size label at base
    geom_text(
      aes(y = y_max * 0.05, label = label_text),
      size = 2.6,
      color = "grey35",
      fontface = "bold"
    ) +
    facet_grid(~ facet_header, drop = FALSE) +
    scale_fill_manual(
      name = key_title,
      values = fill_palette,
      limits = PLAN_LEVELS,
      drop = FALSE
    ) +
    scale_x_discrete(drop = FALSE) +
    scale_y_continuous(
      limits = c(0, y_max),
      expand = c(0, 0)
    )

  # Significance stars
  if (nrow(star_annotations) > 0) {
    p <- p + geom_text(
      data = star_annotations,
      aes(x = x_pos, y = y_pos, label = stars),
      inherit.aes = FALSE,
      size = 4.4,
      color = "#C62828",
      fontface = "bold"
    )
  }

  p <- p +
    labs(
      title = paste0("Possibility ", cond$id, ": ", cond$name),
      subtitle = paste0(var_title, " by Plan (Gemini 3.5 Flash) | ", cond$prompt_text),
      x = "Evaluated Object & Account Plan",
      y = y_label,
      caption = "Significance (t-test):  * p < 0.10   ** p < 0.05   *** p < 0.01"
    ) +
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 13, margin = margin(b = 4)),
      plot.subtitle = element_text(size = 9.2, color = "grey30", margin = margin(b = 10)),
      axis.title.x = element_text(face = "bold", size = 10.5, margin = margin(t = 8)),
      axis.title.y = element_text(face = "bold", size = 10.5, margin = margin(r = 8)),
      axis.text.x = element_text(angle = 0, hjust = 0.5, size = 9.0, face = "bold", color = "black"),
      axis.text.y = element_text(size = 8.5, color = "black"),
      strip.background = element_rect(fill = "#ECEFF1", color = "grey70", linewidth = 0.6),
      strip.text = element_text(face = "bold", size = 10.5, color = "#263238"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_line(color = "grey88", linetype = "dashed"),
      legend.position = "bottom",
      legend.box = "horizontal",
      legend.title = element_text(face = "bold", size = 9.5),
      legend.text = element_text(size = 9),
      legend.key = element_rect(color = "black", linewidth = 0.4),
      legend.key.size = unit(0.45, "cm"),
      plot.caption = element_text(hjust = 1, size = 9.0, face = "bold", color = "grey25", margin = margin(t = 6)),
      plot.margin = margin(t = 12, r = 16, b = 10, l = 14)
    )

  return(p)
}

# Backwards compatibility alias
create_token_spend_barplot <- create_continuous_metric_barplot

# ------------------------------------------------------------------------------
# 4. Generate 20 Figures (5 Metrics × 4 Conditions)
# ------------------------------------------------------------------------------
figure_list <- list()
file_names <- character()
fig_idx <- 1

cat("Generating figures across 4 experimental conditions...\n")

for (cond in CONDITIONS) {
  cat(sprintf("\n-- Processing Condition %d: %s --\n", cond$id, cond$name))

  # 1. Decision Figure
  p_dec <- create_segmented_pvf_barplot(cond, var_type = "decision")
  figure_list[[fig_idx]] <- p_dec
  file_names[fig_idx] <- sprintf("pvf_%02d_%s_decision.png", fig_idx, cond$slug)
  cat(sprintf("  [%02d/20] Created Decision figure\n", fig_idx))
  fig_idx <- fig_idx + 1

  # 2. Mentions Hart Figure
  p_hart <- create_segmented_pvf_barplot(cond, var_type = "mentions_hart")
  figure_list[[fig_idx]] <- p_hart
  file_names[fig_idx] <- sprintf("pvf_%02d_%s_mentions_hart.png", fig_idx, cond$slug)
  cat(sprintf("  [%02d/20] Created Mentions Hart figure\n", fig_idx))
  fig_idx <- fig_idx + 1

  # 3. Input Tokens Figure
  p_in_tok <- create_continuous_metric_barplot(cond, metric_type = "input_tokens")
  figure_list[[fig_idx]] <- p_in_tok
  file_names[fig_idx] <- sprintf("pvf_%02d_%s_input_tokens.png", fig_idx, cond$slug)
  cat(sprintf("  [%02d/20] Created Input Tokens figure\n", fig_idx))
  fig_idx <- fig_idx + 1

  # 4. Output Tokens Figure
  p_out_tok <- create_continuous_metric_barplot(cond, metric_type = "output_tokens")
  figure_list[[fig_idx]] <- p_out_tok
  file_names[fig_idx] <- sprintf("pvf_%02d_%s_output_tokens.png", fig_idx, cond$slug)
  cat(sprintf("  [%02d/20] Created Output Tokens figure\n", fig_idx))
  fig_idx <- fig_idx + 1

  # 5. Latency Seconds Figure
  p_lat <- create_continuous_metric_barplot(cond, metric_type = "latency_seconds")
  figure_list[[fig_idx]] <- p_lat
  file_names[fig_idx] <- sprintf("pvf_%02d_%s_latency.png", fig_idx, cond$slug)
  cat(sprintf("  [%02d/20] Created Latency figure\n", fig_idx))
  fig_idx <- fig_idx + 1
}

# ------------------------------------------------------------------------------
# 5. Save Outputs: Multi-Page PDF (ai_figures_pvf.pdf) & Individual PNG Images
# ------------------------------------------------------------------------------

# 5.1 Multi-Page PDF Document: ai_figures_pvf.pdf
pdf_path <- file.path(output_dir, "ai_figures_pvf.pdf")
cat("\nSaving multi-page PDF document to:\n  ->", pdf_path, "\n")

pdf(pdf_path, width = 10.5, height = 6.0, onefile = TRUE)
for (i in seq_along(figure_list)) {
  print(figure_list[[i]])
}
invisible(dev.off())

# 5.2 Individual High-Resolution PNG Images (300 DPI)
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
  cat(sprintf("  [%02d/20] Saved %s\n", i, file_names[i]))
}

cat("\n=================================================================\n")
cat(" Successfully generated all 20 Paid vs. Free figures!\n")
cat(" Multi-page PDF : ", pdf_path, "\n")
cat(" Images folder  : ", output_dir, "\n")
cat("=================================================================\n")
