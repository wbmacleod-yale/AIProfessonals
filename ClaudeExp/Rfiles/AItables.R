# Summary statistics and crosstabs for descriptive_statistics.csv.
# Run from the workspace root: Rscript Rfiles/AItables.R
# Also writes plain Markdown tables to AItables-copy.qmd beside the CSV.

input_file <- file.path("Rfiles", "descriptive_statistics.csv")
if (!file.exists(input_file) && file.exists("descriptive_statistics.csv")) {
  input_file <- "descriptive_statistics.csv"
}

if (!file.exists(input_file)) {
  stop("CSV file not found: ", normalizePath(input_file, mustWork = FALSE))
}

dat <- read.csv(input_file, stringsAsFactors = FALSE)

# Treat the first four columns as grouping factors.
factor_cols <- c("model", "vehicle", "judge_role", "rule_given")
dat[factor_cols] <- lapply(dat[factor_cols], factor)

# ---- Overall summary statistics ----------------------------------------
numeric_cols <- c(
  "Decision_mean", "Decision_variance",
  "Probs_mean", "Probs_variance"
)

overall_summary <- do.call(
  rbind,
  lapply(numeric_cols, function(col) {
    values <- dat[[col]]
    data.frame(
      variable = col,
      n = sum(!is.na(values)),
      mean = mean(values, na.rm = TRUE),
      sd = sd(values, na.rm = TRUE),
      min = min(values, na.rm = TRUE),
      median = median(values, na.rm = TRUE),
      max = max(values, na.rm = TRUE),
      row.names = NULL
    )
  })
)

# ---- Summary of Decision_mean by model ---------------------------------
decision_by_model <- aggregate(
  Decision_mean ~ model,
  data = dat,
  FUN = function(x) c(mean = mean(x), sd = sd(x), min = min(x), max = max(x))
)
decision_by_model <- do.call(data.frame, decision_by_model)
names(decision_by_model) <- c("model", "mean", "sd", "min", "max")

# ---- Crosstabs of mean Decision ----------------------------------------
# Cell means of Decision_mean averaged over the remaining factors.
ct_model_vehicle <- tapply(
  dat$Decision_mean, list(dat$model, dat$vehicle), mean
)

ct_role_rule <- tapply(
  dat$Decision_mean, list(dat$judge_role, dat$rule_given), mean
)

ct_model_rule <- tapply(
  dat$Decision_mean, list(dat$model, dat$rule_given), mean
)

# ---- Crosstab of mean Probs --------------------------------------------
ct_probs_model_vehicle <- tapply(
  dat$Probs_mean, list(dat$model, dat$vehicle), mean, na.rm = TRUE
)

tables <- list(
  overall_summary = overall_summary,
  decision_by_model = decision_by_model,
  ct_model_vehicle = ct_model_vehicle,
  ct_role_rule = ct_role_rule,
  ct_model_rule = ct_model_rule,
  ct_probs_model_vehicle = ct_probs_model_vehicle
)

# ---- Copy-ready Quarto document (no executable code chunks) -------------
table_section <- function(heading, table, caption, row.names = FALSE) {
  c(
    heading,
    "",
    as.character(knitr::kable(
      table, format = "pipe", digits = 3,
      row.names = row.names, caption = caption
    )),
    ""
  )
}

output_file <- file.path(dirname(input_file), "AItables-copy.qmd")
quarto_document <- c(
  "---",
  'title: "AI Decisions: Summary Statistics and Crosstabs"',
  "format: html",
  "---",
  "",
  sprintf(
    "The data contain %d cells defined by model, vehicle, judge role, and rule given.",
    nrow(dat)
  ),
  "These tables summarize pre-aggregated cell means and variances, not individual trials.",
  "Means across conditions give each cell equal weight; missing probability means are excluded.",
  "",
  table_section(
    "## Overall summary statistics", overall_summary,
    "Summary of cell-level means and variances"
  ),
  table_section(
    "## Decision by model", decision_by_model,
    "Mean Decision by model (across all conditions)"
  ),
  "## Crosstabs of mean Decision",
  "",
  table_section(
    "### Model x Vehicle", ct_model_vehicle,
    "Mean Decision: model x vehicle", row.names = TRUE
  ),
  table_section(
    "### Judge role x Rule given", ct_role_rule,
    "Mean Decision: judge_role x rule_given", row.names = TRUE
  ),
  table_section(
    "### Model x Rule given", ct_model_rule,
    "Mean Decision: model x rule_given", row.names = TRUE
  ),
  "## Crosstab of mean Probs",
  "",
  table_section(
    "### Model x Vehicle", ct_probs_model_vehicle,
    "Mean Probs: model x vehicle", row.names = TRUE
  )
)
writeLines(quarto_document, output_file, useBytes = TRUE)

tables
