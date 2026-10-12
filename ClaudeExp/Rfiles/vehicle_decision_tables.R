# One table per vehicle: mean Decision with standard deviation underneath.
# Run from the workspace root or the Rfiles directory.
# Output: vehicle_decision_tables.qmd beside descriptive_statistics.csv.

input_file <- file.path("Rfiles", "descriptive_statistics.csv")
if (!file.exists(input_file) && file.exists("descriptive_statistics.csv")) {
  input_file <- "descriptive_statistics.csv"
}
if (!file.exists(input_file)) {
  stop("CSV file not found: ", input_file)
}

dat <- read.csv(input_file, stringsAsFactors = FALSE)
group_columns <- c("model", "vehicle", "judge_role", "rule_given")
required_columns <- c(group_columns, "Decision_mean", "Decision_variance")
if (!all(required_columns %in% names(dat))) {
  stop("Missing required columns: ",
       paste(setdiff(required_columns, names(dat)), collapse = ", "))
}
if (anyDuplicated(dat[group_columns])) {
  stop("Expected one observation per model x vehicle x judge x rule cell.")
}
if (anyNA(dat[group_columns])) {
  stop("Grouping columns must not contain missing values.")
}
if (!all(dat$judge_role %in% c("False", "True")) ||
    !all(dat$rule_given %in% c("False", "True"))) {
  stop("Judge and rule values must be False or True.")
}
if (any(dat$Decision_variance < 0, na.rm = TRUE)) {
  stop("Decision variances must be nonnegative.")
}

conditions <- data.frame(
  judge_role = c("False", "False", "True", "True"),
  rule_given = c("False", "True", "False", "True"),
  label = c(
    "Judge: No / Rule: No", "Judge: No / Rule: Yes",
    "Judge: Yes / Rule: No", "Judge: Yes / Rule: Yes"
  ),
  stringsAsFactors = FALSE
)
models <- sort(unique(dat$model))
vehicles <- sort(unique(dat$vehicle))
digits <- 3L

format_statistic <- function(value, parentheses = FALSE) {
  text <- if (is.na(value)) "NA" else sprintf("%.*f", digits, value)
  if (parentheses) paste0("(", text, ")") else text
}

make_vehicle_table <- function(vehicle_name) {
  vehicle_data <- dat[dat$vehicle == vehicle_name, , drop = FALSE]
  entries <- matrix("", nrow = 2L * length(models), ncol = 5L)
  colnames(entries) <- c("Model", conditions$label)

  for (i in seq_along(models)) {
    mean_row <- 2L * i - 1L
    entries[mean_row, 1L] <- models[i]
    for (j in seq_len(nrow(conditions))) {
      cell <- vehicle_data[
        vehicle_data$model == models[i] &
          vehicle_data$judge_role == conditions$judge_role[j] &
          vehicle_data$rule_given == conditions$rule_given[j],
        , drop = FALSE
      ]
      cell_mean <- if (nrow(cell) == 0L) NA_real_ else cell$Decision_mean
      cell_sd <- if (nrow(cell) == 0L) NA_real_ else sqrt(cell$Decision_variance)
      entries[mean_row, j + 1L] <- format_statistic(cell_mean)
      entries[mean_row + 1L, j + 1L] <- format_statistic(cell_sd, parentheses = TRUE)
    }
  }
  as.data.frame(entries, stringsAsFactors = FALSE, check.names = FALSE)
}

vehicle_tables <- setNames(lapply(vehicles, make_vehicle_table), vehicles)
table_sections <- lapply(vehicles, function(vehicle_name) {
  c(
    paste0("## Vehicle: ", vehicle_name),
    "",
    as.character(knitr::kable(
      vehicle_tables[[vehicle_name]], format = "pipe",
      row.names = FALSE, align = c("l", "r", "r", "r", "r"),
      caption = paste0("Mean Decision by model and judge x rule: ", vehicle_name)
    )),
    ""
  )
})

output_file <- file.path(dirname(input_file), "vehicle_decision_tables.qmd")
quarto_document <- c(
  "---",
  'title: "Decision by Vehicle, Model, and Judge x Rule"',
  "format: html",
  "---",
  "",
  "Each model occupies a mean row followed by its standard deviations in parentheses.",
  "Standard deviations are the square roots of the within-cell Decision variances; they are not standard errors.",
  "Judge and Rule correspond to judge_role and rule_given (No = False; Yes = True).",
  "Values are rounded to three decimal places. NA denotes a missing statistic or condition.",
  "",
  unlist(table_sections, use.names = FALSE)
)
writeLines(quarto_document, output_file, useBytes = TRUE)

vehicle_tables