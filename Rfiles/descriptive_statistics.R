# Summarize Decision and Probs by the first four category columns in trials.
# Run from the workspace root.

input_file <- file.path("Rfiles", "trials_combined.RData")
output_file <- file.path("Rfiles", "descriptive_statistics.csv")

if (!file.exists(input_file)) {
  stop("RData file not found: ", normalizePath(input_file, mustWork = FALSE))
}

load(input_file)

category_columns <- names(trials)[1:4]
expected_categories <- c("model", "vehicle", "judge_role", "rule_given")
if (!identical(category_columns, expected_categories)) {
  stop("The first four columns must be: ", paste(expected_categories, collapse = ", "))
}

summarize_variable <- function(values) {
  non_missing <- values[!is.na(values)]
  c(
    n = length(non_missing),
    mean = if (length(non_missing) > 0) mean(non_missing) else NA_real_,
    variance = if (length(non_missing) > 1) var(non_missing) else NA_real_
  )
}

group_ids <- interaction(trials[category_columns], drop = TRUE, lex.order = TRUE)
group_rows <- split(seq_len(nrow(trials)), group_ids)

descriptive_statistics <- do.call(
  rbind,
  lapply(group_rows, function(row_ids) {
    decision_stats <- summarize_variable(trials$Decision[row_ids])
    probability_stats <- summarize_variable(trials$Probs[row_ids])

    data.frame(
      trials[row_ids[1], category_columns, drop = FALSE],
      Decision_n = decision_stats[["n"]],
      Decision_mean = decision_stats[["mean"]],
      Decision_variance = decision_stats[["variance"]],
      Probs_n = probability_stats[["n"]],
      Probs_mean = probability_stats[["mean"]],
      Probs_variance = probability_stats[["variance"]],
      row.names = NULL,
      check.names = FALSE
    )
  })
)

write.csv(descriptive_statistics, output_file, row.names = FALSE)
print(descriptive_statistics)
message("Saved descriptive statistics to ", output_file)
