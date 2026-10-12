# Import the combined CLeF experiment trial data and save it as RData.
# Run from the workspace root.

input_file <- file.path(
  "ClaudeExp",
  "results",
  "trials_20261003_170319_to_20261004_062339_combined.csv"
)
output_file <- file.path("Rfiles", "trials_combined.RData")

if (!file.exists(input_file)) {
  stop("CSV file not found: ", normalizePath(input_file, mustWork = FALSE))
}

trials <- read.csv(
  input_file,
  stringsAsFactors = FALSE,
  na.strings = ""
)

trials$provider <- NULL
trials <- trials[trials$vehicle != "airplane", , drop = FALSE]
trials$trial <- NULL

first_line <- trimws(sub("[\r\n].*$", "", trials$response))
trials$Decision <- ifelse(
  grepl("^yes\\b", first_line, ignore.case = TRUE, perl = TRUE),
  1L,
  ifelse(
    grepl("^no\\b", first_line, ignore.case = TRUE, perl = TRUE),
    -1L,
    ifelse(
      grepl("^(uncertain|undecided)\\b", first_line, ignore.case = TRUE, perl = TRUE),
      0L,
      0L
    )
  )
)

response_details <- trimws(sub("^[^\r\n]*[\r\n]+", "", trials$response, perl = TRUE))
probability_match <- regexpr(
  "[0-9]+(?:\\.[0-9]+)?\\s*%?",
  response_details,
  perl = TRUE
)
raw_probability <- rep(NA_real_, nrow(trials))
has_probability <- !is.na(probability_match) & probability_match > 0
raw_probability[has_probability] <- as.numeric(
  sub("%", "", regmatches(response_details, probability_match)[has_probability], fixed = TRUE)
)
percentage_probability <- !is.na(raw_probability) & raw_probability > 1
raw_probability[percentage_probability] <- raw_probability[percentage_probability] / 100

trials$Probs <- ifelse(
  trials$Decision == 1L,
  raw_probability,
  ifelse(trials$Decision == -1L, 1 - raw_probability, NA_real_)
)

save(trials, file = output_file)
message("Saved ", nrow(trials), " trials to ", output_file)
