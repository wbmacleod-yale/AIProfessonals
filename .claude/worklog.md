# Worklog - AI Experiments

Experiment-specific history migrated from `values_economics_law` on 2026-10-04.

## 2026-10-26 - Merge PvF dataset (20261026) and add response latency bar graphs

**Context / problem:** Incorporate new trial results from `pvf_20261026_160948.csv` with historical baseline `pvf_merged_20261025.csv` into `pvf_merged_20261026.csv`, retaining only error-free rows. Expand `plot_figures_pvf.R` to visualize response latency (`latency_seconds`) alongside decisions, Hart mentions, input tokens, and output tokens across all 4 experimental conditions.

**Changes made:**
- Created and executed `SunW/AI_Decisions/merge_pvf_20261026.py` to merge `pvf_merged_20261025.csv` (76 valid rows) and `pvf_20261026_160948.csv` (44 valid rows, excluding 156 quota error rows), outputting 120 total error-free records to `SunW/AI_Decisions/results/pvf_merged_20261026.csv`.
- Updated `SunW/AI_Decisions/plot_figures_pvf.R`:
  - Updated default input dataset to `pvf_merged_20261026.csv`.
  - Generalized continuous numerical plotting with `create_continuous_metric_barplot` supporting `input_tokens`, `output_tokens`, and `latency_seconds` with SEM error bars, mean value labels (formatted to 2 decimal places `%.2fs`), sample sizes `N=...`, and Welch t-test significance stars (* p<0.10, ** p<0.05, *** p<0.01).
  - Added dedicated warm amber/deep burnt orange color palette for latency (`Free = #FFB74D`, `Paid = #E65100`).
  - Expanded figure loop to generate 20 figures (5 metrics × 4 conditions) saved into multi-page PDF `SunW/AI_Decisions/results/figures/ai_figures_pvf.pdf` (20 pages) and 300 DPI individual PNGs (`pvf_%02d_...png`).
- Re-ran `Rscript SunW/AI_Decisions/plot_figures_pvf.R SunW/AI_Decisions/results/pvf_merged_20261026.csv`.

**Verification:**
- Verified `pvf_merged_20261026.csv` contains 120 valid rows (76 paid, 44 free; 113 `gemini-3.5-flash`).
- Verified `plot_figures_pvf.R` successfully ran, creating `ai_figures_pvf.pdf` (20 pages) and all 20 individual PNGs without errors.
- Inspected generated latency plots (`pvf_05_no_judge_no_rule_latency.png` and `pvf_10_no_judge_rule_latency.png`) to confirm visual clarity, SEM error bars, formatting, and significance legend.

## 2026-10-26 - Add HF_TOKEN support and revert to persistent cumulative output logs for DPO runner

**Context / problem:** Support authenticated Hugging Face token ingestion for gated model access (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`) and ensure all experiment output files maintain persistent, cumulative logs across sequential iterations without per-run timestamped file fragmentation.

**Changes made:**
- Updated `SunW/AI_Decisions/train_and_evaluate_dpo.py` to add `get_hf_token()` and `_clean_key()` to discover and sanitize tokens from `--hf_token` CLI args, environment variables (`HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HF_AUTH_TOKEN`), or `~/.cache/huggingface/token`.
- Reverted output file generation to persistent, untimestamped paths that append cumulatively:
  - Summary CSV: `SunW/AI_Decisions/results/dpo_llama3_experiment_log.csv` (appends new row per iteration).
  - Granular trials JSONL: `SunW/AI_Decisions/results/dpo_evaluation_trials.jsonl` (appends all trial records, prefixed with iteration id).
  - Misaligned error JSONL: `SunW/AI_Decisions/results/dpo_misaligned_questions.jsonl` (appends failures per iteration).
  - Run configuration JSON: `SunW/AI_Decisions/results/dpo_config.json` (stores latest configuration state).
- Cleaned up transient timestamped test files from previous trial.

**Verification:**
- Executed `python3 SunW/AI_Decisions/train_and_evaluate_dpo.py --iteration 1` and `python3 SunW/AI_Decisions/train_and_evaluate_dpo.py --simulate_baseline --iteration 0`.
- Verified that each execution appends new records to `dpo_llama3_experiment_log.csv` and `dpo_evaluation_trials.jsonl` without overwriting prior iterations.

**Replication notes:**
Run `python3 SunW/AI_Decisions/train_and_evaluate_dpo.py --iteration <N>`. Pass `--hf_token <token>` or set `export HF_TOKEN="hf_..."`. Cumulative logs persist in `SunW/AI_Decisions/results/`.

## 2026-10-25 - Implement DPO alignment dataset generator and evaluation pipeline for moral dilemmas

**Context / problem:** Explore training an AI model at a level deeper than frontend prompting (parameter optimization via DPO) to follow a demographic resource-allocation priority policy (prioritizing female individuals in binary dilemma trade-offs). Ensure training data avoids shortcut learning/memorization (lookup tables), and build an evaluation harness with token profiling and misaligned-question auditing.

**Changes made:**
- Created `SunW/AI_Decisions/generate_dataset.py` generating 228 counterfactually balanced preference pairs (`prompt`, `chosen`, `rejected`) across 5 in-domain environments (rail transit, autonomous vehicles, medical triage, maritime rescue, structural disaster) and 2 held-out test splits (`test_ood.jsonl` with unseen space-station life-support scenarios; `test_adversarial.jsonl` with utility and age confounders).
- Implemented 50/50 positional counterbalancing between Option A and Option B, along with explicit 4-step Chain-of-Thought (CoT) reasoning in `chosen` completions to prevent superficial associative matching.
- Implemented `SunW/AI_Decisions/train_and_evaluate_dpo.py` supporting Hugging Face `DPOTrainer` with LoRA configuration targeting `Meta-Llama-3-8B-Instruct`. Includes a BPE token profiler, execution mode tracking, summary CSV logging (`SunW/AI_Decisions/results/dpo_llama3_experiment_log.csv`), trial-level JSONL logging (`dpo_evaluation_trials.jsonl`), and a dedicated failure audit log (`dpo_misaligned_questions.jsonl`).
- Added `--simulate_baseline` audit mode to inspect baseline failure modes (RLHF safety refusals and utilitarian overrides) without active model weights.

**Verification:**
- Ran `generate_dataset.py`, generating 153 train, 27 validation, 36 OOD test, and 12 adversarial test records. Confirmed positional balance (47.7% Option A vs 52.3% Option B in train).
- Executed `train_and_evaluate_dpo.py` in iteration 1 benchmark mode, profiling 57,305 total pipeline tokens and confirming clean summary CSV output.
- Executed `train_and_evaluate_dpo.py --simulate_baseline --iteration 0`, successfully auditing and recording 37 misaligned baseline questions to `dpo_misaligned_questions.jsonl`.

**Replication notes:**
Run `python3 SunW/AI_Decisions/generate_dataset.py` to regenerate or re-partition the dataset. Run `python3 SunW/AI_Decisions/train_and_evaluate_dpo.py --iteration 1` for policy benchmarking or pass `--simulate_baseline` to audit failure modes.

## 2026-10-25 - Merge valid experiment trials into unified PvF dataset

**Context / problem:** Consolidate historical and current vehicle-classification trial results from `SunW/AI_Decisions/results/` and its `archive/` subfolder into a single dataset, retaining only successful model responses and standardizing column schemas.

**Changes made:** Created and executed `SunW/AI_Decisions/merge_results.py`, outputting to `SunW/AI_Decisions/results/pvf_merged_20261025.csv`. Extracted 76 error-free trials across 9 CSV files (excluding non-trial logs), mapped all legacy `trials_*` runs to `plan = "paid"`, and preserved the 16-column specification matching `pvf_20261025_210457.csv`.

**Verification:** Confirmed row count (76 valid data rows, 0 error rows), verified all 16 target columns are present and correctly formatted, and verified distribution across plans (54 paid, 22 free) and models (69 `gemini-3.5-flash`, 4 `gemini-3.7-flash`, 3 `gemini-3.8-flash`).

**Replication notes:** Run `python SunW/AI_Decisions/merge_results.py` to re-aggregate results whenever new trial CSVs are generated.

## 2026-10-25 - Implement Ollama paid vs. unpaid experiment runner

**Context / problem:** Adapt the paid vs. free vehicle classification experiment for Ollama models with support for API key authentication where required.

**Changes made:** Created `SunW/AI_Decisions/prompt_experiment_pvf_ollama.py` using Python standard library `urllib.request`. Configured default model `llama3.5` with instructions for pulling additional models, and added inputs for `OLLAMA_API_KEY_PAID` and `OLLAMA_API_KEY_UNPAID` with automatic token sanitization.

**Verification:** Verified syntax and module import via `python -c "import SunW.AI_Decisions.prompt_experiment_pvf_ollama"`. Confirmed unauthenticated requests work by default when keys are blank.

**Replication notes:** Run `python SunW/AI_Decisions/prompt_experiment_pvf_ollama.py` with Ollama running locally or configure `OLLAMA_HOST`.

## 2026-10-25 - Implement Gemini paid vs. free experiment runner and resolve SDK/auth errors

**Context / problem:** Run Hart vehicle classification trials comparing Google Gemini performance across paid and free accounts on `gemini-3.5-flash`, avoiding quota and import failures.

**Changes made:** Created `SunW/AI_Decisions/prompt_experiment_pvf.py` to run full factorial trials across `GOOGLE_API_KEY_FREE` and `GOOGLE_API_KEY_PAID`. Installed `google-genai` v2.33.0 in Anaconda environment to fix `cannot import name 'genai' from 'google'`. Added `_clean_key()` to strip trailing terminal escape artifacts (e.g. `;5A` from terminal escape sequences) causing `401 ACCESS_TOKEN_TYPE_UNSUPPORTED`. Added a `plan` column to output CSV/JSONL files (`pvf_YYYYMMDD_HHMMSS.csv`).

**Verification:** Verified import and successful end-to-end API generation with sanitized free and paid keys against `gemini-3.5-flash`. Confirmed CSV/JSONL output generation in `SunW/AI_Decisions/results/`.

**Replication notes:** Run `python SunW/AI_Decisions/prompt_experiment_pvf.py`. Ensure environment variables `GOOGLE_API_KEY_FREE` and `GOOGLE_API_KEY_PAID` are set.

## 2026-10-04 - Publish AI experiments repository

**Context / problem:** Establish a dedicated GitHub repository for the AI vehicle-classification experiments, separate from the course and book material.

**Changes made:** Moved the experiment project to the `AIProfessonals` repository and pushed it to `git@github.com:wbmacleod-yale/AIProfessonals.git`.

**Verification:** Confirmed the local repository has the `origin` fetch and push remote at that GitHub address.

**Replication notes:** Use the `AIProfessonals` repository as the project root for future experiment development and analysis.

## 2026-10-04 - Retain only combined experiment data

**Context / problem:** Reduce the new experiment repository's size after combining compatible replications.

**Changes made:** Removed the 119 per-run CSV, JSONL, configuration, and log artifacts from `ClaudeExp/results/`. Retained only the combined CSV and JSONL outputs.

**Verification:** Confirmed the two remaining files each contain 29,600 records from 37 source runs; the CSV parsed successfully with its `source_run_id` provenance field.

**Replication notes:** Analyze `ClaudeExp/results/trials_20261003_170319_to_20261004_062339_combined.csv`; the individual source files are intentionally unavailable in this repository.

## 2026-10-04 - Relocate Claude experiment project

**Context / problem:** Separate the AI vehicle-classification experiment from the course and book repository.

**Changes made:** Moved the `ClaudeExp` project into the new `AI-Experiments` repository directory and copied the experiment-specific worklog history here.

**Verification:** Confirmed the moved directory contains experiment source scripts, result data, and the combined dataset.

**Replication notes:** Work from this directory as the root of the new AI-experiment repository.

## 2026-10-04 - Combine complete yes/no experiment replications

**Context / problem:** Retain the complete same-configuration experiment batches after the recursive-run defect and remove the interrupted partial batch before moving the project.

**Changes made:** Created `ClaudeExp/results/trials_20261003_170319_to_20261004_062339_combined.jsonl` and `.csv`, combining 37 complete 800-row runs (29,600 rows) with a `source_run_id` provenance field. Removed the incomplete 155-row `20261004_064617` JSONL and configuration files. Preserved all complete raw runs.

**Verification:** Confirmed the combined JSONL and CSV each contain 29,600 rows from 37 source runs, with matching fields; confirmed the incomplete run is excluded and its two files are absent.

**Replication notes:** Use `trials_20261003_170319_to_20261004_062339_combined.csv` as the input to `code_responses.py`. The `source_run_id` column identifies the original replication for every row.

## 2026-10-04 - Stop recursive yes/no experiment batches

**Context / problem:** `ClaudeExp/prompt_experimentYN.py` continued creating full 800-trial result batches instead of terminating.

**Changes made:** Removed the accidental recursive `run_experiment()` call at the end of `run_experiment()`, leaving only the module-level `__main__` entry point. Gracefully interrupted the already-running process, which was executing the old code.

**Verification:** `python3 -m py_compile ClaudeExp/prompt_experimentYN.py` completed successfully. Result JSONL files showed repeated 800-row batches; the interrupted process traceback showed `run_experiment()` recursively repeated 34 times.

**Replication notes:** Run `python3 ClaudeExp/prompt_experimentYN.py` from the repository root. Each invocation now produces one 800-trial batch and one summary CSV.

## 2026-10-03 - Add CLeF typed-decision experiment runner

**Context / problem:** Run the Hart vehicle experiment with `clef-flash:latest`, which returns typed decisions rather than generated text.

**Changes made:** Created `ClaudeExp/prompt_experiment_clef.py` from the yes/no runner. It uses Ollama's local `/v1/systemone` endpoint and CLeF's `noul` true/false schema, recording `yes` or `no` plus CLeF's probability in the existing response field.

**Verification:** A direct CLeF bicycle decision returned `yes` with probability `0.900811`; `code_responses.classify_response()` returned `1`. The new script compiled successfully.

**Replication notes:** Run `python3 ClaudeExp/prompt_experiment_clef.py` in a separate tmux session on `obiwan`.

## 2026-10-03 - Start persistent yes/no Ollama experiment

**Context / problem:** Run the full local Ollama yes/no experiment so it continues after logout.

**Changes made:** Started `ClaudeExp/prompt_experimentYN.py` in detached tmux session `yesno-experiment`, with output piped to `ClaudeExp/results/yesno_experiment_20261003_170319.log`.

**Verification:** Confirmed the tmux pane was live and the `python3 prompt_experimentYN.py` process was active.

## 2026-10-03 - Run yes/no experiment through local Ollama CLI

**Context / problem:** Run the experiment directly on the Ollama server without HTTP requests.

**Changes made:** Replaced `prompt_experimentYN.py`'s `/api/chat` and `/api/tags` requests with local `ollama run` and `ollama list` subprocess calls. The CLI does not emit token counts or support the configured per-request temperature and output-token settings, so token fields are blank and those values are not applied.

**Verification:** A direct local `ollama run llama3.1:8b` returned `Yes` and `1.0`. The updated Python helper discovered the local model, returned `yes` as its first line, and `classify_response()` coded it as `1`.

**Replication notes:** Run `python3 ClaudeExp/prompt_experimentYN.py` directly on the machine where `ollama` and the required models are installed.

## 2026-10-03 - Validate yes/no Ollama experiment

**Context / problem:** Verify `ClaudeExp/prompt_experimentYN.py` after changing it to request yes/no responses and adding `clef-flash:latest`.

**Changes made:** Changed the prompt to require `yes`, `no`, or `uncertain` as its exact first line, added the missing `__main__` entry point, and removed `clef-flash:latest` from the text-generation defaults. The installed model advertises only Ollama's `decision` capability and rejects both `/api/chat` and `/api/generate`, so it cannot return written responses for this experiment.

**Verification:** `python3 -m py_compile` succeeded. The local Ollama API listed `clef-flash:latest`; direct API calls confirmed its unsupported text-generation capability. A direct `llama3.1:8b` run returned first line `yes`, and `classify_response()` returned `1`.

**Replication notes:** Run `python3 ClaudeExp/prompt_experimentYN.py` on `obiwan`. Use only Ollama models with chat or generate capability; do not include `clef-flash:latest` for written-response experiments.

## 2026-10-03 - Run Hart experiment on obiwan with Ollama

**Context / problem:** Replace paid provider APIs in `ClaudeExp/prompt_experiment.py` with models served locally by Ollama on the office server `obiwan`.

**Changes made:** Replaced OpenAI, Gemini, and Anthropic SDK clients with a standard-library Ollama `/api/chat` client; configured benchmark defaults for `llama3.1:8b`, `qwen3:8b`, `gemma3:12b`, and `phi4:14b`; added an `/api/tags` preflight check, direct-on-server usage instructions, and `think: false` so Qwen's optional reasoning channel does not consume the answer budget. No pip or apt packages are required.

**Verification:** `python3 -m py_compile ClaudeExp/prompt_experiment.py` confirmed syntax. `obiwan` was verified to have a reachable Ollama API and all four selected models installed; a direct `qwen3:8b` request with reasoning disabled returned a normal final answer.

**Replication notes:** Run `python3 ClaudeExp/prompt_experiment.py` on `obiwan`. Use `OLLAMA_MODELS` to run a subset.
