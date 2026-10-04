# Worklog - AI Experiments

Experiment-specific history migrated from `values_economics_law` on 2026-10-04.

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
