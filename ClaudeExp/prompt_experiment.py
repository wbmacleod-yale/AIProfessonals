#!/usr/bin/env python3
"""
AI Dilemma Experiment Runner

Runs repeated, independent (stateless) trials of a prompt against Ollama
models. Each API call is a fresh request with no conversation history,
ensuring responses are fully independent.

Usage:
            1. On the Ollama server, confirm that Ollama is running and list models:
                ollama list
            2. Optionally select a subset of the installed models:
                export OLLAMA_MODELS="llama3.1:8b,qwen3:8b"
            3. Edit the EXPERIMENT CONFIG section below.
            4. Run:  python3 prompt_experiment.py

Results are saved incrementally to JSONL (crash-safe) and a summary CSV
is generated at the end. The experiment config is also saved for reproducibility.
"""

import csv
import itertools
import json
import os
import time
import random
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path


# ============================================================================
#  EXPERIMENT CONFIG — Edit this section for your study
# ============================================================================

# Vehicles to test in the "Is X a vehicle?" question (Hart's dilemma)
VEHICLES = ["airplane", "bicycle", "car", "scooter", "stroller"]

# Core question template ({vehicle} is replaced at runtime)
QUESTION_TEMPLATE = "Is a {vehicle} a vehicle?"

# Sentence prepended when judge_role condition is True
JUDGE_SENTENCE = "Imagine that you are a judge."

# Sentence inserted when rule_given condition is True
RULE_SENTENCE = "There is a rule that says: no vehicles in the park."


def build_prompt(vehicle: str, judge_role: bool, rule_given: bool) -> str:
    """Construct the full user prompt from experimental condition variables."""
    parts = []
    if judge_role:
        parts.append(JUDGE_SENTENCE)
    if rule_given:
        parts.append(RULE_SENTENCE)
    parts.append(QUESTION_TEMPLATE.format(vehicle=vehicle))
    return " ".join(parts)

# Number of independent trials per model
N_TRIALS = 10

# Sampling temperature (higher = more varied; 0 = near-deterministic)
TEMPERATURE = 1.0

# Maximum tokens per response
MAX_TOKENS = 1024

# Seconds to wait between API calls (helps avoid rate limits)
DELAY_BETWEEN_CALLS = 1.0

# Maximum retry attempts for transient errors (rate limits, server errors)
MAX_RETRIES = 3

# Models to test on obiwan's 16 GB GPU. Override with a comma-separated
# OLLAMA_MODELS environment variable to select a subset or other models.
DEFAULT_OLLAMA_MODELS = (
    "llama3.1:8b,"
    "qwen3:8b,"
    "gemma3:12b,"
    "phi4:14b,"
    "clef-flash:latest"
)
MODELS = [
    {"provider": "ollama", "model": model.strip()}
    for model in os.environ.get("OLLAMA_MODELS", DEFAULT_OLLAMA_MODELS).split(",")
    if model.strip()
]

# Output directory (relative to this script)
OUTPUT_DIR = "results"


# ============================================================================
#  OLLAMA CONNECTION
# ============================================================================

# The script is intended to run on the Ollama server, using its local API.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


# ============================================================================
#  IMPLEMENTATION
# ============================================================================

@dataclass
class TrialResult:
    """Result of a single experimental trial."""
    provider: str
    model: str
    vehicle: str
    judge_role: bool
    rule_given: bool
    trial: int
    prompt: str
    response: str
    timestamp: str
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
#  Ollama API caller
# ---------------------------------------------------------------------------

def call_ollama(model: str, prompt: str, system_prompt: str,
                temperature: float, max_tokens: int) -> dict:
    """Stateless call to Ollama's local /api/chat endpoint."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.load(response)
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_HOST}. Start the SSH tunnel and "
            "confirm that Ollama is running on the office server."
        ) from error

    return {
        "response": result["message"]["content"],
        "input_tokens": result.get("prompt_eval_count"),
        "output_tokens": result.get("eval_count"),
    }


def available_ollama_models() -> set[str]:
    """Return model names exposed by the configured Ollama server."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=10) as response:
            return {model["name"] for model in json.load(response)["models"]}
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_HOST}. Start Ollama and retry."
        ) from error


PROVIDER_FN = {
    "ollama": call_ollama,
}


# ---------------------------------------------------------------------------
#  Retry wrapper with exponential backoff
# ---------------------------------------------------------------------------

def call_with_retry(fn, *args, max_retries: int = MAX_RETRIES) -> dict:
    """Call an API function with retry + exponential backoff on failure."""
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args)
        except Exception as e:
            err_str = str(e).lower()
            is_transient = any(kw in err_str for kw in [
                "rate", "limit", "429", "500", "502", "503", "overloaded",
                "timeout", "connection", "temporarily",
            ])
            if is_transient and attempt < max_retries:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"      ⏳ Retry {attempt}/{max_retries} in {wait:.1f}s — {e}")
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
#  Trial runner
# ---------------------------------------------------------------------------

def run_trial(provider: str, model: str, vehicle: str,
              judge_role: bool, rule_given: bool,
              trial_num: int) -> TrialResult:
    """Execute a single independent trial for one experimental condition."""
    prompt = build_prompt(vehicle, judge_role, rule_given)
    timestamp = datetime.now(timezone.utc).isoformat()
    call_fn = PROVIDER_FN[provider]

    # No system prompt — all experimental manipulation is in the user prompt
    start = time.perf_counter()
    try:
        result = call_with_retry(
            call_fn, model, prompt, "", TEMPERATURE, MAX_TOKENS
        )
        latency = time.perf_counter() - start
        return TrialResult(
            provider=provider,
            model=model,
            vehicle=vehicle,
            judge_role=judge_role,
            rule_given=rule_given,
            trial=trial_num,
            prompt=prompt,
            response=result["response"],
            timestamp=timestamp,
            latency_seconds=round(latency, 3),
            input_tokens=result.get("input_tokens"),
            output_tokens=result.get("output_tokens"),
        )
    except Exception as e:
        latency = time.perf_counter() - start
        return TrialResult(
            provider=provider,
            model=model,
            vehicle=vehicle,
            judge_role=judge_role,
            rule_given=rule_given,
            trial=trial_num,
            prompt=prompt,
            response="",
            timestamp=timestamp,
            latency_seconds=round(latency, 3),
            error=str(e),
        )


# ---------------------------------------------------------------------------
#  Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment():
    """Run the full factorial experiment:
    models × vehicles × judge_role × rule_given × N_TRIALS."""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = output_dir / f"trials_{run_id}.jsonl"
    csv_path = output_dir / f"trials_{run_id}.csv"
    config_path = output_dir / f"config_{run_id}.json"

    # ── Save experiment config for reproducibility ──
    config = {
        "run_id": run_id,
        "question_template": QUESTION_TEMPLATE,
        "judge_sentence": JUDGE_SENTENCE,
        "rule_sentence": RULE_SENTENCE,
        "vehicles": VEHICLES,
        "judge_role_values": [False, True],
        "rule_given_values": [False, True],
        "n_trials": N_TRIALS,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "delay_between_calls": DELAY_BETWEEN_CALLS,
        "models": MODELS,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"📋 Config saved to {config_path}\n")

    active_models = MODELS
    if not active_models:
        print("\n❌ No Ollama models configured. Set OLLAMA_MODELS and retry.")
        return

    installed_models = available_ollama_models()
    missing_models = [m["model"] for m in active_models if m["model"] not in installed_models]
    if missing_models:
        print("\n❌ These configured models are not installed:")
        for model in missing_models:
            print(f"   {model}")
        print("   Run `ollama list` and set OLLAMA_MODELS to installed model names.")
        return

    # ── Build all experimental conditions ──
    conditions = list(itertools.product(
        active_models,
        VEHICLES,
        [False, True],  # judge_role
        [False, True],  # rule_given
    ))
    total = len(conditions) * N_TRIALS
    n_cond_per_model = len(VEHICLES) * 2 * 2
    print(f"\n🚀 Starting experiment: {len(active_models)} model(s) × "
          f"{len(VEHICLES)} vehicles × 2 judge × 2 rule × {N_TRIALS} trials "
          f"= {total} API calls\n")

    results: list[TrialResult] = []
    completed = 0

    for model_cfg, vehicle, judge_role, rule_given in conditions:
        provider = model_cfg["provider"]
        model = model_cfg["model"]
        cond_label = (f"{model} | {vehicle} | "
                      f"judge={'Y' if judge_role else 'N'} | "
                      f"rule={'Y' if rule_given else 'N'}")
        print(f"── {cond_label} {'─' * max(1, 60 - len(cond_label))}")

        for trial in range(1, N_TRIALS + 1):
            result = run_trial(provider, model, vehicle,
                               judge_role, rule_given, trial)
            results.append(result)
            completed += 1

            # Save incrementally (crash-safe)
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

            status = "✓" if not result.error else f"✗ {result.error[:80]}"
            tokens = ""
            if result.input_tokens or result.output_tokens:
                tokens = f"  tok={result.input_tokens or '?'}→{result.output_tokens or '?'}"
            print(
                f"  Trial {trial:3d}/{N_TRIALS}  "
                f"{result.latency_seconds:6.1f}s  "
                f"{status}{tokens}  "
                f"[{completed}/{total}]"
            )

            # Delay between calls to respect rate limits
            if completed < total:
                time.sleep(DELAY_BETWEEN_CALLS)

        print()

    # ── Write summary CSV ──
    if results:
        fieldnames = list(asdict(results[0]).keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))

    # ── Print summary ──
    print("=" * 70)
    print("📊 EXPERIMENT SUMMARY")
    print("=" * 70)

    for model_cfg in active_models:
        model = model_cfg["model"]
        model_results = [r for r in results if r.model == model]
        successes = [r for r in model_results if not r.error]
        errors = len(model_results) - len(successes)
        avg_latency = (
            sum(r.latency_seconds for r in successes) / len(successes)
            if successes else 0
        )
        avg_out_tokens = None
        out_tokens = [r.output_tokens for r in successes if r.output_tokens]
        if out_tokens:
            avg_out_tokens = sum(out_tokens) / len(out_tokens)

        print(
            f"  {model:35s}  "
            f"ok={len(successes):3d}  err={errors:2d}  "
            f"avg_latency={avg_latency:5.1f}s"
            + (f"  avg_tokens={avg_out_tokens:.0f}" if avg_out_tokens else "")
        )

    print(f"\n📁 Output files:")
    print(f"   JSONL  : {jsonl_path}")
    print(f"   CSV    : {csv_path}")
    print(f"   Config : {config_path}")
    print(f"\n📐 Conditions per model: {len(VEHICLES)} vehicles × 2 judge × 2 rule = "
          f"{n_cond_per_model} conditions × {N_TRIALS} trials")
    print()


if __name__ == "__main__":
    run_experiment()
