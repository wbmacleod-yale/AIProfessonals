#!/usr/bin/env python3
"""
AI Dilemma Experiment Runner

Runs repeated, independent (stateless) trials of a prompt across OpenAI,
Google Gemini, and Anthropic Claude models. Each API call is a fresh request
with no conversation history, ensuring responses are fully independent.

Usage on the command line:
    1. Install dependencies:  pip install -r requirements.txt
    2. Set API keys as environment variables (or in a .env file):
         export OPENAI_API_KEY="sk-..."
         export GOOGLE_API_KEY="AIza..."       (or GEMINI_API_KEY)
         export ANTHROPIC_API_KEY="sk-ant-..."
    3. Edit the EXPERIMENT CONFIG section below. In particular, edit the MODELS list
       to only include the models you want to test. Otherwise, you may waste tokens
       and quickly reach quota limits. Similarly, consider changing N_TRIALS.
    4. Run:  python prompt_experiment.py

Results are saved incrementally to JSONL (crash-safe) and a summary CSV.
"""

import csv
import itertools
import json
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path


# ============================================================================
#  EXPERIMENT CONFIG — Edit this section for your study
# ============================================================================

# Objects to test in the "Is X a vehicle?" question (Hart's dilemma)
OBJECTS = ["airplane", "bicycle", "car", "scooter", "stroller"]

# Core question template ({object} is replaced at runtime)
QUESTION_TEMPLATE = "Is a {object} a vehicle?"

# Sentence prepended when judge_role condition is True
JUDGE_SENTENCE = "Imagine that you are a judge."

# Sentence inserted when rule_given condition is True
RULE_SENTENCE = "There is a rule that says: no vehicles in the park."


def build_prompt(obj: str, judge_role: bool, rule_given: bool) -> str:
    """Construct the full user prompt from experimental condition variables."""
    parts = []
    if judge_role:
        parts.append(JUDGE_SENTENCE)
    if rule_given:
        parts.append(RULE_SENTENCE)
    parts.append(QUESTION_TEMPLATE.format(object=obj))
    return " ".join(parts)


def check_mentions_hart(text: str) -> bool:
    """Check if the response mentions H.L.A. Hart or the Hart-Fuller debate."""
    if not text:
        return False
    # Matches "Hart", "H.L.A. Hart", "HLA Hart", "H. L. A. Hart", "Hart's"
    pattern = r"\b(h\.?\s*l\.?\s*a\.?\s*)?hart('s)?\b"
    return bool(re.search(pattern, text, re.IGNORECASE))


def classify_decision(text: str, obj: str) -> int:
    """
    Classify the AI's final decision:
      1 = AI decides that the object is a vehicle
     -1 = AI decides the object is NOT a vehicle
      0 = AI does not come to a conclusion / inconclusive / error
    """
    if not text or not text.strip():
        return 0

    # Strip markdown emphasis and lowercase for consistent matching
    cleaned = re.sub(r"[*_#`]", "", text).strip()
    lower = cleaned.lower()
    obj_lower = obj.lower()

    # 1. Check for explicit rulings / verdicts / decisions
    ruling_pos = [
        r"(?:ruling|verdict|decision|holding|conclusion)\s*:\s*(?:yes|an?\s+" + re.escape(obj_lower) + r"\s+is\s+a\s+vehicle)",
        r"(?:my\s+(?:ruling|decision|judgment|verdict)\s+(?:is|would\s+be)\s*:\s*yes)",
        r"(?:i\s+(?:would\s+)?rule\s+(?:that\s+)?(?:yes|an?\s+" + re.escape(obj_lower) + r"\s+is\s+a\s+vehicle))",
    ]
    for pat in ruling_pos:
        if re.search(pat, lower):
            return 1

    ruling_neg = [
        r"(?:ruling|verdict|decision|holding|conclusion)\s*:\s*(?:no|an?\s+" + re.escape(obj_lower) + r"\s+is\s+not\s+a\s+vehicle)",
        r"(?:my\s+(?:ruling|decision|judgment|verdict)\s+(?:is|would\s+be)\s*:\s*no)",
        r"(?:i\s+(?:would\s+)?rule\s+(?:that\s+)?(?:no|an?\s+" + re.escape(obj_lower) + r"\s+is\s+not\s+a\s+vehicle))",
    ]
    for pat in ruling_neg:
        if re.search(pat, lower):
            return -1

    # 2. Check the opening sentence or first line
    first_line = lower.split("\n")[0].strip()
    first_sentence = re.split(r"[.!?]", lower)[0].strip()

    if re.match(r"^yes\b", first_line) or re.match(r"^yes\b", first_sentence):
        if not re.search(r"\byes\s+(?:and|or)\s+no\b|\bnot\s+a\s+vehicle\b", first_sentence):
            return 1

    if re.match(r"^no\b", first_line) or re.match(r"^no\b", first_sentence):
        if not re.search(r"\bno\s+(?:and|or)\s+yes\b", first_sentence):
            return -1

    # 3. Check for substantive assertion phrases throughout the text
    pos_patterns = [
        r"\b" + re.escape(obj_lower) + r"\s+is\s+(?:definitely\s+|clearly\s+|unequivocally\s+|indeed\s+|certainly\s+|generally\s+considered\s+)?a\s+vehicle\b",
        r"\b" + re.escape(obj_lower) + r"\s+qualifies\s+as\s+a\s+vehicle\b",
        r"\b" + re.escape(obj_lower) + r"\s+counts\s+as\s+a\s+vehicle\b",
        r"\b" + re.escape(obj_lower) + r"\s+falls\s+under\s+the\s+definition\s+of\s+(?:a\s+)?vehicle\b",
    ]
    has_pos = any(re.search(p, lower) for p in pos_patterns)

    neg_patterns = [
        r"\b" + re.escape(obj_lower) + r"\s+is\s+(?:definitely\s+|clearly\s+|unequivocally\s+|generally\s+)?not\s+a\s+vehicle\b",
        r"\b" + re.escape(obj_lower) + r"\s+is\s+not\s+considered\s+a\s+vehicle\b",
        r"\b" + re.escape(obj_lower) + r"\s+does\s+not\s+qualify\s+as\s+a\s+vehicle\b",
        r"\b" + re.escape(obj_lower) + r"\s+does\s+not\s+count\s+as\s+a\s+vehicle\b",
        r"\bnot\s+a\s+vehicle\b",
    ]
    has_neg = any(re.search(p, lower) for p in neg_patterns)

    if has_pos and not has_neg:
        return 1
    elif has_neg and not has_pos:
        return -1

    # 4. Inconclusive or ambiguous
    return 0


# Number of independent trials per model
N_TRIALS = 5

# Sampling temperature (higher = more varied; 0 = near-deterministic)
TEMPERATURE = 1.0

# Maximum tokens per response
MAX_TOKENS = 1024

# Seconds to wait between API calls (helps avoid rate limits)
DELAY_BETWEEN_CALLS = 1.0

# Maximum retry attempts for transient errors (rate limits, server errors)
MAX_RETRIES = 3

# Models to test — comment out any you don't need
MODELS = [
    # --- OpenAI ---
    {"provider": "openai",    "model": "gpt-4o"},
    {"provider": "openai",    "model": "gpt-4o-mini"},
    {"provider": "openai",    "model": "o3-mini"},
    # --- Google Gemini ---
    # {"provider": "google",    "model": "gemini-2.5-flash"},
    {"provider": "google",    "model": "gemini-3.7-flash"},
    {"provider": "google",    "model": "gemini-3.5-flash"},
    # --- Anthropic Claude ---
    {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
    {"provider": "anthropic", "model": "claude-haiku-4-20250414"},
]

# Output directory (relative to this script)
OUTPUT_DIR = "results"


# ============================================================================
#  API KEYS — Set via environment variables (recommended) or paste below
# ============================================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GOOGLE_API_KEY = (
    os.environ.get("GOOGLE_API_KEY", "")
    or os.environ.get("GEMINI_API_KEY", "")
)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


# ============================================================================
#  IMPLEMENTATION
# ============================================================================

@dataclass
class TrialResult:
    """Result of a single experimental trial."""
    provider: str
    model: str
    object: str
    judge_role: bool
    rule_given: bool
    trial: int
    prompt: str
    decision: int | None          # 1 = vehicle, -1 = not a vehicle, 0 = inconclusive, None = error
    mentions_hart: bool | None    # True if output mentions H.L.A. Hart / Hart, None = error
    response: str
    timestamp: str
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
#  Provider-specific API callers
# ---------------------------------------------------------------------------

def call_openai(model: str, prompt: str, system_prompt: str,
                temperature: float, max_tokens: int) -> dict:
    """Stateless call to the OpenAI Chat Completions API."""
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # o-series reasoning models don't support temperature or max_tokens
    kwargs = {}
    if model.startswith("o"):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["temperature"] = temperature
        kwargs["max_tokens"] = max_tokens

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )

    return {
        "response": response.choices[0].message.content,
        "input_tokens": response.usage.prompt_tokens if response.usage else None,
        "output_tokens": response.usage.completion_tokens if response.usage else None,
    }


def call_google(model: str, prompt: str, system_prompt: str,
                temperature: float, max_tokens: int) -> dict:
    """Stateless call to the Google Gemini API."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GOOGLE_API_KEY)

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    if system_prompt:
        config.system_instruction = system_prompt

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    usage = response.usage_metadata
    return {
        "response": response.text,
        "input_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
    }


def call_anthropic(model: str, prompt: str, system_prompt: str,
                   temperature: float, max_tokens: int) -> dict:
    """Stateless call to the Anthropic Messages API."""
    from anthropic import Anthropic

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    response = client.messages.create(**kwargs)

    return {
        "response": response.content[0].text,
        "input_tokens": response.usage.input_tokens if response.usage else None,
        "output_tokens": response.usage.output_tokens if response.usage else None,
    }


PROVIDER_FN = {
    "openai": call_openai,
    "google": call_google,
    "anthropic": call_anthropic,
}

PROVIDER_KEY = {
    "openai": lambda: OPENAI_API_KEY,
    "google": lambda: GOOGLE_API_KEY,
    "anthropic": lambda: ANTHROPIC_API_KEY,
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
                "timeout", "connection", "temporarily", "resource_exhausted", "quota",
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

def run_trial(provider: str, model: str, obj: str,
              judge_role: bool, rule_given: bool,
              trial_num: int) -> TrialResult:
    """Execute a single independent trial for one experimental condition."""
    prompt = build_prompt(obj, judge_role, rule_given)
    timestamp = datetime.now(timezone.utc).isoformat()
    call_fn = PROVIDER_FN[provider]

    # No system prompt — all experimental manipulation is in the user prompt
    start = time.perf_counter()
    try:
        result = call_with_retry(
            call_fn, model, prompt, "", TEMPERATURE, MAX_TOKENS
        )
        latency = time.perf_counter() - start
        resp_text = result.get("response", "") or ""
        decision = classify_decision(resp_text, obj) if resp_text else None
        mentions_hart = check_mentions_hart(resp_text) if resp_text else None

        return TrialResult(
            provider=provider,
            model=model,
            object=obj,
            judge_role=judge_role,
            rule_given=rule_given,
            trial=trial_num,
            prompt=prompt,
            decision=decision,
            mentions_hart=mentions_hart,
            response=resp_text,
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
            object=obj,
            judge_role=judge_role,
            rule_given=rule_given,
            trial=trial_num,
            prompt=prompt,
            decision=None,
            mentions_hart=None,
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
    models × objects × judge_role × rule_given × N_TRIALS."""
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
        "objects": OBJECTS,
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

    # ── Initialize CSV with headers ──
    csv_fieldnames = list(TrialResult.__dataclass_fields__.keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()

    # ── Filter to models whose API keys are available ──
    active_models = []
    for m in MODELS:
        key = PROVIDER_KEY[m["provider"]]()
        if not key:
            print(f"⚠️  Skipping {m['model']} — no API key set for {m['provider']}")
        else:
            active_models.append(m)

    if not active_models:
        print("\n❌ No API keys configured. Set environment variables and retry.")
        print("   OPENAI_API_KEY, GOOGLE_API_KEY (or GEMINI_API_KEY), ANTHROPIC_API_KEY")
        return

    # ── Build all experimental conditions ──
    conditions = list(itertools.product(
        active_models,
        OBJECTS,
        [False, True],  # judge_role
        [False, True],  # rule_given
    ))
    total = len(conditions) * N_TRIALS
    n_cond_per_model = len(OBJECTS) * 2 * 2
    print(f"\n🚀 Starting experiment: {len(active_models)} model(s) × "
          f"{len(OBJECTS)} objects × 2 judge × 2 rule × {N_TRIALS} trials "
          f"= {total} API calls\n")

    results: list[TrialResult] = []
    completed = 0

    for model_cfg, obj, judge_role, rule_given in conditions:
        provider = model_cfg["provider"]
        model = model_cfg["model"]
        cond_label = (f"{model} | {obj} | "
                      f"judge={'Y' if judge_role else 'N'} | "
                      f"rule={'Y' if rule_given else 'N'}")
        print(f"── {cond_label} {'─' * max(1, 60 - len(cond_label))}")

        for trial in range(1, N_TRIALS + 1):
            result = run_trial(provider, model, obj,
                               judge_role, rule_given, trial)
            results.append(result)
            completed += 1

            # Save incrementally (crash-safe JSONL + real-time CSV)
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                writer.writerow(asdict(result))

            status = "✓" if not result.error else f"✗ {result.error[:80]}"
            tokens = ""
            if result.input_tokens or result.output_tokens:
                tokens = f" tok={result.input_tokens or '?'}→{result.output_tokens or '?'}"
            dec_str = f" dec={result.decision:+d}" if result.decision is not None else ""
            hart_str = " [Hart]" if result.mentions_hart else ""
            print(
                f"  Trial {trial:3d}/{N_TRIALS}  "
                f"{result.latency_seconds:6.1f}s  "
                f"{status}{dec_str}{hart_str}{tokens}  "
                f"[{completed}/{total}]"
            )

            # Delay between calls to respect rate limits
            if completed < total:
                time.sleep(DELAY_BETWEEN_CALLS)

        print()



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

        # Decision breakdown
        pos_dec = sum(1 for r in successes if r.decision == 1)
        neg_dec = sum(1 for r in successes if r.decision == -1)
        inc_dec = sum(1 for r in successes if r.decision == 0)
        hart_cnt = sum(1 for r in successes if r.mentions_hart)

        print(
            f"  {model:35s}  "
            f"ok={len(successes):3d}  err={errors:2d}  "
            f"dec(+1/-1/0)={pos_dec}/{neg_dec}/{inc_dec}  "
            f"hart={hart_cnt}"
        )

    print(f"\n📁 Output files:")
    print(f"   JSONL  : {jsonl_path}")
    print(f"   CSV    : {csv_path}")
    print(f"   Config : {config_path}")
    print(f"\n📐 Conditions per model: {len(OBJECTS)} objects × 2 judge × 2 rule = "
          f"{n_cond_per_model} conditions × {N_TRIALS} trials")
    print()


if __name__ == "__main__":
    run_experiment()
