#!/usr/bin/env python3
"""
AI Dilemma Experiment Runner — Paid vs. Free (PvF) Plan Comparison

Runs repeated, independent (stateless) trials of a prompt across Google Gemini
using two distinct Google API keys: one for a free plan and one for a paid plan.
Each API call is a fresh request with no conversation history, ensuring responses
are fully independent.

Configuration & Usage:
    1. Dependencies: The required `google-genai` package is already installed in
       your environment. If running on a new machine, install via:
         pip install -r requirements.txt
    2. API Keys: Both Free and Paid Google API keys are pre-configured directly
       in the code below (with automatic sanitization to strip shell escape artifacts).
       You can optionally override them via environment variables:
         export GOOGLE_API_KEY_FREE="AQ..."
         export GOOGLE_API_KEY_PAID="AQ..."
    3. Run:
         python SunW/AI_Decisions/prompt_experiment_pvf.py

Results are saved incrementally to JSONL (crash-safe) and summary CSV in results/:
    - pvf_YYYYMMDD_HHMMSS.csv
    - pvf_YYYYMMDD_HHMMSS.jsonl
    - pvf_config_YYYYMMDD_HHMMSS.json
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

from google import genai
from google.genai import types


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


# Number of independent trials per model / condition
N_TRIALS = 5

# Sampling temperature (higher = more varied; 0 = near-deterministic)
TEMPERATURE = 1.0

# Maximum tokens per response
MAX_TOKENS = 1024

# Seconds to wait between API calls (helps avoid rate limits)
DELAY_BETWEEN_CALLS = 1.0

# Maximum retry attempts for transient errors (rate limits, server errors)
MAX_RETRIES = 3

# Models to test — Calling Gemini 3.5 Flash only per experiment spec
MODELS = [
    {"provider": "google", "model": "gemini-3.5-flash"},
]

# Output directory (points to results folder in SunW/AI_Decisions)
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


# ============================================================================
#  API KEYS — Free plan vs Paid plan Google API Keys
# ============================================================================

def _clean_key(k: str) -> str:
    """Sanitize key string, removing trailing quotes, whitespace, and terminal artifacts like ';5A'."""
    if not k:
        return ""
    k = k.strip().strip("'\"")
    if ";" in k:
        k = k.split(";")[0].strip()
    return k


GOOGLE_API_KEY_FREE = _clean_key(
    os.environ.get("GOOGLE_API_KEY_FREE", "")
    or os.environ.get("GEMINI_API_KEY_FREE", "")
)

GOOGLE_API_KEY_PAID = _clean_key(
    os.environ.get("GOOGLE_API_KEY_PAID", "")
    or os.environ.get("GEMINI_API_KEY_PAID", "")
)

PLAN_KEYS = {
    "free": lambda: GOOGLE_API_KEY_FREE,
    "paid": lambda: GOOGLE_API_KEY_PAID,
}


# ============================================================================
#  IMPLEMENTATION
# ============================================================================

@dataclass
class TrialResult:
    """Result of a single experimental trial."""
    plan: str                     # "free" or "paid"
    provider: str                 # e.g. "google"
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
#  Google Gemini API Caller
# ---------------------------------------------------------------------------

def call_google(model: str, prompt: str, system_prompt: str,
                temperature: float, max_tokens: int, api_key: str) -> dict:
    """Stateless call to the Google Gemini API using the specified API key."""
    client = genai.Client(api_key=api_key)

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

def run_trial(plan: str, api_key: str, model: str, obj: str,
              judge_role: bool, rule_given: bool,
              trial_num: int) -> TrialResult:
    """Execute a single independent trial for one experimental condition."""
    prompt = build_prompt(obj, judge_role, rule_given)
    timestamp = datetime.now(timezone.utc).isoformat()

    start = time.perf_counter()
    try:
        result = call_with_retry(
            call_google, model, prompt, "", TEMPERATURE, MAX_TOKENS, api_key
        )
        latency = time.perf_counter() - start
        resp_text = result.get("response", "") or ""
        decision = classify_decision(resp_text, obj) if resp_text else None
        mentions_hart = check_mentions_hart(resp_text) if resp_text else None

        return TrialResult(
            plan=plan,
            provider="google",
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
            plan=plan,
            provider="google",
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
    """
    Run the full factorial experiment across both Free and Paid Google plans:
    plans (free vs paid) × models (gemini-3.5-flash) × objects × judge_role × rule_given × N_TRIALS.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = OUTPUT_DIR / f"pvf_{run_id}.jsonl"
    csv_path = OUTPUT_DIR / f"pvf_{run_id}.csv"
    config_path = OUTPUT_DIR / f"pvf_config_{run_id}.json"

    # ── Verify active plans based on configured keys ──
    active_plans = []
    for plan_name in ["free", "paid"]:
        key = PLAN_KEYS[plan_name]()
        if key:
            active_plans.append(plan_name)
        else:
            print(f"⚠️  Skipping '{plan_name}' plan — no API key set "
                  f"(set GOOGLE_API_KEY_{plan_name.upper()} or GEMINI_API_KEY_{plan_name.upper()})")

    if not active_plans:
        print("\n❌ No Google API keys configured. Set environment variables and retry:")
        print("   export GOOGLE_API_KEY_FREE=\"AIza...\"")
        print("   export GOOGLE_API_KEY_PAID=\"AIza...\"")
        return

    # ── Save experiment config for reproducibility ──
    config = {
        "run_id": run_id,
        "plans": active_plans,
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
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"📋 Config saved to {config_path}\n")

    # ── Initialize CSV with headers (includes new 'plan' column) ──
    csv_fieldnames = list(TrialResult.__dataclass_fields__.keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()

    # ── Build all experimental conditions ──
    # plans × models × objects × judge_role × rule_given
    conditions = list(itertools.product(
        active_plans,
        MODELS,
        OBJECTS,
        [False, True],  # judge_role
        [False, True],  # rule_given
    ))
    total = len(conditions) * N_TRIALS
    n_cond_per_plan = len(OBJECTS) * 2 * 2 * len(MODELS)

    print(f"🚀 Starting Paid vs. Free experiment:")
    print(f"   Plans   : {', '.join(active_plans)}")
    print(f"   Model   : {', '.join(m['model'] for m in MODELS)}")
    print(f"   Objects : {len(OBJECTS)} objects × 2 judge × 2 rule × {N_TRIALS} trials")
    print(f"   Total   : {total} API calls ({n_cond_per_plan * N_TRIALS} calls per active plan)\n")

    results: list[TrialResult] = []
    completed = 0

    for plan, model_cfg, obj, judge_role, rule_given in conditions:
        model = model_cfg["model"]
        api_key = PLAN_KEYS[plan]()
        cond_label = (f"[{plan.upper()}] {model} | {obj} | "
                      f"judge={'Y' if judge_role else 'N'} | "
                      f"rule={'Y' if rule_given else 'N'}")
        print(f"── {cond_label} {'─' * max(1, 60 - len(cond_label))}")

        for trial in range(1, N_TRIALS + 1):
            result = run_trial(plan, api_key, model, obj,
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
    print("📊 EXPERIMENT SUMMARY: PAID VS. FREE")
    print("=" * 70)

    for plan in active_plans:
        for model_cfg in MODELS:
            model = model_cfg["model"]
            plan_results = [r for r in results if r.model == model and r.plan == plan]
            successes = [r for r in plan_results if not r.error]
            errors = len(plan_results) - len(successes)
            avg_latency = (
                sum(r.latency_seconds for r in successes) / len(successes)
                if successes else 0.0
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
                f"  [{plan.upper():4s}] {model:20s}  "
                f"ok={len(successes):3d}  err={errors:2d}  "
                f"dec(+1/-1/0)={pos_dec}/{neg_dec}/{inc_dec}  "
                f"hart={hart_cnt}  "
                f"avg_lat={avg_latency:.2f}s"
            )

    print(f"\n📁 Output files:")
    print(f"   JSONL  : {jsonl_path}")
    print(f"   CSV    : {csv_path}")
    print(f"   Config : {config_path}")
    print(f"\n📐 Conditions per plan: {len(OBJECTS)} objects × 2 judge × 2 rule = "
          f"{n_cond_per_plan} conditions × {N_TRIALS} trials = {n_cond_per_plan * N_TRIALS} runs per plan")
    print()


if __name__ == "__main__":
    run_experiment()
