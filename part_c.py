"""Privacy-safe ticket classification, evaluation, and segment-aware routing."""

from __future__ import annotations

import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix

import mock_llm


ROOT = Path(__file__).resolve().parent
RAW_TICKETS = ROOT / "data" / "raw" / "tickets.csv"
SEGMENT_PATH = ROOT / "outputs" / "customer_segments.csv"
OUTPUTS = ROOT / "outputs"
CATEGORIES = ["billing", "delivery", "technical", "fraud"]
MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\b[6-9]\d{9}\b")
ORDER_RE = re.compile(r"ORD-\d{6}")
REDACTION_PATTERNS = [
    ("emails", EMAIL_RE, "[EMAIL]"),
    ("phones", PHONE_RE, "[PHONE]"),
    ("order_ids", ORDER_RE, "[ORDER_ID]"),
]

ZERO_SHOT_PROMPT = """You are a support ticket classifier. CLASSIFY the customer's issue into exactly one allowed category: billing, delivery, technical, fraud. Return a JSON-only reply in this exact form: {\"category\": \"...\", \"reason\": \"...\"}. Do not include markdown or text outside the JSON object."""

FEW_SHOT_PROMPT = """You are a support ticket classifier. CLASSIFY the customer's issue into exactly one allowed category: billing, delivery, technical, fraud. Return a JSON-only reply in this exact form: {\"category\": \"...\", \"reason\": \"...\"}. Do not include markdown or text outside the JSON object.
EXAMPLES:
Billing example — input: I was charged twice for [ORDER_ID]. Output: {\"category\": \"billing\", \"reason\": \"duplicate charge\"}
Fraud example — input: An unauthorized transaction was made on my account. Output: {\"category\": \"fraud\", \"reason\": \"unauthorized transaction\"}
Technical example — input: The app crashes when I open the cart. Output: {\"category\": \"technical\", \"reason\": \"app crash\"}"""


def redact(text: str) -> tuple[str, dict[str, int]]:
    redacted = text
    counts: dict[str, int] = {}
    for name, pattern, placeholder in REDACTION_PATTERNS:
        redacted, count = pattern.subn(placeholder, redacted)
        counts[name] = count
    return redacted, counts


def redaction_audit(tickets: pd.DataFrame) -> dict:
    counts: Counter[str] = Counter()
    leftovers: Counter[str] = Counter()
    for text in tickets["text"].astype(str):
        redacted, per_type = redact(text)
        counts.update(per_type)
        for name, pattern, _ in REDACTION_PATTERNS:
            leftovers[name] += len(pattern.findall(redacted))
    return {
        "substitutions": dict(counts),
        "total_substitutions": int(sum(counts.values())),
        "matches_after_redaction": dict(leftovers),
        "total_matches_after_redaction": int(sum(leftovers.values())),
    }


def classify_reply(text: str) -> tuple[str, str, str]:
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", (text or "").lower())
        matches = [word for word in words if word in CATEGORIES]
        if len(matches) == 1:
            return "classified_keyword", matches[0], text
        return "flagged_invalid_output", "", text

    if isinstance(payload, dict) and payload.get("category") in CATEGORIES:
        return "classified_json", payload["category"], str(payload.get("reason", ""))
    # Valid JSON with a disallowed category is invalid; do not fall back to keywords.
    return "flagged_invalid_output", "", text


def route_rows(rows: list[dict], segments: pd.DataFrame) -> pd.DataFrame:
    segment_lookup = segments.set_index("customer_id")[
        ["segment_name", "is_premium"]
    ].to_dict(orient="index")
    output = []
    human_statuses = {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted",
        "permanent_error",
    }
    for row in rows:
        segment = segment_lookup.get(row["customer_id"], {})
        segment_name = segment.get("segment_name", "")
        is_premium = str(segment.get("is_premium", "False")).casefold() == "true"
        status, category = row["status"], row["category"]
        if status in human_statuses:
            priority, route = "NORMAL", "human_review"
        elif category == "fraud":
            priority, route = "HIGH", "security_team"
        elif is_premium:
            priority, route = "HIGH", "premium_desk"
        else:
            priority, route = "NORMAL", "support_queue"
        output.append(
            {
                "ticket_id": row["ticket_id"],
                "customer_id": row["customer_id"],
                "status": status,
                "category": category,
                "segment_name": segment_name,
                "priority": priority,
                "route": route,
                "tokens_used": int(row["tokens_used"]),
            }
        )
    return pd.DataFrame(
        output,
        columns=[
            "ticket_id",
            "customer_id",
            "status",
            "category",
            "segment_name",
            "priority",
            "route",
            "tokens_used",
        ],
    )


def run_pipeline(
    run_name: str,
    system_prompt: str,
    token_budget: int | None = None,
    retry_log: list[str] | None = None,
) -> dict:
    # Reset immediately before every independent full dataset run.
    mock_llm.reset()
    tickets = pd.read_csv(RAW_TICKETS, keep_default_na=False)
    segments = pd.read_csv(SEGMENT_PATH, keep_default_na=False)
    rng = random.Random(42)
    total_tokens = 0
    response_tokens_by_ticket: dict[str, int] = {}
    rows: list[dict] = []
    called_ticket_ids: set[str] = set()

    for ticket in tickets.to_dict(orient="records"):
        raw_text = str(ticket["text"])
        redacted_text, _ = redact(raw_text)
        record = {
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "true_category": ticket["true_category"],
            "status": "",
            "category": "",
            "reason": "",
            "tokens_used": 0,
        }
        if not raw_text.strip():
            record["status"] = "skipped_empty"
            rows.append(record)
            continue
        if mock_llm.est_tokens(raw_text) > MAX_INPUT_TOKENS:
            record["status"] = "flagged_long"
            rows.append(record)
            continue

        # The exact user message is only the redacted source ticket text.
        assert redacted_text == redact(raw_text)[0]
        for _, pattern, _ in REDACTION_PATTERNS:
            assert pattern.search(redacted_text) is None
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": redacted_text},
        ]
        for retry_number in range(MAX_RETRIES + 1):
            if token_budget is not None and total_tokens >= token_budget:
                record["status"] = "budget_exhausted"
                break
            # Assert the actual outbound user message has no raw identifier pattern.
            assert messages[-1]["content"] == redacted_text
            assert all(pattern.search(messages[-1]["content"]) is None for _, pattern, _ in REDACTION_PATTERNS)
            called_ticket_ids.add(ticket["ticket_id"])
            try:
                response = mock_llm.chat(messages)
            except mock_llm.TransientLLMError:
                if retry_number >= MAX_RETRIES:
                    record["status"] = "permanent_error"
                    break
                wait = BASE_DELAY * (2**retry_number) * rng.uniform(0.9, 1.15)
                line = (
                    f"run={run_name} ticket_id={ticket['ticket_id']} "
                    f"retry={retry_number} wait={wait:.6f}s"
                )
                if retry_log is not None:
                    retry_log.append(line)
                time.sleep(wait)
                continue

            used = int(response.usage.total_tokens)
            total_tokens += used
            record["tokens_used"] += used
            response_tokens_by_ticket[ticket["ticket_id"]] = record["tokens_used"]
            reply = response.choices[0].message.content or ""
            status, category, reason = classify_reply(reply)
            record.update(status=status, category=category, reason=reason)
            break
        rows.append(record)

    skipped_ids = {
        row["ticket_id"] for row in rows if row["status"] in {"skipped_empty", "flagged_long"}
    }
    assert not (called_ticket_ids & skipped_ids)
    assert len(rows) == 164
    status_counts = Counter(row["status"] for row in rows)
    classified_rows = [row for row in rows if row["status"] in {"classified_json", "classified_keyword"}]
    correct = sum(row["category"] == row["true_category"] for row in classified_rows)
    accuracy = correct / len(classified_rows) if classified_rows else 0.0
    confusion = confusion_matrix(
        [row["true_category"] for row in classified_rows],
        [row["category"] for row in classified_rows],
        labels=CATEGORIES,
    ).tolist()
    return {
        "run_name": run_name,
        "rows": rows,
        "status_counts": dict(status_counts),
        "eligible_calls_by_ticket": called_ticket_ids,
        "classified": len(classified_rows),
        "correct": int(correct),
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "tokens_per_classified_ticket": total_tokens / len(classified_rows) if classified_rows else 0.0,
        "confusion_matrix_labels": CATEGORIES,
        "confusion_matrix": confusion,
    }


def public_run_summary(run: dict) -> dict:
    return {
        "classified": run["classified"],
        "correct": run["correct"],
        "accuracy": run["accuracy"],
        "total_tokens": run["total_tokens"],
        "tokens_per_classified_ticket": run["tokens_per_classified_ticket"],
        "status_counts": run["status_counts"],
        "confusion_matrix_labels": run["confusion_matrix_labels"],
        "confusion_matrix": run["confusion_matrix"],
    }


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    tickets = pd.read_csv(RAW_TICKETS, keep_default_na=False)
    audit = redaction_audit(tickets)
    assert audit["substitutions"] == {"emails": 55, "phones": 91, "order_ids": 132}
    assert audit["total_substitutions"] == 278
    assert audit["total_matches_after_redaction"] == 0
    assert sum(not str(value).strip() for value in tickets["text"]) == 4
    long_token_counts = {
        mock_llm.est_tokens(str(value))
        for value in tickets["text"]
        if mock_llm.est_tokens(str(value)) > MAX_INPUT_TOKENS
    }
    assert len(long_token_counts) == 1 and next(iter(long_token_counts)) == 734

    retry_log: list[str] = []
    zero_shot = run_pipeline("zero_shot", ZERO_SHOT_PROMPT, retry_log=retry_log)
    few_shot = run_pipeline("few_shot", FEW_SHOT_PROMPT, retry_log=retry_log)
    assert zero_shot["classified"] == 147
    assert few_shot["classified"] == 147
    assert zero_shot["status_counts"].get("classified_json") == 135
    assert zero_shot["status_counts"].get("classified_keyword") == 12
    assert zero_shot["status_counts"].get("flagged_invalid_output") == 9
    assert zero_shot["status_counts"].get("permanent_error", 0) == 0
    assert few_shot["status_counts"].get("classified_json") == 135
    assert few_shot["status_counts"].get("classified_keyword") == 12
    assert few_shot["status_counts"].get("flagged_invalid_output") == 9
    assert few_shot["status_counts"].get("permanent_error", 0) == 0
    assert abs(zero_shot["accuracy"] - 0.857142857) < 0.005
    assert abs(few_shot["accuracy"] - 1.0) < 0.005

    budget_demo = run_pipeline(
        "zero_shot_budget_6000", ZERO_SHOT_PROMPT, token_budget=6000, retry_log=retry_log
    )
    budget_exhausted_count = budget_demo["status_counts"].get("budget_exhausted", 0)
    assert budget_exhausted_count >= 1

    final_few_shot = run_pipeline(
        "few_shot_final_100000", FEW_SHOT_PROMPT, token_budget=100000, retry_log=retry_log
    )
    assert final_few_shot["status_counts"].get("budget_exhausted", 0) == 0
    assert final_few_shot["status_counts"].get("permanent_error", 0) == 0
    assert final_few_shot["status_counts"].get("classified_json") == 135
    assert final_few_shot["status_counts"].get("classified_keyword") == 12
    assert final_few_shot["status_counts"].get("flagged_invalid_output") == 9

    retry_counts_by_run: Counter[str] = Counter()
    for line in retry_log:
        name = re.search(r"run=([^ ]+)", line).group(1)
        retry_counts_by_run[name] += 1
    assert retry_counts_by_run["zero_shot"] == 19
    assert retry_counts_by_run["few_shot"] == 19
    assert retry_counts_by_run["few_shot_final_100000"] == 19

    segments = pd.read_csv(SEGMENT_PATH, keep_default_na=False)
    triage = route_rows(final_few_shot["rows"], segments)
    triage.to_csv(OUTPUTS / "triage_results.csv", index=False)
    budget_triage = route_rows(budget_demo["rows"], segments)
    budget_triage.to_csv(OUTPUTS / "triage_budget_demo.csv", index=False)
    (OUTPUTS / "triage_retry_log.txt").write_text("\n".join(retry_log) + "\n", encoding="utf-8")

    report = {
        "ticket_count": int(len(tickets)),
        "guardrails": {
            "max_input_tokens": MAX_INPUT_TOKENS,
            "empty_tickets": int(sum(not str(value).strip() for value in tickets["text"])),
            "long_tickets": int(sum(mock_llm.est_tokens(str(value)) > MAX_INPUT_TOKENS for value in tickets["text"])),
            "long_ticket_estimated_tokens": sorted(
                {mock_llm.est_tokens(str(value)) for value in tickets["text"] if mock_llm.est_tokens(str(value)) > MAX_INPUT_TOKENS}
            ),
        },
        "redaction": audit,
        "zero_shot": public_run_summary(zero_shot),
        "few_shot": public_run_summary(few_shot),
        "zero_shot_budget_demo": {
            **public_run_summary(budget_demo),
            "token_budget": 6000,
            "budget_exhausted": int(budget_exhausted_count),
        },
        "few_shot_final": {
            **public_run_summary(final_few_shot),
            "token_budget": 100000,
            "output_rows": int(len(triage)),
        },
        "retry_log_lines": len(retry_log),
        "retry_counts_by_run": dict(retry_counts_by_run),
        "retry_max_wait_seconds": max(
            (float(re.search(r"wait=([0-9.]+)s", line).group(1)) for line in retry_log),
            default=0.0,
        ),
        "routes": triage["route"].value_counts().to_dict(),
        "priorities": triage["priority"].value_counts().to_dict(),
    }
    (OUTPUTS / "part_c_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
