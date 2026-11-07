"""Policy retrieval and a checkpointable, offline ReAct refund agent."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm


ROOT = Path(__file__).resolve().parent
POLICY_DIR = ROOT / "data" / "policies"
DB_PATH = ROOT / "data" / "instamart.db"
RAW_DIR = ROOT / "data" / "raw"
OUTPUTS = ROOT / "outputs"
STATE_DIR = ROOT / "state"
POLICY_FILES = [
    "refund_policy.txt",
    "delivery_policy.txt",
    "escalation_policy.txt",
    "payment_policy.txt",
    "privacy_policy.txt",
]
CHUNK_WORDS = 40
CHUNK_OVERLAP = 10

POLICY_SYSTEM_PROMPT = (
    "You are a POLICY ASSISTANT. Answer only from the supplied policy context. "
    "If the context does not contain the answer, say: I could not find that information in the policy documents."
)
AGENT_SYSTEM_PROMPT = (
    "You are an AGENT for simulated refunds. Use the tools to inspect orders and policy before deciding. "
    "Only completed orders qualify; over 45 delivery minutes earns the full order amount; amounts over INR 1000 "
    "must be escalated. Finish with REFUND, ESCALATE, DENIED, or NOT_FOUND and a short explanation."
)


def build_chunks(overlap_words: int = CHUNK_OVERLAP) -> list[dict]:
    if overlap_words < 0 or overlap_words >= CHUNK_WORDS:
        raise ValueError("overlap_words must be from 0 through 39")
    step = CHUNK_WORDS - overlap_words
    chunks = []
    for source in POLICY_FILES:
        words = (POLICY_DIR / source).read_text(encoding="utf-8").split()
        start = 0
        while start < len(words):
            end = min(start + CHUNK_WORDS, len(words))
            chunks.append({"source": source, "text": " ".join(words[start:end])})
            if end >= len(words):
                break
            start += step
    return chunks


class PolicyRetriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([chunk["text"] for chunk in chunks])

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        indexes = sorted(range(len(self.chunks)), key=lambda index: (-scores[index], index))[:k]
        return [
            {
                "source": self.chunks[index]["source"],
                "score": float(scores[index]),
                "text": self.chunks[index]["text"],
            }
            for index in indexes
        ]


_DEFAULT_RETRIEVER: PolicyRetriever | None = None


def default_retriever() -> PolicyRetriever:
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        _DEFAULT_RETRIEVER = PolicyRetriever(build_chunks(CHUNK_OVERLAP))
    return _DEFAULT_RETRIEVER


def retrieve(query: str, k: int = 3) -> list[dict]:
    """Retrieve the top-k policy chunks as source, score, and text dictionaries."""
    return default_retriever().retrieve(query, k)


def answer_with_retriever(question: str, retriever: PolicyRetriever) -> dict:
    results = retriever.retrieve(question, k=3)
    context = "\n".join(f"[{item['source']}]\n{item['text']}" for item in results)
    user_message = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    response = mock_llm.chat(
        [
            {"role": "system", "content": POLICY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
    )
    text = response.choices[0].message.content or ""
    sources = list(dict.fromkeys(item["source"] for item in results))
    return {"answer": text, "sources": sources}


def answer(question: str) -> dict:
    """Answer one policy question using the default overlapping chunk set."""
    return answer_with_retriever(question, default_retriever())


def evaluate_retrieval(retriever: PolicyRetriever) -> dict:
    questions = pd.read_csv(RAW_DIR / "policy_eval.csv", keep_default_na=False)
    answerable = []
    unanswerable = []
    per_question = []
    for row in questions.to_dict(orient="records"):
        found = answer_with_retriever(row["question"], retriever)
        top_chunks = retriever.retrieve(row["question"], k=3)
        if row["expected_doc"] == "NONE":
            passed = found["answer"].startswith("I could not find")
            unanswerable.append(passed)
            per_question.append(
                {"question": row["question"], "answer": found["answer"], "fallback": passed}
            )
        else:
            doc_found = row["expected_doc"] in [chunk["source"] for chunk in top_chunks]
            fact_found = row["expected_fact"] in found["answer"]
            answerable.append((doc_found, fact_found))
            per_question.append(
                {
                    "question": row["question"],
                    "expected_doc": row["expected_doc"],
                    "expected_fact": row["expected_fact"],
                    "sources": [chunk["source"] for chunk in top_chunks],
                    "answer": found["answer"],
                    "expected_doc_in_top3": doc_found,
                    "expected_fact_in_answer": fact_found,
                }
            )
    failures = [
        item
        for item in per_question
        if item.get("expected_doc_in_top3") is False or item.get("expected_fact_in_answer") is False
    ]
    return {
        "answerable_count": len(answerable),
        "expected_doc_in_top3": sum(doc for doc, _ in answerable),
        "expected_fact_in_answer": sum(fact for _, fact in answerable),
        "unanswerable_count": len(unanswerable),
        "correct_fallbacks": sum(unanswerable),
        "failures": failures,
        "per_question": per_question,
    }


def _ledger_append(tool_name: str, order_id: str, refund_inr: float) -> None:
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
    }
    with (OUTPUTS / "refund_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_order(order_id: str) -> dict:
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute(
            "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
            (order_id,),
        ).fetchone()
    if row is None:
        return {"error": "order not found"}
    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": float(row[2]),
        "delivery_minutes": None if row[3] is None else int(row[3]),
    }


def search_policy(query: str) -> dict:
    return {"results": retrieve(query, k=3)}


def calculate_refund(order_amount: float, delivery_minutes: int | None) -> dict:
    refund = float(order_amount) if delivery_minutes is not None and int(delivery_minutes) > 45 else 0.0
    return {"refund_inr": round(refund, 2)}


def _verify_refund_request(order_id: str, refund_inr: float) -> None:
    order = get_order(order_id)
    if "error" in order or order["status"] != "completed":
        raise ValueError("Only a completed order can be refunded")
    expected = calculate_refund(order["order_amount"], order["delivery_minutes"])["refund_inr"]
    if expected <= 0 or abs(expected - float(refund_inr)) >= 0.01:
        raise ValueError("Refund amount does not match the completed late-delivery order")


def issue_refund(order_id: str, refund_inr: float) -> dict:
    amount = round(float(refund_inr), 2)
    if amount > 1000:
        raise ValueError("Refunds above INR 1000 require human approval")
    _verify_refund_request(order_id, amount)
    _ledger_append("issue_refund", order_id, amount)
    return {"refund_inr": amount}


def escalate_to_human(order_id: str, refund_inr: float) -> dict:
    amount = round(float(refund_inr), 2)
    if amount <= 1000:
        raise ValueError("Only refunds above INR 1000 require escalation")
    _verify_refund_request(order_id, amount)
    _ledger_append("escalate_to_human", order_id, amount)
    return {"refund_inr": amount}


TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Look up one order and return its status, amount, and delivery duration.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    },
    {
        "name": "search_policy",
        "description": "Retrieve the three policy chunks most relevant to a query.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    {
        "name": "calculate_refund",
        "description": "Calculate the refund amount for a completed order from its delivery duration.",
        "parameters": {
            "type": "object",
            "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": ["integer", "null"]}},
            "required": ["order_amount", "delivery_minutes"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Record an automatically approved refund of no more than INR 1000.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
            "required": ["order_id", "refund_inr"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Record a refund above INR 1000 for human support lead approval.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
            "required": ["order_id", "refund_inr"],
        },
    },
]
TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


class SimulatedCrash(RuntimeError):
    """Raised only by the crash-and-resume demonstration."""


def _checkpoint_path(run_id: str) -> Path:
    if re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise ValueError("run_id may contain only letters, digits, underscores, and hyphens")
    return STATE_DIR / f"{run_id}.json"


def _save_checkpoint(state: dict, started_at: float) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state["elapsed_seconds"] = max(0.0, time.monotonic() - started_at)
    checkpoint = {
        "run_id": state["run_id"],
        "scenario_id": state["scenario_id"],
        "messages": state["messages"],
        "tokens_used": state["tokens_used"],
        "step_number": state["step_number"],
        "tools_executed": state["tools_executed"],
        "steps": state["steps"],
        "llm_calls": state["llm_calls"],
        "max_iterations": state["max_iterations"],
        "token_limit": state["token_limit"],
        "time_limit_seconds": state["time_limit_seconds"],
        "elapsed_seconds": state["elapsed_seconds"],
    }
    _checkpoint_path(state["run_id"]).write_text(
        json.dumps(checkpoint, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )


def _agent_result(state: dict, status: str, final_answer: str) -> dict:
    return {
        "run_id": state["run_id"],
        "scenario_id": state["scenario_id"],
        "status": status,
        "final_answer": final_answer,
        "steps": state["steps"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
        "tools_executed": state["tools_executed"],
    }


def _fallback(state: dict, status: str, reason: str) -> dict:
    completed = len(state["steps"])
    final = (
        f"{status}: hand this case to a human agent. {reason} "
        f"Completed {completed} tool step(s); {state['llm_calls']} model call(s) used."
    )
    return _agent_result(state, status, final)


def _continue_agent(
    state: dict,
    started_at: float,
    crash_after_tool_count: int | None = None,
) -> dict:
    while state["llm_calls"] < state["max_iterations"]:
        elapsed = time.monotonic() - started_at
        if elapsed >= state["time_limit_seconds"]:
            return _fallback(state, "FALLBACK_TIMEOUT", "The time limit was reached.")
        if state["tokens_used"] >= state["token_limit"]:
            return _fallback(state, "FALLBACK_TOKEN_LIMIT", "The token limit was reached before another call.")

        response = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += int(response.usage.total_tokens)
        message = response.choices[0].message
        if state["tokens_used"] > state["token_limit"]:
            return _fallback(state, "FALLBACK_TOKEN_LIMIT", "The latest model response exceeded the token limit; its requested tool was not run.")

        tool_calls = list(message.tool_calls or [])
        if not tool_calls:
            final_answer = message.content or ""
            status = final_answer.split(":", 1)[0].strip()
            return _agent_result(state, status, final_answer)

        assistant_calls = []
        for tool_call in tool_calls:
            arguments = tool_call.arguments
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            assistant_calls.append(
                {"id": tool_call.id, "name": tool_call.name, "arguments": arguments}
            )
        state["messages"].append(
            {"role": "assistant", "content": None, "tool_calls": assistant_calls}
        )

        for tool_call in assistant_calls:
            tool_name = tool_call["name"]
            if tool_name not in TOOL_MAP:
                observation = {"error": "unknown tool"}
            else:
                observation = TOOL_MAP[tool_name](**tool_call["arguments"])
            observation_json = json.dumps(observation, ensure_ascii=False, sort_keys=True)
            state["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_name,
                    "content": observation_json,
                }
            )
            step = {
                "tool_name": tool_name,
                "arguments": tool_call["arguments"],
                "observation": observation,
            }
            state["steps"].append(step)
            state["tools_executed"].append(tool_name)
            state["step_number"] += 1
            _save_checkpoint(state, started_at)
            if (
                crash_after_tool_count is not None
                and len(state["tools_executed"]) == crash_after_tool_count
            ):
                raise SimulatedCrash(
                    f"Simulated crash after tool {crash_after_tool_count}; checkpoint saved."
                )

    return _fallback(
        state,
        "FALLBACK_MAX_ITERATIONS",
        f"Stopped after {state['max_iterations']} model calls without a final answer.",
    )


def run_agent(
    user_message: str,
    scenario_id: str,
    max_iterations: int = 6,
    token_limit: int = 6000,
    time_limit_seconds: float = 10,
    *,
    run_id: str | None = None,
    crash_after_tool_count: int | None = None,
) -> dict:
    actual_run_id = run_id or scenario_id
    state = {
        "run_id": actual_run_id,
        "scenario_id": scenario_id,
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "tokens_used": 0,
        "step_number": 0,
        "tools_executed": [],
        "steps": [],
        "llm_calls": 0,
        "max_iterations": int(max_iterations),
        "token_limit": int(token_limit),
        "time_limit_seconds": float(time_limit_seconds),
        "elapsed_seconds": 0.0,
    }
    started_at = time.monotonic()
    return _continue_agent(state, started_at, crash_after_tool_count)


def resume_agent(run_id: str) -> dict:
    path = _checkpoint_path(run_id)
    state = json.loads(path.read_text(encoding="utf-8"))
    started_at = time.monotonic() - float(state.get("elapsed_seconds", 0.0))
    return _continue_agent(state, started_at)


def _append_jsonl(path: Path, item: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")


def _scenario_refund(result: dict) -> float:
    for step in reversed(result["steps"]):
        if "refund_inr" in step["observation"]:
            return float(step["observation"]["refund_inr"])
        if "refund_inr" in step["arguments"]:
            return float(step["arguments"]["refund_inr"])
    return 0.0


def _agent_log_record(result: dict) -> dict:
    return {
        "scenario_id": result["scenario_id"],
        "status": result["status"],
        "final_answer": result["final_answer"],
        "llm_calls": result["llm_calls"],
        "tokens_used": result["tokens_used"],
        "steps": result["steps"],
    }


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    STATE_DIR.mkdir(exist_ok=True)

    chunks = build_chunks(CHUNK_OVERLAP)
    retriever = PolicyRetriever(chunks)
    eval_overlap = evaluate_retrieval(retriever)
    zero_chunks = build_chunks(0)
    zero_retriever = PolicyRetriever(zero_chunks)
    eval_zero = evaluate_retrieval(zero_retriever)
    refund_top3 = retrieve("refund policy late delivery", k=3)
    expected_chunks = {
        "refund_policy.txt": 4,
        "delivery_policy.txt": 3,
        "escalation_policy.txt": 3,
        "payment_policy.txt": 3,
        "privacy_policy.txt": 2,
    }
    assert {source: sum(chunk["source"] == source for chunk in chunks) for source in POLICY_FILES} == expected_chunks
    assert eval_overlap["expected_doc_in_top3"] == 12
    assert eval_overlap["expected_fact_in_answer"] == 12
    assert eval_overlap["correct_fallbacks"] == 2
    assert eval_zero["failures"], "Zero-overlap chunks should lose at least one boundary fact."
    assert all(source == "refund_policy.txt" for source in [item["source"] for item in refund_top3])

    ledger_path = OUTPUTS / "refund_ledger.jsonl"
    runs_path = OUTPUTS / "agent_runs.jsonl"
    ledger_path.write_text("", encoding="utf-8")
    runs_path.write_text("", encoding="utf-8")
    scenarios = pd.read_csv(RAW_DIR / "agent_scenarios.csv", keep_default_na=False)
    scenario_results = []
    for scenario in scenarios.to_dict(orient="records"):
        result = run_agent(
            scenario["message"],
            scenario["scenario_id"],
            max_iterations=6,
            token_limit=6000,
            time_limit_seconds=10,
            run_id=f"scenario_{scenario['scenario_id']}",
        )
        _append_jsonl(runs_path, _agent_log_record(result))
        expected_status = scenario["expected_outcome"]
        if expected_status == "FALLBACK":
            expected_status = "FALLBACK_MAX_ITERATIONS"
        assert result["status"] == expected_status, (scenario["scenario_id"], result["status"], expected_status)
        expected_amount = float(scenario["expected_refund_inr"])
        actual_amount = _scenario_refund(result)
        assert abs(actual_amount - expected_amount) < 0.01, (
            scenario["scenario_id"], actual_amount, expected_amount
        )
        scenario_results.append(
            {
                "scenario_id": scenario["scenario_id"],
                "status": result["status"],
                "refund_inr": round(actual_amount, 2),
                "llm_calls": result["llm_calls"],
                "tokens_used": result["tokens_used"],
                "expected_outcome": scenario["expected_outcome"],
            }
        )
    assert scenario_results[0]["llm_calls"] == 5
    assert scenario_results[-1]["status"] == "FALLBACK_MAX_ITERATIONS"
    assert scenario_results[4]["llm_calls"] == 3
    assert scenario_results[5]["llm_calls"] == 2
    assert scenario_results[-1]["llm_calls"] == 6

    s1 = scenarios.loc[scenarios["scenario_id"].eq("S1")].iloc[0]
    token_demo = run_agent(
        s1["message"], "S1", token_limit=200, run_id="demo_S1_token_limit"
    )
    _append_jsonl(runs_path, _agent_log_record(token_demo))
    timeout_demo = run_agent(
        s1["message"], "S1", time_limit_seconds=0, run_id="demo_S1_timeout"
    )
    _append_jsonl(runs_path, _agent_log_record(timeout_demo))
    assert token_demo["status"] == "FALLBACK_TOKEN_LIMIT"
    assert timeout_demo["status"] == "FALLBACK_TIMEOUT"
    assert timeout_demo["llm_calls"] == 0

    crash_run_id = "demo_S1_crash_resume"
    try:
        run_agent(
            s1["message"],
            "S1",
            run_id=crash_run_id,
            crash_after_tool_count=2,
        )
    except SimulatedCrash:
        pass
    else:
        raise AssertionError("The simulated crash did not occur.")
    partial_checkpoint = json.loads(_checkpoint_path(crash_run_id).read_text(encoding="utf-8"))
    _append_jsonl(
        runs_path,
        {
            "scenario_id": "S1_CRASHED_PARTIAL",
            "status": "SIMULATED_CRASH",
            "final_answer": "Simulated crash after the second tool; checkpoint saved for resume.",
            "llm_calls": partial_checkpoint["llm_calls"],
            "tokens_used": partial_checkpoint["tokens_used"],
            "steps": partial_checkpoint["steps"],
        },
    )
    resumed = resume_agent(crash_run_id)
    _append_jsonl(runs_path, _agent_log_record(resumed))
    expected_resume_tools = ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert resumed["status"] == "REFUND"
    assert abs(_scenario_refund(resumed) - 128.25) < 0.01
    assert resumed["tools_executed"] == expected_resume_tools
    assert partial_checkpoint["tools_executed"] == expected_resume_tools[:2]

    no_overlap_failures = eval_zero["failures"]
    report = {
        "chunk_counts_overlap_10": dict(
            (source, sum(chunk["source"] == source for chunk in chunks)) for source in POLICY_FILES
        ),
        "chunk_total_overlap_10": len(chunks),
        "chunk_counts_overlap_0": dict(
            (source, sum(chunk["source"] == source for chunk in zero_chunks)) for source in POLICY_FILES
        ),
        "chunk_total_overlap_0": len(zero_chunks),
        "refund_query_top3_sources": [item["source"] for item in refund_top3],
        "retrieval_evaluation_overlap_10": eval_overlap,
        "retrieval_evaluation_overlap_0": eval_zero,
        "agent_scenarios": scenario_results,
        "s1_token_limit_demo": {
            "status": token_demo["status"],
            "llm_calls": token_demo["llm_calls"],
            "tokens_used": token_demo["tokens_used"],
        },
        "s1_timeout_demo": {
            "status": timeout_demo["status"],
            "llm_calls": timeout_demo["llm_calls"],
            "tokens_used": timeout_demo["tokens_used"],
        },
        "crash_resume": {
            "status": resumed["status"],
            "refund_inr": _scenario_refund(resumed),
            "partial_tools": partial_checkpoint["tools_executed"],
            "resumed_tools": resumed["tools_executed"][len(partial_checkpoint["tools_executed"]):],
            "tools_across_both": resumed["tools_executed"],
            "llm_calls": resumed["llm_calls"],
            "tokens_used": resumed["tokens_used"],
        },
        "plan_and_execute_comparison": (
            "ReAct uses 5 calls for S1, 3 for S5, and 2 for S6 in these runs, paying more calls and tokens "
            "as each tool result returns to the model. A plan-and-execute design can ask for a whole known "
            "sequence once and use fewer calls, but it must replan when an observation differs from the plan: "
            "NOT_FOUND should stop lookup, and a cancelled order must not proceed to refund calculation. "
            "For refunds, ReAct is safer because each decision follows the observed database and policy result, "
            "while the fixed six-call bound provides a clear human handoff for loops."
        ),
        "ledger_entries": sum(1 for _ in ledger_path.open(encoding="utf-8")),
        "agent_run_entries": sum(1 for _ in runs_path.open(encoding="utf-8")),
        "zero_overlap_failure_questions": [item["question"] for item in no_overlap_failures],
    }
    (OUTPUTS / "part_d_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
