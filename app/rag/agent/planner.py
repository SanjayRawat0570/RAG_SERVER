"""Agentic retrieval — the state machine behind ``mode="agentic"``.

States::

    plan ──▶ search ──▶ evaluate ──▶ answer
      ▲                    │
      └────── rewrite ◀────┘

* **plan**     — decide which tool calls to make (or that we can answer now)
* **search**   — execute the pending tool calls (tenant injected server-side)
* **evaluate** — is the gathered evidence enough? enforce the hard limits
* **rewrite**  — not enough: reformulate (e.g. hop to a bridge entity) and plan again
* **answer**   — stop; the normal generate step writes the cited answer

Two planners share the same states:

* **LLM planner** (Gemini function calling): the model sees the question and the
  tool results so far and either calls more tools or replies ``DONE``. Its
  reasoning stays internal — only tool calls and rewritten queries are traced.
* **Heuristic planner** (offline / no tool-capable provider): search the
  question, look up named entities in the graph, and when evidence coverage is
  low, rewrite the question around a *bridge entity* found in the evidence
  ("the team that owns Project Atlas" → "the Data Platform team").

Hard limits (checked before every plan and after every search): max tool rounds,
wall-clock timeout, and an LLM token budget.
"""
from __future__ import annotations

import re
import time
from typing import Any

from app.config import settings
from app.rag.agent.tools import TOOL_NAMES, TOOL_SPECS, ToolContext, run_tool
from app.rag.cost import BudgetExceededError
from app.rag.llm.gateway import call_llm, is_configured
from app.rag.llm.registry import get_llm
from app.rag.search import tokenize

ANSWER, PLAN, SEARCH, EVALUATE, REWRITE = "answer", "plan", "search", "evaluate", "rewrite"
MAX_CALLS_PER_ROUND = 3
COVERAGE_TO_ANSWER = 0.8

_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are", "was", "were",
    "that", "which", "who", "whom", "whose", "what", "how", "when", "where", "why", "do", "does",
    "did", "can", "could", "with", "at", "by", "be", "as", "it", "this", "our", "my", "many", "much",
}
_ENTITY = re.compile(r"\b[A-Z][\w-]*(?:\s+[A-Z][\w-]*)*")
_QUESTION_WORDS = {"Who", "What", "Which", "Where", "When", "How", "Why", "In", "Is", "Are", "Does", "Do", "Can", "The"}
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

PLANNER_SYSTEM = (
    "You are the retrieval planner for an enterprise knowledge base. Use the tools to gather "
    "the evidence needed to answer the user's question. Questions may need several hops — "
    "for example first find which team owns a project, then who leads that team. Prefer "
    "specific, rewritten queries over repeating the question. You have {rounds_left} tool "
    "round(s) left. When the evidence is sufficient, or nothing more can be found, reply with "
    "the single word DONE and call no tools. Do not write the final answer yourself."
)


# ── State ─────────────────────────────────────────────────────────────────────

def new_state(question: str, tenant: str, provider: str, *, model: str | None = None,
              max_rounds: int | None = None,
              timeout_s: float | None = None, token_budget: int | None = None) -> dict[str, Any]:
    return {
        "question": question,
        "tenant": tenant,
        "provider": provider,
        "model": model,
        "planner": "llm" if supports_tools(provider) else "heuristic",
        "phase": PLAN,
        "round": 0,
        "max_rounds": max_rounds or settings.agent_max_rounds,
        "timeout_s": timeout_s or settings.agent_timeout_s,
        "token_budget": token_budget or settings.agent_token_budget,
        "tokens_used": 0,
        "started_at": time.time(),
        "pending_calls": [],
        "next_calls": [],
        "queries": [],
        "answer_query": question,
        "evidence": [],
        "facts": [],
        "transcript": [{"role": "user", "content": f"Question: {question}"}],
        "steps": [],
        "stop_reason": None,
        "rewritten_query": None,
    }


def supports_tools(provider: str) -> bool:
    try:
        return bool(getattr(get_llm(provider), "supports_tools", False)) and is_configured(provider)
    except ValueError:
        return False


def limit_reached(state: dict[str, Any]) -> str | None:
    if state["round"] >= state["max_rounds"]:
        return "max_rounds"
    if time.time() - state["started_at"] > state["timeout_s"]:
        return "timeout"
    if state["tokens_used"] >= state["token_budget"]:
        return "token_budget"
    return None


def _step(state: dict[str, Any], kind: str, **data: Any) -> None:
    state["steps"].append({"type": kind, "round": state["round"], **data})


def finish(state: dict[str, Any], reason: str) -> dict[str, Any]:
    state["phase"] = ANSWER
    state["stop_reason"] = reason
    state["pending_calls"] = []
    rewrites = [s["query"] for s in state["steps"] if s["type"] == "rewrite"]
    state["rewritten_query"] = rewrites[-1] if rewrites else None
    _step(state, "answer", reason=reason, evidence=len(state["evidence"]),
          answer_query=state["answer_query"])
    return state


# ── Evidence helpers ──────────────────────────────────────────────────────────

_IRREGULAR = {"led": "lead", "ran": "run", "built": "build", "made": "make", "paid": "pay",
              "held": "hold", "sold": "sell", "bought": "buy", "kept": "keep", "met": "meet"}
# "the team that owns X", "the person responsible for Y" — needs an intermediate hop.
_MULTI_HOP = re.compile(r"\bthe\s+\w+\s+(?:that|which|who|responsible\s+for)\b", re.IGNORECASE)


def _stem(token: str) -> str:
    token = _IRREGULAR.get(token, token)
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def _content(text: str) -> set[str]:
    return {_stem(t) for t in tokenize(text) if t not in _STOP}


def coverage(query: str, evidence: list[dict[str, Any]], facts: list[str]) -> float:
    """Best share of the query's content words found in any *single* passage.

    Per passage, not pooled: words scattered over several chunks do not make an
    answer, and the final answer is grounded in individual passages.
    """
    want = _content(query)
    if not want:
        return 1.0
    texts = [str(hit.get("metadata", {}).get("text", "")) for hit in evidence] + facts
    return max((len(want & _content(t)) / len(want) for t in texts), default=0.0)


def needs_hop(state: dict[str, Any]) -> bool:
    """The original question is relational and no hop has been made yet."""
    return bool(_MULTI_HOP.search(state["question"])) and state["answer_query"] == state["question"]


def question_entities(question: str) -> list[str]:
    out = []
    for m in _ENTITY.finditer(question):
        words = m.group(0).split()
        while words and words[0] in _QUESTION_WORDS:
            words = words[1:]
        if words:
            out.append(" ".join(words))
    return list(dict.fromkeys(out))


def bridge_rewrite(state: dict[str, Any]) -> str | None:
    """Rewrite a multi-hop question around an entity found in the evidence.

    "Who leads the team that owns Project Atlas?" + evidence "Project Atlas is
    owned by the Data Platform team" → "Who leads the Data Platform team?"
    """
    question = state["question"]
    entities = question_entities(question)
    if not entities:
        return None
    qwords = [w.lower().strip("?.,") for w in question.split()]
    texts = [str(h.get("metadata", {}).get("text", "")) for h in state["evidence"]] + state["facts"]
    for text in texts:
        for sentence in _SENTENCE.split(text.replace("\n", " ")):
            if not any(e in sentence for e in entities):
                continue
            for m in _ENTITY.finditer(sentence):
                cand = m.group(0)
                if cand in _QUESTION_WORDS or any(cand in e or e in cand for e in entities):
                    continue
                cand = " ".join(w for w in cand.split() if w not in _QUESTION_WORDS)
                if not cand:
                    continue
                # The noun right after the bridge entity ("team") anchors the rewrite.
                after = sentence[m.end():].split()
                noun = after[0].lower().strip(".,") if after else ""
                if noun and noun in qwords:
                    head = question.split()[: qwords.index(noun)]
                    if head and head[-1].lower() == "the":
                        head = head[:-1]
                    return " ".join([*head, "the", cand, noun]).strip() + "?"
    return None


def _add_evidence(state: dict[str, Any], hits: list[dict[str, Any]], *, front: bool = False) -> int:
    """Merge *hits* into the evidence (deduplicated by id); returns how many were new.

    ``front=True`` moves these hits — new or already known — to the front in their
    ranked order, so the most recent (usually rewritten) search leads the context.
    """
    known = {h["id"] for h in state["evidence"]}
    fresh = [h for h in dict((h["id"], h) for h in hits if h.get("id")).values()]
    added = sum(1 for h in fresh if h["id"] not in known)
    if front:
        ids = {h["id"] for h in fresh}
        state["evidence"] = fresh + [h for h in state["evidence"] if h["id"] not in ids]
    else:
        state["evidence"].extend(h for h in fresh if h["id"] not in known)
    return added


# ── States ────────────────────────────────────────────────────────────────────

async def plan(state: dict[str, Any]) -> dict[str, Any]:
    if state["phase"] == ANSWER:
        return state
    if reason := limit_reached(state):
        return finish(state, reason)

    if state["planner"] == "llm":
        try:
            calls = await _llm_plan(state)
        except BudgetExceededError:
            return finish(state, "budget_exceeded")
        except Exception as exc:  # provider down → continue offline
            state["planner"] = "heuristic"
            _step(state, "planner_fallback", error=str(exc)[:200])
            calls = _heuristic_plan(state)
    else:
        calls = _heuristic_plan(state)

    calls = [c for c in calls if c.get("name") in TOOL_NAMES][:MAX_CALLS_PER_ROUND]
    if not calls:
        return finish(state, "planner_done")
    state["pending_calls"] = calls
    state["phase"] = SEARCH
    _step(state, "plan", calls=[{"tool": c["name"], "args": c.get("args", {})} for c in calls])
    return state


async def search(state: dict[str, Any]) -> dict[str, Any]:
    if state["phase"] != SEARCH:
        return state
    import asyncio

    ctx = ToolContext(tenant=state["tenant"])           # tenant from the server, never the model
    calls = state["pending_calls"]
    results = await asyncio.gather(*(run_tool(c["name"], c.get("args"), ctx) for c in calls))
    state["round"] += 1
    responses = []
    for call, res in zip(calls, results):
        added = _add_evidence(state, res["hits"], front=res["name"] == "search_docs")
        state["facts"].extend(f for f in res["facts"] if f not in state["facts"])
        if res["name"] == "search_docs" and "query" in res["args"]:
            state["queries"].append(str(res["args"]["query"]))
        _step(state, "tool_call", tool=res["name"], args=res["args"],
              results=len(res["hits"]) or len(res["facts"]), new_evidence=added,
              sources=sorted({h.get("metadata", {}).get("source") for h in res["hits"]} - {None}),
              dropped_args=res["dropped_args"], error=res["error"], duration_ms=res["duration_ms"])
        responses.append({"functionResponse": {"name": res["name"],
                                               "response": res["summary"] or {"error": res["error"]}}})
    if state["planner"] == "llm":
        state["transcript"].append({"role": "user", "content": "", "parts": responses})
    state["pending_calls"] = []
    state["phase"] = EVALUATE
    return state


async def evaluate(state: dict[str, Any]) -> dict[str, Any]:
    if state["phase"] != EVALUATE:
        return state
    # Judge against the current query: after a hop, the rewritten one.
    cov = round(coverage(state["answer_query"], state["evidence"], state["facts"]), 3)
    _step(state, "evaluate", coverage=cov, evidence=len(state["evidence"]))
    hop = state["planner"] == "heuristic" and needs_hop(state)
    if not hop and (cov >= 0.999 or (state["planner"] == "heuristic" and cov >= COVERAGE_TO_ANSWER)):
        return finish(state, "sufficient_evidence")
    if reason := limit_reached(state):
        return finish(state, reason)

    if state["planner"] == "heuristic":
        rewritten = bridge_rewrite(state)
        if not rewritten or rewritten in state["queries"]:
            return finish(state, "no_new_query")
        state["next_calls"] = [{"name": "search_docs", "args": {"query": rewritten}}]
        state["answer_query"] = rewritten
        _step(state, "rewrite", query=rewritten)
    state["phase"] = REWRITE
    return state


# ── Planners ──────────────────────────────────────────────────────────────────

def _heuristic_plan(state: dict[str, Any]) -> list[dict[str, Any]]:
    if state["round"] == 0:
        calls = [{"name": "search_docs", "args": {"query": state["question"]}}]
        calls += [{"name": "lookup_entity", "args": {"name": e}}
                  for e in question_entities(state["question"])[:2]]
        return calls
    calls, state["next_calls"] = state["next_calls"], []
    return calls


async def _llm_plan(state: dict[str, Any]) -> list[dict[str, Any]]:
    rounds_left = state["max_rounds"] - state["round"]
    # The agent deadline also bounds the call itself, not just the gaps between calls.
    remaining = state["timeout_s"] - (time.time() - state["started_at"])
    request = {"messages": [
        {"role": "system", "content": PLANNER_SYSTEM.format(rounds_left=rounds_left)},
        *state["transcript"],
    ]}
    response = await call_llm(
        state["provider"], request,
        {"tools": TOOL_SPECS, "temperature": 0.0, "max_tokens": 1024,
         "timeout": max(2.0, min(float(settings.llm_timeout), remaining)),
         **({"model": state["model"]} if state.get("model") else {})},
        tenant=state["tenant"], operation="agent_plan",
    )
    usage = response.usage or {}
    state["tokens_used"] += int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
    if response.tool_calls:
        state["transcript"].append({"role": "assistant", "content": "", "parts": response.parts})
        for call in response.tool_calls:
            if call["name"] == "search_docs" and call["args"].get("query") not in (None, state["question"]):
                state["answer_query"] = state["question"]  # the LLM writes the final answer itself
                _step(state, "rewrite", query=call["args"]["query"])
    return response.tool_calls
