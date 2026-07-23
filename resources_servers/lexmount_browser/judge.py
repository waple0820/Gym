# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trajectory-level LLM-judge reward for the lexmount_browser environment.

This ports the reward that produced the validated 0721 growth curve
(Qwen3-8B GRPO, reward mean ~0.105 -> ~0.289 over 60 steps) from the internal
verl/NeMo-Gym WebVoyager service (`nemorl-webagent@3220bc5`,
`training/ascend/nemo_gym_webvoyager_server.py`) into this environment:

  * ``TASK_EVIDENCE_FINAL_ANSWER_JUDGE_PROMPT`` is the VERBATIM validated judge
    prompt (binary, trajectory-level, structured ``{"reason","verdict"}``).
  * ``_truncate_middle`` / ``_render_transcript`` (shared evidence budget across
    turns) / ``_extract_structured_judge_result`` / ``_sanitize_final_answer``
    are ported 1:1; only the transcript event source differs (this env's
    Responses-API ``function_call`` / ``function_call_output`` items instead of
    the internal ``operation/instruction`` event log). The judge sees only
    policy-delivered evidence plus the final live-browser snapshot, matching
    the validated run's ``judge_evidence: policy_delivered_only`` contract.
  * The judge call is retried up to ``JUDGE_MAX_ATTEMPTS`` (default 3, the
    validated ``LEXBROWSER_JUDGE_MAX_ATTEMPTS``) with temperature 0.0 and
    ``max_tokens`` 1024; a failed judge is a distinguishable
    ``status="error"`` (never silently reward=0-as-verdict).

The judge is strictly OPT-IN: it activates only when the three ``JUDGE_*``
environment variables are set AND the task (``verifier_metadata.judge: true``)
or the server config (``judge_default: true``) asks for it. With no JUDGE_* env
vars there is zero behavior change to the rule-based ``verify()``.

Environment variables:
  JUDGE_BASE_URL   OpenAI-compatible base URL, e.g. https://www.dmxapi.cn/v1
  JUDGE_API_KEY    bearer token for that gateway
  JUDGE_MODEL      judge model id — the validated run used ``deepseek-v4-flash``
  JUDGE_MAX_ATTEMPTS            optional, default 3 (validated)
  JUDGE_TRANSCRIPT_CHAR_LIMIT   optional, default 60000 (validated)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

LOGGER = logging.getLogger(__name__)

# Validated default (internal LEXBROWSER_JUDGE_TRANSCRIPT_CHAR_LIMIT).
DEFAULT_TRANSCRIPT_CHAR_LIMIT = 60_000
DEFAULT_MAX_ATTEMPTS = 3
TRUNCATION_MARKER = "...(truncated)..."

# --------------------------------------------------------------------------- #
# VERBATIM judge prompt from nemorl-webagent@3220bc5
# (training/lexbrowser_webvoyager/src/lexbrowser_webvoyager_no_anti_bot/
#  environment.py :: TASK_EVIDENCE_FINAL_ANSWER_JUDGE_PROMPT).
# Do not edit the wording: the 0721 growth curve was produced with exactly this
# prompt, and recipe-level reproducibility depends on it.
# --------------------------------------------------------------------------- #
TASK_EVIDENCE_FINAL_ANSWER_JUDGE_PROMPT = """Judge whether the browser agent completed the task.

The input contains the task, optional rubric, initial environment state, execution status, complete recorded action/tool-result trajectory, final URL, final DOM/accessibility snapshot, optional screenshot evidence, and the policy's final response. Use the available evidence and be reasonably permissive when it supports completion.

Return exactly one JSON object with no Markdown and no additional keys:
{{"reason":"a short evidence-based explanation","verdict":"yes|no"}}

Task:
```
{question}
```

Rubric:
```
{rubric}
```

Initial Environment State:
```
{initial_state}
```

Execution Status:
```json
{execution_status}
```

Action and Tool-Result Trajectory:
```
{response}
```

Final URL:
```
{final_url}
```

Final DOM / Accessibility Snapshot:
```
{final_state}
```

Final Screenshot / Key Screenshots:
```
{screenshot_evidence}
```

Policy Final Response:
```
{final_answer}
```"""


@dataclass
class JudgeConfig:
    """Judge endpoint settings; ``enabled`` only when all three vars are set."""

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    transcript_char_limit: int = DEFAULT_TRANSCRIPT_CHAR_LIMIT
    request_timeout_s: float = 120.0

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    @classmethod
    def from_env(cls) -> "JudgeConfig":
        return cls(
            base_url=os.environ.get("JUDGE_BASE_URL", "").strip().rstrip("/"),
            api_key=os.environ.get("JUDGE_API_KEY", "").strip(),
            model=os.environ.get("JUDGE_MODEL", "").strip(),
            max_attempts=max(1, int(os.environ.get("JUDGE_MAX_ATTEMPTS", str(DEFAULT_MAX_ATTEMPTS)))),
            transcript_char_limit=int(
                os.environ.get("JUDGE_TRANSCRIPT_CHAR_LIMIT", str(DEFAULT_TRANSCRIPT_CHAR_LIMIT))
            ),
        )


@dataclass
class JudgeResult:
    """Outcome of one trajectory judgement.

    ``status`` distinguishes a *judge failure* from a *no verdict*:
      ok        — the judge returned a valid verdict; ``reward`` is authoritative
      skipped   — nothing to judge (no tool calls / empty transcript); reward 0
      error     — the judge endpoint failed after all attempts; reward is 0 but
                  MUST NOT be read as "task failed" (see ``error_message``)
    """

    reward: float
    status: str                     # "ok" | "skipped" | "error"
    verdict: Optional[str] = None   # "yes" | "no" when status == "ok"
    reason: str = ""
    attempt_count: int = 0
    duration_seconds: float = 0.0
    error_message: str = ""
    raw_response: str = ""
    attempts: List[Dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Ported 1:1 from nemorl-webagent@3220bc5 nemo_gym_webvoyager_server.py
# --------------------------------------------------------------------------- #
def _truncate_middle(text: str, limit: int) -> Tuple[str, bool]:
    """Match Verl's middle truncation while reporting evidence loss."""
    if len(text) <= limit:
        return text, False
    if limit <= len(TRUNCATION_MARKER):
        return text[:limit], True
    content_budget = limit - len(TRUNCATION_MARKER)
    left = content_budget // 2
    right = content_budget - left
    return text[:left] + TRUNCATION_MARKER + text[-right:], True


def _extract_structured_judge_result(raw_text: str) -> Optional[Dict[str, str]]:
    """Accept only a non-empty reason and a binary verdict."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    if set(payload) != {"reason", "verdict"}:
        return None
    if not isinstance(payload.get("verdict"), str) or not isinstance(payload.get("reason"), str):
        return None
    verdict = payload["verdict"].strip().lower()
    reason = payload["reason"].strip()
    if verdict not in {"yes", "no"} or not reason:
        return None
    return {"reason": reason, "verdict": verdict}


def _sanitize_final_answer(text: str) -> str:
    """Defense in depth: reasoning tags must never enter the Judge prompt."""
    cleaned = re.sub(
        r"<think\b[^>]*>.*?</think\s*>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    if re.search(r"</?think\b", cleaned, flags=re.IGNORECASE):
        return ""
    return re.sub(
        r"(?:<\|im_start\|>\s*assistant\s*|<\|im_end\|>|<\|endoftext\|>)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()


def render_transcript(
    events: List[Dict[str, str]], transcript_char_limit: int = DEFAULT_TRANSCRIPT_CHAR_LIMIT
) -> Tuple[str, bool]:
    """Share the Judge budget across all turns so no interaction disappears.

    ``events`` are ``{"call": "<one-line tool call>", "result": "<tool result>"}``
    pairs. The budget-sharing algorithm is the validated one; only the call-line
    wording differs (this env's tool names instead of the internal
    ``browser(operation=..., instruction=...)``).
    """
    if not events:
        return "", False
    call_lines = [f"TOOL_CALL {event['call']}" for event in events]
    fixed_chars = sum(len(line) + len("\nTOOL_RESULT: \n") for line in call_lines)
    result_budget = max(512, (transcript_char_limit - fixed_chars) // len(events))
    rendered: List[str] = []
    truncated = False
    for line, event in zip(call_lines, events):
        evidence, shortened = _truncate_middle(event["result"], result_budget)
        rendered.append(f"{line}\nTOOL_RESULT: {evidence}")
        truncated = truncated or shortened
    transcript, total_shortened = _truncate_middle("\n".join(rendered), transcript_char_limit)
    return transcript, truncated or total_shortened


# --------------------------------------------------------------------------- #
# Adapter: NeMo-Gym Responses-API trajectory -> judge evidence
# --------------------------------------------------------------------------- #
def events_from_response_output(output: List[Any]) -> Tuple[List[Dict[str, str]], str]:
    """Map this env's rollout trace to (tool events, final assistant answer).

    ``output`` is the agent-accumulated ``response.output`` list: assistant
    ``message`` items, ``function_call`` items and ``function_call_output``
    items, in rollout order (the policy-delivered evidence — the same evidence
    contract as the validated run).
    """

    def _get(item: Any, key: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    events: List[Dict[str, str]] = []
    results_by_call_id: Dict[str, str] = {}
    for item in output:
        if _get(item, "type") == "function_call_output":
            call_id = str(_get(item, "call_id", "") or "")
            results_by_call_id[call_id] = str(_get(item, "output", "") or "")

    final_answer = ""
    for item in output:
        item_type = _get(item, "type")
        if item_type == "function_call":
            name = str(_get(item, "name", "unknown_tool") or "unknown_tool")
            arguments = str(_get(item, "arguments", "{}") or "{}")
            call_id = str(_get(item, "call_id", "") or "")
            events.append(
                {
                    "call": f"{name}({arguments})",
                    "result": results_by_call_id.get(call_id, ""),
                }
            )
            if name == "browser_finish":
                try:
                    parsed = json.loads(arguments)
                    if isinstance(parsed, dict) and parsed.get("answer"):
                        final_answer = str(parsed["answer"])
                except (json.JSONDecodeError, TypeError):
                    pass
        elif item_type == "message" and _get(item, "role") == "assistant":
            content = _get(item, "content", [])
            texts: List[str] = []
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for part in content:
                    text = _get(part, "text", None)
                    if text:
                        texts.append(str(text))
            joined = "\n".join(t for t in texts if t).strip()
            if joined:
                # Later assistant prose supersedes earlier prose, but an explicit
                # browser_finish answer always wins (set above, never overwritten).
                if not final_answer:
                    final_answer = joined
    return events, final_answer


def question_from_create_params(params: Any) -> str:
    """Extract the user task text from ``responses_create_params.input``."""

    def _get(item: Any, key: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    inputs = _get(params, "input", None)
    if isinstance(inputs, str):
        return inputs
    if not isinstance(inputs, list):
        return ""
    for item in inputs:
        if _get(item, "role") == "user":
            content = _get(item, "content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [str(_get(p, "text", "") or "") for p in content]
                return "\n".join(p for p in parts if p)
    return ""


# --------------------------------------------------------------------------- #
# Judge call (retry x3, temperature 0.0, max_tokens 1024 — validated settings)
# --------------------------------------------------------------------------- #
async def judge_trajectory(
    config: JudgeConfig,
    *,
    question: str,
    events: List[Dict[str, str]],
    final_answer: str,
    initial_state: str,
    final_url: str,
    final_state: str,
    rubric: str = "",
    execution_status: Optional[Dict[str, Any]] = None,
) -> JudgeResult:
    """Score one trajectory 0/1 with the validated binary judge."""
    started = time.monotonic()
    transcript, transcript_truncated = render_transcript(events, config.transcript_char_limit)
    if not transcript:
        return JudgeResult(
            reward=0.0,
            status="skipped",
            reason="no_tool_calls",
            duration_seconds=time.monotonic() - started,
        )

    final_answer = _sanitize_final_answer(final_answer)
    status_payload: Dict[str, Any] = {
        "tool_call_count": len(events),
        "final_answer_present": bool(final_answer),
        "transcript_truncated": transcript_truncated,
    }
    if execution_status:
        status_payload.update(execution_status)

    prompt = TASK_EVIDENCE_FINAL_ANSWER_JUDGE_PROMPT.format(
        question=question,
        response=transcript,
        final_answer=final_answer.strip(),
        execution_status=json.dumps(status_payload, ensure_ascii=False, sort_keys=True),
        rubric=rubric.strip() or "Not provided for this task.",
        initial_state=initial_state,
        final_url=final_url or "Unavailable",
        final_state=final_state,
        screenshot_evidence="Unavailable: this run uses a text-only Judge.",
    )

    attempts: List[Dict[str, Any]] = []
    last_raw_text = ""
    last_error = ""
    timeout = aiohttp.ClientTimeout(total=config.request_timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for attempt in range(1, config.max_attempts + 1):
            try:
                async with session.post(
                    f"{config.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {config.api_key}"},
                    json={
                        "model": config.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "max_tokens": 1024,
                    },
                ) as http_response:
                    body = await http_response.json(content_type=None)
                    if http_response.status != 200:
                        raise RuntimeError(f"judge HTTP {http_response.status}: {json.dumps(body)[:500]}")
                message = body["choices"][0]["message"]
                last_raw_text = message.get("content") or ""
                reasoning_content = message.get("reasoning_content") or ""
                result = _extract_structured_judge_result(last_raw_text)
                if result is None and not last_raw_text.strip():
                    # Reasoning models can burn the budget in reasoning_content and
                    # return an empty assistant message (validated fallback).
                    result = _extract_structured_judge_result(reasoning_content)
                attempts.append(
                    {
                        "attempt": attempt,
                        "raw_response": last_raw_text,
                        "parsed_result": result,
                        "finish_reason": body["choices"][0].get("finish_reason"),
                    }
                )
                if result is not None:
                    return JudgeResult(
                        reward=1.0 if result["verdict"] == "yes" else 0.0,
                        status="ok",
                        verdict=result["verdict"],
                        reason=result["reason"],
                        attempt_count=attempt,
                        duration_seconds=time.monotonic() - started,
                        raw_response=last_raw_text,
                        attempts=attempts,
                    )
                last_error = "Judge returned empty or invalid reason/verdict JSON"
            except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, RuntimeError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                attempts.append(
                    {"attempt": attempt, "error_type": type(exc).__name__, "error_message": str(exc)}
                )
            LOGGER.warning("judge attempt %d/%d failed: %s", attempt, config.max_attempts, last_error)

    LOGGER.error("judge failed after %d attempts: %s", config.max_attempts, last_error)
    return JudgeResult(
        reward=0.0,
        status="error",
        attempt_count=config.max_attempts,
        duration_seconds=time.monotonic() - started,
        error_message=last_error or "Judge produced no valid reason/verdict JSON",
        raw_response=last_raw_text,
        attempts=attempts,
    )
