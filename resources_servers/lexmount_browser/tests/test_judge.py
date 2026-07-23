# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for the opt-in trajectory-level LLM judge (mocked endpoint only).

No GPU, no Gym stack, no real judge calls. Run with:

    uv run --no-project --with aiohttp --with pytest --with pytest-asyncio \
        python -m pytest tests/test_judge.py -q
"""

import hashlib
import json

import pytest
from aiohttp import web

from judge import (
    TASK_EVIDENCE_FINAL_ANSWER_JUDGE_PROMPT,
    JudgeConfig,
    _extract_structured_judge_result,
    _sanitize_final_answer,
    _truncate_middle,
    events_from_response_output,
    judge_trajectory,
    question_from_create_params,
    render_transcript,
)

# SHA-256 of the validated judge prompt as committed at nemorl-webagent@3220bc5
# (TASK_EVIDENCE_FINAL_ANSWER_JUDGE_PROMPT). The 0721 growth curve depends on
# this exact wording — this test fails on ANY edit to the prompt.
VALIDATED_PROMPT_SHA256 = "cb77e1ff49121feabdae9a1216271c9eaf66bfcbb406791e393d5d8390b8d7db"


def test_judge_prompt_is_verbatim():
    actual = hashlib.sha256(TASK_EVIDENCE_FINAL_ANSWER_JUDGE_PROMPT.encode("utf-8")).hexdigest()
    assert actual == VALIDATED_PROMPT_SHA256, (
        "TASK_EVIDENCE_FINAL_ANSWER_JUDGE_PROMPT no longer matches the validated "
        "0721 prompt (nemorl-webagent@3220bc5). Do not edit the prompt wording."
    )


# ----- structured verdict parsing (ported behavior) ------------------------- #
def test_extract_structured_judge_result():
    ok = _extract_structured_judge_result('{"reason":"navigated and verified","verdict":"yes"}')
    assert ok == {"reason": "navigated and verified", "verdict": "yes"}
    # markdown fence stripped
    fenced = _extract_structured_judge_result('```json\n{"reason":"r","verdict":"no"}\n```')
    assert fenced == {"reason": "r", "verdict": "no"}
    # rejected: extra keys, empty reason, non-binary verdict, non-JSON
    assert _extract_structured_judge_result('{"reason":"r","verdict":"yes","x":1}') is None
    assert _extract_structured_judge_result('{"reason":"","verdict":"yes"}') is None
    assert _extract_structured_judge_result('{"reason":"r","verdict":"maybe"}') is None
    assert _extract_structured_judge_result("the task was completed") is None


def test_sanitize_final_answer_strips_reasoning():
    assert _sanitize_final_answer("<think>secret chain</think>The answer is 42") == "The answer is 42"
    # unbalanced think tag -> drop everything (defense in depth)
    assert _sanitize_final_answer("<think>oops no close tag") == ""
    assert _sanitize_final_answer("plain answer<|im_end|>") == "plain answer"


def test_truncate_middle_marks_loss():
    text, truncated = _truncate_middle("x" * 100, 200)
    assert (text, truncated) == ("x" * 100, False)
    text, truncated = _truncate_middle("a" * 300, 100)
    assert truncated and len(text) == 100 and "...(truncated)..." in text


def test_render_transcript_shares_budget_across_turns():
    events = [{"call": f"browser_observe({{}})#{i}", "result": "r" * 50_000} for i in range(4)]
    transcript, truncated = render_transcript(events, transcript_char_limit=10_000)
    assert truncated
    assert len(transcript) <= 10_000
    # every turn survives (shared budget — no interaction disappears)
    for i in range(4):
        assert f"browser_observe({{}})#{i}" in transcript


# ----- Responses-API trace adapter ------------------------------------------ #
def _trace():
    return [
        {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": "I will look at the page."}]},
        {"type": "function_call", "name": "browser_observe", "arguments": "{}", "call_id": "c1"},
        {"type": "function_call_output", "call_id": "c1",
         "output": '{"observation":"URL: https://a.test\\nTITLE: Home","done":false}'},
        {"type": "function_call", "name": "browser_click",
         "arguments": '{"element_id": 3}', "call_id": "c2"},
        {"type": "function_call_output", "call_id": "c2",
         "output": '{"observation":"URL: https://a.test/done","done":false}'},
        {"type": "function_call", "name": "browser_finish",
         "arguments": '{"answer": "Task complete: 42"}', "call_id": "c3"},
        {"type": "function_call_output", "call_id": "c3", "output": '{"observation":"","done":true}'},
    ]


def test_events_from_response_output():
    events, final_answer = events_from_response_output(_trace())
    assert len(events) == 3
    assert events[0]["call"] == "browser_observe({})"
    assert "a.test" in events[0]["result"]
    assert events[1]["call"] == 'browser_click({"element_id": 3})'
    # explicit browser_finish answer wins over assistant prose
    assert final_answer == "Task complete: 42"


def test_question_from_create_params():
    params = {"input": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Find the star count."},
    ]}
    assert question_from_create_params(params) == "Find the star count."


# ----- end-to-end against a mocked judge endpoint --------------------------- #
class _MockJudge:
    """OpenAI-compatible /chat/completions stub with scripted replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    async def handle(self, request: web.Request) -> web.Response:
        payload = await request.json()
        self.requests.append(payload)
        reply = self.replies.pop(0) if self.replies else self.replies_exhausted()
        if isinstance(reply, int):  # HTTP error status
            return web.json_response({"error": "mock failure"}, status=reply)
        return web.json_response(
            {"choices": [{"message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}]}
        )

    @staticmethod
    def replies_exhausted():
        raise AssertionError("mock judge called more times than scripted")


async def _serve(mock: _MockJudge):
    app = web.Application()
    app.router.add_post("/v1/chat/completions", mock.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}/v1"


def _cfg(base_url: str) -> JudgeConfig:
    return JudgeConfig(base_url=base_url, api_key="test-key", model="mock-judge", request_timeout_s=10.0)


async def _judge_with(mock: _MockJudge, events=None):
    runner, base_url = await _serve(mock)
    try:
        return await judge_trajectory(
            _cfg(base_url),
            question="Find the star count.",
            events=events if events is not None else events_from_response_output(_trace())[0],
            final_answer="Task complete: 42",
            initial_state='{"start_url": "https://a.test"}',
            final_url="https://a.test/done",
            final_state="URL: https://a.test/done\nTITLE: Done",
        )
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_judge_yes_gives_reward_1():
    mock = _MockJudge(['{"reason":"trajectory shows completion","verdict":"yes"}'])
    result = await _judge_with(mock)
    assert (result.status, result.verdict, result.reward) == ("ok", "yes", 1.0)
    assert result.attempt_count == 1
    # the mocked endpoint received the verbatim prompt with our evidence in it
    sent = mock.requests[0]
    assert sent["model"] == "mock-judge"
    assert sent["temperature"] == 0.0 and sent["max_tokens"] == 1024
    prompt = sent["messages"][0]["content"]
    assert "Find the star count." in prompt
    assert "browser_click" in prompt
    assert "https://a.test/done" in prompt
    assert '{"reason":"a short evidence-based explanation","verdict":"yes|no"}' in prompt


@pytest.mark.asyncio
async def test_judge_no_gives_reward_0_with_ok_status():
    mock = _MockJudge(['{"reason":"never reached the target page","verdict":"no"}'])
    result = await _judge_with(mock)
    assert (result.status, result.verdict, result.reward) == ("ok", "no", 0.0)


@pytest.mark.asyncio
async def test_judge_retries_invalid_json_then_succeeds():
    mock = _MockJudge(["not json at all", '{"reason":"ok on retry","verdict":"yes"}'])
    result = await _judge_with(mock)
    assert (result.status, result.reward, result.attempt_count) == ("ok", 1.0, 2)


@pytest.mark.asyncio
async def test_judge_failure_is_distinguishable_from_no_verdict():
    # 3 scripted failures (HTTP 500, invalid JSON, HTTP 503) exhaust the
    # validated 3 attempts -> status "error", NOT a "no" verdict.
    mock = _MockJudge([500, "still not json", 503])
    result = await _judge_with(mock)
    assert result.status == "error"
    assert result.reward == 0.0
    assert result.verdict is None
    assert result.attempt_count == 3
    assert result.error_message
    assert len(mock.requests) == 3


@pytest.mark.asyncio
async def test_empty_trajectory_is_skipped_without_judge_call():
    mock = _MockJudge([])
    result = await _judge_with(mock, events=[])
    assert (result.status, result.reward) == ("skipped", 0.0)
    assert mock.requests == []  # no-tool-call short circuit never hits the judge
