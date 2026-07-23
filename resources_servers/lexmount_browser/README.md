# Lexmount Browser — NeMo-Gym interactive-browser environment

A NeMo-Gym **stateful resources server** that turns a real browser into an RL
environment. Each rollout (`session_id`) owns one isolated live browser context;
the policy drives it with tool calls (navigate / click / type / observe / finish);
`verify()` returns the task reward from the live browser state.

This is the interactive complement to the existing read-only browsing envs
(`google_search`, `browsecomp_advanced_harness`): those search-and-extract text;
this one *operates* pages (stateful, multi-step web agency).

## Pluggable backend

The server depends only on the small `BrowserBackend` contract (`backend.py`):
`open / goto / click / type / observe / current_url / text / close`. Switch backend
with one config line; nothing else in the environment changes.

- `backend: playwright` — open-source reference (headless Chromium). Runnable today,
  used for local dev and CI (no proprietary deps). **Default.**
- `backend: lexmount` — production. Each rollout gets an isolated browser session in
  the Lexmount cloud (browser runs off the training node); we connect over CDP and
  reuse the same page-driving logic.

### Using the Lexmount cloud backend

1. Register at **https://browser.lexmount.com**, create a project, and copy your
   **API key** and **project ID**.
2. Install the SDK and export credentials (never commit them):
   ```bash
   pip install "lexmount>=0.5.13"
   export LEXMOUNT_API_KEY=<your-api-key>
   export LEXMOUNT_PROJECT_ID=<your-project-id>
   export LEXMOUNT_BASE_URL=https://api.lexmount.com   # API base shown in your dashboard
   ```
3. Switch one line in `configs/lexmount_browser.yaml` (nothing else changes —
   tools, observation, and reward are identical to the Playwright backend):
   ```yaml
   resources_servers:
     lexmount_browser:
       backend: lexmount        # was: playwright
   ```

## Tools & observation

Tools: `browser_navigate(url)`, `browser_click(element_id)`,
`browser_type(element_id, text)`, `browser_observe()`, `browser_finish(answer)`.
Observation is a **compact numbered list of interactive elements** (`[id] role: name`)
plus URL/title — deliberately token-cheap (raw HTML/pixels are far too expensive for
small policies and for training context length). `element_id`s come from the most
recent observation.

## Reward

`verify()` scores a per-task success spec in `verifier_metadata`:
`final_url` / `url_contains` / `dom_contains` / `answer_equals`. Sparse 0/1 outcome
reward by default (least reward-hackable); extend `_score()` with new keys as needed.
The rule reward is deterministic, free, and CI-safe — a natural extension point is a
judge-backed `_score()` branch behind a new `verifier_metadata` key for tasks with no
rule-checkable end state.

## Run

`example.sh` wraps everything below into one idempotent, fail-fast script with two
stages (`rollout` / `rollout --backend lexmount`). Read on for what each does.

### 0. Serve a policy model (the #1 reviewer stumbling block)

Stage A collects rollouts, so it needs a model endpoint that satisfies **both**:

1. **Speaks the Responses API** (`POST /v1/responses`). Chat-completions-only
   gateways do NOT work — the agent calls `/v1/responses` on the upstream with no
   chat fallback. Recent vLLM serves `/v1/responses` natively.
2. **Parses tool calls into structured `function_call` items.** If the server
   returns the model's tool-call markup as plain text (e.g. a literal
   `<tool_call>{"name": "browser_observe", ...}</tool_call>` string), the agent
   sees zero tool calls, the browser is never driven, and every rollout
   "succeeds" with **reward 0.0** — silently. For vLLM + Qwen-family models,
   launch with `--enable-auto-tool-choice --tool-call-parser hermes`.

Pick either:

- **A generic OpenAI-compatible endpoint (no local GPU).** Point at any server that
  implements the Responses API and export three vars; `example.sh` / the `openai_model`
  config read them:
  ```bash
  export POLICY_BASE_URL=https://your-endpoint/v1
  export POLICY_API_KEY=sk-...            # use any non-empty token if your gateway ignores it
  export POLICY_MODEL=your-model-name     # e.g. gpt-4.1
  ```
- **A locally served vLLM model (needs a GPU).** vLLM exposes the Responses API; serve
  a small tool-calling model and select the `vllm_model` config:
  ```bash
  export POLICY_KIND=vllm
  export POLICY_MODEL=Qwen/Qwen3-4B       # HF id or local checkpoint path
  ```

> The provided `example_rollouts.jsonl` was collected against a generic
> Responses-API endpoint. No base URL or key is stored in the committed rollouts.

### 1. Standalone backend test (no GPU, no Gym serving stack) — proves it works
```bash
uv run --no-project --with playwright python -m playwright install chromium
uv run --no-project --with playwright --with pytest --with pytest-asyncio python -m pytest tests/test_backend.py -q
```
Drives headless Chromium against the bundled offline `site/` (deterministic,
ToS-safe) and asserts navigate/click/type/observe + reward logic end-to-end.

> **Bare containers/VMs**: Chromium needs system libraries (libnss3, libgbm1,
> ...). If the install prints `Host system is missing dependencies to run
> browsers`, run (root/sudo required — `example.sh` attempts this automatically
> when it can):
> ```bash
> uv run --no-project --with playwright python -m playwright install-deps chromium
> ```
> Network note: setup downloads from pypi.org (deps), astral.sh (uv), and
> cdn.playwright.dev (~190 MB Chromium).

### 2. Stage A — rollouts as a NeMo-Gym environment (no GPU)
```bash
# after exporting POLICY_* from step 0:
bash example.sh rollout
```
which is equivalent to (run from the repo root, current-main CLI):
```bash
gym env start --resources-server lexmount_browser \
  --model-type openai_model --model "$POLICY_MODEL" \
  --model-url "$POLICY_BASE_URL" --model-api-key "$POLICY_API_KEY" &
gym eval run --no-serve --agent lexmount_browser_simple_agent \
  --input resources_servers/lexmount_browser/data/example.jsonl \
  --output resources_servers/lexmount_browser/data/example_rollouts.jsonl --limit 2
```
For training, plug into NeMo-RL GRPO via `examples/nemo_gym/run_grpo_nemo_gym.py`.

### 3. Stage C — same rollout on the Lexmount cloud backend (one flag)

Two Stage-C-specific facts:

- **The SDK must live in the *server* venv** — the resources server runs in its
  own venv at `resources_servers/lexmount_browser/.venv` (created by the first
  `gym env start`, e.g. by running Stage A once). Installing `lexmount` into the
  repo-root venv does nothing for the server process.
- **Stage C rolls out on real-web tasks**, not the bundled offline `site/`: the
  offline tasks are local `file://` URIs, which a cloud browser cannot load.
  `example.sh` uses the 3 bundled real-web tasks in `data/cloud_example.jsonl`
  (stable public pages: example.com / iana.org / wikipedia.org, with
  rule-checkable `url_contains` / `dom_contains` rewards) and writes rollouts
  to `data/cloud_rollouts.jsonl`.

```bash
bash example.sh rollout            # Stage A once, so the server venv exists
uv pip install --python resources_servers/lexmount_browser/.venv/bin/python "lexmount>=0.5.13"
export LEXMOUNT_API_KEY=... LEXMOUNT_PROJECT_ID=... LEXMOUNT_BASE_URL=...
bash example.sh rollout --backend lexmount
```

## Files (Gym `new-environment` spec)
- [x] `app.py` — resources server (seed_session + tools + verify)
- [x] `backend.py` — `BrowserBackend` + `PlaywrightBackend` + `LexmountBackend` (cloud SDK)
- [x] `configs/lexmount_browser.yaml`
- [x] `site/` — bundled offline test site (deterministic tasks/CI)
- [x] `generate_data.py` + `data/example.jsonl` — 5 example tasks (Responses-API inputs)
- [x] `data/cloud_example.jsonl` — 3 real-web tasks for the cloud backend (Stage C)
- [x] `tests/test_backend.py` — standalone e2e backend test
- [x] `example.sh` — one-script Stage A/C reproduction
- [x] `requirements.txt`, `README.md`
- [x] `data/example_rollouts.jsonl` — 5 rollouts collected against a Responses-API endpoint (reward 1.0 on the offline site)
- [x] reward wiring exercised end-to-end (Stage A rollouts score via `verify()`)
- [ ] GRPO training-signal run (train via NeMo-RL's `examples/nemo_gym/run_grpo_nemo_gym.py`; not part of this PR)

## Licensing
- Environment code: Apache 2.0 (matches NeMo-Gym).
- Reference backend: Playwright (Apache 2.0).
- Example tasks: bundled offline `site/` and the real-web tasks in
  `data/cloud_example.jsonl` are original (Apache 2.0).
- Lexmount cloud SDK: a separate, optional dependency installed by the operator (not bundled); only needed for `backend: lexmount`.
