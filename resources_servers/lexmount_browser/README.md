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

**Two reward paths ship in this env:**

1. **Rule-based (default).** Deterministic, free, CI-safe — right for the bundled
   offline `site/` tasks. Zero external dependencies; nothing changes unless you
   opt in to the judge.
2. **Trajectory-level LLM judge (opt-in; the validated reward).** Real WebVoyager
   tasks have no rule-checkable end state — the reference training result under
   [Training](#training-reference-result) was produced by a binary trajectory
   judge, and `judge.py` ports it with the **verbatim** validated judge prompt
   (structured `{"reason","verdict":"yes|no"}` output; a SHA-256 unit test pins
   the wording). The judge sees the policy-delivered action/tool-result
   trajectory, the final live-browser URL/snapshot and the final answer
   (`<think>` blocks are stripped), samples at temperature 0 with up to 3
   attempts, and a failed judge is reported as `judge_status: "error"` — never
   conflated with a "no" verdict (the rollout still gets reward 0.0 for
   training-side safety, but the failure is visible in logs and in the verify
   response).

   Activation requires **both**:
   ```bash
   export JUDGE_BASE_URL=https://your-openai-compatible-gateway/v1
   export JUDGE_API_KEY=sk-...
   export JUDGE_MODEL=deepseek-v4-flash    # the validated judge model
   ```
   and a task that opts in (`verifier_metadata: {"judge": true}` — every row
   emitted by `scripts/convert_webvoyager.py --judge` / `fetch_webvoyager.sh`),
   or `judge_default: true` in `configs/lexmount_browser.yaml` to judge all
   tasks. With no `JUDGE_*` env vars, behavior is byte-for-byte the rule-based
   default. Unit tests (`tests/test_judge.py`) run against a mocked endpoint.

## Run

`example.sh` wraps everything below into one idempotent, fail-fast script with three
stages (`rollout` / `train` / `rollout --backend lexmount`). Read on for what each does.

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

### 3. Stage B — GRPO training smoke (1 GPU, via NeMo-RL)
```bash
bash example.sh train   # prints the exact NeMo-RL launch; gated behind a GPU
```
Uses `configs/grpo_lexmount_browser_smoke.yaml` (ports SXH's validated 0721
hyperparameters to a 1-GPU smoke; every value is annotated with its provenance) with
NeMo-RL's `examples/nemo_gym/run_grpo_nemo_gym.py`.

### 4. Stage C — same rollout on the Lexmount cloud backend (one flag)

Two Stage-C-specific facts:

- **The SDK must live in the *server* venv** — the resources server runs in its
  own venv at `resources_servers/lexmount_browser/.venv` (created by the first
  `gym env start`, e.g. by running Stage A once). Installing `lexmount` into the
  repo-root venv does nothing for the server process.
- **Stage C rolls out on real-web tasks**, not the bundled offline `site/`: the
  offline tasks are local `file://` URIs, which a cloud browser cannot load.
  `example.sh` uses the 3 bundled WebVoyager sample tasks
  (`data/webvoyager_sample.jsonl`, already in this env's input format) and
  writes rollouts to `data/webvoyager_rollouts.jsonl` (conservative
  `url_contains` reward — see [Data](#data-webvoyager-bridge)).

```bash
bash example.sh rollout            # Stage A once, so the server venv exists
uv pip install --python resources_servers/lexmount_browser/.venv/bin/python "lexmount>=0.5.13"
export LEXMOUNT_API_KEY=... LEXMOUNT_PROJECT_ID=... LEXMOUNT_BASE_URL=...
bash example.sh rollout --backend lexmount
```

## Data (WebVoyager bridge + the validated 168-task training set)

`scripts/convert_webvoyager.py` maps WebVoyager-style task JSON
(`{web_name, id, ques, web}`, [MinorJerry/WebVoyager](https://github.com/MinorJerry/WebVoyager),
**MIT license**) into this env's `example.jsonl` format (`initial_url` +
`verifier_metadata`), following the cleaning conventions of the validated 0721
pipeline (row-count / SHA-256 validation, duplicate-id rejection, source-id
preservation, task-agnostic system prompt — no answers or synthetic data injected).
The full WebVoyager set is **not bundled** (fetch it from upstream); three
sample tasks (Allrecipes--0 / Amazon--0 / GitHub--0) ship **already converted** in
`data/webvoyager_sample.jsonl` (directly usable as rollout input — Stage C uses it);
their raw upstream form is embedded in the converter for `--selftest`.

**`webvoyager-clean` — exactly what the validated run trained on.** One command
downloads the official tasks and reproduces the 168-task training set with every
step SHA-256-verified (byte-identical to the validated run's task manifest):

```bash
bash scripts/fetch_webvoyager.sh
#   upstream 643 tasks (pinned commit, sha256-checked)
#   -> drop "Cambridge Dictionary"                    = 600 tasks  sha256 b901adc3...
#   -> keep ArXiv / BBC News / Coursera / GitHub      = 168 tasks  sha256 db0dd8c1...
#      (the 4 sites that passed the site-availability probe; IDs cross-checked
#       against the committed data/webvoyager_clean_task_ids.txt)
#   -> data/webvoyager_clean.jsonl (env rollout inputs, verifier_metadata.judge: true)
```

Converted rows also carry a **conservative** `url_contains` spec (agent
reached/stayed on the task host) as the rule-based fallback — a rule pass is
necessary but not sufficient; full success needs the LLM judge (see
[Reward](#reward)).
```bash
python scripts/convert_webvoyager.py --selftest                       # no external data
python scripts/convert_webvoyager.py --source WebVoyager_data.jsonl \
    --output data/webvoyager_example.jsonl --limit 3 --judge
```

## Training (reference result)

The production recipe (colleague SXH's 0721 experiment) is a GRPO, full-parameter FSDP
run on **2×8 Ascend 910B** with **Qwen3-8B**: 8 tasks/step × 8 rollouts/task = 64
rollouts/step, 60 steps, 4 epochs, lr 5e-6 constant, context 40960 (4096 prompt /
36864 response), 10 assistant + 10 user turns, trajectory-level **LLM judge** reward
(`deepseek-v4-flash`), data = the 168-task `webvoyager-clean` set. The cloud
(Lexmount) arm's mean reward rose from **≈0.105** (first 10 steps) to **≈0.289**
(last 10 steps). `configs/grpo_lexmount_browser_smoke.yaml` scales this to a 1-GPU
smoke; `configs/grpo_lexmount_browser_full.yaml` is the full recipe on 8× H100.
Every value in both configs is annotated `validated:` or `adapted:`.

## Reproducing the RL growth curve

Everything needed is in this PR. Target: one node, **8× H100 80GB** (Qwen3-8B
full-parameter FSDP + colocated vLLM TP=4 rollouts fit comfortably; the
validated run needed 2×8 accelerators only because of 64 GB/NPU).

> **Framework/hardware caveat (read first).** The reference numbers were
> produced with **verl on Ascend 910B**; this PR trains with **NeMo-RL on
> NVIDIA GPUs**. Same data (SHA-verified), same judge prompt (SHA-pinned), same
> GRPO geometry, optimizer and budgets — different framework, kernels, and
> hardware. The promise is **recipe-level consistency**: expect the reward mean
> to climb from ≈0.10 (first 10 steps) to ≈0.29 (last 10 steps) over 60 steps,
> with per-step noise (64-rollout batches); do not expect bit-identical curves
> or per-step matches.

```bash
# 0. One-time setup: NeMo-RL + this Gym branch
git clone https://github.com/NVIDIA-NeMo/RL nemo-rl && cd nemo-rl
uv venv && source .venv/bin/activate && uv sync   # per NeMo-RL docs
git clone -b feat/lexmount-browser https://github.com/waple0820/Gym.git \
    3rdparty/Gym-workspace/Gym
ENV=3rdparty/Gym-workspace/Gym/resources_servers/lexmount_browser

# 1. Build the validated 168-task training set (SHA-verified end to end)
bash $ENV/scripts/fetch_webvoyager.sh

# 2. Secrets (never committed) — copy the template and fill it in
cp $ENV/secrets.env.example $ENV/secrets.env && chmod 600 $ENV/secrets.env
#    LEXMOUNT_API_KEY / LEXMOUNT_PROJECT_ID / LEXMOUNT_BASE_URL  (cloud browser)
#    JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL=deepseek-v4-flash (judge)
set -a; source $ENV/secrets.env; set +a

# 3. Launch the full recipe (60 steps; ~64 concurrent browser sessions at peak)
HF_HOME=$PWD/.cache/ uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
    --config=$ENV/configs/grpo_lexmount_browser_full.yaml \
    ++env.nemo_gym.lexmount_browser.resources_servers.lexmount_browser.backend=lexmount

# 4. Read the curve: TensorBoard scalar for per-step mean reward
tensorboard --logdir logs/grpo-lexmount-browser-full
```

Secrets checklist (all read from the environment at launch, see
`secrets.env.example` for per-variable provenance):

| Variable | Used by | Notes |
|---|---|---|
| `LEXMOUNT_API_KEY` / `LEXMOUNT_PROJECT_ID` / `LEXMOUNT_BASE_URL` | cloud browser backend | project must allow ~64 concurrent sessions (validated concurrency: 64 sessions / 16 creates) |
| `JUDGE_BASE_URL` / `JUDGE_API_KEY` / `JUDGE_MODEL` | LLM-judge reward | validated model `deepseek-v4-flash`; ≤64 judge calls per step |
| `POLICY_*` | example.sh rollout stages only | not used by training (NeMo-RL serves the policy) |

Sanity checks before the 60-step run: `bash example.sh rollout` (env wiring),
`bash example.sh rollout --backend lexmount` (cloud browser + creds),
`bash example.sh train` (1-GPU smoke), and a 2-step full-config dry run
(`++grpo.max_num_steps=2`) to confirm judge calls succeed
(`judge_status: "ok"` in logs — `"error"`/`"unconfigured"` means the judge
gateway or env vars need fixing, and rewards would silently fall back to 0/rule).

## Files (Gym `new-environment` spec)
- [x] `app.py` — resources server (seed_session + tools + verify)
- [x] `backend.py` — `BrowserBackend` + `PlaywrightBackend` + `LexmountBackend` (cloud SDK)
- [x] `judge.py` — opt-in trajectory-level LLM-judge reward (verbatim validated prompt)
- [x] `configs/lexmount_browser.yaml`
- [x] `configs/grpo_lexmount_browser_smoke.yaml` — 1-GPU GRPO smoke (NeMo-RL)
- [x] `configs/grpo_lexmount_browser_full.yaml` — full validated recipe on 8× H100 (NeMo-RL)
- [x] `site/` — bundled offline test site (deterministic tasks/CI)
- [x] `generate_data.py` + `data/example.jsonl` — 5 example tasks (Responses-API inputs)
- [x] `scripts/convert_webvoyager.py` + `data/webvoyager_sample.jsonl` — WebVoyager data bridge
- [x] `scripts/fetch_webvoyager.sh` + `data/webvoyager_clean_task_ids.txt` — the validated 168-task `webvoyager-clean` set, reproducible + SHA-verified
- [x] `secrets.env.example` — credentials template with per-variable provenance
- [x] `tests/test_backend.py` — standalone e2e backend test
- [x] `tests/test_judge.py` — judge unit tests against a mocked endpoint
- [x] `example.sh` — one-script Stage A/B/C reproduction (`train --full` = growth-curve recipe)
- [x] `requirements.txt`, `README.md`
- [x] `data/example_rollouts.jsonl` — 5 rollouts collected against a Responses-API endpoint (reward 1.0 on the offline site)
- [x] reward wiring validated end-to-end (Stage A); GRPO training-signal run documented above (Ascend 910B, Qwen3-8B)

## Licensing
- Environment code: Apache 2.0 (matches NeMo-Gym).
- Reference backend: Playwright (Apache 2.0).
- Example tasks: bundled offline `site/` is original (Apache 2.0). WebVoyager sample
  tasks in `data/webvoyager_sample.jsonl` are from WebVoyager (MIT); the full dataset
  is not redistributed here.
- Lexmount cloud SDK: a separate, optional dependency installed by the operator (not bundled); only needed for `backend: lexmount`.
