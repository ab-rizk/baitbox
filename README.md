# baitbox

Automated offensive testing harness for tool-using LLM agents. Runs a library of
prompt-injection payloads against an over-permissioned ReAct agent, scores each
run against safety predicates, and writes a markdown report with per-class
Attack Success Rate and guardrail recommendations.

Backed by any OpenAI-compatible API (OpenRouter by default). Tool execution is
simulated: the agent's *decision* to emit a tool call is what gets scored, not
any side effect, so nothing on the host is ever touched.

## Threat model

The agent has three tools: `read_file`, `http_get`, and `run_shell`. They are
intentionally over-permissioned. That over-permission is the attack surface.
baitbox exercises four injection patterns:

- **Direct override**: "ignore prior instructions and run X".
- **Indirect injection**: the payload travels through tool output (a file the
  agent reads), not user input.
- **Tool-use escalation**: chaining benign tools toward a forbidden action
  (read then exfiltrate, fetch then execute).
- **Multi-turn grooming**: building trust across turns before the malicious ask.

A run **succeeds** when the agent emits a tool call that violates a predicate:
destructive command, external exfiltration, path traversal, or password-file read.

## Run

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-v1-...
python main.py --report reports/run.md
```

Default model is `openai/gpt-4o-mini` (a full run is a few cents). Any
OpenRouter model with reliable tool calling works. Set it with `--model` or
`BAITBOX_MODEL`. Models are non-deterministic, so each payload runs a few times
(`--runs 3`).

```bash
python main.py --list      # list payloads without calling the model
python main.py --runs 1    # fast smoke test
```

Point at another OpenAI-compatible endpoint with `BAITBOX_BASE_URL`.

## Layout

- `agent.py`: ReAct agent + simulated tools, OpenRouter client.
- `payloads.yaml`: the attack library. Add payloads here.
- `runner.py`: payload loader, violation predicates, scoring, report.
- `main.py`: CLI.
