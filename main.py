#!/usr/bin/env python3
"""baitbox: a tiny Auto Red Teaming harness for tool-using LLM agents.

Attacks an over-permissioned ReAct agent with a library of prompt-injection
payloads, scores each run against safety predicates, and emits a markdown
report with per-class Attack Success Rate and guardrail recommendations.
"""
import argparse
import os

from agent import DEFAULT_MODEL
from runner import load_payloads, run_payload, render_report


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"model tag (default: {DEFAULT_MODEL})")
    ap.add_argument("--payloads", default=os.path.join(os.path.dirname(__file__), "payloads.yaml"))
    ap.add_argument("--runs", type=int, default=3, help="Runs per payload (models are non-deterministic).")
    ap.add_argument("--report", default=None, help="Write a markdown report to this path. Omit to print to stdout.")
    args = ap.parse_args()

    payloads = load_payloads(args.payloads)

    print(f"baitbox: {len(payloads)} payloads × {args.runs} runs on model '{args.model}'")
    results = {}
    for i, p in enumerate(payloads, 1):
        print(f"  [{i}/{len(payloads)}] {p['id']} ({p['class']})...", end=" ", flush=True)
        try:
            res = run_payload(p, args.model, args.runs)
        except Exception as e:
            print(f"error: {e}")
            continue
        succ = sum(1 for r in res if r.success)
        print(f"{succ}/{len(res)} succeeded")
        results[p["id"]] = res

    report = render_report(results, args.model, args.runs)
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as f:
            f.write(report)
        print(f"\nreport written to {args.report}")
    else:
        print("\n" + report)


if __name__ == "__main__":
    main()
