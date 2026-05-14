#!/usr/bin/env python3
"""Summarize vitabench results.

For cross-domain runs (--domain delivery,instore,ota), shows aggregate only —
per-domain breakdown is not available because all tasks share domain='delivery,ota,instore'.
For single-domain runs (--domain delivery), per-domain rows appear automatically.

Pass columns:
  Pass@k  = fraction of tasks where ≥k trials passed  (from vita API pass_at_n)
  Pass^k  = statistically estimated P(k randomly sampled trials all pass)  (pass_hat_ks)

Usage:
    python score_summary.py                    # reads data/simulations/
    python score_summary.py data/simulations/
    python score_summary.py data/simulations/Qwen3.6-27B  # single file
"""

import json
import sys
import io
from collections import defaultdict
from pathlib import Path

SKIP_SUFFIXES = {".txt", ".json"}
N_TRIALS = 3  # expected trials (matches config_vllm.yaml num_trials)


def load_result(results_file: Path) -> dict | None:
    """Load vita Results for a single flat file."""
    try:
        _stdout = sys.stdout
        sys.stdout = io.StringIO()
        from vita.data_model.simulation import Results
        from vita.metrics.agent_metrics import compute_metrics
        sys.stdout = _stdout
        results = Results.load(results_file)
        m = compute_metrics(results)
    except Exception as e:
        sys.stdout = sys.__stdout__
        print(f"  WARNING: {results_file.name}: {e}", file=sys.stderr)
        return None

    n_tasks = len(results.tasks)
    n_sims = len(results.simulations)

    # Detect domains from tasks
    domains = sorted(set(getattr(t, "domain", "unknown") for t in results.tasks))
    domain_str = domains[0] if len(domains) == 1 else "cross-domain"

    return {
        "name": results_file.name,
        "tasks": n_tasks,
        "simulations": n_sims,
        "domain": domain_str,
        "avg_reward": m.avg_reward,
        "pass_at": dict(m.pass_at_n),
        "pass_hat": dict(m.pass_hat_ks),
    }


def find_result_files(root: Path) -> dict[str, Path]:
    """Find flat result files under root."""
    return {
        f.name: f
        for f in sorted(root.iterdir())
        if f.is_file() and f.suffix not in SKIP_SUFFIXES
    }


def _fmt(val, width=7) -> str:
    return f"{val:.3f}".rjust(width) if val is not None else "—".rjust(width)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/simulations")
    if not root.exists():
        print(f"Path not found: {root}")
        sys.exit(1)

    # Single file mode
    if root.is_file():
        r = load_result(root)
        if r:
            print(f"File:   {root}")
            print(f"Tasks:  {r['tasks']}")
            print(f"Domain: {r['domain']}")
            print(f"Avg:    {r['avg_reward']:.4f}")
            for k in sorted(set(r["pass_at"]) | set(r["pass_hat"])):
                at  = r["pass_at"].get(k)
                hat = r["pass_hat"].get(k)
                print(f"  Pass@{k}: {at:.4f}   Pass^{k}: {hat:.4f}")
        return

    result_files = find_result_files(root)
    if not result_files:
        print(f"No result files found in {root}")
        sys.exit(1)

    # Load all results
    cache: dict[str, dict] = {}
    max_k = 1
    for name, f in result_files.items():
        r = load_result(f)
        if r and r["tasks"] > 0:
            cache[name] = r
            if r["pass_at"]:
                max_k = max(max_k, max(r["pass_at"].keys()))
            if r["pass_hat"]:
                max_k = max(max_k, max(r["pass_hat"].keys()))
    max_k = max(max_k, N_TRIALS)

    k_values = list(range(1, max_k + 1))
    sorted_models = sorted(cache.keys())

    # ── Header ────────────────────────────────────────────────────────────────
    # Pass@1 (any pass) differs from Pass^1 (per-trial rate) only when k>1.
    # For k≥2 Pass@k == Pass^k (both = fraction with ≥k passes), so omit Pass@k for k≥2.
    hat_cols = "  ".join(f"{'Pass^'+str(k):>7s}" for k in k_values if k >= 2)
    header = f"{'Model':<25s}  {'Domain':<12s}  {'Tasks':>5s}  {'Avg':>5s}  {'Pass@1':>7s}  {'Pass^1':>7s}  {hat_cols}"
    print(header)
    print("─" * len(header))

    json_out = []
    for name in sorted_models:
        r = cache[name]
        pass_vals = (
            _fmt(r["pass_at"].get(1)) + "  " +
            _fmt(r["pass_hat"].get(1)) + "  " +
            "  ".join(_fmt(r["pass_hat"].get(k)) for k in k_values if k >= 2)
        )
        print(
            f"{name:<25s}  {r['domain']:<12s}  {r['tasks']:>5d}"
            f"  {r['avg_reward']:>5.3f}  {pass_vals}"
        )
        json_out.append({
            "model": name,
            "tasks": r["tasks"],
            "simulations": r["simulations"],
            "domain": r["domain"],
            "avg_reward": round(r["avg_reward"], 4),
            "pass_at": {str(k): round(v, 4) for k, v in r["pass_at"].items()},
            "pass_hat": {str(k): round(v, 4) for k, v in r["pass_hat"].items()},
        })

    if any(r["domain"] == "cross-domain" for r in cache.values()):
        print(
            "\nNote: cross-domain tasks (delivery+instore+ota combined). "
            "Per-domain breakdown requires separate single-domain runs "
            "(--domain delivery / --domain instore / --domain ota)."
        )

    print()
    print(f"Pass@1 = fraction of tasks where ≥1 trial passed  (optimistic; upper bound on capability)")
    print(f"Pass^1 = per-trial pass rate: passing trials / total trials  (differs from Pass@1 when n_trials > 1)")
    print(f"Pass^k = fraction of tasks where ≥k trials passed  (k≥2; Pass@k omitted — same value)")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out_file = root / "score_summary.json"
    with open(out_file, "w") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved to {out_file}")


if __name__ == "__main__":
    main()
