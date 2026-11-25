"""Per-run records, task-diversity checks and aggregation for sequential evaluations.

Used by scripts/run_pointmaze_corrected.sh (works for any task run with
`+eval_seed_override`, e.g. Wall). Subcommands:

  record    write <run_dir>/run.json after one plan.py run
  precheck  sample start/goal for a seed range straight from the env (no GPU, no model)
  verify    check that all completed runs have distinct start/goal pairs
  aggregate metrics, comparison with the previous protocol, figures, LaTeX table

Only runs whose run.json says "complete": true are ever counted.
"""
import argparse, glob, json, os, pickle, re, sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_SR = {"point_maze": 0.98, "wall": 0.96}
DECIMALS = 4  # start/goal states are compared after rounding to this many decimals


def _final_eval(run_dir):
    path = os.path.join(run_dir, "logs.json")
    final, mpc_iters = None, 0
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            if any(k.startswith("final_eval/") for k in j):
                final = {k.split("/", 1)[1]: v for k, v in j.items()}
            elif "step" in j:
                mpc_iters = max(mpc_iters, int(j["step"]))
    return final, mpc_iters


def _targets(run_dir):
    path = os.path.join(run_dir, "plan_targets.pkl")
    if not os.path.exists(path):
        return None, None
    p = pickle.load(open(path, "rb"))
    return np.asarray(p["state_0"]).reshape(-1), np.asarray(p["state_g"]).reshape(-1)


def _pair_key(s0, sg):
    return tuple(np.round(np.concatenate([s0, sg]), DECIMALS).tolist())


# ---------------------------------------------------------------- record
def cmd_record(a):
    final, mpc_iters = _final_eval(a.run_dir)
    s0, sg = _targets(a.run_dir)
    printed_seed = None
    if a.stdout and os.path.exists(a.stdout):
        m = re.search(r"eval_seed:\s*\[([^\]]*)\]", open(a.stdout, errors="ignore").read())
        if m:
            printed_seed = [int(x) for x in m.group(1).split(",") if x.strip()]
    seed_ok = printed_seed == [a.eval_seed]
    complete = a.exit_code == 0 and final is not None and s0 is not None and seed_ok
    if a.timed_out:
        status = "timeout"
    elif not seed_ok and printed_seed is not None:
        status = "seed_mismatch"
    elif not complete:
        status = "error"
    else:
        status = "success" if final["success_rate"] >= 1.0 else "failure"
    rec = dict(
        eval_seed=a.eval_seed, printed_eval_seed=printed_seed, global_seed=a.global_seed,
        status=status, complete=complete, exit_code=a.exit_code, timed_out=bool(a.timed_out),
        success=bool(final and final["success_rate"] >= 1.0),
        final_state_dist=final and final.get("mean_state_dist"),
        final_visual_dist=final and final.get("mean_visual_dist"),
        final_proprio_dist=final and final.get("mean_proprio_dist"),
        mpc_iterations=mpc_iters, max_mpc_iters=a.max_mpc_iters,
        initial_state=None if s0 is None else s0.tolist(),
        goal_state=None if sg is None else sg.tolist(),
        output_dir=os.path.relpath(os.path.abspath(a.run_dir), ROOT),
        stdout_log=a.stdout and os.path.relpath(os.path.abspath(a.stdout), ROOT),
        wall_time_s=a.wall_time, recorded_at=datetime.now().isoformat(timespec="seconds"),
    )
    with open(os.path.join(a.run_dir, "run.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[record] seed {a.eval_seed}: {status}"
          + (f", final state dist {rec['final_state_dist']:.3f}" if rec["final_state_dist"] is not None else ""))
    return 0 if complete else 1


# ---------------------------------------------------------------- load
def load_runs(exp_dir):
    runs = []
    for p in sorted(glob.glob(os.path.join(exp_dir, "results", "seed_*", "run.json"))):
        r = json.load(open(p))
        if r.get("complete"):
            runs.append(r)
    return sorted(runs, key=lambda r: r["eval_seed"])


def diversity(pairs):
    keys = [_pair_key(np.asarray(s0), np.asarray(sg)) for s0, sg in pairs]
    dup = {}
    for i, k in enumerate(keys):
        dup.setdefault(k, []).append(i)
    return len(set(keys)), [v for v in dup.values() if len(v) > 1]


# ---------------------------------------------------------------- precheck
def cmd_precheck(a):
    sys.path.insert(0, ROOT)
    os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    import gym, env  # noqa: F401  (registers the envs)
    e = gym.make(a.env)
    seeds = list(range(a.first, a.last + 1))
    pairs = []
    for s in seeds:
        s0, sg = e.sample_random_init_goal_states(s)
        pairs.append((np.asarray(s0).reshape(-1), np.asarray(sg).reshape(-1)))
    n, dups = diversity(pairs)
    print(f"[precheck] env={a.env} seeds {a.first}..{a.last}: {n} distinct start/goal pairs out of {len(seeds)}")
    for s, (s0, sg) in list(zip(seeds, pairs))[: a.show]:
        print(f"  seed {s:>2}: start {np.round(s0[:2], 3).tolist()}  goal {np.round(sg[:2], 3).tolist()}")
    for d in dups:
        print("  DUPLICATE seeds:", [seeds[i] for i in d])
    # the old protocol (n_evals=1, any seed) always used eval seed 1
    old = e.sample_random_init_goal_states(1)
    print(f"  old protocol (eval_seed always 1): start {np.round(np.asarray(old[0])[:2], 3).tolist()}")
    return 0 if n == len(seeds) else 1


# ---------------------------------------------------------------- verify
def cmd_verify(a):
    runs = load_runs(a.exp_dir)
    if not runs:
        print("[verify] no completed runs yet"); return 1
    n, dups = diversity([(r["initial_state"], r["goal_state"]) for r in runs])
    seeds = [r["eval_seed"] for r in runs]
    bad_seed = [r["eval_seed"] for r in runs if r["printed_eval_seed"] != [r["eval_seed"]]]
    print(f"[verify] completed runs: {len(runs)}; distinct start/goal pairs: {n}")
    print(f"[verify] eval seeds used: {seeds[0]}..{seeds[-1]} ({len(set(seeds))} unique)")
    for d in dups:
        print("  DUPLICATE task across eval seeds:", [runs[i]["eval_seed"] for i in d])
    if bad_seed:
        print("  plan.py printed a different eval seed for:", bad_seed)
    ok = n == len(runs) and not bad_seed
    print("[verify] PASS: every completed run is a distinct task" if ok else "[verify] FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------- previous protocol
def previous_runs(log_glob):
    out = []
    for log in sorted(glob.glob(os.path.join(ROOT, log_glob))):
        t = open(log, errors="ignore").read()
        m = re.search(r"Planning result saved dir: (\S+)", t)
        if not m:
            continue
        d = m.group(1)
        final, _ = _final_eval(d)
        s0, sg = _targets(d)
        if final is None or s0 is None:
            continue
        out.append(dict(success=final["success_rate"] >= 1.0, final_state_dist=final["mean_state_dist"],
                        final_visual_dist=final["mean_visual_dist"], final_proprio_dist=final["mean_proprio_dist"],
                        initial_state=s0.tolist(), goal_state=sg.tolist()))
    return out


def summarize(runs):
    if not runs:
        return None
    n_distinct, _ = diversity([(r["initial_state"], r["goal_state"]) for r in runs])
    m = lambda k: float(np.mean([r[k] for r in runs]))
    return dict(total=len(runs), distinct=n_distinct, successes=sum(r["success"] for r in runs),
                success_rate=sum(r["success"] for r in runs) / len(runs),
                mean_state_dist=m("final_state_dist"), mean_visual_dist=m("final_visual_dist"),
                mean_proprio_dist=m("final_proprio_dist"))


# ---------------------------------------------------------------- aggregate
def cmd_aggregate(a):
    runs = load_runs(a.exp_dir)
    all_recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.exp_dir, "results", "seed_*", "run.json")))]
    incomplete = sorted(r["eval_seed"] for r in all_recs if not r.get("complete"))
    cur, prev = summarize(runs), summarize(previous_runs(a.previous_logs)) if a.previous_logs else None
    paper = PAPER_SR.get(a.env)
    res_dir = os.path.join(a.exp_dir, "results"); fig_dir = os.path.join(a.exp_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    if cur is None:
        print("[aggregate] no completed runs yet"); return 1
    finished = cur["total"] == a.expected and not incomplete

    # per-run CSV
    import csv
    with open(os.path.join(res_dir, "per_run.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eval_seed", "status", "success", "mpc_iterations", "final_state_dist", "final_visual_dist",
                    "final_proprio_dist", "initial_state", "goal_state", "output_dir"])
        for r in runs:
            w.writerow([r["eval_seed"], r["status"], r["success"], r["mpc_iterations"], r["final_state_dist"],
                        r["final_visual_dist"], r["final_proprio_dist"],
                        " ".join(f"{x:.4f}" for x in r["initial_state"]),
                        " ".join(f"{x:.4f}" for x in r["goal_state"]), r["output_dir"]])

    summary = dict(env=a.env, expected_runs=a.expected, finished=finished, incomplete_seeds=incomplete,
                   failed_seeds=[r["eval_seed"] for r in runs if not r["success"]],
                   corrected=cur, previous=prev, paper_success_rate=paper)
    json.dump(summary, open(os.path.join(res_dir, "summary.json"), "w"), indent=1)

    pct = lambda x: f"{100 * x:.0f}%"
    status_note = "" if finished else f" (PARTIAL: {cur['total']}/{a.expected} runs complete)"
    rows = [("Total runs", prev and prev["total"], cur["total"]),
            ("Distinct start/goal tasks", prev and prev["distinct"], cur["distinct"]),
            ("Successful runs", prev and prev["successes"], cur["successes"]),
            ("Success rate", prev and pct(prev["success_rate"]), pct(cur["success_rate"])),
            ("Mean final state distance", prev and f"{prev['mean_state_dist']:.3f}", f"{cur['mean_state_dist']:.3f}"),
            ("Mean final visual distance", prev and f"{prev['mean_visual_dist']:.3f}", f"{cur['mean_visual_dist']:.3f}"),
            ("Mean final proprio distance", prev and f"{prev['mean_proprio_dist']:.3f}", f"{cur['mean_proprio_dist']:.3f}")]
    md = [f"# {a.env} corrected evaluation{status_note}", ""]
    if prev:
        md += ["Previous evaluation = 50 repeated runs of ONE task (eval_seed was always 1); "
               "it is not a valid 50-task benchmark.", ""]
    md += ["| Metric | Previous evaluation (1 task x 50) | Corrected evaluation |", "|---|---|---|"]
    md += [f"| {k} | {'-' if p is None else p} | {c} |" for k, p, c in rows]
    if paper is not None:
        md += ["", f"Paper (DINO-WM, MPC, 50 tasks): {pct(paper)}. Corrected: {pct(cur['success_rate'])} "
               f"({cur['successes']}/{cur['total']})."]
    if summary["failed_seeds"]:
        md += ["", f"Failed tasks (eval seeds): {summary['failed_seeds']}"]
    if incomplete:
        md += ["", f"Runs that did not complete (crash/timeout; re-run them): {incomplete}"]
    open(os.path.join(res_dir, "comparison.md"), "w").write("\n".join(md) + "\n")

    tex = ["% generated by scripts/eval_records.py aggregate -- do not edit by hand",
           r"\begin{tabular}{lcc}", r"\toprule",
           r"\textbf{Metric} & \textbf{Previous (1 task $\times$ 50)} & \textbf{Corrected (50 tasks)} \\", r"\midrule"]
    tex += [f"{k} & {'--' if p is None else str(p).replace('%', chr(92) + '%')} & {str(c).replace('%', chr(92) + '%')} \\\\"
            for k, p, c in rows]
    tex += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(res_dir, "comparison_table.tex"), "w").write("\n".join(tex) + "\n")

    # figures
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, GOOD, CRIT, INK2, GRID = "#2a78d6", "#0ca30c", "#d03b3b", "#52514e", "#e4e3df"
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": GRID, "axes.axisbelow": True, "savefig.dpi": 200, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(6.5, 6))
    if a.env == "point_maze":  # free space of the U-maze (same test as PointMazeWrapper)
        ax.add_patch(plt.Rectangle((1.1, 0.5), 1.4, 2.0, color="#d9d8d4", zorder=0, label="Wall"))
    for r in runs:
        s0, sg = r["initial_state"], r["goal_state"]
        c = GOOD if r["success"] else CRIT
        ax.annotate("", xy=sg[:2], xytext=s0[:2], arrowprops=dict(arrowstyle="->", color=c, lw=1.1, alpha=0.8))
    ax.scatter([r["initial_state"][0] for r in runs], [r["initial_state"][1] for r in runs], s=26, color=BLUE,
               edgecolor="white", linewidth=1, zorder=3, label="Start")
    ax.scatter([r["goal_state"][0] for r in runs], [r["goal_state"][1] for r in runs], s=60, color=BLUE,
               marker="*", edgecolor="white", linewidth=0.6, zorder=3, label="Goal")
    ax.plot([], [], color=GOOD, label="Task solved"); ax.plot([], [], color=CRIT, label="Task failed")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
    ax.set_title(f"{cur['distinct']} distinct start/goal tasks\n(arrows link start to goal; not the path taken)",
                 fontsize=10)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.savefig(os.path.join(fig_dir, "task_layout.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.bar([r["eval_seed"] for r in runs], [r["final_state_dist"] for r in runs], width=0.8,
           color=[BLUE if r["success"] else CRIT for r in runs])
    ax.set_xlabel("Evaluation seed (task)"); ax.set_ylabel("Final state distance"); ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(fig_dir, "final_state_dist_per_task.png")); plt.close(fig)

    print("\n".join(md))
    print(f"\n[aggregate] wrote {res_dir}/summary.json, per_run.csv, comparison.md, comparison_table.tex; figures in {fig_dir}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--run-dir", required=True); r.add_argument("--eval-seed", type=int, required=True)
    r.add_argument("--global-seed", type=int); r.add_argument("--exit-code", type=int, required=True)
    r.add_argument("--timed-out", type=int, default=0); r.add_argument("--stdout")
    r.add_argument("--max-mpc-iters", type=int); r.add_argument("--wall-time", type=float)
    pc = sub.add_parser("precheck")
    pc.add_argument("--env", default="point_maze"); pc.add_argument("--first", type=int, default=1)
    pc.add_argument("--last", type=int, default=50); pc.add_argument("--show", type=int, default=5)
    v = sub.add_parser("verify"); v.add_argument("--exp-dir", required=True)
    g = sub.add_parser("aggregate")
    g.add_argument("--exp-dir", required=True); g.add_argument("--env", default="point_maze")
    g.add_argument("--expected", type=int, default=50)
    g.add_argument("--previous-logs", default="evaluation_logs/eval_[0-9]*.log",
                   help="glob (relative to repo root) of the old n_evals=1 PointMaze logs; '' to skip")
    a = p.parse_args()
    sys.exit({"record": cmd_record, "precheck": cmd_precheck, "verify": cmd_verify, "aggregate": cmd_aggregate}[a.cmd](a))


if __name__ == "__main__":
    main()
