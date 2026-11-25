"""Aggregate Original (baseline) vs Adaptive CEM runs into metrics, tables and figures.

Reads   logs/runs/<method>/task_XX/{status.json, logs.json, cem_calls.jsonl,
                                    .hydra/config.yaml, plan_targets.pkl}
Writes  results/per_call.csv              one row per CEM optimisation call (one MPC replanning step)
        results/baseline_metrics.csv      one row per navigation task (original CEM)
        results/adaptive_metrics.csv      one row per navigation task (adaptive CEM)
        results/comparison_summary.csv    summary table
        results/comparison_summary.json   summary + reductions + paired statistics + task check
        results/summary.md                short written summary (wording follows the numbers)
        figures/cem_iterations_comparison.png
        figures/planning_time_comparison.png
        figures/per_task_iterations.png
        figures/convergence_example.png

Terminology: a *navigation task* is one MPC episode. Each MPC replanning step
runs one *CEM planning call*, and each call runs 1..10 *CEM iterations*. All
totals include every MPC replanning call, so any extra replanning caused by
adaptive stopping is charged to it.

Only tasks completed by BOTH methods with verified-identical task inputs enter
the comparison (paired design). Everything else is listed, never averaged in.

Usage (repo root):  python experiments/adaptive_cem/scripts/aggregate.py
"""
import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import random
import statistics as st

import numpy as np

EXP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METHODS = ["baseline", "adaptive"]
LABELS = {"baseline": "Original CEM", "adaptive": "Adaptive CEM"}
# Categorical slots 1 and 2 of the validated default palette, fixed order.
COLORS = {"baseline": "#2a78d6", "adaptive": "#eb6834"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
N_BOOT = 10000


# ----------------------------------------------------------------------------- loading
def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_hydra_cfg(run_dir):
    path = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.exists(path):
        return {}
    from omegaconf import OmegaConf

    return OmegaConf.to_container(OmegaConf.load(path), resolve=False)


def file_sha1(path):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def targets_fingerprint(run_dir):
    """Hash of the start/goal states+observations the planner actually received."""
    path = os.path.join(run_dir, "plan_targets.pkl")
    if not os.path.exists(path):
        return None
    d = pickle.load(open(path, "rb"))
    h = hashlib.sha1()
    for k in ("state_0", "state_g"):
        h.update(np.ascontiguousarray(np.asarray(d[k], dtype=np.float64)).tobytes())
    for k in ("obs_0", "obs_g"):
        for kk in sorted(d[k]):
            h.update(np.ascontiguousarray(np.asarray(d[k][kk])).tobytes())
    return h.hexdigest()[:16]


def load_run(run_dir):
    status_path = os.path.join(run_dir, "status.json")
    if not os.path.exists(status_path):
        return None, "no status.json (not run / still running)"
    status = json.load(open(status_path))
    if status["exit_code"] != 0:
        return None, f"exit_code={status['exit_code']}" + (" (timeout)" if status.get("timed_out") else "")
    final = [r for r in read_jsonl(os.path.join(run_dir, "logs.json")) if "final_eval/success_rate" in r]
    calls = read_jsonl(os.path.join(run_dir, "cem_calls.jsonl"))
    if not final:
        return None, "no final_eval entry in logs.json"
    if not calls:
        return None, "no cem_calls.jsonl (stats_filename not set?)"
    cfg = read_hydra_cfg(run_dir)
    return {"status": status, "final": final[-1], "calls": calls, "cfg": cfg, "dir": run_dir}, None


# ----------------------------------------------------------------------------- per task
def wm_count(call, key):
    c = call.get("wm_forward_counts") or {}
    return c.get(key, 0)


def summarise_task(method, task, run, manifest):
    calls, final, cfg = run["calls"], run["final"], run["cfg"]
    success = bool(final["final_eval/success_rate"] >= 1.0)
    n_calls = len(calls)
    goal_file = cfg.get("goal_file_path")
    mt = manifest.get(os.path.basename(goal_file)) if goal_file else None
    planner = cfg.get("planner", {}).get("sub_planner", {})
    return {
        "method": method,
        "task": task,
        "task_file": os.path.basename(goal_file) if goal_file else None,
        "task_hash": mt["hash"] if mt else None,
        "task_file_sha1": file_sha1(goal_file),
        "received_targets_sha1": targets_fingerprint(run["dir"]),
        "seed": cfg.get("seed"),
        "start_goal_xy_dist": mt["start_goal_xy_dist"] if mt else None,
        "num_samples": planner.get("num_samples"),
        "topk": planner.get("topk"),
        "horizon": planner.get("horizon"),
        "max_cem_iterations": planner.get("opt_steps"),
        "success": int(success),
        "final_state_dist": final["final_eval/mean_state_dist"],
        "mpc_planning_calls": n_calls,
        "hit_mpc_cap": int((not success) and n_calls >= run["status"]["max_mpc_steps"]),
        "total_cem_iterations": sum(c["iterations"] for c in calls),
        "avg_cem_iterations_per_call": sum(c["iterations"] for c in calls) / n_calls,
        "total_candidates": sum(c["candidates_evaluated"] for c in calls),
        "wm_rollout_calls": sum(c.get("wm_rollout_calls", c["iterations"]) for c in calls),
        "encoder_forward_calls": sum(wm_count(c, "encoder_forward_calls") for c in calls),
        "encoder_forward_samples": sum(wm_count(c, "encoder_forward_samples") for c in calls),
        "predictor_forward_calls": sum(wm_count(c, "predictor_forward_calls") for c in calls),
        "predictor_forward_samples": sum(wm_count(c, "predictor_forward_samples") for c in calls),
        "planning_time_s": sum(c["compute_time_s"] for c in calls),
        "cem_eval_time_s": sum(c["eval_time_s"] for c in calls),
        "cem_wall_time_s": sum(c["wall_time_s"] for c in calls),
        "process_wall_time_s": run["status"]["process_wall_time_s"],
        "stops_converged": sum(c["stop_reason"] == "converged" for c in calls),
        "stops_success": sum(c["stop_reason"] == "success" for c in calls),
        "stops_max_iterations": sum(c["stop_reason"] == "max_iterations" for c in calls),
    }


# ----------------------------------------------------------------------------- stats
def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def reduction(base, new):
    return None if not base else 100.0 * (base - new) / base


def bootstrap_reduction(pairs, key, n_boot=N_BOOT, seed=0):
    """95% percentile CI of R = (sum base - sum adaptive)/sum base, resampling tasks."""
    rng = random.Random(seed)
    n = len(pairs)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        b = sum(pairs[i][0][key] for i in idx)
        a = sum(pairs[i][1][key] for i in idx)
        if b > 0:
            vals.append(100.0 * (b - a) / b)
    if not vals:
        return None
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]]


def bootstrap_mean_ci(xs, n_boot=N_BOOT, seed=0):
    rng = random.Random(seed)
    n = len(xs)
    if n < 2:
        return None
    m = sorted(st.mean(xs[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot))
    return [m[int(0.025 * n_boot)], m[int(0.975 * n_boot) - 1]]


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------------- figures
def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "axes.titlelocation": "left",
    })
    return plt


def _bars(ax, values, fmt, ylabel, title, err=None):
    xs = range(len(METHODS))
    bars = ax.bar(xs, values, width=0.56, color=[COLORS[m] for m in METHODS],
                  yerr=err, capsize=4 if err else 0, error_kw={"ecolor": INK2, "elinewidth": 1})
    ax.set_xticks(list(xs))
    ax.set_xticklabels([LABELS[m] for m in METHODS])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    top = max(values) if err is None else max(v + e for v, e in zip(values, err[1]))
    ax.set_ylim(0, top * 1.18 if top > 0 else 1)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * 0.02, fmt(v), ha="center",
                va="bottom", color=INK, fontsize=9)


def matplotlib_ticker():
    import matplotlib.ticker as t

    return t


def _subtitle(fig, text, width=120):
    import textwrap

    fig.text(0.012, 0.012, "\n".join(textwrap.wrap(text, width)), color=INK2, fontsize=8.5, ha="left", va="bottom")


def make_figures(S, pairs, calls, fig_dir):
    plt = _style()
    B, A, R = S["original"], S["adaptive"], S["reductions_pct"]
    n = S["n_paired_tasks"]

    # 1) CEM iterations
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7))
    _bars(axes[0], [B["total_cem_iterations"], A["total_cem_iterations"]], lambda v: f"{v:,.0f}",
          "CEM iterations (all MPC calls)", f"Total CEM iterations ({n} tasks)")
    _bars(axes[1], [B["avg_cem_iterations_per_call"], A["avg_cem_iterations_per_call"]], lambda v: f"{v:.2f}",
          "Iterations per planning call", "Average per CEM planning call")
    ci = S["bootstrap_95ci"]["R_I"]
    _subtitle(fig, f"Iteration reduction R_I = {R['R_I']:.1f}%" + (f" (95% CI {ci[0]:.1f} to {ci[1]:.1f}%)" if ci else "")
              + ". Max 10 iterations per call; totals include all MPC replanning calls.")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(os.path.join(fig_dir, "cem_iterations_comparison.png"), dpi=200)
    plt.close(fig)

    # 2) planning time
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), gridspec_kw={"width_ratios": [1, 1.05]})
    means = [B["avg_planning_time_per_task_s"], A["avg_planning_time_per_task_s"]]
    cis = [S["bootstrap_95ci"]["mean_planning_time_baseline"], S["bootstrap_95ci"]["mean_planning_time_adaptive"]]
    err = None
    if all(cis):
        err = [[m - c[0] for m, c in zip(means, cis)], [c[1] - m for m, c in zip(means, cis)]]
    _bars(axes[0], means, lambda v: f"{v:.2f} s", "Seconds per task", "Average planning time per task", err=err)
    b = [p[0]["planning_time_s"] for p in pairs]
    a = [p[1]["planning_time_s"] for p in pairs]
    lo, lim = min(b + a) * 0.7, max(b + a) * 1.4
    ax = axes[1]
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot([lo, lim], [lo, lim], color=INK2, linewidth=1, linestyle="--", zorder=1)
    ax.text(lim * 0.8, lim * 0.55, "equal time", ha="right", va="top", rotation=45,
            rotation_mode="anchor", color=INK2, fontsize=8)
    ax.scatter(b, a, s=34, color=COLORS["adaptive"], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_xlim(lo, lim); ax.set_ylim(lo, lim); ax.set_aspect("equal")
    ax.set_xlabel("Original CEM (s, log)"); ax.set_ylabel("Adaptive CEM (s, log)")
    ax.set_title("Per task (below line = adaptive faster)", fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ci = S["bootstrap_95ci"]["R_T"]
    _subtitle(fig, f"Planning-time reduction R_T = {R['R_T']:.1f}%" + (f" (95% CI {ci[0]:.1f} to {ci[1]:.1f}%)" if ci else "")
              + ". World-model inference + CEM updates, GPU-synchronised; excludes simulator check and rendering.")
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(os.path.join(fig_dir, "planning_time_comparison.png"), dpi=200)
    plt.close(fig)

    # 3) per-task iterations (dumbbell)
    tasks = [p[0]["task"] for p in pairs]
    fig, ax = plt.subplots(figsize=(max(6.5, 0.2 * len(tasks) + 2.5), 4.0))
    xs = np.arange(len(tasks))
    for x, p in zip(xs, pairs):
        ax.plot([x - 0.14, x + 0.14], [p[0]["total_cem_iterations"], p[1]["total_cem_iterations"]], color=GRID, linewidth=2, zorder=1)
    for j, m in enumerate(METHODS):
        ys = [p[j]["total_cem_iterations"] for p in pairs]
        ok = [p[j]["success"] for p in pairs]
        xo = xs + (j - 0.5) * 0.28  # side by side so equal values stay visible
        ax.scatter([x for x, s in zip(xo, ok) if s], [y for y, s in zip(ys, ok) if s], s=40, color=COLORS[m],
                   edgecolor=SURFACE, linewidth=1.5, zorder=3, label=f"{LABELS[m]} (success)")
        if not all(ok):
            ax.scatter([x for x, s in zip(xo, ok) if not s], [y for y, s in zip(ys, ok) if not s], s=40,
                       facecolor=SURFACE, edgecolor=COLORS[m], linewidth=1.8, zorder=3, label=f"{LABELS[m]} (failure)")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(t) for t in tasks], fontsize=7 if len(tasks) > 20 else 9)
    ax.set_xlabel("Task")
    ax.set_ylabel("Total CEM iterations (log scale)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.set_yscale("log")
    ax.set_ylim(0.7, max(max(p[0]["total_cem_iterations"], p[1]["total_cem_iterations"]) for p in pairs) * 1.5)
    ax.yaxis.set_major_formatter(matplotlib_ticker().FuncFormatter(lambda v, _: f"{v:g}"))
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.06), borderaxespad=0)
    ax.set_title("Per-task CEM iterations, same task under both planners", pad=26)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "per_task_iterations.png"), dpi=200)
    plt.close(fig)

    # 4) convergence example: first CEM call of a task where adaptive stopped on convergence
    by = {(c["method"], c["task"], c["mpc_step"]): c for c in calls}
    cands = [(t, k) for (m, t, k), c in by.items() if m == "adaptive" and c["stop_reason"] == "converged"
             and ("baseline", t, k) in by]
    if cands:
        cands.sort(key=lambda tk: (tk[1], -by[("baseline",) + tk]["iterations"], tk[0]))
        t, k = cands[0]
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for m, ls in (("baseline", "-"), ("adaptive", "-")):
            c = by[(m, t, k)]
            it = np.arange(1, len(c["best_cost"]) + 1)
            ax.plot(it, c["best_cost"], ls, color=COLORS[m], linewidth=2, marker="o", markersize=5,
                    label=f"{LABELS[m]}: {c['iterations']} iterations ({c['stop_reason'].replace('_', ' ')})", zorder=3 if m == "adaptive" else 2)
        stop = by[("adaptive", t, k)]["iterations"]
        ax.axvline(stop, color=INK2, linewidth=1, linestyle=":")
        ax.text(stop + 0.1, ax.get_ylim()[1], " adaptive stop", va="top", color=INK2, fontsize=8)
        ax.set_xlabel("CEM iteration")
        ax.set_ylabel("Best predicted goal-distance cost")
        ax.set_title(f"Convergence example: task {t}, MPC call {k + 1}")
        ax.set_xticks(range(1, 11))
        ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8.5)
        _subtitle(fig, "Both planners use identical sampling noise until the adaptive planner stops. "
                       "Cost = DINO feature MSE to goal (world-model prediction).", width=95)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        fig.savefig(os.path.join(fig_dir, "convergence_example.png"), dpi=200)
        plt.close(fig)
        return {"task": t, "mpc_call": k + 1}
    return None


# ----------------------------------------------------------------------------- summary text
def write_markdown(S, path):
    B, A, R, CI, P = S["original"], S["adaptive"], S["reductions_pct"], S["bootstrap_95ci"], S["paired_stats"]
    n = S["n_paired_tasks"]
    lines = [f"# Adaptive CEM vs Original CEM — PointMaze ({n} paired tasks)", ""]
    if n < 50:
        lines += [f"> **Preliminary:** only {n} paired tasks. Treat these numbers as a correctness check, "
                  "not as evidence.", ""]
    lines += ["| Metric | Original CEM | Adaptive CEM | Change |", "|---|---|---|---|"]
    lines += [f"| {r['metric']} | {r['original']} | {r['adaptive']} | {r['change']} |" for r in S["table"]]

    def ci_txt(k):
        return f" (95% bootstrap CI {CI[k][0]:.1f}% to {CI[k][1]:.1f}%)" if CI.get(k) else ""

    def verdict(k):
        v, ci = R[k], CI.get(k)
        if v is None:
            return "could not be computed"
        if ci and ci[0] > 0:
            return f"was reduced by {v:.1f}%{ci_txt(k)}"
        if ci and ci[1] < 0:
            return f"increased by {-v:.1f}%{ci_txt(k)}"
        return f"changed by {0.0 - v:+.1f}%{ci_txt(k)}; the interval includes zero, so no reliable change was measured"

    work_reduced = bool(CI.get("R_I") and CI["R_I"][0] > 0)
    time_line = f"- Total planning time {verdict('R_T')}."
    if not work_reduced:
        time_line = (f"- Total planning time changed by {0.0 - R['R_T']:+.1f}%{ci_txt('R_T')}. Iteration counts did not "
                     "fall reliably, so this difference is attributed to timing variability, not to adaptive stopping.")

    lines += ["", "## Computation", "",
              f"- Total CEM iterations {verdict('R_I')}.",
              time_line,
              f"- Total candidate sequences evaluated {verdict('R_C')}.",
              f"- Adaptive stopping ended {A['convergence_stops']} of {A['n_planning_calls']} planning calls; "
              f"MPC replanning calls per task: {B['avg_mpc_calls_per_task']:.2f} → {A['avg_mpc_calls_per_task']:.2f} "
              "(all replanning cost is included in the totals).",
              f"- Per task: adaptive used fewer CEM iterations on {P['tasks_adaptive_fewer_iterations']}, more on "
              f"{P['tasks_adaptive_more_iterations']}, and the same number on {P['tasks_identical_iterations']} of {n} tasks "
              f"(median planning-time difference {P['planning_time_diff_s_median']:+.2f} s). Savings are concentrated in a few tasks.",
              ] + ([
              f"- Sensitivity: excluding tasks where either planner hit the MPC step cap "
              f"({S['sensitivity_excluding_mpc_cap_tasks']['excluded_tasks']}), R_I = "
              f"{S['sensitivity_excluding_mpc_cap_tasks']['R_I']:.1f}% and R_T = {S['sensitivity_excluding_mpc_cap_tasks']['R_T']:.1f}% "
              f"over {S['sensitivity_excluding_mpc_cap_tasks']['n_tasks']} tasks. Capped tasks inflate totals, because the cap "
              "(not the planner) sets their cost."] if S["sensitivity_excluding_mpc_cap_tasks"]["excluded_tasks"] else []) + [
              "", "## Navigation performance", "",
              f"- Success: {B['successful_tasks']}/{n} → {A['successful_tasks']}/{n} "
              f"({S['change']['success_rate_pp']:+.1f} percentage points).",
              f"- Paired outcomes: solved only by Original = {P['success_original_only']}, only by Adaptive = "
              f"{P['success_adaptive_only']}; exact McNemar p = {P['success_mcnemar_exact_p']:.3f}.",
              f"- Mean final state distance: {B['mean_final_state_dist']:.3f} → {A['mean_final_state_dist']:.3f}."]
    if P["success_mcnemar_exact_p"] >= 0.05:
        lines.append("- No statistically significant change in success was detected. With this sample size, "
                     "small differences cannot be ruled out, so this is not proof that the two planners are equivalent.")
    else:
        lines.append("- The change in success is statistically significant at the 5% level.")
    lines += ["", "## Notes", "",
              "- Planning time = GPU-synchronised world-model inference and CEM updates. It excludes the original "
              "per-iteration simulator success check, rendering, video writing and model loading.",
              f"- Final state distance is the env's full-state distance (position and velocity); "
              "success uses position only (< 0.5).",
              f"- Task inputs were verified identical for both planners on all {n} paired tasks.",
              f"- Excluded runs: {S['excluded'] or 'none'}."]
    open(path, "w").write("\n".join(lines) + "\n")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default=EXP_DIR)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    runs_dir = os.path.join(args.exp_dir, "logs", "runs")
    res_dir = os.path.join(args.exp_dir, "results")
    fig_dir = os.path.join(args.exp_dir, "figures")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    mpath = os.path.join(args.exp_dir, "configs", "tasks", "manifest.json")
    manifest = {t["file"]: t for t in json.load(open(mpath))["tasks"]} if os.path.exists(mpath) else {}

    tasks, excluded, per_call = {}, [], []
    for m in METHODS:
        mdir = os.path.join(runs_dir, m)
        if not os.path.isdir(mdir):
            continue
        for d in sorted(os.listdir(mdir)):
            if not d.startswith("task_"):
                continue
            t = int(d.split("_")[1])
            run, err = load_run(os.path.join(mdir, d))
            if err:
                excluded.append({"method": m, "task": t, "reason": err})
                continue
            tasks.setdefault(t, {})[m] = summarise_task(m, t, run, manifest)
            for c in run["calls"]:
                inst = c["instances"][0]
                per_call.append({
                    "method": m, "task": t, "mpc_step": c["call_index"], "iterations": c["iterations"],
                    "stop_reason": c["stop_reason"], "candidates_evaluated": c["candidates_evaluated"],
                    "wm_rollout_calls": c.get("wm_rollout_calls"),
                    "predictor_forward_calls": wm_count(c, "predictor_forward_calls"),
                    "encoder_forward_calls": wm_count(c, "encoder_forward_calls"),
                    "compute_time_s": c["compute_time_s"], "eval_time_s": c["eval_time_s"],
                    "wall_time_s": c["wall_time_s"],
                    "best_cost": inst["best_cost"], "final_best_cost": inst["best_cost"][-1] if inst["best_cost"] else None,
                })

    for m in METHODS:
        write_csv(os.path.join(res_dir, f"{m}_metrics.csv"), [tasks[t][m] for t in sorted(tasks) if m in tasks[t]])
    write_csv(os.path.join(res_dir, "per_call.csv"),
              [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in c.items()} for c in per_call])

    # pairing + task-identity verification
    paired = []
    for t in sorted(tasks):
        v = tasks[t]
        if not all(m in v for m in METHODS):
            excluded.append({"task": t, "reason": "not completed by both methods"})
            continue
        b, a = v["baseline"], v["adaptive"]
        same = all(b[k] == a[k] for k in ("task_file", "task_file_sha1", "received_targets_sha1", "seed",
                                           "num_samples", "topk", "horizon", "max_cem_iterations"))
        if not same:
            excluded.append({"task": t, "reason": "task inputs/settings differ between methods"})
            continue
        paired.append(t)
    if not paired:
        print("No task completed by both methods yet. Excluded:", excluded)
        return
    pairs = [(tasks[t]["baseline"], tasks[t]["adaptive"]) for t in paired]
    pset = set(paired)
    pcalls = [c for c in per_call if c["task"] in pset]

    def agg(j, m):
        rows = [p[j] for p in pairs]
        calls = [c for c in pcalls if c["method"] == m]
        return {
            "evaluated_tasks": len(rows),
            "successful_tasks": sum(r["success"] for r in rows),
            "success_rate": st.mean(r["success"] for r in rows),
            "total_cem_iterations": sum(r["total_cem_iterations"] for r in rows),
            "avg_cem_iterations_per_call": st.mean(c["iterations"] for c in calls),
            "n_planning_calls": len(calls),
            "avg_mpc_calls_per_task": st.mean(r["mpc_planning_calls"] for r in rows),
            "total_candidates": sum(r["total_candidates"] for r in rows),
            "total_wm_rollout_calls": sum(r["wm_rollout_calls"] for r in rows),
            "total_predictor_forward_calls": sum(r["predictor_forward_calls"] for r in rows),
            "total_encoder_forward_calls": sum(r["encoder_forward_calls"] for r in rows),
            "avg_planning_time_per_task_s": st.mean(r["planning_time_s"] for r in rows),
            "total_planning_time_s": sum(r["planning_time_s"] for r in rows),
            "mean_final_state_dist": st.mean(r["final_state_dist"] for r in rows),
            "convergence_stops": sum(c["stop_reason"] == "converged" for c in calls),
            "success_stops": sum(c["stop_reason"] == "success" for c in calls),
            "max_iteration_stops": sum(c["stop_reason"] == "max_iterations" for c in calls),
            "tasks_hitting_mpc_cap": sum(r["hit_mpc_cap"] for r in rows),
            "avg_process_wall_time_per_task_s": st.mean(r["process_wall_time_s"] for r in rows),
        }

    B, A = agg(0, "baseline"), agg(1, "adaptive")
    R = {
        "R_I": reduction(B["total_cem_iterations"], A["total_cem_iterations"]),
        "R_T": reduction(B["total_planning_time_s"], A["total_planning_time_s"]),
        "R_C": reduction(B["total_candidates"], A["total_candidates"]),
        "R_predictor_calls": reduction(B["total_predictor_forward_calls"], A["total_predictor_forward_calls"]),
    }
    CI = {
        "R_I": bootstrap_reduction(pairs, "total_cem_iterations"),
        "R_T": bootstrap_reduction(pairs, "planning_time_s"),
        "R_C": bootstrap_reduction(pairs, "total_candidates"),
        "mean_planning_time_baseline": bootstrap_mean_ci([p[0]["planning_time_s"] for p in pairs]),
        "mean_planning_time_adaptive": bootstrap_mean_ci([p[1]["planning_time_s"] for p in pairs]),
    }
    nocap = [p for p in pairs if not (p[0]["hit_mpc_cap"] or p[1]["hit_mpc_cap"])]
    sens = {
        "excluded_tasks": [p[0]["task"] for p in pairs if p not in nocap],
        "n_tasks": len(nocap),
        "R_I": reduction(sum(p[0]["total_cem_iterations"] for p in nocap), sum(p[1]["total_cem_iterations"] for p in nocap)),
        "R_T": reduction(sum(p[0]["planning_time_s"] for p in nocap), sum(p[1]["planning_time_s"] for p in nocap)),
        "R_C": reduction(sum(p[0]["total_candidates"] for p in nocap), sum(p[1]["total_candidates"] for p in nocap)),
        "R_I_ci": bootstrap_reduction(nocap, "total_cem_iterations") if nocap else None,
        "R_T_ci": bootstrap_reduction(nocap, "planning_time_s") if nocap else None,
    }
    o_only = sum(p[0]["success"] and not p[1]["success"] for p in pairs)
    a_only = sum(p[1]["success"] and not p[0]["success"] for p in pairs)
    dt = [p[1]["planning_time_s"] - p[0]["planning_time_s"] for p in pairs]
    di = [p[1]["total_cem_iterations"] - p[0]["total_cem_iterations"] for p in pairs]

    def ch(b, a, nd=0, unit=""):
        d = a - b
        r = reduction(b, a)
        d = 0.0 if abs(d) < 0.5 * 10 ** -nd else d
        r = None if r is None else (0.0 if abs(r) < 0.05 else r)
        return f"{d:+,.{nd}f}{unit}" + ("" if r is None else f" ({0.0 - r:+.1f}%)")

    table = [
        {"metric": "Evaluated tasks", "original": B["evaluated_tasks"], "adaptive": A["evaluated_tasks"], "change": "paired"},
        {"metric": "Successful tasks", "original": B["successful_tasks"], "adaptive": A["successful_tasks"],
         "change": f"{A['successful_tasks'] - B['successful_tasks']:+d}"},
        {"metric": "Success rate", "original": f"{100 * B['success_rate']:.1f}%", "adaptive": f"{100 * A['success_rate']:.1f}%",
         "change": f"{100 * (A['success_rate'] - B['success_rate']):+.1f} pp"},
        {"metric": "Total CEM iterations", "original": f"{B['total_cem_iterations']:,}", "adaptive": f"{A['total_cem_iterations']:,}",
         "change": ch(B["total_cem_iterations"], A["total_cem_iterations"])},
        {"metric": "Average CEM iterations per planning call", "original": f"{B['avg_cem_iterations_per_call']:.2f}",
         "adaptive": f"{A['avg_cem_iterations_per_call']:.2f}",
         "change": ch(B["avg_cem_iterations_per_call"], A["avg_cem_iterations_per_call"], 2)},
        {"metric": "Average MPC replanning calls per task", "original": f"{B['avg_mpc_calls_per_task']:.2f}",
         "adaptive": f"{A['avg_mpc_calls_per_task']:.2f}", "change": ch(B["avg_mpc_calls_per_task"], A["avg_mpc_calls_per_task"], 2)},
        {"metric": "Total candidate sequences evaluated", "original": f"{B['total_candidates']:,}", "adaptive": f"{A['total_candidates']:,}",
         "change": ch(B["total_candidates"], A["total_candidates"])},
        {"metric": "Total predictor forward calls (batched)", "original": f"{B['total_predictor_forward_calls']:,}",
         "adaptive": f"{A['total_predictor_forward_calls']:,}",
         "change": ch(B["total_predictor_forward_calls"], A["total_predictor_forward_calls"])},
        {"metric": "Average planning time per task", "original": f"{B['avg_planning_time_per_task_s']:.2f} s",
         "adaptive": f"{A['avg_planning_time_per_task_s']:.2f} s",
         "change": ch(B["avg_planning_time_per_task_s"], A["avg_planning_time_per_task_s"], 2, " s")},
        {"metric": "Total planning time", "original": f"{B['total_planning_time_s']:.1f} s", "adaptive": f"{A['total_planning_time_s']:.1f} s",
         "change": ch(B["total_planning_time_s"], A["total_planning_time_s"], 1, " s")},
        {"metric": "Mean final state distance (pos+vel)", "original": f"{B['mean_final_state_dist']:.3f}",
         "adaptive": f"{A['mean_final_state_dist']:.3f}", "change": ch(B["mean_final_state_dist"], A["mean_final_state_dist"], 3)},
        {"metric": "Convergence-triggered stops (calls)", "original": B["convergence_stops"], "adaptive": A["convergence_stops"], "change": ""},
        {"metric": "Tasks hitting MPC step cap", "original": B["tasks_hitting_mpc_cap"], "adaptive": A["tasks_hitting_mpc_cap"], "change": ""},
    ]
    S = {
        "n_paired_tasks": len(paired),
        "paired_tasks": paired,
        "excluded": excluded,
        "task_identity_check": "task file sha1, received start/goal targets, seed and CEM settings identical for every paired task",
        "original": B,
        "adaptive": A,
        "reductions_pct": R,
        "bootstrap_95ci": CI,
        "sensitivity_excluding_mpc_cap_tasks": sens,
        "change": {"success_rate_pp": 100.0 * (A["success_rate"] - B["success_rate"])},
        "paired_stats": {
            "success_original_only": o_only,
            "success_adaptive_only": a_only,
            "success_mcnemar_exact_p": mcnemar_exact(o_only, a_only),
            "planning_time_diff_s_mean": st.mean(dt),
            "planning_time_diff_s_median": st.median(dt),
            "tasks_adaptive_faster": sum(d < 0 for d in dt),
            "tasks_identical_iterations": sum(d == 0 for d in di),
            "iteration_diff_mean": st.mean(di),
            "tasks_adaptive_fewer_iterations": sum(d < 0 for d in di),
            "tasks_adaptive_more_iterations": sum(d > 0 for d in di),
        },
        "table": table,
    }
    if not args.no_figures:
        S["convergence_example"] = make_figures(S, pairs, pcalls, fig_dir)
    json.dump(S, open(os.path.join(res_dir, "comparison_summary.json"), "w"), indent=2)
    rows = [dict(r) for r in table]
    rows += [{"metric": "Iteration reduction R_I (%)", "original": "", "adaptive": "", "change": f"{R['R_I']:.2f}"},
             {"metric": "Planning time reduction R_T (%)", "original": "", "adaptive": "", "change": f"{R['R_T']:.2f}"},
             {"metric": "Candidate evaluation reduction R_C (%)", "original": "", "adaptive": "", "change": f"{R['R_C']:.2f}"}]
    write_csv(os.path.join(res_dir, "comparison_summary.csv"), rows)
    print(write_markdown(S, os.path.join(res_dir, "summary.md")))
    print(f"\nWrote results to {res_dir} and figures to {fig_dir}")


if __name__ == "__main__":
    main()
