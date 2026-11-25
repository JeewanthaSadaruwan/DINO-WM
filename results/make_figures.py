"""Regenerate every figure in results/figures/ from the evaluation logs and plan_outputs.

Run from the repo root:  python results/make_figures.py
Writes: results/figures/*.png, results/figures/videos/*.mp4, results/summary.json
"""
import json, os, re, shutil, pickle, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")
VID = os.path.join(FIG, "videos")
os.makedirs(VID, exist_ok=True)

# palette (validated reference palette: categorical slots 1-2, status good/critical)
BLUE, ORANGE = "#2a78d6", "#eb6834"
GOOD, CRIT = "#0ca30c", "#d03b3b"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 200,
    "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})

PAPER = {"PointMaze": 0.98, "PushT": 0.90, "Wall": 0.96}  # arXiv 2411.04983, Table 1 (MPC)


def parse(log):
    t = open(log, errors="ignore").read()
    m = re.search(r"Planning result saved dir: (\S+)", t)
    d = m.group(1) if m else None
    fin = None
    if d and os.path.exists(f"{d}/logs.json"):
        for line in open(f"{d}/logs.json"):
            j = json.loads(line)
            if any(k.startswith("final_eval") for k in j):
                fin = {k.split("/")[1]: v for k, v in j.items()}
    sg = None
    if d and os.path.exists(f"{d}/plan_targets.pkl"):
        p = pickle.load(open(f"{d}/plan_targets.pkl", "rb"))
        sg = (np.asarray(p["state_0"]).reshape(-1), np.asarray(p["state_g"]).reshape(-1))
    return dict(log=os.path.relpath(log, ROOT), dir=d, final=fin,
                mpc_iters=len(re.findall(r"MPC iter \d+ Eval", t)),
                oom="OutOfMemoryError" in t, s0=sg and sg[0], sg=sg and sg[1])


pm = [parse(f"{ROOT}/evaluation_logs/eval_{i}.log") for i in range(1, 51)]
pt = [parse(f"{ROOT}/evaluation_logs/pusht/eval_{i}.log") for i in range(1, 51)]
wl = [parse(f"{ROOT}/evaluation_logs/wall/test.log")]
wall50 = f"{ROOT}/evaluation_logs/wall/eval_50.log"
wall50 = parse(wall50) if os.path.exists(wall50) else None

for r in pt:
    s0, sg = r["s0"][:7], r["sg"][:7]
    a = abs(s0[4] - sg[4]); a = min(a, 2 * np.pi - a)
    r["block_dist"] = float(np.linalg.norm(s0[2:4] - sg[2:4]))
    r["angle_deg"] = float(np.degrees(a))
    r["success"] = bool(r["final"] and r["final"]["success_rate"] == 1.0)


def stats(runs, key):
    a = np.array([r["final"][key] for r in runs if r["final"]])
    return dict(mean=float(a.mean()), median=float(np.median(a)), min=float(a.min()), max=float(a.max()))


summary = {
    "paper": PAPER,
    "pointmaze": dict(runs=50, successes=sum(r["final"]["success_rate"] for r in pm),
                      distinct_tasks=len({r["s0"][:4].round(3).tobytes() + r["sg"][:4].round(3).tobytes() for r in pm}),
                      state_dist=stats(pm, "mean_state_dist"), visual_dist=stats(pm, "mean_visual_dist"),
                      proprio_dist=stats(pm, "mean_proprio_dist")),
    "pusht": dict(runs=50, successes=sum(r["success"] for r in pt), unfinished=[r["log"] for r in pt if not r["final"]],
                  distinct_tasks=len({r["s0"][:5].round(3).tobytes() + r["sg"][:5].round(3).tobytes() for r in pt}),
                  state_dist=stats(pt, "mean_state_dist"), visual_dist=stats(pt, "mean_visual_dist"),
                  proprio_dist=stats(pt, "mean_proprio_dist"),
                  block_dist=dict(min=min(r["block_dist"] for r in pt), median=float(np.median([r["block_dist"] for r in pt])),
                                  max=max(r["block_dist"] for r in pt)),
                  angle_deg=dict(min=min(r["angle_deg"] for r in pt), median=float(np.median([r["angle_deg"] for r in pt])),
                                 max=max(r["angle_deg"] for r in pt)),
                  one_iter_runs=sum(r["mpc_iters"] == 1 for r in pt)),
    "wall_test": dict(runs=1, successes=wl[0]["final"]["success_rate"], final=wl[0]["final"], mpc_iters=wl[0]["mpc_iters"]),
    "wall_50": wall50 and wall50["final"],
}
json.dump(summary, open(os.path.join(ROOT, "results", "summary.json"), "w"), indent=1, default=float)

# ---------------- Fig 1: success rate vs paper ----------------
ours = {"PointMaze": summary["pointmaze"]["successes"] / 50, "PushT": summary["pusht"]["successes"] / 50,
        "Wall": (wall50["final"]["success_rate"] if wall50 and wall50["final"] else summary["wall_test"]["successes"])}
notes = {"PointMaze": "50 runs of\n1 task", "PushT": f"{summary['pusht']['successes']}/50\ntasks",
         "Wall": "50 tasks" if wall50 and wall50["final"] else "1 test\ntask"}
fig, ax = plt.subplots(figsize=(7.5, 4.2))
x = np.arange(3); w = 0.36
envs = list(PAPER)
b1 = ax.bar(x - w / 2 - 0.01, [ours[e] * 100 for e in envs], w, color=BLUE, label="Our reproduction")
b2 = ax.bar(x + w / 2 + 0.01, [PAPER[e] * 100 for e in envs], w, color=ORANGE, label="DINO-WM paper (MPC)")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f}%",
                ha="center", va="bottom", fontsize=10, color=INK)
for i, e in enumerate(envs):
    ax.text(i - w / 2, 4, notes[e], ha="center", va="bottom", fontsize=8.5, color="white")
ax.set_xticks(x, envs); ax.set_ylim(0, 115); ax.set_ylabel("Success rate (%)")
ax.set_yticks([0, 25, 50, 75, 100]); ax.grid(axis="x", visible=False)
ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.12))
fig.savefig(f"{FIG}/fig01_success_vs_paper.png"); plt.close(fig)

# ---------------- Fig 3: PointMaze spread across 50 repeats ----------------
d = np.sort([r["final"]["mean_state_dist"] for r in pm])
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.bar(np.arange(1, 51), d, width=0.8, color=BLUE)
ax.axhline(np.mean(d), color=INK2, ls="--", lw=1.2)
ax.text(50.5, np.mean(d), f" mean {np.mean(d):.2f}", va="center", ha="left", fontsize=9, color=INK2)
ax.set_xlabel("Run (sorted by final state distance)"); ax.set_ylabel("Final state distance")
ax.set_xlim(0, 51); ax.grid(axis="x", visible=False)
fig.savefig(f"{FIG}/fig03_pointmaze_repeat_spread.png"); plt.close(fig)

# ---------------- Fig 4: PushT difficulty vs outcome ----------------
fig, ax = plt.subplots(figsize=(7.5, 4.4))
s = [r for r in pt if r["success"]]; f = [r for r in pt if not r["success"]]
ax.scatter([r["block_dist"] for r in s], [r["angle_deg"] for r in s], s=46, color=GOOD,
           edgecolor="white", linewidth=1.5, label=f"Success ({len(s)})", zorder=3)
ax.scatter([r["block_dist"] for r in f], [r["angle_deg"] for r in f], s=110, color=CRIT, marker="X",
           edgecolor="white", linewidth=1.5, label=f"Not reached, run crashed ({len(f)})", zorder=4)
for r in f:
    run = re.search(r"eval_(\d+)", r["log"]).group(1)
    ax.annotate(f"run {run}", (r["block_dist"], r["angle_deg"]), xytext=(8, 6),
                textcoords="offset points", fontsize=9, color=INK)
ax.set_xlabel("Initial block distance to goal (px)"); ax.set_ylabel("Initial block angle error (°)")
ax.legend(frameon=False, loc="upper right")
fig.savefig(f"{FIG}/fig04_pusht_difficulty_vs_outcome.png"); plt.close(fig)


# ---------------- example images & videos ----------------
def copy(src, dst):
    shutil.copy(src, os.path.join(FIG, dst))


def vid(d, pattern, dst):
    m = sorted(glob.glob(os.path.join(d, pattern)))
    if m: shutil.copy(m[0], os.path.join(VID, dst))


copy(f"{pm[0]['dir']}/plan0.png", "fig02_pointmaze_example.png")
vid(pm[0]["dir"], "plan0_0_*.mp4", "pointmaze_example.mp4")

ok = sorted(s, key=lambda r: r["block_dist"])
picks = {"easy": ok[len(ok) // 4], "medium": ok[len(ok) // 2], "hard": ok[-1]}
for i, (k, r) in enumerate(picks.items()):
    copy(f"{r['dir']}/plan0.png", f"fig0{5 + i}_pusht_{k}.png")
    vid(r["dir"], "plan0_0_*.mp4", f"pusht_{k}.mp4")
    summary["pusht"][f"example_{k}"] = dict(log=r["log"], block_dist=r["block_dist"], angle_deg=r["angle_deg"])

fr = f[0]
copy(f"{fr['dir']}/plan0.png", "fig08a_pusht_failure_first_plan.png")
last = max(glob.glob(f"{fr['dir']}/plan[0-9]*.png"), key=lambda p: int(re.search(r"plan(\d+)\.png", p).group(1)))
im = Image.open(last); W, H = im.size
im.crop((W - 1584, 0, W, H)).save(f"{FIG}/fig08b_pusht_failure_last_frames.png")
summary["pusht"]["example_failure"] = dict(log=fr["log"], block_dist=fr["block_dist"], angle_deg=fr["angle_deg"],
                                           mpc_iters=fr["mpc_iters"], last_plan=os.path.basename(last))
vid(fr["dir"], "plan0_0_*.mp4", "pusht_failure_first_plan.mp4")

wdir = (wall50 or wl[0])["dir"]
copy(f"{wl[0]['dir']}/plan0.png", "fig09_wall_example.png")
vid(wl[0]["dir"], "plan0_0_*.mp4", "wall_example.mp4")

json.dump(summary, open(os.path.join(ROOT, "results", "summary.json"), "w"), indent=1, default=float)
print(json.dumps(summary, indent=1, default=float))


# =====================================================================================
# Corrected PointMaze evaluation (50 distinct tasks, scripts/run_pointmaze_corrected.sh)
# =====================================================================================
import csv
PMC = f"{ROOT}/experiments/pointmaze_corrected"
if os.path.exists(f"{PMC}/results/summary.json"):
    rows = list(csv.DictReader(open(f"{PMC}/results/per_run.csv")))
    for r in rows:
        r["seed"] = int(r["eval_seed"]); r["ok"] = r["success"] == "True"; r["iters"] = int(r["mpc_iterations"])
        s0 = np.array(r["initial_state"].split(), float); sg = np.array(r["goal_state"].split(), float)
        r["xy_dist"] = float(np.linalg.norm(s0[:2] - sg[:2]))
    agg = json.load(open(f"{PMC}/results/summary.json"))
    okr = [r for r in rows if r["ok"]]; bad = [r for r in rows if not r["ok"]]
    summary["pointmaze_corrected"] = dict(
        agg["corrected"], failed_seeds=agg["failed_seeds"],
        mpc_iters_hist={k: sum(r["iters"] == k for r in rows) for k in sorted({r["iters"] for r in rows})},
        start_goal_xy_dist=dict(min=min(r["xy_dist"] for r in rows), median=float(np.median([r["xy_dist"] for r in rows])),
                                max=max(r["xy_dist"] for r in rows)),
        state_dist_successes=dict(median=float(np.median([float(r["final_state_dist"]) for r in okr])),
                                  min=min(float(r["final_state_dist"]) for r in okr),
                                  max=max(float(r["final_state_dist"]) for r in okr)))

    # Fig 1 again, now with the corrected PointMaze number
    ours["PointMaze"] = agg["corrected"]["success_rate"]
    notes["PointMaze"] = f"{agg['corrected']['successes']}/50\ntasks"
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    b1 = ax.bar(x - w / 2 - 0.01, [ours[e] * 100 for e in envs], w, color=BLUE, label="Our reproduction")
    b2 = ax.bar(x + w / 2 + 0.01, [PAPER[e] * 100 for e in envs], w, color=ORANGE, label="DINO-WM paper (MPC)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f}%",
                    ha="center", va="bottom", fontsize=10, color=INK)
    for i, e in enumerate(envs):
        ax.text(i - w / 2, 4, notes[e], ha="center", va="bottom", fontsize=8.5, color="white")
    ax.set_xticks(x, envs); ax.set_ylim(0, 115); ax.set_ylabel("Success rate (%)")
    ax.set_yticks([0, 25, 50, 75, 100]); ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.12))
    fig.savefig(f"{FIG}/fig01_success_vs_paper.png"); plt.close(fig)

    # Fig 10: task layout (made by scripts/eval_records.py aggregate)
    copy(f"{PMC}/figures/task_layout.png", "fig10_pointmaze_corrected_tasks.png")

    # Fig 11: final state distance per task
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.bar([r["seed"] for r in okr], [float(r["final_state_dist"]) for r in okr], width=0.8, color=BLUE,
           label=f"Solved ({len(okr)})")
    ax.bar([r["seed"] for r in bad], [float(r["final_state_dist"]) for r in bad], width=0.8, color=CRIT,
           label=f"Not solved within {bad[0]['iters'] if bad else 20} MPC iterations ({len(bad)})")
    ax.set_xlabel("Evaluation seed (task)"); ax.set_ylabel("Final state distance")
    ax.set_xlim(0, 51); ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.savefig(f"{FIG}/fig11_pointmaze_corrected_per_task.png"); plt.close(fig)

    # Fig 12: start-goal distance vs MPC iterations needed
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.scatter([r["xy_dist"] for r in okr], [r["iters"] for r in okr], s=46, color=GOOD, edgecolor="white",
               linewidth=1.5, label=f"Solved ({len(okr)})", zorder=3)
    ax.scatter([r["xy_dist"] for r in bad], [r["iters"] for r in bad], s=110, color=CRIT, marker="X",
               edgecolor="white", linewidth=1.5, label=f"Not solved, stopped at the cap ({len(bad)})", zorder=4)
    for r in bad:
        ax.annotate(f"seed {r['seed']}", (r["xy_dist"], r["iters"]), xytext=(-12, -16),
                    textcoords="offset points", fontsize=9, color=INK, ha="right")
    ax.set_xlabel("Straight-line start-to-goal distance (x, y)"); ax.set_ylabel("MPC iterations used")
    ax.set_yticks([1, 2, 3, 5, 10, 15, 20]); ax.set_ylim(0, 22)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(f"{FIG}/fig12_pointmaze_corrected_distance_vs_iters.png"); plt.close(fig)

    # Fig 13: a solved task that needed re-planning (most MPC iterations among successes)
    rep = max(okr, key=lambda r: (r["iters"], r["xy_dist"]))
    copy(f"{ROOT}/{rep['output_dir']}/output_final.png", "fig13_pointmaze_corrected_replanning.png")
    vid(f"{ROOT}/{rep['output_dir']}", "output_final_0_*.mp4", "pointmaze_corrected_replanning.mp4")
    summary["pointmaze_corrected"]["example_replanning"] = dict(seed=rep["seed"], iters=rep["iters"], xy_dist=rep["xy_dist"])

    # Fig 14a/b: the failed task, first plan and last frames
    if bad:
        fb = bad[0]; d = f"{ROOT}/{fb['output_dir']}"
        copy(f"{d}/plan0.png", "fig14a_pointmaze_failure_first_plan.png")
        im = Image.open(f"{d}/output_final.png"); W, H = im.size
        im.crop((W - 1584, 0, W, H)).save(f"{FIG}/fig14b_pointmaze_failure_last_frames.png")
        vid(d, "output_final_0_*.mp4", "pointmaze_failure.mp4")
        summary["pointmaze_corrected"]["example_failure"] = dict(seed=fb["seed"], iters=fb["iters"], xy_dist=fb["xy_dist"],
                                                                 initial_state=fb["initial_state"], goal_state=fb["goal_state"])

    shutil.copy(f"{PMC}/results/comparison_table.tex", f"{ROOT}/results/pointmaze_comparison_table.tex")
    shutil.copy(f"{PMC}/results/per_run.csv", f"{ROOT}/results/pointmaze_corrected_per_run.csv")
    json.dump(summary, open(os.path.join(ROOT, "results", "summary.json"), "w"), indent=1, default=float)
    print(json.dumps(summary["pointmaze_corrected"], indent=1, default=float))


# ---------------- Adaptive CEM extension (experiments/adaptive_cem) ----------------
# Figures are produced by experiments/adaptive_cem/scripts/aggregate.py; copied here
# so all report figures live in results/figures/.
ACEM = f"{ROOT}/experiments/adaptive_cem"
if os.path.exists(f"{ACEM}/results/comparison_summary.json"):
    for src, dst in [("cem_iterations_comparison.png", "fig15_adaptive_cem_iterations.png"),
                     ("planning_time_comparison.png", "fig16_adaptive_cem_planning_time.png"),
                     ("per_task_iterations.png", "fig17_adaptive_cem_per_task_iterations.png"),
                     ("convergence_example.png", "fig18_adaptive_cem_convergence_example.png")]:
        copy(f"{ACEM}/figures/{src}", dst)
    shutil.copy(f"{ACEM}/results/comparison_summary.csv", f"{ROOT}/results/adaptive_cem_comparison_summary.csv")
    acs = json.load(open(f"{ACEM}/results/comparison_summary.json"))
    summary = json.load(open(os.path.join(ROOT, "results", "summary.json")))
    summary["adaptive_cem"] = {k: acs[k] for k in ("n_paired_tasks", "original", "adaptive", "reductions_pct",
                               "bootstrap_95ci", "sensitivity_excluding_mpc_cap_tasks", "change", "paired_stats")}
    json.dump(summary, open(os.path.join(ROOT, "results", "summary.json"), "w"), indent=1, default=float)
    print("adaptive CEM figures copied (fig15-fig18)")


# ---------------- Corrected Wall evaluation (50 distinct tasks, experiments/wall_corrected) ----------------
WC = f"{ROOT}/experiments/wall_corrected"
WALL_X, DOOR_Y, DOOR_HALF, WALL_W = 32, 30, 4, 6  # env/wall: fix_wall_location, hole_y, door_space, wall_width
if os.path.exists(f"{WC}/results/summary.json"):
    import csv
    from matplotlib.patches import Rectangle
    summary = json.load(open(os.path.join(ROOT, "results", "summary.json")))
    wagg = json.load(open(f"{WC}/results/summary.json"))
    wrows = list(csv.DictReader(open(f"{WC}/results/per_run.csv")))
    for r in wrows:
        r["seed"] = int(r["eval_seed"]); r["ok"] = r["success"] == "True"; r["iters"] = int(r["mpc_iterations"])
        r["s0"] = np.array(r["initial_state"].split(), float); r["sg"] = np.array(r["goal_state"].split(), float)
        door = np.array([WALL_X, DOOR_Y], float)
        r["via_door"] = float(np.linalg.norm(r["s0"][:2] - door) + np.linalg.norm(door - r["sg"][:2]))
    wok = [r for r in wrows if r["ok"]]; wbad = [r for r in wrows if not r["ok"]]
    by_via = sorted(wrows, key=lambda r: r["via_door"])
    summary["wall_corrected"] = dict(
        wagg["corrected"], failed_seeds=wagg["failed_seeds"],
        mpc_iters_hist={k: sum(r["iters"] == k for r in wrows) for k in sorted({r["iters"] for r in wrows})},
        tasks_crossing_wall=int(sum((r["s0"][0] - WALL_X) * (r["sg"][0] - WALL_X) < 0 for r in wrows)),
        via_door_path=dict(min=by_via[0]["via_door"], median=float(np.median([r["via_door"] for r in wrows])),
                           max=by_via[-1]["via_door"]),
        failed_via_door_rank={r["seed"]: by_via.index(r) + 1 for r in wbad})

    # Fig 1 again, now with the corrected Wall number
    ours["Wall"] = wagg["corrected"]["success_rate"]
    notes["Wall"] = f"{wagg['corrected']['successes']}/50\ntasks"
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    b1 = ax.bar(x - w / 2 - 0.01, [ours[e] * 100 for e in envs], w, color=BLUE, label="Our reproduction")
    b2 = ax.bar(x + w / 2 + 0.01, [PAPER[e] * 100 for e in envs], w, color=ORANGE, label="DINO-WM paper (MPC)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f}%",
                    ha="center", va="bottom", fontsize=10, color=INK)
    for i, e in enumerate(envs):
        ax.text(i - w / 2, 4, notes[e], ha="center", va="bottom", fontsize=8.5, color="white")
    ax.set_xticks(x, envs); ax.set_ylim(0, 115); ax.set_ylabel("Success rate (%)")
    ax.set_yticks([0, 25, 50, 75, 100]); ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.12))
    fig.savefig(f"{FIG}/fig01_success_vs_paper.png"); plt.close(fig)

    # Fig 19: task layout with the wall and door drawn (y axis down, as in the rendered images)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.add_patch(Rectangle((WALL_X - WALL_W / 2, 0), WALL_W, DOOR_Y - DOOR_HALF, color="#9a9994", zorder=1))
    ax.add_patch(Rectangle((WALL_X - WALL_W / 2, DOOR_Y + DOOR_HALF), WALL_W, 64 - DOOR_Y - DOOR_HALF,
                           color="#9a9994", zorder=1, label="Wall (gap = door)"))
    for r in wrows:
        c = GOOD if r["ok"] else CRIT
        ax.annotate("", xy=r["sg"][:2], xytext=r["s0"][:2],
                    arrowprops=dict(arrowstyle="->", color=c, lw=2.0 if not r["ok"] else 1.0,
                                    alpha=1 if not r["ok"] else 0.55), zorder=2 if r["ok"] else 4)
    ax.scatter([r["s0"][0] for r in wrows], [r["s0"][1] for r in wrows], s=26, color=BLUE, edgecolor="white",
               linewidth=1, zorder=5, label="Start")
    ax.scatter([r["sg"][0] for r in wrows], [r["sg"][1] for r in wrows], s=60, marker="*", color=BLUE,
               edgecolor="white", linewidth=0.6, zorder=5, label="Goal")
    for r in wbad:
        ax.annotate(f"seed {r['seed']}", r["s0"][:2], xytext=(4, -16), textcoords="offset points",
                    fontsize=9, color=INK, ha="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURF, ec="none", alpha=0.85), zorder=6)
    ax.plot([], [], color=GOOD, label="Task solved"); ax.plot([], [], color=CRIT, lw=2, label="Task failed")
    ax.set_xlim(0, 64); ax.set_ylim(64, 0); ax.set_aspect("equal"); ax.grid(False)
    ax.set_xlabel("x"); ax.set_ylabel("y (down, as in the rendered images)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    fig.savefig(f"{FIG}/fig19_wall_corrected_tasks.png"); plt.close(fig)

    # Fig 20: shortest path through the door vs MPC iterations
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.scatter([r["via_door"] for r in wok], [r["iters"] for r in wok], s=46, color=GOOD, edgecolor="white",
               linewidth=1.5, label=f"Solved ({len(wok)})", zorder=3)
    ax.scatter([r["via_door"] for r in wbad], [r["iters"] for r in wbad], s=110, color=CRIT, marker="X",
               edgecolor="white", linewidth=1.5, label=f"Not solved, stopped at the cap ({len(wbad)})", zorder=4)
    for r in wbad:
        ax.annotate(f"seed {r['seed']}", (r["via_door"], r["iters"]), xytext=(-12, -16),
                    textcoords="offset points", fontsize=9, color=INK, ha="right")
    ax.set_xlabel("Path length start → door centre → goal (straight segments)"); ax.set_ylabel("MPC iterations used")
    ax.set_yticks([1, 2, 3, 4, 5, 10, 15, 20]); ax.set_ylim(0, 22)
    ax.legend(frameon=False, loc="center left")
    fig.savefig(f"{FIG}/fig20_wall_corrected_path_vs_iters.png"); plt.close(fig)

    # Fig 21: solved task with the most re-planning
    wrep = max(wok, key=lambda r: (r["iters"], r["via_door"]))
    copy(f"{ROOT}/{wrep['output_dir']}/output_final.png", "fig21_wall_corrected_replanning.png")
    vid(f"{ROOT}/{wrep['output_dir']}", "output_final_0_*.mp4", "wall_corrected_replanning.mp4")
    summary["wall_corrected"]["example_replanning"] = dict(seed=wrep["seed"], iters=wrep["iters"], via_door=wrep["via_door"])

    # Fig 22a/b: last frames of each failed task
    for tag, r in zip("ab", sorted(wbad, key=lambda r: r["seed"])):
        im = Image.open(f"{ROOT}/{r['output_dir']}/output_final.png"); W, H = im.size
        im.crop((W - 1584, 0, W, H)).save(f"{FIG}/fig22{tag}_wall_failure_seed{r['seed']:02d}_last_frames.png")
        vid(f"{ROOT}/{r['output_dir']}", "output_final_0_*.mp4", f"wall_failure_seed{r['seed']:02d}.mp4")

    shutil.copy(f"{WC}/results/comparison_table.tex", f"{ROOT}/results/wall_comparison_table.tex")
    shutil.copy(f"{WC}/results/per_run.csv", f"{ROOT}/results/wall_corrected_per_run.csv")
    json.dump(summary, open(os.path.join(ROOT, "results", "summary.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in summary["wall_corrected"].items()}, indent=1, default=float))
