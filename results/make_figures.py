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
