"""Failure analysis of PushT runs 2 and 9 from SAVED outputs only (no inference, no GPU).

Usage (repo root):  python scripts/pusht_failure_analysis.py
Writes results/failure_analysis/run_XX/{actual_simulation, predicted_rollout, actual_vs_predicted,
actual_vs_openloop_prediction}.mp4, three comparison PNGs, per_iteration.csv, progress.png, analysis.json.

What the saved files contain (planning/evaluator.py, planning/mpc.py, planning/cem.py):
  plan_K_output_N.png  CEM evaluation of the current mean plan after CEM step N in MPC iteration K.
                       Top row: simulator executing that plan from the CURRENT real state;
                       bottom row: decoded world-model prediction of the same plan from the same
                       observation; right column: goal. One column per world-model step (5 actions).
                       N=30 is the final CEM step, i.e. the plan that MPC executed.
  planK_0_failure.mp4  MPC evaluation after iteration K: ALL executed actions so far replayed from the
                       episode's START. Top-left: simulator (one frame per action);
                       bottom-left: world model rolled out OPEN-LOOP from the first observation over the
                       whole action sequence (each WM frame repeated 5x); right: goal.
Env frames in both are darkened by 0.3 in [-1,1] space (= 38 grey levels) by the plotting code;
undarken() reverses this exactly (verified against the clean obs_0 in plan_targets.pkl).
"""
import csv, glob, json, os, pickle, re, sys
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pusht_pose import TFitter, agent_xy, ang_diff, undarken, rot  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "failure_analysis")
RUNS = [2, 9]
FPS = 20
FS = 5            # frameskip: actions per world-model step
H = 5             # planning horizon (WM steps) = n_taken_actions
TH_POS, TH_ANG, TH_AGENT = 10.0, 5.0, 10.0   # divergence thresholds (world units / degrees)
AGENT_R = 15.0

import matplotlib  # noqa: E402
from matplotlib import font_manager  # noqa: E402
FONT = font_manager.findfont("DejaVu Sans"); FONT_B = font_manager.findfont("DejaVu Sans:bold")
f = lambda s, b=False: ImageFont.truetype(FONT_B if b else FONT, s)


# ------------------------------------------------------------------ locate run outputs
def locate(run):
    log = os.path.join(ROOT, "evaluation_logs", "pusht", f"eval_{run}.log")
    t = open(log, errors="ignore").read()
    d = re.search(r"Planning result saved dir: (\S+)", t).group(1)
    ov = re.search(r"Error executing job with overrides: (\[.*?\])", t)
    iters = [int(x) for x in re.findall(r"MPC iter (\d+) Eval", t)]
    oom = re.search(r"(torch\.OutOfMemoryError: [^\n]*)", t)
    frames = re.findall(r'File "(/home/[^"]*dino_wm/[^"]+)", line (\d+), in (\w+)', t)
    return dict(run=run, log=os.path.relpath(log, ROOT), dir=d, overrides=ov and ov.group(1),
                seed_arg=int(re.search(r"seed=(\d+)", ov.group(1)).group(1)) if ov else None,
                eval_seed=re.search(r"eval_seed:\s+(\[[^\]]*\])", t).group(1),
                mpc_eval_started=iters, oom=oom and oom.group(1),
                oom_stack=[f"{os.path.relpath(p, ROOT)}:{l} {fn}" for p, l, fn in frames][-6:])


def panels(png):
    im = np.array(Image.open(png).convert("RGB"))
    P = lambda r, c: im[2 + r * 226: 2 + r * 226 + 224, 2 + c * 226: 2 + c * 226 + 224]
    return [undarken(P(0, c)) for c in range(H + 1)], [P(1, c).copy() for c in range(H + 1)]


# ------------------------------------------------------------------ drawing helpers
def up(img, k=2):
    return np.repeat(np.repeat(img, k, 0), k, 1)


def label_frame(img, top_lines, bottom_lines=(), w=None, border=None):
    """img: HxWx3 uint8 -> adds a header (and optional footer) band."""
    h, wd = img.shape[:2]; w = w or wd
    head = 30 * len(top_lines) + 8; foot = 26 * len(bottom_lines) + (8 if bottom_lines else 0)
    can = Image.new("RGB", (w, head + h + foot), (255, 255, 255))
    can.paste(Image.fromarray(img), ((w - wd) // 2, head))
    d = ImageDraw.Draw(can)
    for i, (txt, size, bold, col) in enumerate(top_lines):
        d.text((w // 2, 6 + 30 * i), txt, fill=col, font=f(size, bold), anchor="ma")
    for i, (txt, size, bold, col) in enumerate(bottom_lines):
        d.text((w // 2, head + h + 6 + 26 * i), txt, fill=col, font=f(size, bold), anchor="ma")
    if border:
        d.rectangle([(w - wd) // 2, head, (w - wd) // 2 + wd - 1, head + h - 1], outline=border, width=6)
    return np.array(can)


def hstack(imgs, gap=16):
    h = max(i.shape[0] for i in imgs)
    out = [np.pad(i, ((0, h - i.shape[0]), (0, 0), (0, 0)), constant_values=255) for i in imgs]
    sep = np.full((h, gap, 3), 255, np.uint8)
    res = out[0]
    for i in out[1:]: res = np.concatenate([res, sep, i], 1)
    return res


def pad16(img):
    h, w = img.shape[:2]
    return np.pad(img, ((0, (-h) % 16), (0, (-w) % 16), (0, 0)), constant_values=255)


def writer(path):
    return imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=16,
                              ffmpeg_log_level="error")


INK, RED, BLUE, GREY = (11, 11, 11), (208, 59, 59), (42, 120, 214), (82, 81, 78)


def strip(rows, goal, title, sub, col_labels, row_labels):
    """rows: list of lists of 224x224 frames; goal frame appended as the last column of each row."""
    cell = 224; gap = 6; lw = 230; th = 70; ch = 26
    n = len(rows[0]) + 1
    W = lw + n * (cell + gap); Hh = th + ch + len(rows) * (cell + gap)
    can = Image.new("RGB", (W, Hh), (255, 255, 255)); d = ImageDraw.Draw(can)
    d.text((10, 8), title, fill=INK, font=f(22, True)); d.text((10, 40), sub, fill=GREY, font=f(16))
    for c, cl in enumerate(col_labels + ["GOAL OBSERVATION"]):
        d.text((lw + c * (cell + gap) + cell // 2, th + 2), cl, fill=INK, font=f(14, c == n - 1), anchor="ma")
    for r, (row, rl) in enumerate(zip(rows, row_labels)):
        y = th + ch + r * (cell + gap)
        for li, line in enumerate(rl.split("\n")):
            d.text((10, y + cell // 2 - 12 + 22 * li), line, fill=INK, font=f(16, li == 0))
        for c, im in enumerate(list(row) + [goal]):
            can.paste(Image.fromarray(im), (lw + c * (cell + gap), y))
    return can


# ------------------------------------------------------------------ analysis
def analyse(run):
    info = locate(run)
    d = info["dir"]; tag = f"run_{run:02d}"; od = os.path.join(OUT, tag); os.makedirs(od, exist_ok=True)
    print(f"== {tag}: {d}")
    tg = pickle.load(open(f"{d}/plan_targets.pkl", "rb"))
    s0, sg = tg["state_0"][0], tg["state_g"][0]
    obs0, goal = tg["obs_0"]["visual"][0, 0], tg["obs_g"]["visual"][0, 0]
    F = TFitter(obs0, s0)
    gfit = F.fit(goal); gagent = agent_xy(goal)

    plan_imgs = sorted(int(m) for m in re.findall(r"plan(\d+)\.png", " ".join(os.listdir(d))))
    cem_iters = sorted({int(m) for m in re.findall(r"plan_(\d+)_output_30\.png", " ".join(os.listdir(d)))})
    n_cem_steps = {k: len(glob.glob(f"{d}/plan_{k}_output_*.png")) for k in cem_iters}
    last_exec = plan_imgs[-1]                        # last MPC iteration whose evaluation completed
    executed = list(range(last_exec + 1))
    n_actions = (last_exec + 1) * H * FS
    mp4 = f"{d}/plan{last_exec}_0_failure.mp4"
    vid = imageio.mimread(mp4, memtest=False)
    act_cont = [undarken(v[:224, :224]) for v in vid]
    ol_pred = [v[224:, :224].copy() for v in vid]
    assert len(act_cont) == n_actions + 1, (len(act_cont), n_actions)

    logs = [json.loads(l) for l in open(f"{d}/logs.json")]
    mpc_logs = [l for l in logs if "mpc/success_rate" in l]

    # ---------- per-iteration: actual vs predicted (CEM final plan, from the current real state)
    rows, A, P = [], {}, {}
    prev = None
    for k in cem_iters:
        a, p = panels(f"{d}/plan_{k}_output_30.png"); A[k], P[k] = a, p
        af = [F.fit(x) for x in a]; pf = [F.fit(x) for x in p]
        aa = [agent_xy(x) for x in a]; pa = [agent_xy(x) for x in p]
        e = lambda u, v: float(np.hypot(u["x"] - v["x"], u["y"] - v["y"]))
        r = dict(iter=k, executed=k in executed,
                 err_block_pos=e(af[H], pf[H]), err_block_ang=ang_diff(af[H]["angle"], pf[H]["angle"]),
                 err_agent=float(np.linalg.norm(aa[H] - pa[H])) if aa[H] is not None and pa[H] is not None else None,
                 err_block_pos_t0=e(af[0], pf[0]), pred_iou_min=min(x["iou"] for x in pf),
                 act_block_move=e(af[0], af[H]), act_block_rot=ang_diff(af[0]["angle"], af[H]["angle"]),
                 pred_block_move=e(pf[0], pf[H]), pred_block_rot=ang_diff(pf[0]["angle"], pf[H]["angle"]),
                 act_agent_move=float(np.linalg.norm(aa[H] - aa[0])), pred_agent_move=float(np.linalg.norm(pa[H] - pa[0])),
                 block_goal_dist=e(af[H], gfit), block_goal_ang=ang_diff(af[H]["angle"], gfit["angle"]),
                 agent_goal_dist=float(np.linalg.norm(aa[H] - gagent)),
                 pred_block_goal_dist=e(pf[H], gfit), pred_block_goal_ang=ang_diff(pf[H]["angle"], gfit["angle"]),
                 pred_agent_goal_dist=float(np.linalg.norm(pa[H] - gagent)))
        dv_a = np.array([af[H]["x"] - af[0]["x"], af[H]["y"] - af[0]["y"]])
        dv_p = np.array([pf[H]["x"] - pf[0]["x"], pf[H]["y"] - pf[0]["y"]])
        r["move_dir_cos"] = (float(dv_a @ dv_p / np.linalg.norm(dv_a) / np.linalg.norm(dv_p))
                             if np.linalg.norm(dv_a) > 3 and np.linalg.norm(dv_p) > 3 else None)
        # minimum agent-block gap over the iteration (continuous trajectory, every action)
        if k in executed:
            gaps = []
            for t in range(k * 25, k * 25 + 26, 1):
                ag = agent_xy(act_cont[t])
                if t % 5 == 0 or prev is None:
                    prev = F.fit(act_cont[t]) or prev
                w = F.body @ rot(prev["angle"]).T + [prev["x"], prev["y"]]
                gaps.append(float(np.min(np.linalg.norm(w - ag, axis=1)) - AGENT_R))
            r["min_agent_block_gap"] = min(gaps)
            # sanity: continuous replay vs CEM evaluator rollout at the same action index
            cf = [F.fit(act_cont[k * 25 + 5 * j]) for j in (0, H)]
            r["sync_block_pos_diff"] = max(e(cf[0], af[0]), e(cf[1], af[H]))
            r["sync_agent_diff"] = max(float(np.linalg.norm(agent_xy(act_cont[k * 25 + 5 * j]) - aa[jj]))
                                       for j, jj in ((0, 0), (H, H)))
        r["divergent"] = (r["err_block_pos"] > TH_POS or r["err_block_ang"] > TH_ANG
                          or (r["err_agent"] or 0) > TH_AGENT)
        rows.append(r)
        print(f"  iter {k:2d} pred err pos {r['err_block_pos']:5.1f} ang {r['err_block_ang']:4.1f} agent "
              f"{r['err_agent'] or -1:5.1f} | act block move {r['act_block_move']:5.1f} rot {r['act_block_rot']:4.1f} "
              f"| goal d {r['block_goal_dist']:5.1f} ang {r['block_goal_ang']:5.1f} agent {r['agent_goal_dist']:5.1f}")

    # ---------- open-loop (whole trajectory from the start) divergence
    ol = []
    for t in range(0, n_actions + 1, 5):
        af, pf = F.fit(act_cont[t]), F.fit(ol_pred[t])
        aa, pa = agent_xy(act_cont[t]), agent_xy(ol_pred[t])
        ol.append(dict(t=t, err_block_pos=float(np.hypot(af["x"] - pf["x"], af["y"] - pf["y"])) if pf else None,
                       err_block_ang=ang_diff(af["angle"], pf["angle"]) if pf else None,
                       pred_iou=pf["iou"] if pf else 0.0,
                       err_agent=float(np.linalg.norm(aa - pa)) if aa is not None and pa is not None else None))
    def ol_div(o):
        return (o["err_block_pos"] is None or o["err_block_pos"] > TH_POS or o["err_block_ang"] > TH_ANG
                or (o["err_agent"] or 0) > TH_AGENT)
    ol_first = next((o for o in ol if ol_div(o)), None)

    # ---------- summary numbers
    ex = [r for r in rows if r["executed"]]
    first_div = next((r for r in ex if r["divergent"]), None)
    still = [r for r in ex if r["act_block_move"] < 2 and r["act_block_rot"] < 1]
    contact = [r for r in ex if r["min_agent_block_gap"] < 5]
    best = min(ex, key=lambda r: r["block_goal_dist"])
    last_block_move = max((r["iter"] for r in ex if r["act_block_move"] >= 2 or r["act_block_rot"] >= 1), default=None)
    with open(os.path.join(od, "per_iteration.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

    # ---------- figures
    cols = [f"t+{5 * j} actions" + (" (start)" if j == 0 else "") for j in range(H + 1)]
    def it_strip(k, title):
        return strip([A[k], P[k]], goal, title,
                     f"Run {run}, MPC iteration {k}: CEM's final plan (5 world-model steps = 25 actions), "
                     "executed in the simulator vs predicted by DINO-WM from the same observation.",
                     cols, ["ACTUAL ENVIRONMENT\n(simulator)", "WORLD MODEL\nPREDICTION\n(decoded)"])
    it_strip(0, "Initial prediction").save(os.path.join(od, "initial_prediction.png"))
    it_strip(last_exec, f"Final available comparison (last executed MPC iteration, {last_exec})").save(
        os.path.join(od, "final_available_comparison.png"))
    t0 = ol_first["t"]
    ts = sorted({0, t0, t0 + 50, n_actions // 4, n_actions // 2, n_actions})
    strip([[act_cont[t] for t in ts], [ol_pred[t] for t in ts]], goal,
          f"Open-loop drift: world model rolled out from the episode start (first over threshold at action {t0})",
          f"Run {run}: all executed actions fed to DINO-WM from the FIRST observation, never re-anchored on real frames. "
          "Diagnostic only: the planner never uses this; it re-plans from the real state every 25 actions.",
          [f"action {t}" for t in ts],
          ["ACTUAL ENVIRONMENT\n(simulator)", "WORLD MODEL\nOPEN-LOOP\nFROM START"]).save(
        os.path.join(od, "openloop_divergence.png"))
    if first_div is not None:
        r = first_div
        what = ", ".join(s for s, c in [(f"block position error {r['err_block_pos']:.1f} (threshold {TH_POS:.0f})", r["err_block_pos"] > TH_POS),
                                        (f"block angle error {r['err_block_ang']:.1f}° (threshold {TH_ANG:.0f}°)", r["err_block_ang"] > TH_ANG),
                                        (f"agent position error {r['err_agent']:.1f} (threshold {TH_AGENT:.0f})", (r["err_agent"] or 0) > TH_AGENT)] if c)
        it_strip(r["iter"], f"First divergence over threshold: MPC iteration {r['iter']} ({what} after 25 actions)").save(
            os.path.join(od, "first_visible_divergence.png"))
        div_note = f"per-iteration prediction, MPC iteration {r['iter']}: {what}"
    else:
        t0 = ol_first["t"]; ts = [max(0, t0 - 10), max(0, t0 - 5), t0, t0 + 5, t0 + 10, t0 + 15]
        ts = [t for t in ts if t <= n_actions]
        strip([[act_cont[t] for t in ts], [ol_pred[t] for t in ts]], goal,
              f"First visible divergence: open-loop rollout, action {t0}",
              f"Run {run}: NO per-iteration (planner) prediction exceeded the threshold. This is the world model rolled "
              f"out open-loop from the episode start over all executed actions (diagnostic only; not used by the planner).",
              [f"action {t}" for t in ts],
              ["ACTUAL ENVIRONMENT\n(simulator)", "WORLD MODEL\nOPEN-LOOP\nFROM START"]).save(
            os.path.join(od, "first_visible_divergence.png"))
        div_note = f"open-loop rollout from the start, action {t0} (no per-iteration divergence)"

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True)
    it = [r["iter"] for r in ex]
    ax[0].plot(it, [r["block_goal_dist"] for r in ex], color="#2a78d6", lw=2, label="block position error to goal")
    ax[0].plot(it, [r["agent_goal_dist"] for r in ex], color="#eb6834", lw=2, label="agent position error to goal")
    ax[0].set_ylabel("World units"); ax[0].legend(frameon=False, fontsize=9)
    ax[1].plot(it, [r["block_goal_ang"] for r in ex], color="#2a78d6", lw=2)
    ax[1].axhline(20, color="#52514e", ls="--", lw=1); ax[1].text(it[-1], 21, "success needs < 20°", ha="right", fontsize=8)
    ax[1].set_ylabel("Block angle error\nto goal (°)")
    ax[2].plot(it, [r["err_block_pos"] for r in ex], color="#d03b3b", lw=2, label="planner prediction: block position error")
    ax[2].plot(it, [r["err_agent"] or np.nan for r in ex], color="#52514e", lw=1.5, label="planner prediction: agent position error")
    ax[2].plot([o["t"] / 25 for o in ol], [o["err_block_pos"] if o["err_block_pos"] is not None else np.nan for o in ol],
               color="#d03b3b", lw=1, ls=":", label="open-loop from start: block position error")
    ax[2].axhline(TH_POS, color="#52514e", ls="--", lw=1)
    ax[2].set_ylabel("Prediction error\n(world units)"); ax[2].set_xlabel("MPC iteration (25 actions each)")
    ax[2].legend(frameon=False, fontsize=8); ax[2].set_ylim(0, max(60, TH_POS * 2))
    for a_ in ax: a_.spines[["top", "right"]].set_visible(False); a_.grid(color="#e4e3df")
    fig.suptitle(f"PushT run {run}: progress towards the goal and prediction error per MPC iteration", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(od, "progress.png"), dpi=160); plt.close(fig)

    # ---------- videos
    G2 = up(goal)
    hdr = lambda t: (t, 18, True, INK)
    # A: actual
    with writer(os.path.join(od, "actual_simulation.mp4")) as w:
        for t, fr in enumerate(act_cont):
            k = min(t // 25, last_exec)
            w.append_data(pad16(label_frame(up(fr), [hdr("ACTUAL ENVIRONMENT"), (f"Run {run} | MPC iter {k} | action {t}/{n_actions}", 15, False, GREY)])))
    # B: predicted, per MPC planning call
    with writer(os.path.join(od, "predicted_rollout.mp4")) as w:
        for k in cem_iters:
            note = "" if k in executed else " (planned, never executed: run crashed)"
            for t in range(26):
                j = t // 5
                w.append_data(pad16(label_frame(up(P[k][j]), [hdr("WORLD MODEL PREDICTION"),
                    (f"MPC iter {k}{note}", 15, False, RED if note else GREY),
                    (f"WM step {j}/5 = action {5 * j} of this plan" + (" (decoded start obs)" if j == 0 else ""), 14, False, GREY)],
                    border=RED if t < 6 else None)))
    # C: actual vs predicted (synchronised per MPC iteration)
    with writer(os.path.join(od, "actual_vs_predicted.mp4")) as w:
        for k in executed:
            for t in range(26):
                j = t // 5; new = t < 6
                lp = label_frame(up(act_cont[k * 25 + t]), [hdr("ACTUAL ENVIRONMENT"), (f"action {t}/25 of this plan", 14, False, GREY)])
                mp = label_frame(up(P[k][j]), [hdr("WORLD MODEL PREDICTION"),
                                 (f"WM step {j} (= action {5 * j})" + ("  exact match" if t % 5 == 0 else "  held"), 14, False, GREY)],
                                 border=RED if new else None)
                gp = label_frame(G2, [hdr("GOAL OBSERVATION"), ("fixed", 14, False, GREY)])
                body = hstack([lp, mp, gp])
                foot = Image.new("RGB", (body.shape[1], 44), (255, 255, 255)); dd = ImageDraw.Draw(foot)
                dd.text((body.shape[1] // 2, 8),
                        f"Run {run}  |  MPC iteration {k}/{last_exec}  |  total action {k * 25 + t}/{n_actions}"
                        + ("  |  NEW PLAN from the real state" if new else ""),
                        fill=RED if new else INK, font=f(17, new), anchor="ma")
                w.append_data(pad16(np.concatenate([body, np.array(foot)], 0)))
    # D (extra): open-loop prediction from the start
    with writer(os.path.join(od, "actual_vs_openloop_prediction.mp4")) as w:
        for t in range(n_actions + 1):
            lp = label_frame(up(act_cont[t]), [hdr("ACTUAL ENVIRONMENT"), (f"action {t}/{n_actions}", 14, False, GREY)])
            mp = label_frame(up(ol_pred[t]), [hdr("WM OPEN-LOOP FROM START"), (f"WM step {t // 5} (diagnostic only)", 14, False, GREY)])
            gp = label_frame(G2, [hdr("GOAL OBSERVATION"), ("fixed", 14, False, GREY)])
            body = hstack([lp, mp, gp])
            foot = Image.new("RGB", (body.shape[1], 44), (255, 255, 255)); dd = ImageDraw.Draw(foot)
            dd.text((body.shape[1] // 2, 8), "Open-loop: the world model is NOT re-anchored on real observations "
                    "(the planner never uses this; it re-plans every 25 actions)", fill=GREY, font=f(15), anchor="ma")
            w.append_data(pad16(np.concatenate([body, np.array(foot)], 0)))

    # ---------- verification of the written videos
    vcheck = {}
    for v in ["actual_simulation", "predicted_rollout", "actual_vs_predicted", "actual_vs_openloop_prediction"]:
        p = os.path.join(od, f"{v}.mp4"); rd = imageio.get_reader(p); n = rd.count_frames(); m = rd.get_meta_data()
        rd.close(); vcheck[v] = dict(frames=n, fps=m["fps"], size=m["size"], seconds=round(n / m["fps"], 1))
    expected = dict(actual_simulation=n_actions + 1, predicted_rollout=26 * len(cem_iters),
                    actual_vs_predicted=26 * len(executed), actual_vs_openloop_prediction=n_actions + 1)
    for v, n in expected.items(): vcheck[v]["frame_count_ok"] = vcheck[v]["frames"] == n

    final = mpc_logs[-1]
    an = dict(
        run=run, source=info, output_dir=os.path.relpath(d, ROOT),
        files_used=dict(actual_continuous=os.path.relpath(mp4, ROOT) + " (top-left quadrant, undarkened)",
                        openloop_prediction=os.path.relpath(mp4, ROOT) + " (bottom-left quadrant)",
                        per_iteration=f"plan_K_output_30.png, K={cem_iters[0]}..{cem_iters[-1]}",
                        goal="plan_targets.pkl obs_g (clean)"),
        initial_state=s0.tolist(), goal_state=sg.tolist(),
        mpc=dict(completed_iterations=len(executed), planned_iterations=len(cem_iters),
                 cem_steps_per_iteration=sorted(set(n_cem_steps.values())),
                 executed_actions=n_actions,
                 final_recorded=dict(after_iteration=last_exec, state_dist=final["mpc/mean_state_dist"],
                                     visual_dist=final["mpc/mean_visual_dist"], proprio_dist=final["mpc/mean_proprio_dist"],
                                     success=bool(final["mpc/success_rate"])),
                 termination=(f"CUDA out of memory in MPC iteration {cem_iters[-1]}'s evaluation, while decoding the open-loop "
                              f"visualisation of all {(cem_iters[-1] + 1) * H} world-model steps "
                              "(planning/evaluator.py:132 -> models/visual_world_model.py decode_obs -> models/vqvae.py decode)"),
                 oom_message=info["oom"]),
        estimates=dict(
            note="block pose and agent position estimated from images (scripts/pusht_pose.py); world units = env pixels (512 frame)",
            estimator_validation_on_goal_image=dict(block_pos_err=float(np.hypot(gfit["x"] - sg[2], gfit["y"] - sg[3])),
                                                    block_ang_err_deg=ang_diff(gfit["angle"], sg[4]),
                                                    agent_pos_err=float(np.linalg.norm(gagent - sg[:2]))),
            start=dict(block_goal_dist=float(np.hypot(s0[2] - sg[2], s0[3] - sg[3])), block_goal_ang=ang_diff(s0[4], sg[4]),
                       agent_goal_dist=float(np.linalg.norm(s0[:2] - sg[:2]))),
            end=dict(block_goal_dist=ex[-1]["block_goal_dist"], block_goal_ang=ex[-1]["block_goal_ang"],
                     agent_goal_dist=ex[-1]["agent_goal_dist"]),
            closest_block=dict(iter=best["iter"], block_goal_dist=best["block_goal_dist"], block_goal_ang=best["block_goal_ang"]),
            last_iteration_block_moved=last_block_move,
            iterations_block_still=len(still), iterations_with_contact=len(contact),
            per_iteration_prediction=dict(
                threshold=dict(block_pos=TH_POS, block_ang_deg=TH_ANG, agent=TH_AGENT),
                mean_err_block_pos=float(np.mean([r["err_block_pos"] for r in ex])),
                max_err_block_pos=float(max(r["err_block_pos"] for r in ex)),
                max_err_block_ang=float(max(r["err_block_ang"] for r in ex)),
                max_err_agent=float(max(r["err_agent"] or 0 for r in ex)),
                divergent_iterations=[r["iter"] for r in ex if r["divergent"]],
                first_divergent_iteration=first_div and first_div["iter"]),
            openloop_from_start=dict(first_divergent_action=ol_first and ol_first["t"],
                                     err_block_pos_at_end=ol[-1]["err_block_pos"], pred_iou_at_end=ol[-1]["pred_iou"]),
            sync_check=dict(max_block_pos_diff=float(max(r["sync_block_pos_diff"] for r in ex)),
                            max_agent_diff=float(max(r["sync_agent_diff"] for r in ex)),
                            note="continuous replay (planK mp4) vs CEM-evaluator rollout of the same plan at actions 0 and 25 of each iteration")),
        first_visible_divergence=div_note,
        videos=vcheck,
    )
    json.dump(an, open(os.path.join(od, "analysis.json"), "w"), indent=1, default=float)
    print(json.dumps({k: an[k] for k in ("mpc", "estimates", "videos")}, indent=1, default=float))
    return an


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    res = [analyse(r) for r in (map(int, sys.argv[1:]) if len(sys.argv) > 1 else RUNS)]
