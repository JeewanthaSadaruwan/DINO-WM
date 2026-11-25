"""Estimate PushT block pose (x, y, angle) and agent position from 224x224 RGB frames.

World <-> pixel: pixel = world * 224/512 (checked against state_0 in plan_targets.pkl).
The T template is taken from a clean frame with a known block pose (state_0) and fitted to other
frames by rotation + translation (IoU on the block mask). Validated on the goal frame (known state_g).
"""
import numpy as np

S = 224 / 512.0
DARKEN = 38  # evaluator draws env frames with "- 0.3" in [-1,1] space = 0.15*255 = 38 grey levels


def undarken(img):
    return np.clip(img.astype(int) + DARKEN, 0, 255).astype(np.uint8)


def masks(img):
    i = img.astype(int); r, g, b = i[..., 0], i[..., 1], i[..., 2]
    agent = (b - r > 90) & (b > 170)
    block = (b - r > 15) & (b - r <= 90) & (g < b) & (r < 205) & ~agent
    return block, agent


def agent_xy(img):
    _, a = masks(img)
    if a.sum() < 15: return None
    ys, xs = np.nonzero(a)
    return np.array([xs.mean(), ys.mean()]) / S


def rot(t):
    c, s = np.cos(t), np.sin(t); return np.array([[c, -s], [s, c]])


class TFitter:
    def __init__(self, clean_img, state):
        blk, _ = masks(clean_img)
        ys, xs = np.nonzero(blk)
        sub = np.array([(dx, dy) for dx in (0.25, 0.75) for dy in (0.25, 0.75)])  # 2x2 sub-pixel samples (no holes when rotated)
        w = (np.stack([xs, ys], 1)[:, None, :] + sub[None]).reshape(-1, 2) / S   # world coords of block pixels
        self.body = (w - state[2:4]) @ rot(state[4])              # body frame: R^T (w - p)
        self.c_body = self.body.mean(0)

    def render(self, x, y, t):
        w = self.body @ rot(t).T + np.array([x, y])
        p = np.floor(w * S).astype(int)
        p = p[(p[:, 0] >= 0) & (p[:, 0] < 224) & (p[:, 1] >= 0) & (p[:, 1] < 224)]
        m = np.zeros((224, 224), bool); m[p[:, 1], p[:, 0]] = True
        return m

    def fit(self, img):
        blk, _ = masks(img)
        if blk.sum() < 50: return None
        ys, xs = np.nonzero(blk); cen = np.array([xs.mean() + 0.5, ys.mean() + 0.5]) / S

        def score(x, y, t):
            m = self.render(x, y, t); return (m & blk).sum() / max((m | blk).sum(), 1)

        def origin(t, c):  # body origin so that the template centroid sits at c
            return c - rot(t) @ self.c_body
        best = max(((score(*origin(t, cen), t), t) for t in np.radians(np.arange(0, 360, 3))))
        t0 = best[1]; c0 = cen.copy(); bs = best[0]
        for step_t, step_c in ((np.radians(1), 2.0), (np.radians(0.25), 0.7)):
            improved = True
            while improved:
                improved = False
                for dt in (-step_t, 0, step_t):
                    for dx in (-step_c, 0, step_c):
                        for dy in (-step_c, 0, step_c):
                            c = c0 + [dx, dy]; s = score(*origin(t0 + dt, c), t0 + dt)
                            if s > bs + 1e-9: bs, t0, c0, improved = s, t0 + dt, c, True
        x, y = origin(t0, c0)
        return dict(x=float(x), y=float(y), angle=float(t0 % (2 * np.pi)), iou=float(bs))


def ang_diff(a, b):
    d = (a - b) % (2 * np.pi); return float(np.degrees(min(d, 2 * np.pi - d)))
