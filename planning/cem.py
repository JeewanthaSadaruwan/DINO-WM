import json
import time

import torch
import numpy as np
from einops import rearrange, repeat
from omegaconf import DictConfig, OmegaConf
from .base_planner import BasePlanner
from utils import move_to_device


# Adaptive CEM (convergence-based early stopping) is OFF by default. With
# enabled=False the optimisation loop is numerically identical to the original
# DINO-WM CEM planner (same RNG calls in the same order, same updates, same
# success-based termination, same returned actions).
DEFAULT_ADAPTIVE_CEM = {
    "enabled": False,
    "min_iterations": 3,  # K_min: never stop on convergence before this many iterations
    "relative_tolerance": 0.01,  # tau: r_k < tau counts as "no meaningful improvement"
    "patience": 2,  # P: consecutive low-improvement iterations required to stop
    "delta": 1e-8,  # guards the relative-improvement denominator
    "return_best": True,  # on a convergence stop, return the lowest-cost sampled sequence
}


def _to_plain_dict(cfg):
    if cfg is None:
        return {}
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    return dict(cfg)


class CEMPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        topk,
        num_samples,
        var_scale,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="plan_0",
        log_filename="logs.json",
        adaptive_cem=None,
        stats_filename=None,
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.horizon = horizon
        self.topk = topk
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix

        unknown = set(_to_plain_dict(adaptive_cem)) - set(DEFAULT_ADAPTIVE_CEM)
        if unknown:
            raise ValueError(f"Unknown adaptive_cem keys: {sorted(unknown)}")
        acfg = {**DEFAULT_ADAPTIVE_CEM, **_to_plain_dict(adaptive_cem)}
        self.adaptive_enabled = bool(acfg["enabled"])
        self.adaptive_min_iterations = int(acfg["min_iterations"])
        self.adaptive_rel_tol = float(acfg["relative_tolerance"])
        self.adaptive_patience = int(acfg["patience"])
        self.adaptive_delta = float(acfg["delta"])
        self.adaptive_return_best = bool(acfg["return_best"])
        if self.adaptive_enabled:
            assert self.adaptive_min_iterations >= 1, "min_iterations must be >= 1"
            assert self.adaptive_patience >= 1, "patience must be >= 1"
            assert self.adaptive_rel_tol >= 0, "relative_tolerance must be >= 0"
            assert self.adaptive_delta > 0, "delta must be > 0"
        self.adaptive_cfg = acfg

        # Per-CEM-call instrumentation (JSON lines). None disables it.
        self.stats_filename = stats_filename
        self.call_index = 0

    def init_mu_sigma(self, obs_0, actions=None):
        """
        actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        mu, sigma could depend on current obs, but obs_0 is only used for providing n_evals for now
        """
        n_evals = obs_0["visual"].shape[0]
        sigma = self.var_scale * torch.ones([n_evals, self.horizon, self.action_dim])
        if actions is None:
            mu = torch.zeros(n_evals, 0, self.action_dim)
        else:
            mu = actions
        device = mu.device
        t = mu.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            new_mu = torch.zeros(n_evals, remaining_t, self.action_dim)
            mu = torch.cat([mu, new_mu.to(device)], dim=1)
        return mu, sigma

    def _sync(self):
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _install_wm_counters(self):
        """Count batched forward calls of the DINO encoder and the predictor.

        Forward hooks only observe; they do not change any computation. Counting
        is paused while the evaluator (simulator success check) runs, so only
        planning-time world-model usage is counted.
        """
        counters = {
            "active": True,
            "encoder_forward_calls": 0,
            "encoder_forward_samples": 0,
            "predictor_forward_calls": 0,
            "predictor_forward_samples": 0,
        }
        handles = []
        # DINO-WM calls `encoder.forward()` -> `base_model.forward_features()`
        # directly, which bypasses module hooks; `patch_embed` is invoked via
        # __call__ exactly once per encoder pass, so it is counted instead.
        encoder = getattr(self.wm, "encoder", None)
        encoder_probe = getattr(getattr(encoder, "base_model", None), "patch_embed", encoder)
        for name, module in (("encoder", encoder_probe), ("predictor", getattr(self.wm, "predictor", None))):
            if not isinstance(module, torch.nn.Module):
                continue

            def hook(_m, inputs, _out, name=name):
                if counters["active"]:
                    counters[f"{name}_forward_calls"] += 1
                    x = inputs[0] if inputs else None
                    if isinstance(x, torch.Tensor) and x.ndim > 0:
                        counters[f"{name}_forward_samples"] += int(x.shape[0])

            handles.append(module.register_forward_hook(hook))
        return counters, handles

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        """
        if self.stats_filename is None:
            return self._plan(obs_0, obs_g, actions, counters=None)
        counters, handles = self._install_wm_counters()
        try:
            return self._plan(obs_0, obs_g, actions, counters=counters)
        finally:
            for h in handles:
                h.remove()

    def _plan(self, obs_0, obs_g, actions, counters):
        self._sync()
        t_call_start = time.perf_counter()
        compute_time = 0.0  # world-model encoding/rollouts + CEM updates only
        eval_time = 0.0  # simulator evaluation + rendering inside the loop

        t0 = time.perf_counter()
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        z_obs_g = self.wm.encode_obs(trans_obs_g)

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]
        self._sync()
        compute_time += time.perf_counter() - t0

        # ---- convergence tracking (also recorded when adaptive stopping is off) ----
        best_cost = np.full(n_evals, np.inf)  # C^best per instance
        best_action = mu.clone()  # lowest-cost sampled sequence per instance
        stall = np.zeros(n_evals, dtype=int)  # consecutive iterations with r_k < tau
        converged = np.zeros(n_evals, dtype=bool)
        converged_at = [None] * n_evals  # 1-based iteration index
        active_iters = np.zeros(n_evals, dtype=int)
        cost_trace = [[] for _ in range(n_evals)]  # C_k
        best_trace = [[] for _ in range(n_evals)]  # C_k^best
        rel_trace = [[] for _ in range(n_evals)]  # r_k (None for k = 1)
        stop_reason = "max_iterations"
        n_iterations = 0

        for i in range(self.opt_steps):
            n_iterations = i + 1
            self._sync()
            t0 = time.perf_counter()
            # optimize individual instances
            losses = []
            for traj in range(n_evals):
                if self.adaptive_enabled and converged[traj]:
                    continue  # this instance is frozen at its best sequence
                cur_trans_obs_0 = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in trans_obs_0.items()
                }
                cur_z_obs_g = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in z_obs_g.items()
                }
                action = (
                    torch.randn(self.num_samples, self.horizon, self.action_dim).to(
                        self.device
                    )
                    * sigma[traj]
                    + mu[traj]
                )
                action[0] = mu[traj]  # optional: make the first one mu itself
                with torch.no_grad():
                    i_z_obses, i_zs = self.wm.rollout(
                        obs_0=cur_trans_obs_0,
                        act=action,
                    )

                loss = self.objective_fn(i_z_obses, cur_z_obs_g)
                topk_idx = torch.argsort(loss)[: self.topk]
                topk_action = action[topk_idx]
                iter_cost = loss[topk_idx[0]].item()  # C_k for this instance
                losses.append(iter_cost)

                # best-so-far bookkeeping (C^best_k = min(C^best_{k-1}, C_k))
                prev_best = best_cost[traj]
                if iter_cost < prev_best:
                    best_cost[traj] = iter_cost
                    best_action[traj] = action[topk_idx[0]].detach().clone()
                if np.isfinite(prev_best):
                    r_k = (prev_best - best_cost[traj]) / max(
                        abs(prev_best), self.adaptive_delta
                    )
                    stall[traj] = stall[traj] + 1 if r_k < self.adaptive_rel_tol else 0
                else:
                    r_k = None  # first iteration: no previous best to compare with
                active_iters[traj] += 1
                cost_trace[traj].append(iter_cost)
                best_trace[traj].append(float(best_cost[traj]))
                rel_trace[traj].append(None if r_k is None else float(r_k))

                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)

                if (
                    self.adaptive_enabled
                    and active_iters[traj] >= self.adaptive_min_iterations
                    and stall[traj] >= self.adaptive_patience
                ):
                    converged[traj] = True
                    converged_at[traj] = i + 1
                    if self.adaptive_return_best:
                        # freeze the instance at the best evaluated sequence rather
                        # than the (unevaluated) refitted mean
                        mu[traj] = best_action[traj]
            self._sync()
            compute_time += time.perf_counter() - t0

            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": np.mean(losses), "step": i + 1}
            )
            if self.evaluator is not None and i % self.eval_every == 0:
                if counters is not None:
                    counters["active"] = False  # do not count evaluator WM usage
                t0 = time.perf_counter()
                logs, successes, _, _ = self.evaluator.eval_actions(
                    mu, filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                eval_time += time.perf_counter() - t0
                if counters is not None:
                    counters["active"] = True
                if np.all(successes):
                    stop_reason = "success"
                    break  # terminate planning if all success
            if self.adaptive_enabled and np.all(converged):
                stop_reason = "converged"
                break  # every instance has converged

        self._sync()
        wall_time = time.perf_counter() - t_call_start
        self._write_stats(
            n_evals=n_evals,
            n_iterations=n_iterations,
            stop_reason=stop_reason,
            converged_at=converged_at,
            active_iters=active_iters,
            cost_trace=cost_trace,
            best_trace=best_trace,
            rel_trace=rel_trace,
            compute_time=compute_time,
            eval_time=eval_time,
            wall_time=wall_time,
            counters=counters,
        )
        self.call_index += 1

        return mu, np.full(n_evals, np.inf)  # all actions are valid

    def _write_stats(
        self,
        n_evals,
        n_iterations,
        stop_reason,
        converged_at,
        active_iters,
        cost_trace,
        best_trace,
        rel_trace,
        compute_time,
        eval_time,
        wall_time,
        counters=None,
    ):
        if self.stats_filename is None:
            return
        instances = []
        for b in range(n_evals):
            if converged_at[b] is not None:
                reason = "converged"
            elif stop_reason == "success":
                reason = "success"
            else:
                reason = "max_iterations"
            instances.append(
                {
                    "instance": b,
                    "iterations": int(active_iters[b]),
                    "stop_reason": reason,
                    "converged_at": converged_at[b],
                    "returned": "best_sample"
                    if (converged_at[b] is not None and self.adaptive_return_best)
                    else "cem_mean",
                    "candidates_evaluated": int(active_iters[b]) * self.num_samples,
                    "cost": cost_trace[b],
                    "best_cost": best_trace[b],
                    "rel_improvement": rel_trace[b],
                }
            )
        record = {
            "call_index": self.call_index,
            "logging_prefix": self.logging_prefix,
            "adaptive_enabled": self.adaptive_enabled,
            "adaptive_cfg": self.adaptive_cfg if self.adaptive_enabled else None,
            "num_samples": self.num_samples,
            "topk": self.topk,
            "horizon": self.horizon,
            "max_iterations": self.opt_steps,
            "n_evals": n_evals,
            "iterations": n_iterations,  # outer CEM iterations executed
            "stop_reason": stop_reason,  # success | converged | max_iterations
            "candidates_evaluated": int(np.sum(active_iters)) * self.num_samples,
            # one batched wm.rollout per active instance per iteration
            "wm_rollout_calls": int(np.sum(active_iters)),
            "wm_forward_counts": None
            if counters is None
            else {k: v for k, v in counters.items() if k != "active"},
            "compute_time_s": compute_time,  # WM inference + CEM updates (GPU-synced)
            "eval_time_s": eval_time,  # in-loop simulator check + rendering
            "wall_time_s": wall_time,
            "instances": instances,
        }
        with open(self.stats_filename, "a") as f:
            f.write(json.dumps(record) + "\n")
