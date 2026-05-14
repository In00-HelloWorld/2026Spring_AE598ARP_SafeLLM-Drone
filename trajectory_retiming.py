from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from problems import get_problem
from schemas import AgentTrajectory, MissionSolution, ProblemSpec, Waypoint


@dataclass(frozen=True)
class RetimingConfig:
    method: str = "slsqp"
    min_segment_dt: float = 0.05
    moving_epsilon: float = 1.0e-7
    velocity_margin: float = 1.0e-4
    limit_buffer: float = 0.995
    max_segment_scale: float = 8.0
    solver_max_iter: int = 150
    solver_ftol: float = 1.0e-7
    feasibility_tol: float = 1.0e-5
    preserve_holds: bool = True
    enforce_jerk: bool = True
    enforce_snap: bool = True
    enforce_curvature: bool = True
    local_max_iterations: int = 800
    local_shrink_sweeps: int = 4
    target_time: float | None = None


def _arrays(trajectory: AgentTrajectory) -> tuple[np.ndarray, np.ndarray]:
    points = np.array([[w.x, w.y, w.z] for w in trajectory.waypoints], dtype=float)
    times = np.array([w.t for w in trajectory.waypoints], dtype=float)
    return points, times


def _kinematics(points: np.ndarray, dt: np.ndarray) -> dict[str, np.ndarray]:
    deltas = np.diff(points, axis=0)
    velocities = deltas / dt[:, np.newaxis]
    accels = (
        np.diff(velocities, axis=0) / dt[1:, np.newaxis]
        if len(velocities) > 1
        else np.empty((0, 3))
    )
    jerks = (
        np.diff(accels, axis=0) / dt[2:, np.newaxis]
        if len(accels) > 1
        else np.empty((0, 3))
    )
    snaps = (
        np.diff(jerks, axis=0) / dt[3:, np.newaxis]
        if len(jerks) > 1
        else np.empty((0, 3))
    )
    return {
        "velocities": velocities,
        "accels": accels,
        "jerks": jerks,
        "snaps": snaps,
    }


def _constraint_values(
    points: np.ndarray,
    dt: np.ndarray,
    problem: ProblemSpec,
    config: RetimingConfig,
) -> np.ndarray:
    kin = _kinematics(points, dt)
    values: list[float] = []
    max_velocity = problem.max_velocity * config.limit_buffer
    max_acceleration = problem.max_acceleration * config.limit_buffer
    max_jerk = problem.max_jerk * config.limit_buffer
    max_snap = problem.max_snap * config.limit_buffer
    max_curvature = problem.max_curvature * config.limit_buffer

    velocities = kin["velocities"]
    values.extend(max_velocity**2 - np.sum(velocities * velocities, axis=1))

    accels = kin["accels"]
    if len(accels):
        values.extend(max_acceleration**2 - np.sum(accels * accels, axis=1))

    jerks = kin["jerks"]
    if config.enforce_jerk and len(jerks):
        values.extend(max_jerk**2 - np.sum(jerks * jerks, axis=1))

    snaps = kin["snaps"]
    if config.enforce_snap and len(snaps):
        values.extend(max_snap**2 - np.sum(snaps * snaps, axis=1))

    if config.enforce_curvature and len(accels):
        for accel_index, accel in enumerate(accels):
            velocity = 0.5 * (velocities[accel_index] + velocities[accel_index + 1])
            speed = float(np.linalg.norm(velocity))
            if speed <= 0.30:
                values.append(max_curvature)
                continue
            curvature = float(np.linalg.norm(np.cross(velocity, accel)) / speed**3)
            values.append(max_curvature - curvature)

    return np.array(values, dtype=float)


def _metric_summary(
    points: np.ndarray,
    dt: np.ndarray,
    problem: ProblemSpec,
    config: RetimingConfig,
) -> dict[str, float]:
    kin = _kinematics(points, dt)
    velocities = kin["velocities"]
    accels = kin["accels"]
    jerks = kin["jerks"]
    snaps = kin["snaps"]
    curvatures = []
    if len(accels):
        for accel_index, accel in enumerate(accels):
            velocity = 0.5 * (velocities[accel_index] + velocities[accel_index + 1])
            speed = float(np.linalg.norm(velocity))
            if speed <= 0.30:
                continue
            curvatures.append(float(np.linalg.norm(np.cross(velocity, accel)) / speed**3))
    constraints = _constraint_values(points, dt, problem, config)
    return {
        "max_v": float(np.max(np.linalg.norm(velocities, axis=1))) if len(velocities) else 0.0,
        "max_a": float(np.max(np.linalg.norm(accels, axis=1))) if len(accels) else 0.0,
        "max_jerk": float(np.max(np.linalg.norm(jerks, axis=1))) if len(jerks) else 0.0,
        "max_snap": float(np.max(np.linalg.norm(snaps, axis=1))) if len(snaps) else 0.0,
        "max_curvature": float(max(curvatures, default=0.0)),
        "min_constraint_margin": float(np.min(constraints)) if constraints.size else 0.0,
    }


def _requested_target_time(problem: ProblemSpec, config: RetimingConfig) -> float:
    if config.target_time is not None:
        return float(config.target_time)
    if problem.target_time is not None:
        return float(problem.target_time)
    raise ValueError("A requested target_time is required for retiming.")


def _estimated_limit_multipliers(required_time: float, target_time: float) -> dict[str, float]:
    if target_time <= 1.0e-9:
        scale = float("inf")
    else:
        scale = required_time / target_time
    return {
        "time_scale": float(scale),
        "max_velocity_multiplier": float(scale),
        "max_acceleration_multiplier": float(scale**2),
        "max_jerk_multiplier": float(scale**3),
        "max_snap_multiplier": float(scale**4),
        "note": (
            "Approximate multipliers assume the same geometry is uniformly time-compressed; "
            "curvature is primarily a geometry constraint."
        ),
    }


def _apply_target_window(
    dt: np.ndarray,
    problem: ProblemSpec,
    config: RetimingConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    target_time = _requested_target_time(problem, config)
    total_time = float(np.sum(dt))
    info: dict[str, Any] = {
        "requested_target_time": target_time,
        "target_time": target_time,
        "fastest_feasible_time": total_time,
        "feasible_by_requested_time": total_time <= target_time + config.feasibility_tol,
        "required_mission_time": total_time if total_time > target_time else target_time,
        "required_time_increase": max(0.0, total_time - target_time),
        "estimated_required_limit_multipliers": (
            _estimated_limit_multipliers(total_time, target_time)
            if total_time > target_time
            else None
        ),
    }
    if total_time > target_time + config.feasibility_tol:
        return dt, info

    scale = target_time / max(total_time, 1.0e-9)
    stretched = dt * scale
    info["time_after_target_stretch"] = float(np.sum(stretched))
    return stretched, info


def _initial_bounds(
    points: np.ndarray,
    times: np.ndarray,
    problem: ProblemSpec,
    config: RetimingConfig,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    original_dt = np.diff(times)
    if np.any(original_dt <= 0):
        raise ValueError("Trajectory timestamps must be strictly increasing before retiming.")

    deltas = np.diff(points, axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    lower = np.full_like(original_dt, config.min_segment_dt, dtype=float)
    moving = distances > config.moving_epsilon
    lower[moving] = np.maximum(
        lower[moving],
        distances[moving]
        / (problem.max_velocity * config.limit_buffer)
        * (1.0 + config.velocity_margin),
    )
    if config.preserve_holds:
        lower[~moving] = original_dt[~moving]

    upper = np.maximum.reduce(
        [
            original_dt * config.max_segment_scale,
            lower * config.max_segment_scale,
            lower + 0.25,
        ]
    )
    if config.preserve_holds:
        upper[~moving] = lower[~moving]

    x0 = np.clip(original_dt, lower, upper)
    return x0, [(float(lo), float(hi)) for lo, hi in zip(lower, upper)]


def _inflate_until_feasible(
    x0: np.ndarray,
    bounds: list[tuple[float, float]],
    points: np.ndarray,
    problem: ProblemSpec,
    config: RetimingConfig,
) -> np.ndarray:
    dt = x0.copy()
    lower = np.array([item[0] for item in bounds], dtype=float)
    upper = np.array([item[1] for item in bounds], dtype=float)
    fixed = np.isclose(lower, upper)
    for _ in range(40):
        constraints = _constraint_values(points, dt, problem, config)
        if not constraints.size or float(np.min(constraints)) >= -config.feasibility_tol:
            return dt
        candidate = dt.copy()
        candidate[~fixed] = np.minimum(upper[~fixed], candidate[~fixed] * 1.10 + 0.01)
        if np.allclose(candidate, dt):
            return dt
        dt = candidate
    return dt


def _time_scaled_durations(
    original_dt: np.ndarray,
    lower_bounds: np.ndarray,
    moving_mask: np.ndarray,
    scale: float,
) -> np.ndarray:
    dt = original_dt.copy()
    dt[moving_mask] = np.maximum(
        lower_bounds[moving_mask],
        original_dt[moving_mask] * scale,
    )
    return dt


def _uniform_scale_optimize(
    points: np.ndarray,
    times: np.ndarray,
    problem: ProblemSpec,
    config: RetimingConfig,
) -> tuple[np.ndarray, bool, str]:
    original_dt = np.diff(times)
    x0, bounds = _initial_bounds(points, times, problem, config)
    lower_bounds = np.array([item[0] for item in bounds], dtype=float)
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    moving_mask = distances > config.moving_epsilon
    if not np.any(moving_mask):
        return original_dt, True, "No moving segments to retime."

    def feasible(scale: float) -> tuple[bool, np.ndarray]:
        dt = _time_scaled_durations(original_dt, lower_bounds, moving_mask, scale)
        values = _constraint_values(points, dt, problem, config)
        ok = not values.size or float(np.min(values)) >= -config.feasibility_tol
        return ok, dt

    lower_scale = float(
        np.max(lower_bounds[moving_mask] / np.maximum(original_dt[moving_mask], 1.0e-9))
    )
    lower_scale = max(0.0, min(lower_scale, 1.0))
    high = 1.0
    high_ok, high_dt = feasible(high)
    while not high_ok and high < config.max_segment_scale:
        high *= 1.25
        high_ok, high_dt = feasible(high)
    if not high_ok:
        inflated = _inflate_until_feasible(x0, bounds, points, problem, config)
        inflated_values = _constraint_values(points, inflated, problem, config)
        inflated_ok = (
            not inflated_values.size
            or float(np.min(inflated_values)) >= -config.feasibility_tol
        )
        return inflated, inflated_ok, "Uniform retiming could not find a feasible scale."

    low = lower_scale
    low_ok, low_dt = feasible(low)
    if low_ok:
        return low_dt, True, f"Uniform retiming scale={low:.4f}."

    best_dt = high_dt
    for _ in range(80):
        mid = 0.5 * (low + high)
        mid_ok, mid_dt = feasible(mid)
        if mid_ok:
            high = mid
            best_dt = mid_dt
        else:
            low = mid
    return best_dt, True, f"Uniform retiming scale={high:.4f}."


def _inflate_indices(
    dt: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    fixed: np.ndarray,
    indices: range,
    factor: float,
) -> bool:
    changed = False
    factor = float(np.clip(factor, 1.001, 1.35))
    for index in indices:
        if index < 0 or index >= len(dt) or fixed[index]:
            continue
        old_value = dt[index]
        new_value = min(upper[index], max(old_value * factor, old_value + 0.002))
        new_value = max(new_value, lower[index])
        if new_value > old_value + 1.0e-10:
            dt[index] = new_value
            changed = True
    return changed


def _local_inflate_once(
    points: np.ndarray,
    dt: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    fixed: np.ndarray,
    problem: ProblemSpec,
    config: RetimingConfig,
) -> bool:
    kin = _kinematics(points, dt)
    changed = False
    max_velocity = problem.max_velocity * config.limit_buffer
    max_acceleration = problem.max_acceleration * config.limit_buffer
    max_jerk = problem.max_jerk * config.limit_buffer
    max_snap = problem.max_snap * config.limit_buffer
    max_curvature = problem.max_curvature * config.limit_buffer

    velocities = kin["velocities"]
    for index, velocity_norm in enumerate(np.linalg.norm(velocities, axis=1)):
        if velocity_norm > max_velocity:
            changed |= _inflate_indices(
                dt,
                lower,
                upper,
                fixed,
                range(index, index + 1),
                velocity_norm / max_velocity * 1.01,
            )

    accels = kin["accels"]
    for index, accel_norm in enumerate(np.linalg.norm(accels, axis=1)):
        if accel_norm > max_acceleration:
            changed |= _inflate_indices(
                dt,
                lower,
                upper,
                fixed,
                range(index, index + 2),
                np.sqrt(accel_norm / max_acceleration) * 1.01,
            )

    jerks = kin["jerks"]
    if config.enforce_jerk:
        for index, jerk_norm in enumerate(np.linalg.norm(jerks, axis=1)):
            if jerk_norm > max_jerk:
                changed |= _inflate_indices(
                    dt,
                    lower,
                    upper,
                    fixed,
                    range(index, index + 3),
                    (jerk_norm / max_jerk) ** (1.0 / 3.0) * 1.01,
                )

    snaps = kin["snaps"]
    if config.enforce_snap:
        for index, snap_norm in enumerate(np.linalg.norm(snaps, axis=1)):
            if snap_norm > max_snap:
                changed |= _inflate_indices(
                    dt,
                    lower,
                    upper,
                    fixed,
                    range(index, index + 4),
                    (snap_norm / max_snap) ** 0.25 * 1.01,
                )

    if config.enforce_curvature and len(accels):
        for index, accel in enumerate(accels):
            velocity = 0.5 * (velocities[index] + velocities[index + 1])
            speed = float(np.linalg.norm(velocity))
            if speed <= 0.30:
                continue
            curvature = float(np.linalg.norm(np.cross(velocity, accel)) / speed**3)
            if curvature > max_curvature:
                for segment_index in range(index, index + 2):
                    if segment_index >= len(velocities):
                        continue
                    segment_speed = float(np.linalg.norm(velocities[segment_index]))
                    if segment_speed <= 0.29:
                        continue
                    changed |= _inflate_indices(
                        dt,
                        lower,
                        upper,
                        fixed,
                        range(segment_index, segment_index + 1),
                        segment_speed / 0.29 * 1.01,
                    )
    return changed


def _local_scale_optimize(
    points: np.ndarray,
    times: np.ndarray,
    problem: ProblemSpec,
    config: RetimingConfig,
) -> tuple[np.ndarray, bool, str]:
    _, bounds = _initial_bounds(points, times, problem, config)
    lower = np.array([item[0] for item in bounds], dtype=float)
    upper = np.array([item[1] for item in bounds], dtype=float)
    fixed = np.isclose(lower, upper)
    dt = lower.copy()

    for _ in range(config.local_max_iterations):
        values = _constraint_values(points, dt, problem, config)
        if not values.size or float(np.min(values)) >= -config.feasibility_tol:
            break
        changed = _local_inflate_once(points, dt, lower, upper, fixed, problem, config)
        if not changed:
            break

    for _ in range(config.local_shrink_sweeps):
        any_change = False
        for index in range(len(dt)):
            if fixed[index] or dt[index] <= lower[index] + 1.0e-9:
                continue
            candidate = dt.copy()
            candidate[index] = max(lower[index], lower[index] + 0.90 * (dt[index] - lower[index]))
            values = _constraint_values(points, candidate, problem, config)
            if not values.size or float(np.min(values)) >= -config.feasibility_tol:
                dt = candidate
                any_change = True
        if not any_change:
            break

    values = _constraint_values(points, dt, problem, config)
    feasible = not values.size or float(np.min(values)) >= -config.feasibility_tol
    reason = (
        "Local segment retiming satisfied all dynamic limits."
        if feasible
        else "Local segment retiming hit bounds before satisfying all dynamic limits."
    )
    return dt, bool(feasible), reason


def retime_agent_trajectory(
    problem: ProblemSpec,
    trajectory: AgentTrajectory,
    config: RetimingConfig | None = None,
) -> tuple[AgentTrajectory, dict[str, Any]]:
    config = config or RetimingConfig()
    points, times = _arrays(trajectory)
    if len(points) < 2:
        return trajectory, {
            "success": False,
            "reason": "Trajectory has fewer than two waypoints.",
        }

    before_time = float(times[-1] - times[0])
    if config.method == "local":
        chosen_dt, success, reason = _local_scale_optimize(points, times, problem, config)
        chosen_dt, target_info = _apply_target_window(chosen_dt, problem, config)
        success = success and bool(target_info["feasible_by_requested_time"])
        if not target_info["feasible_by_requested_time"]:
            reason = (
                f"{reason} Fastest feasible time {target_info['fastest_feasible_time']:.2f}s "
                f"exceeds requested target time {target_info['requested_target_time']:.2f}s."
            )
        new_times = np.concatenate([[0.0], np.cumsum(chosen_dt)])
        retimed = AgentTrajectory(
            agent_id=trajectory.agent_id,
            waypoints=[
                Waypoint(x=float(point[0]), y=float(point[1]), z=float(point[2]), t=float(t))
                for point, t in zip(points, new_times)
            ],
        )
        summary = _metric_summary(points, chosen_dt, problem, config)
        summary.update(
            {
                "success": success,
                "reason": reason,
                "method": config.method,
                "time_before": before_time,
                "time_after": float(new_times[-1]),
                "time_saved": float(before_time - new_times[-1]),
                "segment_count": int(len(chosen_dt)),
                **target_info,
            }
        )
        return retimed, summary

    if config.method == "uniform":
        chosen_dt, success, reason = _uniform_scale_optimize(points, times, problem, config)
        chosen_dt, target_info = _apply_target_window(chosen_dt, problem, config)
        success = success and bool(target_info["feasible_by_requested_time"])
        if not target_info["feasible_by_requested_time"]:
            reason = (
                f"{reason} Fastest feasible time {target_info['fastest_feasible_time']:.2f}s "
                f"exceeds requested target time {target_info['requested_target_time']:.2f}s."
            )
        new_times = np.concatenate([[0.0], np.cumsum(chosen_dt)])
        retimed = AgentTrajectory(
            agent_id=trajectory.agent_id,
            waypoints=[
                Waypoint(x=float(point[0]), y=float(point[1]), z=float(point[2]), t=float(t))
                for point, t in zip(points, new_times)
            ],
        )
        summary = _metric_summary(points, chosen_dt, problem, config)
        summary.update(
            {
                "success": success,
                "reason": reason,
                "method": config.method,
                "time_before": before_time,
                "time_after": float(new_times[-1]),
                "time_saved": float(before_time - new_times[-1]),
                "segment_count": int(len(chosen_dt)),
                **target_info,
            }
        )
        return retimed, summary

    x0, bounds = _initial_bounds(points, times, problem, config)
    x0 = _inflate_until_feasible(x0, bounds, points, problem, config)

    def objective(dt: np.ndarray) -> float:
        return float(np.sum(dt))

    def constraints(dt: np.ndarray) -> np.ndarray:
        return _constraint_values(points, dt, problem, config)

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": constraints}],
        options={"maxiter": config.solver_max_iter, "ftol": config.solver_ftol, "disp": False},
    )
    candidate_dt = np.asarray(result.x if result.x is not None else x0, dtype=float)
    candidate_constraints = _constraint_values(points, candidate_dt, problem, config)
    feasible = (
        not candidate_constraints.size
        or float(np.min(candidate_constraints)) >= -config.feasibility_tol
    )

    original_constraints = _constraint_values(points, np.diff(times), problem, config)
    original_feasible = (
        not original_constraints.size
        or float(np.min(original_constraints)) >= -config.feasibility_tol
    )
    original_time = float(np.sum(np.diff(times)))
    candidate_time = float(np.sum(candidate_dt))
    if not feasible or candidate_time > original_time:
        if original_feasible:
            chosen_dt = np.diff(times)
            success = False
            reason = f"Retiming optimizer kept original timestamps: {result.message}"
        else:
            chosen_dt = candidate_dt
            success = bool(feasible)
            reason = str(result.message)
    else:
        chosen_dt = candidate_dt
        success = bool(result.success or feasible)
        reason = str(result.message)

    chosen_dt, target_info = _apply_target_window(chosen_dt, problem, config)
    success = success and bool(target_info["feasible_by_requested_time"])
    if not target_info["feasible_by_requested_time"]:
        reason = (
            f"{reason} Fastest feasible time {target_info['fastest_feasible_time']:.2f}s "
            f"exceeds requested target time {target_info['requested_target_time']:.2f}s."
        )

    new_times = np.concatenate([[0.0], np.cumsum(chosen_dt)])
    retimed = AgentTrajectory(
        agent_id=trajectory.agent_id,
        waypoints=[
            Waypoint(x=float(point[0]), y=float(point[1]), z=float(point[2]), t=float(t))
            for point, t in zip(points, new_times)
        ],
    )
    summary = _metric_summary(points, chosen_dt, problem, config)
    summary.update(
        {
            "success": success,
            "reason": reason,
            "method": config.method,
            "time_before": before_time,
            "time_after": float(new_times[-1]),
            "time_saved": float(before_time - new_times[-1]),
            "segment_count": int(len(chosen_dt)),
            **target_info,
        }
    )
    return retimed, summary


def retime_solution(
    problem_id: str,
    solution: MissionSolution,
    config: RetimingConfig | None = None,
) -> tuple[MissionSolution, dict[str, Any]]:
    config = config or RetimingConfig()
    problem = get_problem(problem_id)
    retimed_trajectories: list[AgentTrajectory] = []
    agent_metrics: dict[str, dict[str, Any]] = {}
    for trajectory in solution.agent_trajectories:
        retimed, metrics = retime_agent_trajectory(problem, trajectory, config)
        retimed_trajectories.append(retimed)
        agent_metrics[trajectory.agent_id] = metrics

    before_total = max(
        (metrics["time_before"] for metrics in agent_metrics.values()),
        default=0.0,
    )
    after_total = max(
        (metrics["time_after"] for metrics in agent_metrics.values()),
        default=0.0,
    )
    target_time = _requested_target_time(problem, config)
    feasible_by_requested_time = (
        after_total <= target_time + config.feasibility_tol
        and all(metrics.get("success", False) for metrics in agent_metrics.values())
    )
    required_mission_time = max(
        (metrics.get("required_mission_time", after_total) for metrics in agent_metrics.values()),
        default=after_total,
    )
    return MissionSolution(agent_trajectories=retimed_trajectories), {
        "enabled": True,
        "method": config.method,
        "requested_target_time": target_time,
        "target_time": target_time,
        "feasible_by_requested_time": bool(feasible_by_requested_time),
        "required_mission_time": float(required_mission_time),
        "required_time_increase": float(max(0.0, required_mission_time - target_time)),
        "time_before": float(before_total),
        "time_after": float(after_total),
        "time_saved": float(before_total - after_total),
        "agents": agent_metrics,
    }
