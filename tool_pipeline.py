import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import instructor
import numpy as np
from openai import OpenAI

from full_motor_nmpc import FullMotorNMPCConfig, execute_with_full_motor_nmpc
from problems import build_problem_prompt, get_problem
from nmpc_cbf import execute_with_nmpc_cbf
from schemas import AgentSpec, AgentTrajectory, KeyWaypoint, MissionSolution, ProblemSpec, ToolPlanResponse, Waypoint
from trajectory_retiming import RetimingConfig, retime_solution
from verifier import verify
from visualization import plot_payload


client = None


def _execution_metrics_payload(metrics) -> dict:
    return {
        "max_tracking_error": metrics.max_tracking_error,
        "mean_tracking_error": metrics.mean_tracking_error,
        "cbf_adjustment_count": metrics.cbf_adjustment_count,
        "max_cbf_adjustment": metrics.max_cbf_adjustment,
        "max_roll_deg": metrics.max_roll_deg,
        "max_pitch_deg": metrics.max_pitch_deg,
        "max_roll_rate_deg_s": metrics.max_roll_rate_deg_s,
        "max_pitch_rate_deg_s": metrics.max_pitch_rate_deg_s,
        "min_thrust_accel": metrics.min_thrust_accel,
        "max_thrust_accel": metrics.max_thrust_accel,
        "max_thrust_rate_accel_s": metrics.max_thrust_rate_accel_s,
        "max_jerk": metrics.max_jerk,
        "max_angular_rate_deg_s": metrics.max_angular_rate_deg_s,
        "max_motor_thrust_rate_accel_s": metrics.max_motor_thrust_rate_accel_s,
    }


def _full_motor_metrics_payload(metrics) -> dict:
    return {
        "max_tracking_error": metrics.max_tracking_error,
        "mean_tracking_error": metrics.mean_tracking_error,
        "max_roll_deg": metrics.max_roll_deg,
        "max_pitch_deg": metrics.max_pitch_deg,
        "max_angular_rate_deg_s": metrics.max_angular_rate_deg_s,
        "min_motor_thrust_accel": metrics.min_motor_thrust_accel,
        "max_motor_thrust_accel": metrics.max_motor_thrust_accel,
        "mean_solve_time_s": metrics.mean_solve_time_s,
        "max_solve_time_s": metrics.max_solve_time_s,
        "solve_count": metrics.solve_count,
        "min_obstacle_clearance": metrics.min_obstacle_clearance,
        "min_agent_clearance": metrics.min_agent_clearance,
    }


def _full_motor_config_for_problem(problem: ProblemSpec) -> FullMotorNMPCConfig:
    config = FullMotorNMPCConfig()
    return replace(
        config,
        max_velocity=problem.max_velocity,
        max_acceleration=problem.max_acceleration,
        max_jerk=problem.max_jerk,
        max_angular_rate_deg_s=problem.max_attitude_rate_deg_s,
    )


def _execute_reference_solution(
    problem_id: str,
    reference_solution: MissionSolution,
    nmpc_backend: str,
    full_apply_steps: int,
    full_max_duration: float | None,
) -> tuple[MissionSolution, dict, str]:
    if nmpc_backend == "cascade":
        executed_solution, metrics = execute_with_nmpc_cbf(problem_id, reference_solution)
        return executed_solution, _execution_metrics_payload(metrics), "nmpc_cbf"
    if nmpc_backend == "full":
        problem = get_problem(problem_id)
        config = _full_motor_config_for_problem(problem)
        executed_solution, metrics = execute_with_full_motor_nmpc(
            problem_id,
            reference_solution,
            config=config,
            apply_steps=full_apply_steps,
            max_duration=full_max_duration,
        )
        return executed_solution, _full_motor_metrics_payload(metrics), "full_motor_nmpc_cbf"
    raise ValueError(f"Unsupported nmpc_backend={nmpc_backend!r}.")


def _retiming_issue_text(retiming_metrics: dict | None) -> str | None:
    if not retiming_metrics:
        return None
    issues: list[str] = []
    if not retiming_metrics.get("feasible_by_requested_time", True):
        target_time = retiming_metrics.get("requested_target_time") or retiming_metrics.get("target_time")
        required = retiming_metrics.get("required_mission_time")
        increase = retiming_metrics.get("required_time_increase")
        issues.append(
            "retiming target time infeasible"
            + (
                f": requires at least {required:.2f}s, "
                f"{increase:.2f}s above requested target {target_time:.2f}s"
                if required is not None and increase is not None and target_time is not None
                else ""
            )
        )
    for agent_id, metrics in retiming_metrics.get("agents", {}).items():
        if not metrics.get("success", True):
            issues.append(f"{agent_id}: retiming failed ({metrics.get('reason', 'no reason')})")
    return " Retiming issue: " + " ".join(issues) if issues else None


def _apply_retiming_issues(result: dict, retiming_metrics: dict | None) -> None:
    issue_text = _retiming_issue_text(retiming_metrics)
    if not issue_text:
        return
    result["pass"] = False
    reason = result.get("reason", "")
    if reason == "Success! Multi-agent mission feasible.":
        result["reason"] = issue_text.strip()
    else:
        result["reason"] = f"{reason}{issue_text}"


def _make_retiming_config(
    target_time: float | None = None,
) -> RetimingConfig:
    return RetimingConfig(target_time=target_time)


def _print_nmpc_time_feasibility(estimate: dict | None) -> None:
    if not estimate:
        return
    print("NMPC time feasibility:")
    print(f"  Requested: {estimate['requested_time']:.2f}s")
    print(f"  Required: {estimate['required_mission_time']}")
    print(f"  Increase: {estimate['required_time_increase']}")
    if estimate.get("reason"):
        print(f"  Reason: {estimate['reason']}")


def _verify_for_target_time(problem_id: str, solution: MissionSolution, target_time: float | None) -> dict:
    return verify(problem_id, solution, target_time=target_time)


def _run_nmpc_candidate(
    problem_id: str,
    geometric_solution: MissionSolution,
    candidate_time: float,
    nmpc_backend: str,
    full_apply_steps: int,
    full_max_duration: float | None,
) -> tuple[bool, dict]:
    reference_solution, retiming_metrics = retime_solution(
        problem_id,
        geometric_solution,
        _make_retiming_config(target_time=candidate_time),
    )
    retiming_issue = _retiming_issue_text(retiming_metrics)
    if retiming_issue:
        return False, {
            "candidate_time": candidate_time,
            "stage": "retiming",
            "retiming": retiming_metrics,
            "reason": retiming_issue.strip(),
        }
    executed, metrics, backend = _execute_reference_solution(
        problem_id,
        reference_solution,
        nmpc_backend,
        full_apply_steps,
        full_max_duration,
    )
    result = _verify_for_target_time(problem_id, executed, candidate_time)
    return bool(result["pass"]), {
        "candidate_time": candidate_time,
        "stage": "execution",
        "backend": backend,
        "verification": result,
        "retiming": retiming_metrics,
        "execution_metrics": metrics,
    }


def estimate_required_nmpc_time(
    problem_id: str,
    geometric_solution: MissionSolution,
    requested_time: float,
    nmpc_backend: str = "cascade",
    full_apply_steps: int = 4,
    full_max_duration: float | None = None,
    coarse_step: float = 5.0,
    binary_iterations: int = 5,
) -> dict:
    problem = get_problem(problem_id)
    upper_limit = requested_time + 60.0
    candidates: list[dict] = []

    ok, payload = _run_nmpc_candidate(
        problem_id,
        geometric_solution,
        requested_time,
        nmpc_backend,
        full_apply_steps,
        full_max_duration,
    )
    candidates.append(payload)
    if ok:
        return {
            "requested_time": requested_time,
            "feasible_by_requested_time": True,
            "required_mission_time": requested_time,
            "required_time_increase": 0.0,
            "candidates": candidates,
        }

    low = requested_time
    high = requested_time
    while high < upper_limit:
        high += coarse_step
        ok, payload = _run_nmpc_candidate(
            problem_id,
            geometric_solution,
            high,
            nmpc_backend,
            full_apply_steps,
            full_max_duration,
        )
        candidates.append(payload)
        if ok:
            break
        low = high

    if not ok:
        return {
            "requested_time": requested_time,
            "feasible_by_requested_time": False,
            "required_mission_time": None,
            "required_time_increase": None,
            "search_upper_limit": upper_limit,
            "reason": "No NMPC-feasible execution found within search upper limit.",
            "candidates": candidates,
        }

    for _ in range(binary_iterations):
        mid = 0.5 * (low + high)
        ok_mid, payload = _run_nmpc_candidate(
            problem_id,
            geometric_solution,
            mid,
            nmpc_backend,
            full_apply_steps,
            full_max_duration,
        )
        candidates.append(payload)
        if ok_mid:
            high = mid
        else:
            low = mid

    return {
        "requested_time": requested_time,
        "feasible_by_requested_time": False,
        "required_mission_time": high,
        "required_time_increase": max(0.0, high - requested_time),
        "candidates": candidates,
    }


def get_client():
    global client
    if client is None:
        client = instructor.from_openai(OpenAI())
    return client


def _distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    return float(np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float)))


def _as_xyz(point: KeyWaypoint | Sequence[float]) -> list[float]:
    if isinstance(point, KeyWaypoint):
        return [point.x, point.y, point.z]
    return [float(point[0]), float(point[1]), float(point[2])]


def _same_position(p1: Sequence[float], p2: Sequence[float]) -> bool:
    return _distance(p1, p2) <= 1e-6


def _normalize_key_waypoints(
    problem_id: str,
    agent_id: str,
    key_waypoints: list[KeyWaypoint],
) -> list[KeyWaypoint]:
    problem = get_problem(problem_id)
    agent = next(agent for agent in problem.agents if agent.agent_id == agent_id)
    start = problem.charging_stations[agent.start_station_index]
    goal = agent.final_goal

    normalized = list(key_waypoints)
    if not normalized or not _same_position(_as_xyz(normalized[0]), start):
        normalized.insert(0, KeyWaypoint(x=start[0], y=start[1], z=start[2], note="start"))
    if not _same_position(_as_xyz(normalized[-1]), goal):
        normalized.append(
            KeyWaypoint(x=goal[0], y=goal[1], z=goal[2], hold_seconds=1.0, note="goal")
        )
    return normalized


def _curve_point(
    start: np.ndarray,
    end: np.ndarray,
    start_tangent: np.ndarray,
    end_tangent: np.ndarray,
    u: float,
) -> np.ndarray:
    h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
    h10 = u**3 - 2.0 * u**2 + u
    h01 = -2.0 * u**3 + 3.0 * u**2
    h11 = u**3 - u**2
    return h00 * start + h10 * start_tangent + h01 * end + h11 * end_tangent


def _minimum_jerk_point(start: np.ndarray, end: np.ndarray, u: float) -> np.ndarray:
    blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    return start + blend * (end - start)


def _curve_length(
    start: Sequence[float],
    end: Sequence[float],
    start_tangent: Sequence[float],
    end_tangent: Sequence[float],
    samples: int = 30,
) -> float:
    start_arr = np.array(start, dtype=float)
    end_arr = np.array(end, dtype=float)
    start_tangent_arr = np.array(start_tangent, dtype=float)
    end_tangent_arr = np.array(end_tangent, dtype=float)
    points = [
        _curve_point(start_arr, end_arr, start_tangent_arr, end_tangent_arr, u)
        for u in np.linspace(0.0, 1.0, samples)
    ]
    return float(
        sum(np.linalg.norm(points[i] - points[i - 1]) for i in range(1, len(points)))
    )


def _minimum_jerk_length(
    start: Sequence[float],
    end: Sequence[float],
    samples: int = 30,
) -> float:
    start_arr = np.array(start, dtype=float)
    end_arr = np.array(end, dtype=float)
    points = [
        _minimum_jerk_point(start_arr, end_arr, u)
        for u in np.linspace(0.0, 1.0, samples)
    ]
    return float(
        sum(np.linalg.norm(points[i] - points[i - 1]) for i in range(1, len(points)))
    )


def _sample_curve_by_arclength(
    start: Sequence[float],
    end: Sequence[float],
    start_tangent: Sequence[float],
    end_tangent: Sequence[float],
    max_spacing: float,
    samples: int = 80,
) -> list[tuple[np.ndarray, float]]:
    use_minimum_jerk = _distance(start_tangent, [0.0, 0.0, 0.0]) <= 1e-9 or _distance(
        end_tangent, [0.0, 0.0, 0.0]
    ) <= 1e-9
    start_arr = np.array(start, dtype=float)
    end_arr = np.array(end, dtype=float)
    start_tangent_arr = np.array(start_tangent, dtype=float)
    end_tangent_arr = np.array(end_tangent, dtype=float)
    dense_points = []
    for u in np.linspace(0.0, 1.0, samples + 1):
        point = (
            _minimum_jerk_point(start_arr, end_arr, u)
            if use_minimum_jerk
            else _curve_point(start_arr, end_arr, start_tangent_arr, end_tangent_arr, u)
        )
        dense_points.append(point)
    segment_lengths = [
        float(np.linalg.norm(dense_points[index] - dense_points[index - 1]))
        for index in range(1, len(dense_points))
    ]
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(cumulative[-1])
    if total_length <= 1e-9:
        return []

    sample_count = max(1, int(np.ceil(total_length / max_spacing)))
    distances = np.linspace(total_length / sample_count, total_length, sample_count)
    sampled: list[tuple[np.ndarray, float]] = []
    for distance_along in distances:
        dense_index = int(np.searchsorted(cumulative, distance_along, side="right") - 1)
        dense_index = min(dense_index, len(dense_points) - 2)
        local_length = cumulative[dense_index + 1] - cumulative[dense_index]
        if local_length <= 1e-9:
            point = dense_points[dense_index + 1]
        else:
            alpha = (distance_along - cumulative[dense_index]) / local_length
            point = dense_points[dense_index] + alpha * (
                dense_points[dense_index + 1] - dense_points[dense_index]
            )
        sampled.append((point, float(distance_along)))
    return sampled


def _append_smooth_segment(
    waypoints: list[Waypoint],
    start: Sequence[float],
    end: Sequence[float],
    start_tangent: Sequence[float],
    end_tangent: Sequence[float],
    current_time: float,
    max_acceleration: float,
    max_velocity: float,
    max_spacing: float = 0.14,
) -> float:
    sampled = _sample_curve_by_arclength(
        start,
        end,
        start_tangent,
        end_tangent,
        max_spacing=max_spacing,
    )
    if not sampled:
        return current_time

    previous_distance = 0.0
    for point, distance_along in sampled:
        step_length = max(0.0, distance_along - previous_distance)
        velocity_limited = step_length / max(max_velocity, 1e-9)
        acceleration_limited = np.sqrt(
            2.0 * step_length / max(max_acceleration, 1e-9)
        )
        current_time += max(velocity_limited, acceleration_limited, 0.02)
        waypoints.append(
            Waypoint(x=float(point[0]), y=float(point[1]), z=float(point[2]), t=float(current_time))
        )
        previous_distance = distance_along
    return current_time


def _point_tangent(
    points: list[list[float]],
    hold_seconds: list[float],
    index: int,
    scale: float = 0.35,
) -> np.ndarray:
    if index == 0 or index == len(points) - 1 or hold_seconds[index] > 0.0:
        return np.zeros(3)
    previous_point = np.array(points[index - 1], dtype=float)
    next_point = np.array(points[index + 1], dtype=float)
    return scale * (next_point - previous_point)


def _append_hold(
    waypoints: list[Waypoint],
    position: Sequence[float],
    current_time: float,
    hold_seconds: float,
) -> float:
    if hold_seconds <= 0.0:
        return current_time
    current_time += hold_seconds
    waypoints.append(
        Waypoint(x=position[0], y=position[1], z=position[2], t=current_time)
    )
    return current_time


def _horizontal_direction(
    origin: Sequence[float],
    candidate: Sequence[float],
    fallback: Sequence[float],
) -> np.ndarray:
    delta = np.array(candidate[:2], dtype=float) - np.array(origin[:2], dtype=float)
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-9:
        delta = np.array(fallback[:2], dtype=float)
        norm = float(np.linalg.norm(delta))
    if norm <= 1e-9:
        return np.array([1.0, 0.0], dtype=float)
    return delta / norm


def _reshape_vertical_visit_keys(
    problem: ProblemSpec,
    agent: AgentSpec,
    keys: list[KeyWaypoint],
) -> list[KeyWaypoint]:
    reshaped = [key.model_copy() for key in keys]
    offset = min(0.40, max(0.20, 1.4 * problem.visit_radius))

    for index in range(1, len(reshaped) - 1):
        key = reshaped[index]
        previous_key = reshaped[index - 1]
        next_key = reshaped[index + 1]
        key_xyz = np.array(_as_xyz(key), dtype=float)
        prev_xyz = np.array(_as_xyz(previous_key), dtype=float)
        next_xyz = np.array(_as_xyz(next_key), dtype=float)
        vertical_stack = (
            key.hold_seconds > 0.0
            and np.linalg.norm(prev_xyz[:2] - key_xyz[:2]) <= 1e-6
            and np.linalg.norm(next_xyz[:2] - key_xyz[:2]) <= 1e-6
            and prev_xyz[2] - key_xyz[2] > 0.25
            and next_xyz[2] - key_xyz[2] > 0.25
        )
        if not vertical_stack:
            continue

        earlier = _as_xyz(reshaped[index - 2]) if index >= 2 else _as_xyz(previous_key)
        later = _as_xyz(reshaped[index + 2]) if index + 2 < len(reshaped) else _as_xyz(next_key)
        incoming = _horizontal_direction(key_xyz, earlier, [-1.0, 0.0])
        outgoing = _horizontal_direction(key_xyz, later, [1.0, 0.0])

        previous_key.x = float(key_xyz[0] + offset * incoming[0])
        previous_key.y = float(key_xyz[1] + offset * incoming[1])
        next_key.x = float(key_xyz[0] + offset * outgoing[0])
        next_key.y = float(key_xyz[1] + offset * outgoing[1])

    return reshaped


def smooth_key_waypoints(
    problem_id: str,
    agent_key_waypoints: dict[str, list[KeyWaypoint]],
) -> MissionSolution:
    problem = get_problem(problem_id)
    agent_trajectories: list[AgentTrajectory] = []

    for agent in problem.agents:
        keys = _normalize_key_waypoints(
            problem_id,
            agent.agent_id,
            agent_key_waypoints.get(agent.agent_id, []),
        )
        keys = _reshape_vertical_visit_keys(problem, agent, keys)
        points = [_as_xyz(key) for key in keys]
        hold_seconds = [key.hold_seconds for key in keys]
        first = points[0]
        waypoints = [Waypoint(x=first[0], y=first[1], z=first[2], t=0.0)]
        current_position = first
        current_time = _append_hold(waypoints, current_position, 0.0, keys[0].hold_seconds)

        for index, key in enumerate(keys[1:], start=1):
            destination = points[index]
            current_time = _append_smooth_segment(
                waypoints,
                current_position,
                destination,
                _point_tangent(points, hold_seconds, index - 1),
                _point_tangent(points, hold_seconds, index),
                current_time,
                problem.max_acceleration,
                problem.max_velocity,
            )
            current_position = destination
            current_time = _append_hold(
                waypoints, current_position, current_time, key.hold_seconds
            )

        agent_trajectories.append(
            AgentTrajectory(agent_id=agent.agent_id, waypoints=waypoints)
        )

    return MissionSolution(agent_trajectories=agent_trajectories)


def run_tool_pipeline(
    problem_id: str,
    feedback: list[str] | None = None,
    use_nmpc_cbf: bool = True,
    use_retiming: bool = True,
    target_time: float | None = None,
    nmpc_backend: str = "cascade",
    full_apply_steps: int = 4,
    full_max_duration: float | None = None,
    estimate_required_time_on_failure: bool = False,
):
    problem = get_problem(problem_id)
    requested_time = target_time if target_time is not None else problem.target_time
    messages = [
        {
            "role": "system",
            "content": (
                "You are the high-level multi-agent Crazyflie planner. You must choose "
                "the sparse key waypoints for each drone, including obstacle detours, "
                "charging station stops, target visits, and final goals. The helper tool "
                "only smooths the route geometry and optimizes timestamps after your "
                "waypoints; it does not choose the route for you. The timestamp optimizer "
                "targets the requested mission time, so do not output speed commands. "
                "If feedback reports acceleration, jerk, snap, or curvature violations, "
                "change the geometry with wider turns, more gradual altitude transitions, "
                "and fewer near-duplicate waypoints."
            ),
        },
        {"role": "user", "content": build_problem_prompt(problem_id)},
    ]
    for item in feedback or []:
        messages.append(
            {
                "role": "user",
                "content": f"Verifier feedback from the previous attempt: {item}",
            }
        )
    response = get_client().chat.completions.create(
        model="gpt-5.5",
        response_model=ToolPlanResponse,
        messages=messages,
    )
    requested_waypoints = {
        agent_plan.agent_id: agent_plan.key_waypoints for agent_plan in response.agent_plans
    }
    reference_solution = smooth_key_waypoints(
        problem_id=problem_id,
        agent_key_waypoints=requested_waypoints,
    )
    retiming_metrics = None
    if use_retiming:
        reference_solution, retiming_metrics = retime_solution(
            problem_id,
            reference_solution,
            _make_retiming_config(requested_time),
        )
    solution = reference_solution
    execution_metrics = None
    execution_backend = None
    retiming_has_issues = _retiming_issue_text(retiming_metrics) is not None
    if use_nmpc_cbf and not retiming_has_issues:
        solution, execution_metrics, execution_backend = _execute_reference_solution(
            problem_id,
            reference_solution,
            nmpc_backend,
            full_apply_steps,
            full_max_duration,
        )
    result = verify(
        problem_id,
        solution,
        target_time=requested_time,
    )
    if retiming_metrics is not None:
        result["details"]["retiming"] = retiming_metrics
        _apply_retiming_issues(result, retiming_metrics)
    if execution_metrics is not None:
        result["details"]["execution_backend"] = execution_backend
        result["details"][execution_backend] = execution_metrics
    if (
        estimate_required_time_on_failure
        and use_nmpc_cbf
        and requested_time is not None
        and not result["pass"]
    ):
        result["details"]["nmpc_time_feasibility"] = estimate_required_nmpc_time(
            problem_id,
            smooth_key_waypoints(
                problem_id=problem_id,
                agent_key_waypoints=requested_waypoints,
            ),
            requested_time,
            nmpc_backend=nmpc_backend,
            full_apply_steps=full_apply_steps,
            full_max_duration=full_max_duration,
        )
    return result, response.strategy, solution


def build_reference_solution(
    problem_id: str = "MA1",
    target_time: float | None = None,
) -> MissionSolution:
    problem = get_problem(problem_id)
    requested_time = target_time if target_time is not None else problem.target_time
    solution = smooth_key_waypoints(
        problem_id=problem_id,
        agent_key_waypoints=problem.recommended_key_waypoints,
    )
    retimed_solution, _ = retime_solution(
        problem_id,
        solution,
        _make_retiming_config(requested_time),
    )
    return retimed_solution


def build_geometric_reference_solution(problem_id: str = "MA1") -> MissionSolution:
    problem = get_problem(problem_id)
    return smooth_key_waypoints(
        problem_id=problem_id,
        agent_key_waypoints=problem.recommended_key_waypoints,
    )


def build_executed_reference_solution(
    problem_id: str = "MA1",
    use_retiming: bool = True,
    target_time: float | None = None,
    nmpc_backend: str = "cascade",
    full_apply_steps: int = 4,
    full_max_duration: float | None = None,
) -> tuple[MissionSolution, dict]:
    reference_solution = (
        build_reference_solution(problem_id, target_time)
        if use_retiming
        else build_geometric_reference_solution(problem_id)
    )
    executed_solution, metrics, execution_backend = _execute_reference_solution(
        problem_id,
        reference_solution,
        nmpc_backend,
        full_apply_steps,
        full_max_duration,
    )
    return executed_solution, {"backend": execution_backend, "metrics": metrics}


def save_reference_trajectory(
    output_path: str | Path = "reference_trajectory.json",
    problem_id: str = "MA1",
    use_retiming: bool = True,
    target_time: float | None = None,
) -> dict:
    problem = get_problem(problem_id)
    requested_time = target_time if target_time is not None else problem.target_time
    base_solution = build_geometric_reference_solution(problem_id)
    retiming_metrics = None
    if use_retiming:
        solution, retiming_metrics = retime_solution(
            problem_id,
            base_solution,
            _make_retiming_config(requested_time),
        )
    else:
        solution = base_solution
    result = verify(
        problem_id,
        solution,
        target_time=requested_time,
    )
    if retiming_metrics is not None:
        result["details"]["retiming"] = retiming_metrics
        _apply_retiming_issues(result, retiming_metrics)
    payload = {
        "problem_id": problem_id,
        "verification": result,
        "retiming": retiming_metrics,
        "agent_trajectories": [
            trajectory.model_dump() for trajectory in solution.agent_trajectories
        ],
    }
    path = Path(output_path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def save_executed_reference_trajectory(
    output_path: str | Path = "executed_trajectory.json",
    problem_id: str = "MA1",
    use_retiming: bool = True,
    target_time: float | None = None,
    nmpc_backend: str = "cascade",
    full_apply_steps: int = 4,
    full_max_duration: float | None = None,
) -> dict:
    solution, execution_metrics = build_executed_reference_solution(
        problem_id,
        use_retiming=use_retiming,
        target_time=target_time,
        nmpc_backend=nmpc_backend,
        full_apply_steps=full_apply_steps,
        full_max_duration=full_max_duration,
    )
    result = verify(
        problem_id,
        solution,
        target_time=target_time if target_time is not None else get_problem(problem_id).target_time,
    )
    payload = {
        "problem_id": problem_id,
        "verification": result,
        "execution_metrics": execution_metrics,
        "agent_trajectories": [
            trajectory.model_dump() for trajectory in solution.agent_trajectories
        ],
    }
    path = Path(output_path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def save_llm_trajectory(
    output_path: str | Path = "llm_executed_trajectory.json",
    problem_id: str = "MA1",
    feedback: list[str] | None = None,
    use_nmpc_cbf: bool = True,
    use_retiming: bool = True,
    target_time: float | None = None,
    nmpc_backend: str = "cascade",
    full_apply_steps: int = 4,
    full_max_duration: float | None = None,
) -> dict:
    result, strategy, solution = run_tool_pipeline(
        problem_id=problem_id,
        feedback=feedback,
        use_nmpc_cbf=use_nmpc_cbf,
        use_retiming=use_retiming,
        target_time=target_time,
        nmpc_backend=nmpc_backend,
        full_apply_steps=full_apply_steps,
        full_max_duration=full_max_duration,
    )
    payload = {
        "problem_id": problem_id,
        "strategy": strategy,
        "verification": result,
        "agent_trajectories": [
            trajectory.model_dump() for trajectory in solution.agent_trajectories
        ],
    }
    path = Path(output_path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_llm_refinement(
    problem_id: str = "MA1",
    max_refinement_turns: int = 2,
    output_path: str | Path = "llm_refined_trajectory.json",
    use_nmpc_cbf: bool = True,
    use_retiming: bool = True,
    target_time: float | None = None,
    nmpc_backend: str = "cascade",
    full_apply_steps: int = 4,
    full_max_duration: float | None = None,
    estimate_required_time_on_failure: bool = True,
    plot: bool = False,
) -> dict:
    feedback_history: list[str] = []
    turn_records: list[dict] = []
    final_payload: dict | None = None
    output = Path(output_path)

    for turn in range(max_refinement_turns + 1):
        result, strategy, solution = run_tool_pipeline(
            problem_id=problem_id,
            feedback=feedback_history,
            use_nmpc_cbf=use_nmpc_cbf,
            use_retiming=use_retiming,
            target_time=target_time,
            nmpc_backend=nmpc_backend,
            full_apply_steps=full_apply_steps,
            full_max_duration=full_max_duration,
            estimate_required_time_on_failure=(
                estimate_required_time_on_failure and turn == max_refinement_turns
            ),
        )
        turn_payload = {
            "problem_id": problem_id,
            "turn": turn + 1,
            "strategy": strategy,
            "feedback_history": list(feedback_history),
            "verification": result,
            "agent_trajectories": [
                trajectory.model_dump() for trajectory in solution.agent_trajectories
            ],
        }
        turn_path = output.with_name(f"{output.stem}_turn{turn + 1}{output.suffix}")
        turn_path.write_text(json.dumps(turn_payload, indent=2), encoding="utf-8")
        if plot:
            plot_payload(turn_path, turn_path.with_suffix(".png"), problem_id)
        turn_records.append(
            {
                "turn": turn + 1,
                "pass": result["pass"],
                "strategy": strategy,
                "reason": result["reason"],
                "json_path": str(turn_path),
                "plot_path": str(turn_path.with_suffix(".png")) if plot else None,
            }
        )
        status = "PASS" if result["pass"] else "FAIL"
        print(f"[Refine] Trial {turn + 1}/{max_refinement_turns + 1}: {status}")
        print(f"[Refine] Strategy: {strategy}")
        print(f"[Refine] Reason: {result['reason']}")
        print(f"[Refine] JSON: {turn_path}")
        if plot:
            print(f"[Refine] Plot: {turn_path.with_suffix('.png')}")
        print()
        final_payload = turn_payload
        if result["pass"]:
            break
        feedback_history.append(result["reason"])

    assert final_payload is not None
    payload = {
        **final_payload,
        "refinement": {
            "max_refinement_turns": max_refinement_turns,
            "turn_records": turn_records,
        },
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if plot:
        plot_payload(output, output.with_suffix(".png"), problem_id)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the LLM waypoint planner and NMPC/CBF execution pipeline."
    )
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Use built-in reference waypoints instead of the default LLM refinement pipeline.",
    )
    parser.add_argument("--problem-id", default="MA1")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--max-refinement-turns",
        type=int,
        default=2,
        help="Number of feedback retries after the first LLM attempt.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not save 3D PNG visualizations next to output JSON files.",
    )
    parser.add_argument(
        "--no-nmpc-cbf",
        action="store_true",
        help="Skip NMPC tracking and CBF shielding after waypoint smoothing.",
    )
    parser.add_argument(
        "--no-retiming",
        action="store_true",
        help="Skip geometric-path time-scaling optimization before NMPC execution.",
    )
    parser.add_argument(
        "--execute-reference",
        action="store_true",
        help="Also run the selected NMPC/CBF backend on the built-in reference trajectory.",
    )
    parser.add_argument(
        "--estimate-required-time",
        action="store_true",
        help="When execution fails, search for the longer mission time required by NMPC.",
    )
    parser.add_argument(
        "--target-time",
        type=float,
        default=None,
        help="Requested mission deadline in seconds. Defaults to the problem's target_time.",
    )
    parser.add_argument(
        "--nmpc-backend",
        choices=["cascade", "full"],
        default="cascade",
        help=(
            "Execution backend after waypoint smoothing. 'cascade' uses the existing "
            "lightweight NMPC+CBF tracker. 'full' uses CasADi/IPOPT full motor-level "
            "quadrotor NMPC with embedded CBF safety constraints and can be much slower."
        ),
    )
    parser.add_argument(
        "--full-apply-steps",
        type=int,
        default=4,
        help="For --nmpc-backend full, apply this many predicted steps before replanning.",
    )
    parser.add_argument(
        "--full-max-duration",
        type=float,
        default=None,
        help="Optional smoke-test duration limit in seconds for --nmpc-backend full.",
    )
    args = parser.parse_args()

    plot_outputs = not args.no_plot
    reference_mode = args.reference_only or args.execute_reference
    if not reference_mode:
        output = args.output or "llm_refined_trajectory.json"
        payload = run_llm_refinement(
            output_path=output,
            problem_id=args.problem_id,
            max_refinement_turns=args.max_refinement_turns,
            use_nmpc_cbf=not args.no_nmpc_cbf,
            use_retiming=not args.no_retiming,
            target_time=args.target_time,
            nmpc_backend=args.nmpc_backend,
            full_apply_steps=args.full_apply_steps,
            full_max_duration=args.full_max_duration,
            estimate_required_time_on_failure=args.estimate_required_time,
            plot=plot_outputs,
        )
        print("Refinement trial summary:")
        for record in payload.get("refinement", {}).get("turn_records", []):
            status = "PASS" if record["pass"] else "FAIL"
            print(f"  Trial {record['turn']}: {status}")
            print(f"    Reason: {record['reason']}")
            print(f"    JSON: {record['json_path']}")
            if record.get("plot_path"):
                print(f"    Plot: {record['plot_path']}")
        verification = payload["verification"]
        print(f"Saved {output}")
        if plot_outputs:
            print(f"Saved {Path(output).with_suffix('.png')}")
        print(f"LLM strategy: {payload['strategy']}")
        print(f"Pass: {verification['pass']}")
        print(f"Reason: {verification['reason']}")
        _print_nmpc_time_feasibility(
            verification.get("details", {}).get("nmpc_time_feasibility")
        )
    else:
        reference_output = args.output or "reference_trajectory.json"
        payload = save_reference_trajectory(
            reference_output,
            args.problem_id,
            use_retiming=not args.no_retiming,
            target_time=args.target_time,
        )
        verification = payload["verification"]
        print(f"Saved {reference_output}")
        print(f"Pass: {verification['pass']}")
        print(f"Reason: {verification['reason']}")
        if plot_outputs:
            plot_payload(reference_output, Path(reference_output).with_suffix(".png"), args.problem_id)
            print(f"Saved {Path(reference_output).with_suffix('.png')}")
        if args.execute_reference:
            executed_payload = save_executed_reference_trajectory(
                "executed_trajectory.json",
                args.problem_id,
                use_retiming=not args.no_retiming,
                target_time=args.target_time,
                nmpc_backend=args.nmpc_backend,
                full_apply_steps=args.full_apply_steps,
                full_max_duration=args.full_max_duration,
            )
            executed_verification = executed_payload["verification"]
            print("Saved executed_trajectory.json")
            print(f"Executed pass: {executed_verification['pass']}")
            print(f"Executed reason: {executed_verification['reason']}")
            if plot_outputs:
                plot_payload("executed_trajectory.json", "executed_trajectory.png", args.problem_id)
                print("Saved executed_trajectory.png")
            if args.estimate_required_time and not executed_verification["pass"]:
                requested_time = args.target_time or get_problem(args.problem_id).target_time
                estimate = estimate_required_nmpc_time(
                    args.problem_id,
                    build_geometric_reference_solution(args.problem_id),
                    requested_time,
                    nmpc_backend=args.nmpc_backend,
                    full_apply_steps=args.full_apply_steps,
                    full_max_duration=args.full_max_duration,
                )
                _print_nmpc_time_feasibility(estimate)
