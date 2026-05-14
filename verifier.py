from typing import Any, Sequence

import numpy as np

from problems import get_problem
from schemas import MissionSolution, ObstacleSpec


def _distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    return float(np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float)))


def _inside_station(
    point: Sequence[float], charging_stations: list[list[float]], charge_radius: float
) -> bool:
    return any(_distance(point, station) <= charge_radius for station in charging_stations)


def _point_in_expanded_box(
    point: Sequence[float], obstacle: ObstacleSpec, clearance: float
) -> bool:
    p = np.array(point, dtype=float)
    box_min = np.array(obstacle.min_corner, dtype=float) - clearance
    box_max = np.array(obstacle.max_corner, dtype=float) + clearance
    return bool(np.all(p >= box_min) and np.all(p <= box_max))


def _segment_hits_obstacle(
    start: Sequence[float],
    end: Sequence[float],
    obstacle: ObstacleSpec,
    clearance: float,
    sample_step: float = 0.05,
) -> bool:
    start_arr = np.array(start, dtype=float)
    end_arr = np.array(end, dtype=float)
    length = float(np.linalg.norm(end_arr - start_arr))
    sample_count = max(2, int(np.ceil(length / sample_step)) + 1)
    for alpha in np.linspace(0.0, 1.0, sample_count):
        point = start_arr + alpha * (end_arr - start_arr)
        if _point_in_expanded_box(point, obstacle, clearance):
            return True
    return False


def _point_segment_distance(
    point: Sequence[float], start: Sequence[float], end: Sequence[float]
) -> float:
    p = np.array(point, dtype=float)
    a = np.array(start, dtype=float)
    b = np.array(end, dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))
    alpha = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    closest = a + alpha * ab
    return float(np.linalg.norm(p - closest))


def _position_at(pts: np.ndarray, ts: np.ndarray, t: float) -> np.ndarray:
    if t <= ts[0]:
        return pts[0]
    if t >= ts[-1]:
        return pts[-1]
    idx = int(np.searchsorted(ts, t, side="right") - 1)
    dt = ts[idx + 1] - ts[idx]
    if dt <= 0.0:
        return pts[idx]
    alpha = (t - ts[idx]) / dt
    return pts[idx] + alpha * (pts[idx + 1] - pts[idx])


def verify(
    problem_id: str,
    solution: MissionSolution,
    target_time: float | None = None,
) -> dict[str, Any]:
    problem = get_problem(problem_id)
    effective_target_time = (
        target_time if target_time is not None else problem.target_time
    )
    details: dict[str, Any] = {
        "agents": {},
        "separation": {"pass": True, "min_distance": float("inf")},
        "time": {"pass": True, "total_time": 0.0},
    }
    reasons: list[str] = []

    expected_agents = {agent.agent_id: agent for agent in problem.agents}
    provided = {traj.agent_id: traj for traj in solution.agent_trajectories}
    missing_agents = [agent_id for agent_id in expected_agents if agent_id not in provided]
    if missing_agents:
        return {
            "pass": False,
            "details": details,
            "reason": f"Error: Missing trajectories for agents {missing_agents}.",
        }

    trajectory_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for agent_id, agent in expected_agents.items():
        trajectory = provided[agent_id]
        agent_details = {
            "kinematics": {"pass": True, "max_v": 0.0, "max_a": 0.0},
            "smoothness": {
                "pass": True,
                "max_jerk": 0.0,
                "max_snap": 0.0,
                "max_curvature": 0.0,
            },
            "battery": {"pass": True, "min_soc": problem.battery_start},
            "mission": {"pass": True, "visited_count": 0, "final_reached": False},
            "obstacles": {"pass": True},
        }
        details["agents"][agent_id] = agent_details

        if len(trajectory.waypoints) < 2:
            reasons.append(f"{agent_id}: fewer than 2 waypoints.")
            agent_details["mission"]["pass"] = False
            continue

        pts = np.array([[w.x, w.y, w.z] for w in trajectory.waypoints], dtype=float)
        ts = np.array([w.t for w in trajectory.waypoints], dtype=float)
        trajectory_arrays[agent_id] = (pts, ts)
        dt = np.diff(ts)

        if np.any(dt <= 0):
            reasons.append(f"{agent_id}: time must strictly increase.")
            agent_details["kinematics"]["pass"] = False
            continue

        start_station = problem.charging_stations[agent.start_station_index]
        if _distance(pts[0], start_station) > problem.charge_radius:
            agent_details["mission"]["pass"] = False
            reasons.append(f"{agent_id}: trajectory does not start at its assigned station.")

        vels = np.diff(pts, axis=0) / dt[:, np.newaxis]
        v_norms = np.linalg.norm(vels, axis=1)
        max_v = float(np.max(v_norms)) if v_norms.size else 0.0
        agent_details["kinematics"]["max_v"] = max_v
        if max_v > problem.max_velocity:
            agent_details["kinematics"]["pass"] = False
            reasons.append(
                f"{agent_id}: velocity exceeded {max_v:.2f} m/s "
                f"(limit {problem.max_velocity:.2f})."
            )

        if len(vels) > 1:
            accel_dt = dt[1:]
            accels = np.diff(vels, axis=0) / accel_dt[:, np.newaxis]
            a_norms = np.linalg.norm(accels, axis=1)
            max_a = float(np.max(a_norms)) if a_norms.size else 0.0
            agent_details["kinematics"]["max_a"] = max_a
            if max_a > problem.max_acceleration:
                agent_details["kinematics"]["pass"] = False
                reasons.append(
                    f"{agent_id}: acceleration exceeded {max_a:.2f} m/s^2 "
                    f"(limit {problem.max_acceleration:.2f})."
                )

            if len(accels) > 1:
                jerk_dt = accel_dt[1:]
                jerks = np.diff(accels, axis=0) / jerk_dt[:, np.newaxis]
                jerk_norms = np.linalg.norm(jerks, axis=1)
                max_jerk = float(np.max(jerk_norms)) if jerk_norms.size else 0.0
                agent_details["smoothness"]["max_jerk"] = max_jerk
                if max_jerk > problem.max_jerk:
                    agent_details["smoothness"]["pass"] = False
                    reasons.append(
                        f"{agent_id}: jerk exceeded {max_jerk:.2f} m/s^3 "
                        f"(limit {problem.max_jerk:.2f})."
                    )

                if len(jerks) > 1:
                    snap_dt = jerk_dt[1:]
                    snaps = np.diff(jerks, axis=0) / snap_dt[:, np.newaxis]
                    snap_norms = np.linalg.norm(snaps, axis=1)
                    max_snap = float(np.max(snap_norms)) if snap_norms.size else 0.0
                    agent_details["smoothness"]["max_snap"] = max_snap
                    if max_snap > problem.max_snap:
                        agent_details["smoothness"]["pass"] = False
                        reasons.append(
                            f"{agent_id}: snap exceeded {max_snap:.2f} m/s^4 "
                            f"(limit {problem.max_snap:.2f})."
                        )

            if len(accels) > 0:
                curvatures = []
                for accel_index, accel in enumerate(accels):
                    velocity = 0.5 * (vels[accel_index] + vels[accel_index + 1])
                    speed = float(np.linalg.norm(velocity))
                    if speed <= 0.30:
                        continue
                    curvature = float(np.linalg.norm(np.cross(velocity, accel)) / speed**3)
                    curvatures.append(curvature)
                max_curvature = float(max(curvatures, default=0.0))
                agent_details["smoothness"]["max_curvature"] = max_curvature
                if max_curvature > problem.max_curvature:
                    agent_details["smoothness"]["pass"] = False
                    reasons.append(
                        f"{agent_id}: curvature exceeded {max_curvature:.2f} 1/m "
                        f"(limit {problem.max_curvature:.2f})."
                    )

        soc = problem.battery_start
        visited_targets = {target_index: False for target_index in agent.required_target_indices}

        for i in range(1, len(pts)):
            start = pts[i - 1]
            end = pts[i]
            dist = float(np.linalg.norm(end - start))
            soc -= dist * problem.battery_loss_per_meter
            if _inside_station(start, problem.charging_stations, problem.charge_radius) and _inside_station(
                end, problem.charging_stations, problem.charge_radius
            ):
                soc += dt[i - 1] * problem.charge_rate
            soc = min(problem.battery_start, soc)
            agent_details["battery"]["min_soc"] = float(
                min(agent_details["battery"]["min_soc"], soc)
            )

            if soc < problem.battery_floor:
                agent_details["battery"]["pass"] = False
                reasons.append(
                    f"{agent_id}: battery floor violated at t={ts[i]:.1f}s "
                    f"({soc:.1f}% < {problem.battery_floor:.1f}%)."
                )
                break

            for obstacle in problem.obstacles:
                if _segment_hits_obstacle(
                    start, end, obstacle, problem.obstacle_clearance
                ):
                    agent_details["obstacles"]["pass"] = False
                    reasons.append(
                        f"{agent_id}: segment {i - 1}->{i} intersects {obstacle.obstacle_id}."
                    )
                    break

            for target_index in agent.required_target_indices:
                target = problem.targets[target_index]
                if _point_segment_distance(target, start, end) <= problem.visit_radius:
                    visited_targets[target_index] = True

        agent_details["mission"]["visited_count"] = sum(visited_targets.values())
        if not all(visited_targets.values()):
            missing = [
                target_index
                for target_index, visited in visited_targets.items()
                if not visited
            ]
            agent_details["mission"]["pass"] = False
            reasons.append(f"{agent_id}: missed assigned targets {missing}.")

        final_distance = _distance(pts[-1], agent.final_goal)
        agent_details["mission"]["final_reached"] = final_distance <= problem.final_radius
        if final_distance > problem.final_radius:
            agent_details["mission"]["pass"] = False
            reasons.append(
                f"{agent_id}: final goal missed by {final_distance:.2f} m."
            )

    if len(trajectory_arrays) >= 2:
        total_time = max(float(ts[-1]) for _, ts in trajectory_arrays.values())
        event_times = np.concatenate([ts for _, ts in trajectory_arrays.values()])
        grid_times = np.arange(0.0, total_time + 0.05, 0.10)
        sample_times = np.unique(np.concatenate([event_times, grid_times]))
        agent_ids = list(expected_agents.keys())
        min_distance = float("inf")
        for t in sample_times:
            for i, first_id in enumerate(agent_ids):
                first_pts, first_ts = trajectory_arrays[first_id]
                first_pos = _position_at(first_pts, first_ts, float(t))
                for second_id in agent_ids[i + 1 :]:
                    second_pts, second_ts = trajectory_arrays[second_id]
                    second_pos = _position_at(second_pts, second_ts, float(t))
                    min_distance = min(min_distance, _distance(first_pos, second_pos))
        details["separation"]["min_distance"] = min_distance
        if min_distance < problem.min_agent_separation:
            details["separation"]["pass"] = False
            reasons.append(
                f"Agent separation violated: {min_distance:.2f} m "
                f"(limit {problem.min_agent_separation:.2f})."
            )

    total_time = max(
        (float(ts[-1]) for _, ts in trajectory_arrays.values()),
        default=0.0,
    )
    details["time"]["total_time"] = total_time
    details["time"]["target_time"] = effective_target_time
    time_tolerance = 1.0e-6
    if effective_target_time is not None and total_time > effective_target_time + time_tolerance:
        details["time"]["pass"] = False
        reasons.append(
            f"Timeout: {total_time:.1f}s (target {effective_target_time:.1f}s)."
        )

    agent_passes = [
        all(component["pass"] for component in agent_details.values() if "pass" in component)
        for agent_details in details["agents"].values()
    ]
    overall_pass = (
        all(agent_passes)
        and details["separation"]["pass"]
        and details["time"]["pass"]
        and len(agent_passes) == len(problem.agents)
    )
    return {
        "pass": overall_pass,
        "details": details,
        "reason": " ".join(reasons) if not overall_pass else "Success! Multi-agent mission feasible.",
    }
