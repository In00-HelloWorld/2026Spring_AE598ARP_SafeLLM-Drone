from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from problems import get_problem
from schemas import AgentTrajectory, MissionSolution, ObstacleSpec, Waypoint


@dataclass(frozen=True)
class NMPCConfig:
    dynamics_model: str = "full"
    dt: float = 0.25
    horizon_steps: int = 10
    position_weight: float = 10.0
    velocity_weight: float = 1.2
    terminal_position_weight: float = 18.0
    terminal_velocity_weight: float = 2.0
    control_weight: float = 0.08
    settle_seconds: float = 8.0
    goal_tolerance: float = 0.08
    gravity: float = 9.81
    drag_coefficient: float = 0.08
    max_tilt_deg: float = 28.0
    max_jerk: float = 8.0
    max_attitude_rate_deg_s: float = 120.0
    max_thrust_rate_accel_s: float = 12.0
    min_thrust_accel: float = 4.0
    max_thrust_accel: float = 16.0
    attitude_time_constant: float = 0.12
    thrust_time_constant: float = 0.10
    mass: float = 0.027
    arm_length: float = 0.046
    inertia: tuple[float, float, float] = (1.4e-5, 1.4e-5, 2.17e-5)
    yaw_torque_coeff: float = 0.006
    attitude_kp: tuple[float, float, float] = (2.6e-4, 2.6e-4, 1.0e-4)
    attitude_kd: tuple[float, float, float] = (5.2e-5, 5.2e-5, 2.5e-5)
    max_torque: tuple[float, float, float] = (3.0e-4, 3.0e-4, 1.2e-4)
    max_angular_rate_deg_s: float = 220.0
    motor_time_constant: float = 0.06
    max_motor_thrust_rate_accel_s: float = 45.0


@dataclass(frozen=True)
class CBFConfig:
    obstacle_influence: float = 0.45
    agent_influence: float = 0.75
    safety_margin: float = 0.03
    barrier_rate: float = 0.85
    slack_penalty: float = 5000.0
    enabled: bool = True


@dataclass
class ExecutionMetrics:
    max_tracking_error: dict[str, float] = field(default_factory=dict)
    mean_tracking_error: dict[str, float] = field(default_factory=dict)
    cbf_adjustment_count: int = 0
    max_cbf_adjustment: float = 0.0
    max_roll_deg: dict[str, float] = field(default_factory=dict)
    max_pitch_deg: dict[str, float] = field(default_factory=dict)
    max_roll_rate_deg_s: dict[str, float] = field(default_factory=dict)
    max_pitch_rate_deg_s: dict[str, float] = field(default_factory=dict)
    min_thrust_accel: dict[str, float] = field(default_factory=dict)
    max_thrust_accel: dict[str, float] = field(default_factory=dict)
    max_thrust_rate_accel_s: dict[str, float] = field(default_factory=dict)
    max_jerk: dict[str, float] = field(default_factory=dict)
    max_angular_rate_deg_s: dict[str, float] = field(default_factory=dict)
    max_motor_thrust_rate_accel_s: dict[str, float] = field(default_factory=dict)


@dataclass
class DroneState:
    position: np.ndarray
    velocity: np.ndarray
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    thrust_accel: float = 9.81
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    motor_thrust_accels: np.ndarray = field(
        default_factory=lambda: np.full(4, 9.81 / 4.0)
    )


def _norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def _clip_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    value = _norm(vector)
    if value <= limit or value <= 1e-12:
        return vector
    return vector * (limit / value)


def _trajectory_arrays(trajectory: AgentTrajectory) -> tuple[np.ndarray, np.ndarray]:
    points = np.array([[w.x, w.y, w.z] for w in trajectory.waypoints], dtype=float)
    times = np.array([w.t for w in trajectory.waypoints], dtype=float)
    return points, times


def _position_at(points: np.ndarray, times: np.ndarray, t: float) -> np.ndarray:
    if t <= times[0]:
        return points[0].copy()
    if t >= times[-1]:
        return points[-1].copy()
    idx = int(np.searchsorted(times, t, side="right") - 1)
    dt = times[idx + 1] - times[idx]
    if dt <= 1e-9:
        return points[idx].copy()
    alpha = (t - times[idx]) / dt
    return points[idx] + alpha * (points[idx + 1] - points[idx])


def _velocity_at(points: np.ndarray, times: np.ndarray, t: float) -> np.ndarray:
    if len(points) < 2:
        return np.zeros(3)
    if t <= times[0]:
        idx = 0
    elif t >= times[-1]:
        idx = len(points) - 2
    else:
        idx = int(np.searchsorted(times, t, side="right") - 1)
        idx = min(idx, len(points) - 2)
    dt = times[idx + 1] - times[idx]
    if dt <= 1e-9:
        return np.zeros(3)
    return (points[idx + 1] - points[idx]) / dt


def _reference_stack(
    points: np.ndarray,
    times: np.ndarray,
    start_t: float,
    dt: float,
    horizon_steps: int,
) -> np.ndarray:
    refs: list[np.ndarray] = []
    for step in range(1, horizon_steps + 1):
        t = start_t + step * dt
        refs.append(_position_at(points, times, t))
        refs.append(_velocity_at(points, times, t))
    return np.concatenate(refs)


def _body_z_axis(roll: float, pitch: float, yaw: float = 0.0) -> np.ndarray:
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    return np.array(
        [
            cy * sp * cr + sy * sr,
            sy * sp * cr - cy * sr,
            cp * cr,
        ],
        dtype=float,
    )


def _rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _euler_rates_from_body_rates(
    roll: float, pitch: float, angular_velocity: np.ndarray
) -> np.ndarray:
    p, q, r = angular_velocity
    cp = max(np.cos(pitch), 1e-4)
    sr = np.sin(roll)
    cr = np.cos(roll)
    tp = np.sin(pitch) / cp
    return np.array(
        [
            p + sr * tp * q + cr * tp * r,
            cr * q - sr * r,
            (sr / cp) * q + (cr / cp) * r,
        ],
        dtype=float,
    )


def _clip_tilt(roll: float, pitch: float, max_tilt_rad: float) -> tuple[float, float]:
    tilt = float(np.hypot(roll, pitch))
    if tilt <= max_tilt_rad or tilt <= 1e-12:
        return roll, pitch
    scale = max_tilt_rad / tilt
    return roll * scale, pitch * scale


def _accel_to_drone_command(
    desired_accel: np.ndarray,
    velocity: np.ndarray,
    config: NMPCConfig,
) -> tuple[float, float, float]:
    gravity = np.array([0.0, 0.0, config.gravity])
    force = desired_accel + gravity + config.drag_coefficient * velocity
    thrust = float(np.clip(_norm(force), config.min_thrust_accel, config.max_thrust_accel))
    if thrust <= 1e-9:
        return 0.0, 0.0, config.min_thrust_accel

    body_z = force / thrust
    roll_cmd = float(np.arcsin(np.clip(-body_z[1], -1.0, 1.0)))
    pitch_cmd = float(np.arctan2(body_z[0], max(body_z[2], 1e-6)))
    roll_cmd, pitch_cmd = _clip_tilt(
        roll_cmd, pitch_cmd, np.deg2rad(config.max_tilt_deg)
    )
    return roll_cmd, pitch_cmd, thrust


def _accel_from_drone_state(state: DroneState, config: NMPCConfig) -> np.ndarray:
    total_thrust_accel = (
        float(np.sum(state.motor_thrust_accels))
        if config.dynamics_model == "full"
        else state.thrust_accel
    )
    thrust_axis = _body_z_axis(state.roll, state.pitch, state.yaw)
    gravity = np.array([0.0, 0.0, config.gravity])
    return total_thrust_accel * thrust_axis - gravity - config.drag_coefficient * state.velocity


def _motor_mix_accels(
    total_thrust_accel: float,
    torque_cmd: np.ndarray,
    config: NMPCConfig,
) -> np.ndarray:
    mass = config.mass
    arm = config.arm_length
    yaw_coeff = config.yaw_torque_coeff
    tau_x, tau_y, tau_z = torque_cmd
    total_force = mass * total_thrust_accel
    forces = np.array(
        [
            total_force / 4.0 - tau_y / (2.0 * arm) + tau_z / (4.0 * yaw_coeff),
            total_force / 4.0 + tau_x / (2.0 * arm) - tau_z / (4.0 * yaw_coeff),
            total_force / 4.0 + tau_y / (2.0 * arm) + tau_z / (4.0 * yaw_coeff),
            total_force / 4.0 - tau_x / (2.0 * arm) - tau_z / (4.0 * yaw_coeff),
        ],
        dtype=float,
    )
    per_motor_max = mass * config.max_thrust_accel / 4.0
    per_motor_min = mass * max(config.min_thrust_accel, 0.0) / 4.0
    return np.clip(forces, per_motor_min, per_motor_max) / mass


def _motor_torque_from_accels(motor_thrust_accels: np.ndarray, config: NMPCConfig) -> np.ndarray:
    forces = config.mass * motor_thrust_accels
    f_front, f_right, f_back, f_left = forces
    tau_x = config.arm_length * (f_right - f_left)
    tau_y = config.arm_length * (f_back - f_front)
    tau_z = config.yaw_torque_coeff * (f_front - f_right + f_back - f_left)
    return np.array([tau_x, tau_y, tau_z], dtype=float)


def _step_full_quadrotor_state(
    state: DroneState,
    roll_cmd: float,
    pitch_cmd: float,
    thrust_cmd: float,
    dt: float,
    config: NMPCConfig,
) -> DroneState:
    roll_cmd, pitch_cmd = _clip_tilt(roll_cmd, pitch_cmd, np.deg2rad(config.max_tilt_deg))
    attitude_error = np.array(
        [roll_cmd - state.roll, pitch_cmd - state.pitch, 0.0],
        dtype=float,
    )
    kp = np.array(config.attitude_kp, dtype=float)
    kd = np.array(config.attitude_kd, dtype=float)
    torque_cmd = kp * attitude_error - kd * state.angular_velocity
    torque_limit = np.array(config.max_torque, dtype=float)
    torque_cmd = np.clip(torque_cmd, -torque_limit, torque_limit)

    motor_cmd = _motor_mix_accels(thrust_cmd, torque_cmd, config)
    max_motor_delta = config.max_motor_thrust_rate_accel_s * dt
    limited_motor_cmd = state.motor_thrust_accels + np.clip(
        motor_cmd - state.motor_thrust_accels,
        -max_motor_delta,
        max_motor_delta,
    )
    motor_alpha = 1.0 - np.exp(-dt / max(config.motor_time_constant, 1e-6))
    motor_thrust_accels = state.motor_thrust_accels + motor_alpha * (
        limited_motor_cmd - state.motor_thrust_accels
    )
    motor_thrust_accels = np.clip(
        motor_thrust_accels,
        config.min_thrust_accel / 4.0,
        config.max_thrust_accel / 4.0,
    )

    torque = _motor_torque_from_accels(motor_thrust_accels, config)
    inertia = np.array(config.inertia, dtype=float)
    omega = state.angular_velocity
    omega_dot = (torque - np.cross(omega, inertia * omega)) / inertia
    next_omega = omega + dt * omega_dot
    next_omega = _clip_norm(next_omega, np.deg2rad(config.max_angular_rate_deg_s))

    euler_dot = _euler_rates_from_body_rates(state.roll, state.pitch, next_omega)
    roll = state.roll + dt * euler_dot[0]
    pitch = state.pitch + dt * euler_dot[1]
    yaw = state.yaw + dt * euler_dot[2]
    roll, pitch = _clip_tilt(roll, pitch, np.deg2rad(config.max_tilt_deg))

    total_thrust_accel = float(np.sum(motor_thrust_accels))
    rotation = _rotation_matrix(roll, pitch, yaw)
    gravity = np.array([0.0, 0.0, config.gravity])
    accel = (
        rotation @ np.array([0.0, 0.0, total_thrust_accel])
        - gravity
        - config.drag_coefficient * state.velocity
    )
    position = state.position + dt * state.velocity + 0.5 * dt**2 * accel
    velocity = state.velocity + dt * accel
    return DroneState(
        position=position,
        velocity=velocity,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        thrust_accel=total_thrust_accel,
        angular_velocity=next_omega,
        motor_thrust_accels=motor_thrust_accels,
    )


def _step_drone_state(
    state: DroneState,
    roll_cmd: float,
    pitch_cmd: float,
    thrust_cmd: float,
    dt: float,
    config: NMPCConfig,
) -> DroneState:
    if config.dynamics_model == "full":
        return _step_full_quadrotor_state(
            state, roll_cmd, pitch_cmd, thrust_cmd, dt, config
        )

    attitude_alpha = 1.0 - np.exp(-dt / max(config.attitude_time_constant, 1e-6))
    thrust_alpha = 1.0 - np.exp(-dt / max(config.thrust_time_constant, 1e-6))

    max_attitude_delta = np.deg2rad(config.max_attitude_rate_deg_s) * dt
    filtered_roll_cmd = state.roll + float(
        np.clip(roll_cmd - state.roll, -max_attitude_delta, max_attitude_delta)
    )
    filtered_pitch_cmd = state.pitch + float(
        np.clip(pitch_cmd - state.pitch, -max_attitude_delta, max_attitude_delta)
    )
    roll = state.roll + attitude_alpha * (filtered_roll_cmd - state.roll)
    pitch = state.pitch + attitude_alpha * (filtered_pitch_cmd - state.pitch)
    roll, pitch = _clip_tilt(roll, pitch, np.deg2rad(config.max_tilt_deg))
    max_thrust_delta = config.max_thrust_rate_accel_s * dt
    filtered_thrust_cmd = state.thrust_accel + float(
        np.clip(thrust_cmd - state.thrust_accel, -max_thrust_delta, max_thrust_delta)
    )
    thrust = state.thrust_accel + thrust_alpha * (filtered_thrust_cmd - state.thrust_accel)
    thrust = float(np.clip(thrust, config.min_thrust_accel, config.max_thrust_accel))

    updated = DroneState(
        position=state.position.copy(),
        velocity=state.velocity.copy(),
        roll=roll,
        pitch=pitch,
        yaw=state.yaw,
        thrust_accel=thrust,
    )
    accel = _accel_from_drone_state(updated, config)
    position = state.position + dt * state.velocity + 0.5 * dt**2 * accel
    velocity = state.velocity + dt * accel
    return DroneState(
        position=position,
        velocity=velocity,
        roll=roll,
        pitch=pitch,
        yaw=state.yaw,
        thrust_accel=thrust,
    )


def _prediction_matrices(dt: float, horizon_steps: int) -> tuple[np.ndarray, np.ndarray]:
    a = np.block(
        [
            [np.eye(3), dt * np.eye(3)],
            [np.zeros((3, 3)), np.eye(3)],
        ]
    )
    b = np.vstack([0.5 * dt**2 * np.eye(3), dt * np.eye(3)])
    state_dim = 6
    control_dim = 3
    mx = np.zeros((state_dim * horizon_steps, state_dim))
    mu = np.zeros((state_dim * horizon_steps, control_dim * horizon_steps))

    a_power = np.eye(state_dim)
    for row in range(horizon_steps):
        a_power = a @ a_power
        mx[row * state_dim : (row + 1) * state_dim, :] = a_power
        for col in range(row + 1):
            transition = np.linalg.matrix_power(a, row - col)
            block = transition @ b
            mu[
                row * state_dim : (row + 1) * state_dim,
                col * control_dim : (col + 1) * control_dim,
            ] = block
    return mx, mu


def _solve_nmpc_acceleration(
    position: np.ndarray,
    velocity: np.ndarray,
    reference: tuple[np.ndarray, np.ndarray],
    t: float,
    max_acceleration: float,
    config: NMPCConfig,
) -> np.ndarray:
    points, times = reference
    horizon = config.horizon_steps
    state = np.concatenate([position, velocity])
    refs = _reference_stack(points, times, t, config.dt, horizon)
    mx, mu = _prediction_matrices(config.dt, horizon)

    weights: list[float] = []
    for step in range(horizon):
        terminal = step == horizon - 1
        weights.extend(
            [config.terminal_position_weight if terminal else config.position_weight] * 3
        )
        weights.extend(
            [config.terminal_velocity_weight if terminal else config.velocity_weight] * 3
        )
    q = np.diag(weights)
    r = config.control_weight * np.eye(3 * horizon)
    lhs = mu.T @ q @ mu + r
    rhs = -mu.T @ q @ (mx @ state - refs)

    try:
        controls = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        controls = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return _clip_norm(controls[:3], max_acceleration)


def _solve_nmpc_drone_acceleration(
    state: DroneState,
    reference: tuple[np.ndarray, np.ndarray],
    t: float,
    max_acceleration: float,
    config: NMPCConfig,
) -> np.ndarray:
    desired_accel = _solve_nmpc_acceleration(
        state.position,
        state.velocity,
        reference,
        t,
        max_acceleration,
        config,
    )
    roll_cmd, pitch_cmd, thrust_cmd = _accel_to_drone_command(
        desired_accel,
        state.velocity,
        config,
    )
    feasible_axis = _body_z_axis(roll_cmd, pitch_cmd, state.yaw)
    feasible_accel = (
        thrust_cmd * feasible_axis
        - np.array([0.0, 0.0, config.gravity])
        - config.drag_coefficient * state.velocity
    )
    return _clip_norm(feasible_accel, max_acceleration)


def _expanded_box(
    obstacle: ObstacleSpec, clearance: float, margin: float
) -> tuple[np.ndarray, np.ndarray]:
    box_min = np.array(obstacle.min_corner, dtype=float) - clearance - margin
    box_max = np.array(obstacle.max_corner, dtype=float) + clearance + margin
    return box_min, box_max


def _nearest_box_halfspace(
    point: np.ndarray,
    obstacle: ObstacleSpec,
    clearance: float,
    config: CBFConfig,
) -> tuple[np.ndarray, float] | None:
    box_min, box_max = _expanded_box(obstacle, clearance, config.safety_margin)
    closest = np.minimum(np.maximum(point, box_min), box_max)
    delta = point - closest
    distance = _norm(delta)
    inside = bool(np.all(point >= box_min) and np.all(point <= box_max))

    if not inside and distance > config.obstacle_influence:
        return None

    if distance > 1e-9:
        normal = delta / distance
        boundary = float(normal @ closest)
        return normal, boundary

    distances = np.array(
        [
            abs(point[0] - box_min[0]),
            abs(box_max[0] - point[0]),
            abs(point[1] - box_min[1]),
            abs(box_max[1] - point[1]),
            abs(point[2] - box_min[2]),
            abs(box_max[2] - point[2]),
        ]
    )
    axis_choice = int(np.argmin(distances))
    normals = [
        np.array([-1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    bounds = [-box_min[0], box_max[0], -box_min[1], box_max[1], -box_min[2], box_max[2]]
    return normals[axis_choice], float(bounds[axis_choice])


def _linear_constraints(
    positions: list[np.ndarray],
    velocities: list[np.ndarray],
    nominal_accels: list[np.ndarray],
    agent_ids: list[str],
    problem_id: str,
    nmpc_config: NMPCConfig,
    cbf_config: CBFConfig,
) -> list[tuple[np.ndarray, float]]:
    problem = get_problem(problem_id)
    dt = nmpc_config.dt
    coeff = 0.5 * dt**2
    constraints: list[tuple[np.ndarray, float]] = []

    for agent_index, position in enumerate(positions):
        base_next = position + dt * velocities[agent_index]
        nominal_next = base_next + coeff * nominal_accels[agent_index]
        for obstacle in problem.obstacles:
            halfspace = _nearest_box_halfspace(
                nominal_next, obstacle, problem.obstacle_clearance, cbf_config
            )
            if halfspace is None:
                continue
            normal, boundary = halfspace
            current_h = float(normal @ position - boundary)
            required = boundary + max(0.0, (1.0 - cbf_config.barrier_rate) * current_h)
            c = np.zeros(3 * len(agent_ids))
            c[3 * agent_index : 3 * agent_index + 3] = coeff * normal
            b = required - float(normal @ base_next)
            constraints.append((c, b))

    min_sep = problem.min_agent_separation + cbf_config.safety_margin
    for i, first_id in enumerate(agent_ids):
        for j, second_id in enumerate(agent_ids[i + 1 :], start=i + 1):
            base_rel = (positions[i] + dt * velocities[i]) - (
                positions[j] + dt * velocities[j]
            )
            nominal_rel = base_rel + coeff * (nominal_accels[i] - nominal_accels[j])
            rel_norm = _norm(nominal_rel)
            current_distance = _norm(positions[i] - positions[j])
            if (
                rel_norm > min_sep + cbf_config.agent_influence
                and current_distance > min_sep + cbf_config.agent_influence
            ):
                continue
            if rel_norm <= 1e-9:
                nominal_rel = positions[i] - positions[j]
                rel_norm = max(_norm(nominal_rel), 1e-9)
            normal = nominal_rel / rel_norm
            current_h = float(normal @ (positions[i] - positions[j]) - min_sep)
            required = min_sep + max(0.0, (1.0 - cbf_config.barrier_rate) * current_h)
            c = np.zeros(3 * len(agent_ids))
            c[3 * i : 3 * i + 3] = coeff * normal
            c[3 * j : 3 * j + 3] = -coeff * normal
            b = required - float(normal @ base_rel)
            constraints.append((c, b))

    return constraints


def _project_with_cbf(
    positions: list[np.ndarray],
    velocities: list[np.ndarray],
    nominal_accels: list[np.ndarray],
    agent_ids: list[str],
    problem_id: str,
    nmpc_config: NMPCConfig,
    cbf_config: CBFConfig,
) -> list[np.ndarray]:
    if not cbf_config.enabled:
        return nominal_accels

    problem = get_problem(problem_id)
    accel_limit = problem.max_acceleration
    accel_x0 = np.concatenate(nominal_accels)
    constraints = _linear_constraints(
        positions,
        velocities,
        nominal_accels,
        agent_ids,
        problem_id,
        nmpc_config,
        cbf_config,
    )

    if not constraints:
        return [_clip_norm(accel, accel_limit) for accel in nominal_accels]

    slack_count = len(constraints)
    x0 = np.concatenate([accel_x0, np.zeros(slack_count)])
    accel_dim = len(accel_x0)
    scipy_constraints = []
    for slack_index, (c, b) in enumerate(constraints):
        scipy_constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, c=c, b=b, slack_index=slack_index: c @ x[:accel_dim]
                + x[accel_dim + slack_index]
                - b,
            }
        )

    for slack_index in range(slack_count):
        scipy_constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, slack_index=slack_index: x[accel_dim + slack_index],
            }
        )

    for idx, velocity in enumerate(velocities):
        start = 3 * idx
        scipy_constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, start=start: accel_limit**2
                - float(np.dot(x[start : start + 3], x[start : start + 3])),
            }
        )
        scipy_constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, velocity=velocity, start=start: problem.max_velocity**2
                - float(
                    np.dot(
                        velocity + nmpc_config.dt * x[start : start + 3],
                        velocity + nmpc_config.dt * x[start : start + 3],
                    )
                ),
            }
        )

    result = minimize(
        lambda x: 0.5 * float(np.dot(x[:accel_dim] - accel_x0, x[:accel_dim] - accel_x0))
        + 0.5 * cbf_config.slack_penalty * float(np.dot(x[accel_dim:], x[accel_dim:])),
        x0,
        method="SLSQP",
        bounds=[(-accel_limit, accel_limit)] * accel_dim + [(0.0, None)] * slack_count,
        constraints=scipy_constraints,
        options={"maxiter": 80, "ftol": 1e-9, "disp": False},
    )
    if result.success:
        projected = result.x[:accel_dim]
    else:
        projected = _fallback_project(accel_x0, constraints, accel_limit)

    return [
        _clip_norm(projected[3 * idx : 3 * idx + 3], accel_limit)
        for idx in range(len(agent_ids))
    ]


def _fallback_project(
    accelerations: np.ndarray,
    constraints: list[tuple[np.ndarray, float]],
    accel_limit: float,
) -> np.ndarray:
    projected = accelerations.copy()
    for _ in range(12):
        for c, b in constraints:
            violation = b - float(c @ projected)
            denom = float(c @ c)
            if violation > 0.0 and denom > 1e-12:
                projected += (violation / denom) * c
        for idx in range(len(projected) // 3):
            start = 3 * idx
            projected[start : start + 3] = _clip_norm(
                projected[start : start + 3], accel_limit
            )
    return projected


def _limit_jerk(
    desired_accel: np.ndarray,
    previous_accel: np.ndarray,
    dt: float,
    max_jerk: float,
) -> np.ndarray:
    max_delta = max_jerk * dt
    return previous_accel + _clip_norm(desired_accel - previous_accel, max_delta)


def execute_with_nmpc_cbf(
    problem_id: str,
    reference_solution: MissionSolution,
    nmpc_config: NMPCConfig | None = None,
    cbf_config: CBFConfig | None = None,
) -> tuple[MissionSolution, ExecutionMetrics]:
    problem = get_problem(problem_id)
    nmpc = nmpc_config or NMPCConfig()
    cbf = cbf_config or CBFConfig()

    references = {
        trajectory.agent_id: _trajectory_arrays(trajectory)
        for trajectory in reference_solution.agent_trajectories
    }
    agent_ids = [agent.agent_id for agent in problem.agents]
    missing = [agent_id for agent_id in agent_ids if agent_id not in references]
    if missing:
        raise ValueError(f"Missing reference trajectories for agents {missing}.")

    states = [
        DroneState(
            position=references[agent_id][0][0].copy(),
            velocity=np.zeros(3),
            thrust_accel=nmpc.gravity,
            angular_velocity=np.zeros(3),
            motor_thrust_accels=np.full(4, nmpc.gravity / 4.0),
        )
        for agent_id in agent_ids
    ]
    executed = {
        agent_id: [
            Waypoint(
                x=float(states[idx].position[0]),
                y=float(states[idx].position[1]),
                z=float(states[idx].position[2]),
                t=0.0,
            )
        ]
        for idx, agent_id in enumerate(agent_ids)
    }
    tracking_errors = {agent_id: [] for agent_id in agent_ids}
    roll_history = {agent_id: [] for agent_id in agent_ids}
    pitch_history = {agent_id: [] for agent_id in agent_ids}
    roll_rate_history = {agent_id: [] for agent_id in agent_ids}
    pitch_rate_history = {agent_id: [] for agent_id in agent_ids}
    thrust_history = {agent_id: [] for agent_id in agent_ids}
    thrust_rate_history = {agent_id: [] for agent_id in agent_ids}
    angular_rate_history = {agent_id: [] for agent_id in agent_ids}
    motor_rate_history = {agent_id: [] for agent_id in agent_ids}
    jerk_history = {agent_id: [] for agent_id in agent_ids}
    previous_accels = [np.zeros(3) for _ in agent_ids]
    metrics = ExecutionMetrics()
    reference_end = max(float(times[-1]) for _, times in references.values())
    total_end = reference_end + nmpc.settle_seconds
    t = 0.0

    while t + 1e-9 < total_end:
        nominal_accels = []
        for idx, agent_id in enumerate(agent_ids):
            reference = references[agent_id]
            nominal = _solve_nmpc_drone_acceleration(
                states[idx],
                reference,
                t,
                problem.max_acceleration,
                nmpc,
            )
            nominal_accels.append(nominal)

        shielded_accels = _project_with_cbf(
            [state.position for state in states],
            [state.velocity for state in states],
            nominal_accels,
            agent_ids,
            problem_id,
            nmpc,
            cbf,
        )

        for nominal, shielded in zip(nominal_accels, shielded_accels):
            delta = _norm(shielded - nominal)
            if delta > 1e-6:
                metrics.cbf_adjustment_count += 1
                metrics.max_cbf_adjustment = max(metrics.max_cbf_adjustment, delta)

        step_dt = min(nmpc.dt, total_end - t)
        for idx, agent_id in enumerate(agent_ids):
            accel = _limit_jerk(
                _clip_norm(shielded_accels[idx], problem.max_acceleration),
                previous_accels[idx],
                step_dt,
                min(nmpc.max_jerk, problem.max_jerk),
            )
            jerk_history[agent_id].append(_norm(accel - previous_accels[idx]) / step_dt)
            previous_accels[idx] = accel
            roll_cmd, pitch_cmd, thrust_cmd = _accel_to_drone_command(
                accel,
                states[idx].velocity,
                nmpc,
            )
            previous_roll = states[idx].roll
            previous_pitch = states[idx].pitch
            previous_thrust = states[idx].thrust_accel
            previous_motors = states[idx].motor_thrust_accels.copy()
            next_state = _step_drone_state(
                states[idx],
                roll_cmd,
                pitch_cmd,
                thrust_cmd,
                step_dt,
                nmpc,
            )
            next_state.velocity = _clip_norm(next_state.velocity, problem.max_velocity)
            states[idx] = next_state
            t_next = t + step_dt
            ref_position = _position_at(references[agent_id][0], references[agent_id][1], t_next)
            tracking_errors[agent_id].append(_norm(states[idx].position - ref_position))
            roll_history[agent_id].append(abs(np.rad2deg(states[idx].roll)))
            pitch_history[agent_id].append(abs(np.rad2deg(states[idx].pitch)))
            roll_rate_history[agent_id].append(
                abs(np.rad2deg(states[idx].roll - previous_roll)) / step_dt
            )
            pitch_rate_history[agent_id].append(
                abs(np.rad2deg(states[idx].pitch - previous_pitch)) / step_dt
            )
            thrust_history[agent_id].append(states[idx].thrust_accel)
            thrust_rate_history[agent_id].append(
                abs(states[idx].thrust_accel - previous_thrust) / step_dt
            )
            angular_rate_history[agent_id].append(
                float(np.rad2deg(np.linalg.norm(states[idx].angular_velocity)))
            )
            motor_rate_history[agent_id].append(
                float(np.max(np.abs(states[idx].motor_thrust_accels - previous_motors)) / step_dt)
            )
            executed[agent_id].append(
                Waypoint(
                    x=float(states[idx].position[0]),
                    y=float(states[idx].position[1]),
                    z=float(states[idx].position[2]),
                    t=float(t_next),
                )
            )
        t += step_dt

        goal_errors = []
        for idx, agent in enumerate(problem.agents):
            goal_errors.append(
                _norm(states[idx].position - np.array(agent.final_goal, dtype=float))
            )
        if t >= reference_end and max(goal_errors, default=0.0) <= nmpc.goal_tolerance:
            break

    for agent_id, values in tracking_errors.items():
        if values:
            metrics.max_tracking_error[agent_id] = float(max(values))
            metrics.mean_tracking_error[agent_id] = float(np.mean(values))
        else:
            metrics.max_tracking_error[agent_id] = 0.0
            metrics.mean_tracking_error[agent_id] = 0.0
        metrics.max_roll_deg[agent_id] = float(max(roll_history[agent_id], default=0.0))
        metrics.max_pitch_deg[agent_id] = float(max(pitch_history[agent_id], default=0.0))
        metrics.max_roll_rate_deg_s[agent_id] = float(
            max(roll_rate_history[agent_id], default=0.0)
        )
        metrics.max_pitch_rate_deg_s[agent_id] = float(
            max(pitch_rate_history[agent_id], default=0.0)
        )
        metrics.min_thrust_accel[agent_id] = float(
            min(thrust_history[agent_id], default=nmpc.gravity)
        )
        metrics.max_thrust_accel[agent_id] = float(
            max(thrust_history[agent_id], default=nmpc.gravity)
        )
        metrics.max_thrust_rate_accel_s[agent_id] = float(
            max(thrust_rate_history[agent_id], default=0.0)
        )
        metrics.max_jerk[agent_id] = float(max(jerk_history[agent_id], default=0.0))
        metrics.max_angular_rate_deg_s[agent_id] = float(
            max(angular_rate_history[agent_id], default=0.0)
        )
        metrics.max_motor_thrust_rate_accel_s[agent_id] = float(
            max(motor_rate_history[agent_id], default=0.0)
        )

    solution = MissionSolution(
        agent_trajectories=[
            AgentTrajectory(agent_id=agent_id, waypoints=executed[agent_id])
            for agent_id in agent_ids
        ]
    )
    return solution, metrics
