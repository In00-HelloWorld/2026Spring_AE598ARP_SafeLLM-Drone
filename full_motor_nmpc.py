from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import argparse
import json
from typing import Any

import numpy as np

from problems import get_problem
from schemas import AgentTrajectory, MissionSolution, Waypoint


@dataclass(frozen=True)
class FullMotorNMPCConfig:
    dt: float = 0.10
    horizon_steps: int = 20
    gravity: float = 9.81
    mass: float = 0.027
    arm_length: float = 0.046
    inertia: tuple[float, float, float] = (1.4e-5, 1.4e-5, 2.17e-5)
    yaw_torque_coeff: float = 0.006
    drag_coefficient: float = 0.08
    motor_time_constant: float = 0.06
    min_total_thrust_accel: float = 4.0
    max_total_thrust_accel: float = 16.0
    max_tilt_deg: float = 28.0
    max_velocity: float = 1.6
    max_acceleration: float = 1.8
    max_jerk: float = 8.0
    max_angular_rate_deg_s: float = 220.0
    position_weight: float = 40.0
    velocity_weight: float = 5.0
    acceleration_weight: float = 0.4
    jerk_weight: float = 0.08
    attitude_weight: float = 1.0
    angular_rate_weight: float = 0.05
    motor_input_weight: float = 0.02
    motor_smoothness_weight: float = 0.08
    terminal_position_weight: float = 120.0
    terminal_velocity_weight: float = 12.0
    ipopt_max_iter: int = 120
    ipopt_print_level: int = 0
    cbf_enabled: bool = True
    cbf_barrier_rate: float = 0.95
    cbf_obstacle_margin: float = 0.01
    cbf_agent_margin: float = 0.03
    cbf_slack_weight: float = 1.0e5
    cbf_disabled_value: float = 1.0e3
    max_obstacle_constraints: int = 8
    max_agent_constraints: int = 4


@dataclass
class FullMotorNMPCResult:
    first_motor_command: np.ndarray
    predicted_states: np.ndarray
    predicted_inputs: np.ndarray
    solve_time_s: float | None
    objective: float


@dataclass
class FullMotorExecutionMetrics:
    max_tracking_error: dict[str, float]
    mean_tracking_error: dict[str, float]
    max_roll_deg: dict[str, float]
    max_pitch_deg: dict[str, float]
    max_angular_rate_deg_s: dict[str, float]
    min_motor_thrust_accel: dict[str, float]
    max_motor_thrust_accel: dict[str, float]
    mean_solve_time_s: dict[str, float]
    max_solve_time_s: dict[str, float]
    solve_count: dict[str, int]
    min_obstacle_clearance: dict[str, float]
    min_agent_clearance: dict[str, float]


def casadi_available() -> bool:
    try:
        import casadi  # noqa: F401

        return True
    except ImportError:
        return False


def acados_available() -> bool:
    try:
        import acados_template  # noqa: F401

        return True
    except ImportError:
        return False


def _require_casadi():
    try:
        import casadi as ca

        return ca
    except ImportError as exc:
        raise ImportError(
            "CasADi is required for full motor-level NMPC. Install it with "
            "`pip install casadi`, then rerun the full NMPC option."
        ) from exc


def _rotation_matrix_ca(ca, roll, pitch, yaw):
    cr = ca.cos(roll)
    sr = ca.sin(roll)
    cp = ca.cos(pitch)
    sp = ca.sin(pitch)
    cy = ca.cos(yaw)
    sy = ca.sin(yaw)
    rx = ca.vertcat(
        ca.horzcat(1, 0, 0),
        ca.horzcat(0, cr, -sr),
        ca.horzcat(0, sr, cr),
    )
    ry = ca.vertcat(
        ca.horzcat(cp, 0, sp),
        ca.horzcat(0, 1, 0),
        ca.horzcat(-sp, 0, cp),
    )
    rz = ca.vertcat(
        ca.horzcat(cy, -sy, 0),
        ca.horzcat(sy, cy, 0),
        ca.horzcat(0, 0, 1),
    )
    return rz @ ry @ rx


def _euler_rates_ca(ca, roll, pitch, omega):
    p = omega[0]
    q = omega[1]
    r = omega[2]
    cp = ca.cos(pitch)
    cp_safe = ca.if_else(ca.fabs(cp) < 1e-4, 1e-4, cp)
    sr = ca.sin(roll)
    cr = ca.cos(roll)
    tp = ca.sin(pitch) / cp_safe
    return ca.vertcat(
        p + sr * tp * q + cr * tp * r,
        cr * q - sr * r,
        (sr / cp_safe) * q + (cr / cp_safe) * r,
    )


def _full_quadrotor_rhs(ca, x, u, config: FullMotorNMPCConfig):
    position = x[0:3]
    velocity = x[3:6]
    euler = x[6:9]
    omega = x[9:12]
    motors = x[12:16]

    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]
    rotation = _rotation_matrix_ca(ca, roll, pitch, yaw)
    total_thrust_accel = ca.sum1(motors)
    gravity = ca.vertcat(0, 0, config.gravity)
    acceleration = (
        rotation @ ca.vertcat(0, 0, total_thrust_accel)
        - gravity
        - config.drag_coefficient * velocity
    )

    forces = config.mass * motors
    f_front = forces[0]
    f_right = forces[1]
    f_back = forces[2]
    f_left = forces[3]
    tau_x = config.arm_length * (f_right - f_left)
    tau_y = config.arm_length * (f_back - f_front)
    tau_z = config.yaw_torque_coeff * (f_front - f_right + f_back - f_left)
    torque = ca.vertcat(tau_x, tau_y, tau_z)
    inertia = ca.DM(config.inertia)
    omega_dot = (torque - ca.cross(omega, inertia * omega)) / inertia
    motor_dot = (u - motors) / config.motor_time_constant

    return ca.vertcat(
        velocity,
        acceleration,
        _euler_rates_ca(ca, roll, pitch, omega),
        omega_dot,
        motor_dot,
    )


def _rk4_step(ca, x, u, config: FullMotorNMPCConfig):
    dt = config.dt
    k1 = _full_quadrotor_rhs(ca, x, u, config)
    k2 = _full_quadrotor_rhs(ca, x + 0.5 * dt * k1, u, config)
    k3 = _full_quadrotor_rhs(ca, x + 0.5 * dt * k2, u, config)
    k4 = _full_quadrotor_rhs(ca, x + dt * k3, u, config)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _config_key(config: FullMotorNMPCConfig) -> tuple[Any, ...]:
    return tuple(getattr(config, field) for field in config.__dataclass_fields__)


@lru_cache(maxsize=8)
def _build_casadi_solver_cached(config_key: tuple[Any, ...]):
    config = FullMotorNMPCConfig(
        **dict(zip(FullMotorNMPCConfig.__dataclass_fields__.keys(), config_key))
    )
    return _build_casadi_solver_uncached(config)


def build_casadi_full_motor_nmpc(config: FullMotorNMPCConfig | None = None) -> dict[str, Any]:
    config = config or FullMotorNMPCConfig()
    return _build_casadi_solver_cached(_config_key(config))


def _build_casadi_solver_uncached(config: FullMotorNMPCConfig) -> dict[str, Any]:
    ca = _require_casadi()
    nx = 16
    nu = 4
    ny = 6
    n = config.horizon_steps
    obstacle_param_dim = 7 * config.max_obstacle_constraints
    agent_param_dim = 5 * config.max_agent_constraints * (n + 1)
    safety_param_dim = obstacle_param_dim + agent_param_dim

    x = ca.MX.sym("X", nx, n + 1)
    u = ca.MX.sym("U", nu, n)
    x0_param = ca.MX.sym("x0", nx)
    ref_param = ca.MX.sym("ref", ny, n + 1)
    safety_param = ca.MX.sym("safety", safety_param_dim)
    obstacle_param = ca.reshape(
        safety_param[0:obstacle_param_dim],
        7,
        config.max_obstacle_constraints,
    )
    agent_param = ca.reshape(
        safety_param[obstacle_param_dim:],
        5,
        config.max_agent_constraints * (n + 1),
    )

    objective = 0
    constraints = [x[:, 0] - x0_param]
    q_pos = config.position_weight
    q_vel = config.velocity_weight
    q_acc = config.acceleration_weight
    q_jerk = config.jerk_weight
    q_att = config.attitude_weight
    q_omega = config.angular_rate_weight
    r_motor = config.motor_input_weight
    r_delta_motor = config.motor_smoothness_weight
    hover_motor = config.gravity / 4.0
    path_constraints = []
    predicted_accels = []

    for k in range(n):
        position_error = x[0:3, k] - ref_param[0:3, k]
        velocity_error = x[3:6, k] - ref_param[3:6, k]
        attitude = x[6:9, k]
        omega = x[9:12, k]
        motor_error = u[:, k] - hover_motor
        objective += q_pos * ca.dot(position_error, position_error)
        objective += q_vel * ca.dot(velocity_error, velocity_error)
        objective += q_att * ca.dot(attitude, attitude)
        objective += q_omega * ca.dot(omega, omega)
        objective += r_motor * ca.dot(motor_error, motor_error)
        if k > 0:
            delta_motor = u[:, k] - u[:, k - 1]
            objective += r_delta_motor * ca.dot(delta_motor, delta_motor)
        constraints.append(x[:, k + 1] - _rk4_step(ca, x[:, k], u[:, k], config))
        accel_k = (x[3:6, k + 1] - x[3:6, k]) / config.dt
        predicted_accels.append(accel_k)
        objective += q_acc * ca.dot(accel_k, accel_k)
        path_constraints.append(
            config.max_acceleration**2 - ca.dot(accel_k, accel_k)
        )
        if k > 0:
            jerk_k = (accel_k - predicted_accels[k - 1]) / config.dt
            objective += q_jerk * ca.dot(jerk_k, jerk_k)
            path_constraints.append(config.max_jerk**2 - ca.dot(jerk_k, jerk_k))

    terminal_position_error = x[0:3, n] - ref_param[0:3, n]
    terminal_velocity_error = x[3:6, n] - ref_param[3:6, n]
    objective += config.terminal_position_weight * ca.dot(
        terminal_position_error, terminal_position_error
    )
    objective += config.terminal_velocity_weight * ca.dot(
        terminal_velocity_error, terminal_velocity_error
    )

    safety_constraints = []
    if config.cbf_enabled:
        one_minus_gamma = 1.0 - config.cbf_barrier_rate
        for obstacle_index in range(config.max_obstacle_constraints):
            enabled = obstacle_param[0, obstacle_index]
            lower = obstacle_param[1:4, obstacle_index]
            upper = obstacle_param[4:7, obstacle_index]
            obstacle_h = []
            for k in range(n + 1):
                position = x[0:3, k]
                below = ca.fmax(lower - position, 0)
                above = ca.fmax(position - upper, 0)
                distance_sq = ca.dot(below, below) + ca.dot(above, above)
                h = distance_sq - config.cbf_obstacle_margin**2
                relaxed_h = (1 - enabled) * config.cbf_disabled_value + enabled * h
                safety_constraints.append(relaxed_h)
                obstacle_h.append(h)
            for k in range(n):
                cbf_h = obstacle_h[k + 1] - one_minus_gamma * obstacle_h[k]
                relaxed_cbf_h = (
                    (1 - enabled) * config.cbf_disabled_value + enabled * cbf_h
                )
                safety_constraints.append(relaxed_cbf_h)

        for agent_index in range(config.max_agent_constraints):
            agent_h = []
            for k in range(n + 1):
                column = agent_index * (n + 1) + k
                enabled = agent_param[0, column]
                other_position = agent_param[1:4, column]
                min_distance_sq = agent_param[4, column]
                delta = x[0:3, k] - other_position
                h = ca.dot(delta, delta) - min_distance_sq
                relaxed_h = (1 - enabled) * config.cbf_disabled_value + enabled * h
                safety_constraints.append(relaxed_h)
                agent_h.append(h)
            for k in range(n):
                column = agent_index * (n + 1) + k
                enabled = agent_param[0, column]
                cbf_h = agent_h[k + 1] - one_minus_gamma * agent_h[k]
                relaxed_cbf_h = (
                    (1 - enabled) * config.cbf_disabled_value + enabled * cbf_h
                )
                safety_constraints.append(relaxed_cbf_h)

    safety_count = len(safety_constraints)
    safety_slack = ca.MX.sym("safety_slack", safety_count) if safety_count else None
    if safety_count:
        objective += config.cbf_slack_weight * ca.dot(safety_slack, safety_slack)
        safety_constraints = [
            constraint + safety_slack[index]
            for index, constraint in enumerate(safety_constraints)
        ]

    opt_blocks = [ca.reshape(x, -1, 1), ca.reshape(u, -1, 1)]
    if safety_count:
        opt_blocks.append(safety_slack)
    opt_vars = ca.vertcat(*opt_blocks)
    params = ca.vertcat(x0_param, ca.reshape(ref_param, -1, 1), safety_param)
    g = ca.vertcat(*constraints, *path_constraints, *safety_constraints)
    nlp = {"x": opt_vars, "f": objective, "g": g, "p": params}
    solver = ca.nlpsol(
        "full_motor_nmpc",
        "ipopt",
        nlp,
        {
            "ipopt.print_level": config.ipopt_print_level,
            "ipopt.max_iter": config.ipopt_max_iter,
            "print_time": False,
            "record_time": True,
        },
    )

    motor_min = config.min_total_thrust_accel / 4.0
    motor_max = config.max_total_thrust_accel / 4.0
    max_tilt = np.deg2rad(config.max_tilt_deg)
    max_omega = np.deg2rad(config.max_angular_rate_deg_s)
    x_lbx = np.full((nx, n + 1), -np.inf)
    x_ubx = np.full((nx, n + 1), np.inf)
    x_lbx[3:6, :] = -config.max_velocity
    x_ubx[3:6, :] = config.max_velocity
    x_lbx[6:8, :] = -max_tilt
    x_ubx[6:8, :] = max_tilt
    x_lbx[9:12, :] = -max_omega
    x_ubx[9:12, :] = max_omega
    x_lbx[12:16, :] = motor_min
    x_ubx[12:16, :] = motor_max
    u_lbx = np.full((nu, n), motor_min)
    u_ubx = np.full((nu, n), motor_max)
    lbx = np.concatenate(
        [x_lbx.reshape(-1, order="F"), u_lbx.reshape(-1, order="F")]
    )
    ubx = np.concatenate(
        [x_ubx.reshape(-1, order="F"), u_ubx.reshape(-1, order="F")]
    )
    if safety_count:
        slack_lbx = np.zeros(safety_count)
        slack_ubx = np.full(safety_count, np.inf)
        lbx = np.concatenate([lbx, slack_lbx])
        ubx = np.concatenate([ubx, slack_ubx])
    equality_count = nx * (n + 1)
    path_count = len(path_constraints)
    lbg = np.concatenate(
        [np.zeros(equality_count), np.zeros(path_count), np.zeros(safety_count)]
    )
    ubg = np.concatenate(
        [
            np.zeros(equality_count),
            np.full(path_count, np.inf),
            np.full(safety_count, np.inf),
        ]
    )

    return {
        "solver": solver,
        "config": config,
        "nx": nx,
        "nu": nu,
        "ny": ny,
        "safety_param_dim": safety_param_dim,
        "equality_count": equality_count,
        "safety_count": safety_count,
        "path_count": path_count,
        "lbx": lbx,
        "ubx": ubx,
        "lbg": lbg,
        "ubg": ubg,
    }


def make_initial_full_state(
    position: np.ndarray,
    velocity: np.ndarray | None = None,
    yaw: float = 0.0,
    config: FullMotorNMPCConfig | None = None,
) -> np.ndarray:
    config = config or FullMotorNMPCConfig()
    state = np.zeros(16)
    state[0:3] = np.asarray(position, dtype=float)
    if velocity is not None:
        state[3:6] = np.asarray(velocity, dtype=float)
    state[8] = yaw
    state[12:16] = config.gravity / 4.0
    return state


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


def reference_from_trajectory(
    trajectory: AgentTrajectory,
    start_t: float,
    config: FullMotorNMPCConfig | None = None,
) -> np.ndarray:
    config = config or FullMotorNMPCConfig()
    points, times = _trajectory_arrays(trajectory)
    reference = np.zeros((6, config.horizon_steps + 1))
    for k in range(config.horizon_steps + 1):
        t = start_t + k * config.dt
        reference[0:3, k] = _position_at(points, times, t)
        reference[3:6, k] = _velocity_at(points, times, t)
    return reference


def _safety_param_dim(config: FullMotorNMPCConfig) -> int:
    return (
        7 * config.max_obstacle_constraints
        + 5 * config.max_agent_constraints * (config.horizon_steps + 1)
    )


def empty_safety_parameters(config: FullMotorNMPCConfig | None = None) -> np.ndarray:
    config = config or FullMotorNMPCConfig()
    return np.zeros(_safety_param_dim(config), dtype=float)


def _distance_to_expanded_box(
    point: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> float:
    below = np.maximum(box_min - point, 0.0)
    above = np.maximum(point - box_max, 0.0)
    return float(np.linalg.norm(below + above))


def _min_obstacle_clearance(
    point: np.ndarray,
    problem_id: str,
    config: FullMotorNMPCConfig,
) -> float:
    problem = get_problem(problem_id)
    best = float("inf")
    for obstacle in problem.obstacles:
        box_min = np.array(obstacle.min_corner, dtype=float) - problem.obstacle_clearance
        box_max = np.array(obstacle.max_corner, dtype=float) + problem.obstacle_clearance
        clearance = _distance_to_expanded_box(point, box_min, box_max)
        best = min(best, clearance - config.cbf_obstacle_margin)
    return best


def _min_agent_clearance(
    point: np.ndarray,
    t: float,
    other_trajectories: dict[str, AgentTrajectory],
    minimum_separation: float,
) -> float:
    best = float("inf")
    for trajectory in other_trajectories.values():
        points, times = _trajectory_arrays(trajectory)
        other_position = _position_at(points, times, t)
        best = min(best, float(np.linalg.norm(point - other_position) - minimum_separation))
    return best


def safety_parameters_from_problem(
    problem_id: str,
    agent_id: str,
    other_trajectories: dict[str, AgentTrajectory],
    start_t: float,
    config: FullMotorNMPCConfig | None = None,
) -> np.ndarray:
    config = config or FullMotorNMPCConfig()
    problem = get_problem(problem_id)
    if len(problem.obstacles) > config.max_obstacle_constraints:
        raise ValueError(
            f"Problem has {len(problem.obstacles)} obstacles but config allows "
            f"{config.max_obstacle_constraints}."
        )
    other_agent_ids = [
        agent.agent_id
        for agent in problem.agents
        if agent.agent_id != agent_id and agent.agent_id in other_trajectories
    ]
    if len(other_agent_ids) > config.max_agent_constraints:
        raise ValueError(
            f"Problem has {len(other_agent_ids)} neighboring agents but config allows "
            f"{config.max_agent_constraints}."
        )

    params = empty_safety_parameters(config)
    obstacle_dim = 7 * config.max_obstacle_constraints
    for obstacle_index, obstacle in enumerate(problem.obstacles):
        base = 7 * obstacle_index
        box_min = np.array(obstacle.min_corner, dtype=float) - problem.obstacle_clearance
        box_max = np.array(obstacle.max_corner, dtype=float) + problem.obstacle_clearance
        params[base : base + 7] = np.concatenate([[1.0], box_min, box_max])

    minimum_separation = problem.min_agent_separation + config.cbf_agent_margin
    min_distance_sq = minimum_separation**2
    for agent_slot, other_agent_id in enumerate(other_agent_ids):
        trajectory = other_trajectories[other_agent_id]
        points, times = _trajectory_arrays(trajectory)
        for k in range(config.horizon_steps + 1):
            t = start_t + k * config.dt
            position = _position_at(points, times, t)
            column = agent_slot * (config.horizon_steps + 1) + k
            base = obstacle_dim + 5 * column
            params[base : base + 5] = np.concatenate([[1.0], position, [min_distance_sq]])
    return params


def trajectory_from_payload(payload: dict, agent_id: str) -> AgentTrajectory:
    for item in payload["agent_trajectories"]:
        if item["agent_id"] == agent_id:
            return AgentTrajectory(
                agent_id=agent_id,
                waypoints=[Waypoint(**waypoint) for waypoint in item["waypoints"]],
            )
    raise ValueError(f"Trajectory for agent_id={agent_id!r} not found.")


def solve_casadi_full_motor_nmpc(
    initial_state: np.ndarray,
    reference: np.ndarray,
    config: FullMotorNMPCConfig | None = None,
    warm_start: dict[str, np.ndarray] | None = None,
    safety_parameters: np.ndarray | None = None,
) -> FullMotorNMPCResult:
    config = config or FullMotorNMPCConfig()
    built = build_casadi_full_motor_nmpc(config)
    ca = _require_casadi()
    nx = built["nx"]
    nu = built["nu"]
    n = config.horizon_steps
    expected_reference_shape = (built["ny"], n + 1)
    if reference.shape != expected_reference_shape:
        raise ValueError(
            f"reference must have shape {expected_reference_shape}, got {reference.shape}"
        )
    if safety_parameters is None:
        safety_parameters = empty_safety_parameters(config)
    safety_parameters = np.asarray(safety_parameters, dtype=float)
    if safety_parameters.shape != (built["safety_param_dim"],):
        raise ValueError(
            f"safety_parameters must have shape {(built['safety_param_dim'],)}, "
            f"got {safety_parameters.shape}"
        )

    if warm_start is not None:
        x_guess = warm_start.get("X")
        u_guess = warm_start.get("U")
        if x_guess is not None:
            x_guess = np.asarray(x_guess, dtype=float).copy()
            x_guess[:, 0] = np.asarray(initial_state, dtype=float)
    else:
        x_guess = None
        u_guess = None

    if x_guess is None:
        x_guess = np.repeat(np.asarray(initial_state, dtype=float)[:, None], n + 1, axis=1)
        x_guess[0:3, :] = reference[0:3, :]
        x_guess[3:6, :] = reference[3:6, :]
    if u_guess is None:
        u_guess = np.full((nu, n), config.gravity / 4.0)

    param = np.concatenate(
        [initial_state, reference.reshape(-1, order="F"), safety_parameters]
    )
    guess_blocks = [x_guess.reshape(-1, order="F"), u_guess.reshape(-1, order="F")]
    safety_count = built.get("safety_count", 0)
    if safety_count:
        guess_blocks.append(np.zeros(safety_count))
    opt_guess = np.concatenate(guess_blocks)
    result = built["solver"](
        x0=opt_guess,
        p=param,
        lbx=built["lbx"],
        ubx=built["ubx"],
        lbg=built["lbg"],
        ubg=built["ubg"],
    )
    opt = np.array(result["x"]).reshape(-1)
    x_end = nx * (n + 1)
    u_end = x_end + nu * n
    x_opt = opt[:x_end].reshape(nx, n + 1, order="F")
    u_opt = opt[x_end:u_end].reshape(nu, n, order="F")
    stats = built["solver"].stats()
    solve_time = stats.get("t_wall_total", None)
    return FullMotorNMPCResult(
        first_motor_command=u_opt[:, 0],
        predicted_states=x_opt,
        predicted_inputs=u_opt,
        solve_time_s=solve_time,
        objective=float(result["f"]),
    )


def execute_with_full_motor_nmpc(
    problem_id: str,
    reference_solution: MissionSolution,
    config: FullMotorNMPCConfig | None = None,
    apply_steps: int = 4,
    max_duration: float | None = None,
) -> tuple[MissionSolution, FullMotorExecutionMetrics]:
    config = config or FullMotorNMPCConfig()
    if apply_steps < 1:
        raise ValueError("apply_steps must be at least 1.")
    apply_steps = min(apply_steps, config.horizon_steps)
    problem = get_problem(problem_id)
    reference_by_agent = {
        trajectory.agent_id: trajectory for trajectory in reference_solution.agent_trajectories
    }
    missing = [
        agent.agent_id for agent in problem.agents if agent.agent_id not in reference_by_agent
    ]
    if missing:
        raise ValueError(f"Missing reference trajectories for agents {missing}.")

    trajectories: list[AgentTrajectory] = []
    metrics = FullMotorExecutionMetrics(
        max_tracking_error={},
        mean_tracking_error={},
        max_roll_deg={},
        max_pitch_deg={},
        max_angular_rate_deg_s={},
        min_motor_thrust_accel={},
        max_motor_thrust_accel={},
        mean_solve_time_s={},
        max_solve_time_s={},
        solve_count={},
        min_obstacle_clearance={},
        min_agent_clearance={},
    )
    executed_by_agent: dict[str, AgentTrajectory] = {}

    for agent in problem.agents:
        agent_id = agent.agent_id
        trajectory = reference_by_agent[agent_id]
        other_trajectories = {
            other.agent_id: executed_by_agent.get(
                other.agent_id,
                reference_by_agent[other.agent_id],
            )
            for other in problem.agents
            if other.agent_id != agent_id and other.agent_id in reference_by_agent
        }
        points, times = _trajectory_arrays(trajectory)
        mission_end = float(times[-1])
        total_end = mission_end
        if max_duration is not None:
            total_end = min(total_end, max_duration)
        initial_state = make_initial_full_state(
            points[0],
            np.zeros(3),
            config=config,
        )
        state = initial_state
        current_t = 0.0
        waypoints = [
            Waypoint(
                x=float(state[0]),
                y=float(state[1]),
                z=float(state[2]),
                t=0.0,
            )
        ]
        tracking_errors: list[float] = []
        roll_values: list[float] = []
        pitch_values: list[float] = []
        angular_rates: list[float] = []
        motor_values: list[float] = []
        solve_times: list[float] = []
        obstacle_clearances: list[float] = []
        agent_clearances: list[float] = []
        solve_count = 0
        warm_start: dict[str, np.ndarray] | None = None

        while current_t + 1e-9 < total_end:
            reference = reference_from_trajectory(trajectory, current_t, config)
            safety_parameters = safety_parameters_from_problem(
                problem_id,
                agent_id,
                other_trajectories,
                current_t,
                config,
            )
            try:
                result = solve_casadi_full_motor_nmpc(
                    state,
                    reference,
                    config,
                    warm_start=warm_start,
                    safety_parameters=safety_parameters,
                )
            except RuntimeError:
                result = solve_casadi_full_motor_nmpc(
                    state,
                    reference,
                    config,
                    warm_start=None,
                    safety_parameters=safety_parameters,
                )
            solve_count += 1
            if result.solve_time_s is not None:
                solve_times.append(float(result.solve_time_s))
            warm_start = {
                "X": np.column_stack(
                    [result.predicted_states[:, 1:], result.predicted_states[:, -1]]
                ),
                "U": np.column_stack(
                    [result.predicted_inputs[:, 1:], result.predicted_inputs[:, -1]]
                ),
            }

            steps_to_apply = min(
                apply_steps,
                int(np.ceil((total_end - current_t) / config.dt)),
            )
            for step in range(1, steps_to_apply + 1):
                state = result.predicted_states[:, step].copy()
                current_t = min(current_t + config.dt, total_end)
                ref_position = _position_at(points, times, current_t)
                tracking_errors.append(float(np.linalg.norm(state[0:3] - ref_position)))
                roll_values.append(abs(float(np.rad2deg(state[6]))))
                pitch_values.append(abs(float(np.rad2deg(state[7]))))
                angular_rates.append(float(np.rad2deg(np.linalg.norm(state[9:12]))))
                motor_values.extend([float(value) for value in state[12:16]])
                obstacle_clearances.append(
                    _min_obstacle_clearance(state[0:3], problem_id, config)
                )
                if other_trajectories:
                    agent_clearances.append(
                        _min_agent_clearance(
                            state[0:3],
                            current_t,
                            other_trajectories,
                            problem.min_agent_separation + config.cbf_agent_margin,
                        )
                    )
                waypoints.append(
                    Waypoint(
                        x=float(state[0]),
                        y=float(state[1]),
                        z=float(state[2]),
                        t=float(current_t),
                    )
                )
                if current_t + 1e-9 >= total_end:
                    break

        executed_trajectory = AgentTrajectory(agent_id=agent_id, waypoints=waypoints)
        trajectories.append(executed_trajectory)
        executed_by_agent[agent_id] = executed_trajectory
        metrics.max_tracking_error[agent_id] = float(max(tracking_errors, default=0.0))
        metrics.mean_tracking_error[agent_id] = float(
            np.mean(tracking_errors) if tracking_errors else 0.0
        )
        metrics.max_roll_deg[agent_id] = float(max(roll_values, default=0.0))
        metrics.max_pitch_deg[agent_id] = float(max(pitch_values, default=0.0))
        metrics.max_angular_rate_deg_s[agent_id] = float(max(angular_rates, default=0.0))
        metrics.min_motor_thrust_accel[agent_id] = float(min(motor_values, default=0.0))
        metrics.max_motor_thrust_accel[agent_id] = float(max(motor_values, default=0.0))
        metrics.mean_solve_time_s[agent_id] = float(
            np.mean(solve_times) if solve_times else 0.0
        )
        metrics.max_solve_time_s[agent_id] = float(max(solve_times, default=0.0))
        metrics.solve_count[agent_id] = solve_count
        metrics.min_obstacle_clearance[agent_id] = float(
            min(obstacle_clearances, default=0.0)
        )
        metrics.min_agent_clearance[agent_id] = float(
            min(agent_clearances, default=0.0)
        )

    return MissionSolution(agent_trajectories=trajectories), metrics


def build_acados_full_motor_ocp(config: FullMotorNMPCConfig | None = None):
    config = config or FullMotorNMPCConfig()
    ca = _require_casadi()
    try:
        from acados_template import AcadosModel, AcadosOcp
    except ImportError as exc:
        raise ImportError(
            "acados_template is required to export the acados OCP. Install acados and "
            "its Python interface, then rerun this function."
        ) from exc

    nx = 16
    nu = 4
    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)
    xdot = ca.MX.sym("xdot", nx)
    f_expl = _full_quadrotor_rhs(ca, x, u, config)
    model = AcadosModel()
    model.name = "full_motor_quadrotor"
    model.x = x
    model.u = u
    model.xdot = xdot
    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot - f_expl

    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = config.horizon_steps
    ocp.solver_options.tf = config.dt * config.horizon_steps
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    return ocp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build or run a full nonlinear motor-level quadrotor NMPC problem."
    )
    parser.add_argument("--input", default="executed_trajectory.json")
    parser.add_argument("--agent-id", default="cf1")
    parser.add_argument("--start-t", type=float, default=0.0)
    parser.add_argument(
        "--solve",
        action="store_true",
        help="Attempt one CasADi/IPOPT NMPC solve. Requires casadi to be installed.",
    )
    args = parser.parse_args()

    print(f"CasADi available: {casadi_available()}")
    print(f"acados_template available: {acados_available()}")
    config = FullMotorNMPCConfig()
    if not args.solve:
        print("Use --solve to run one full motor-level NMPC solve when CasADi is installed.")
    else:
        payload = json.loads(open(args.input, encoding="utf-8").read())
        trajectory = trajectory_from_payload(payload, args.agent_id)
        points, times = _trajectory_arrays(trajectory)
        initial_position = _position_at(points, times, args.start_t)
        initial_velocity = _velocity_at(points, times, args.start_t)
        initial_state = make_initial_full_state(initial_position, initial_velocity, config=config)
        reference = reference_from_trajectory(trajectory, args.start_t, config)
        result = solve_casadi_full_motor_nmpc(initial_state, reference, config)
        print(f"Objective: {result.objective:.6g}")
        print(f"First motor command [m/s^2 per motor]: {result.first_motor_command}")
