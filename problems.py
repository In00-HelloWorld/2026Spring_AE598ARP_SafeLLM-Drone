from schemas import AgentSpec, KeyWaypoint, ObstacleSpec, ProblemSpec


PROBLEMS = {
    "MA1": ProblemSpec(
        problem_id="MA1",
        title="Two-Crazyflie Recharge-Aware Obstacle Mission",
        difficulty="Hard",
        description=(
            "Plan a coordinated mission for two Crazyflie drones. Each drone must pass "
            "through its assigned inspection waypoints, avoid virtual obstacles placed "
            "inside the box (-1, 2) x (-1, 2) x (0, 1.5), keep enough battery margin by "
            "using the two charging stations, and reach its own final destination. The "
            "drones must remain separated by at least 0.3 m throughout the mission."
        ),
        deliverable=(
            "Return timed waypoint(smoothed) trajectories for cf1 and cf2. Each trajectory must "
            "start at the drone's assigned charging station, visit its assigned targets, "
            "use charging stops when needed, avoid all obstacle boxes, maintain pairwise "
            "separation, and finish at its final goal."
        ),
        charging_stations=[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        obstacles=[
            ObstacleSpec(
                obstacle_id="obs_left",
                min_corner=[-0.80, 0.65, 0.0],
                max_corner=[-0.4, 1.75, 1.30],
            ),
            ObstacleSpec(
                obstacle_id="obs_front",
                min_corner=[-0.0, 1.25, 0.0],
                max_corner=[0.85, 1.95, 1.20],
            ),
            ObstacleSpec(
                obstacle_id="obs_right",
                min_corner=[1.20, -0.60, 0.0],
                max_corner=[2.00, 0.40, 1.40],
            ),
        ],
        targets=[
            [-1.0, 1.85, 1.00],
            [2.45, 2.25, 1.10],
            [2.35, -1.05, 1.00],
            [-1.05, -0.95, 1.10],
        ],
        agents=[
            AgentSpec(
                agent_id="cf1",
                start_station_index=0,
                required_target_indices=[0, 1],
                final_goal=[2.15, -0.85, 1.00],
            ),
            AgentSpec(
                agent_id="cf2",
                start_station_index=1,
                required_target_indices=[2, 3],
                final_goal=[-1.85, 2.15, 1.00],
            ),
        ],
        max_velocity=1.6,
        max_acceleration=1.8,
        max_jerk=8.0,
        max_snap=80.0,
        max_curvature=2.8,
        max_attitude_rate_deg_s=120.0,
        battery_start=100.0,
        battery_floor=35.0,
        battery_loss_per_meter=2.0,
        charge_rate=18.0,
        charge_radius=0.35,
        visit_radius=0.25,
        final_radius=0.25,
        min_agent_separation=0.30,
        obstacle_clearance=0.05,
        target_time=35.0,
        recommended_key_waypoints={
            "cf1": [
                KeyWaypoint(x=0.0, y=0.0, z=0.0, note="start S0"),
                KeyWaypoint(x=0.0, y=0.0, z=1.85, note="climb above obstacles"),
                KeyWaypoint(x=-1.0, y=1.85, z=1.85, note="over T0"),
                KeyWaypoint(x=-1.0, y=1.85, z=1.0, hold_seconds=1.0, note="visit T0"),
                KeyWaypoint(x=-1.0, y=1.85, z=1.85, note="leave T0"),
                KeyWaypoint(x=2.45, y=2.25, z=1.85, note="over T1"),
                KeyWaypoint(x=2.45, y=2.25, z=1.1, hold_seconds=1.0, note="visit T1"),
                KeyWaypoint(x=2.45, y=2.25, z=1.85, note="leave T1"),
                KeyWaypoint(x=2.75, y=-0.85, z=1.85, note="over cf1 goal"),
                KeyWaypoint(x=2.75, y=-0.85, z=1.0, hold_seconds=1.0, note="cf1 goal"),
            ],
            "cf2": [
                KeyWaypoint(x=1.0, y=1.0, z=0.0, note="start S1"),
                KeyWaypoint(x=1.0, y=1.0, z=2.30, note="climb above obstacles"),
                KeyWaypoint(x=2.35, y=-1.05, z=2.30, note="over T2"),
                KeyWaypoint(x=2.35, y=-1.05, z=1.0, hold_seconds=1.0, note="visit T2"),
                KeyWaypoint(x=2.35, y=-1.05, z=2.30, note="leave T2"),
                KeyWaypoint(x=-1.05, y=-0.95, z=2.30, note="over T3"),
                KeyWaypoint(x=-1.05, y=-0.95, z=1.1, hold_seconds=1.0, note="visit T3"),
                KeyWaypoint(x=-1.05, y=-0.95, z=2.30, note="leave T3"),
                KeyWaypoint(x=-1.85, y=2.15, z=2.30, note="over cf2 goal"),
                KeyWaypoint(x=-1.85, y=2.15, z=1.0, hold_seconds=1.0, note="cf2 goal"),
            ],
        },
        recommended_speed=0.8,
    ),
    "MA2": ProblemSpec(
        problem_id="MA2",
        title="Mandatory Mid-Mission Recharge Mission",
        difficulty="Hard",
        description=(
            "Plan a long single-Crazyflie inspection obstacle avoidance mission "
            "The drone must visit both inspection targets and finish at the "
            "final goal while avoiding obstacle boxes.(Do not pass the boxes or go under the boxes)"
        ),
        deliverable=(
            "Return one timed trajectory for cf1. The trajectory must start at S0, "
            "visit targets [0, 1], and can visit a charging stop at S1 long enough to keep "
            "battery above the floor, avoid obstacles, and finish at the final goal."
        ),
        charging_stations=[
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        obstacles=[
            ObstacleSpec(
                obstacle_id="wall_low",
                min_corner=[1.0, -0.45, 0.0],
                max_corner=[2.0, 0.45, 1.05],
            ),
            ObstacleSpec(
                obstacle_id="wall_high",
                min_corner=[4.7, 0.45, 0.0],
                max_corner=[5.7, 1.45, 1.20],
            ),
        ],
        targets=[
            [4.8, 1.85, 1.00],
            [7.15, -1.10, 1.05],
        ],
        agents=[
            AgentSpec(
                agent_id="cf1",
                start_station_index=0,
                required_target_indices=[0, 1],
                final_goal=[8.50, 0.90, 1.00],
            ),
        ],
        max_velocity=1.5,
        max_acceleration=1.7,
        max_jerk=8.0,
        max_snap=80.0,
        max_curvature=2.8,
        max_attitude_rate_deg_s=120.0,
        battery_start=100.0,
        battery_floor=35.0,
        battery_loss_per_meter=5.0,
        charge_rate=12.0,
        charge_radius=0.35,
        visit_radius=0.25,
        final_radius=0.25,
        min_agent_separation=0.30,
        obstacle_clearance=0.05,
        target_time=45.0,
        recommended_key_waypoints={
            "cf1": [
                KeyWaypoint(x=0.0, y=0.0, z=0.0, note="start S0"),
                KeyWaypoint(x=0.0, y=-0.70, z=1.35, note="climb and detour below wall_low"),
                KeyWaypoint(x=3.0, y=-0.70, z=1.35, note="approach charger S1"),
                KeyWaypoint(x=3.0, y=0.0, z=0.0, hold_seconds=4.0, note="mandatory recharge at S1"),
                KeyWaypoint(x=3.0, y=0.0, z=1.45, note="leave charger"),
                KeyWaypoint(x=3.8, y=1.85, z=1.45, note="north approach around wall_high"),
                KeyWaypoint(x=4.8, y=1.85, z=1.45, note="over T0"),
                KeyWaypoint(x=4.8, y=1.85, z=1.0, hold_seconds=1.0, note="visit T0"),
                KeyWaypoint(x=4.8, y=1.85, z=1.45, note="leave T0"),
                KeyWaypoint(x=6.25, y=1.85, z=1.45, note="east of wall_high"),
                KeyWaypoint(x=6.25, y=-1.35, z=1.45, note="wide turn around wall_high"),
                KeyWaypoint(x=7.15, y=-1.10, z=1.05, hold_seconds=1.0, note="visit T1"),
                KeyWaypoint(x=8.50, y=0.90, z=1.00, hold_seconds=1.0, note="final goal"),
            ],
        },
        recommended_speed=0.8,
    ),
    "MA3": ProblemSpec(
        problem_id="MA3",
        title="Intentionally Time-Infeasible Mission",
        difficulty="Infeasible",
        description=(
            "Plan a deliberately impossible single-Crazyflie mission. The spatial route "
            "is simple and obstacle-free, but the requested time window is too short for "
            "the given velocity, acceleration, jerk, and snap limits."
        ),
        deliverable=(
            "Return one timed trajectory for cf1 if possible. A correct solver should "
            "report that the requested time window is infeasible and estimate the "
            "minimum required time or needed limit increases."
        ),
        charging_stations=[
            [0.0, 0.0, 0.0],
        ],
        obstacles=[],
        targets=[
            [7.50, 0.0, 1.00],
        ],
        agents=[
            AgentSpec(
                agent_id="cf1",
                start_station_index=0,
                required_target_indices=[0],
                final_goal=[10.00, 0.0, 1.00],
            ),
        ],
        max_velocity=1.0,
        max_acceleration=1.0,
        max_jerk=4.0,
        max_snap=40.0,
        max_curvature=2.8,
        max_attitude_rate_deg_s=100.0,
        battery_start=100.0,
        battery_floor=20.0,
        battery_loss_per_meter=1.0,
        charge_rate=10.0,
        charge_radius=0.35,
        visit_radius=0.25,
        final_radius=0.25,
        min_agent_separation=0.30,
        obstacle_clearance=0.05,
        target_time=8.0,
        recommended_key_waypoints={
            "cf1": [
                KeyWaypoint(x=0.0, y=0.0, z=0.0, note="start S0"),
                KeyWaypoint(x=0.0, y=0.0, z=1.0, note="climb"),
                KeyWaypoint(x=7.50, y=0.0, z=1.0, hold_seconds=1.0, note="visit T0"),
                KeyWaypoint(x=10.00, y=0.0, z=1.0, hold_seconds=1.0, note="final goal"),
            ],
        },
        recommended_speed=0.8,
    )
}


def get_problem(problem_id: str) -> ProblemSpec:
    try:
        return PROBLEMS[problem_id]
    except KeyError as exc:
        raise ValueError(f"Unknown problem_id: {problem_id}") from exc


def build_problem_prompt(problem_id: str) -> str:
    problem = get_problem(problem_id)
    station_text = ", ".join(
        f"S{i}={station}" for i, station in enumerate(problem.charging_stations)
    )
    target_text = ", ".join(f"T{i}={target}" for i, target in enumerate(problem.targets))
    obstacle_text = ", ".join(
        f"{obs.obstacle_id}: min={obs.min_corner}, max={obs.max_corner}"
        for obs in problem.obstacles
    )
    agent_text = "\n".join(
        (
            f"- {agent.agent_id}: starts at S{agent.start_station_index}, "
            f"must visit targets {agent.required_target_indices}, "
            f"final_goal={agent.final_goal}"
        )
        for agent in problem.agents
    )
    shared = f"""
Multi-Agent Crazyflie Mission {problem.problem_id}: {problem.title}

Description:
{problem.description}

Charging stations: {station_text}
Targets: {target_text}
Obstacles: {obstacle_text}
Agents:
{agent_text}

Constraints:
1. Each Crazyflie battery starts at {problem.battery_start:.1f}%.
2. Battery loss is {problem.battery_loss_per_meter:.1f}% per meter.
3. Each battery must always remain above {problem.battery_floor:.1f}%.
4. Charging is allowed only within {problem.charge_radius:.2f} m of either charging station.
5. Charging rate is {problem.charge_rate:.1f}% per second.
6. Each Crazyflie must pass within {problem.visit_radius:.2f} m of every assigned target.
7. Each Crazyflie must finish within {problem.final_radius:.2f} m of its final goal.
8. Max velocity: {problem.max_velocity:.1f} m/s.
9. Max acceleration: {problem.max_acceleration:.1f} m/s^2.
10. Max jerk: {problem.max_jerk:.1f} m/s^3 for smooth quadrotor-feasible turns.
11. Max snap: {problem.max_snap:.1f} m/s^4 and max curvature: {problem.max_curvature:.1f} 1/m.
12. Max attitude rate: {problem.max_attitude_rate_deg_s:.1f} deg/s in the execution model.
13. Every path segment must avoid all obstacle boxes with {problem.obstacle_clearance:.2f} m clearance.
14. The two Crazyflies must stay at least {problem.min_agent_separation:.2f} m apart.
15. Mission should be feasible within the requested mission time {problem.target_time:.1f} seconds.

Deliverable:
{problem.deliverable}
"""
    return (
        shared
        + "\nReturn sparse key_waypoints(sparse but enough points to avoid obstacles) for each Crazyflie. This is the actual plan: "
        + "include start, obstacle-avoidance detour points, target visits, any charging station stops, "
        + "and the final goal. The helper tool will only smooth the path geometry and optimize timestamps; "
        + "it will not choose target order, charging stops, obstacle detours, or final routing for you. "
        + "If you need to return to a station for a mid-mission recharge, you can stop at the station"
        + "for as long as needed to keep battery above the floor for the rest of the mission.(But the maximum of battery is 100)"
        + "Do not try to control timing or speed; the optimizer assigns timestamps to fit the requested mission time. "
        + "If verifier feedback reports dynamic infeasibility, change the route geometry with wider, smoother turns "
        + "and avoid near-duplicate key waypoints or sharp vertical drops. "
    )
