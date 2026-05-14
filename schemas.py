from typing import List

from pydantic import BaseModel, Field


class Waypoint(BaseModel):
    x: float
    y: float
    z: float
    t: float


class AgentTrajectory(BaseModel):
    agent_id: str
    waypoints: List[Waypoint]


class MissionSolution(BaseModel):
    agent_trajectories: List[AgentTrajectory]


class KeyWaypoint(BaseModel):
    x: float
    y: float
    z: float
    hold_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Optional hover/charge duration after reaching this key waypoint",
    )
    note: str = Field(
        default="",
        description="Short label such as start, target, charge, detour, or goal",
    )


class ToolAgentPlan(BaseModel):
    agent_id: str = Field(description="Crazyflie id, for example cf1 or cf2")
    key_waypoints: List[KeyWaypoint] = Field(
        description=(
            "Sparse LLM-planned waypoints. The tool smooths the path between "
            "these points and optimizes timestamps separately."
        )
    )


class ToolPlanResponse(BaseModel):
    agent_plans: List[ToolAgentPlan] = Field(
        description="One sparse key-waypoint plan for each Crazyflie"
    )
    strategy: str = Field(description="Reasoning behind the selected multi-agent plan")


class ObstacleSpec(BaseModel):
    obstacle_id: str
    min_corner: List[float]
    max_corner: List[float]


class AgentSpec(BaseModel):
    agent_id: str
    start_station_index: int
    required_target_indices: List[int]
    final_goal: List[float]


class ProblemSpec(BaseModel):
    problem_id: str
    title: str
    difficulty: str
    description: str
    deliverable: str
    agents: List[AgentSpec]
    charging_stations: List[List[float]]
    obstacles: List[ObstacleSpec]
    targets: List[List[float]]
    max_velocity: float
    max_acceleration: float
    max_jerk: float = 8.0
    max_snap: float = 80.0
    max_curvature: float = 8.0
    max_attitude_rate_deg_s: float = 120.0
    battery_start: float
    battery_floor: float
    battery_loss_per_meter: float
    charge_rate: float
    charge_radius: float
    visit_radius: float
    final_radius: float
    min_agent_separation: float
    obstacle_clearance: float
    target_time: float | None = None
    recommended_key_waypoints: dict[str, List[KeyWaypoint]]
    recommended_speed: float = 1.2
