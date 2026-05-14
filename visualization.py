from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from problems import get_problem
from schemas import AgentTrajectory, MissionSolution, Waypoint


COLORS = {
    "cf1": "#1f77b4",
    "cf2": "#d62728",
}


def _set_axes_equal(ax) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = limits.mean(axis=1)
    radius = 0.5 * max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([max(0.0, centers[2] - radius), centers[2] + radius])


def _box_faces(box_min: np.ndarray, box_max: np.ndarray) -> list[list[tuple[float, float, float]]]:
    x0, y0, z0 = box_min
    x1, y1, z1 = box_max
    vertices = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    return [
        [vertices[i] for i in [0, 1, 2, 3]],
        [vertices[i] for i in [4, 5, 6, 7]],
        [vertices[i] for i in [0, 1, 5, 4]],
        [vertices[i] for i in [2, 3, 7, 6]],
        [vertices[i] for i in [1, 2, 6, 5]],
        [vertices[i] for i in [0, 3, 7, 4]],
    ]


def _draw_box(ax, box_min, box_max, color, alpha, label=None) -> None:
    faces = _box_faces(np.array(box_min, dtype=float), np.array(box_max, dtype=float))
    collection = Poly3DCollection(
        faces,
        facecolors=color,
        edgecolors="#333333",
        linewidths=0.7,
        alpha=alpha,
    )
    if label:
        collection.set_label(label)
    ax.add_collection3d(collection)


def _trajectory_from_payload(payload: dict) -> MissionSolution:
    trajectories = []
    for item in payload["agent_trajectories"]:
        waypoints = [Waypoint(**waypoint) for waypoint in item["waypoints"]]
        trajectories.append(AgentTrajectory(agent_id=item["agent_id"], waypoints=waypoints))
    return MissionSolution(agent_trajectories=trajectories)


def _failed_segments(payload: dict) -> dict[str, set[tuple[int, int]]]:
    reason = payload.get("verification", {}).get("reason", "")
    failed: dict[str, set[tuple[int, int]]] = {}
    for agent_id, start, end in re.findall(r"(\w+): segment (\d+)->(\d+) intersects", reason):
        failed.setdefault(agent_id, set()).add((int(start), int(end)))
    return failed


def plot_solution(
    problem_id: str,
    solution: MissionSolution,
    output_path: str | Path = "trajectory_3d.png",
    title: str | None = None,
    failed_segments: dict[str, set[tuple[int, int]]] | None = None,
) -> Path:
    problem = get_problem(problem_id)
    failed_segments = failed_segments or {}

    fig = plt.figure(figsize=(10, 8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title or f"{problem_id} trajectory")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    for obstacle in problem.obstacles:
        _draw_box(
            ax,
            obstacle.min_corner,
            obstacle.max_corner,
            color="#808080",
            alpha=0.22,
            label="obstacle" if obstacle == problem.obstacles[0] else None,
        )
        expanded_min = np.array(obstacle.min_corner) - problem.obstacle_clearance
        expanded_max = np.array(obstacle.max_corner) + problem.obstacle_clearance
        _draw_box(ax, expanded_min, expanded_max, color="#ff7f0e", alpha=0.08)
        center = 0.5 * (np.array(obstacle.min_corner) + np.array(obstacle.max_corner))
        ax.text(center[0], center[1], center[2], obstacle.obstacle_id, fontsize=8)

    stations = np.array(problem.charging_stations, dtype=float)
    ax.scatter(
        stations[:, 0],
        stations[:, 1],
        stations[:, 2],
        marker="s",
        s=80,
        color="#2ca02c",
        label="charging station",
    )
    for idx, station in enumerate(stations):
        ax.text(station[0], station[1], station[2] + 0.08, f"S{idx}", color="#2ca02c")

    targets = np.array(problem.targets, dtype=float)
    ax.scatter(
        targets[:, 0],
        targets[:, 1],
        targets[:, 2],
        marker="*",
        s=130,
        color="#9467bd",
        label="target",
    )
    for idx, target in enumerate(targets):
        ax.text(target[0], target[1], target[2] + 0.08, f"T{idx}", color="#9467bd")

    for agent in problem.agents:
        goal = np.array(agent.final_goal, dtype=float)
        color = COLORS.get(agent.agent_id, "#111111")
        ax.scatter(goal[0], goal[1], goal[2], marker="X", s=95, color=color)
        ax.text(goal[0], goal[1], goal[2] + 0.08, f"{agent.agent_id} goal", color=color)

    for trajectory in solution.agent_trajectories:
        color = COLORS.get(trajectory.agent_id, "#111111")
        points = np.array([[w.x, w.y, w.z] for w in trajectory.waypoints], dtype=float)
        ax.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=color,
            linewidth=2.0,
            label=trajectory.agent_id,
        )
        ax.scatter(points[0, 0], points[0, 1], points[0, 2], color=color, marker="o", s=45)
        ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], color=color, marker="^", s=55)

        for start, end in failed_segments.get(trajectory.agent_id, set()):
            if 0 <= start < len(points) and 0 <= end < len(points):
                seg = points[[start, end]]
                ax.plot(
                    seg[:, 0],
                    seg[:, 1],
                    seg[:, 2],
                    color="#ff0000",
                    linewidth=5.0,
                    label="failed segment",
                )

    all_points = []
    for trajectory in solution.agent_trajectories:
        all_points.extend([[w.x, w.y, w.z] for w in trajectory.waypoints])
    all_points.extend(problem.targets)
    all_points.extend(problem.charging_stations)
    all_points.extend([agent.final_goal for agent in problem.agents])
    all_points = np.array(all_points, dtype=float)
    padding = 0.35
    ax.set_xlim(float(all_points[:, 0].min() - padding), float(all_points[:, 0].max() + padding))
    ax.set_ylim(float(all_points[:, 1].min() - padding), float(all_points[:, 1].max() + padding))
    ax.set_zlim(0.0, float(max(all_points[:, 2].max() + padding, 2.6)))
    _set_axes_equal(ax)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper left")
    ax.view_init(elev=24, azim=-55)
    fig.tight_layout()

    output = Path(output_path)
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_payload(
    input_path: str | Path,
    output_path: str | Path | None = None,
    problem_id: str | None = None,
) -> Path:
    path = Path(input_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    solution = _trajectory_from_payload(payload)
    selected_problem = problem_id or payload.get("problem_id", "MA1")
    output = output_path or path.with_suffix("").as_posix() + "_3d.png"
    title = f"{selected_problem}: {'PASS' if payload.get('verification', {}).get('pass') else 'FAIL'}"
    return plot_solution(
        selected_problem,
        solution,
        output,
        title=title,
        failed_segments=_failed_segments(payload),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a mission problem and trajectory JSON.")
    parser.add_argument("--input", default="llm_executed_trajectory.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--problem-id", default=None)
    args = parser.parse_args()
    saved = plot_payload(args.input, args.output, args.problem_id)
    print(f"Saved {saved}")
