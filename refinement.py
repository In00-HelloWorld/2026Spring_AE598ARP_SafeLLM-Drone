import json
from pathlib import Path

from tool_pipeline import run_tool_pipeline


def run_refinement(
    problem_id: str = "MA1",
    trials: int = 3,
    max_refinement_turns: int = 2,
    output_path: str | Path = "llm_refinement_results.json",
):
    table: list[list[bool]] = []
    records: list[dict] = []
    for trial in range(trials):
        print(f"--- Refinement Trial {trial + 1} ({problem_id}) ---")
        trial_results: list[bool] = []
        turn_records: list[dict] = []
        feedback_history: list[str] = []
        for turn in range(max_refinement_turns + 1):
            result, strategy, solution = run_tool_pipeline(problem_id, feedback=feedback_history)
            trial_results.append(result["pass"])
            print(f"Turn {turn + 1}: {'PASS' if result['pass'] else 'FAIL'}")
            print(f"Strategy: {strategy}")
            turn_records.append(
                {
                    "turn": turn + 1,
                    "pass": result["pass"],
                    "strategy": strategy,
                    "verification": result,
                    "agent_trajectories": [
                        trajectory.model_dump()
                        for trajectory in solution.agent_trajectories
                    ],
                }
            )
            if result["pass"]:
                break
            feedback_history.append(result["reason"])
            print(f"Feedback for next turn: {result['reason']}")
        table.append(trial_results)
        records.append({"trial": trial + 1, "turns": turn_records})
        print()
    payload = {
        "problem_id": problem_id,
        "trials": trials,
        "max_refinement_turns": max_refinement_turns,
        "pass_table": table,
        "records": records,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {output_path}")
    return table


if __name__ == "__main__":
    run_refinement()
