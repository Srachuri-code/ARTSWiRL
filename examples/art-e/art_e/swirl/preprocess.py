import json
import argparse
from pathlib import Path
from typing import Any, Dict, List


def _assistant_messages(messages_and_choices: List[Any]) -> List[Dict[str, Any]]:
    """Return list of message dicts that were produced by the agent."""
    assistant_msgs: List[Dict[str, Any]] = []
    for m in messages_and_choices:
        # Trajectory stores either raw dict (for prompted models) or dict-like choice objects
        if isinstance(m, dict):
            if m.get("role") == "assistant":
                assistant_msgs.append(m)
        else:
            # Fallback: not expected here – ignore
            pass
    return assistant_msgs


def explode_to_substeps(traj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a full trajectory into SWiRL sub-step records.

    Each sub-step contains:
        prefix : str  – JSON string of messages up to (but not incl.) current assistant
        action : str  – JSON string of the assistant message
        reward : float – final trajectory reward
    """
    sub: List[Dict[str, Any]] = []
    messages: List[Any] = traj["messages_and_choices"]
    reward: float = traj.get("reward", 0.0)

    assistant_indices = [i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "assistant"]

    for idx in assistant_indices:
        prefix_msgs = messages[:idx]  # everything before current assistant
        action_msg = messages[idx]
        record = {
            "prefix": json.dumps(prefix_msgs, ensure_ascii=False),
            "action": json.dumps(action_msg, ensure_ascii=False),
            "reward": reward,
        }
        sub.append(record)
    return sub


def main():
    parser = argparse.ArgumentParser(description="Convert raw trajectories → SWiRL dataset")
    parser.add_argument("--input", type=str, required=True, help="Path to raw_trajectories.jsonl")
    parser.add_argument(
        "--output", type=str, default="data/swirl/substeps.jsonl", help="Output JSONL file"
    )
    parser.add_argument(
        "--reward-threshold",
        type=float,
        default=None,
        help="If set, keep only trajectories with reward >= threshold",
    )
    parser.add_argument(
        "--exclude-final-answer",
        action="store_true",
        help="Skip sub-steps whose action is a return_final_answer tool call (process-only dataset).",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept, total, sub_count = 0, 0, 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            total += 1
            traj = json.loads(line)
            if args.reward_threshold is not None and traj.get("reward", 0.0) < args.reward_threshold:
                continue
            kept += 1
            for rec in explode_to_substeps(traj):
                if args.exclude_final_answer and "return_final_answer" in rec["action"]:
                    continue
                json.dump(rec, fout, ensure_ascii=False)
                fout.write("\n")
                sub_count += 1

    print(f"Read {total} trajectories, kept {kept}. Wrote {sub_count} sub-steps → {out_path}")


if __name__ == "__main__":
    main() 