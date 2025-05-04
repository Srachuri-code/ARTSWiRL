import asyncio
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

import art
from dotenv import load_dotenv

from art_e.data.query_iterators import load_synthetic_queries
from art_e.rollout import rollout  # Uses same environment
from art_e.project_types import ProjectPolicyConfig

# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect full trajectories for SWiRL offline training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Teacher model to use when generating trajectories.",
    )
    parser.add_argument(
        "--litellm-name",
        type=str,
        default="openai/gpt-4o",
        help="litellm model identifier (if different from --model)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/swirl/raw_trajectories.jsonl",
        help="Path of JSONL file to write",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of scenarios to collect",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="If set, disables OpenAI function-calling; model must return a JSON object instead. Useful when endpoints strip tool calls.",
    )
    return parser.parse_args()


async def _collect(args: argparse.Namespace) -> None:
    load_dotenv()

    # Create output dir
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize (prompted) teacher model – not trainable
    teacher = art.Model(
        name=args.model,
        project="email_agent",
        config=ProjectPolicyConfig(
            litellm_model_name=args.litellm_name,
            use_tools=not args.no_tools,
        ),
    )
    # Mark as non-trainable so rollout knows not to treat it as a fine-tunable ART model.
    # We have to bypass Pydantic's attribute validation.
    object.__setattr__(teacher, "trainable", False)

    scenarios = load_synthetic_queries(split="train", limit=args.limit)

    print(f"Collecting {len(scenarios)} trajectories with model={args.model} …")

    # Gather trajectories concurrently via art utility
    trajs = await art.gather_trajectories(
        (rollout(teacher, s) for s in scenarios),
        pbar_desc="collect",
    )

    # Write to JSONL
    with out_path.open("w") as f:
        for t in trajs:
            if not isinstance(t, art.Trajectory):
                continue
            # Ensure we store every assistant step (messages_and_choices already contains them)
            record = t.model_dump()
            record["collected_at"] = datetime.utcnow().isoformat()
            json.dump(record, f)
            f.write("\n")

    print(f"Wrote {len(trajs)} trajectories → {out_path}")


def main():
    args = _parse_args()
    asyncio.run(_collect(args))


if __name__ == "__main__":
    main() 