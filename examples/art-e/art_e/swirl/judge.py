import asyncio
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

import litellm
from litellm import acompletion
from litellm.caching.caching import Cache, LiteLLMCacheType

# Enable disk cache so repeated scoring is cheap
litellm.cache = Cache(type=LiteLLMCacheType.DISK)

PROMPT = (
    "You are an expert tutor. A student is solving a problem step-by-step. "
    "Judge ONLY the CURRENT step for clarity, correctness, and usefulness toward answering the original question. "
    "Return a single float from –1 (very poor) to 1 (excellent). Do NOT judge previous or future steps."
)

FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")


async def _score_step(model_name: str, question: str, step: str) -> float:
    messages = [
        {"role": "system", "content": PROMPT},
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n---\nStep by student:\n{step}\n\n"
                "Respond with a single float in [-1, 1]."
            ),
        },
    ]
    resp = await acompletion(model=model_name, messages=messages, temperature=0, max_tokens=4, caching=True)
    txt = resp.choices[0].message.content  # type: ignore
    if txt is None:
        return 0.0
    m = FLOAT_RE.search(txt)
    try:
        if m:
            val = float(m.group())
            return max(-1.0, min(1.0, val))
    except ValueError:
        pass
    return 0.0


async def _process_traj(traj_json: str, model_name: str) -> str:
    traj: Dict[str, Any] = json.loads(traj_json)
    messages: List[Any] = traj["messages_and_choices"]

    # Extract original question (first user message)
    question = ""
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            question = m.get("content", "")
            break

    assistant_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]

    if not assistant_msgs:
        traj["reward"] = 0.0
        return json.dumps(traj, ensure_ascii=False)

    # Score each assistant step
    scores = []
    for a in assistant_msgs:
        step_text = a.get("content", "")
        score = await _score_step(model_name, question, step_text)
        scores.append(score)

    # Trajectory reward = mean of step rewards (paper §2.2)
    traj_reward = sum(scores) / len(scores)
    traj["reward"] = traj_reward
    traj["step_rewards"] = scores
    return json.dumps(traj, ensure_ascii=False)


async def main_async(args: argparse.Namespace):
    load_dotenv()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = in_path.read_text().splitlines()
    print(f"Scoring {len(lines)} trajectories with model={args.model} …")

    out_lines: List[str] = []
    # Process sequentially to keep rate-limit simple (easy to add concurrency)
    for i, line in enumerate(lines, 1):
        scored = await _process_traj(line, args.model)
        out_lines.append(scored)
        if i % 50 == 0:
            print(f"… {i}/{len(lines)} done")

    out_path.write_text("\n".join(out_lines))
    print(f"Wrote {len(out_lines)} scored trajectories → {out_path}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser("Generative judge – assign reward to each trajectory")
    p.add_argument("--input", required=True, help="raw_trajectories.jsonl path")
    p.add_argument("--output", required=True, help="Where to write scored trajectories")
    p.add_argument("--model", default="openai/gpt-4o", help="litellm model identifier to use as judge")
    return p.parse_args()


def main():
    args = _parse()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main() 