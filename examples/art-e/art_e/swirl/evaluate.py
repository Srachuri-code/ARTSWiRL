import asyncio
import argparse
import json
from pathlib import Path
import time

import torch
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from art.utils import limit_concurrency

from art_e.data.query_iterators import load_synthetic_queries
from art_e.rollout import rollout, FinalRubric  # Use same environment and rubric
from art_e.project_types import ProjectPolicyConfig

# Reuse the PEFT loader logic, slightly modified to apply existing adapters
from .trainer_peft import load_lora  # Assuming this loads base + applies LoRA


# --- Evaluation Logic ---

@limit_concurrency(10) # Limit concurrent rollouts
async def evaluate_scenario(model, scenario):
    """Runs rollout for one scenario and returns the trajectory."""
    try:
        # Ensure model config matches rollout expectations (no tools for SWiRL)
        # If load_lora doesn't set config, we might need to manually create it
        if not hasattr(model, 'config') or not isinstance(model.config, ProjectPolicyConfig):
             # Create a default config matching our training setup
             model.config = ProjectPolicyConfig(
                 use_tools=False, # Critical for SWiRL evaluation consistency
                 max_turns=5,     # Match rollout limits if needed
                 max_tokens=512, # Match rollout limits if needed
                 # Add other necessary fields if load_lora doesn't provide them
             )
             # Add other necessary attributes if load_lora doesn't provide them
             if not hasattr(model, 'inference_base_url'): model.inference_base_url = None
             if not hasattr(model, 'inference_api_key'): model.inference_api_key = None
             if not hasattr(model, 'trainable'): model.trainable = False # Evaluation model isn't training


        traj = await rollout(model=model, scenario=scenario)
        return traj
    except Exception as e:
        print(f"Error during rollout for scenario {scenario.id}: {e}")
        # Return a dummy trajectory or rubric indicating failure
        rubric = FinalRubric(answer_correct=False) # Mark as incorrect on error
        # You might want a more detailed dummy trajectory if needed
        return type('obj', (object,), {'metrics': rubric.to_metrics(), 'reward': -2.0, 'logs': [f"Rollout failed: {e}"]})


async def run_evaluation(args: argparse.Namespace):
    """Loads model, runs evaluation, prints results."""
    print(f"Loading base model '{args.base_model}' and applying LoRA from '{args.lora_path}'...")
    # Modify load_lora or create a new function if needed to handle applying adapters
    # This assumes load_lora can take the adapter path directly or we modify it
    # For now, assuming load_lora returns a model ready for inference after applying PEFT
    # We pass the lora_path to load_lora
    model, tokenizer = load_lora(args.base_model, args.lora_path)
    model = model.to(args.device)
    model.eval() # Set model to evaluation mode

    print(f"Loading evaluation data (split='{args.split}', limit={args.limit})...")
    scenarios = load_synthetic_queries(split=args.split, limit=args.limit)

    print(f"Running evaluation on {len(scenarios)} scenarios...")
    start_time = time.time()

    tasks = [evaluate_scenario(model, s) for s in scenarios]
    results = await tqdm_asyncio.gather(*tasks)

    end_time = time.time()
    print(f"Evaluation finished in {end_time - start_time:.2f} seconds.")

    # --- Aggregate results ---
    total_scenarios = len(results)
    correct_answers = 0
    total_reward = 0
    rollout_failures = 0

    output_trajectories = []

    for i, traj in enumerate(results):
        # Handle potential dummy trajectory from errors
        if isinstance(traj, dict) and 'metrics' in traj: # Basic check for our dummy error object
             metrics = traj['metrics']
             reward = traj['reward']
             output_trajectories.append({
                 "scenario_id": scenarios[i].id,
                 "question": scenarios[i].question,
                 "correct_answer_expected": scenarios[i].answer,
                 "generated_answer": "ROLLOUT_FAILED",
                 "reward": reward,
                 "metrics": metrics,
                 "logs": traj.get('logs', [])
             })
             rollout_failures += 1
        elif hasattr(traj, 'metrics') and isinstance(traj.metrics, dict):
             metrics = traj.metrics
             reward = traj.reward
             correct_answers += int(metrics.get('answer_correct', 0))
             total_reward += reward
             # Append full trajectory details if needed
             # For brevity, just storing key info now
             output_trajectories.append({
                 "scenario_id": scenarios[i].id,
                 "question": scenarios[i].question,
                 "correct_answer_expected": scenarios[i].answer,
                 "generated_answer": traj.messages_and_choices[-1]['content'] if traj.messages_and_choices else "NO_ANSWER", # Approximate last message
                 "reward": reward,
                 "metrics": metrics,
                 "logs": getattr(traj, 'logs', [])
            })
        else:
            print(f"Warning: Unexpected result type for scenario {i}: {type(traj)}")
            rollout_failures +=1


    accuracy = (correct_answers / (total_scenarios - rollout_failures)) * 100 if (total_scenarios - rollout_failures) > 0 else 0
    avg_reward = (total_reward / (total_scenarios - rollout_failures)) if (total_scenarios - rollout_failures) > 0 else 0

    print("--- Evaluation Results ---")
    print(f"Total Scenarios: {total_scenarios}")
    print(f"Successful Rollouts: {total_scenarios - rollout_failures}")
    print(f"Rollout Failures: {rollout_failures}")
    print(f"Correct Answers: {correct_answers}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average Reward: {avg_reward:.4f}")

    # Save detailed results
    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w') as f:
            for entry in output_trajectories:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        print(f"Saved detailed evaluation results to {out_path}")


# --- CLI ---

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned SWiRL model.")
    parser.add_argument("--base-model", type=str, required=True, help="Path or HF ID of the base model.")
    parser.add_argument("--lora-path", type=str, required=True, help="Path to the trained LoRA adapter directory.")
    parser.add_argument("--split", type=str, default="test", help="Data split to evaluate on (e.g., 'test', 'train').")
    parser.add_argument("--limit", type=int, default=100, help="Number of scenarios to evaluate.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on (cuda/cpu).")
    parser.add_argument("--output-file", type=str, default="evaluation_results.jsonl", help="Optional path to save detailed trajectory results.")
    # Add other relevant args like batch size if needed for concurrency later
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_evaluation(args)) 