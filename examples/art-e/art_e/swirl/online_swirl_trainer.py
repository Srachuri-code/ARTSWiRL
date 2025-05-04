"""
Online Trainer for SWiRL using ART-E components.

This script integrates online trajectory generation (rollouts)
with the SWiRL policy gradient update step.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader # May not be needed if batching manually
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, PreTrainedTokenizerBase
from peft import PeftModel # Assuming PEFT model usage
from tqdm.auto import tqdm
import nest_asyncio

# ART-E specific imports (adjust paths based on execution context if needed)
# Assuming execution from the repository root or proper PYTHONPATH setup
try:
    from art_e.dataset import load_synthetic_queries, SyntheticQuery
    from art_e.rollout import rollout # Key component for online generation
    from art_e.swirl.trainer_peft import load_lora # Reuse PEFT loading utility
    from art_e.types import Trajectory # Rollout returns Trajectory objects
except ImportError:
    print("Error: Make sure ART-E components are importable. Run from repo root or check PYTHONPATH.")
    exit(1)

nest_asyncio.apply() # Needed for running async rollout within sync loop potentially


# --- SWiRL Helper Functions (Adapted for Online Use) ---

def _explode_trajectory_to_swirl_substeps(
    traj: Trajectory, tokenizer: PreTrainedTokenizerBase
) -> List[Dict[str, Any]]:
    """Convert a full trajectory into SWiRL sub-step records with tokenized IDs.

    Each sub-step contains:
        prefix_ids : torch.Tensor – Token IDs of messages up to (but not incl.) current assistant msg
        action_ids : torch.Tensor – Token IDs of the assistant message
        reward : float – final trajectory reward
    """
    sub_records_tokenized: List[Dict[str, Any]] = []
    # Use traj.messages_and_choices directly which should contain dicts or Choice objects
    messages: List[Any] = []
    for item in traj.messages_and_choices:
         # Handle Choice objects if present, otherwise assume dict
         # We need the dictionary representation for apply_chat_template
         if hasattr(item, 'message') and hasattr(item.message, 'model_dump'):
             messages.append(item.message.model_dump()) # Use pydantic model_dump if available
         elif isinstance(item, dict):
             messages.append(item)
         # Ignore other types for simplicity in this adaptation

    reward: float = traj.reward

    assistant_indices = [i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "assistant"]

    for idx in assistant_indices:
        prefix_msgs = messages[:idx]  # Messages before current assistant
        action_msg = messages[idx]    # The assistant message itself

        # Tokenize prefix and action here
        try:
            # Create text representation for prefix and action
            # Use add_generation_prompt=False as we manually structure prefix/action
            prefix_text = tokenizer.apply_chat_template(prefix_msgs, tokenize=False, add_generation_prompt=False) if prefix_msgs else ""
            action_text = tokenizer.apply_chat_template([action_msg], tokenize=False, add_generation_prompt=False)

            # Tokenize - keep BOS for prefix, remove for action
            prefix_ids = tokenizer(prefix_text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length // 2).input_ids.squeeze(0)
            action_ids = tokenizer(action_text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length // 2).input_ids.squeeze(0)

            # Remove BOS token from action if present and action is not empty
            if action_ids.numel() > 0 and action_ids[0] == tokenizer.bos_token_id:
                 action_ids = action_ids[1:]

            # Skip if action becomes empty after removing BOS, or if prefix/action are too long (implicit truncation)
            if action_ids.numel() == 0:
                 continue

            record = {
                "prefix_ids": prefix_ids,
                "action_ids": action_ids,
                "reward": reward,
            }
            sub_records_tokenized.append(record)
        except Exception as e:
            print(f"Warning: Skipping substep due to tokenization/processing error: {e}")
            print(f"  Prefix Msgs: {prefix_msgs}")
            print(f"  Action Msg: {action_msg}")
            continue # Skip problematic substeps

    return sub_records_tokenized


def _collate_swirl_substeps(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Prepare padded batch for SWiRL step-wise RL.

    Assumes batch contains dicts with 'prefix_ids', 'action_ids', 'reward'.
    Input tensors are expected to be 1D.
    """
    seqs = []
    lbls = []
    rewards_list = [] # Keep rewards separate initially

    for rec in batch:
        prefix = rec["prefix_ids"] # Should already be a tensor
        action = rec["action_ids"] # Should already be a tensor

        # Basic validation
        if not isinstance(prefix, torch.Tensor) or not isinstance(action, torch.Tensor):
             print(f"Warning: Skipping record in collate due to non-tensor input: {type(prefix)}, {type(action)}")
             continue
        if prefix.dim() > 1 or action.dim() > 1:
             print(f"Warning: Skipping record in collate due to unexpected dimensions: {prefix.shape}, {action.shape}")
             continue


        # Handle empty prefix case (first turn)
        if prefix.numel() == 0:
             seq = action
             label = action # No prefix to mask
        else:
            # Ensure prefix and action are 1D before concatenating
            if prefix.dim() == 0: prefix = prefix.unsqueeze(0)
            if action.dim() == 0: action = action.unsqueeze(0)

            seq = torch.cat([prefix, action])
            label = torch.cat([torch.full_like(prefix, -100), action])

        seqs.append(seq)
        lbls.append(label)
        rewards_list.append(rec["reward"]) # Collect rewards

    # Padding (use 0 for input_ids, -100 for labels)
    if not seqs:
        return {"input_ids": torch.empty((0,0), dtype=torch.long),
                "labels": torch.empty((0,0), dtype=torch.long),
                "reward": torch.empty((0,), dtype=torch.float)}

    try:
        input_ids = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
        labels = torch.nn.utils.rnn.pad_sequence(lbls, batch_first=True, padding_value=-100)
        rewards = torch.tensor(rewards_list, dtype=torch.float)
    except RuntimeError as e:
         print(f"Error during padding: {e}")
         # Print shapes for debugging
         for i, s in enumerate(seqs): print(f" Seq {i} shape: {s.shape}")
         for i, l in enumerate(lbls): print(f" Lbl {i} shape: {l.shape}")
         # Return empty batch on error
         return {"input_ids": torch.empty((0,0), dtype=torch.long),
                 "labels": torch.empty((0,0), dtype=torch.long),
                 "reward": torch.empty((0,), dtype=torch.float)}


    return {"input_ids": input_ids, "labels": labels, "reward": rewards}

# --- End SWiRL Helper Functions ---


async def online_train(args):
    print("Starting Online SWiRL Training...")
    device = torch.device(args.device)

    # 1. Load Model and Tokenizer
    print(f"Loading base model '{args.base_model}' with LoRA...")
    # load_lora handles PEFT setup and potentially quantization
    model, tokenizer, _ = load_lora(args.base_model)
    model.to(device) # Ensure model is on the correct device after loading
    print("Model and tokenizer loaded.")

    if tokenizer.pad_token is None:
        print("Warning: Tokenizer does not have a pad token. Setting to EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
        # Update model config if needed
        model.config.pad_token_id = tokenizer.eos_token_id


    # 2. Load Scenarios (Data Source for Rollouts)
    print(f"Loading scenarios (split='{args.data_split}', limit={args.num_scenarios_per_step})...")
    # We load scenarios once, and sample from them in the loop
    # Alternatively, load a larger set and iterate through it.
    # Using limit=None to load all available training scenarios
    all_scenarios: List[SyntheticQuery] = load_synthetic_queries(split=args.data_split, limit=None)
    if not all_scenarios:
        print(f"Error: No scenarios found for split '{args.data_split}'. Exiting.")
        return
    print(f"Loaded {len(all_scenarios)} scenarios.")

    # 3. Initialize Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    # Calculate total steps for scheduler (approximate)
    num_training_steps = args.training_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
    )

    # 4. Online Training Loop
    model.train() # Set model to training mode
    global_step = 0
    progress_bar = tqdm(total=num_training_steps, desc="Training Steps")

    while global_step < num_training_steps:

        # --- Rollout Phase ---
        print(f"\nStep {global_step + 1}/{num_training_steps}: Starting rollout phase...")
        rollout_tasks = []
        # Sample scenarios for this step's rollouts
        # Ensure we don't sample more scenarios than available
        num_to_sample = min(args.num_scenarios_per_step, len(all_scenarios))
        selected_scenario_indices = torch.randperm(len(all_scenarios))[:num_to_sample]
        selected_scenarios = [all_scenarios[i] for i in selected_scenario_indices]

        print(f"Generating {args.rollouts_per_scenario * len(selected_scenarios)} trajectories...")
        for scenario in selected_scenarios:
            for _ in range(args.rollouts_per_scenario):
                # Pass the current model state to the rollout function
                # Ensure rollout uses model.generate or similar internally
                rollout_tasks.append(rollout(model=model, scenario=scenario)) # Async call

        # Gather completed trajectories
        completed_trajectories_or_errors = await asyncio.gather(*rollout_tasks, return_exceptions=True)

        # Filter out successful trajectories
        valid_trajectories: List[Trajectory] = []
        num_errors = 0
        for result in completed_trajectories_or_errors:
             if isinstance(result, Trajectory):
                 valid_trajectories.append(result)
             else:
                 print(f"Rollout Error: {result}")
                 num_errors += 1
        print(f"Rollout phase complete. Got {len(valid_trajectories)} valid trajectories, {num_errors} errors.")

        if not valid_trajectories:
            print("Warning: No valid trajectories generated in this step. Skipping update.")
            global_step += 1 # Increment step even if no update occurs
            progress_bar.update(1)
            continue

        # --- Processing Phase ---
        print("Processing trajectories into SWiRL substeps...")
        all_substeps_for_update: List[Dict[str, Any]] = []
        for traj in valid_trajectories:
            substeps = _explode_trajectory_to_swirl_substeps(traj, tokenizer)
            all_substeps_for_update.extend(substeps)

        if not all_substeps_for_update:
            print("Warning: No valid SWiRL substeps generated from trajectories. Skipping update.")
            global_step += 1
            progress_bar.update(1)
            continue

        print(f"Generated {len(all_substeps_for_update)} substeps for update.")

        # --- Update Phase ---
        print("Performing SWiRL update...")
        # Collate all substeps into a single batch for this update
        # TODO: Implement mini-batching within this large batch if needed (memory constraints)
        update_batch = _collate_swirl_substeps(all_substeps_for_update)

        if update_batch["input_ids"].numel() == 0:
             print("Warning: Collated batch is empty. Skipping update.")
             global_step += 1
             progress_bar.update(1)
             continue

        input_ids = update_batch["input_ids"].to(device)
        labels = update_batch["labels"].to(device)
        rewards = update_batch["reward"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass to get logits
        # Use autocast for mixed precision if applicable
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
             outputs = model(input_ids=input_ids)
             logits = outputs.logits

        # SWiRL Loss Calculation (adapted from trainer.py)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        token_nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_nll = token_nll.view(shift_labels.size())

        token_mask = (shift_labels != -100).float()
        log_probs_per_example = -(token_nll * token_mask).sum(dim=1)

        baseline = rewards.mean()
        advantage = rewards - baseline

        loss = -(advantage * log_probs_per_example).mean()

        # Backward pass and optimizer step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        print(f"Update complete. Loss: {loss.item():.4f}")
        tqdm.write(f"Step: {global_step+1}, Loss: {loss.item():.4f}, Avg Reward: {rewards.mean().item():.2f}") # Use tqdm.write for clean output

        # --- Step Increment ---
        global_step += 1
        progress_bar.update(1)

        # Optional: Save checkpoint periodically
        if global_step % args.save_steps == 0:
            print(f"\nSaving checkpoint at step {global_step}...")
            ckpt_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"Checkpoint saved to {ckpt_dir}")
            model.train() # Ensure model stays in train mode after saving


    # 5. Save Final Model
    progress_bar.close()
    print("\nTraining finished. Saving final model...")
    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Final model saved to {final_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Online SWiRL Trainer")
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Base model ID from Hugging Face.")
    parser.add_argument("--output-dir", type=str, default="models/online-swirl-agent", help="Directory to save trained model checkpoints and final model.")
    parser.add_argument("--data-split", type=str, default="train", help="Data split for loading scenarios (e.g., 'train', 'test').")
    parser.add_argument("--training-steps", type=int, default=1000, help="Total number of training steps (rollout+update cycles)." )
    parser.add_argument("--num-scenarios-per-step", type=int, default=4, help="Number of scenarios to sample for rollouts in each step.")
    parser.add_argument("--rollouts-per-scenario", type=int, default=2, help="Number of rollouts to generate per selected scenario.")
    # Training Hyperparameters
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Number of warmup steps for the scheduler.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping value.")
    parser.add_argument("--save-steps", type=int, default=100, help="Save checkpoint every N steps.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training (cuda/cpu).")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(online_train(args)) 