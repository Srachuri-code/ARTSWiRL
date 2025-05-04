import argparse
import json
from pathlib import Path
import os
import boto3

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, BitsAndBytesConfig
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model

from .dataset import SwirlDataset


def collate_fn(batch):
    """Prepare padded batch for step-wise RL.

    For each record we have prefix_ids (the context) and action_ids (the
    sub-step the model should generate).  We concatenate them into one
    sequence that the model will process autoregressively.  Tokens coming
    from the prefix are masked out in *labels* so the loss / policy‐grad
    signal only applies to the action tokens.
    """

    seqs = []
    lbls = []
    for rec in batch:
        prefix = rec["prefix_ids"]
        action = rec["action_ids"]

        seq = torch.cat([prefix, action])
        label = torch.cat([torch.full_like(prefix, -100), action])

        seqs.append(seq)
        lbls.append(label)

    input_ids = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(lbls, batch_first=True, padding_value=-100)
    rewards = torch.stack([rec["reward"] for rec in batch])

    return {
        "input_ids": input_ids,
        "labels": labels,
        "reward": rewards,
    }


def train(args):
    model, tokenizer = load_lora(args.base_model)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    dataset = SwirlDataset(args.dataset, tokenizer_name=args.base_model)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=args.epochs*len(loader))

    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].to(args.device)
            rewards = batch["reward"].to(args.device)

            # Forward pass with logits (we need per-token log-probs)
            outputs = model(input_ids=input_ids)  # labels handled manually
            logits = outputs.logits  # (B, T, V)

            # Shift so that tokens <t> predict token t
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Mask ignores prefix tokens (label == -100)
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            token_nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            token_nll = token_nll.view(shift_labels.size())  # (B, T-1)

            # Only consider tokens where label != -100 (action tokens)
            token_mask = (shift_labels != -100).float()
            log_probs_per_example = -(token_nll * token_mask).sum(dim=1)  # sum log-probs of action tokens

            # Advantage (baseline is batch mean)
            baseline = rewards.mean()
            advantage = rewards - baseline

            loss = -(advantage * log_probs_per_example).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs}   loss={total_loss/len(loader):.4f}")
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)

    # --------------------------------------------------------
    # Optional: upload results to an S3 bucket (same convention
    # the GRPO training script uses).  Requires that the
    # environment variable BACKUP_BUCKET is set and that the
    # credentials in the environment have write permission.
    # --------------------------------------------------------
    bucket = os.getenv("BACKUP_BUCKET")
    if bucket:
        s3_prefix = f"{out_path.name}/"  # folder in bucket matches model dir
        print(f"Uploading {out_path} → s3://{bucket}/{s3_prefix} …")

        s3 = boto3.client("s3")
        for file_path in out_path.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(out_path).as_posix()
                key = f"{s3_prefix}{rel_path}"
                s3.upload_file(str(file_path), bucket, key)
        print("S3 upload complete.")
    else:
        print("BACKUP_BUCKET not set – skipping S3 upload.")


def _parse():
    parser = argparse.ArgumentParser("SWiRL trainer")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--output-dir", type=str, default="models/swirl-agent")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse()
    train(args) 