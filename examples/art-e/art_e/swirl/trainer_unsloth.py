import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig
from unsloth import FastLanguageModel
import bitsandbytes as bnb
import boto3

from .dataset import SwirlDataset
from .trainer import collate_fn  # reuse existing batching logic


def train(args):
    # ------------------------------------------------------------------
    # 1. Load base model in 4-bit with Unsloth + add LoRA adapters
    # ------------------------------------------------------------------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        dtype="float16",
        load_in_4bit=True,
        use_flash_attention=False,  # avoid xformers build on macOS
    )

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = FastLanguageModel.get_peft_model(model, lora_cfg)

    dataset = SwirlDataset(args.dataset, tokenizer_name=args.base_model)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # 8-bit Adam optimiser from bitsandbytes
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=args.epochs * len(loader)
    )

    model.train()
    device = torch.device(args.device)
    model.to(device)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            rewards = batch["reward"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids)
            logits = outputs.logits  # (B, T, V)

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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs}  loss={total_loss/len(loader):.4f}")

    # ------------------------------------------------------------------
    # Save LoRA adapters + tokenizer
    # ------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Optional S3 upload (same as original)
    bucket = os.getenv("BACKUP_BUCKET")
    if bucket:
        s3_prefix = f"{out_dir.name}/"
        print(f"Uploading {out_dir} → s3://{bucket}/{s3_prefix} …")
        s3 = boto3.client("s3")
        for file_path in out_dir.rglob("*"):
            if file_path.is_file():
                key = f"{s3_prefix}{file_path.relative_to(out_dir).as_posix()}"
                s3.upload_file(str(file_path), bucket, key)
        print("S3 upload complete.")
    else:
        print("BACKUP_BUCKET not set – skipping S3 upload.")


def _parse():
    p = argparse.ArgumentParser("SWiRL trainer (Unsloth/LoRA)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--base-model", default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--output-dir", default="models/swirl-3b-lora")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    train(args) 