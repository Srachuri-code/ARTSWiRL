import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
import boto3

from .dataset import SwirlDataset
from .trainer import collate_fn  # reuse batching + loss


def load_lora(base_model: str):
    """Load base model with LoRA.

    Try 4-bit QLoRA when a new enough bitsandbytes is available; otherwise
    fall back to float16.  This avoids the strict >=0.43.0 requirement that
    fails on macOS/CPU boxes while still giving memory savings when
    possible.
    """

    from importlib import metadata
    use_4bit = False
    try:
        bnb_version = metadata.version("bitsandbytes")
        from packaging import version
        if version.parse(bnb_version) >= version.parse("0.43.0"):
            use_4bit = True
    except metadata.PackageNotFoundError:
        pass

    if use_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_cfg)
        print("Loaded model in 4-bit (bitsandbytes).")
    else:
        print("bitsandbytes >=0.43 not available – loading model in float16.")
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    return model, tokenizer


def train(args):
    model, tokenizer = load_lora(args.base_model)
    device = torch.device(args.device)
    model.to(device)

    dataset = SwirlDataset(args.dataset, tokenizer_name=args.base_model)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=args.epochs * len(loader)
    )

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            rewards = batch["reward"].to(device)

            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            token_nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.size())
            token_mask = (shift_labels != -100).float()
            log_probs = -(token_nll * token_mask).sum(dim=1)

            baseline = rewards.mean()
            advantage = rewards - baseline
            loss = -(advantage * log_probs).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs}  loss={total_loss/len(loader):.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    bucket = os.getenv("BACKUP_BUCKET")
    if bucket:
        print(f"Uploading → s3://{bucket}/{out_dir.name}/ …")
        s3 = boto3.client("s3")
        for fp in out_dir.rglob("*"):
            if fp.is_file():
                s3.upload_file(str(fp), bucket, f"{out_dir.name}/{fp.relative_to(out_dir)}")


def _parse():
    p = argparse.ArgumentParser("SWiRL trainer – PEFT LoRA 4-bit")
    p.add_argument("--dataset", required=True)
    p.add_argument("--base-model", default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--output-dir", default="models/swirl-3b-lora")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    train(args) 