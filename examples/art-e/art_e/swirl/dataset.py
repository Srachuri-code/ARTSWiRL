import json
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class SwirlDataset(Dataset):
    """Iterable of sub-steps for SWiRL."""

    def __init__(self, path: str | Path, tokenizer_name: str):
        self.records: List[Dict] = [json.loads(l) for l in Path(path).read_text().splitlines()]
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        prefix_ids = self.tokenizer(rec["prefix"], truncation=True, padding=False, return_tensors="pt").input_ids[0]
        action_ids = self.tokenizer(rec["action"], truncation=True, padding=False, return_tensors="pt").input_ids[0]
        reward = torch.tensor(rec["reward"], dtype=torch.float)
        return {
            "prefix_ids": prefix_ids,
            "action_ids": action_ids,
            "reward": reward,
        } 