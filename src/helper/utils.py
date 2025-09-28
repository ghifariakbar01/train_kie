import json
import random

# src/utils_training.py
import os, json
from typing import Optional, Tuple
from datasets import DatasetDict
from transformers import DefaultDataCollator

def load_label_maps(path: str) -> Tuple[dict, dict]:
    """Load label2id/id2label; fallback to label_list if explicit maps missing."""
    with open(path, "r", encoding="utf-8") as f:
        info = json.load(f)

    label2id = {str(k): int(v) for k, v in info.get("label2id", {}).items()} or {
        l: i for i, l in enumerate(info["label_list"])
    }
    id2label = {int(k): v for k, v in info.get("id2label", {}).items()} or {
        i: l for l, i in label2id.items()
    }
    return label2id, id2label

def ensure_splits(ds):
    """Return (train, val, test?) from a saved HF dataset."""
    if isinstance(ds, DatasetDict):
        if "train" in ds and "validation" in ds:
            return ds["train"], ds["validation"], ds.get("test")
        if "train" in ds and "valid" in ds:
            return ds["train"], ds["valid"], ds.get("test")
        if "train" in ds and len(ds) == 1:
            tmp = ds["train"].train_test_split(test_size=0.1, seed=42)
            return tmp["train"], tmp["test"], None
        name = next(iter(ds.keys()))
        tmp = ds[name].train_test_split(test_size=0.1, seed=42)
        return tmp["train"], tmp["test"], None
    tmp = ds.train_test_split(test_size=0.1, seed=42)
    return tmp["train"], tmp["test"], None

def maybe_build_collators(noise_config_path: Optional[str], model_name: str):
    """
    If a noise config is provided, build (train_collator with noise, eval_collator without).
    Otherwise return (None, DefaultDataCollator) to keep training simple.
    """
    if not noise_config_path:
        return None, DefaultDataCollator(return_tensors="pt")

    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)

        with open(noise_config_path, "r", encoding="utf-8") as f:
            aug_cfg = json.load(f)

        try:
            from augment.collator import DataCollatorForLayoutLMv3Noise
        except Exception:
            from augment.collator import DataCollatorForLayoutLMv3Noise  # fallback import

        train_collator = DataCollatorForLayoutLMv3Noise.from_config(processor, aug_cfg)

        eval_cfg = dict(aug_cfg)
        eval_cfg.update({
            "bbox_jitter_prob": 0.0,
            "token_dropout_prob": 0.0,
            "max_tokens_dropped": 0
        })
        eval_collator = DataCollatorForLayoutLMv3Noise.from_config(processor, eval_cfg)

        return train_collator, eval_collator

    except Exception as e:
        print(f"[WARN] Failed to build noise collators from {noise_config_path}: {e}")
        return None, DefaultDataCollator(return_tensors="pt")
    
    