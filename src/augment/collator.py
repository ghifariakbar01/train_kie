# src/augment/collator.py
from typing import List, Dict, Any, Optional
import random
import torch
from transformers import PreTrainedTokenizerBase
from .augmentation import jitter_bboxes, random_token_dropout

class DataCollatorForLayoutNoise:
    def __init__(
        self,
        processor: Optional[Any] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        bbox_jitter_prob: float = 0.5,
        bbox_jitter_scale: float = 0.02,
        token_dropout_prob: float = 0.05,
        max_tokens_dropped: int = 3,
        label_mask_value: int = -100,
        seed: int = 42,
    ):
        # accept either a tokenizer directly or processor.tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif processor is not None and hasattr(processor, "tokenizer"):
            self.tokenizer = processor.tokenizer
        else:
            raise ValueError("Collator needs a tokenizer or a processor with .tokenizer")

        self.processor = processor
        self.bbox_jitter_prob = bbox_jitter_prob
        self.bbox_jitter_scale = bbox_jitter_scale
        self.token_dropout_prob = token_dropout_prob
        self.max_tokens_dropped = max_tokens_dropped
        self.label_mask_value = label_mask_value
        self.rng = random.Random(seed)

    @classmethod
    def from_config(cls, processor=None, tokenizer=None, cfg: Dict[str, Any] = None):
        cfg = cfg or {}
        return cls(
            processor=processor,
            tokenizer=tokenizer,
            bbox_jitter_prob=cfg.get("bbox_jitter_prob", 0.5),
            bbox_jitter_scale=cfg.get("bbox_jitter_scale", 0.02),
            token_dropout_prob=cfg.get("token_dropout_prob", 0.05),
            max_tokens_dropped=cfg.get("max_tokens_dropped", 3),
            label_mask_value=cfg.get("label_mask_value", -100),
            seed=cfg.get("seed", 42),
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        proc = []
        any_pixel_values = any("pixel_values" in ex for ex in features)

        for ex in features:
            input_ids = ex["input_ids"].tolist() if isinstance(ex["input_ids"], torch.Tensor) else ex["input_ids"]
            labels = ex.get("labels")
            if isinstance(labels, torch.Tensor): labels = labels.tolist()

            bboxes = ex.get("bbox")
            if isinstance(bboxes, torch.Tensor): bboxes = bboxes.tolist()

            # jitter in 0–1000 space
            if bboxes is not None and self.bbox_jitter_prob > 0:
                bboxes = jitter_bboxes(bboxes, self.bbox_jitter_scale, self.rng, self.bbox_jitter_prob)

            # token dropout
            if input_ids is not None and labels is not None and self.token_dropout_prob > 0:
                mask_id = getattr(self.tokenizer, "mask_token_id", None)
                input_ids, labels = random_token_dropout(
                    input_ids, labels, mask_id, self.token_dropout_prob, self.max_tokens_dropped, self.rng
                )

            item = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long) if labels is not None else None,
                "bbox": torch.tensor(bboxes, dtype=torch.long) if bboxes is not None else None,
            }

            if any_pixel_values:
                pv = ex.get("pixel_values")
                if isinstance(pv, torch.Tensor):
                    if pv.dim() == 4 and pv.size(0) == 1:
                        pv = pv.squeeze(0)   # [1,3,H,W] -> [3,H,W]
                elif pv is not None:
                    pv = torch.tensor(pv, dtype=torch.float)
                item["pixel_values"] = pv  # may be None for text-only models

            proc.append(item)

        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in proc]),
        }
        pad_id = getattr(self.tokenizer, "pad_token_id", 0)
        batch["attention_mask"] = (batch["input_ids"] != pad_id).long()

        if proc[0]["labels"] is not None:
            batch["labels"] = torch.stack([f["labels"] for f in proc])
        if proc[0]["bbox"] is not None:
            batch["bbox"] = torch.stack([f["bbox"] for f in proc])
        if any_pixel_values and proc[0].get("pixel_values") is not None:
            batch["pixel_values"] = torch.stack([f["pixel_values"] for f in proc])

        return batch
