# src/collator.py
from typing import List, Dict, Any, Optional
import random
import torch
from transformers import PreTrainedTokenizerBase
from .augmentation import jitter_bboxes, random_token_dropout  # adjust import to your package layout

class DataCollatorForLayoutLMv3Noise:
    def __init__(
        self,
        processor,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        bbox_jitter_prob: float = 0.5,
        bbox_jitter_scale: float = 0.02,
        token_dropout_prob: float = 0.05,
        max_tokens_dropped: int = 3,
        label_mask_value: int = -100,
        seed: int = 42,
    ):
        self.processor = processor
        self.tokenizer = tokenizer or processor.tokenizer
        self.bbox_jitter_prob = bbox_jitter_prob
        self.bbox_jitter_scale = bbox_jitter_scale
        self.token_dropout_prob = token_dropout_prob
        self.max_tokens_dropped = max_tokens_dropped
        self.label_mask_value = label_mask_value
        self.rng = random.Random(seed)

    @classmethod
    def from_config(cls, processor, cfg: Dict[str, Any]):
        return cls(
            processor=processor,
            tokenizer=None,
            bbox_jitter_prob=cfg.get("bbox_jitter_prob", 0.5),
            bbox_jitter_scale=cfg.get("bbox_jitter_scale", 0.02),
            token_dropout_prob=cfg.get("token_dropout_prob", 0.05),
            max_tokens_dropped=cfg.get("max_tokens_dropped", 3),
            label_mask_value=cfg.get("label_mask_value", -100),
            seed=cfg.get("seed", 42),
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        proc_features = []
        for ex in features:
            # tensors -> lists
            input_ids = ex["input_ids"]
            if isinstance(input_ids, torch.Tensor): input_ids = input_ids.tolist()

            labels = ex.get("labels")
            if isinstance(labels, torch.Tensor): labels = labels.tolist()

            bboxes = ex.get("bbox")
            if isinstance(bboxes, torch.Tensor): bboxes = bboxes.tolist()

            # bbox jitter in 0–1000 space
            if bboxes is not None and self.bbox_jitter_prob > 0:
                bboxes = jitter_bboxes(bboxes, self.bbox_jitter_scale, self.rng, self.bbox_jitter_prob)

            # token dropout (mask id if exists)
            if input_ids is not None and labels is not None and self.token_dropout_prob > 0:
                mask_id = getattr(self.tokenizer, "mask_token_id", None)
                input_ids, labels = random_token_dropout(
                    input_ids, labels, mask_id, self.token_dropout_prob, self.max_tokens_dropped, self.rng
                )

            # pixel_values: ensure shape [3,H,W] (some datasets store [1,3,H,W])
            pv = ex.get("pixel_values")
            if isinstance(pv, torch.Tensor):
                if pv.dim() == 4 and pv.size(0) == 1:
                    pv = pv.squeeze(0)  # [1,3,H,W] -> [3,H,W]
            else:
                pv = torch.tensor(pv, dtype=torch.float)

            proc_features.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long) if labels is not None else None,
                "bbox": torch.tensor(bboxes, dtype=torch.long) if bboxes is not None else None,
                "pixel_values": pv,  # [3,H,W]
            })

        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in proc_features]),
            "pixel_values": torch.stack([f["pixel_values"] for f in proc_features]),
        }
        if proc_features[0]["labels"] is not None:
            batch["labels"] = torch.stack([f["labels"] for f in proc_features])
        if proc_features[0]["bbox"] is not None:
            batch["bbox"] = torch.stack([f["bbox"] for f in proc_features])

        pad_id = getattr(self.tokenizer, "pad_token_id", 0)
        batch["attention_mask"] = (batch["input_ids"] != pad_id).long()
        return batch
