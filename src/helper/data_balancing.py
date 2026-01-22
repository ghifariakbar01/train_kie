# src/data_balancing.py
from collections import Counter
import numpy as np

def compute_label_weights(dataset, num_labels):
    counter = Counter()
    for ex in dataset:
        labels = ex["labels"]
        for l in labels:
            if l != -100:
                counter[l] += 1

    total = sum(counter.values())
    weights = np.zeros(num_labels, dtype=np.float32)

    for i in range(num_labels):
        freq = counter.get(i, 1)
        weights[i] = total / freq

    # normalize
    weights = np.clip(weights, 0.2, 10.0)
    weights = weights / weights.mean()
    return weights
import torch
import torch.nn as nn
from transformers import Trainer

class WeightedTokenTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is None:
            raise ValueError("class_weights is required")
        self._class_weights = torch.tensor(class_weights, dtype=torch.float)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # labels must exist for token classification training
        labels = inputs.get("labels")
        if labels is None:
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        labels = labels.long()
        outputs = model(**inputs)  # outputs.logits: (B, T, C)
        logits = outputs.logits

        weight = self._class_weights.to(logits.device)
        loss_fct = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)

        loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss
