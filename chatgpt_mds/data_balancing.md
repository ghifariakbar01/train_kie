Good question — **data imbalance is very common in KIE**, and handling it correctly can noticeably improve **recall and F1**, especially for rare labels like `B-TOTAL`, `B-DATE`, etc.

I’ll explain **what “balancing” means in KIE**, **what you can and cannot balance**, and then give you **practical ways to apply it to your existing script**.

---

## 1️⃣ What “data imbalance” means in KIE (important)

In **token classification KIE**, imbalance exists at **token level**, not image level.

Typical situation:

* `O` label = **80–95%** of tokens
* `B-ITEM`, `B-PRICE` = many tokens
* `B-TOTAL`, `B-DATE`, `B-RECEIPT_NO` = very few tokens

So the model learns:

> “Predict `O` most of the time and I’ll get good loss”

This **hurts recall** on rare fields.

⚠️ **You cannot simply oversample images** and expect it to fix token imbalance.
You need **label-aware strategies**.

---

## 2️⃣ What balancing methods work for KIE (ranked by impact)

### ✅ 1. **Loss weighting (MOST IMPORTANT)**

This is the **correct and safest** method.

You tell the model:

> “Mistakes on rare labels are more expensive.”

### ✅ 2. **Targeted data augmentation**

You already do this (bbox jitter, token dropout) — but we can **apply it selectively**.

### ⚠️ 3. **Oversampling documents**

Helps slightly, but risky if done blindly.

### ❌ 4. Removing `O` labels

Never do this — the model must learn background.

---

## 3️⃣ Method 1 — Class-weighted loss (RECOMMENDED)

### Step A — Compute label frequencies (once)

Create a small script (or add to training setup):

```python
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
    weights = weights / weights.mean()
    return weights
```

---

### Step B — Use weighted CrossEntropyLoss

In **train.py**, replace the default loss by **subclassing the model**:

```python
import torch.nn as nn
from transformers import LayoutLMv3ForTokenClassification

class WeightedLayoutLMv3(LayoutLMv3ForTokenClassification):
    def __init__(self, config, class_weights):
        super().__init__(config)
        self.loss_fct = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights),
            ignore_index=-100
        )

    def forward(self, **kwargs):
        labels = kwargs.get("labels")
        outputs = super().forward(**kwargs)
        if labels is not None:
            logits = outputs.logits
            loss = self.loss_fct(
                logits.view(-1, self.num_labels),
                labels.view(-1)
            )
            outputs.loss = loss
        return outputs
```

---

### Step C — Plug it into your training script

Replace this:

```python
model = LayoutLMv3ForTokenClassification.from_pretrained(...)
```

With:

```python
weights = compute_label_weights(train_ds, num_labels=len(id2label))

model = WeightedLayoutLMv3.from_pretrained(
    args.model_name,
    num_labels=len(id2label),
    id2label=id2label,
    label2id={v: k for k, v in id2label.items()},
    class_weights=weights
)
```

✅ **This alone often improves recall + F1 by 5–15%** for rare fields.

---

## 4️⃣ Method 2 — Noise augmentation only on rare-label pages

You already have **noise-aware collators**. Improve them:

### Idea

Apply strong augmentation **only when rare labels appear**.

```python
def has_rare_label(labels, rare_ids):
    return any(l in rare_ids for l in labels if l != -100)
```

Inside your collator:

```python
if has_rare_label(labels, rare_ids):
    bbox_jitter_prob = 0.8
    token_dropout_prob = 0.2
else:
    bbox_jitter_prob = 0.2
```

✅ Makes the model **robust where it matters most**.

---

## 5️⃣ Method 3 — Document oversampling (optional)

Only oversample documents that contain **rare entities**.

```python
def contains_label(example, target_ids):
    return any(l in target_ids for l in example["labels"])

rare_docs = train_ds.filter(lambda x: contains_label(x, rare_ids))
balanced_train = concatenate_datasets([train_ds, rare_docs])
```

⚠️ Use **small duplication (1–2× only)**.

---

## 6️⃣ What NOT to do ❌

❌ Random SMOTE — does not work for sequences
❌ Removing `O` labels
❌ Equalizing token counts manually

These **break sequence structure**.

---

## 7️⃣ How to verify balancing worked

After training, check:

| Metric    | Expected Change  |
| --------- | ---------------- |
| Loss      | Slightly higher  |
| Precision | Slight drop      |
| Recall    | **Big increase** |
| F1        | **Net increase** |

Specifically check **per-label recall**:

```python
eval_results["eval_recall_per_label"]
```

Rare labels should improve the most.

---

## 8️⃣ Minimal recommendation for your setup

Given your **RTX 2060 + current scripts**, do **only this first**:

1. ✅ Add **class-weighted loss**
2. ✅ Keep noise-aware collator
3. ❌ Skip oversampling initially

This is **safe, clean, and thesis-worthy**.

---

If you want, I can:

* Add **exact code patches** into your current `train.py`
* Help you **report balanced vs unbalanced results** properly in your thesis
* Show **per-label confusion analysis**

Just tell me 👍
