import json
import random

def load_jsonl(path):
    """Load dataset from JSONL file (produced in Step 1)."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_json(obj, path):
    """Save a Python dict as a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def make_splits(data, train_split=0.8, val_split=0.1, seed=42):
    """Split dataset into train/val/test subsets."""
    random.seed(seed)
    random.shuffle(data)
    n = len(data)
    n_train = int(train_split * n)
    n_val = int(val_split * n)
    return {
        "train": data[:n_train],
        "val": data[n_train:n_train+n_val],
        "test": data[n_train+n_val:]
    }

def build_label_map(raw_data, out_path=None):
    """
    Create label2id and id2label mappings from dataset labels.
    raw_data: list of dicts with 'labels' field
    """
    all_labels = sorted({l for ex in raw_data for l in ex["labels"]})
    id2label = {i: l for i, l in enumerate(all_labels)}
    label2id = {l: i for i, l in id2label.items()}

    if out_path:
        save_json({"id2label": id2label, "label2id": label2id}, out_path)

    return label2id, id2label
