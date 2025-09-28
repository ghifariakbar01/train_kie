"""
Refactor: Convert processed.jsonl (tokens, bboxes, labels) into a Hugging Face Dataset
ready for LayoutLMv3 training. Expects the original images to exist on disk.

Usage:
  python refactor_prepare_hf.py \
    --jsonl ./data/processed.jsonl \
    --images ./data/raw/images \
    --out ./data/features

Assumptions:
- processed.jsonl has: id (str), tokens (List[str]), bboxes (List[List[int]] 4 or 8 long), labels (List[str])
- Images live at <images>/<id>.{jpg|png|jpeg}
"""

import argparse, os, json
from pathlib import Path
from typing import Optional, List

import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoProcessor
from PIL import Image

def find_image_path(img_dir: str, stem: str) -> Optional[str]:
    for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"):
        p = os.path.join(img_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None

def quad_to_aabb(box8: List[int]) -> List[int]:
    xs = box8[0::2]; ys = box8[1::2]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

def ensure_aabb(box) -> List[int]:
    # Accepts 4 or 8 numbers; returns 4 ints in pixel space
    if isinstance(box, (list, tuple)):
        if len(box) == 4:
            x1, y1, x2, y2 = map(int, box)
            return [x1, y1, x2, y2]
        if len(box) == 8:
            return quad_to_aabb(list(map(int, box)))
    raise ValueError(f"Unsupported bbox format: {box}")

def clamp(v, lo, hi): 
    return max(lo, min(hi, v))

def normalize_box_0_1000_px(aabb_px: List[int], W: int, H: int) -> List[int]:
    """
    Convert pixel aabb [x1,y1,x2,y2] to 0–1000 space for LayoutLMv3.
    Clamps and fixes order / zero-area after rounding.
    """
    x1, y1, x2, y2 = aabb_px

    # fix inverted
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1

    # clamp to image frame
    x1 = clamp(x1, 0, W - 1); x2 = clamp(x2, 0, W - 1)
    y1 = clamp(y1, 0, H - 1); y2 = clamp(y2, 0, H - 1)

    # avoid zero-area boxes
    if x2 == x1: x2 = clamp(x1 + 1, 0, W - 1)
    if y2 == y1: y2 = clamp(y1 + 1, 0, H - 1)

    # scale to 0–1000
    nx1 = int(round(1000 * x1 / W))
    ny1 = int(round(1000 * y1 / H))
    nx2 = int(round(1000 * x2 / W))
    ny2 = int(round(1000 * y2 / H))

    # clamp to [0,1000] and ensure non-zero after rounding
    nx1 = clamp(nx1, 0, 1000); ny1 = clamp(ny1, 0, 1000)
    nx2 = clamp(nx2, 0, 1000); ny2 = clamp(ny2, 0, 1000)
    if nx2 <= nx1: nx2 = min(1000, nx1 + 1)
    if ny2 <= ny1: ny2 = min(1000, ny1 + 1)
    return [nx1, ny1, nx2, ny2]

def main(args):
    jsonl_path = args.jsonl
    img_dir    = args.images
    out_dir    = args.out

    # 1) Load JSONL
    ds_all = load_dataset("json", data_files={"train": jsonl_path})["train"]
    print(f"Loaded {len(ds_all)} examples from {jsonl_path}")

    # 2) Attach image_path and filter
    def add_image_path(example):
        p = find_image_path(img_dir, example["id"])
        if p:
            example["image_path"] = p
        return example

    ds_all = ds_all.map(add_image_path)
    ds_all = ds_all.filter(lambda ex: "image_path" in ex and ex["image_path"] is not None)

    if len(ds_all) == 0:
        print("WARNING: No images matched the ids. Check --images directory and file extensions.")

    # 3) Splits (80/10/10)
    tmp = ds_all.train_test_split(test_size=0.2, seed=42)
    val_test = tmp["test"].train_test_split(test_size=0.5, seed=42)
    ds = DatasetDict(train=tmp["train"], validation=val_test["train"], test=val_test["test"])

    # 4) Label maps
    all_labels = set()
    for row in ds_all:
        all_labels.update(row["labels"])
    label_list = sorted(all_labels)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    print(f"Labels ({len(label_list)}): {label_list}")

    # 5) Processor
    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

    # 6) Encode (normalize to 0–1000; return unbatched tensors via squeeze(0))
    def encode(example):
        image = Image.open(example["image_path"]).convert("RGB")
        W, H = image.size

        words = example["tokens"]
        boxes_in = example["bboxes"]
        labels_str = example["labels"]

        if not (len(words) == len(boxes_in) == len(labels_str)):
            raise ValueError(
                f"Length mismatch for id={example.get('id')}: "
                f"tokens={len(words)} boxes={len(boxes_in)} labels={len(labels_str)}"
            )

        boxes_px = [ensure_aabb(b) for b in boxes_in]
        boxes_1000 = [normalize_box_0_1000_px(b, W, H) for b in boxes_px]
        word_labels = [label2id[l] for l in labels_str]

        enc = processor(
            image,
            words,
            boxes=boxes_1000,
            word_labels=word_labels,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",  # get per-sample tensors
        )
        # Remove the per-sample batch dim so Trainer can batch later
        enc = {k: (v.squeeze(0) if isinstance(v, torch.Tensor) else v) for k, v in enc.items()}
        return enc

    encoded = ds.map(
        encode,
        remove_columns=ds["train"].column_names,
        desc="Encoding examples",
    )

    # 7) Save
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    encoded.save_to_disk(out_dir)
    print(f"✅ Saved encoded dataset to {out_dir}")

    with open(os.path.join(out_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"label_list": label_list, "label2id": label2id, "id2label": id2label},
            f, indent=2, ensure_ascii=False
        )
    print(f"✅ Saved label maps to {os.path.join(out_dir, 'label_map.json')}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl",  default="./data/processed.jsonl", help="Path to processed.jsonl")
    p.add_argument("--images", default="./data/raw/images", help="Directory containing original images")
    p.add_argument("--out",    default="./data/features", help="Output directory (save_to_disk)")
    args = p.parse_args()
    main(args)
