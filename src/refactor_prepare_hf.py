
"""
Refactor: Convert processed.jsonl (tokens, bboxes, labels) into a Hugging Face Dataset
ready for LayoutLMv3 training. Expects the original images to exist on disk.

Usage (example):
    python refactor_prepare_hf.py \
        --jsonl /mnt/data/processed.jsonl \
        --images /path/to/images \
        --out    /mnt/data/features

Notes:
- processed.jsonl lines must contain: id, tokens, bboxes (quad), labels
- We will search for images as <images>/<id>.{jpg|png|jpeg}
- Boxes are converted from quadrilaterals to [xmin,ymin,xmax,ymax] and normalized by processor.
"""

import argparse, os, json
from pathlib import Path
from typing import List, Dict

from datasets import load_dataset, DatasetDict
from transformers import AutoProcessor

from PIL import Image

def find_image_path(img_dir: str, stem: str) -> str | None:
    for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"):
        p = os.path.join(img_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None

def normalize_box_from_quad(box: List[int]) -> List[int]:
    # box: [x1,y1,x2,y2,x3,y3,x4,y4]
    xs = box[0::2]
    ys = box[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return [int(xmin), int(ymin), int(xmax), int(ymax)]

def main(args):
    jsonl_path = args.jsonl
    img_dir    = args.images
    out_dir    = args.out

    # 1) Load JSONL
    ds_all = load_dataset("json", data_files={"train": jsonl_path})["train"]
    n = len(ds_all)
    print(f"Loaded {n} examples from {jsonl_path}")

    # 2) Attach image_path
    def add_image_path(example):
        p = find_image_path(img_dir, example["id"])
        if p: example["image_path"] = p
        return example

    ds_all = ds_all.map(add_image_path)
    ds_all = ds_all.filter(lambda ex: "image_path" in ex and ex["image_path"] is not None)

    if len(ds_all) == 0:
        print("WARNING: No images found that match ids. Check your --images folder and file extensions.")
        print("Keeping dataset empty; you can still inspect schema. Exiting after save attempt.")
    
    # 3) Splits (80/10/10)
    tmp = ds_all.train_test_split(test_size=0.2, seed=42)
    val_test = tmp["test"].train_test_split(test_size=0.5, seed=42)
    ds = DatasetDict(train=tmp["train"], validation=val_test["train"], test=val_test["test"])

    # 4) Label maps
    all_labels = set()
    for row in ds_all:
        for y in row["labels"]:
            all_labels.add(y)
    label_list = sorted(all_labels)
    label2id = {l:i for i,l in enumerate(label_list)}
    id2label = {i:l for l,i in label2id.items()}
    print(f"Labels ({len(label_list)}): {label_list}")

    # 5) Processor
    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

    # 6) Encode
    def encode(example):
        image = Image.open(example["image_path"]).convert("RGB")
        words = example["tokens"]
        boxes_quad = example["bboxes"]
        labels_str = example["labels"]

        boxes = [normalize_box_from_quad(b) for b in boxes_quad]
        word_labels = [label2id[l] for l in labels_str]

        encoded = processor(
            image,
            words,
            boxes=boxes,
            word_labels=word_labels,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors=None
        )
        return encoded

    encoded = ds.map(encode, remove_columns=ds["train"].column_names, desc="Encoding examples")

    # 7) Save
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    encoded.save_to_disk(out_dir)
    print(f"✅ Saved encoded dataset to {out_dir}")
    # Also save the label maps for training scripts
    with open(os.path.join(out_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"label_list": label_list, "label2id": label2id, "id2label": id2label}, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved label maps to {os.path.join(out_dir, 'label_map.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default="./data/processed.jsonl", required=True, help="Path to processed.jsonl")
    parser.add_argument("--images", default="./data/raw/images", required=True, help="Directory containing original images")
    parser.add_argument("--out", default="./data/output",required=True, help="Output directory for HF dataset (save_to_disk)")
    args = parser.parse_args()
    main(args)
