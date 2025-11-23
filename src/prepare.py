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
from transformers import AutoProcessor, AutoTokenizer
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
    # 4) Label maps (prefer existing; otherwise build from data)
    existing_map = os.path.join(os.path.dirname(jsonl_path), "label_map.json")
    if os.path.exists(existing_map):
        with open(existing_map, "r", encoding="utf-8") as f:
            info = json.load(f)
        label_list = info.get("label_list")
        label2id   = {str(k): int(v) for k, v in info.get("label2id", {}).items()} if info.get("label2id") else {
            l: i for i, l in enumerate(label_list)
        }
        id2label   = {int(k): v for k, v in info.get("id2label", {}).items()} if info.get("id2label") else {
            i: l for l, i in label2id.items()
        }
        # sanity: ensure all labels in data exist in the map
        data_labels = {y for row in ds_all for y in row["labels"]}
        missing = sorted(list(data_labels - set(label2id.keys())))
        if missing:
            raise ValueError(f"Label map missing classes present in data: {missing}")
    else:
        # build from data if no map is present
        all_labels = sorted({y for row in ds_all for y in row["labels"]})
        label_list = all_labels
        label2id   = {l: i for i, l in enumerate(label_list)}
        id2label   = {i: l for l, i in label2id.items()}

    
    def is_layoutlm_v1(name: str) -> bool:
        return "layoutlm-base" in name.lower()

    def is_lilt(name: str) -> bool:
        return "lilt" in name.lower()

    def is_vision_layout(name: str) -> bool:
        n = name.lower()
        return ("layoutlmv2" in n) or ("layoutlmv3" in n) or ("layoutxlm" in n)
    
    model_name = args.model_name

    if is_lilt(model_name) or is_layoutlm_v1(model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)   # text-only models
        processor = None
    elif is_vision_layout(model_name):
        processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)  # vision models
        tokenizer = getattr(processor, "tokenizer", None)
    else:
        # default: assume vision model
        processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)
        tokenizer = getattr(processor, "tokenizer", None)

    # Clear, explicit boolean:
    text_only = is_lilt(model_name) or is_layoutlm_v1(model_name)

    if text_only:
        tokenizer = AutoTokenizer.from_pretrained(model_name)  # LayoutLM v1 / LiLT
    else:
        processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)  # v2/v3/xlm

    def encode_vision(example):
        image = Image.open(example["image_path"]).convert("RGB")
        W, H = image.size
        words = example["tokens"]
        boxes_px = [ensure_aabb(b) for b in example["bboxes"]]
        boxes_1000 = [normalize_box_0_1000_px(b, W, H) for b in boxes_px]
        word_labels = [label2id[l] for l in example["labels"]]

        enc = processor(
            images=image,                                  # <-- explicit name
            text=words,       
            boxes=boxes_1000,
            word_labels=word_labels,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: (v.squeeze(0) if isinstance(v, torch.Tensor) else v) for k, v in enc.items()}
        return enc
    

    def encode_text_only(example):
        words = example["tokens"]
        # no image; still normalize bboxes to 0–1000 using the image size if you want consistent scale
        image = Image.open(example["image_path"]).convert("RGB")
        W, H = image.size
        boxes_px = [ensure_aabb(b) for b in example["bboxes"]]
        boxes_1000 = [normalize_box_0_1000_px(b, W, H) for b in boxes_px]
        word_labels = [label2id[l] for l in example["labels"]]

        # tokenize words
        tok = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

        # align bboxes/labels to subwords using word_ids()
        word_ids = tok.word_ids(batch_index=0)  # list[Optional[int]] length = seq_len
        aligned_boxes, aligned_labels = [], []
        previous_word_idx = None
        for wi in word_ids:
            if wi is None:
                aligned_boxes.append([0, 0, 0, 0])
                aligned_labels.append(-100)          # mask specials/pad
            else:
                # label only the first subword of each word (common practice)
                if wi != previous_word_idx:
                    aligned_boxes.append(boxes_1000[wi])
                    aligned_labels.append(word_labels[wi])
                else:
                    aligned_boxes.append(boxes_1000[wi])
                    aligned_labels.append(-100)      # mask subsequent subwords
            previous_word_idx = wi

        # build the batch dict; no pixel_values for v1/LiLT
        enc = {
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "bbox": torch.tensor(aligned_boxes, dtype=torch.long),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }
        return enc

    def encode_lilt(example):
        words = example["tokens"]
        image = Image.open(example["image_path"]).convert("RGB")
        W, H = image.size
        boxes_px = [ensure_aabb(b) for b in example["bboxes"]]
        boxes_1000 = [normalize_box_0_1000_px(b, W, H) for b in boxes_px]
        word_labels = [label2id[l] for l in example["labels"]]

        # Some versions support is_split_into_words, some don't. Try w/ it, fallback w/o.
        try:
            tok = tokenizer(
                words,
                boxes=boxes_1000,
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            )
        except TypeError:
            # older tokenizer API: no is_split_into_words kw; still accepts list[str] and boxes
            tok = tokenizer(
                words,
                boxes=boxes_1000,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            )

        # Align labels to subwords using word_ids (works in both cases)
        word_ids = tok.word_ids(0)
        aligned_labels = []
        prev = None
        for wi in word_ids:
            if wi is None:
                aligned_labels.append(-100)
            else:
                if wi != prev:
                    aligned_labels.append(word_labels[wi])  # first subword of a word
                else:
                    aligned_labels.append(-100)             # subsequent subwords
            prev = wi

        return {
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "bbox": tok["bbox"].squeeze(0),   # already expanded to subwords by tokenizer
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }

    if is_lilt(model_name):
        encode_fn = encode_lilt
    elif is_layoutlm_v1(model_name):
        encode_fn = encode_text_only  # your LayoutLMv1 path
    else:
        encode_fn = encode_vision

    encoded = ds.map(
        encode_fn,
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
    p.add_argument("--model_name", default="", help="HF model/processor name to encode with")
    args = p.parse_args()
    main(args)
