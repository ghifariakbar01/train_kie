import os
import json
import argparse
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoModelForTokenClassification, AutoProcessor


# ---------- I/O helpers ----------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_id2label(path: str) -> Dict[int, str]:
    with open(path, "r", encoding="utf-8") as f:
        lm = json.load(f)

    # supports {"id2label": {"0": "O", ...}}
    if "id2label" in lm:
        return {int(k): str(v) for k, v in lm["id2label"].items()}

    # supports flat {"0": "O", ...}
    if lm and all(str(k).isdigit() for k in lm.keys()):
        return {int(k): str(v) for k, v in lm.items()}

    raise ValueError("Unsupported label_map format. Expected id2label or flat digit-key dict.")


# ---------- bbox helpers ----------
def quad_to_aabb(box8: List[int]) -> List[int]:
    xs = box8[0::2]
    ys = box8[1::2]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def ensure_aabb(box: Any) -> List[int]:
    """
    Accepts 4 or 8 numbers; returns [x1,y1,x2,y2] as ints in pixel space.
    """
    if isinstance(box, (list, tuple)):
        if len(box) == 4:
            x1, y1, x2, y2 = map(int, box)
            return [x1, y1, x2, y2]
        if len(box) == 8:
            return quad_to_aabb(list(map(int, box)))
    raise ValueError(f"Unsupported bbox format: {box}")


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def normalize_box_0_1000_px(aabb_px: List[int], W: int, H: int) -> List[int]:
    """
    Convert pixel aabb [x1,y1,x2,y2] to 0–1000 space for LayoutLMv3.
    Fixes inverted coords, clamps to image, avoids zero-area, then scales.
    """
    x1, y1, x2, y2 = aabb_px

    # fix inverted
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    # clamp to image frame
    x1 = clamp(x1, 0, W - 1)
    x2 = clamp(x2, 0, W - 1)
    y1 = clamp(y1, 0, H - 1)
    y2 = clamp(y2, 0, H - 1)

    # avoid zero-area boxes
    if x2 == x1:
        x2 = clamp(x1 + 1, 0, W - 1)
    if y2 == y1:
        y2 = clamp(y1 + 1, 0, H - 1)

    # scale to 0–1000
    nx1 = int(round(1000 * x1 / W))
    ny1 = int(round(1000 * y1 / H))
    nx2 = int(round(1000 * x2 / W))
    ny2 = int(round(1000 * y2 / H))

    # clamp to [0,1000] and ensure non-zero after rounding
    nx1 = clamp(nx1, 0, 1000)
    ny1 = clamp(ny1, 0, 1000)
    nx2 = clamp(nx2, 0, 1000)
    ny2 = clamp(ny2, 0, 1000)
    if nx2 <= nx1:
        nx2 = min(1000, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(1000, ny1 + 1)

    return [nx1, ny1, nx2, ny2]


def find_image(images_dir: str, ex_id: str) -> str:
    for ext in ("png", "jpg", "jpeg", "webp", "PNG", "JPG", "JPEG", "WEBP"):
        p = os.path.join(images_dir, f"{ex_id}.{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Image not found for id='{ex_id}' under '{images_dir}'")


# ---------- inference ----------
@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    model_dir = os.path.abspath(args.model_dir)
    images_dir = os.path.abspath(args.images_dir)
    input_path = os.path.abspath(args.input)
    label_map_path = os.path.abspath(args.label_map)
    output_path = os.path.abspath(args.output)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    id2label = load_id2label(label_map_path)

    # Processor source:
    # - If you saved processor files with the run folder, pass that directory.
    # - Otherwise use "microsoft/layoutlmv3-base" (online/cached).
    processor_source = args.processor_source
    processor = AutoProcessor.from_pretrained(processor_source, apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(model_dir, local_files_only=True)

    model.to(device)
    model.eval()

    use_autocast = args.fp16 and device == "cuda"

    data = load_jsonl(input_path)
    outputs: List[Dict[str, Any]] = []

    for ex in data:
        ex_id = str(ex.get("id"))
        tokens = ex.get("tokens") or ex.get("words")
        bboxes_raw = ex.get("bboxes") or ex.get("bbox")

        if not ex_id:
            raise ValueError("Missing 'id' in an example.")
        if tokens is None:
            raise ValueError(f"id={ex_id}: missing 'tokens' (or 'words').")
        if bboxes_raw is None:
            raise ValueError(f"id={ex_id}: missing 'bboxes' (or 'bbox').")

        if len(tokens) != len(bboxes_raw):
            raise ValueError(f"id={ex_id}: tokens({len(tokens)}) != bboxes({len(bboxes_raw)})")

        img_path = find_image(images_dir, ex_id)
        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        # Normalize bboxes to 0–1000
        boxes_px = [ensure_aabb(b) for b in bboxes_raw]
        boxes_1000 = [normalize_box_0_1000_px(b, W, H) for b in boxes_px]

        enc = processor(
            image,
            tokens,
            boxes=boxes_1000,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_autocast):
            logits = model(**enc).logits  # [1, L, C]

        pred_ids = logits.argmax(dim=-1).squeeze(0).detach().cpu().tolist()

        attn = enc["attention_mask"].squeeze(0).detach().cpu().tolist()
        kept_ids = [pid for pid, m in zip(pred_ids, attn) if m == 1]
        kept_labels = [id2label.get(int(pid), "O") for pid in kept_ids]

        outputs.append(
            {
                "id": ex_id,
                "image_path": img_path,
                "tokens": tokens,
                "pred_ids": kept_ids,
                "pred_labels": kept_labels,
            }
        )

    write_jsonl(output_path, outputs)
    print(f"Saved: {output_path} ({len(outputs)} examples) on device={device}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", required=True, help="Directory containing images named {id}.png/jpg/jpeg/webp")
    p.add_argument("--model_dir", required=True, help="Checkpoint directory (e.g., .../checkpoint-4000)")
    p.add_argument("--input", required=True, help="processed.jsonl with id,tokens,bboxes")
    p.add_argument("--label_map", required=True, help="label_map.json with id2label/label2id")
    p.add_argument("--output", default="predictions.jsonl")
    p.add_argument("--max_length", type=int, default=512)

    # This avoids the 'preprocessor_config.json missing in checkpoint' problem:
    # point to run folder OR a hub name.
    p.add_argument(
        "--processor_source",
        default="microsoft/layoutlmv3-base",
        help="Path to processor folder OR hub name (default: microsoft/layoutlmv3-base)",
    )

    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--fp16", action="store_true", help="Use fp16 autocast on CUDA")

    args = p.parse_args()
    main(args)
