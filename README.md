
# Refactor: From `processed.jsonl` to a Hugging Face dataset for LayoutLMv3

This repo shows how to take your `processed.jsonl` (with `id`, `tokens`, `bboxes`, `labels`) and convert it into an on-disk Hugging Face dataset that LayoutLMv3 can train on.

## Files created here
- `refactor_prepare_hf.py` — CLI script to convert JSONL into a HF dataset (requires images).
- `label_map.json` — saved alongside the dataset with your label mapping.
- Output dataset saved via `Dataset.save_to_disk()`.

## Quickstart

```bash
python ./src/refactor_prepare_hf.py --jsonl ./data/processed.jsonl --images ./data/raw/images --out ./data/features
```

**Important:** The script looks for images by filename stem: `<images>/<id>.jpg|.png|.jpeg`. Adjust your image paths or edit the function `find_image_path` if needed.

## What the script does

1. Loads your JSONL with `load_dataset("json")`.
2. Adds `image_path` for each example by matching `<id>` to files in `--images`.
3. Splits into train/validation/test (80/10/10).
4. Builds `label_list` and `{label2id,id2label}` from the JSONL.
5. Uses `AutoProcessor("microsoft/layoutlmv3-base", apply_ocr=False)` to encode `(image, words, boxes, labels)`.
6. Saves the encoded dataset to `--out` with `save_to_disk`.

You can then train a model with Hugging Face `Trainer` by loading the dataset with `load_from_disk("--out")`.
