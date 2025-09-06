import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datasets import Dataset, DatasetDict
from transformers import LayoutLMv3Processor
from PIL import Image
from src.utils import load_jsonl, make_splits, build_label_map


def encode_batch(batch, processor, label2id, image_dir):
    """Convert one batch into LayoutLMv3 features."""
    images = [Image.open(os.path.join(image_dir, f"{i}.jpg")).convert("RGB") for i in batch["id"]]
    encodings = processor(
        images=images,
        text=batch["tokens"],
        boxes=batch["bboxes"],
        word_labels=[[label2id[l] for l in labels] for labels in batch["labels"]],
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    return {k: v for k, v in encodings.items()}


def prepare_features(jsonl_path, image_dir, out_dir="data/features"):
    # 1. Load raw data
    raw_data = load_jsonl(jsonl_path)

    # 2. Build label map
    label2id, id2label = build_label_map(raw_data, out_path=os.path.join(out_dir, "label_map.json"))

    # 3. Split
    splits = make_splits(raw_data)
    dataset_dict = {}
    for split_name, examples in splits.items():
        if len(examples) > 0:
            dataset_dict[split_name] = Dataset.from_list(examples)
        else:
            print(f"⚠️ Skipping empty split: {split_name}")

    dataset_dict = DatasetDict(dataset_dict)

    # 4. Load processor
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

    # 5. Encode
    print("🔄 Encoding dataset with LayoutLMv3Processor...")
    encoded = dataset_dict.map(
        lambda batch: encode_batch(batch, processor, label2id, image_dir),
        batched=True,
        remove_columns=next(iter(dataset_dict.values())).column_names,
        desc="📦 Converting examples to tensors"
    )

    # 6. Save
    encoded.save_to_disk(out_dir)
    print(f"✅ Saved encoded dataset to {out_dir}")


if __name__ == "__main__":
    prepare_features(
        jsonl_path="data/processed.jsonl",
        image_dir="data/raw/images",
        out_dir="data/features"
    )
