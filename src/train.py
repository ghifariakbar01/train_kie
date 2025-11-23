# train_baseline.py


import sys, os
from pathlib import Path

# add project root and ./src to sys.path so "src.*" works regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))


import os, json, argparse
from datasets import load_from_disk
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

from helper.metrics import compute_metrics_builder
from helper.utils import (
    load_label_maps,
    ensure_splits,
    maybe_build_collators,
)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",  default="./data/features", help="HF dataset (save_to_disk).")
    ap.add_argument("--label_map", default=None, help="label_map.json (defaults to <data_dir>/label_map.json).")
    ap.add_argument("--out_dir",   default="./data/output/", help="Output dir for checkpoints & report.")
    ap.add_argument("--model_name", default="")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr",     type=float, default=5e-5)
    ap.add_argument("--train_bsz", type=int, default=2)
    ap.add_argument("--eval_bsz",  type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)

    # device/precision
    ap.add_argument("--no_cuda", action="store_true", help="Force CPU.")
    ap.add_argument("--fp16", action="store_true", help="Enable FP16 (CUDA only).")
    ap.add_argument("--bf16", action="store_true", help="Enable BF16 (CUDA only).")

    # optional noise-aware collator
    ap.add_argument("--noise_config", default="./config/augmentation.json", help="Path to augmentation.json to enable noise-aware collator.")
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    # 1) Load dataset + label maps
    ds = load_from_disk(args.data_dir)
    label_map_path = args.label_map or os.path.join(args.data_dir, "label_map.json")
    label2id, id2label = load_label_maps(label_map_path)
    train_ds, eval_ds, test_ds = ensure_splits(ds)
    print(f"Train size: {len(train_ds)} | Val size: {len(eval_ds)} | Test size: {len(test_ds) if test_ds else 0}")

    # 2) Model
    # 2) Model (architecture-agnostic)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    )

    # 3) Collators (optional noise-aware)
    train_collator, eval_collator = maybe_build_collators(args.noise_config, args.model_name)
    if train_collator is None:
        eval_collator = train_collator = None  # Trainer will use default

    # 4) Training args (tqdm enabled)
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.train_bsz,
        per_device_eval_batch_size=args.eval_bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        # no_cuda=args.no_cuda,
        fp16=True,
        # bf16=bool(args.bf16 and not args.no_cuda),

        logging_steps=50,
        logging_first_step=True,
        disable_tqdm=False,   # progress bars on
        report_to=[],         # no TB/W&B by default
        save_total_limit=2,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics_builder(id2label),
        data_collator=train_collator,  # may be None; Trainer falls back to default
    )

    report = {}

    # === BEFORE (clean eval) ===
    if eval_collator is not None:
        trainer.data_collator = eval_collator
    val_before = trainer.evaluate()
    report["val_before"] = val_before
    if test_ds is not None:
        test_before = trainer.evaluate(test_ds)
        report["test_before"] = test_before

    # === TRAIN ===
    if train_collator is not None:
        trainer.data_collator = train_collator
    trainer.train()

    # === AFTER (clean eval) ===
    if eval_collator is not None:
        trainer.data_collator = eval_collator
    val_after = trainer.evaluate()
    report["val_after"] = val_after
    if test_ds is not None:
        test_after = trainer.evaluate(test_ds)
        report["test_after"] = test_after

        # Optional robustness: evaluate with noise-aware collator
        if args.noise_config and train_collator is not None:
            if hasattr(train_collator, "rng"):
                train_collator.rng.seed(args.seed)
            trainer.data_collator = train_collator
            test_after_noise = trainer.evaluate(test_ds)
            report["test_after_noise_aware"] = test_after_noise
            trainer.data_collator = eval_collator  # restore

    # 5) Save report
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "training_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Saved:", os.path.join(args.out_dir, "training_report.json"))

if __name__ == "__main__":
    main()
