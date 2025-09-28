# src/metrics.py
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

def compute_metrics_builder(id2label: dict):
    """Token-level seqeval metrics with -100 masking respected."""
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        true_labels, true_preds = [], []
        for p_seq, l_seq in zip(preds, labels):
            tl, tp = [], []
            for p, l in zip(p_seq, l_seq):
                if l == -100:
                    continue
                tl.append(id2label[int(l)])
                tp.append(id2label[int(p)])
            true_labels.append(tl)
            true_preds.append(tp)

        return {
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
            "f1": f1_score(true_labels, true_preds),
        }
    return compute_metrics