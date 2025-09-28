# src/augmentation.py
import random
from typing import List

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def jitter_bboxes(bboxes: List[List[int]], jitter_scale: float, rng: random.Random, prob: float) -> List[List[int]]:
    """
    Jitter in 0–1000 space (LayoutLMv3). Keeps boxes valid and within [0,1000].
    """
    out = []
    for x0, y0, x1, y1 in bboxes:
        if rng.random() < prob:
            w = max(1, x1 - x0)
            h = max(1, y1 - y0)
            dx = int(w * jitter_scale * rng.uniform(-1, 1))
            dy = int(h * jitter_scale * rng.uniform(-1, 1))
            nx0, ny0 = x0 + dx, y0 + dy
            nx1, ny1 = x1 + dx, y1 + dy

            # order + clamp
            if nx1 < nx0: nx0, nx1 = nx1, nx0
            if ny1 < ny0: ny0, ny1 = ny1, ny0
            nx0 = _clamp(nx0, 0, 1000); nx1 = _clamp(nx1, 0, 1000)
            ny0 = _clamp(ny0, 0, 1000); ny1 = _clamp(ny1, 0, 1000)

            # avoid zero-area after clamp/round
            if nx1 <= nx0: nx1 = min(1000, nx0 + 1)
            if ny1 <= ny0: ny1 = min(1000, ny0 + 1)

            out.append([nx0, ny0, nx1, ny1])
        else:
            out.append([x0, y0, x1, y1])
    return out

def random_token_dropout(input_ids, labels, tokenizer_mask_id, drop_prob, max_drop, rng: random.Random):
    """
    Replace up to max_drop tokens with [MASK] (if available) and set labels to -100 for those positions.
    """
    new_input_ids, new_labels = [], []
    dropped = 0
    for tok, lab in zip(input_ids, labels):
        if rng.random() < drop_prob and dropped < max_drop:
            new_input_ids.append(tokenizer_mask_id if tokenizer_mask_id is not None else tok)
            new_labels.append(-100)
            dropped += 1
        else:
            new_input_ids.append(tok)
            new_labels.append(lab)
    return new_input_ids, new_labels
