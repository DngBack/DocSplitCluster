#!/usr/bin/env python3
"""Page-level semantic boundary baseline using ColSmol embeddings.

Pipeline:
1) Load one PDF entry from `data/grouth_truth.json`
2) Render each page to PIL image
3) Compute one embedding per page with `vidore/colSmol-256M`
4) Score adjacent-page continuity using cosine similarity
5) Predict boundaries by threshold
6) Compute hard + soft boundary metrics
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    import torch
    from PIL import Image


DEFAULT_MODEL = "vidore/colSmol-256M"
DEFAULT_GT = "data/grouth_truth.json"
DEFAULT_PDF_ROOT = "data/files"
DEFAULT_FILE_NAME = "eng_diff_topics/eng_diff_topics_top4_shuffled.pdf"


@dataclass
class Sample:
    file_name: str
    pdf_path: Path
    gt_page_end_labels: List[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run page-level semantic boundary baseline with ColSmol embeddings."
    )
    parser.add_argument("--ground-truth", default=DEFAULT_GT, help="Path to grouth_truth.json")
    parser.add_argument(
        "--pdf-root",
        default=DEFAULT_PDF_ROOT,
        help="Root folder for PDF files referenced by file_name in GT",
    )
    parser.add_argument(
        "--file-name",
        default=DEFAULT_FILE_NAME,
        help="file_name entry in grouth_truth.json to evaluate",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="HF model id")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PDF render DPI. 120-180 is usually enough for this baseline.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Boundary threshold on cosine score (< threshold => cut). "
        "If omitted, script searches best threshold from scores + GT.",
    )
    parser.add_argument(
        "--soft-window",
        type=int,
        default=1,
        help="Allowed page offset for soft boundary matching.",
    )
    parser.add_argument("--max-chunk-pages", type=int, default=4, help="Hard max pages per chunk.")
    parser.add_argument(
        "--report-path",
        default="data/reports/eng_diff_topics_top4_metrics.json",
        help="Where to save metrics JSON report.",
    )
    return parser.parse_args()


def load_sample(gt_path: Path, pdf_root: Path, target_file_name: str) -> Sample:
    records = json.loads(gt_path.read_text(encoding="utf-8"))
    selected = None
    for item in records:
        if item.get("file_name") == target_file_name:
            selected = item
            break
    if selected is None:
        available = [item.get("file_name", "<unknown>") for item in records]
        raise ValueError(
            f"file_name='{target_file_name}' not found in {gt_path}. "
            f"Available entries: {available}"
        )
    labels = selected.get("grouth truth")
    if not isinstance(labels, list) or not labels:
        raise ValueError("Invalid 'grouth truth' labels in selected entry.")

    pdf_path = pdf_root / target_file_name
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing PDF: {pdf_path}")
    return Sample(file_name=target_file_name, pdf_path=pdf_path, gt_page_end_labels=labels)


def render_pdf_pages(pdf_path: Path, dpi: int) -> List[Image.Image]:
    import fitz  # type: ignore[import-not-found]  # pymupdf
    from PIL import Image

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pages: List[Image.Image] = []
    doc = fitz.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(img)
    doc.close()
    return pages


def choose_device(device_arg: str) -> torch.device:
    import torch

    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    import torch

    moved = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def pool_embeddings(tensor: torch.Tensor) -> torch.Tensor:
    import torch

    # Accept [B, D] or [B, T, D]
    if tensor.ndim == 2:
        emb = tensor
    elif tensor.ndim == 3:
        emb = tensor.mean(dim=1)
    else:
        raise ValueError(f"Unexpected embedding shape: {tuple(tensor.shape)}")
    emb = torch.nn.functional.normalize(emb.float(), dim=-1)
    return emb


def infer_embeddings_from_output(output) -> torch.Tensor:
    import torch

    # Common patterns for vision-language models / custom heads
    for attr in ("embeddings", "image_embeds", "last_hidden_state", "pooler_output"):
        if hasattr(output, attr):
            value = getattr(output, attr)
            if isinstance(value, torch.Tensor):
                return pool_embeddings(value)
    if isinstance(output, torch.Tensor):
        return pool_embeddings(output)
    raise ValueError("Could not infer embeddings from model output.")


def embed_pages(
    pages: Sequence[Image.Image],
    model_name: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoProcessor  # type: ignore[import-not-found]

    processor = None
    model = None

    # Prefer native colpali_engine loaders for ColSmol/ColIdefics3 checkpoints.
    # This avoids PEFT adapter mismatches when loading via generic AutoModel.
    try:
        from colpali_engine.models import BiIdefics3, BiIdefics3Processor  # type: ignore[import-not-found]

        processor = BiIdefics3Processor.from_pretrained(model_name)
        model = BiIdefics3.from_pretrained(model_name)
    except Exception:
        # Fallback for non-colpali checkpoints.
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    model = model.to(device)
    model.eval()

    all_embeddings: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(pages), batch_size):
            batch_pages = pages[start : start + batch_size]
            if hasattr(processor, "process_images"):
                inputs = processor.process_images(list(batch_pages))
            else:
                inputs = processor(images=list(batch_pages), return_tensors="pt")
            inputs = move_to_device(inputs, device)

            if hasattr(model, "get_image_features"):
                try:
                    output = model.get_image_features(**inputs)
                except TypeError:
                    if "pixel_values" not in inputs:
                        raise
                    output = model.get_image_features(pixel_values=inputs["pixel_values"])
            else:
                output = model(**inputs)

            emb = infer_embeddings_from_output(output)
            all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def cosine_adjacent(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.shape[0] < 2:
        return np.array([], dtype=np.float32)
    sims = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    return sims.astype(np.float32)


def apply_chunk_constraints(edge_pred: np.ndarray, max_chunk_pages: int) -> np.ndarray:
    """Force a boundary when current chunk reaches max length."""
    constrained = edge_pred.copy()
    run_len = 1
    for idx in range(len(constrained)):
        if constrained[idx] == 1:
            run_len = 1
            continue
        if run_len >= max_chunk_pages:
            constrained[idx] = 1
            run_len = 1
        else:
            run_len += 1
    return constrained


def page_end_to_edge_labels(page_end_labels: Sequence[int]) -> np.ndarray:
    # page_end labels have length N pages; edge labels have length N-1.
    if len(page_end_labels) < 2:
        return np.array([], dtype=np.int64)
    return np.asarray(page_end_labels[:-1], dtype=np.int64)


def edge_to_page_end_labels(edge_labels: np.ndarray) -> np.ndarray:
    # Always add final page boundary.
    if edge_labels.size == 0:
        return np.array([1], dtype=np.int64)
    return np.concatenate([edge_labels.astype(np.int64), np.array([1], dtype=np.int64)], axis=0)


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def hard_boundary_metrics(gt_edge: np.ndarray, pred_edge: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum((gt_edge == 1) & (pred_edge == 1)))
    fp = int(np.sum((gt_edge == 0) & (pred_edge == 1)))
    fn = int(np.sum((gt_edge == 1) & (pred_edge == 0)))
    p, r, f1 = precision_recall_f1(tp, fp, fn)
    over_cut = fp / max(int(np.sum(gt_edge == 0)), 1)
    miss_cut = fn / max(int(np.sum(gt_edge == 1)), 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": p,
        "recall": r,
        "f1": f1,
        "over_cut_rate": over_cut,
        "miss_cut_rate": miss_cut,
    }


def soft_boundary_metrics(gt_edge: np.ndarray, pred_edge: np.ndarray, window: int) -> Dict[str, float]:
    gt_pos = np.where(gt_edge == 1)[0].tolist()
    pred_pos = np.where(pred_edge == 1)[0].tolist()
    used_gt = set()
    soft_tp = 0
    soft_score_sum = 0.0

    for pidx in pred_pos:
        candidates = [g for g in gt_pos if g not in used_gt and abs(g - pidx) <= window]
        if not candidates:
            continue
        best = min(candidates, key=lambda g: abs(g - pidx))
        dist = abs(best - pidx)
        # Exponential decay gives partial credit for near hits.
        sigma = max(window, 1)
        credit = math.exp(-dist / sigma)
        soft_score_sum += credit
        soft_tp += 1
        used_gt.add(best)

    fp = len(pred_pos) - soft_tp
    fn = len(gt_pos) - soft_tp
    p, r, f1 = precision_recall_f1(soft_tp, fp, fn)
    mean_match_credit = soft_score_sum / max(soft_tp, 1)
    return {
        "window": int(window),
        "soft_tp": soft_tp,
        "soft_fp": fp,
        "soft_fn": fn,
        "soft_precision": p,
        "soft_recall": r,
        "soft_f1": f1,
        "mean_match_credit": mean_match_credit,
    }


def pick_threshold(scores: np.ndarray, gt_edge: np.ndarray, max_chunk_pages: int) -> float:
    if scores.size == 0:
        return 0.5
    candidates = sorted(set(scores.tolist()))
    # Include two outer points to avoid edge-only solutions.
    candidates = [min(candidates) - 1e-4] + candidates + [max(candidates) + 1e-4]

    best_tau = candidates[0]
    best_f1 = -1.0
    for tau in candidates:
        pred_edge = (scores < tau).astype(np.int64)
        pred_edge = apply_chunk_constraints(pred_edge, max_chunk_pages=max_chunk_pages)
        f1 = hard_boundary_metrics(gt_edge, pred_edge)["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau
    return float(best_tau)


def run(args: argparse.Namespace) -> Dict[str, object]:
    gt_path = Path(args.ground_truth)
    pdf_root = Path(args.pdf_root)
    report_path = Path(args.report_path)

    sample = load_sample(gt_path, pdf_root, args.file_name)
    pages = render_pdf_pages(sample.pdf_path, dpi=args.dpi)
    if len(sample.gt_page_end_labels) != len(pages):
        raise ValueError(
            f"GT labels length ({len(sample.gt_page_end_labels)}) != page count ({len(pages)}) "
            f"for {sample.file_name}"
        )

    device = choose_device(args.device)
    embeddings = embed_pages(
        pages=pages,
        model_name=args.model_name,
        device=device,
        batch_size=args.batch_size,
    )
    if embeddings.shape[0] != len(pages):
        raise RuntimeError("Embedding rows do not match page count.")

    scores = cosine_adjacent(embeddings)
    gt_edge = page_end_to_edge_labels(sample.gt_page_end_labels)
    if gt_edge.shape[0] != scores.shape[0]:
        raise RuntimeError("GT edge labels length does not match number of adjacency scores.")

    tau = args.threshold if args.threshold is not None else pick_threshold(
        scores=scores, gt_edge=gt_edge, max_chunk_pages=args.max_chunk_pages
    )

    pred_edge = (scores < tau).astype(np.int64)
    pred_edge = apply_chunk_constraints(pred_edge, max_chunk_pages=args.max_chunk_pages)
    pred_page_end = edge_to_page_end_labels(pred_edge).tolist()

    hard = hard_boundary_metrics(gt_edge, pred_edge)
    soft = soft_boundary_metrics(gt_edge, pred_edge, window=args.soft_window)

    report = {
        "config": {
            "model_name": args.model_name,
            "device": str(device),
            "batch_size": args.batch_size,
            "dpi": args.dpi,
            "threshold": tau,
            "max_chunk_pages": args.max_chunk_pages,
            "soft_window": args.soft_window,
            "ground_truth_path": str(gt_path),
            "pdf_root": str(pdf_root),
            "file_name": sample.file_name,
        },
        "counts": {
            "num_pages": len(pages),
            "num_edges": int(scores.shape[0]),
            "num_gt_boundaries": int(np.sum(gt_edge)),
            "num_pred_boundaries": int(np.sum(pred_edge)),
        },
        "metrics": {"hard": hard, "soft": soft},
        "diagnostics": {
            "adjacent_cosine_scores": scores.tolist(),
            "gt_page_end_labels": sample.gt_page_end_labels,
            "pred_page_end_labels": pred_page_end,
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    report = run(args)

    hard = report["metrics"]["hard"]
    soft = report["metrics"]["soft"]
    cfg = report["config"]
    print(f"Model: {cfg['model_name']}")
    print(f"File: {cfg['file_name']}")
    print(f"Threshold: {cfg['threshold']:.6f}")
    print("--- Hard Boundary ---")
    print(
        "P={:.4f} R={:.4f} F1={:.4f} | over_cut={:.4f} miss_cut={:.4f}".format(
            hard["precision"],
            hard["recall"],
            hard["f1"],
            hard["over_cut_rate"],
            hard["miss_cut_rate"],
        )
    )
    print("--- Soft Boundary ---")
    print(
        "window={} | P={:.4f} R={:.4f} F1={:.4f} credit={:.4f}".format(
            soft["window"],
            soft["soft_precision"],
            soft["soft_recall"],
            soft["soft_f1"],
            soft["mean_match_credit"],
        )
    )
    print("Saved report to:", args.report_path)


if __name__ == "__main__":
    main()
