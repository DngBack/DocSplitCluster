#!/usr/bin/env python3
"""Run boundary split + late-interaction chunk classification in one pass.

Workflow:
1) Split pages using multiview embeddings + local margin (or legacy full-page cosine).
2) Classify each predicted chunk with late interaction (MaxSim) using ColSmol.
3) Save chunk labels and combined metrics (boundary + classification).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple

import numpy as np

from run_boundary_baseline import (
    DEFAULT_FILE_NAME,
    DEFAULT_GT,
    DEFAULT_MODEL,
    DEFAULT_PDF_ROOT,
    choose_device,
    edge_to_page_end_labels,
    hard_boundary_metrics,
    infer_embeddings_from_output,
    load_sample,
    move_to_device,
    page_end_to_edge_labels,
    predict_page_boundaries,
    render_pdf_pages,
    soft_boundary_metrics,
)

if TYPE_CHECKING:
    import torch
    from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split document pages then classify chunks with late interaction MaxSim."
    )
    parser.add_argument("--ground-truth", default=DEFAULT_GT)
    parser.add_argument("--pdf-root", default=DEFAULT_PDF_ROOT)
    parser.add_argument("--file-name", default=DEFAULT_FILE_NAME)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Multiview: boundary when local margin m_i < threshold. "
        "Legacy: when adjacent full-page cosine < threshold.",
    )
    parser.add_argument(
        "--legacy-full-page-cosine",
        action="store_true",
        help="Boundary: single full-page embedding + adjacent cosine (old baseline).",
    )
    parser.add_argument("--min-chunk-pages", type=int, default=1)
    parser.add_argument("--max-chunk-pages", type=int, default=4)
    parser.add_argument("--soft-window", type=int, default=1)
    parser.add_argument(
        "--keywords-json",
        default=None,
        help="Optional JSON file mapping label -> list of query keywords.",
    )
    parser.add_argument(
        "--chunk-labels-path",
        default="outputs/eng_diff_topics_top4_chunk_labels.json",
        help="Output path for per-chunk labels and scores.",
    )
    parser.add_argument(
        "--report-path",
        default="outputs/eng_diff_topics_top4_split_classify_metrics.json",
        help="Output path for combined metrics report.",
    )
    return parser.parse_args()


def load_gt_record(gt_path: Path, target_file_name: str) -> Dict[str, object]:
    records = json.loads(gt_path.read_text(encoding="utf-8"))
    for item in records:
        if item.get("file_name") == target_file_name:
            return item
    raise ValueError(f"file_name='{target_file_name}' not found in {gt_path}")


def build_page_topic_labels(record: Dict[str, object], expected_pages: int) -> List[str]:
    meta = record.get("meta", {})
    segments = meta.get("segments", []) if isinstance(meta, dict) else []
    labels: List[str] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        source = seg.get("source")
        used = seg.get("pages_used")
        if not isinstance(source, str) or not isinstance(used, int):
            continue
        # Example source:
        # data/files/eng_diff_topics/finance/xxx.pdf -> label "finance"
        parts = Path(source).parts
        label = parts[-2] if len(parts) >= 2 else "unknown"
        labels.extend([label] * max(used, 0))

    if len(labels) != expected_pages:
        return ["unknown"] * expected_pages
    return labels


def build_default_keywords(labels: Sequence[str]) -> Dict[str, List[Dict[str, str]]]:
    uniq = sorted(set(labels))
    keywords: Dict[str, List[Dict[str, str]]] = {}
    for label in uniq:
        if label == "finance":
            keywords[label] = [
                {
                    "keyword": "finance report",
                    "description": "Corporate financial reporting pages with monetary values and statements.",
                },
                {
                    "keyword": "financial statement",
                    "description": "Documents containing balance sheet, income statement, or cash-flow content.",
                },
                {
                    "keyword": "accounting document",
                    "description": "Accounting-oriented forms, totals, and period-based financial tables.",
                },
            ]
        elif label == "health":
            keywords[label] = [
                {
                    "keyword": "medical document",
                    "description": "Healthcare-related pages including diagnosis, treatment, or patient guidance.",
                },
                {
                    "keyword": "health fact sheet",
                    "description": "Public health educational sheets and disease/prevention information.",
                },
                {
                    "keyword": "clinical information",
                    "description": "Clinical terminology, screening references, and care instructions.",
                },
            ]
        elif label == "math":
            keywords[label] = [
                {
                    "keyword": "mathematics lecture notes",
                    "description": "Math course pages with definitions, theorems, and derivations.",
                },
                {
                    "keyword": "homework problems",
                    "description": "Exercise sheets containing problem statements and solution structure.",
                },
                {
                    "keyword": "math equations",
                    "description": "Pages dominated by formulas, symbols, and mathematical expressions.",
                },
            ]
        else:
            keywords[label] = [
                {
                    "keyword": label,
                    "description": f"General pages about topic '{label}'.",
                }
            ]
    return keywords


def load_keywords(path: str | None, labels: Sequence[str]) -> Dict[str, List[Dict[str, str]]]:
    defaults = build_default_keywords(labels)
    if path is None:
        return defaults

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("--keywords-json must be object: label -> list[str|object] or str")

    merged = defaults.copy()
    for k, v in raw.items():
        label = str(k)
        if isinstance(v, str):
            merged[label] = [{"keyword": v, "description": f"Custom query for '{label}'."}]
        elif isinstance(v, list):
            normalized_specs: List[Dict[str, str]] = []
            for item in v:
                if isinstance(item, str):
                    normalized_specs.append(
                        {"keyword": item, "description": f"Custom query for '{label}'."}
                    )
                    continue
                if isinstance(item, dict):
                    kw = item.get("keyword")
                    desc = item.get("description")
                    if isinstance(kw, str):
                        normalized_specs.append(
                            {
                                "keyword": kw,
                                "description": desc
                                if isinstance(desc, str)
                                else f"Custom query for '{label}'.",
                            }
                        )
                        continue
                raise ValueError(
                    f"Invalid keyword spec for label '{label}'. "
                    "Each item must be string or object with 'keyword' and optional 'description'."
                )
            merged[label] = normalized_specs
        else:
            raise ValueError(f"Invalid keywords for label '{label}'")
    return merged


def build_keyword_queries(
    keyword_specs: Dict[str, List[Dict[str, str]]],
) -> Dict[str, List[str]]:
    queries: Dict[str, List[str]] = {}
    for label, specs in keyword_specs.items():
        label_queries: List[str] = []
        for spec in specs:
            keyword = spec.get("keyword", "").strip()
            description = spec.get("description", "").strip()
            if not keyword:
                continue
            # Enrich query text with short semantic description.
            if description:
                label_queries.append(f"{keyword}. {description}")
            else:
                label_queries.append(keyword)
        if label_queries:
            queries[label] = label_queries
    return queries


def page_end_to_chunks(page_end_labels: Sequence[int]) -> List[Tuple[int, int]]:
    chunks: List[Tuple[int, int]] = []
    start = 0
    for idx, flag in enumerate(page_end_labels):
        if flag == 1:
            chunks.append((start, idx))
            start = idx + 1
    if start < len(page_end_labels):
        chunks.append((start, len(page_end_labels) - 1))
    return chunks


def normalize_tokens(tokens: "torch.Tensor") -> "torch.Tensor":
    import torch

    return torch.nn.functional.normalize(tokens.float(), dim=-1)


def embed_pages_multivector(
    pages: Sequence["Image.Image"],
    model_name: str,
    device: "torch.device",
    batch_size: int,
) -> List["torch.Tensor"]:
    import torch
    from colpali_engine.models import ColIdefics3, ColIdefics3Processor  # type: ignore[import-not-found]

    processor = ColIdefics3Processor.from_pretrained(model_name)
    model = ColIdefics3.from_pretrained(model_name).to(device)
    model.eval()

    all_page_tokens: List[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(pages), batch_size):
            batch_pages = list(pages[start : start + batch_size])
            inputs = processor.process_images(batch_pages)
            moved = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            output = model(**moved)
            output = normalize_tokens(output)
            for i in range(output.shape[0]):
                all_page_tokens.append(output[i].detach().cpu())
    return all_page_tokens


def embed_keyword_queries(
    keywords: Dict[str, List[str]],
    model_name: str,
    device: "torch.device",
) -> Dict[str, List["torch.Tensor"]]:
    import torch
    from colpali_engine.models import ColIdefics3, ColIdefics3Processor  # type: ignore[import-not-found]

    processor = ColIdefics3Processor.from_pretrained(model_name)
    model = ColIdefics3.from_pretrained(model_name).to(device)
    model.eval()

    out: Dict[str, List[torch.Tensor]] = defaultdict(list)
    with torch.inference_mode():
        for label, phrases in keywords.items():
            if not phrases:
                continue
            inputs = processor.process_queries(phrases)
            moved = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            q_emb = model(**moved)
            q_emb = normalize_tokens(q_emb)
            for i in range(q_emb.shape[0]):
                out[label].append(q_emb[i].detach().cpu())
    return out


def maxsim_score(query_tokens: "torch.Tensor", doc_tokens: "torch.Tensor") -> float:
    import torch

    sim = query_tokens @ doc_tokens.T
    token_max = torch.max(sim, dim=1).values
    # Mean-normalized MaxSim is more stable across queries with different token lengths.
    return float(torch.mean(token_max).item())


def classify_chunk_maxsim_aggregate(
    chunk_page_tokens: Sequence["torch.Tensor"],
    keyword_query_tokens: Dict[str, List["torch.Tensor"]],
) -> Tuple[str, Dict[str, float]]:
    label_scores: Dict[str, float] = {}
    for label, query_vecs in keyword_query_tokens.items():
        if not query_vecs:
            label_scores[label] = float("-inf")
            continue
        # Compute score for each page, then aggregate to one document score.
        # This enforces one label for the whole chunk/document.
        per_page_scores: List[float] = []
        for page_tokens in chunk_page_tokens:
            best_page = max(maxsim_score(qt, page_tokens) for qt in query_vecs)
            per_page_scores.append(best_page)
        label_scores[label] = float(np.mean(per_page_scores)) if per_page_scores else float("-inf")
    pred_label = max(label_scores.items(), key=lambda x: x[1])[0]
    return pred_label, label_scores


def majority_label(labels: Sequence[str]) -> str:
    if not labels:
        return "unknown"
    return Counter(labels).most_common(1)[0][0]


def classification_metrics(gt: Sequence[str], pred: Sequence[str]) -> Dict[str, object]:
    labels = sorted(set(gt) | set(pred))
    correct = sum(1 for g, p in zip(gt, pred) if g == p)
    acc = correct / max(len(gt), 1)

    per_class: Dict[str, Dict[str, float]] = {}
    macro_p = 0.0
    macro_r = 0.0
    macro_f1 = 0.0
    for label in labels:
        tp = sum(1 for g, p in zip(gt, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gt, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gt, pred) if g == label and p != label)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_class[label] = {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        macro_p += p
        macro_r += r
        macro_f1 += f1

    n = max(len(labels), 1)
    confusion: Dict[str, Dict[str, int]] = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(gt, pred):
        confusion[g][p] += 1

    return {
        "accuracy": acc,
        "macro_precision": macro_p / n,
        "macro_recall": macro_r / n,
        "macro_f1": macro_f1 / n,
        "labels": labels,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def run(args: argparse.Namespace) -> Dict[str, object]:
    if args.min_chunk_pages < 1:
        raise ValueError("--min-chunk-pages must be >= 1")
    if args.max_chunk_pages < args.min_chunk_pages:
        raise ValueError("--max-chunk-pages must be >= --min-chunk-pages")

    gt_path = Path(args.ground_truth)
    pdf_root = Path(args.pdf_root)

    sample = load_sample(gt_path, pdf_root, args.file_name)
    record = load_gt_record(gt_path, args.file_name)
    pages = render_pdf_pages(sample.pdf_path, dpi=args.dpi)

    device = choose_device(args.device)

    # ----- Step 1: boundary split (shared with run_boundary_baseline.predict_page_boundaries) -----
    gt_edge = page_end_to_edge_labels(sample.gt_page_end_labels)
    pred_edge, tau, boundary_diag = predict_page_boundaries(
        pages,
        model_name=args.model_name,
        device=device,
        batch_size=args.batch_size,
        gt_edge=gt_edge,
        legacy_full_page_cosine=args.legacy_full_page_cosine,
        threshold=args.threshold,
        min_chunk_pages=args.min_chunk_pages,
        max_chunk_pages=args.max_chunk_pages,
    )
    pred_page_end = edge_to_page_end_labels(pred_edge).tolist()

    boundary_metrics = {
        "hard": hard_boundary_metrics(gt_edge, pred_edge),
        "soft": soft_boundary_metrics(gt_edge, pred_edge, window=args.soft_window),
    }

    # ----- Step 2: Late-interaction classification -----
    page_topic_labels = build_page_topic_labels(record, expected_pages=len(pages))
    keyword_specs = load_keywords(args.keywords_json, labels=page_topic_labels)
    keyword_queries = build_keyword_queries(keyword_specs)

    page_tokens = embed_pages_multivector(
        pages=pages,
        model_name=args.model_name,
        device=device,
        batch_size=args.batch_size,
    )
    keyword_query_tokens = embed_keyword_queries(
        keywords=keyword_queries,
        model_name=args.model_name,
        device=device,
    )

    pred_chunks = page_end_to_chunks(pred_page_end)
    gt_chunk_labels: List[str] = []
    pred_chunk_labels: List[str] = []
    pred_page_topic_labels: List[str] = ["unknown"] * len(pages)
    chunk_rows: List[Dict[str, object]] = []

    for idx, (start, end) in enumerate(pred_chunks):
        chunk_token_list = page_tokens[start : end + 1]
        if not chunk_token_list:
            continue
        pred_label, class_scores = classify_chunk_maxsim_aggregate(
            chunk_token_list, keyword_query_tokens
        )
        gt_label = majority_label(page_topic_labels[start : end + 1])

        gt_chunk_labels.append(gt_label)
        pred_chunk_labels.append(pred_label)
        for p in range(start, end + 1):
            pred_page_topic_labels[p] = pred_label
        chunk_rows.append(
            {
                "chunk_index": idx,
                "start_page": start,
                "end_page": end,
                "num_pages": end - start + 1,
                "gt_label": gt_label,
                "pred_label": pred_label,
                "pred_page_labels": [pred_label] * (end - start + 1),
                "scores": class_scores,
            }
        )

    cls_metrics = classification_metrics(gt_chunk_labels, pred_chunk_labels)

    chunk_labels_path = Path(args.chunk_labels_path)
    chunk_labels_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_labels_payload = {
        "file_name": args.file_name,
        "keywords": keyword_specs,
        "chunks": chunk_rows,
    }
    chunk_labels_path.write_text(
        json.dumps(chunk_labels_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = {
        "config": {
            "file_name": args.file_name,
            "model_name": args.model_name,
            "device": str(device),
            "threshold": tau,
            "batch_size": args.batch_size,
            "dpi": args.dpi,
            "max_chunk_pages": args.max_chunk_pages,
            "min_chunk_pages": args.min_chunk_pages,
            "soft_window": args.soft_window,
            "legacy_full_page_cosine": args.legacy_full_page_cosine,
            "keywords_json": args.keywords_json,
            "keyword_queries": keyword_queries,
        },
        "counts": {
            "num_pages": len(pages),
            "num_pred_chunks": len(pred_chunks),
            "num_gt_boundaries": int(np.sum(gt_edge)),
            "num_pred_boundaries": int(np.sum(pred_edge)),
        },
        "metrics": {
            "boundary": boundary_metrics,
            "classification": cls_metrics,
        },
        "artifacts": {
            "chunk_labels_path": str(chunk_labels_path),
        },
        "diagnostics": {
            **boundary_diag,
            "gt_page_end_labels": sample.gt_page_end_labels,
            "pred_page_end_labels": pred_page_end,
            "page_topic_labels": page_topic_labels,
            "pred_page_topic_labels": pred_page_topic_labels,
        },
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    report = run(args)
    b = report["metrics"]["boundary"]["hard"]
    c = report["metrics"]["classification"]
    print(f"File: {report['config']['file_name']}")
    print("Boundary mode:", report["diagnostics"]["boundary_mode"])
    print(
        "Boundary F1={:.4f} (P={:.4f}, R={:.4f})".format(
            b["f1"], b["precision"], b["recall"]
        )
    )
    print(
        "Classification Acc={:.4f}, Macro-F1={:.4f}".format(
            c["accuracy"], c["macro_f1"]
        )
    )
    print("Chunk labels:", report["artifacts"]["chunk_labels_path"])
    print("Report:", args.report_path)


if __name__ == "__main__":
    main()
