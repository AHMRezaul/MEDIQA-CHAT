"""
Evaluation for MTS-Dialog Task.

MTS-Dialog metrics:
  - section_header_accuracy (exact match, case-insensitive)
  - ROUGE-1, BERTScore (deberta-xlarge-mnli), BLEURT-20


Usage:
  python src/scorer.py \
    --task mts \
    --predictions results/mts_test_preds.csv \
    --output results/mts_test_scores.json
"""

import argparse
import json
import os
import sys

import evaluate as hf_evaluate
import numpy as np
import pandas as pd

# ── Section header accuracy (MTS-Dialog) ─────────────────────────────────────

def header_accuracy(preds: list[str], refs: list[str]) -> float:
    correct = sum(p.strip().lower() == r.strip().lower() for p, r in zip(preds, refs))
    return correct / len(refs) if refs else 0.0


# ── Metric loaders (lazy, cached) ────────────────────────────────────────────

_rouge = None
_bertscore_scorer = None
_bleurt = None


def _get_rouge():
    global _rouge
    if _rouge is None:
        _rouge = hf_evaluate.load("rouge")
    return _rouge


def _get_bleurt():
    global _bleurt
    if _bleurt is None:
        _bleurt = hf_evaluate.load("bleurt", config_name="BLEURT-20")
    return _bleurt


def compute_rouge(preds: list[str], refs: list[str]) -> dict:
    results = _get_rouge().compute(predictions=preds, references=refs, use_aggregator=True)
    return {k: float(v) for k, v in results.items()}


def compute_bertscore(preds: list[str], refs: list[str]) -> dict:
    import bert_score as _bs
    global _bertscore_scorer
    if _bertscore_scorer is None:
        _bertscore_scorer = _bs.BERTScorer(model_type="microsoft/deberta-xlarge-mnli")
        tok = _bertscore_scorer._tokenizer
        # transformers 5.x removed build_inputs_with_special_tokens from all tokenizers.
        # bert_score v0.3.x calls it to detect special token positions, so patch it back.
        if not callable(getattr(tok, "build_inputs_with_special_tokens", None)):
            tok.build_inputs_with_special_tokens = lambda ids_0, ids_1=None: (
                ([tok.cls_token_id] if tok.cls_token_id is not None else [])
                + list(ids_0)
                + ([tok.sep_token_id] if tok.sep_token_id is not None else [])
            )
        if getattr(tok, "model_max_length", 0) > 100_000:
            tok.model_max_length = 512
    P, R, F = _bertscore_scorer.score(preds, refs)
    return {
        "bertscore_precision": float(P.mean()),
        "bertscore_recall": float(R.mean()),
        "bertscore_f1": float(F.mean()),
    }


def compute_bleurt(preds: list[str], refs: list[str]) -> dict:
    results = _get_bleurt().compute(predictions=preds, references=refs)
    return {"bleurt": float(np.mean(results["scores"]))}


def compute_all_metrics(preds: list[str], refs: list[str]) -> dict:
    metrics = {}
    metrics.update(compute_rouge(preds, refs))
    metrics.update(compute_bertscore(preds, refs))
    metrics.update(compute_bleurt(preds, refs))
    return metrics


# ── MTS-Dialog evaluation ─────────────────────────────────────────────────────

def evaluate_mts(df: pd.DataFrame) -> dict:
    pred_headers = df["pred_section_header"].fillna("").tolist()
    ref_headers = df["ref_section_header"].fillna("").tolist()
    pred_texts = df["pred_section_text"].fillna("").tolist()
    ref_texts = df["ref_section_text"].fillna("").tolist()

    results = {
        "num_examples": len(df),
        "section_header_accuracy": header_accuracy(pred_headers, ref_headers),
    }
    results.update(compute_all_metrics(pred_texts, ref_texts))

    results["aggregate_score"] = float(np.mean([
        results["rouge1"],
        results["bertscore_f1"],
        results["bleurt"],
    ]))

    header_correct = {}
    for pred, ref in zip(pred_headers, ref_headers):
        ref_norm = ref.strip().lower()
        if ref_norm not in header_correct:
            header_correct[ref_norm] = {"correct": 0, "total": 0}
        header_correct[ref_norm]["total"] += 1
        if pred.strip().lower() == ref_norm:
            header_correct[ref_norm]["correct"] += 1
    results["per_header_accuracy"] = {
        k: v["correct"] / v["total"] for k, v in header_correct.items()
    }

    return results


# ── ACI-Bench evaluation ──────────────────────────────────────────────────────

_SECTION_TAGGER = None

def _get_section_tagger():
    global _SECTION_TAGGER
    if _SECTION_TAGGER is None:
        tagger_dir = "/scratch/akarim9/mediqa-chat/dataset/aci/baselines"
        sys.path.insert(0, tagger_dir)
        from sectiontagger import SectionTagger
        _SECTION_TAGGER = SectionTagger()
    return _SECTION_TAGGER


ACI_DIVISIONS = ["subjective", "objective_exam", "objective_results", "assessment_and_plan"]


def _split_note_by_sections(text: str) -> dict[str, str]:
    tagger = _get_section_tagger()
    text_nl = text.replace("__lf1__", "\n")
    divisions = tagger.divide_note_by_metasections(text_nl)
    sections = {}
    for label, _, _, start, _, end in divisions:
        sections[label] = text_nl[start:end]
    return sections


def evaluate_aci(df: pd.DataFrame) -> dict:
    preds = df["pred_note"].fillna("").tolist()
    refs = df["ref_note"].fillna("").tolist()
    n = len(df)

    results = {"num_examples": n}

    full_metrics = compute_all_metrics(preds, refs)
    results["full_note"] = full_metrics

    for division in ACI_DIVISIONS:
        div_preds, div_refs = [], []
        for pred, ref in zip(preds, refs):
            pred_secs = _split_note_by_sections(pred)
            ref_secs = _split_note_by_sections(ref)
            div_preds.append(pred_secs.get(division, ""))
            div_refs.append(ref_secs.get(division, ""))
        try:
            results[f"section_{division}"] = compute_all_metrics(div_preds, div_refs)
        except Exception as e:
            results[f"section_{division}"] = {"error": str(e)}

    results["aggregate_score"] = float(np.mean([
        full_metrics["rouge1"],
        full_metrics["bertscore_f1"],
        full_metrics["bleurt"],
    ]))

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["mts", "aci"], required=True)
    p.add_argument("--predictions", required=True, help="CSV file from inference.py")
    p.add_argument("--output", required=True, help="JSON file to write scores to")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.predictions)

    if args.task == "mts":
        scores = evaluate_mts(df)
    else:
        scores = evaluate_aci(df)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(scores, f, indent=2)

    print(f"Results saved → {args.output}")

    print("\n── Summary ──")
    if args.task == "mts":
        print(f"  Header accuracy : {scores['section_header_accuracy']:.4f}")
        print(f"  ROUGE-1         : {scores['rouge1']:.4f}")
        print(f"  ROUGE-2         : {scores['rouge2']:.4f}")
        print(f"  ROUGE-L         : {scores['rougeL']:.4f}")
        print(f"  ROUGE-Lsum      : {scores['rougeLsum']:.4f}")
        print(f"  BERTScore P     : {scores['bertscore_precision']:.4f}")
        print(f"  BERTScore R     : {scores['bertscore_recall']:.4f}")
        print(f"  BERTScore F1    : {scores['bertscore_f1']:.4f}")
        print(f"  BLEURT          : {scores['bleurt']:.4f}")
        print(f"  Aggregate       : {scores['aggregate_score']:.4f}")
    else:
        fn = scores["full_note"]
        print(f"  ROUGE-1         : {fn['rouge1']:.4f}")
        print(f"  ROUGE-2         : {fn['rouge2']:.4f}")
        print(f"  ROUGE-L         : {fn['rougeL']:.4f}")
        print(f"  ROUGE-Lsum      : {fn['rougeLsum']:.4f}")
        print(f"  BERTScore P     : {fn['bertscore_precision']:.4f}")
        print(f"  BERTScore R     : {fn['bertscore_recall']:.4f}")
        print(f"  BERTScore F1    : {fn['bertscore_f1']:.4f}")
        print(f"  BLEURT          : {fn['bleurt']:.4f}")
        print(f"  Aggregate       : {scores['aggregate_score']:.4f}")


if __name__ == "__main__":
    main()
