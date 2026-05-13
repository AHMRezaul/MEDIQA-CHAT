"""
Retrieval-based few-shot prompting for clinical dialogue summarization.

Encodes all training examples with a sentence-transformer, builds a FAISS index,
and retrieves the top-k most similar training examples at inference time.

Usage:
  # Build index from MTS-Dialog training set
  python src/retrieval.py build \
    --task mts \
    --index_path retrieval_index/mts_train.faiss

  # Query (smoke test)
  python src/retrieval.py query \
    --index_path retrieval_index/mts_train.faiss \
    --query "Patient complains of chest pain for 3 days."
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from data_utils import (
    aci_to_examples,
    get_train_examples,
    load_aci_bench,
    load_mts_dialog,
    mts_to_examples,
)

_DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


class DialogueRetriever:
    """
    Dense retriever over a fixed set of training examples.
    Encodes dialogues with a sentence-transformer; retrieval is cosine-similarity search.
    """

    def __init__(self, encoder_name: str = _DEFAULT_ENCODER, device: str = "cpu"):
        self.encoder_name = encoder_name
        self.encoder = SentenceTransformer(encoder_name, device=device)
        self.examples: list[dict] = []
        self.embeddings: np.ndarray | None = None

    def build(self, examples: list[dict], batch_size: int = 64, show_progress: bool = True):
        self.examples = examples
        dialogues = [ex["dialogue"] for ex in examples]
        print(f"Encoding {len(dialogues)} training dialogues with {self.encoder_name}…")
        self.embeddings = self.encoder.encode(
            dialogues,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """Return the top-k most similar training examples to the query dialogue."""
        if self.embeddings is None:
            raise RuntimeError("Call build() before retrieve().")
        q_emb = self.encoder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )
        scores = (self.embeddings @ q_emb.T).squeeze(-1)
        top_k_idx = np.argsort(scores)[::-1][:k]
        return [self.examples[i] for i in top_k_idx]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "examples": self.examples,
                "embeddings": self.embeddings,
                "encoder_name": self.encoder_name,
            }, f)
        print(f"Index saved → {path}")

    @classmethod
    def load(cls, path: str, encoder_name: str = _DEFAULT_ENCODER, device: str = "cpu") -> "DialogueRetriever":
        with open(path, "rb") as f:
            data = pickle.load(f)
        saved_encoder = data.get("encoder_name", None)
        if saved_encoder and saved_encoder != encoder_name:
            print(
                f"WARNING: index was built with '{saved_encoder}' "
                f"but loading with '{encoder_name}'. Rebuild the index to avoid mismatch."
            )
            encoder_name = saved_encoder
        retriever = cls(encoder_name=encoder_name, device=device)
        retriever.examples = data["examples"]
        retriever.embeddings = data["embeddings"]
        print(f"Loaded retrieval index ({len(retriever.examples)} examples, encoder: {encoder_name}) from {path}")
        return retriever


def load_retriever(path: str, device: str = "cpu") -> DialogueRetriever:
    return DialogueRetriever.load(path, device=device)


# ── CLI ───────────────────────────────────────────────────────────────────────

def cmd_build(args):
    if args.task == "mts":
        examples = mts_to_examples(load_mts_dialog("train"))
    else:
        examples = aci_to_examples(load_aci_bench("train", "full"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    retriever = DialogueRetriever(encoder_name=args.encoder, device=device)
    retriever.build(examples)
    retriever.save(args.index_path)


def cmd_query(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    retriever = load_retriever(args.index_path, device=device)
    results = retriever.retrieve(args.query, k=args.top_k)
    for i, ex in enumerate(results, 1):
        print(f"\n─── Result {i} ───")
        print(f"Dialogue (first 200 chars): {ex['dialogue'][:200]}")
        if ex["task"] == "mts":
            print(f"Section: {ex['section_header']}")
            print(f"Summary: {ex['section_text'][:200]}")
        else:
            print(f"Note (first 300 chars): {ex['note'][:300]}")


def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")

    build_p = sub.add_parser("build")
    build_p.add_argument("--task", choices=["mts", "aci"], default="mts")
    build_p.add_argument("--index_path", default="retrieval_index/mts_train.pkl")
    build_p.add_argument("--encoder", default=_DEFAULT_ENCODER)

    query_p = sub.add_parser("query")
    query_p.add_argument("--index_path", required=True)
    query_p.add_argument("--query", required=True)
    query_p.add_argument("--top_k", type=int, default=3)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "build":
        cmd_build(args)
    elif args.command == "query":
        cmd_query(args)
    else:
        print("Usage: python src/retrieval.py {build,query} [options]")
