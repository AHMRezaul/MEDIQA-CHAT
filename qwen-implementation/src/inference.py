"""
Inference for section-aware clinical dialogue summarization.

Modes:
  --mode zero_shot   : no fine-tuning, prompt-only with base model
  --mode finetuned   : load LoRA checkpoint from --model_dir
  --mode retrieval   : zero-shot with retrieved few-shot examples (see retrieval.py)

Output: CSV with columns [id, dialogue, section_header (pred), section_text (pred),
                          reference_header, reference_text]  for MTS

Usage:
  python src/inference.py \
    --task mts \
    --mode finetuned \
    --model_dir runs/mts_lora/best \
    --split test \
    --output_file results/mts_test_preds.csv
"""

import argparse
import os
import re

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import (
    SYSTEM_ACI,
    SYSTEM_MTS,
    get_test_examples,
    get_val_examples,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["mts", "aci"], default="mts")
    p.add_argument("--mode", choices=["zero_shot", "finetuned", "retrieval"], default="finetuned")
    p.add_argument("--model_id", default="meta-llama/Llama-3.3-70B-Instruct")
    p.add_argument("--model_dir", default=None, help="Path to LoRA checkpoint dir")
    p.add_argument("--split", default="test1",
                   help="For MTS: test1 / test2 / validation. For ACI: test1 / test2 / test3 / validation")
    p.add_argument("--output_file", default="results/predictions.csv")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--retrieval_index", default=None, help="Path to retrieval index (for retrieval mode)")
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_model(args):
    tok_source = args.model_dir if args.mode == "finetuned" else args.model_id
    print(f"[1/3] Loading tokenizer from {tok_source} …")
    tokenizer = AutoTokenizer.from_pretrained(
        tok_source,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad for batch generation

    print(f"[2/3] Loading base model {args.model_id} in bf16 …")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )

    if args.mode == "finetuned":
        print(f"[3/3] Merging LoRA weights from {args.model_dir} …")
        model = PeftModel.from_pretrained(base, args.model_dir)
        model = model.merge_and_unload()
        print("      LoRA merge complete.")
    else:
        print("[3/3] Using base model (no LoRA).")
        model = base

    model.eval()
    return model, tokenizer


def parse_mts_output(text: str) -> tuple[str, str]:
    """Extract section header and summary from model output."""
    header, summary = "", ""
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("section:"):
            header = line[len("section:"):].strip()
        elif line.lower().startswith("summary:"):
            summary = line[len("summary:"):].strip()
    # fallback: grab everything after the first newline as summary
    if not header and not summary:
        parts = text.strip().split("\n", 1)
        header = parts[0].strip()
        summary = parts[1].strip() if len(parts) > 1 else ""
    return header, summary



def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    device: str,
) -> list[str]:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=3072,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    results = []
    for i, seq in enumerate(out):
        prompt_len = enc["input_ids"].shape[1]
        generated = seq[prompt_len:]
        results.append(tokenizer.decode(generated, skip_special_tokens=True).strip())
    return results


def run_inference(args):
    model, tokenizer = load_model(args)
    system = SYSTEM_MTS if args.task == "mts" else SYSTEM_ACI

    if args.split in ("test", "test1", "test2", "test3"):
        aci_split = args.split if args.split.startswith("test") else "test1"
        mts_split = args.split if args.split in ("test1", "test2") else "test1"
        examples = get_test_examples(dataset=args.task, mts_split=mts_split, aci_split=aci_split)
    else:
        examples = get_val_examples(dataset=args.task)

    retriever = None
    if args.retrieval_index:
        from retrieval import load_retriever
        retriever = load_retriever(args.retrieval_index)

    prompts = []
    for ex in examples:
        few_shots = retriever.retrieve(ex["dialogue"], k=args.top_k) if retriever else None

        if few_shots:
            shots_text = []
            for fs in few_shots:
                if args.task == "mts":
                    shots_text.append(
                        f"Dialogue:\n{fs['dialogue']}\nSection: {fs['section_header']}\nSummary: {fs['section_text']}"
                    )
                else:
                    shots_text.append(f"Transcript:\n{fs['dialogue']}\nNote:\n{fs['note']}")
            context = "\n\n---\n\n".join(shots_text) + "\n\n---\n\n"
            user_content = context + ex["user"]
        else:
            user_content = ex["user"]

        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        prompts.append(
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    n_batches = (len(examples) + args.batch_size - 1) // args.batch_size
    print(f"\nRunning inference on {len(examples)} examples ({n_batches} batches, batch_size={args.batch_size})")

    records = []
    gen_bar = tqdm(
        range(0, len(examples), args.batch_size),
        total=n_batches,
        desc="Generating",
        dynamic_ncols=True,
        unit="batch",
    )
    for i in gen_bar:
        batch_ex = examples[i : i + args.batch_size]
        batch_prompts = prompts[i : i + args.batch_size]
        outputs = generate_batch(model, tokenizer, batch_prompts, args.max_new_tokens, args.device)
        gen_bar.set_postfix(done=f"{min(i + args.batch_size, len(examples))}/{len(examples)}")

        for ex, raw_out in zip(batch_ex, outputs):
            rec = {"id": ex.get("file", i), "dialogue": ex["dialogue"]}
            if args.task == "mts":
                pred_header, pred_text = parse_mts_output(raw_out)
                rec.update({
                    "pred_section_header": pred_header,
                    "pred_section_text": pred_text,
                    "ref_section_header": ex["section_header"],
                    "ref_section_text": ex["section_text"],
                    "raw_output": raw_out,
                })
            else:
                rec.update({
                    "pred_note": raw_out,
                    "ref_note": ex["note"],
                })
            records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(args.output_file, index=False)
    print(f"Saved {len(df)} predictions → {args.output_file}")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
