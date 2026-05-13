"""
LoRA fine-tuning of Llama-3.3-8B-Instruct on MTS-Dialog.

Usage:
  accelerate launch --num_processes 4 src/train.py --config configs/train_config.yaml

Supports:
  - Full LoRA in bf16 (default)
  - Optional 4-bit QLoRA via --load_in_4bit
  - MTS-Dialog
"""

import argparse
import math
import os
from functools import partial

# Must be set before any CUDA allocation to reduce fragmentation-driven OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml
from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

from data_utils import (
    SummarizationDataset,
    get_train_examples,
    get_val_examples,
    make_collate_fn,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_config.yaml")
    # allow CLI overrides of any config key
    p.add_argument("--model_id", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--use_mts", action="store_true", default=None)
    p.add_argument("--use_aci", action="store_true", default=None)
    p.add_argument("--no_mts_augmented", action="store_true", default=False,
                   help="Disable augmented MTS-Dialog training data (included by default)")
    p.add_argument("--load_in_4bit", action="store_true", default=False)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--per_device_batch_size", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--max_length", type=int, default=None)
    p.add_argument("--lora_r", type=int, default=None)
    p.add_argument("--lora_alpha", type=int, default=None)
    p.add_argument("--lora_dropout", type=float, default=None)
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model_and_tokenizer(cfg: dict, load_in_4bit: bool):
    model_id = cfg["model_id"]

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map=None,  # accelerate handles placement
        token=os.environ.get("HF_TOKEN"),
    )
    model.config.use_cache = False

    if load_in_4bit:
        # Required for QLoRA: enables gradient checkpointing + fp32 layer norms.
        # Without this, activations fill VRAM and cause OOM on large models.
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def train():
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    for key in ["model_id", "output_dir", "epochs", "lr", "per_device_batch_size",
                "grad_accum", "max_length", "lora_r", "lora_alpha", "lora_dropout"]:
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
    if args.use_mts is not None:
        cfg["use_mts"] = args.use_mts
    if args.use_aci is not None:
        cfg["use_aci"] = args.use_aci
    if args.no_mts_augmented:
        cfg["use_mts_augmented"] = False
    elif "use_mts_augmented" not in cfg:
        cfg["use_mts_augmented"] = True

    accelerator = Accelerator(gradient_accumulation_steps=cfg["grad_accum"])

    if accelerator.is_main_process:
        os.makedirs(cfg["output_dir"], exist_ok=True)

    model, tokenizer = build_model_and_tokenizer(cfg, args.load_in_4bit)

    train_examples = get_train_examples(
        use_mts=cfg["use_mts"],
        use_aci=cfg["use_aci"],
        use_mts_augmented=cfg.get("use_mts_augmented", True),
    )
    val_examples = get_val_examples(dataset="mts" if cfg["use_mts"] else "aci")

    train_ds = SummarizationDataset(train_examples, tokenizer, max_length=cfg["max_length"])
    val_ds = SummarizationDataset(val_examples, tokenizer, max_length=cfg["max_length"])

    collate = make_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["per_device_batch_size"], shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["per_device_batch_size"], shuffle=False, collate_fn=collate
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01)
    )

    total_steps = math.ceil(len(train_loader) * cfg["epochs"] / cfg["grad_accum"])
    warmup_steps = int(total_steps * cfg.get("warmup_ratio", 0.05))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = 0.0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{cfg['epochs']} [train]",
            disable=not accelerator.is_main_process,
            dynamic_ncols=True,
        )
        for step, batch in enumerate(train_bar):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.get("max_grad_norm", 1.0))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.detach().float()
            global_step += 1
            avg_loss = epoch_loss / (step + 1)
            lr_now = scheduler.get_last_lr()[0]
            train_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr_now:.2e}", step=global_step)

        # Validation
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch+1}/{cfg['epochs']} [val]  ",
            disable=not accelerator.is_main_process,
            dynamic_ncols=True,
        )
        with torch.no_grad():
            for batch in val_bar:
                outputs = model(**batch)
                val_loss += outputs.loss.detach().float()
                val_bar.set_postfix(val_loss=f"{val_loss / (val_bar.n + 1):.4f}")
        val_loss /= len(val_loader)
        val_loss = accelerator.reduce(val_loss, reduction="mean").item()

        if accelerator.is_main_process:
            print(f"\nepoch {epoch+1}/{cfg['epochs']} | val_loss {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(os.path.join(cfg["output_dir"], "best"))
                tokenizer.save_pretrained(os.path.join(cfg["output_dir"], "best"))
                print(f"  → saved best checkpoint (val_loss={best_val_loss:.4f})")

        # Save per-epoch checkpoint
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(os.path.join(cfg["output_dir"], f"epoch_{epoch+1}"))
            tokenizer.save_pretrained(os.path.join(cfg["output_dir"], f"epoch_{epoch+1}"))

    if accelerator.is_main_process:
        print("Training complete.")


if __name__ == "__main__":
    train()
