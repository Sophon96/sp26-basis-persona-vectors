"""LoRA SFT with optional preventive-steering hook.

The paper (Section 5.2) introduces **preventive steering**: during finetuning, add
α·v at the persona-vector's layer in every forward pass. Intuitively, this
"pre-pays" for the persona shift the training data would otherwise cause — the
model no longer has to move along v to minimize loss, so post-training trait
expression stays low.

This is implemented here by keeping the same forward hook from `persona_lib.hooks`
active across every training forward pass. The hook operates on the *base* model's
residual stream; LoRA adapters update on top of that.

Dataset format: JSONL with either:
    {"messages": [{"role": ..., "content": ...}, ...]}         (preferred)
or  {"user": ..., "assistant": ..., "system": ... (optional)}  (auto-converted)
"""

from __future__ import annotations

import json
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from persona_lib.data import read_jsonl


def _rows_to_messages(row: dict) -> list[dict]:
    if "messages" in row:
        return row["messages"]
    msgs = []
    if row.get("system"):
        msgs.append({"role": "system", "content": row["system"]})
    msgs.append({"role": "user", "content": row["user"]})
    msgs.append({"role": "assistant", "content": row["assistant"]})
    return msgs


def _prep_sft_dataset(path: str | Path, tokenizer, max_seq_length: int):
    from datasets import Dataset

    rows = read_jsonl(path)
    texts = [
        tokenizer.apply_chat_template(_rows_to_messages(r), tokenize=False, add_generation_prompt=False)
        for r in rows
    ]
    return Dataset.from_dict({"text": texts})


@contextmanager
def _maybe_steering(hf_model, preventive: tuple[np.ndarray, int, float] | None) -> Iterator[None]:
    """Wrap a block in a steering hook if `preventive` is set, else a no-op."""
    if preventive is None:
        yield
        return
    from persona_lib.hooks import steering_hook

    vec, layer, alpha = preventive
    vec_t = torch.from_numpy(np.asarray(vec)).float()
    with steering_hook(hf_model, {int(layer): (float(alpha), vec_t)}):
        yield


def finetune(
    model_id: str,
    dataset_path: str,
    *,
    preventive_steering: tuple[np.ndarray, int, float] | None = None,
    output_dir: str | Path = "./runs/adapter",
    num_epochs: int = 1,
    learning_rate: float = 1e-4,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_seq_length: int = 1024,
    dtype: str = "bfloat16",
    seed: int = 0,
) -> str:
    """Finetune `model_id` on `dataset_path` with LoRA; return the adapter path.

    If `preventive_steering=(vector, layer, alpha)` is given, the persona vector is
    added to the residual stream at `layer` during every forward pass throughout
    training (but NOT at inference — the adapter alone is saved).
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=getattr(torch, dtype), device_map="auto"
    )
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(base, lora_cfg)

    train_ds = _prep_sft_dataset(dataset_path, tokenizer, max_seq_length)

    sft_cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_length=max_seq_length,
        logging_steps=10,
        save_strategy="no",
        bf16=(dtype == "bfloat16"),
        fp16=(dtype == "float16"),
        report_to=[],
        seed=seed,
        dataset_text_field="text",
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    # The steering hook must be active for every forward pass during training.
    with _maybe_steering(base, preventive_steering):
        trainer.train()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "preventive_steering.json").write_text(
        json.dumps(
            {
                "layer": preventive_steering[1] if preventive_steering else None,
                "alpha": preventive_steering[2] if preventive_steering else None,
                "dataset": str(dataset_path),
                "base_model": model_id,
            },
            indent=2,
        )
    )
    return str(out)
