#!/usr/bin/env python3
"""Bounded, log-complete VCSD reproduction on public Qwen3-VL and datasets."""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import os
import random
import re
import time
import zipfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


PAPER_NUMBERS = {
    "base": {"MMStar": 57.47, "MathVista": 62.50, "seven_benchmark": 62.27},
    "opsd": {"MMStar": 60.73, "MathVista": 64.70, "seven_benchmark": 64.89},
    "vcsd": {"MMStar": 63.73, "MathVista": 66.10, "seven_benchmark": 67.04},
}


def rank0_print(obj: Any) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        if isinstance(obj, str):
            print(obj, flush=True)
        else:
            print(json.dumps(obj, sort_keys=True), flush=True)


def setup_dist(seed: int) -> tuple[int, int, torch.device]:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    seed_i = seed + rank
    random.seed(seed_i)
    np.random.seed(seed_i)
    torch.manual_seed(seed_i)
    torch.cuda.manual_seed_all(seed_i)
    return rank, world, torch.device("cuda", local_rank)


def clean_question(text: str) -> str:
    return re.sub(r"<image(?:\s+\d+)?>", "", str(text)).strip()


def make_inputs(processor, image: Image.Image, question: str, device, answer_hint=None):
    text = clean_question(question)
    if answer_hint is not None:
        text += (
            "\n\nPrivileged reference answer for the teacher only: "
            f"{answer_hint}\nUse the hint to assess the next response token."
        )
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": text}],
    }]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}


def append_tokens(inputs: dict[str, torch.Tensor], suffix: torch.Tensor):
    out = dict(inputs)
    out["input_ids"] = torch.cat([inputs["input_ids"], suffix], dim=1)
    out["attention_mask"] = torch.ones_like(out["input_ids"])
    return out


def load_virl_subset(n: int, seed: int):
    parquet = hf_hub_download(
        "TIGER-Lab/ViRL39K", "39Krelease.parquet", repo_type="dataset"
    )
    image_zip = hf_hub_download(
        "TIGER-Lab/ViRL39K", "images.zip", repo_type="dataset"
    )
    frame = pd.read_parquet(parquet)
    frame = frame[frame["image"].map(lambda x: len(x) == 1)].copy()
    # Fixed, public, model-relevant stratum: answered at least once by the
    # annotated 7B base but not always, then deterministic qid ordering.
    eligible = frame[
        (frame["PassRate_7BBase"] > 0.0) & (frame["PassRate_7BBase"] < 1.0)
    ]
    if len(eligible) < n:
        eligible = frame[frame["PassRate_32BTrained"].between(0.1, 0.9)]
    eligible = eligible.sort_values("qid")
    rng = np.random.default_rng(seed)
    chosen = eligible.iloc[np.sort(rng.choice(len(eligible), n, replace=False))]
    archive = zipfile.ZipFile(image_zip)
    rows = []
    for row in chosen.to_dict("records"):
        name = row["image"][0]
        with archive.open(name) as handle:
            image = Image.open(io.BytesIO(handle.read())).convert("RGB")
        rows.append({
            "id": row["qid"],
            "question": row["question"],
            "answer": row["answer"],
            "image": image,
            "category": row["category"],
        })
    return rows


def fixed_eval_rows(name: str, n: int, seed: int):
    if name == "MMStar":
        data = load_dataset("rvanova/MMStar", split="train")
        order = sorted(range(len(data)), key=lambda i: int(data[i]["index"]))
        rng = np.random.default_rng(seed + 11)
        indices = sorted(rng.choice(order, n, replace=False).tolist())
        return [{
            "id": str(data[i]["index"]),
            "question": data[i]["question"],
            "answer": data[i]["answer"],
            "image": data[i]["image"].convert("RGB"),
            "answer_type": "choice",
            "choices": None,
        } for i in indices]
    data = load_dataset("AI4Math/MathVista", split="testmini")
    order = sorted(range(len(data)), key=lambda i: int(data[i]["pid"]))
    rng = np.random.default_rng(seed + 29)
    indices = sorted(rng.choice(order, n, replace=False).tolist())
    return [{
        "id": str(data[i]["pid"]),
        "question": data[i]["query"],
        "answer": str(data[i]["answer"]),
        "image": data[i]["decoded_image"].convert("RGB"),
        "answer_type": data[i]["answer_type"],
        "choices": data[i]["choices"],
    } for i in indices]


def normalize_text(x: str) -> str:
    x = str(x).strip().lower().replace(",", "")
    x = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", x)
    x = re.sub(r"[^a-z0-9.+/%-]+", " ", x)
    return " ".join(x.split())


def score_prediction(pred: str, row: dict[str, Any]) -> bool:
    target = normalize_text(row["answer"])
    norm = normalize_text(pred)
    if row["answer_type"] == "choice":
        letters = re.findall(r"\b([abcd])\b", norm)
        if letters:
            return letters[-1] == target.lower()
    if row.get("choices") and target in [normalize_text(x) for x in row["choices"]]:
        correct_i = [normalize_text(x) for x in row["choices"]].index(target)
        letters = re.findall(r"\b([abcd])\b", norm)
        if letters and ord(letters[-1]) - ord("a") == correct_i:
            return True
    numbers = re.findall(r"[-+]?(?:\d*\.)?\d+(?:/\d+)?%?", norm)
    if re.fullmatch(r"[-+]?(?:\d*\.)?\d+(?:/\d+)?%?", target) and numbers:
        return numbers[-1] == target
    return norm.endswith(target) or f" {target} " in f" {norm} "


@torch.inference_mode()
def evaluate(model, processor, rows, cfg, rank, world, device, benchmark, stage):
    module = model.module if isinstance(model, DDP) else model
    module.eval()
    local = []
    for i in range(rank, len(rows), world):
        row = rows[i]
        inputs = make_inputs(processor, row["image"], row["question"], device)
        ids = module.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=cfg["max_new_tokens_eval"],
            use_cache=True,
        )
        suffix = ids[:, inputs["input_ids"].shape[1]:]
        pred = processor.batch_decode(
            suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        local.append((row["id"], pred, score_prediction(pred, row)))
    gathered = [None for _ in range(world)]
    dist.all_gather_object(gathered, local)
    flat = [x for part in gathered for x in part]
    flat.sort(key=lambda x: x[0])
    correct = sum(x[2] for x in flat)
    result = {
        "event": "evaluation",
        "stage": stage,
        "benchmark": benchmark,
        "correct": correct,
        "n": len(flat),
        "accuracy": 100.0 * correct / len(flat),
        "examples": [{"id": x[0], "prediction": x[1], "correct": x[2]} for x in flat[:5]],
    }
    rank0_print(result)
    module.train()
    return result


def token_diagnostic(model, teacher, processor, rows, cfg, rank, world, device, stage):
    module = model.module if isinstance(model, DDP) else model
    source = teacher if teacher is not None else module
    source.eval()
    values = []
    with torch.inference_mode():
        for i in range(rank, min(len(rows), 16), world):
            row = rows[i]
            orig = make_inputs(processor, row["image"], row["question"], device)
            black = Image.new("RGB", row["image"].size, (0, 0, 0))
            ctrl = make_inputs(processor, black, row["question"], device)
            lo = source(**orig, use_cache=False).logits[:, -1].float()
            lc = source(**ctrl, use_cache=False).logits[:, -1].float()
            po = torch.log_softmax(lo / cfg["temperature_kd"], -1)
            pc = torch.log_softmax(lc / cfg["temperature_kd"], -1)
            delta = po - pc
            plausible = po >= po.max(-1, keepdim=True).values + math.log(0.1)
            positive = delta > 0
            p = po.exp()
            scores = po + delta
            scores = scores.masked_fill(~plausible, -torch.inf)
            q = torch.softmax(scores, -1)
            values.append({
                "positive_contrast_mass_original": float((p * positive).sum()),
                "positive_contrast_mass_shaped": float((q * positive).sum()),
                "mean_abs_contrast": float(delta.abs().mean()),
                "max_abs_contrast": float(delta.abs().max()),
                "plausible_support_size": int(plausible.sum()),
            })
    gathered = [None for _ in range(world)]
    dist.all_gather_object(gathered, values)
    all_values = [x for part in gathered for x in part]
    result = {"event": "token_diagnostic", "stage": stage, "n": len(all_values)}
    for key in all_values[0]:
        result[key] = float(np.mean([x[key] for x in all_values]))
    result["shaping_mass_gain"] = (
        result["positive_contrast_mass_shaped"]
        - result["positive_contrast_mass_original"]
    )
    rank0_print(result)
    return result


def ema_update(teacher, student, rate):
    with torch.no_grad():
        for tp, sp in zip(teacher.parameters(), student.parameters()):
            tp.mul_(1.0 - rate).add_(sp.detach(), alpha=rate)


def train(model, teacher, processor, rows, cfg, rank, world, device):
    module = model.module
    optimizer = torch.optim.AdamW(
        module.parameters(), lr=cfg["learning_rate"], weight_decay=0.0
    )
    def precision_context():
        return (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if torch.cuda.is_bf16_supported() else nullcontext()
        )
    history = []
    eos_ids = {
        x for x in [
            processor.tokenizer.eos_token_id,
            processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        ] if isinstance(x, int) and x >= 0
    }
    for step in range(cfg["train_steps"]):
        module.train()
        idx = (step * world + rank) % len(rows)
        row = rows[idx]
        original = make_inputs(processor, row["image"], row["question"], device)
        module.eval()
        with torch.inference_mode():
            generated = module.generate(
                **original,
                do_sample=True,
                temperature=cfg["sampling_temperature"],
                top_p=cfg["top_p"],
                top_k=cfg["top_k"],
                max_new_tokens=cfg["max_new_tokens_train"],
                use_cache=True,
            )
        suffix = generated[:, original["input_ids"].shape[1]:]
        if suffix.shape[1] == 0:
            continue
        student_inputs = append_tokens(original, suffix)
        condition = cfg["condition"]
        if condition == "opsd":
            teacher_prefix = make_inputs(
                processor, row["image"], row["question"], device, row["answer"]
            )
        else:
            teacher_prefix = original
        teacher_inputs = append_tokens(teacher_prefix, suffix)
        teacher.eval()
        with torch.inference_mode(), precision_context():
            to = teacher(**teacher_inputs, use_cache=False).logits[:, -suffix.shape[1]-1:-1]
            logp_orig = torch.log_softmax(to.float() / cfg["temperature_kd"], -1)
            if condition in {"vcsd", "beta0"}:
                black_image = Image.new("RGB", row["image"].size, (0, 0, 0))
                black_prefix = make_inputs(
                    processor, black_image, row["question"], device
                )
                black_inputs = append_tokens(black_prefix, suffix)
                tc = teacher(**black_inputs, use_cache=False).logits[
                    :, -suffix.shape[1]-1:-1
                ]
                logp_ctrl = torch.log_softmax(
                    tc.float() / cfg["temperature_kd"], -1
                )
                delta = logp_orig - logp_ctrl
                for token_id in eos_ids:
                    delta[..., token_id] = 0
                scores = logp_orig + cfg["alpha"] * delta
            else:
                delta = torch.zeros_like(logp_orig)
                scores = logp_orig
            beta = cfg["beta"]
            if beta > 0:
                support = (
                    logp_orig
                    >= logp_orig.max(-1, keepdim=True).values + math.log(beta)
                )
                scores = scores.masked_fill(~support, -torch.inf)
            else:
                support = torch.ones_like(scores, dtype=torch.bool)
            target = torch.softmax(scores, -1)
            log_target = torch.log_softmax(scores, -1)
        module.train()
        optimizer.zero_grad(set_to_none=True)
        with precision_context():
            student_logits = model(**student_inputs, use_cache=False).logits[
                :, -suffix.shape[1]-1:-1
            ]
            logp_student = torch.log_softmax(
                student_logits.float() / cfg["temperature_kd"], -1
            )
            per_token = torch.sum(
                target * (log_target - logp_student), dim=-1
            )
            loss = cfg["temperature_kd"] ** 2 * per_token.mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        if step < cfg["warmup_steps"]:
            lr = cfg["learning_rate"] * (step + 1) / cfg["warmup_steps"]
            for group in optimizer.param_groups:
                group["lr"] = lr
        optimizer.step()
        ema_update(teacher, module, cfg["ema_update_rate"])
        metrics = torch.tensor([
            float(loss.detach()),
            float(grad_norm),
            float(support.float().sum(-1).mean()),
            float(-(target * torch.where(torch.isfinite(log_target), log_target, 0)).sum(-1).mean()),
            float(delta.abs().max()),
            float(torch.isfinite(loss).all()),
        ], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics /= world
        event = {
            "event": "train_step",
            "step": step + 1,
            "condition": condition,
            "loss": float(metrics[0]),
            "grad_norm": float(metrics[1]),
            "support_size": float(metrics[2]),
            "target_entropy": float(metrics[3]),
            "max_abs_contrast": float(metrics[4]),
            "finite_fraction": float(metrics[5]),
        }
        history.append(event)
        rank0_print(event)
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    start = time.time()
    rank, world, device = setup_dist(cfg["seed"])
    rank0_print({
        "event": "protocol",
        "paper_id": "2607.21556",
        "condition": cfg["condition"],
        "config": cfg,
        "backend": "kubernetes",
        "gpu_model": torch.cuda.get_device_name(device),
        "world_size": world,
        "paper_numbers": PAPER_NUMBERS,
        "substitutions": [
            "48 fixed ViRL39K examples instead of the full 38,870",
            "12 updates instead of 90",
            "batch 4 with one rollout per prompt instead of batch 32 with eight rollouts",
            "64 fixed examples each from MMStar and MathVista instead of seven full benchmarks",
            "maximum 24 rollout tokens and 32 evaluation tokens",
        ],
    })
    processor = AutoProcessor.from_pretrained(
        cfg["model_id"],
        min_pixels=cfg["min_pixels"],
        max_pixels=cfg["max_pixels"],
    )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg["model_id"], torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = DDP(model, device_ids=[device.index], find_unused_parameters=True)
    # Every rank resolves exactly the same deterministic public IDs from the
    # pod-wide cache. Do not serialize image objects through an NCCL collective:
    # large broadcast_object_list payloads cause illegal accesses on this stack.
    dist.barrier()
    train_rows = load_virl_subset(cfg["train_examples"], cfg["seed"])
    mmstar = fixed_eval_rows(
        "MMStar", cfg["eval_examples_per_benchmark"], cfg["seed"]
    )
    mathvista = fixed_eval_rows(
        "MathVista", cfg["eval_examples_per_benchmark"], cfg["seed"]
    )
    dist.barrier()
    rank0_print({
        "event": "data",
        "train_ids_sha_material": [x["id"] for x in train_rows],
        "mmstar_ids": [x["id"] for x in mmstar],
        "mathvista_ids": [x["id"] for x in mathvista],
    })
    initial = [
        evaluate(model, processor, mmstar, cfg, rank, world, device, "MMStar", "before"),
        evaluate(model, processor, mathvista, cfg, rank, world, device, "MathVista", "before"),
    ]
    diag_before = token_diagnostic(
        model, None, processor, mmstar, cfg, rank, world, device, "before"
    )
    teacher = None
    history = []
    if cfg["condition"] != "base":
        teacher = copy.deepcopy(model.module).to(device)
        teacher.requires_grad_(False)
        history = train(
            model, teacher, processor, train_rows, cfg, rank, world, device
        )
    final = [
        evaluate(model, processor, mmstar, cfg, rank, world, device, "after"),
        evaluate(model, processor, mathvista, cfg, rank, world, device, "after"),
    ]
    diag_after = token_diagnostic(
        model, teacher, processor, mmstar, cfg, rank, world, device, "after"
    )
    elapsed = (time.time() - start) / 3600
    if rank == 0:
        before_mean = float(np.mean([x["accuracy"] for x in initial]))
        after_mean = float(np.mean([x["accuracy"] for x in final]))
        summary = {
            "event": "FINAL_RESULT",
            "paper_id": "2607.21556",
            "condition": cfg["condition"],
            "backend": "kubernetes",
            "gpu_model": torch.cuda.get_device_name(device),
            "gpu_count": world,
            "elapsed_hours": elapsed,
            "before": {x["benchmark"]: x["accuracy"] for x in initial},
            "after": {x["benchmark"]: x["accuracy"] for x in final},
            "mean_before": before_mean,
            "mean_after": after_mean,
            "mean_delta": after_mean - before_mean,
            "diagnostic_before": diag_before,
            "diagnostic_after": diag_after,
            "train_final": history[-1] if history else None,
            "finite_all_steps": all(x["finite_fraction"] == 1.0 for x in history),
        }
        print("FINAL_RESULT_JSON=" + json.dumps(summary, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
