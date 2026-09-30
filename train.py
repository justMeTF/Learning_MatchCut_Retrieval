from __future__ import annotations

import gc
import json
import random
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from tools.model import TLs, nt_xent_loss
from tools.data import set_frames_syn, set_frames_real, split_frames, EmbeddingsStore, Split
from tools.metrics import evaluate_syn, evaluate_real

def make_batch(
    half_batch: int, split: Split, rng: random.Random,
) -> tuple[list[str], list[str]]:
    comps = split.train_comps
    if half_batch <= len(comps):
        picked_comps = rng.sample(comps, half_batch)
    else:
        raise IndexError('[Batch Size Error] : batch/2 can not be over trian_comps_id')

    na: list[str] = []
    nb: list[str] = []

    cells = split.train_conts
    for c in picked_comps:
        cA, cB = rng.sample(cells, 2)
        na.append(f"{c}_{cA}")
        nb.append(f"{c}_{cB}")

    return sorted(na), sorted(nb)

def train_and_test_(
    store_syn: EmbeddingStore, store_real: EmbeddingStore, split_syn: Split, frame_real: Frame,
    half_batch: int, seed: int, enc: str, *,
    # num_steps: int = 1000, eval_every: int = 200, lr: float = 1e-3, temperature: float = 0.05,
    num_steps: int = 5000, eval_every: int = 100, lr: float = 1e-4, temperature: float = 0.05, 
    use_amp: bool = True, device: str = "cuda",
    patience: int = 5, min_delta: float = 0.0,   # early stopping parameters
) -> dict:
    setting = f"{enc}_{half_batch}_{seed}"
    setting_seed = int(hashlib.md5(setting.encode()).hexdigest()[:8], 16)

    torch.manual_seed(setting_seed)
    model = TLs(fm_dim=store_syn.feat_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    amp = use_amp and device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    rng = random.Random(setting_seed)

    val_curve = []

    best_val = None         # early stopping
    best_step = None        # early stopping
    best_test_syn = None    # early stopping
    best_test_real = None   # early stopping
    no_improve = 0          # early stopping

    for step in range(1, num_steps + 1):
        na, nb = make_batch(half_batch, split_syn, rng)
        xa = store_syn.take(na, device)
        xb = store_syn.take(nb, device)

        model.train() #학습모드
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            ea = model(xa, isTrain=True)
            eb = model(xb, isTrain=True)
            loss = nt_xent_loss(ea, eb, temperature=temperature)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if step % eval_every == 0 or step == num_steps:
            val = evaluate_syn(model, store_syn, split_syn.valid_comps, split_syn.train_conts, device="cuda")
            val_curve.append({"step": step, "loss": float(loss.item()), "mAP": val["mAP"]})
            print(f"[{setting}] : step {step:>5} / loss {loss.item():.4f} / val_mAP {val['mAP']:.4f}")

            # 0913_잠시off
            # if not step % 1000: #
            #     test_real_per1000 = evaluate_real(model, store_real, frame_real, device="cuda") #
            #     mAP_per1000 = test_real_per1000['mAP'] #
            #     print(f"  --> real mAP : {mAP_per1000}") #
            #     del test_real_per1000, mAP_per1000 #

            # early stopping
            if best_val is None or val["mAP"] > best_val + min_delta:
                best_val = val["mAP"]
                best_step = step
                no_improve = 0
                best_test_syn = evaluate_syn(model, store_syn, split_syn.test_comps, split_syn.test_conts, device="cuda")
                best_test_real = evaluate_real(model, store_real, frame_real, device="cuda")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"    --> Early Stopped (best step: {best_step})")
                    val_curve = val_curve[:-patience]
                    break

    # test_syn = evaluate_syn(model, store_syn, split_syn.test_comps, split_syn.test_conts, device="cuda")
    # test_real = evaluate_real(model, store_real, frame_real, device="cuda")
    test_syn = best_test_syn       # early stopping
    test_real = best_test_real     # early stopping

    out = {"setting": setting, "best_step": best_step, "val_curve": val_curve, "test_syn": test_syn, "test_real": test_real}

    del model, opt
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return out

def run_experiments(
    emb_syn_dir: str = "..\\2_embeddings\\data_syn",
    emb_real_dir: str = "..\\2_embeddings\\data_real",
    encoders: list[str] = ("sam2", "sam", "efficientnet", "clip", "resnet", ),
    half_batchs: list[int] = (256, 128, 64, 32, 16, 8, 4, 2),
    seeds: list[int] = (0, 1, 2, 3, 4),
    num_steps: int = 5000,
    eval_every: int = 100,
    lr: float = 1e-4,
    temperature: float = 0.05,
    seed_split: int = 42,
    ratio_comp: tuple[float, float, float] = (0.6, 0.2, 0.2),
    ratio_cont: float = 0.6,
    use_amp: bool = True,
) -> list:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    half_batchs, seeds, encoders = list(half_batchs), list(seeds), list(encoders)

    frame_syn = set_frames_syn(emb_syn_dir)
    splits, names_syn = split_frames(frame_syn, seed_split, ratio_comp, ratio_cont)
    frame_real, names_real = set_frames_real(emb_real_dir)

    results = []

    for enc in encoders:
        store_syn = EmbeddingsStore(emb_syn_dir, enc, names_syn)           # load synthetic data to RAM
        store_real = EmbeddingsStore(emb_real_dir, enc, names_real)        # load real data to RAM

        for seed in seeds:
            for half_batch in half_batchs:
                res = train_and_test_(
                    store_syn, store_real, splits, frame_real,
                    half_batch, seed, enc,
                    num_steps=num_steps, eval_every=eval_every, lr=lr, temperature=temperature,
                    use_amp=use_amp, device=device
                )
                m=res['setting'] #
                s=res["test_syn"]["mAP"] #
                r=res["test_real"]["mAP"] #
                print(f">>> [{m}] syn : {s}, real : {r}") #
                del m, s, r #
                results.append(res)

        del store_syn, store_real
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    
    return results

if __name__ == "__main__":
    run_experiments()