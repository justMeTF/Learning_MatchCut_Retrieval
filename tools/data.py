from __future__ import annotations
import random
import re
from dataclasses import dataclass
from pathlib import Path
import torch
import os
import numpy as np

SYN_RE = re.compile(r"^(ps\d+)_(CU|MS|FS|LS)_(\d+)_(bg\d+)_(ch\d+)$")     # "SYN"thethic data for train, validation and test
REAL_RE = re.compile(r"^([a-zA-Z])(.*)_([^_]+)$")                         # "REAL" data for test

@dataclass(frozen=True)
class Frame_syn:
    name: str          # name : {ps}_{ss}_{lc}_{bg}_{ch}
    ps: str            # pose             : ps00 ~ ps24
    ss: str            # shot size        : CU / MS / FS / LS
    cs: str            # camera settings  : 0 ~ 4 (location, rotation)
    bg: str            # background       : bg00 ~ bg09
    ch: str            # character        : ch00 ~ ch09

    @property
    def comp_id(self) -> str:
        return f"{self.ps}_{self.ss}_{self.cs}"

    @property
    def cont_id(self) -> tuple[str, str]:
        return (self.bg, self.ch)

# parse image names and store them based on 'names.txt'
def set_frames_syn(emb_dir: str | Path) -> list[Frame_syn]:
    names_txt = Path(emb_dir) / "names.txt"
    if not names_txt.exists():
        raise FileNotFoundError(f"No {names_txt}")
    frames = []
    with open(names_txt, 'r', encoding='utf-8') as file:
        for line in file:
            img_name = line.strip()
            parsed = SYN_RE.match(img_name)
            if not parsed:
                continue
            ps, ss, cs, bg, ch = parsed.groups()
            frames.append(Frame_syn(img_name, ps, ss, cs, bg, ch))
    if not frames:
        raise FileNotFoundError(f"No image names in {names_txt}")
    return frames

@dataclass(frozen=True)
class Frame_real:
    name: str          # name : {comp_id}_{cont_id}
    is_query: bool     # m : True, d : False
    comp_id: str       # m[source]_c[cut], d[source]_c00
    cont_id: int       # 0, 1, ...

def set_frames_real(emb_dir: str | Path) -> tuple[list[Frame_real], list[str]]:
    names_txt = Path(emb_dir) / "names.txt"
    if not names_txt.exists():
        raise FileNotFoundError(f"No {names_txt}")
    frames = []
    names = []
    with open(names_txt, 'r', encoding='utf-8') as file:
        for line in file:
            img_name = line.strip()
            parsed = REAL_RE.match(img_name)
            if not parsed:
                continue
            names.append(img_name)
            is_query, comp_id, cont_id = parsed.groups()
            comp_id = is_query + comp_id
            is_query = (is_query == 'm')
            cont_id = int(cont_id)
            frames.append(Frame_real(img_name, is_query, comp_id, cont_id))
    if not frames:
        raise FileNotFoundError(f"No image names in {names_txt}")
    return frames, names

@dataclass
class Split:
    train_comps: list[str]
    valid_comps: list[str]
    test_comps: list[str]
    train_bgs: list[str]
    train_chs: list[str]
    test_bgs: list[str]
    test_chs: list[str]

    @property
    def train_conts(self) -> list[str]:
        return [bg+'_'+ch for bg in self.train_bgs for ch in self.train_chs]

    @property
    def test_conts(self) -> list[str]:
        return [bg+'_'+ch for bg in self.test_bgs for ch in self.test_chs]

def split_frames(
    frames: list[Frame_syn], seed: int = 42,
    ratio_comp: tuple[float, float, float] = (0.6, 0.2, 0.2),
    ratio_cont: float = 0.6
) -> tuple[Split, list[str]]:
    rng = random.Random(seed)
    assert abs(sum(ratio_comp) - 1.0) < 1e-6
    assert 0 < ratio_cont < 1
    
    comp_ids = sorted({f.comp_id for f in frames})
    rng.shuffle(comp_ids)
    comp_tr = round(len(comp_ids) * ratio_comp[0])
    comp_va = round(len(comp_ids) * ratio_comp[1])
    train_comp = comp_ids[:comp_tr]
    valid_comp = comp_ids[comp_tr:comp_tr + comp_va]
    test_comp = comp_ids[comp_tr + comp_va:]

    bgs = sorted({f.bg for f in frames})
    chs = sorted({f.ch for f in frames})
    rng.shuffle(bgs)
    rng.shuffle(chs)
    bg_tr = round(len(bgs) * ratio_cont)
    ch_tr = round(len(chs) * ratio_cont)
    train_bg, test_bg = bgs[:bg_tr], bgs[bg_tr:]
    train_ch, test_ch = chs[:ch_tr], chs[ch_tr:]

    log_dir = Path("data_syn_split_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    save_data = {"train_comp": train_comp, "valid_comp": valid_comp, "test_comp": test_comp,
        "train_bg": train_bg, "train_ch": train_ch, "test_bg": test_bg, "test_ch": test_ch}
    for name, data in save_data.items():
        (log_dir / f"{name}.txt").write_text("\n".join(map(str, data)), encoding="utf-8")

    split_dataset = Split(train_comp, valid_comp, test_comp, train_bg, train_ch, test_bg, test_ch)
    train_frames = [f"{comp}_{cont}" for cont in split_dataset.train_conts for comp in split_dataset.train_comps]
    valid_frames = [f"{comp}_{cont}" for cont in split_dataset.train_conts for comp in split_dataset.valid_comps]
    test_frames = [f"{comp}_{cont}" for cont in split_dataset.test_conts for comp in split_dataset.test_comps]
    used_frames = sorted(train_frames) + sorted(valid_frames) + sorted(test_frames)
    return split_dataset, used_frames

# store Embedding of data(by certain encoder) in RAM
class EmbeddingsStore:
    def __init__(self, emb_dir: str | Path, enc: str, names: list[str]):
        npy_path = Path(emb_dir) / enc / "spatial.npy"
        txt_path = Path(emb_dir) /"names.txt"

        if not npy_path.exists():
            raise FileNotFoundError(f"Do Not Find Embeddings : {npy_path}")

        with open(txt_path, "r", encoding="utf-8") as f:
            pos = {line.strip() : i for i, line in enumerate(f)} # {file_names : index, ... }
        
        # mmap 으로 열고(디스크 상주), 필요한 행만 fancy-index 로 읽어 RAM 에 복사
        mm = np.load(npy_path, mmap_mode="r") # actual feature maps in spatial.npy
        rows = np.array([pos[nm] for nm in names], dtype=np.int64) # [index, index, ...]
        sub = np.ascontiguousarray(mm[rows])              # (len(names), ...) 만 실제 로드
        del mm

        self.arr = torch.from_numpy(sub)                  # fp16 유지 (용량↓)
        self.shape = tuple(self.arr.shape[1:])            # patch:(C,H,W)
        self.index: dict[str, int] = {nm: i for i, nm in enumerate(names)}

    @property
    def feat_dim(self) -> int:
        return self.shape[0] # channel

    def take(self, names: list[str], device) -> torch.Tensor:
        idx = torch.tensor([self.index[n] for n in names], dtype=torch.long)
        x = self.arr.index_select(0, idx)
        return x.to(device=device, dtype=torch.float32)