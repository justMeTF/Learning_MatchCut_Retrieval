from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------- #
# 검색용 임베딩 추출
# ---------------------------------------------------------------------- #
@torch.no_grad()
def embed(model, store, names, device, chunk=512) -> torch.Tensor:
    """검색용 임베딩 (평가). mEnc=h(1024) 정규화 / MLP=정규화 1024. 반환 (N,d)."""
    model.eval() # 평가 모드 전환
    outs = []
    for i in range(0, len(names), chunk):
        x = store.take(names[i:i + chunk], device)
        h = model(x, isTrain=False)
        outs.append(F.normalize(h, p=2, dim=1))
    return torch.cat(outs, dim=0)

def evaluate_syn(model, store, comps, conts, device="cuda") -> dict:
    pairs = sorted((f"{comp}_{cont}", str(comp)) for cont in conts for comp in comps)
    names = np.array([p[0] for p in pairs])
    names_comp = np.array([p[1] for p in pairs])
    N = len(names)

    emb = embed(model, store, names, device)    # (N,d) 정규화
    sim = (emb @ emb.T).float().cpu().numpy()

    APs = []
    retrievals = {}

    for i in range(N):
        cand = np.array([c for c in range(N) if c != i])
        if cand.size == 0:
            continue
        order = np.argsort(-sim[i, cand], kind="stable")
        ranked = cand[order]
        rel = (names_comp[ranked] == names_comp[i])                    # 랭킹된 후보의 relevance

        if int(rel.sum()) == 0:
            continue
        cum = np.cumsum(rel)
        ranks = np.arange(1, len(rel) + 1)
        APs.append(float((cum[rel] / ranks[rel]).mean()))            # Average Precision
        
        retrievals[str(names[i])] = (names[ranked][:5], sim[i, ranked][:5])
        # retrievals.append((names[ranked][:5], sim[i, ranked][:5]))   # top-5 retreived images and their similarity scores
    out = {
        "mAP": float(np.mean(APs)) if APs else 0.0,
        "retrievals": retrievals, #
    }
    return out

def evaluate_real(model, store, frames, device="cuda") -> dict:
    pairs = sorted((frame.name, frame.comp_id, frame.is_query) for frame in frames)
    names = np.array([p[0] for p in pairs])
    names_comp = np.array([p[1] for p in pairs])
    names_is_query = np.where(np.array([p[2] for p in pairs]))[0]
    N = len(names)
    emb = embed(model, store, names, device)    # (N,d) 정규화
    sim = (emb @ emb.T).float().cpu().numpy()

    APs = []
    retrievals = {}

    for i in names_is_query:
        cand = np.array([c for c in range(N) if c != i])
        if cand.size == 0:
            continue
        order = np.argsort(-sim[i, cand], kind="stable")
        ranked = cand[order]
        rel = (names_comp[ranked] == names_comp[i])                    # 랭킹된 후보의 relevance

        if int(rel.sum()) == 0:
            continue
        cum = np.cumsum(rel)
        ranks = np.arange(1, len(rel) + 1)
        APs.append(float((cum[rel] / ranks[rel]).mean()))    # Average Precision

        retrievals[str(names[i])] = (names[ranked][:5], sim[i, ranked][:5])        
        # retrievals.append(names[ranked][:5])                 # top-5 retreived images

    out = {
        "mAP": float(np.mean(APs)) if APs else 0.0,
        "retrievals": retrievals, #
    }
    return out