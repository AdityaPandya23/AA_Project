"""EEGNet comparison under the same splits (CONTRACTS.md §h).

Requires ``requirements-gpu.txt`` (torch, braindecode). Import torch and
braindecode **inside** functions so that the rest of the package works without them.

Training recipe (recorded here and in ``cfg["eegnet"]``):

* braindecode ``EEGNet`` (``EEGNetv4`` in braindecode < 1.0), ``F1``, ``D``,
  ``drop_prob`` from config, raw logits + cross-entropy, Adam(``lr``,
  ``weight_decay``), mini-batches of ``batch_size``, up to ``max_epochs``.
* Early stopping: a stratified ``val_fraction`` of the **training fold** is held
  out. The epoch with the lowest validation loss is kept; training stops after
  ``early_stopping_patience`` epochs without improvement. The test fold is
  never used for model selection.
* Inputs are z-scored per channel with mean/SD from the fit part of the
  training fold only.
* Seeds: ``cfg["eegnet"].get("seed", 0)`` (the contract signature has no seed
  argument). The fold ``k`` uses torch/numpy seed ``seed * 1000 + k``, cuDNN
  deterministic mode.
* Device: ``cfg["eegnet"].get("device")`` or CUDA when available, else CPU.
"""
from __future__ import annotations

import copy
import os
import time
from typing import Any, Sequence

import numpy as np

Split = tuple[np.ndarray, np.ndarray]


def _model_class():
    try:
        from braindecode.models import EEGNet  # braindecode >= 1.0
    except ImportError:  # pragma: no cover
        from braindecode.models import EEGNetv4 as EEGNet
    return EEGNet


def _seed_everything(seed: int) -> None:
    import random

    import torch
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(n_chans: int, n_times: int, ecfg: dict[str, Any]):
    """The EEGNet used for every fold."""
    return _model_class()(n_chans=n_chans, n_outputs=3, n_times=n_times, F1=int(ecfg["F1"]),
                          D=int(ecfg["D"]), drop_prob=float(ecfg["drop_prob"]),
                          # max-norm on the classifier as in the original EEGNet; pinned so
                          # braindecode's announced default change cannot alter the model.
                          final_layer_with_constraint=True)


def _fit_one(Xtr, ytr, Xva, yva, ecfg, device, seed):
    """Train with early stopping on (Xva, yva). Returns ``(model, best_epoch, n_epochs)``."""
    import torch

    _seed_everything(seed)
    model = build_model(Xtr.shape[1], Xtr.shape[2], ecfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(ecfg["lr"]),
                           weight_decay=float(ecfg["weight_decay"]))
    loss_fn = torch.nn.CrossEntropyLoss()
    Xtr_t = torch.as_tensor(Xtr, device=device)
    ytr_t = torch.as_tensor(ytr, device=device)
    Xva_t = torch.as_tensor(Xva, device=device)
    yva_t = torch.as_tensor(yva, device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    bs = int(ecfg["batch_size"])
    patience = int(ecfg["early_stopping_patience"])

    best_loss, best_state, best_epoch, epoch = np.inf, None, -1, -1
    for epoch in range(int(ecfg["max_epochs"])):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=gen).to(device)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:   # BatchNorm needs > 1 sample
                continue
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(loss_fn(model(Xva_t), yva_t))
        if vloss < best_loss - 1e-6:
            best_loss, best_epoch = vloss, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break
    if best_state is not None:  # None only if every val loss was NaN
        model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, epoch + 1


def _predict(model, X, device, batch: int = 256) -> np.ndarray:
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(torch.as_tensor(X[i:i + batch], device=device)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def _latency_ms(model, X, device, max_trials: int = 50) -> float:
    """Mean per-trial inference latency (batch size 1, includes host->device copy)."""
    import torch
    n = min(len(X), max_trials)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    with torch.no_grad():
        for i in range(min(5, n)):  # warm-up
            model(torch.as_tensor(X[i:i + 1], device=device))
        sync()
        t0 = time.perf_counter()
        for i in range(n):
            model(torch.as_tensor(X[i:i + 1], device=device)).argmax(1).cpu()
        sync()
    return (time.perf_counter() - t0) * 1000.0 / n


def carve_validation(train_idx: np.ndarray, y: np.ndarray, val_fraction: float,
                     seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratified ``(fit_idx, val_idx)`` split of ``train_idx`` (absolute, sorted int64)."""
    from sklearn.model_selection import train_test_split
    train_idx = np.asarray(train_idx, dtype=np.int64)
    fit_idx, val_idx = train_test_split(train_idx, test_size=val_fraction,
                                        stratify=y[train_idx], random_state=seed)
    return np.sort(fit_idx), np.sort(val_idx)


def evaluate_eegnet(X: np.ndarray, y: np.ndarray, splits: Sequence[Split],
                    ch_idx: np.ndarray | None = None,
                    cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Train and test braindecode EEGNetv4 on each split.

    Parameters
    ----------
    X
        EEGNet cache array ``(n_trials, 32, 1000)``, float32, µV.
    y
        int64 ``(n_trials,)``.
    splits
        Exactly the same split objects as used for FBCSP on the same file stem.
    ch_idx
        Optional channel indices into axis 1. ``None`` means all 32. For
        ``eegnet_gach``, use
        ``cache.candidate_channel_index(ch_names, cfg, ch_mask)``.
    cfg
        ``load_config()`` dict. Hyperparameters come from ``cfg["eegnet"]``.
        ``None`` → ``load_config()``.

    Early-stopping validation is carved from the **training fold only**
    (``cfg["eegnet"]["val_fraction"]``, stratified). The test fold is never
    used for model selection.

    Returns
    -------
    dict
        ``{"per_fold": list[float], "ba_mean": float, "ba_std": float,
        "train_time_s": float}``. It may add ``"infer_time_ms"`` (mean
        per-trial latency) and ``"n_params"``.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from sklearn.metrics import balanced_accuracy_score

    if cfg is None:
        from .config import load_config
        cfg = load_config()
    ecfg = cfg["eegnet"]
    seed = int(ecfg.get("seed", 0))
    device = torch.device(ecfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if ch_idx is not None:
        X = X[:, np.asarray(ch_idx, dtype=np.int64), :]

    per_fold, train_time, latencies, best_epochs, n_epochs = [], 0.0, [], [], []
    n_params = None
    for k, (tr, te) in enumerate(splits):
        tr = np.asarray(tr, dtype=np.int64)
        te = np.asarray(te, dtype=np.int64)
        if np.intersect1d(tr, te).size:
            raise ValueError(f"fold {k}: train and test overlap")
        t0 = time.perf_counter()
        fit_idx, val_idx = carve_validation(tr, y, float(ecfg["val_fraction"]), seed * 1000 + k)
        assert not np.isin(val_idx, te).any() and not np.isin(fit_idx, te).any()
        mu = X[fit_idx].mean(axis=(0, 2), keepdims=True)
        sd = X[fit_idx].std(axis=(0, 2), keepdims=True) + 1e-6

        def norm(a):
            return ((a - mu) / sd).astype(np.float32)

        model, best_ep, n_ep = _fit_one(norm(X[fit_idx]), y[fit_idx], norm(X[val_idx]),
                                        y[val_idx], ecfg, device, seed * 1000 + k)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_time += time.perf_counter() - t0

        Xte = norm(X[te])
        per_fold.append(float(balanced_accuracy_score(y[te], _predict(model, Xte, device))))
        latencies.append(_latency_ms(model, Xte, device))
        best_epochs.append(int(best_ep))
        n_epochs.append(int(n_ep))
        n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {"per_fold": per_fold, "ba_mean": float(np.mean(per_fold)),
            "ba_std": float(np.std(per_fold)), "train_time_s": float(train_time),
            "infer_time_ms": float(np.mean(latencies)), "n_params": n_params,
            "best_epoch": best_epochs, "n_epochs": n_epochs, "device": str(device),
            "seed": seed}
