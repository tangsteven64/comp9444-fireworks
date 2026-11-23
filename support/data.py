"""data -- shared split and data loading for ESC-50 (generated once, then frozen before any experiment).

Split rules (evaluation protocol):
  · Outer loop uses the official 5 folds (the fold field in meta); round k = fold k is the test fold;
  · In the other 4 folds, about 15% is held out as the validation set, grouped by src_file (original recording)
    and stratified by class -- the same original recording never appears in both training and validation.
  · Cost: with grouping by recording, each round's validation set has about 215–230 clips covering 47–49 classes
    (not all 50); the validation set is only used for early stopping and model selection, and the test fold is
    still the complete official fold, so this is acceptable.
Usage:
  make_split(meta_csv, out_csv)      # called once during data preparation (preprocessing outline in Section 2.2) to create split.csv; its SHA-256 is stored in EXPECTED_SPLIT_SHA256 below
  split = load_split(split_csv)      # raises if the hash differs from the frozen value
  train_idx, val_idx, test_idx = get_round(split, test_fold=1)
  load_logmel(COMMON / "logmel_fp16.npy", idx) / load_waveform(ROOT / "data/ESC-50-master/audio", split, idx)
  logmel_from_waveform(x)            # the same log-mel as the cache, from a waveform (e.g. after add_noise for robustness)
  add_noise(x, snr_db, noise_rng(noise_seed, fold, snr_db))   # white noise at a given SNR, reproducible per (fold, SNR)
Data generation: once, during data preparation (preprocessing outline in Section 2.2; downloads the data, creates split.csv
  and the log-mel cache); the expected split SHA-256 is stored below.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from eval_io import check_hash, sha256

# SHA-256 of split.csv, hard-coded here: load_split() rejects a split whose bytes differ (a byte-identical regenerated split passes)
EXPECTED_SPLIT_SHA256 = "dd83fbbbf08cbadd455dee01310db485f76d9a84c045172bb44d3605a3d47bbf"

N_FOLDS = 5
# log-mel parameters of the cache (128 mel x 431 frames). data preparation (Section 2.2) uses these, so the
# cache and logmel_from_waveform() can never use different settings.
LOGMEL_PARAMS = dict(sample_rate=44100, n_fft=2048, win_length=2048, hop_length=512, n_mels=128, top_db=80, frames=431)
VAL_SPLITS = 7          # each round: split the 4 training folds into 7 grouped, stratified parts and use 1 as validation (≈ 14%)
SEED = 0


def make_split(meta_csv, out_csv) -> pd.DataFrame:
    """Called once during data preparation (Section 2.2): the validation sets are computed here once and written to columns val_r{k}; they are never recomputed afterwards."""
    from sklearn.model_selection import StratifiedGroupKFold   # sklearn is only needed to create the split
    meta = pd.read_csv(meta_csv)
    out = meta[["filename", "fold", "target", "category", "src_file"]].copy()
    for k in range(1, N_FOLDS + 1):
        pool = out[out["fold"] != k]
        sgkf = StratifiedGroupKFold(n_splits=VAL_SPLITS, shuffle=True, random_state=SEED + k)
        _, val_pos = next(sgkf.split(pool, pool["target"], groups=pool["src_file"]))
        out[f"val_r{k}"] = out["filename"].isin(pool.iloc[val_pos]["filename"]).astype(int)
    out.to_csv(out_csv, index=False)
    return out


def load_split(split_csv) -> pd.DataFrame:
    """Read the frozen split; raises if the hash differs from the frozen value hard-coded above (no bypass option)."""
    check_hash(split_csv, EXPECTED_SPLIT_SHA256)
    split = pd.read_csv(split_csv)
    if len(split) != 2000 or set(split["fold"]) != set(range(1, N_FOLDS + 1)):
        raise RuntimeError(f"split.csv should have 2,000 rows with fold 1–{N_FOLDS}")
    return split


def get_round(split: pd.DataFrame, test_fold: int):
    """Return (train_idx, val_idx, test_idx) for round test_fold, as row numbers of split (aligned with the rows of the cache arrays)."""
    test = split["fold"] == test_fold
    val = (split[f"val_r{test_fold}"] == 1) & ~test
    train = ~test & ~val
    tr, va, te = (np.flatnonzero(m.values) for m in (train, val, test))
    if set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te):
        raise RuntimeError("train / validation / test sets overlap")
    if set(split.loc[tr, "src_file"]) & set(split.loc[va, "src_file"]):
        raise RuntimeError("training and validation share original recordings")
    # Note: test and training may share a src_file: 4 original recordings span folds in the official 5-fold split
    return tr, va, te


def load_waveform(audio_dir, split: pd.DataFrame, indices):
    """Read raw waveforms (for the MFCC features of Section 4.1 and the EDA). Returns [(waveform float32, sample_rate), ...]."""
    import soundfile as sf
    out = []
    for i in indices:
        x, sr = sf.read(str(Path(audio_dir) / split.loc[i, "filename"]), dtype="float32")
        if sr != 44100 or len(x) != 220500:
            raise RuntimeError(f"{split.loc[i, 'filename']}: expected 44.1 kHz × 5 s, got {sr} Hz × {len(x)} samples")
        out.append((x, sr))
    return out


_LOGMEL: dict = {}      # each process reads the whole cache once (fp16, about 220 MB), then slices it by row number


def load_logmel(cache_npy, indices):
    """Take samples from the full log-mel cache (row order = split.csv row order); returns float32.
    On the first call: on Colab, copy the cache to the local disk /content first (so it is read from there),
    always check the row count, and check size and SHA-256 against FROZEN.json when that file exists; then load the
    whole array into memory; later calls per batch are fast."""
    cache_npy = Path(cache_npy)
    if cache_npy not in _LOGMEL:
        src = cache_npy
        if Path("/content").is_dir() and not str(cache_npy.resolve()).startswith("/content/fw_cache"):
            local = Path("/content/fw_cache") / cache_npy.name
            if not local.exists() or local.stat().st_size != cache_npy.stat().st_size:
                local.parent.mkdir(exist_ok=True)
                shutil.copyfile(cache_npy, local)
            src = local
        frozen_json = cache_npy.parent / "FROZEN.json"     # looked up next to the original path (common/), so it can be checked after copying to /content
        if frozen_json.exists():
            want = json.loads(frozen_json.read_text())["files"][cache_npy.name]
            if src.stat().st_size != want["bytes"] or sha256(src) != want["sha256"]:
                raise RuntimeError(f"{cache_npy.name} differs from the frozen version (size or sha256 differs): features and labels may be misaligned")
        x = np.load(src)
        if len(x) != 2000:
            raise RuntimeError(f"{cache_npy.name} has {len(x)} rows but split.csv has 2,000: the row order does not match")
        _LOGMEL[cache_npy] = x
    return np.asarray(_LOGMEL[cache_npy][indices], dtype=np.float32)


# ───────────────────────── log-mel from a waveform · noise (robustness) ─────────────────────────
_MEL: dict = {}         # (f_max) -> (MelSpectrogram, AmplitudeToDB), built once per process


def logmel_from_waveform(x, f_max: float = 22050.0, fp16: bool = True) -> np.ndarray:
    """Log-mel of ONE clip, computed exactly like the cache: torchaudio MelSpectrogram (LOGMEL_PARAMS) +
    AmplitudeToDB(power, top_db=80), cropped to 431 frames. x: 1-D float waveform at 44.1 kHz (220,500 samples).
    Returns float32 of shape (1, 128, 431). fp16=True rounds through float16 like the cache files, so the result
    matches load_logmel() rows; fp16=False keeps full precision.
    One clip per call on purpose: AmplitudeToDB's top_db is relative to the maximum of the whole input tensor, so a
    batch of clips would share one maximum and differ from the cache."""
    import torch
    import torchaudio
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"logmel_from_waveform takes one clip as a 1-D waveform, got shape {x.shape}")
    P = LOGMEL_PARAMS
    key = float(f_max)
    if key not in _MEL:
        _MEL[key] = (torchaudio.transforms.MelSpectrogram(sample_rate=P["sample_rate"], n_fft=P["n_fft"], win_length=P["win_length"],
                                                          hop_length=P["hop_length"], n_mels=P["n_mels"], f_max=key),
                     torchaudio.transforms.AmplitudeToDB(stype="power", top_db=P["top_db"]))
    mel, todb = _MEL[key]
    with torch.no_grad():
        out = todb(mel(torch.from_numpy(x))).numpy()[:, :P["frames"]]
    if out.shape != (P["n_mels"], P["frames"]):
        raise RuntimeError(f"log-mel has shape {out.shape}, expected ({P['n_mels']}, {P['frames']}): is the clip 44.1 kHz x 5 s?")
    if fp16:
        out = out.astype(np.float16)
    return out.astype(np.float32)[None]


def noise_rng(noise_seed: int, fold: int, snr_db) -> np.random.Generator:
    """The random generator for one (fold, SNR) robustness condition: default_rng(noise_seed + fold * 1000 + snr_db).
    Create it once per condition and draw the noise for the test clips in split order, so every run gets the same noise."""
    return np.random.default_rng(int(noise_seed) + int(fold) * 1000 + int(snr_db))


def add_noise(x, snr_db, rng: np.random.Generator) -> np.ndarray:
    """White Gaussian noise at snr_db relative to the clip's own mean signal power (10 * log10(P_signal / P_noise) = snr_db).
    Deterministic given rng (see noise_rng). Returns float32 of the same shape; a completely silent clip is returned unchanged."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"add_noise takes ONE clip (1-D waveform), got shape {x.shape}: loop over clips in split order")
    p_signal = float(np.mean(x.astype(np.float64) ** 2))
    noise = rng.standard_normal(x.shape) * np.sqrt(p_signal / 10 ** (float(snr_db) / 10))
    return (x + noise).astype(np.float32)
