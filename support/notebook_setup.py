"""notebook_setup -- loading of the saved results for Section 0.3 of the notebook.

load_saved_results(support)   the split and every saved result in support/frozen/ next to the notebook.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

import data
import eval_io


def _b64(path: Path) -> bytes:
    return base64.b64decode(path.read_text())


def load_saved_results(support: Path) -> tuple:
    """Load the saved results in support/frozen/: the 41 runs (into eval_io.FROZEN_RUNS), the EDA data with the split
    (checked against its SHA-256 fingerprint) and the example clips (Section 3), the nine log-mels of the
    confused-pairs figure (Section 5.1) and the evidence tables (Sections 2.3, 7.1).
    Returns (SPLIT, EDA_DATA, SAMPLE_CLIPS, EVIDENCE, PAIR_MELS)."""
    print("shared code in support/: version", eval_io.COMMON_VERSION)
    frozen = support / "frozen"
    eval_io.FROZEN_RUNS = eval_io.unpack_runs((frozen / "runs.b64").read_text())
    eda_data = json.loads(zlib.decompress(_b64(frozen / "eda.b64")))
    if hashlib.sha256(eda_data["split_csv"].encode()).hexdigest() != data.EXPECTED_SPLIT_SHA256:   # the saved split
        raise ValueError("the split saved in support/frozen/eda.b64 does not match its SHA-256 fingerprint")
    split = pd.read_csv(io.StringIO(eda_data["split_csv"]))
    sample_clips = dict(np.load(io.BytesIO(_b64(frozen / "sample_clips.b64"))))
    evidence = json.loads(zlib.decompress(_b64(frozen / "evidence.b64")))
    pair_mels = dict(np.load(io.BytesIO(_b64(frozen / "pair_mels.b64"))))
    return split, eda_data, sample_clips, evidence, pair_mels
