"""eval_io -- shared result I/O and checks used by every method.

Two phases (the evaluation protocol's "use the test fold only once", enforced in code):
  dev   train + validate, never touch the test fold. Each fold writes:
          fold<k>/metrics.csv     best_epoch, best_val_acc, best_val_correct, val_n, train_seconds, n_params
          fold<k>/val_preds.csv   per-sample predictions on the validation set after reloading the best checkpoint (filename, y_true, y_pred)
          fold<k>/curves.csv      per epoch: fold, epoch, train_loss, val_acc (empty table for methods without epochs)
          fold<k>/model.joblib (SVM / random forest) or ckpt.pt (neural models)   best checkpoint
          fold<k>/_SUCCESS
  final only for runs chosen for the results table: first select_for_final() writes SELECTED (time + reason), then each fold reloads its checkpoint and runs the test fold:
          fold<k>/test_preds.csv  filename, y_true, y_pred, prob_top1 (test accuracy is computed from this file, never self-reported by a method)
          fold<k>/_TESTED         time of testing
  robustness (optional, declared in advance): select_for_final(..., robustness={"noise": "white", "snr_db": [20, 10, 0], "noise_seed": 0})
        writes the declaration into SELECTED together with selected_at (it cannot be added later). After a fold's normal
        test (_TESTED), save_robust() stores predictions on the same test fold with noise added, once per (fold, condition):
          fold<k>/robust_<condition>.csv   filename, y_true, y_pred, prob_top1 (condition names: robust_conditions(), e.g. white_snr20)
          ROBUST_LOG.csv                   time, fold, condition, sha256 (append-only; one row per (fold, condition))
        robust_table() aggregates them like final_table. Each clean test fold is evaluated once per selected configuration
        and seed, after selection on validation data; robustness runs never feed back into the choice of configuration.
Run folder: results/<line>/<YYYY-MM-DD>_<run_name>_s<seed>/ (config.json includes split_sha256 and common_version)
Shared evaluation-code version: the COMMON_VERSION in use when a run is created is recorded in its config.json as
metadata and is not part of config_hash.

Results table: FINAL_RUNS = {"method name": [run paths, one per seed, ...]}; average over seeds within each fold first, then report mean ± SD over the five folds.
Metrics (all computed from test_preds):
  M1 accuracy (primary metric, comparable with the literature) · M2 macro-F1 (mean of the 50 per-class F1 scores) · M3 per-class recall + confusion matrix (which classes fail, and what they are confused with)
The runs shown in the notebook are saved in support/frozen/ and loaded into FROZEN_RUNS in Section 0.3.

Which functions are used where:
  writing runs (used when the runs were trained; the notebook does not call them): new_run, config_problems,
        fold_dir, save_fold, select_for_final, is_selected, save_test, save_robust, robust_declaration,
        read_robust_log, check_run. save_test refuses to save a test result before select_for_final, or a second
        one for the same fold.
  reading the saved runs and computing results (used in this notebook): unpack_runs, load_run_or_none, run_id,
        run_dir, config_hash, final_table, test_acc, test_macro_f1, macro_f1, per_class_recall, confusion,
        load_final_runs, robust_table, robust_conditions, dev_summary.
  reading and checking run folders on disk (not used with the saved results): load_run, done_folds, tested_folds.
  used by the training code of Section 4, by data.py and by the run-writing functions: seed_everything,
        seed_worker, sha256, check_hash.
"""
from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import json
import math
import random
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

DEV_COLS = ["fold", "best_epoch", "best_val_acc", "best_val_correct", "val_n", "train_seconds", "n_params"]
VAL_PRED_COLS = ["filename", "y_true", "y_pred"]
TEST_PRED_COLS = ["filename", "y_true", "y_pred", "prob_top1"]
CURVE_COLS = ["fold", "epoch", "train_loss", "val_acc"]
REQUIRED_CONFIG = ["line", "run_name", "date", "seed", "group", "question", "compare_to", "changed_variable"]
COMMON_VERSION = "1.0.5"   # shared evaluation-code version
IDENTITY_KEYS = {"line", "run_name", "date", "seed", "group", "question", "compare_to", "changed_variable", "split_sha256", "notes",
                 "common_version"}
GROUPS = ("non_deep", "from_scratch", "external_pretrained")   # row groups of the results table, in this order
PLACEHOLDER_RE = re.compile(r"^(TODO|MISSING|CHECK)\b")   # leading placeholder tokens only
REENTRY_IGNORE = {"notes", "question", "split_sha256", "common_version"}   # fields that may change when resuming the same run
PROTOCOL_KEYS = ("input",)                                # protocol fields that must still match in a whole-method comparison across lines
N_FOLDS = 5
N_CLASSES = 50
LINE_RE = re.compile(r"line[1-5]")                       # results/<line>/: the five methods
FOLD_RE = re.compile(r"^fold([1-5])$")                    # canonical fold folder names only
ROBUST_NOISES = ("white",)                                # noise types save_robust accepts (extend together with data.add_noise)
ROBUST_KEYS = ("noise", "snr_db", "noise_seed")
ROBUST_LOG_COLS = ["time", "fold", "condition", "sha256"]
ROBUST_FILE_RE = re.compile(r"^robust_(.+)\.csv$")

SPLIT_SHA256 = None        # set once data.load_split() has verified the split
FROZEN_RUNS: dict = {}     # the saved runs, filled from support/frozen/ in Section 0.3


# ───────────────────────── reproducibility ─────────────────────────
def seed_everything(seed: int):
    """Call before training in every method; returns a torch.Generator for the DataLoader (None if torch is not installed)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True    # deterministic cuDNN settings requested (not bit-exact across runs)
        torch.backends.cudnn.benchmark = False
        g = torch.Generator()
        g.manual_seed(seed)
        return g
    except ImportError:
        return None


def seed_worker(worker_id):
    """DataLoader(..., worker_init_fn=eval_io.seed_worker, generator=g): numpy / random in each worker also follow the seed."""
    import torch
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


# ───────────────────────── split verification ─────────────────────────
def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_hash(path, expected: str) -> None:
    """The split file's hash must equal the frozen value, otherwise fail immediately (prevents runs from reading different splits)."""
    global SPLIT_SHA256
    got = sha256(path)
    if got != expected:
        raise RuntimeError(f"hash of {path} is {got[:12]}…, frozen value is {str(expected)[:12]}…: this is not the team's shared split")
    SPLIT_SHA256 = got


# ───────────────────────── creating runs ─────────────────────────
def run_id(config: dict) -> str:
    return f"{config['date']}_{config['run_name']}_s{config['seed']}"


def run_dir(results_root, config: dict) -> Path:
    return Path(results_root) / config["line"] / run_id(config)


def config_problems(config: dict, results_root=None, compare_root=None) -> list:
    """Hard checks on CONFIG: required keys, group value, placeholders, compare_to and "change only one variable".
    compare_to is looked up under compare_root (default: results_root)."""
    errs = [f"missing required key {k}" for k in REQUIRED_CONFIG if k not in config]
    if errs:
        return errs
    if config["group"] not in GROUPS:
        errs.append(f"group must be one of {GROUPS}")
    for k, v in config.items():
        if isinstance(v, str) and PLACEHOLDER_RE.match(v):
            errs.append(f"{k} is still a placeholder: {v!r}")
        elif isinstance(v, float) and not math.isfinite(v):
            errs.append(f"{k} is not a finite number: {v!r}")
    ref_root = compare_root if compare_root is not None else results_root
    if config["compare_to"] != "baseline" and ref_root is not None and not errs:
        ref = Path(ref_root) / config["compare_to"] / "config.json"
        # without results/, fall back to the reference config in support/frozen/ (Section 0.3)
        ref_cfg = (json.loads(ref.read_text()) if ref.exists()
                   else (FROZEN_RUNS.get(config["compare_to"]) or {}).get("config"))
        if ref_cfg is None:
            errs.append(f"compare_to points to a run that does not exist: {config['compare_to']} (the first run uses 'baseline')")
        elif config["changed_variable"] == "method":
            # whole-method comparison (e.g. the CNN vs the MFCC baseline): model and training settings differ anyway, only check the shared protocol fields
            other, mine = ref_cfg, _norm(config)
            bad = [k for k in PROTOCOL_KEYS if mine.get(k) != other.get(k)]
            if bad:
                errs.append(f"protocol fields differ from {config['compare_to']} {bad}: a whole-method comparison must still use the same input")
        else:
            other, config = ref_cfg, _norm(config)   # tuple→list etc.: compare as written to config.json
            keys = (set(config) | set(other)) - IDENTITY_KEYS
            diff = sorted(k for k in keys if config.get(k) != other.get(k))
            declared = sorted(s.strip() for s in str(config["changed_variable"]).split("+"))
            if diff != declared:
                errs.append(f"compared with {config['compare_to']} you actually changed {diff}, but changed_variable says {declared}"
                            f" (if several fields form one treatment, join them with +, e.g. 'augment+mixup_alpha';"
                            f" for a completely different method write 'method')")
    return errs


def _norm(cfg: dict) -> dict:
    """Compare configs as they look once written to config.json (tuple→list etc.) to avoid false alarms."""
    return json.loads(json.dumps(cfg))


def new_run(results_root, config: dict, compare_root=None) -> Path:
    """Get the folder for this run: the first call creates it and writes config.json; later calls (resuming unfinished
    folds) check that the current CONFIG still matches config.json; different settings require a distinct run
    name.
    compare_root: where compare_to is looked up (default: results_root)."""
    if not SPLIT_SHA256:
        raise RuntimeError("the frozen split has not been verified via data.load_split(); refusing to create a run")
    if not LINE_RE.fullmatch(str(config.get("line", ""))):
        raise ValueError(f"line must be line1 ... line5: {config.get('line')!r}")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", str(config.get("run_name", ""))):
        raise ValueError(f"run_name may only contain letters, digits and . _ -: {config.get('run_name')!r}")
    d = run_dir(results_root, config)
    cj = d / "config.json"
    if cj.exists():
        stored, mine = json.loads(cj.read_text()), _norm(config)
        if stored.get("split_sha256") != SPLIT_SHA256:
            raise ValueError(f"{d.name} was created with a different split (split_sha256 differs); it cannot be resumed")
        diff = sorted(k for k in (set(stored) | set(mine)) - REENTRY_IGNORE if stored.get(k) != mine.get(k))
        if diff:
            raise ValueError(f"{d.name} already exists, but the current CONFIG differs from its config.json {diff}:"
                             f" different settings require a distinct run_name")
        return d
    if d.exists() and any(d.iterdir()):
        raise FileExistsError(f"{d} contains files but no config.json: it was not created by this tool; find out why before doing anything")
    errs = config_problems(config, results_root, compare_root)
    if errs:
        raise ValueError("CONFIG has problems:\n  " + "\n  ".join(errs))
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "config.json.tmp"
    tmp.write_text(json.dumps({**config, "split_sha256": SPLIT_SHA256, "common_version": COMMON_VERSION}, indent=2, ensure_ascii=False))
    tmp.replace(cj)                                # atomic replace: a dropped connection never leaves half a config.json
    return d


def fold_dir(rdir, fold: int) -> Path:
    """Get the fold's folder before training (the checkpoint is saved here)."""
    if fold not in range(1, N_FOLDS + 1):
        raise ValueError(f"fold must be 1–{N_FOLDS}: {fold!r}")
    d = Path(rdir) / f"fold{fold}"
    d.mkdir(exist_ok=True)
    return d


def _folds_with(rdir, marker: str) -> set:
    return {int(m.group(1)) for p in Path(rdir).glob(f"fold*/{marker}") if (m := FOLD_RE.match(p.parent.name))}


def done_folds(rdir) -> set:
    return _folds_with(rdir, "_SUCCESS")


def tested_folds(rdir) -> set:
    return _folds_with(rdir, "_TESTED")


# ───────────────────────── training and validation ─────────────────────────
def save_fold(rdir, fold: int, metrics: dict, val_preds: pd.DataFrame, curves: pd.DataFrame, split=None) -> None:
    """dev: save a fold as soon as it has been trained.
    metrics must contain best_epoch, best_val_acc, best_val_correct, val_n, train_seconds, n_params;
    val_preds must be the validation predictions made *after reloading the best checkpoint* -- the number of correct
    predictions recomputed from it must equal best_val_correct, so the reloaded checkpoint must be the best
    epoch.
    If split is passed, check right away that this is this round's frozen validation set
    with correct labels, instead of waiting for check_run after training."""
    fd = fold_dir(rdir, fold)
    if (fd / "_SUCCESS").exists():
        raise ValueError(f"fold {fold} already has complete dev results")
    _check_cols(val_preds, VAL_PRED_COLS, "val_preds")
    _check_labels(val_preds, f"fold {fold} val_preds")
    missing = [k for k in DEV_COLS[1:] if k not in metrics]
    if missing:
        raise ValueError(f"fold {fold}: metrics is missing {missing}")
    if not all(math.isfinite(float(metrics[k])) for k in DEV_COLS[1:]):
        raise ValueError(f"fold {fold}: metrics contains non-finite values")
    if split is not None:
        err = _val_set_problem(split, fold, val_preds)
        if err:
            raise ValueError(f"fold {fold}: {err}")
    correct = int((val_preds["y_true"] == val_preds["y_pred"]).sum())
    if correct != int(metrics["best_val_correct"]) or len(val_preds) != int(metrics["val_n"]):
        raise AssertionError(f"fold {fold}: after reloading the checkpoint the validation set has {correct}/{len(val_preds)} correct,"
                             f" but training recorded {metrics['best_val_correct']}/{metrics['val_n']} -- the checkpoint saved or reloaded is not the best epoch")
    if abs(float(metrics["best_val_acc"]) - correct / max(len(val_preds), 1)) > 1e-9:
        raise ValueError(f"fold {fold}: best_val_acc ≠ best_val_correct / val_n (compute it as correct / n, do not round)")
    if len(curves):
        row = curves.loc[curves["epoch"] == metrics["best_epoch"]]
        if row.empty or abs(float(row["val_acc"].iloc[0]) - float(metrics["best_val_acc"])) > 5e-4:
            raise ValueError(f"fold {fold}: val_acc of epoch {metrics['best_epoch']} in curves does not match best_val_acc"
                             f" (the learning-curve plot and the results table would disagree; do not round val_acc in curves)")
    c = curves.copy()
    c.insert(0, "fold", fold)
    _check_cols(c, CURVE_COLS, "curves")
    pd.DataFrame([{"fold": fold, **{k: metrics[k] for k in DEV_COLS[1:]}}])[DEV_COLS].to_csv(fd / "metrics.csv", index=False)
    val_preds[VAL_PRED_COLS].to_csv(fd / "val_preds.csv", index=False)
    c[CURVE_COLS].to_csv(fd / "curves.csv", index=False)
    (fd / "env.json").write_text(json.dumps(_env(), ensure_ascii=False))   # hardware and versions: not part of config_hash
    (fd / "_SUCCESS").write_text("ok\n")


# ───────────────────────── final phase ─────────────────────────
def select_for_final(rdirs, reason: str, robustness: dict | None = None) -> None:
    """Call once after choosing a configuration for the results table: pass **the runs for all
    seeds fixed in advance** for that configuration together (each with all five dev folds done, identical config).
    Only then can the test folds be run. The results table checks that runs selected together are all listed: no
    picking seeds.
    robustness (optional): declare the robustness evaluation now, before any test fold is seen, e.g.
        {"noise": "white", "snr_db": [20, 10, 0], "noise_seed": 0}
    It is written into every SELECTED file together with selected_at; SELECTED is written once, so a declaration
    cannot be added or changed later. Only declared conditions can be saved with save_robust()."""
    spec = None if robustness is None else _robust_spec(robustness)
    rdirs = [Path(r) for r in ([rdirs] if isinstance(rdirs, (str, Path)) else rdirs)]
    keys = [f"{r.parent.name}/{r.name}" for r in rdirs]
    hashes = set()
    for r in rdirs:
        if done_folds(r) != set(range(1, N_FOLDS + 1)):
            raise ValueError(f"{r.name}: not all five dev folds are done yet; cannot select it")
        if (r / "SELECTED").exists():
            raise ValueError(f"{r.name} has already been selected")
        hashes.add(config_hash(json.loads((r / "config.json").read_text())))
    if len(hashes) != 1:
        raise ValueError(f"runs selected together must share one configuration (differing only in seed): {keys}")
    rec = {"selected_at": _now(), "reason": reason, "runs": keys}
    if spec is not None:
        rec["robustness"] = {**spec, "conditions": robust_conditions(spec)}
    for r in rdirs:
        (r / "SELECTED").write_text(json.dumps(rec, ensure_ascii=False))


def is_selected(rdir) -> bool:
    return (Path(rdir) / "SELECTED").exists()


def save_test(rdir, fold: int, test_preds: pd.DataFrame) -> None:
    """final: save the test predictions of one selected run and fold, once (the caller reloads the fold's best checkpoint and predicts on the test fold). Test accuracy is computed from test_preds; self-reported numbers are not accepted."""
    rdir = Path(rdir)
    if not is_selected(rdir):
        raise PermissionError(f"{rdir.name} has not been through select_for_final() yet: test folds are only for selected runs")
    fd = rdir / f"fold{fold}"
    if not (fd / "_SUCCESS").exists():
        raise ValueError(f"fold {fold} has no dev results")
    if (fd / "_TESTED").exists():
        raise ValueError(f"the test fold of fold {fold} has already been run (each test fold is used only once)")
    _check_cols(test_preds, TEST_PRED_COLS, "test_preds")
    _check_labels(test_preds, f"fold {fold} test_preds")
    test_preds[TEST_PRED_COLS].to_csv(fd / "test_preds.csv", index=False)
    now = _now()
    (fd / "_TESTED").write_text(now + "\n")
    with open(rdir / "TEST_LOG.csv", "a") as f:   # append-only: re-testing after deleting _TESTED leaves a second line here
        f.write(f"{now},{fold},{sha256(fd / 'test_preds.csv')}\n")


# ───────────────────────── robustness (declared at selection, after the normal test) ─────────────────────────
def _robust_spec(spec) -> dict:
    """Validate a robustness declaration and return it normalised: {"noise": str, "snr_db": [int, ...], "noise_seed": int}."""
    if not isinstance(spec, dict):
        raise ValueError(f"robustness must be a dict like {{'noise': 'white', 'snr_db': [20, 10, 0], 'noise_seed': 0}}, got {spec!r}")
    extra, missing = sorted(set(spec) - set(ROBUST_KEYS) - {"conditions"}), [k for k in ROBUST_KEYS if k not in spec]
    if extra or missing:
        raise ValueError(f"robustness must have exactly the keys {list(ROBUST_KEYS)} (missing {missing}, unknown {extra})")
    if spec["noise"] not in ROBUST_NOISES:
        raise ValueError(f"robustness noise must be one of {ROBUST_NOISES}, got {spec['noise']!r}")
    snrs = spec["snr_db"]
    if not isinstance(snrs, (list, tuple)) or not snrs:
        raise ValueError(f"robustness snr_db must be a non-empty list of numbers, got {snrs!r}")
    out = []
    for v in snrs:
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) or not math.isfinite(float(v)):
            raise ValueError(f"robustness snr_db must contain finite numbers only, got {v!r}")
        if float(v) != int(float(v)):
            raise ValueError(f"robustness snr_db values must be whole dB (e.g. 20, 10, 0, -5), got {v!r}:"
                             f" the condition name and the noise seed are built from the integer value")
        out.append(int(float(v)))
    if len(set(out)) != len(out):
        raise ValueError(f"robustness snr_db has duplicates: {list(snrs)}")
    seed = spec["noise_seed"]
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError(f"robustness noise_seed must be a non-negative int, got {seed!r}")
    return {"noise": spec["noise"], "snr_db": out, "noise_seed": int(seed)}


def robust_conditions(spec) -> list:
    """Condition names of a robustness declaration, in declared order: {"noise": "white", "snr_db": [20, 10, 0], ...}
    -> ["white_snr20", "white_snr10", "white_snr0"] (negative SNR: "white_snr-5"). These are the condition arguments of
    save_robust and the file names fold<k>/robust_<condition>.csv."""
    spec = _robust_spec(spec)
    return [f"{spec['noise']}_snr{s}" for s in spec["snr_db"]]


def robust_declaration(rdir):
    """The robustness declaration stored in SELECTED (dict with noise / snr_db / noise_seed / conditions), or None."""
    p = Path(rdir) / "SELECTED"
    return json.loads(p.read_text()).get("robustness") if p.exists() else None


def read_robust_log(rdir) -> pd.DataFrame:
    """ROBUST_LOG.csv as a DataFrame (empty with the right columns if there is none)."""
    p = Path(rdir) / "ROBUST_LOG.csv"
    if not p.exists():
        return pd.DataFrame(columns=ROBUST_LOG_COLS)
    log = pd.read_csv(p, dtype={"time": str, "condition": str, "sha256": str})
    if list(log.columns) != ROBUST_LOG_COLS:
        raise ValueError(f"{p} has columns {list(log.columns)}, expected {ROBUST_LOG_COLS} (was it edited by hand?)")
    return log


def save_robust(rdir, fold: int, condition: str, preds: pd.DataFrame, split=None) -> None:
    """final, robustness: predictions of the reloaded best checkpoint on the test fold of `fold` with the noise of
    `condition` added (no training, no tuning). Allowed only if
      · the run is SELECTED with a robustness declaration that contains `condition` (declared before testing);
      · the fold's normal test is done (_TESTED exists): the clean test comes first;
      · (fold, condition) has not been saved before -- checked in ROBUST_LOG.csv, not only by the file.
    preds = DataFrame[filename, y_true, y_pred, prob_top1] on exactly the official test fold with correct labels
    (checked against the fold's test_preds.csv, and against split if passed). Writes fold<k>/robust_<condition>.csv
    and appends time,fold,condition,sha256 to ROBUST_LOG.csv (append-only)."""
    rdir = Path(rdir)
    if not is_selected(rdir):
        raise PermissionError(f"{rdir.name} has not been through select_for_final(): robustness runs are only for selected runs")
    decl = robust_declaration(rdir)
    if not decl:
        raise PermissionError(f"{rdir.name} was selected without a robustness declaration: robustness must be declared in"
                              f" select_for_final(..., robustness=...) and cannot be added afterwards")
    declared = robust_conditions(decl)
    if condition not in declared:
        raise ValueError(f"condition {condition!r} was not declared for {rdir.name}; declared conditions: {declared}")
    if fold not in range(1, N_FOLDS + 1):
        raise ValueError(f"fold must be 1–{N_FOLDS}: {fold!r}")
    fd = rdir / f"fold{fold}"
    if not (fd / "_TESTED").exists() or not (fd / "test_preds.csv").exists():
        raise PermissionError(f"fold {fold} has not been tested yet: run the normal test (save_test) before the robustness conditions")
    log = read_robust_log(rdir)
    if ((log["fold"].astype(int) == fold) & (log["condition"] == condition)).any() or (fd / f"robust_{condition}.csv").exists():
        raise ValueError(f"fold {fold} / {condition} has already been evaluated (each robustness condition is run only once per fold;"
                         f" see ROBUST_LOG.csv)")
    _check_cols(preds, TEST_PRED_COLS, "robust preds")
    _check_labels(preds, f"fold {fold} {condition} preds")
    clean = pd.read_csv(fd / "test_preds.csv")
    err = _same_files_and_labels(preds, clean)
    if err:
        raise ValueError(f"fold {fold} {condition}: {err} (compared with this fold's test_preds.csv)")
    if split is not None:
        err = _test_set_problem(split, fold, preds)
        if err:
            raise ValueError(f"fold {fold} {condition}: {err}")
    out = fd / f"robust_{condition}.csv"
    preds[TEST_PRED_COLS].to_csv(out, index=False)
    lp = rdir / "ROBUST_LOG.csv"
    new = not lp.exists()
    with open(lp, "a") as f:                       # append-only, like TEST_LOG.csv
        if new:
            f.write(",".join(ROBUST_LOG_COLS) + "\n")
        f.write(f"{_now()},{fold},{condition},{sha256(out)}\n")


def _same_files_and_labels(a: pd.DataFrame, b: pd.DataFrame):
    if a["filename"].duplicated().any() or set(a["filename"]) != set(b["filename"]):
        return "the files are not the official test fold"
    if (a["y_true"].values != b.set_index("filename").loc[a["filename"], "y_true"].values).any():
        return "y_true does not match the labels of the test fold"
    return None


def _test_set_problem(split: pd.DataFrame, fold: int, tp: pd.DataFrame):
    want = set(split.loc[split["fold"] == fold, "filename"])
    if set(tp["filename"]) != want or tp["filename"].duplicated().any():
        return f"the files are not official fold {fold}"
    if (tp["y_true"].values != split.set_index("filename").loc[tp["filename"], "target"].values).any():
        return "y_true does not match the labels in split.csv"
    return None


# ───────────────────────── reading ─────────────────────────
def load_run(rdir) -> dict:
    rdir = Path(rdir)
    if not rdir.exists():
        raise FileNotFoundError(f"{rdir} does not exist: check the spelling of run_id, or the run has not been run yet")
    cfg = json.loads((rdir / "config.json").read_text())
    folds = sorted(done_folds(rdir))
    if not folds:
        raise ValueError(f"{rdir.name} has no complete fold yet")
    rd = lambda k, f: pd.read_csv(rdir / f"fold{k}" / f)  # noqa: E731
    dev = pd.concat([rd(k, "metrics.csv") for k in folds], ignore_index=True)
    curves = pd.concat([rd(k, "curves.csv") for k in folds], ignore_index=True)
    val_preds = {k: rd(k, "val_preds.csv") for k in folds}
    tested = sorted(tested_folds(rdir))
    test_preds = {k: rd(k, "test_preds.csv") for k in tested}
    sel = json.loads((rdir / "SELECTED").read_text()) if (rdir / "SELECTED").exists() else None
    tested_at = {k: (rdir / f"fold{k}" / "_TESTED").read_text().strip() for k in tested}
    robust = {}                                   # {fold: {condition: DataFrame}} -- every robust_*.csv found, declared or not (check_run judges)
    for p in sorted(rdir.glob("fold*/robust_*.csv")):
        m, mf = ROBUST_FILE_RE.match(p.name), FOLD_RE.match(p.parent.name)
        if m and mf:
            robust.setdefault(int(mf.group(1)), {})[m.group(1)] = pd.read_csv(p)
    env = {}                                      # hardware / versions per fold (cost table); optional metadata
    for k in folds:
        try:
            env[k] = json.loads((rdir / f"fold{k}" / "env.json").read_text())
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue                              # missing or unreadable env.json must not invalidate the run
    return {"key": f"{rdir.parent.name}/{rdir.name}", "config": cfg, "dev": dev, "curves": curves,
            "val_preds": val_preds, "test_preds": test_preds, "selected": sel, "tested_at": tested_at,
            "robust_preds": robust, "env": env}


def load_run_or_none(rdir):
    """Read the run from the saved FROZEN_RUNS (support/frozen/); a run that is not saved there is read from its folder when the folder exists; if neither, print a note and return None instead of raising."""
    rdir = Path(rdir)
    key = f"{rdir.parent.name}/{rdir.name}"
    if key in FROZEN_RUNS:
        return _unpack_run(key, FROZEN_RUNS[key])
    if rdir.exists():
        return load_run(rdir)
    if rdir.parent.exists():                     # results/ exists but the run does not: most likely a typo in run_name / date, do not stay silent
        have = sorted(p.name for p in rdir.parent.iterdir() if p.is_dir())
        print(f"⚠️⚠️ run {key} not found (not run yet, or run_name / date / seed in CONFIG is wrong)."
              f" Existing runs in {rdir.parent.name}: {have or '(none)'}")
        return None
    print(f"{key}: not in the saved results in support/frozen/")
    return None


def test_acc(run: dict) -> pd.Series:
    """M1: test accuracy per fold (computed from test_preds)."""
    return pd.Series({k: float((p["y_true"] == p["y_pred"]).mean()) for k, p in sorted(run["test_preds"].items())})


def macro_f1(y_true, y_pred, n_classes: int = N_CLASSES) -> float:
    """Arithmetic mean of the 50 per-class F1 scores (no sklearn needed; a class that is neither predicted nor present is skipped)."""
    yt, yp = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    f1s = []
    for c in range(n_classes):
        tp = int(((yt == c) & (yp == c)).sum()); fp = int(((yt != c) & (yp == c)).sum()); fn = int(((yt == c) & (yp != c)).sum())
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def test_macro_f1(run: dict) -> pd.Series:
    """M2: test macro-F1 per fold (computed from test_preds)."""
    return pd.Series({k: macro_f1(p["y_true"], p["y_pred"]) for k, p in sorted(run["test_preds"].items())})


def confusion(runs: list, n_classes: int = N_CLASSES) -> pd.DataFrame:
    """M3: confusion matrix summed over all seeds and all five test folds of one method (rows = true class, columns = predicted class, counts)."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for r in runs:
        for p in r["test_preds"].values():
            np.add.at(cm, (p["y_true"].astype(int).values, p["y_pred"].astype(int).values), 1)
    return pd.DataFrame(cm)


def per_class_recall(table_runs: dict, class_names=None) -> pd.DataFrame:
    """M3: per-class recall (%), rows = classes, columns = methods; table_runs = {"method name": [runs, ...]} (output of load_final_runs)."""
    out = {}
    for name, runs in table_runs.items():
        cm = confusion(runs).values
        out[name] = np.round(100 * np.diag(cm) / np.maximum(cm.sum(axis=1), 1), 1)
    df = pd.DataFrame(out)
    if class_names is not None:
        df.index = list(class_names)
    return df


def load_final_runs(results_root, final_runs: dict) -> dict:
    """FINAL_RUNS → {"method name": [runs, ...]}, for per_class_recall / confusion."""
    return {name: [load_run_or_none(Path(results_root) / rel) for rel in ([rels] if isinstance(rels, str) else rels)]
            for name, rels in final_runs.items()}


def dev_summary(run: dict) -> dict:
    d = run["dev"].sort_values("fold")
    return {"run": run["key"], "folds_done": len(d), "val_acc_mean": round(float(d["best_val_acc"].mean()) * 100, 1),
            "train_min": round(float(d["train_seconds"].sum()) / 60, 1), "params_M": round(float(d["n_params"].iloc[0]) / 1e6, 2)}


# ───────────────────────── checks (check_run) ─────────────────────────
def check_run(rdir, split: pd.DataFrame, results_root=None, require_final: bool = False) -> list:
    """Check one run's result files against the frozen split; returns a list of problems (empty = pass).
    This only proves the result files are consistent with the frozen split and the config; it does not run the
    model again."""
    rdir, errs = Path(rdir), []
    try:
        run = load_run(rdir)
    except Exception as e:                       # noqa: BLE001
        return [str(e)]
    cfg = run["config"]
    if rdir.parent.name != cfg.get("line") or rdir.name != run_id(cfg):
        errs.append(f"folder name {rdir.parent.name}/{rdir.name} does not match the config")
    if SPLIT_SHA256 and cfg.get("split_sha256") != SPLIT_SHA256:
        errs.append("split_sha256 in the config is not the frozen split")
    errs += config_problems(cfg, results_root)
    by_name = split.set_index("filename")
    for k, vp in run["val_preds"].items():
        want = set(split.loc[(split[f"val_r{k}"] == 1) & (split["fold"] != k), "filename"])
        err = _val_set_problem(split, k, vp)
        if err:
            errs.append(f"fold {k}: {err}")
        row = run["dev"].loc[run["dev"]["fold"] == k].iloc[0]
        cur_k = run["curves"].loc[run["curves"]["fold"] == k] if len(run["curves"]) else run["curves"]
        if len(cur_k):
            hit = cur_k.loc[cur_k["epoch"] == row["best_epoch"]]
            if hit.empty or abs(float(hit["val_acc"].iloc[0]) - float(row["best_val_acc"])) > 5e-4:
                errs.append(f"fold {k}: val_acc in the best_epoch row of curves does not match metrics")
        if int(row["val_n"]) != len(want):
            errs.append(f"fold {k}: val_n={int(row['val_n'])}, the frozen validation set has {len(want)} clips")
        if abs(float(row["best_val_acc"]) - float(row["best_val_correct"]) / max(int(row["val_n"]), 1)) > 1e-9:
            errs.append(f"fold {k}: best_val_acc ≠ best_val_correct / val_n")
        if not all(math.isfinite(float(row[c])) for c in DEV_COLS[1:]):
            errs.append(f"fold {k}: metrics contains non-finite values")
    if run["dev"]["fold"].duplicated().any():
        errs.append("metrics has duplicate folds")
    cur = run["curves"]
    if len(cur) and cur.duplicated(["fold", "epoch"]).any():
        errs.append("curves has duplicate epochs within a fold")
    if run["test_preds"]:
        if not run["selected"]:
            errs.append("test results exist but there is no SELECTED: the test folds were run before selection")
        for k, tp in run["test_preds"].items():
            want = set(split.loc[split["fold"] == k, "filename"])
            try:
                _check_labels(tp, f"fold {k} test_preds")
            except ValueError as e:
                errs.append(str(e))
            if set(tp["filename"]) != want or tp["filename"].duplicated().any():
                errs.append(f"fold {k}: the files in test_preds are not official fold {k}")
            elif (tp["y_true"].values != by_name.loc[tp["filename"], "target"].values).any():
                errs.append(f"fold {k}: y_true in test_preds does not match the labels in split.csv")
            if run["selected"] and run["tested_at"][k] < run["selected"]["selected_at"]:
                errs.append(f"fold {k}: tested before it was selected")
    if require_final and set(run["test_preds"]) != set(range(1, N_FOLDS + 1)):
        errs.append(f"the results table needs test results for all five folds, currently only {sorted(run['test_preds'])}")
    errs += _robust_problems(rdir, run, split)
    return errs


def _robust_problems(rdir, run: dict, split: pd.DataFrame) -> list:
    """check_run part for robustness files: declared, logged exactly once, after selection and after the clean test, on the official fold."""
    rdir, errs = Path(rdir), []
    files = {(k, c): df for k, d in run["robust_preds"].items() for c, df in d.items()}
    try:
        log = read_robust_log(rdir)
    except Exception as e:                       # noqa: BLE001
        return [str(e)]
    if not files and not len(log):
        return errs
    decl = (run["selected"] or {}).get("robustness")
    if not run["selected"]:
        errs.append("robustness results exist but there is no SELECTED")
        declared = []
    elif not decl:
        errs.append("robustness results exist but SELECTED has no robustness declaration (it must be declared at selection)")
        declared = []
    else:
        try:
            declared = robust_conditions(decl)
        except ValueError as e:
            errs.append(f"robustness declaration in SELECTED is invalid: {e}")
            declared = []
    counts = log.groupby([log["fold"].astype(int), log["condition"]]).size() if len(log) else pd.Series(dtype=int)
    for (k, c), n in counts.items():
        if n > 1:
            errs.append(f"fold {k} / {c}: ROBUST_LOG.csv has {n} rows (each robustness condition may be run only once per fold)")
        if (k, c) not in files:
            errs.append(f"fold {k} / {c}: logged in ROBUST_LOG.csv but fold{k}/robust_{c}.csv is missing")
    for (k, c), df in sorted(files.items()):
        where = f"fold {k} / {c}"
        if declared and c not in declared:
            errs.append(f"{where}: robust_{c}.csv is not a declared condition (declared: {declared})")
        rows = log[(log["fold"].astype(int) == k) & (log["condition"] == c)] if len(log) else log
        if rows.empty:
            errs.append(f"{where}: robust_{c}.csv has no row in ROBUST_LOG.csv (not written by save_robust, or the log was edited)")
        else:
            if rows["sha256"].iloc[-1] != sha256(rdir / f"fold{k}" / f"robust_{c}.csv"):
                errs.append(f"{where}: robust_{c}.csv was changed after it was saved (sha256 differs from ROBUST_LOG.csv)")
            t = str(rows["time"].iloc[0])
            if run["selected"] and t < run["selected"]["selected_at"]:
                errs.append(f"{where}: robustness run before the run was selected")
            if k not in run["tested_at"]:
                errs.append(f"{where}: robustness results for a fold without a normal test (_TESTED)")
            elif t < run["tested_at"][k]:
                errs.append(f"{where}: robustness run before the normal test of fold {k}")
        missing = [col for col in TEST_PRED_COLS if col not in df.columns]
        if missing:
            errs.append(f"{where}: missing columns {missing}")
            continue
        try:
            _check_labels(df, where)
        except ValueError as e:
            errs.append(str(e))
            continue
        err = _test_set_problem(split, k, df)
        if err:
            errs.append(f"{where}: {err}")
    return errs


# ───────────────────────── results table ─────────────────────────
def final_table(results_root, final_runs: dict, expected_split_sha256: str | None = None) -> tuple:
    """final_runs = {"method name": ["line2/<run_id>", ... one per seed]} → (results table, per-seed detail).
    Table columns: mean ± std_over_folds = M1 test accuracy (%); macro_f1 ± macro_f1_std = M2 (%); M3 via per_class_recall / confusion.
    Aggregation: average over seeds within each fold first, then report mean ± SD (sample standard deviation) over the
    five folds; seed_spread = range of the per-seed five-fold means; per-seed numbers go in a separate table.
    All seeds of one method must share one configuration (same config_hash) and no seed may repeat.
    Differences are descriptive only: no significance tests; when the gap between two methods is smaller than both
    seed_spreads, write "no clear difference"."""
    if expected_split_sha256 is None:
        expected_split_sha256 = SPLIT_SHA256 or _frozen_split_sha256()
    rows, detail = [], []
    for name, rels in final_runs.items():
        rels = [rels] if isinstance(rels, str) else list(rels)
        runs = []
        for rel in rels:
            run = load_run_or_none(Path(results_root) / rel)
            if run is None:
                raise FileNotFoundError(f"{rel}: neither in results/ nor in support/frozen/")
            if set(run["test_preds"]) != set(range(1, N_FOLDS + 1)):
                raise ValueError(f"{rel}: test results for all five folds are not there yet (select_for_final first, then run the test folds)")
            if expected_split_sha256 and run["config"].get("split_sha256") != expected_split_sha256:
                raise ValueError(f"{rel}: does not use the frozen split")
            runs.append(run)
        seeds = [r["config"]["seed"] for r in runs]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"{name}: duplicate seeds {seeds}")
        ref = config_hash(runs[0]["config"])
        bad = [r["key"] for r in runs if config_hash(r["config"]) != ref]
        if bad:
            raise ValueError(f"{name}: these runs have a different experiment configuration from the first one (everything except seed / date must be identical): {bad}")
        listed = {r["key"] for r in runs}
        for r in runs:                            # runs selected together must all be listed: no picking seeds afterwards
            chosen = set((r["selected"] or {}).get("runs", [r["key"]]))
            if chosen != listed:
                raise ValueError(f"{name}: the listed runs {sorted(listed)} differ from the runs selected together {sorted(chosen)}"
                                 f" (once a configuration is selected, all its seeds go into the table; do not drop seeds based on test scores)")
        groups = {r["config"]["group"] for r in runs}
        per_seed = pd.DataFrame({r["config"]["seed"]: test_acc(r) for r in runs}) * 100
        per_fold = per_seed.mean(axis=1)
        f1_fold = (pd.DataFrame({r["config"]["seed"]: test_macro_f1(r) for r in runs}) * 100).mean(axis=1)
        seed_means = per_seed.mean(axis=0)
        rows.append({"method": name, "group": groups.pop(), "seeds": len(runs), "config_hash": ref,
                     "mean": round(per_fold.mean(), 1), "std_over_folds": round(per_fold.std(ddof=1), 1),
                     "macro_f1": round(f1_fold.mean(), 1), "macro_f1_std": round(f1_fold.std(ddof=1), 1),
                     "seed_spread": round(seed_means.max() - seed_means.min(), 1) if len(runs) > 1 else float("nan"),
                     "folds": " / ".join(f"{a:.1f}" for a in per_fold),
                     "train_min_total": round(sum(r["dev"]["train_seconds"].sum() for r in runs) / 60, 1)})
        for seed, col in per_seed.items():
            detail.append({"method": name, "seed": seed, **{f"fold{k}": round(v, 1) for k, v in col.items()},
                           "mean": round(col.mean(), 1)})
    df = pd.DataFrame(rows)
    df["_g"] = df["group"].map({g: i for i, g in enumerate(GROUPS)})
    df = df.sort_values(["_g", "mean"], ascending=[True, False]).drop(columns="_g").reset_index(drop=True)
    return df, pd.DataFrame(detail)


def robust_table(table_runs: dict) -> pd.DataFrame:
    """Robustness results: rows = methods, columns = clean (normal test accuracy) and each declared condition, as
    accuracy (%) with a "<column>_sd" column next to it. Same convention as final_table: average over seeds within each
    fold first, then mean ± SD (sample standard deviation) over the five folds.
    table_runs = {"method name": [runs, ...]} (output of load_final_runs). Methods whose runs have no robustness
    declaration are skipped (a note is printed); a declared method with missing (fold, condition) results raises."""
    rows, skipped = [], []
    for name, runs in table_runs.items():
        if any(r is None for r in runs):
            raise FileNotFoundError(f"{name}: some runs are neither in results/ nor in support/frozen/")
        decls = [(r["selected"] or {}).get("robustness") for r in runs]
        if not any(decls):
            skipped.append(name)
            continue
        if not all(decls) or len({json.dumps(d, sort_keys=True) for d in decls}) != 1:
            raise ValueError(f"{name}: the runs do not share one robustness declaration (they must be selected together)")
        conds = robust_conditions(decls[0])
        row = {"method": name, "seeds": len(runs)}
        cols = {"clean": lambda r: {k: p for k, p in r["test_preds"].items()}}
        for c in conds:
            cols[c] = (lambda c: lambda r: {k: d[c] for k, d in r.get("robust_preds", {}).items() if c in d})(c)
        for col, get in cols.items():
            per_seed = {}
            for r in runs:
                preds = get(r)
                if set(preds) != set(range(1, N_FOLDS + 1)):
                    raise ValueError(f"{name}: {r['key']} has {col} results for folds {sorted(preds)}, all five are needed")
                per_seed[r["config"]["seed"]] = pd.Series({k: float((p["y_true"] == p["y_pred"]).mean()) for k, p in sorted(preds.items())})
            per_fold = (pd.DataFrame(per_seed) * 100).mean(axis=1)
            row[col], row[f"{col}_sd"] = round(per_fold.mean(), 1), round(per_fold.std(ddof=1), 1)
        rows.append(row)
    if skipped:
        print(f"robust_table: no robustness declaration for {skipped} (skipped)")
    return pd.DataFrame(rows).set_index("method") if rows else pd.DataFrame()


def config_hash(config: dict) -> str:
    """Experiment-configuration fingerprint: all settings except seed / date / descriptive fields. Must be identical across the seeds of one method."""
    core = {k: v for k, v in config.items() if k not in {"seed", "date", "split_sha256", "notes", "question",
                                                         "compare_to", "changed_variable", "common_version"}}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:12]


# ───────────────────────── reading the saved results in support/frozen/ ─────────────────────────
def unpack_runs(payload: str) -> dict:
    return json.loads(zlib.decompress(base64.b64decode(payload)).decode()) if payload else {}


def _unpack_run(key: str, raw: dict) -> dict:
    return {"key": key, "config": raw["config"], "dev": pd.DataFrame(raw["dev"]),
            "curves": pd.DataFrame(raw["curves"], columns=CURVE_COLS),
            "val_preds": {int(k): pd.DataFrame(v, columns=VAL_PRED_COLS) for k, v in raw["val_preds"].items()},
            "test_preds": {int(k): pd.DataFrame(v, columns=TEST_PRED_COLS) for k, v in raw["test_preds"].items()},
            "selected": raw["selected"], "tested_at": {int(k): v for k, v in raw["tested_at"].items()},
            "robust_preds": {int(k): {c: pd.DataFrame(v, columns=TEST_PRED_COLS) for c, v in d.items()}
                             for k, d in raw.get("robust_preds", {}).items()},
            "env": {int(k): v for k, v in raw.get("env", {}).items()}}     # optional environment metadata (empty dict if absent)


def _check_labels(df: pd.DataFrame, what: str) -> None:
    for c in ("y_true", "y_pred"):
        v = pd.to_numeric(df[c], errors="coerce")
        if v.isna().any() or (v % 1 != 0).any() or not v.between(0, N_CLASSES - 1).all():
            raise ValueError(f"{what}: {c} must be an integer class index 0–{N_CLASSES - 1}")
    if "prob_top1" in df.columns:
        p = pd.to_numeric(df["prob_top1"], errors="coerce")
        if p.isna().any() or not p.between(0, 1).all():
            raise ValueError(f"{what}: prob_top1 must be in [0, 1]")


def _val_set_problem(split: pd.DataFrame, fold: int, vp: pd.DataFrame):
    want = set(split.loc[(split[f"val_r{fold}"] == 1) & (split["fold"] != fold), "filename"])
    if set(vp["filename"]) != want or vp["filename"].duplicated().any():
        return "the files in val_preds are not this round's frozen validation set"
    if (vp["y_true"].values != split.set_index("filename").loc[vp["filename"], "target"].values).any():
        return "y_true in val_preds does not match the labels in split.csv"
    return None


def _frozen_split_sha256():
    """When the split was not verified in this process, use the frozen value hard-coded in data.py."""
    try:
        import data
        h = data.EXPECTED_SPLIT_SHA256
        return None if h.startswith("TODO") else h
    except Exception:                            # noqa: BLE001
        return None


def _env() -> dict:
    import platform
    import sys
    env = {"saved_at": _now(), "python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        env["torch"] = torch.__version__
        env["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    except ImportError:
        env["device"] = "cpu"
    return env


def _check_cols(df: pd.DataFrame, cols: list, what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{what} is missing columns {missing} (the shared format is described at the top of eval_io.py)")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")   # UTC, comparable across devices
