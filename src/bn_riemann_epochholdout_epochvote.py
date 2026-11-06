import os
import re
import numpy as np
from glob import glob
from collections import defaultdict

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


# ================= EDIT THIS PATH =================
DATA_DIR = r"C:\Users\USER\Documents\Gobeshona\Thesis\Thesis data collection\only bn pp files"
# ==================================================

# Labels: 1 to 6
LABEL_MAP = {
    "back": 1,
    "down": 2,
    "forward": 3,
    "left": 4,
    "right": 5,
    "up": 6
}

CLASS_LABELS = [1, 2, 3, 4, 5, 6]
CLASS_NAMES = ["back", "down", "forward", "left", "right", "up"]

# Bangla filename token to English label
BN_MAP = {
    "pichone": "back",
    "niche": "down",
    "samne": "forward",
    "ete": "forward",      # keep this in case old files used ete
    "bame": "left",
    "dane": "right",
    "upore": "up"
}

WIN_SEC = 2.0
OVERLAP = 0.50
DO_Z = True


def parse_meta(meta_path):
    txt = open(meta_path, "r", encoding="utf-8", errors="ignore").read().lower()

    def g(k):
        m = re.search(rf"{k}\s*=\s*([0-9]+(\.[0-9]+)?)", txt)
        return float(m.group(1)) if m else None

    fs = g("srate")
    channels = g("channels")
    epochs = g("epochs")
    pnts_per_epoch = g("pnts_per_epoch")

    if fs is None or channels is None or epochs is None or pnts_per_epoch is None:
        raise ValueError(f"Could not parse meta file correctly: {meta_path}")

    return float(fs), int(channels), int(epochs), int(pnts_per_epoch)


def normalize_token(s):
    s = s.strip().lower()
    s = re.sub(r"[^a-z]", "", s)
    return s


def label_from_filename(base):
    """
    Expected filename examples:
    tonmoy_upore_pp
    rahim_niche_pp
    nibedita_pichone_1_pp

    The label token is usually the second part after splitting by underscore.
    """
    parts = base.split("_")

    if len(parts) < 2:
        raise ValueError(f"Could not parse label from filename: {base}")

    token = normalize_token(parts[1])

    if token in BN_MAP:
        english_label = BN_MAP[token]
        return LABEL_MAP[english_label]

    if token in LABEL_MAP:
        return LABEL_MAP[token]

    raise ValueError(f"Unknown Bangla label token '{token}' in file: {base}")


def find_meta_file(csv_path):
    base = os.path.splitext(os.path.basename(csv_path))[0]

    candidates = [
        os.path.join(DATA_DIR, base + "_meta.txt"),
        os.path.join(DATA_DIR, base.replace("_pp", "") + "_meta.txt"),
        os.path.join(DATA_DIR, base + "_pp_meta.txt"),
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(f"No meta file found for: {csv_path}")


def load_epochs(csv_path):
    base = os.path.splitext(os.path.basename(csv_path))[0]
    meta = find_meta_file(csv_path)

    fs, C, E, T = parse_meta(meta)

    subject = base.split("_")[0]
    y = label_from_filename(base)

    M = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)

    if M.ndim == 1:
        M = M.reshape(-1, C)

    if M.shape[1] < C:
        raise ValueError(f"CSV has fewer columns than expected channels: {csv_path}")

    M = M[:, :C]

    expected_rows = E * T
    if M.shape[0] != expected_rows:
        raise ValueError(
            f"Row count mismatch in {csv_path}\n"
            f"Expected rows: {expected_rows}, got: {M.shape[0]}\n"
            f"E={E}, T={T}, C={C}"
        )

    epochs = np.zeros((E, T, C), dtype=np.float32)

    for ch in range(C):
        epochs[:, :, ch] = M[:, ch].reshape(T, E).T

    return {
        "file_id": base,
        "subject": subject,
        "y": y,
        "epochs": epochs,
        "fs": fs
    }


def windowize(epoch_TxC, fs):
    win = int(WIN_SEC * fs)
    step = int(win * (1 - OVERLAP))

    if step < 1:
        step = 1

    T, C = epoch_TxC.shape
    windows = []

    for start in range(0, T - win + 1, step):
        seg = epoch_TxC[start:start + win, :]

        if DO_Z:
            seg = (seg - seg.mean(axis=0, keepdims=True)) / (
                seg.std(axis=0, keepdims=True) + 1e-6
            )

        windows.append(seg.T)  # pyriemann expects shape: channels x time

    return windows


def majority_vote(preds):
    """
    Majority vote for labels 1..6.
    """
    preds = np.asarray(preds, dtype=int)
    counts = {label: np.sum(preds == label) for label in CLASS_LABELS}
    return max(counts, key=counts.get)


def classwise_from_cm(cm):
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.diag(cm) / cm.sum(axis=1)

    return np.nan_to_num(per_class)


def run_bn_epochheldout_epochvote(items):
    win_true_all = []
    win_pred_all = []

    ep_true_all = []
    ep_pred_all = []

    by_subj = defaultdict(list)

    for item in items:
        by_subj[item["subject"]].append(item)

    for subject, files in sorted(by_subj.items()):
        E = min(item["epochs"].shape[0] for item in files)

        for test_ep in range(E):
            Xtr = []
            ytr = []

            for item in files:
                y = item["y"]
                epochs = item["epochs"]
                fs = item["fs"]

                for e in range(E):
                    if e == test_ep:
                        continue

                    segs = windowize(epochs[e], fs)
                    Xtr.extend(segs)
                    ytr.extend([y] * len(segs))

            Xtr = np.stack(Xtr, axis=0)
            ytr = np.array(ytr, dtype=np.int64)

            pipe = Pipeline([
                ("cov", Covariances(estimator="oas")),
                ("ts", TangentSpace(metric="riemann")),
                ("sc", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=10, gamma="scale"))
            ])

            pipe.fit(Xtr, ytr)

            for item in files:
                y_true = item["y"]
                fs = item["fs"]

                segs = windowize(item["epochs"][test_ep], fs)
                Xte = np.stack(segs, axis=0)

                pred_w = pipe.predict(Xte)

                win_true_all.append(np.full(len(pred_w), y_true, dtype=np.int64))
                win_pred_all.append(pred_w)

                y_epoch_pred = majority_vote(pred_w)

                ep_true_all.append(y_true)
                ep_pred_all.append(y_epoch_pred)

        print("Done subject:", subject)

    win_true = np.concatenate(win_true_all)
    win_pred = np.concatenate(win_pred_all)

    ep_true = np.array(ep_true_all, dtype=np.int64)
    ep_pred = np.array(ep_pred_all, dtype=np.int64)

    win_acc = accuracy_score(win_true, win_pred)
    win_f1 = f1_score(win_true, win_pred, average="macro")

    ep_acc = accuracy_score(ep_true, ep_pred)
    ep_f1 = f1_score(ep_true, ep_pred, average="macro")

    cm = confusion_matrix(ep_true, ep_pred, labels=CLASS_LABELS)
    per_class = classwise_from_cm(cm)

    print("\n=== BN only, no leakage, epoch-heldout ===")
    print("WINDOW-level acc:", win_acc)
    print("WINDOW-level macroF1:", win_f1)

    print("\nEPOCH-VOTED acc:", ep_acc)
    print("EPOCH-VOTED macroF1:", ep_f1)

    print("\nEpoch class-wise accuracy:")
    for label, name, val in zip(CLASS_LABELS, CLASS_NAMES, per_class):
        print(f"Class {label} ({name}): {val:.4f}")

    print("\nEpoch confusion matrix rows=true, cols=pred")
    print("Labels:", CLASS_LABELS)
    print("Names :", CLASS_NAMES)
    print(cm)

    print("\nClassification report:")
    print(classification_report(
        ep_true,
        ep_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    ))


def main():
    files = sorted(glob(os.path.join(DATA_DIR, "*_pp.csv")))

    print("DATA_DIR:", DATA_DIR)
    print("Total Bangla pp files found:", len(files))

    if len(files) == 0:
        raise FileNotFoundError("No *_pp.csv files found. Check DATA_DIR.")

    items = []

    for p in files:
        item = load_epochs(p)
        items.append(item)

    print("Loaded files:", len(items))
    print("Class labels:", CLASS_LABELS)
    print("Class names :", CLASS_NAMES)

    label_counts = defaultdict(int)
    for item in items:
        label_counts[item["y"]] += 1

    print("\nFiles per class:")
    for label, name in zip(CLASS_LABELS, CLASS_NAMES):
        print(f"Class {label} ({name}): {label_counts[label]} files")

    run_bn_epochheldout_epochvote(items)


if __name__ == "__main__":
    main()