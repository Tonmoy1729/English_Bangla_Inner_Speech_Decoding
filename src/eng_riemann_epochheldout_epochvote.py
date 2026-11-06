import os, re
import numpy as np
from glob import glob
from collections import defaultdict

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

DATA_DIR = r"C:\Users\USER\Documents\Gobeshona\Thesis\Thesis data collection\English preprocessed"

LABEL_MAP = {"back":0, "down":1, "forward":2, "left":3, "right":4, "up":5}
IDX_TO_NAME = {v:k for k,v in LABEL_MAP.items()}

FS = 256.0
WIN_SEC = 1.0
OVERLAP = 0.75
DO_Z = True

def parse_meta(meta_path):
    txt = open(meta_path, "r", encoding="utf-8", errors="ignore").read().lower()
    def g(k):
        m = re.search(rf"{k}\s*=\s*([0-9]+(\.[0-9]+)?)", txt)
        return float(m.group(1)) if m else None
    return float(g("srate")), int(g("channels")), int(g("epochs")), int(g("pnts_per_epoch"))

def normalize_direction(raw):
    s = re.sub(r"[^a-z]", "", raw.strip().lower())
    if s.startswith("backward") or s == "back": return "back"
    if s.startswith("forward"): return "forward"
    if s.startswith("left"): return "left"
    if s.startswith("right"): return "right"
    if s.startswith("up"): return "up"
    if s.startswith("down"): return "down"
    return s

def load_epochs(csv_path):
    base = os.path.splitext(os.path.basename(csv_path))[0]  # subject_direction_pp
    meta = os.path.join(DATA_DIR, base + "_meta.txt")
    fs, C, E, T = parse_meta(meta)

    parts = base.split("_")
    subject = parts[0]
    direction = normalize_direction(parts[1])
    y = LABEL_MAP[direction]

    M = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)  # (E*T, C)
    epochs = np.zeros((E, T, C), dtype=np.float32)  # (E, time, ch)
    for ch in range(C):
        epochs[:, :, ch] = M[:, ch].reshape(T, E).T
    return subject, y, epochs, fs  # epochs: (E,T,C)

def windowize(epoch_TxC, fs):
    win = int(WIN_SEC * fs)
    step = int(win * (1 - OVERLAP))
    if step < 1: step = 1
    T, C = epoch_TxC.shape
    out = []
    for s in range(0, T - win + 1, step):
        seg = epoch_TxC[s:s+win, :]  # (win, C)
        if DO_Z:
            seg = (seg - seg.mean(axis=0, keepdims=True)) / (seg.std(axis=0, keepdims=True) + 1e-6)
        out.append(seg)
    return out  # list of (win,C)

def majority_vote(preds, n_classes=6):
    # preds: 1D array of window predictions
    return int(np.bincount(preds, minlength=n_classes).argmax())

def classwise_from_cm(cm):
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.diag(cm) / cm.sum(axis=1)
    return np.nan_to_num(per_class)

def main():
    files = sorted(glob(os.path.join(DATA_DIR, "*_pp.csv")))
    by_subj = defaultdict(list)
    for p in files:
        subj = os.path.basename(p).split("_")[0]
        by_subj[subj].append(p)

    # Store predictions
    win_true_all, win_pred_all = [], []
    ep_true_all, ep_pred_all = [], []

    for subj, flist in sorted(by_subj.items()):
        data = []
        for p in flist:
            s, y, epochs, fs = load_epochs(p)
            data.append((y, epochs, fs))

        if not data:
            continue

        E = data[0][1].shape[0]  # number of repeats/epochs per file (usually 4)

        for test_ep in range(E):
            Xtr_seg, ytr = [], []
            Xte_seg, yte = [], []
            ep_groups = []  # map each test window to an epoch label group (only one epoch here, but keep structure)

            for y, epochs, fs in data:
                for e in range(E):
                    segs = windowize(epochs[e], fs)  # list of (win,C)
                    if e == test_ep:
                        for seg in segs:
                            Xte_seg.append(seg.T)      # (C,win)
                            yte.append(y)
                            ep_groups.append((y, test_ep))  # epoch identity
                    else:
                        for seg in segs:
                            Xtr_seg.append(seg.T)      # (C,win)
                            ytr.append(y)

            Xtr = np.stack(Xtr_seg, axis=0)
            Xte = np.stack(Xte_seg, axis=0)
            ytr = np.array(ytr, dtype=np.int64)
            yte = np.array(yte, dtype=np.int64)

            pipe = Pipeline([
                ("cov", Covariances(estimator="oas")),
                ("ts", TangentSpace(metric="riemann")),
                ("sc", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=10, gamma="scale"))
            ])

            pipe.fit(Xtr, ytr)
            pred_win = pipe.predict(Xte)

            # window-level collect
            win_true_all.append(yte)
            win_pred_all.append(pred_win)

            # epoch-level voting:
            # Here, test set windows come from many directions but all from the same held-out epoch index.
            # We need to vote per (direction-file epoch). Since each direction file contributes one epoch,
            # we can vote per true label chunk by grouping windows by their true label.
            #
            # Better: vote per (file, epoch) but we didn't carry file-id. Since each file has fixed label,
            # vote per label within this fold is equivalent only if window counts per class are equal.
            #
            # So we will vote per file properly by adding file-id in the loop below.

        print("Done subject:", subj)

    # --- The above is window-level correct, but epoch voting must be per file ---
    # Let's do it properly in a second pass that tracks file-id.
    win_true_all, win_pred_all = [], []
    ep_true_all, ep_pred_all = [], []

    for subj, flist in sorted(by_subj.items()):
        # load all subject files once
        file_objs = []
        for p in flist:
            s, y, epochs, fs = load_epochs(p)
            file_id = os.path.splitext(os.path.basename(p))[0]
            file_objs.append((file_id, y, epochs, fs))

        if not file_objs:
            continue

        E = file_objs[0][2].shape[0]

        for test_ep in range(E):
            # Build train windows
            Xtr_seg, ytr = [], []
            for file_id, y, epochs, fs in file_objs:
                for e in range(E):
                    if e == test_ep:
                        continue
                    for seg in windowize(epochs[e], fs):
                        Xtr_seg.append(seg.T)
                        ytr.append(y)

            Xtr = np.stack(Xtr_seg, axis=0)
            ytr = np.array(ytr, dtype=np.int64)

            pipe = Pipeline([
                ("cov", Covariances(estimator="oas")),
                ("ts", TangentSpace(metric="riemann")),
                ("sc", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=10, gamma="scale"))
            ])
            pipe.fit(Xtr, ytr)

            # Test per file: predict windows of held-out epoch and vote
            for file_id, y_true, epochs, fs in file_objs:
                segs = windowize(epochs[test_ep], fs)
                Xte = np.stack([seg.T for seg in segs], axis=0)
                pred_w = pipe.predict(Xte)

                # window-level store
                win_true_all.append(np.full(len(pred_w), y_true, dtype=np.int64))
                win_pred_all.append(pred_w)

                # epoch-level vote store
                y_pred_epoch = majority_vote(pred_w, n_classes=6)
                ep_true_all.append(y_true)
                ep_pred_all.append(y_pred_epoch)

        # end subject
        # print("Done subject:", subj)

    win_true = np.concatenate(win_true_all)
    win_pred = np.concatenate(win_pred_all)
    ep_true = np.array(ep_true_all, dtype=np.int64)
    ep_pred = np.array(ep_pred_all, dtype=np.int64)

    # Window-level metrics
    win_acc = accuracy_score(win_true, win_pred)
    win_f1 = f1_score(win_true, win_pred, average="macro")
    print("\nWINDOW-level acc:", win_acc)
    print("WINDOW-level macroF1:", win_f1)

    # Epoch-level metrics (this is what you should compare to MATLAB 81%)
    ep_acc = accuracy_score(ep_true, ep_pred)
    ep_f1 = f1_score(ep_true, ep_pred, average="macro")
    cm_ep = confusion_matrix(ep_true, ep_pred, labels=np.arange(6))
    per_class_ep = classwise_from_cm(cm_ep)

    print("\nEPOCH-VOTED acc:", ep_acc)
    print("EPOCH-VOTED macroF1:", ep_f1)
    print("Epoch class-wise acc (back,down,forward,left,right,up):")
    print(per_class_ep)
    print("Epoch confusion matrix:\n", cm_ep)

    # Friendly print
    print("\nEpoch class-wise (name -> acc):")
    for i in range(6):
        print(f"{IDX_TO_NAME[i]:>7s}: {per_class_ep[i]:.3f}")

if __name__ == "__main__":
    main()