import os
import re
import math
import warnings
from glob import glob
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import t

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


# ============================================================
# PATHS
# ============================================================
ENG_DIR = (
    r"C:\Users\USER\Documents\Gobeshona\Thesis\Thesis data collection"
    r"\only eng pp files"
)
BN_DIR = (
    r"C:\Users\USER\Documents\Gobeshona\Thesis\Thesis data collection"
    r"\only bn pp files"
)
OUT_DIR = (
    r"C:\Users\USER\Documents\Gobeshona\Thesis\Thesis Revision Work"
    r"\FINAL_CALIBRATED_LOSO"
)


# ============================================================
# SETTINGS
# ============================================================
WIN_SEC = 2.0
OVERLAP = 0.50
DO_Z = True
SVM_C = 10
SVM_GAMMA = "scale"
N_CHANNELS = 12
EXPECTED_POINTS_PER_EPOCH = 1792

# CAL=1: one complete epoch per available direction from the held-out
# subject is used for calibration. Remaining epochs are test epochs.
# With 4 epochs/direction this is 25% calibration, not 20%.
CAL_EPOCHS_PER_DIRECTION = 1

# Rotated calibration removes dependence on one lucky calibration epoch.
# For ordinary 4-epoch recordings, there are 4 rotations.
# If a held-out subject contains a 3-epoch and 4-epoch recording, the
# least common multiple gives 12 rotations so each available epoch is
# selected equally often within its recording.
USE_ROTATED_CALIBRATION = True

CLASS_IDS = [1, 2, 3, 4, 5, 6]
CLASS_NAMES = {
    1: "back",
    2: "down",
    3: "forward",
    4: "left",
    5: "right",
    6: "up",
}
LABEL_TO_ID = {v: k for k, v in CLASS_NAMES.items()}

ENG_MAP = {
    "back": "back",
    "backward": "back",
    "down": "down",
    "forward": "forward",
    "left": "left",
    "right": "right",
    "up": "up",
}
BN_MAP = {
    "pichone": "back",
    "pichonee": "back",
    "niche": "down",
    "samne": "forward",
    "ete": "forward",
    "bame": "left",
    "dane": "right",
    "upore": "up",
}


# ============================================================
# METADATA / LABEL HELPERS
# ============================================================
def parse_meta(meta_path):
    with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().lower()

    def get_value(*keys):
        for key in keys:
            m = re.search(
                rf"{re.escape(key)}\s*=\s*([0-9]+(?:\.[0-9]+)?)",
                text,
            )
            if m:
                return float(m.group(1))
        return None

    fs = get_value("srate", "sampling_rate", "sample_frequency_rate")
    channels = get_value("channels")
    epochs = get_value("epochs")
    points = get_value("pnts_per_epoch", "points_per_epoch")

    if None in (fs, channels, epochs, points):
        raise ValueError(f"Could not parse metadata: {meta_path}")

    return float(fs), int(channels), int(epochs), int(points)


def find_meta_file(csv_path, data_dir):
    base = os.path.splitext(os.path.basename(csv_path))[0]
    candidates = [
        os.path.join(data_dir, base + "_meta.txt"),
        os.path.join(data_dir, base.replace("_pp", "") + "_meta.txt"),
        os.path.join(data_dir, base + "_pp_meta.txt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No metadata file found for {csv_path}")


def normalize_token(text):
    return re.sub(r"[^a-z]", "", text.strip().lower())


def get_label(filename_base, language):
    parts = filename_base.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse label from {filename_base}")
    token = normalize_token(parts[1])
    mapping = ENG_MAP if language == "English" else BN_MAP
    if token not in mapping:
        raise ValueError(
            f"Unknown {language} label '{token}' in file {filename_base}"
        )
    name = mapping[token]
    return LABEL_TO_ID[name], name


# ============================================================
# LOAD DATA
# ============================================================
def load_file(csv_path, data_dir, language):
    base = os.path.splitext(os.path.basename(csv_path))[0]
    meta_path = find_meta_file(csv_path, data_dir)
    fs, C, meta_epochs, T = parse_meta(meta_path)

    if C != N_CHANNELS:
        raise ValueError(f"{base}: expected 12 channels, found {C}")
    if T != EXPECTED_POINTS_PER_EPOCH:
        raise ValueError(
            f"{base}: expected {EXPECTED_POINTS_PER_EPOCH} points/epoch, found {T}"
        )

    data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, C)
    data = data[:, :C]

    if data.shape[0] % T != 0:
        raise ValueError(
            f"{base}: {data.shape[0]} rows cannot form complete {T}-sample epochs"
        )

    actual_epochs = data.shape[0] // T
    if actual_epochs != meta_epochs:
        warnings.warn(
            f"{base}: metadata={meta_epochs} epochs, CSV={actual_epochs}; using CSV count"
        )
    if actual_epochs != 4:
        print(f"NOTICE: {base} has {actual_epochs} valid epochs")

    subject = base.split("_")[0]
    label_id, label_name = get_label(base, language)

    epochs = np.zeros((actual_epochs, T, C), dtype=np.float32)
    for ch in range(C):
        epochs[:, :, ch] = data[:, ch].reshape(T, actual_epochs).T

    return {
        "subject": subject,
        "file_id": base,
        "label_id": label_id,
        "label_name": label_name,
        "epochs": epochs,
        "fs": fs,
        "epoch_count": actual_epochs,
    }


def load_dataset(data_dir, language):
    csv_files = sorted(glob(os.path.join(data_dir, "*_pp.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No *_pp.csv files found in {data_dir}")

    items, skipped = [], []
    for path in csv_files:
        try:
            items.append(load_file(path, data_dir, language))
        except Exception as e:
            print(f"WARNING: skipping {os.path.basename(path)} | {e}")
            skipped.append(
                {
                    "Language": language,
                    "File": os.path.basename(path),
                    "Reason": str(e),
                }
            )

    if skipped:
        pd.DataFrame(skipped).to_csv(
            os.path.join(OUT_DIR, f"{language}_skipped_files.csv"),
            index=False,
        )
    if not items:
        raise RuntimeError(f"No usable {language} data")
    return items


# ============================================================
# WINDOWING
# ============================================================
def windowize(epoch_TxC, fs):
    win = int(WIN_SEC * fs)
    step = int(win * (1.0 - OVERLAP))
    T, _ = epoch_TxC.shape
    windows = []

    for start in range(0, T - win + 1, step):
        x = epoch_TxC[start : start + win, :].copy()
        if DO_Z:
            x = (x - x.mean(axis=0, keepdims=True)) / (
                x.std(axis=0, keepdims=True) + 1e-6
            )
        windows.append(x.T)

    if len(windows) != 6:
        warnings.warn(f"Expected 6 windows, got {len(windows)}")
    return windows


def append_item_epochs(item, epoch_ids, X, y, meta, role):
    for epoch_id in epoch_ids:
        if epoch_id < 0 or epoch_id >= item["epoch_count"]:
            continue
        windows = windowize(item["epochs"][epoch_id], item["fs"])
        for window_id, window in enumerate(windows):
            X.append(window)
            y.append(item["label_id"])
            meta.append(
                {
                    "role": role,
                    "subject": item["subject"],
                    "file_id": item["file_id"],
                    "epoch_id": epoch_id,
                    "window_id": window_id,
                    "label_id": item["label_id"],
                    "label_name": item["label_name"],
                }
            )


# ============================================================
# CALIBRATION SPLIT
# ============================================================
def n_rotations_for_subject(test_items):
    counts = [x["epoch_count"] for x in test_items if x["epoch_count"] > 1]
    if not counts:
        return 1
    if not USE_ROTATED_CALIBRATION:
        return 1
    n = 1
    for c in counts:
        n = math.lcm(n, c)
    return n


def choose_calibration_epochs(item, rotation):
    E = item["epoch_count"]
    if E <= CAL_EPOCHS_PER_DIRECTION:
        # Cannot leave a test epoch if all are calibrated.
        return [], list(range(E))

    # CAL=1 for this paper. The function is kept explicit for clarity.
    cal_idx = rotation % E
    cal_epochs = [cal_idx]
    test_epochs = [e for e in range(E) if e not in cal_epochs]
    return cal_epochs, test_epochs


def build_calibrated_fold(all_items_by_subject, test_subject, rotation):
    X_train, y_train, meta_train = [], [], []
    X_test, y_test, meta_test = [], [], []

    # All epochs from all other subjects are source training data.
    for subject, subject_items in all_items_by_subject.items():
        if subject == test_subject:
            continue
        for item in subject_items:
            append_item_epochs(
                item,
                list(range(item["epoch_count"])),
                X_train,
                y_train,
                meta_train,
                role="source_train",
            )

    # One epoch per AVAILABLE direction from the held-out subject is calibration.
    # Remaining epochs from the same direction are final test data.
    for item in all_items_by_subject[test_subject]:
        cal_epochs, test_epochs = choose_calibration_epochs(item, rotation)
        append_item_epochs(
            item, cal_epochs, X_train, y_train, meta_train, role="calibration"
        )
        append_item_epochs(
            item, test_epochs, X_test, y_test, meta_test, role="test"
        )

    if len(X_train) == 0 or len(X_test) == 0:
        return None

    return (
        np.asarray(X_train, dtype=np.float64),
        np.asarray(y_train, dtype=int),
        pd.DataFrame(meta_train),
        np.asarray(X_test, dtype=np.float64),
        np.asarray(y_test, dtype=int),
        pd.DataFrame(meta_test),
    )


# ============================================================
# 78-D OAS + RIEMANN + RBF-SVM
# ============================================================
def extract_features(X_train_raw, X_test_raw):
    cov = Covariances(estimator="oas")
    ts = TangentSpace(metric="riemann")
    scaler = StandardScaler()

    X_train_cov = cov.fit_transform(X_train_raw)
    X_test_cov = cov.transform(X_test_raw)

    X_train_ts = ts.fit_transform(X_train_cov)
    X_test_ts = ts.transform(X_test_cov)

    if X_train_ts.shape[1] != 78:
        raise RuntimeError(
            f"Expected 78 tangent-space features, got {X_train_ts.shape[1]}"
        )

    X_train = scaler.fit_transform(X_train_ts)
    X_test = scaler.transform(X_test_ts)
    return X_train, X_test


def make_model():
    return SVC(
        kernel="rbf",
        C=SVM_C,
        gamma=SVM_GAMMA,
        decision_function_shape="ovr",
    )


# ============================================================
# EPOCH VOTING / METRICS
# ============================================================
def majority_vote(labels):
    counts = Counter(int(x) for x in labels)
    max_count = max(counts.values())
    winners = [cls for cls, n in counts.items() if n == max_count]
    return min(winners)


def epoch_predictions(meta_test, window_pred):
    df = meta_test.copy()
    df["window_pred"] = window_pred
    rows = []
    for (subject, file_id, epoch_id), g in df.groupby(
        ["subject", "file_id", "epoch_id"], sort=True
    ):
        rows.append(
            {
                "subject": subject,
                "file_id": file_id,
                "epoch_id": int(epoch_id),
                "true_label": int(g["label_id"].iloc[0]),
                "pred_label": majority_vote(g["window_pred"].values),
            }
        )
    return pd.DataFrame(rows)


def calculate_metrics(y_true, y_pred):
    labels = sorted(np.unique(y_true))
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        ),
        "F1_Score": f1_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        ),
        "Macro_F1": f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
    }


def row_normalized_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_IDS).astype(float)
    out = np.full_like(cm, np.nan, dtype=float)
    row_sum = cm.sum(axis=1)
    for i in range(len(CLASS_IDS)):
        if row_sum[i] > 0:
            out[i, :] = 100.0 * cm[i, :] / row_sum[i]
    return out


def class_precision_percent(y_true, y_pred):
    out = np.full(len(CLASS_IDS), np.nan, dtype=float)
    for i, cls in enumerate(CLASS_IDS):
        if np.sum(np.asarray(y_true) == cls) == 0:
            continue
        out[i] = 100.0 * precision_score(
            y_true,
            y_pred,
            labels=[cls],
            average="macro",
            zero_division=0,
        )
    return out


# ============================================================
# CALIBRATED LOSO
# ============================================================
def run_language(items, language):
    by_subject = defaultdict(list)
    for item in items:
        by_subject[item["subject"]].append(item)
    subjects = sorted(by_subject.keys())

    prefix = "E" if language == "English" else "B"
    sid = {s: f"{prefix}{i:02d}" for i, s in enumerate(subjects, start=1)}

    availability_rows = []
    for s in subjects:
        dirs = sorted({x["label_name"] for x in by_subject[s]})
        missing = sorted(set(CLASS_NAMES.values()) - set(dirs))
        availability_rows.append(
            {
                "Language": language,
                "Subject_internal": s,
                "Subject_ID": sid[s],
                "N_Directions": len(dirs),
                "N_Epochs": sum(x["epoch_count"] for x in by_subject[s]),
                "Missing_Directions": ", ".join(missing),
            }
        )
    pd.DataFrame(availability_rows).to_csv(
        os.path.join(OUT_DIR, f"{language}_CAL1_data_availability.csv"),
        index=False,
    )

    subject_rows = []
    all_rotation_rows = []
    all_epoch_rows = []
    subject_main_cms = []
    subject_precision_vectors = []

    print("\n" + "=" * 100)
    print(f"{language.upper()} CALIBRATED LOSO (CAL=1)")
    print("=" * 100)

    for fold_no, test_subject in enumerate(subjects, start=1):
        test_items = by_subject[test_subject]
        n_rot = n_rotations_for_subject(test_items)
        print(
            f"\nSubject {sid[test_subject]} | fold {fold_no}/{len(subjects)} | rotations={n_rot}"
        )

        rotation_metrics = []
        subject_epoch_predictions = []

        for rotation in range(n_rot):
            fold = build_calibrated_fold(by_subject, test_subject, rotation)
            if fold is None:
                continue

            (
                X_train_raw,
                y_train,
                meta_train,
                X_test_raw,
                y_test_windows,
                meta_test,
            ) = fold

            if len(np.unique(y_train)) < 2 or len(X_test_raw) == 0:
                continue

            X_train, X_test = extract_features(X_train_raw, X_test_raw)
            model = make_model()
            model.fit(X_train, y_train)
            window_pred = model.predict(X_test)
            ep = epoch_predictions(meta_test, window_pred)
            if ep.empty:
                continue

            y_true = ep["true_label"].astype(int).values
            y_pred = ep["pred_label"].astype(int).values
            m = calculate_metrics(y_true, y_pred)
            m["Rotation"] = rotation + 1
            m["N_Test_Epochs"] = len(ep)
            rotation_metrics.append(m)

            ep["Language"] = language
            ep["Subject_internal"] = test_subject
            ep["Subject_ID"] = sid[test_subject]
            ep["Rotation"] = rotation + 1
            subject_epoch_predictions.append(ep)
            all_epoch_rows.extend(ep.to_dict(orient="records"))

            all_rotation_rows.append(
                {
                    "Language": language,
                    "Subject_ID": sid[test_subject],
                    "Rotation": rotation + 1,
                    "N_Test_Epochs": len(ep),
                    "Accuracy": m["Accuracy"],
                    "Precision": m["Precision"],
                    "Recall": m["Recall"],
                    "F1_Score": m["F1_Score"],
                    "Macro_F1": m["Macro_F1"],
                }
            )

        if not rotation_metrics:
            print("  No valid calibrated LOSO rotation for this subject")
            continue

        rm = pd.DataFrame(rotation_metrics)
        dirs = sorted({x["label_name"] for x in test_items})
        missing = sorted(set(CLASS_NAMES.values()) - set(dirs))

        subject_rows.append(
            {
                "Language": language,
                "Subject_ID": sid[test_subject],
                "N_Directions": len(dirs),
                "Missing_Directions": ", ".join(missing),
                "N_Rotations": len(rm),
                "Mean_Test_Epochs_per_Rotation": rm["N_Test_Epochs"].mean(),
                "Accuracy": rm["Accuracy"].mean(),
                "Precision": rm["Precision"].mean(),
                "Recall": rm["Recall"].mean(),
                "F1_Score": rm["F1_Score"].mean(),
                "Macro_F1": rm["Macro_F1"].mean(),
                "Accuracy_SD_across_rotations": rm["Accuracy"].std(ddof=1)
                if len(rm) > 1
                else 0.0,
            }
        )

        # Equal weight per held-out subject in the paper confusion matrix.
        subj_all = pd.concat(subject_epoch_predictions, ignore_index=True)
        y_true_sub = subj_all["true_label"].astype(int).values
        y_pred_sub = subj_all["pred_label"].astype(int).values
        subject_main_cms.append(row_normalized_cm(y_true_sub, y_pred_sub))
        subject_precision_vectors.append(
            class_precision_percent(y_true_sub, y_pred_sub)
        )

        print(
            f"  Mean accuracy={100*rm['Accuracy'].mean():.2f}% | "
            f"Macro-F1={100*rm['Macro_F1'].mean():.2f}%"
        )

    subject_df = pd.DataFrame(subject_rows)
    rotation_df = pd.DataFrame(all_rotation_rows)
    epoch_df = pd.DataFrame(all_epoch_rows)

    if subject_df.empty:
        raise RuntimeError(f"No valid {language} calibrated LOSO results")

    # Main paper statistic: mean +/- SD across held-out subjects, like Rakib.
    n = len(subject_df)
    mean_acc = subject_df["Accuracy"].mean()
    sd_acc = subject_df["Accuracy"].std(ddof=1)
    sem_acc = sd_acc / np.sqrt(n) if n > 1 else np.nan
    crit = t.ppf(0.975, df=n - 1) if n > 1 else np.nan
    ci_low = mean_acc - crit * sem_acc if n > 1 else np.nan
    ci_high = mean_acc + crit * sem_acc if n > 1 else np.nan

    summary = pd.DataFrame(
        [
            {
                "Dataset": language,
                "Calibration": "CAL=1 epoch per available direction",
                "N_Subjects": n,
                "Mean_Subject_Accuracy": mean_acc,
                "SD_Subject_Accuracy": sd_acc,
                "CI95_Low": ci_low,
                "CI95_High": ci_high,
                "Mean_Subject_Precision": subject_df["Precision"].mean(),
                "Mean_Subject_Recall": subject_df["Recall"].mean(),
                "Mean_Subject_F1": subject_df["F1_Score"].mean(),
                "Mean_Subject_Macro_F1": subject_df["Macro_F1"].mean(),
            }
        ]
    )

    # Subject-balanced confusion matrix for the Rakib-style figure.
    cm_stack = np.stack(subject_main_cms, axis=0)
    precision_stack = np.stack(subject_precision_vectors, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main_cm = np.nanmean(cm_stack, axis=0)
        precision_vec = np.nanmean(precision_stack, axis=0)

    recall_vec = np.diag(main_cm)
    error_recall = 100.0 - recall_vec
    error_precision = 100.0 - precision_vec

    # Save raw and publication tables.
    subject_df.to_csv(
        os.path.join(OUT_DIR, f"{language}_CAL1_LOSO_subjectwise_raw.csv"),
        index=False,
    )
    rotation_df.to_csv(
        os.path.join(OUT_DIR, f"{language}_CAL1_LOSO_rotationwise.csv"),
        index=False,
    )
    epoch_df.to_csv(
        os.path.join(OUT_DIR, f"{language}_CAL1_LOSO_epoch_predictions.csv"),
        index=False,
    )
    summary.to_csv(
        os.path.join(OUT_DIR, f"{language}_CAL1_LOSO_summary_raw.csv"),
        index=False,
    )

    pub_subject = subject_df[
        [
            "Subject_ID",
            "Accuracy",
            "Precision",
            "Recall",
            "F1_Score",
            "Macro_F1",
        ]
    ].copy()
    for c in ["Accuracy", "Precision", "Recall", "F1_Score", "Macro_F1"]:
        pub_subject[c] = (100.0 * pub_subject[c]).round(2)
    pub_subject.to_csv(
        os.path.join(OUT_DIR, f"{language}_CAL1_LOSO_subjectwise_PUBLICATION.csv"),
        index=False,
    )

    pub_summary = summary.copy()
    for c in [
        "Mean_Subject_Accuracy",
        "SD_Subject_Accuracy",
        "CI95_Low",
        "CI95_High",
        "Mean_Subject_Precision",
        "Mean_Subject_Recall",
        "Mean_Subject_F1",
        "Mean_Subject_Macro_F1",
    ]:
        pub_summary[c] = (100.0 * pub_summary[c]).round(2)
    pub_summary.to_csv(
        os.path.join(OUT_DIR, f"{language}_CAL1_LOSO_summary_PUBLICATION.csv"),
        index=False,
    )

    pd.DataFrame(main_cm, index=[CLASS_NAMES[x] for x in CLASS_IDS],
                 columns=[CLASS_NAMES[x] for x in CLASS_IDS]).to_csv(
        os.path.join(OUT_DIR, f"{language}_CAL1_LOSO_CM_subject_balanced_percent.csv")
    )

    return {
        "language": language,
        "subject_df": subject_df,
        "pub_subject": pub_subject,
        "summary": summary,
        "pub_summary": pub_summary,
        "main_cm": main_cm,
        "recall_vec": recall_vec,
        "error_recall": error_recall,
        "precision_vec": precision_vec,
        "error_precision": error_precision,
    }


# ============================================================
# RAKIB-STYLE CONFUSION MATRIX FIGURE
# ============================================================
BLUE = "#2B83C6"
PEACH = "#F4D2BD"
EDGE = "#7A7A7A"
WHITE = "#FFFFFF"
BLACK = "#111111"


def draw_percent_grid(ax, values, diag_blue=False, blue_row=False, fmt="{:.1f}%"):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    nr, nc = values.shape

    ax.set_xlim(0, nc)
    ax.set_ylim(nr, 0)
    ax.set_aspect("equal")

    for i in range(nr):
        for j in range(nc):
            val = values[i, j]
            use_blue = False
            if diag_blue and nr == nc and i == j:
                use_blue = True
            if blue_row and i == 0:
                use_blue = True

            face = BLUE if use_blue else PEACH
            txt_color = WHITE if use_blue else BLACK
            ax.add_patch(
                Rectangle((j, i), 1, 1, facecolor=face, edgecolor=EDGE, linewidth=0.8)
            )
            text = "" if np.isnan(val) else fmt.format(val)
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center",
                    fontsize=8.5, color=txt_color)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_right_summary(ax, recall_vec, error_vec):
    values = np.column_stack([recall_vec, error_vec])
    nr, nc = values.shape
    ax.set_xlim(0, nc)
    ax.set_ylim(nr, 0)
    ax.set_aspect("equal")
    for i in range(nr):
        for j in range(nc):
            face = BLUE if j == 0 else PEACH
            color = WHITE if j == 0 else BLACK
            ax.add_patch(
                Rectangle((j, i), 1, 1, facecolor=face, edgecolor=EDGE, linewidth=0.8)
            )
            v = values[i, j]
            ax.text(j + 0.5, i + 0.5,
                    "" if np.isnan(v) else f"{v:.1f}%",
                    ha="center", va="center", fontsize=8.5, color=color)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_bottom_summary(ax, precision_vec, error_vec):
    values = np.vstack([precision_vec, error_vec])
    nr, nc = values.shape
    ax.set_xlim(0, nc)
    ax.set_ylim(nr, 0)
    ax.set_aspect("equal")
    for i in range(nr):
        for j in range(nc):
            face = BLUE if i == 0 else PEACH
            color = WHITE if i == 0 else BLACK
            ax.add_patch(
                Rectangle((j, i), 1, 1, facecolor=face, edgecolor=EDGE, linewidth=0.8)
            )
            v = values[i, j]
            ax.text(j + 0.5, i + 0.5,
                    "" if np.isnan(v) else f"{v:.1f}%",
                    ha="center", va="center", fontsize=8.5, color=color)
    ax.set_xticks(np.arange(nc) + 0.5)
    ax.set_xticklabels([CLASS_NAMES[x].capitalize() for x in CLASS_IDS],
                       rotation=35, ha="right", fontsize=8)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_main_cm(ax, matrix, language):
    n = len(CLASS_IDS)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")

    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            diag = i == j
            face = BLUE if diag else PEACH
            color = WHITE if diag else BLACK
            ax.add_patch(
                Rectangle((j, i), 1, 1, facecolor=face, edgecolor=EDGE, linewidth=0.8)
            )
            ax.text(j + 0.5, i + 0.5,
                    "" if np.isnan(v) else f"{v:.1f}%",
                    ha="center", va="center", fontsize=8.5, color=color)

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels([CLASS_NAMES[x].capitalize() for x in CLASS_IDS],
                       rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels([CLASS_NAMES[x].capitalize() for x in CLASS_IDS], fontsize=8)
    ax.set_xlabel("Predicted Class", fontsize=9)
    ax.set_ylabel("True Class", fontsize=9)
    ax.set_title(f"{language} Inner-Speech Confusion Matrix — Calibrated LOSO",
                 fontsize=10, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_rakib_style_figure(eng, bn):
    fig = plt.figure(figsize=(14.5, 6.6))
    outer = fig.add_gridspec(1, 2, wspace=0.28)

    for k, result in enumerate([eng, bn]):
        sub = outer[k].subgridspec(
            2, 2,
            width_ratios=[6, 2],
            height_ratios=[6, 2],
            wspace=0.08,
            hspace=0.12,
        )
        ax_main = fig.add_subplot(sub[0, 0])
        ax_right = fig.add_subplot(sub[0, 1])
        ax_bottom = fig.add_subplot(sub[1, 0])
        ax_blank = fig.add_subplot(sub[1, 1])
        ax_blank.axis("off")

        draw_main_cm(ax_main, result["main_cm"], result["language"])
        draw_right_summary(ax_right, result["recall_vec"], result["error_recall"])
        draw_bottom_summary(ax_bottom, result["precision_vec"], result["error_precision"])

    fig.text(0.255, 0.035, "(a)", ha="center", fontsize=11)
    fig.text(0.755, 0.035, "(b)", ha="center", fontsize=11)

    plt.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.18)

    png = os.path.join(OUT_DIR, "CAL1_LOSO_confusion_matrices_Rakib_style.png")
    pdf = os.path.join(OUT_DIR, "CAL1_LOSO_confusion_matrices_Rakib_style.pdf")
    plt.savefig(png, dpi=600, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    return png, pdf


# ============================================================
# SIDE-BY-SIDE SUBJECT TABLE LIKE RAKIB TABLE 4
# ============================================================
def make_side_by_side_subject_table(eng, bn):
    e = eng["pub_subject"].copy().rename(
        columns={
            "Subject_ID": "ENG_Subject",
            "Accuracy": "ENG_Accuracy",
            "Precision": "ENG_Precision",
            "Recall": "ENG_Recall",
            "F1_Score": "ENG_F1",
            "Macro_F1": "ENG_Macro_F1",
        }
    )
    b = bn["pub_subject"].copy().rename(
        columns={
            "Subject_ID": "BN_Subject",
            "Accuracy": "BN_Accuracy",
            "Precision": "BN_Precision",
            "Recall": "BN_Recall",
            "F1_Score": "BN_F1",
            "Macro_F1": "BN_Macro_F1",
        }
    )

    e = e[["ENG_Subject", "ENG_Accuracy", "ENG_Precision", "ENG_Recall", "ENG_F1", "ENG_Macro_F1"]]
    b = b[["BN_Subject", "BN_Accuracy", "BN_Precision", "BN_Recall", "BN_F1", "BN_Macro_F1"]]

    n = max(len(e), len(b))
    e = e.reindex(range(n)).reset_index(drop=True)
    b = b.reindex(range(n)).reset_index(drop=True)
    out = pd.concat([e, b], axis=1)
    out.to_csv(
        os.path.join(OUT_DIR, "CAL1_LOSO_Rakib_style_subject_table.csv"),
        index=False,
    )
    return out


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 100)
    print("FINAL CALIBRATED LOSO")
    print("CAL=1 epoch per available direction + 78-D Riemann + RBF-SVM")
    print("=" * 100)
    print(
        "NOTE: With 4 epochs per direction, CAL=1 uses 25% of that direction for "
        "calibration and 75% for testing. This is the closest epoch-disjoint "
        "analogue to Rakib et al.'s 20% subject calibration."
    )

    eng_items = load_dataset(ENG_DIR, "English")
    bn_items = load_dataset(BN_DIR, "Bangla")

    eng = run_language(eng_items, "English")
    bn = run_language(bn_items, "Bangla")

    combined_summary = pd.concat(
        [eng["pub_summary"], bn["pub_summary"]], ignore_index=True
    )
    combined_summary.to_csv(
        os.path.join(OUT_DIR, "CAL1_LOSO_overall_summary_PUBLICATION.csv"),
        index=False,
    )

    side = make_side_by_side_subject_table(eng, bn)
    png, pdf = make_rakib_style_figure(eng, bn)

    print("\n" + "=" * 120)
    print("CALIBRATED LOSO SUMMARY")
    print("=" * 120)
    print(combined_summary.to_string(index=False))

    print("\n" + "=" * 120)
    print("SUBJECT-WISE TABLE")
    print("=" * 120)
    print(side.to_string(index=False))

    print("\nSaved figure:")
    print(png)
    print(pdf)
    print("\nAll outputs saved in:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
