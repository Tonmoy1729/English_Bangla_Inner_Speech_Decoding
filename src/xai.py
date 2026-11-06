# ============================================================
# XAI FOR ENGLISH-BANGLA INNER SPEECH EEG
#
# Full workflow:
#
# 1. Load English + Bangla preprocessed EEG
# 2. Reconstruct 7-s epochs
# 3. 2-s windows, 50% overlap
# 4. Window-wise channel z-score
# 5. OAS covariance
# 6. Train-only Riemannian tangent-space mapping
# 7. Train-only StandardScaler
# 8. RBF-SVM, C=10
# 9. Epoch-level majority voting
#
# XAI:
# A. Leave-one-channel-out ablation
# B. Leave-one-region-out ablation
# C. Tangent-feature permutation importance
# D. SHAP on tangent-space RBF-SVM
# E. English vs Bangla channel comparison
# F. Paper-ready PNG + PDF figures
# G. CSV files for future tables
#
# IMPORTANT:
# This script uses all 78 tangent features from 12 channels.
# It DOES NOT invent a 52-feature selection step.
# ============================================================


import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

import mne
import shap


warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(
    r"C:\Users\USER\Documents\Gobeshona\Thesis"
    r"\Thesis data collection"
)


ENG_DIR = ROOT / "English preprocessed"

BN_DIR = ROOT / "Bangla preprocessed"


OUT_DIR = Path(
    r"C:\Users\USER\Documents\Gobeshona\Thesis"
    r"\Thesis Revision Work"
    r"\XAI_RIEMANN_PAPER"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. EEG SETTINGS
# ============================================================

FS = 256

CHANNELS = [
    "F3",
    "Fz",
    "F4",
    "FC3",
    "FC4",
    "C3",
    "Cz",
    "C4",
    "CP5",
    "P3",
    "P4",
    "PO7"
]

N_CHANNELS = len(CHANNELS)


# 7 seconds x 256 Hz
EPOCH_SAMPLES = 1792


# 2-second window
WINDOW_SAMPLES = 512


# 50% overlap
STEP_SAMPLES = 256


WINDOW_STARTS = list(
    range(
        0,
        EPOCH_SAMPLES - WINDOW_SAMPLES + 1,
        STEP_SAMPLES
    )
)

N_WINDOWS_PER_EPOCH = len(
    WINDOW_STARTS
)


# ============================================================
# 3. MODEL SETTINGS
# ============================================================

SVM_C = 10

SVM_GAMMA = "scale"

SEED = 42


# ============================================================
# 4. XAI SETTINGS
# ============================================================

# Permutation repeats.
# 5 is enough for initial paper analysis.
PERMUTATION_REPEATS = 5


# SHAP is substantially slower than ablation/permutation.

RUN_SHAP = True


# Number of training examples used as SHAP background
SHAP_BACKGROUND = 20


# Number of held-out windows explained in each fold
SHAP_TEST_SAMPLES_PER_FOLD = 15


# Approximate KernelSHAP evaluations
SHAP_NSAMPLES = 100


# Number of top tangent features in figures
TOP_FEATURES_PLOT = 15


# ============================================================
# 5. REGIONAL GROUPING
#
# Based on your actual 12-channel montage.
# ============================================================

REGIONS = {

    "Frontal": [
        "F3",
        "Fz",
        "F4"
    ],

    "Fronto-central": [
        "FC3",
        "FC4"
    ],

    "Central": [
        "C3",
        "Cz",
        "C4"
    ],

    "Parietal": [
        "CP5",
        "P3",
        "P4"
    ],

    "Parieto-occipital": [
        "PO7"
    ]
}


# ============================================================
# 6. ALIGNED ENGLISH-BANGLA LABELS
# ============================================================

COMMANDS = [
    "back",
    "down",
    "forward",
    "left",
    "right",
    "up"
]


COMMAND_TO_ID = {
    name: i
    for i, name in enumerate(COMMANDS)
}


TOKEN_TO_COMMAND = {

    # English
    "back": "back",
    "down": "down",
    "forward": "forward",
    "left": "left",
    "right": "right",
    "up": "up",

    # Bangla transliterations
    "pichone": "back",
    "niche": "down",
    "samne": "forward",
    "bame": "left",
    "dane": "right",
    "upore": "up"
}


# ============================================================
# 7. PLOT STYLE
# ============================================================

plt.rcParams.update({

    "font.family": "serif",

    "font.size": 10,

    "axes.titlesize": 11,

    "axes.labelsize": 10,

    "xtick.labelsize": 9,

    "ytick.labelsize": 9,

    "legend.fontsize": 9,

    "figure.dpi": 150
})


# ============================================================
# HELPERS
# ============================================================

def save_figure(
    fig,
    name
):

    png = OUT_DIR / f"{name}.png"

    pdf = OUT_DIR / f"{name}.pdf"


    fig.savefig(
        png,
        dpi=600,
        bbox_inches="tight"
    )


    fig.savefig(
        pdf,
        bbox_inches="tight"
    )


    plt.close(fig)


# ============================================================
# FIND DATASET FOLDER IF NAME DIFFERS SLIGHTLY
# ============================================================

def resolve_directory(
    preferred,
    keyword
):

    if preferred.exists():

        return preferred


    candidates = [
        p
        for p in ROOT.iterdir()
        if (
            p.is_dir()
            and
            keyword.lower()
            in p.name.lower()
        )
    ]


    if len(candidates) == 1:

        print(
            f"Using detected {keyword} folder:",
            candidates[0]
        )

        return candidates[0]


    raise FileNotFoundError(
        f"\nCannot find {keyword} dataset folder.\n"
        f"Expected:\n{preferred}\n"
        f"Please change ENG_DIR / BN_DIR at top."
    )


# ============================================================
# LABEL FROM FILE NAME
# ============================================================

def detect_command(
    filepath
):

    stem = filepath.stem.lower()


    tokens = re.split(
        r"[_\-\s]+",
        stem
    )


    # Work backward because filenames may contain
    # subject_name_command_1_pp.csv

    for token in reversed(tokens):

        if token in TOKEN_TO_COMMAND:

            return TOKEN_TO_COMMAND[
                token
            ]


    raise ValueError(
        f"Could not determine command from file:\n"
        f"{filepath.name}"
    )


# ============================================================
# SUBJECT NAME FROM FILE NAME
# ============================================================

def detect_subject(
    filepath
):

    stem = filepath.stem.lower()


    tokens = re.split(
        r"[_\-\s]+",
        stem
    )


    for i, token in enumerate(tokens):

        if token in TOKEN_TO_COMMAND:

            if i == 0:

                return "unknown"


            return "_".join(
                tokens[:i]
            )


    return stem


# ============================================================
# READ META FILE
# ============================================================

def find_meta_file(
    csv_path
):

    candidates = [

        csv_path.with_name(
            csv_path.stem
            +
            "_meta.txt"
        ),

        csv_path.with_name(
            csv_path.stem.replace(
                "_pp",
                ""
            )
            +
            "_meta.txt"
        ),

        csv_path.with_name(
            csv_path.stem.replace(
                "_pp",
                "_pp_meta"
            )
            +
            ".txt"
        )
    ]


    for p in candidates:

        if p.exists():

            return p


    return None


def parse_meta(
    csv_path,
    n_rows
):

    meta = find_meta_file(
        csv_path
    )


    n_epochs = None

    pnts = EPOCH_SAMPLES


    if meta is not None:

        text = meta.read_text(
            errors="ignore"
        )


        match_epochs = re.search(
            r"(?:epochs|n_epochs)\s*=\s*(\d+)",
            text,
            flags=re.I
        )


        match_pnts = re.search(
            r"(?:pnts_per_epoch|points_per_epoch|pnts)"
            r"\s*=\s*(\d+)",
            text,
            flags=re.I
        )


        if match_epochs:

            n_epochs = int(
                match_epochs.group(1)
            )


        if match_pnts:

            pnts = int(
                match_pnts.group(1)
            )


    # If meta missing or incomplete
    if n_epochs is None:

        n_epochs = (
            n_rows
            //
            pnts
        )


    max_possible = (
        n_rows
        //
        pnts
    )


    n_epochs = min(
        n_epochs,
        max_possible
    )


    return n_epochs, pnts


# ============================================================
# LOAD ONE LANGUAGE
# ============================================================

def load_language(
    language,
    folder
):

    files = sorted(
        folder.rglob(
            "*_pp.csv"
        )
    )


    if len(files) == 0:

        raise RuntimeError(
            f"No *_pp.csv files found in:\n"
            f"{folder}"
        )


    epoch_data = []

    epoch_labels = []

    epoch_positions = []

    epoch_files = []

    epoch_subjects = []


    print(
        f"\nLoading {language}: "
        f"{len(files)} PP files"
    )


    for file_no, file in enumerate(
        files,
        start=1
    ):

        command = detect_command(
            file
        )


        label = COMMAND_TO_ID[
            command
        ]


        subject = detect_subject(
            file
        )


        raw = pd.read_csv(
            file,
            header=None
        ).iloc[
            :,
            :N_CHANNELS
        ].to_numpy(
            dtype=np.float32
        )


        n_epochs, pnts = parse_meta(
            file,
            len(raw)
        )


        if pnts != EPOCH_SAMPLES:

            raise RuntimeError(
                f"\nUnexpected epoch length in "
                f"{file.name}: {pnts}\n"
                f"Expected {EPOCH_SAMPLES}."
            )


        if n_epochs < 1:

            print(
                "Skipping empty file:",
                file.name
            )

            continue


        usable = (
            n_epochs
            *
            EPOCH_SAMPLES
        )


        raw = raw[
            :usable
        ]


        for e in range(
            n_epochs
        ):

            segment = raw[
                e * EPOCH_SAMPLES:
                (e + 1) * EPOCH_SAMPLES
            ]


            # time x channels
            # ->
            # channels x time

            segment = segment.T


            epoch_data.append(
                segment
            )


            epoch_labels.append(
                label
            )


            # critical for epoch-heldout CV
            epoch_positions.append(
                e
            )


            epoch_files.append(
                file.name
            )


            epoch_subjects.append(
                subject
            )


        if (
            file_no % 30 == 0
            or
            file_no == len(files)
        ):

            print(
                f"  {file_no}/"
                f"{len(files)} files"
            )


    epochs = np.stack(
        epoch_data
    ).astype(
        np.float32
    )


    labels = np.asarray(
        epoch_labels,
        dtype=int
    )


    positions = np.asarray(
        epoch_positions,
        dtype=int
    )


    print(
        f"{language} epochs:",
        epochs.shape
    )


    print(
        f"{language} class counts:",
        {
            COMMANDS[c]:
                int(
                    np.sum(
                        labels == c
                    )
                )
            for c in range(6)
        }
    )


    return {

        "language":
            language,

        "epochs":
            epochs,

        "epoch_labels":
            labels,

        "epoch_positions":
            positions,

        "epoch_files":
            np.asarray(
                epoch_files
            ),

        "epoch_subjects":
            np.asarray(
                epoch_subjects
            )
    }


# ============================================================
# CREATE 2-S WINDOWS WITH 50% OVERLAP
# ============================================================

def create_windows(
    data
):

    epochs = data[
        "epochs"
    ]


    n_epochs = len(
        epochs
    )


    windows = []

    window_epoch_ids = []

    window_labels = []


    for epoch_id in range(
        n_epochs
    ):

        epoch = epochs[
            epoch_id
        ]


        for start in WINDOW_STARTS:

            stop = (
                start
                +
                WINDOW_SAMPLES
            )


            w = epoch[
                :,
                start:stop
            ].copy()


            # ================================================
            # Per-window, per-channel z-score
            # ================================================

            mean = w.mean(
                axis=1,
                keepdims=True
            )


            std = w.std(
                axis=1,
                keepdims=True
            )


            w = (
                w
                -
                mean
            ) / (
                std
                +
                1e-8
            )


            windows.append(
                w
            )


            window_epoch_ids.append(
                epoch_id
            )


            window_labels.append(
                data[
                    "epoch_labels"
                ][
                    epoch_id
                ]
            )


    data[
        "windows"
    ] = np.stack(
        windows
    ).astype(
        np.float32
    )


    data[
        "window_epoch_ids"
    ] = np.asarray(
        window_epoch_ids,
        dtype=int
    )


    data[
        "window_labels"
    ] = np.asarray(
        window_labels,
        dtype=int
    )


    print(
        f"{data['language']} windows:",
        data[
            "windows"
        ].shape
    )


# ============================================================
# TANGENT FEATURE NAMES
#
# pyRiemann symmetric upper triangle:
#
# F3-F3, F3-Fz, F3-F4, ...
#
# For 12 channels = 78 features.
# ============================================================

def tangent_feature_names(
    channel_names
):

    names = []

    pairs = []


    for i in range(
        len(channel_names)
    ):

        for j in range(
            i,
            len(channel_names)
        ):

            if i == j:

                name = (
                    f"{channel_names[i]}"
                    f"_variance"
                )

            else:

                name = (
                    f"{channel_names[i]}"
                    f"-"
                    f"{channel_names[j]}"
                )


            names.append(
                name
            )


            pairs.append(
                (
                    channel_names[i],
                    channel_names[j]
                )
            )


    return names, pairs


# ============================================================
# EPOCH MAJORITY VOTE
# ============================================================

def epoch_vote(
    predictions,
    epoch_ids,
    epoch_labels
):

    unique_epochs = np.unique(
        epoch_ids
    )


    y_true = []

    y_pred = []


    for eid in unique_epochs:

        mask = (
            epoch_ids
            ==
            eid
        )


        p = predictions[
            mask
        ]


        counts = np.bincount(
            p.astype(int),
            minlength=6
        )


        final_pred = int(
            np.argmax(
                counts
            )
        )


        y_true.append(
            int(
                epoch_labels[
                    eid
                ]
            )
        )


        y_pred.append(
            final_pred
        )


    return (
        np.asarray(
            y_true
        ),
        np.asarray(
            y_pred
        ),
        unique_epochs
    )


# ============================================================
# EVALUATE ANY CHANNEL SUBSET
#
# This is used for:
#
# - baseline
# - channel ablation
# - region ablation
# ============================================================

def evaluate_channel_subset(
    data,
    keep_channel_indices,
    collect_fold_objects=False
):

    windows = data[
        "windows"
    ][
        :,
        keep_channel_indices,
        :
    ]


    epoch_labels = data[
        "epoch_labels"
    ]


    epoch_positions = data[
        "epoch_positions"
    ]


    window_epoch_ids = data[
        "window_epoch_ids"
    ]


    window_labels = data[
        "window_labels"
    ]


    # ================================================
    # OAS is window-local, so safe to calculate here.
    # ================================================

    covs = Covariances(
        estimator="oas"
    ).transform(
        windows
    )


    fold_results = []

    fold_objects = []


    all_true = []

    all_pred = []


    max_epoch_position = int(
        np.max(
            epoch_positions
        )
    )


    subset_channel_names = [
        CHANNELS[i]
        for i in keep_channel_indices
    ]


    feature_names, feature_pairs = (
        tangent_feature_names(
            subset_channel_names
        )
    )


    # ================================================
    # Epoch-position heldout evaluation
    #
    # Fold 0:
    # test epoch #1 from each recording
    #
    # Fold 1:
    # test epoch #2
    #
    # ...
    #
    # One Bangla recording may have only 3 epochs.
    # That is handled automatically.
    # ================================================

    for fold in range(
        max_epoch_position + 1
    ):

        test_epoch_ids = np.where(
            epoch_positions
            ==
            fold
        )[0]


        train_epoch_ids = np.where(
            epoch_positions
            !=
            fold
        )[0]


        if len(test_epoch_ids) == 0:

            continue


        train_window_mask = np.isin(
            window_epoch_ids,
            train_epoch_ids
        )


        test_window_mask = np.isin(
            window_epoch_ids,
            test_epoch_ids
        )


        train_cov = covs[
            train_window_mask
        ]


        test_cov = covs[
            test_window_mask
        ]


        y_train = window_labels[
            train_window_mask
        ]


        y_test_window = window_labels[
            test_window_mask
        ]


        test_window_epoch_ids = (
            window_epoch_ids[
                test_window_mask
            ]
        )


        # ================================================
        # Train-only tangent-space mapping
        # ================================================

        ts = TangentSpace(
            metric="riemann"
        )


        X_train = ts.fit_transform(
            train_cov
        )


        X_test = ts.transform(
            test_cov
        )


        # ================================================
        # Train-only standardization
        # ================================================

        scaler = StandardScaler()


        X_train = scaler.fit_transform(
            X_train
        )


        X_test = scaler.transform(
            X_test
        )


        # ================================================
        # Final RBF-SVM
        # ================================================

        model = SVC(
            kernel="rbf",
            C=SVM_C,
            gamma=SVM_GAMMA,
            decision_function_shape="ovr"
        )


        model.fit(
            X_train,
            y_train
        )


        pred_window = model.predict(
            X_test
        )


        y_true_epoch, y_pred_epoch, _ = (
            epoch_vote(
                pred_window,
                test_window_epoch_ids,
                epoch_labels
            )
        )


        fold_accuracy = accuracy_score(
            y_true_epoch,
            y_pred_epoch
        )


        fold_results.append(
            {

                "fold":
                    fold + 1,

                "accuracy":
                    fold_accuracy,

                "n_test_epochs":
                    len(
                        y_true_epoch
                    ),

                "n_test_windows":
                    len(
                        pred_window
                    ),

                "n_tangent_features":
                    X_train.shape[1]
            }
        )


        all_true.extend(
            y_true_epoch
        )


        all_pred.extend(
            y_pred_epoch
        )


        if collect_fold_objects:

            fold_objects.append(
                {

                    "fold":
                        fold + 1,

                    "model":
                        model,

                    "ts":
                        ts,

                    "scaler":
                        scaler,

                    "X_train":
                        X_train,

                    "X_test":
                        X_test,

                    "y_train":
                        y_train,

                    "y_test_window":
                        y_test_window,

                    "test_window_epoch_ids":
                        test_window_epoch_ids,

                    "epoch_labels":
                        epoch_labels,

                    "feature_names":
                        feature_names,

                    "feature_pairs":
                        feature_pairs,

                    "classes":
                        model.classes_
                }
            )


    all_true = np.asarray(
        all_true
    )


    all_pred = np.asarray(
        all_pred
    )


    pooled_accuracy = accuracy_score(
        all_true,
        all_pred
    )


    fold_acc = np.asarray(
        [
            x["accuracy"]
            for x in fold_results
        ]
    )


    return {

        "accuracy":
            pooled_accuracy,

        "fold_mean":
            float(
                np.mean(
                    fold_acc
                )
            ),

        "fold_sd":
            float(
                np.std(
                    fold_acc,
                    ddof=1
                )
            )
            if len(fold_acc) > 1
            else 0.0,

        "fold_results":
            fold_results,

        "fold_objects":
            fold_objects,

        "n_features":
            len(
                feature_names
            )
    }


# ============================================================
# CHANNEL ABLATION
# ============================================================

def channel_ablation(
    data,
    baseline_accuracy
):

    rows = []


    print(
        f"\n{data['language']} "
        f"channel ablation"
    )


    for channel_index, channel in enumerate(
        CHANNELS
    ):

        keep = [
            i
            for i in range(
                N_CHANNELS
            )
            if i != channel_index
        ]


        result = evaluate_channel_subset(
            data,
            keep,
            collect_fold_objects=False
        )


        drop_pp = (
            baseline_accuracy
            -
            result[
                "accuracy"
            ]
        ) * 100


        rows.append(
            {

                "language":
                    data[
                        "language"
                    ],

                "channel":
                    channel,

                "baseline_accuracy":
                    baseline_accuracy,

                "ablated_accuracy":
                    result[
                        "accuracy"
                    ],

                "importance_pp":
                    drop_pp
            }
        )


        print(
            f"  remove {channel:4s}: "
            f"acc="
            f"{result['accuracy'] * 100:6.2f}% "
            f"| drop="
            f"{drop_pp:+6.2f} pp"
        )


    df = pd.DataFrame(
        rows
    )


    df[
        "rank"
    ] = df[
        "importance_pp"
    ].rank(
        ascending=False,
        method="min"
    ).astype(int)


    return df.sort_values(
        "rank"
    )


# ============================================================
# REGION ABLATION
# ============================================================

def region_ablation(
    data,
    baseline_accuracy
):

    rows = []


    print(
        f"\n{data['language']} "
        f"regional ablation"
    )


    for region, region_channels in (
        REGIONS.items()
    ):

        remove_indices = [
            CHANNELS.index(
                ch
            )
            for ch in region_channels
        ]


        keep = [
            i
            for i in range(
                N_CHANNELS
            )
            if i not in remove_indices
        ]


        result = evaluate_channel_subset(
            data,
            keep,
            collect_fold_objects=False
        )


        drop_pp = (
            baseline_accuracy
            -
            result[
                "accuracy"
            ]
        ) * 100


        rows.append(
            {

                "language":
                    data[
                        "language"
                    ],

                "region":
                    region,

                "channels":
                    ", ".join(
                        region_channels
                    ),

                "baseline_accuracy":
                    baseline_accuracy,

                "ablated_accuracy":
                    result[
                        "accuracy"
                    ],

                "importance_pp":
                    drop_pp
            }
        )


        print(
            f"  remove "
            f"{region:18s}: "
            f"acc="
            f"{result['accuracy'] * 100:6.2f}% "
            f"| drop="
            f"{drop_pp:+6.2f} pp"
        )


    df = pd.DataFrame(
        rows
    )


    df[
        "rank"
    ] = df[
        "importance_pp"
    ].rank(
        ascending=False,
        method="min"
    ).astype(int)


    return df.sort_values(
        "rank"
    )


# ============================================================
# TANGENT-FEATURE PERMUTATION IMPORTANCE
#
# Important:
#
# We calculate importance using EPOCH-VOTED accuracy,
# not just window accuracy.
#
# This keeps XAI aligned with the final reported metric.
# ============================================================

def permutation_feature_importance(
    language,
    fold_objects
):

    rng = np.random.default_rng(
        SEED
    )


    all_rows = []


    print(
        f"\n{language} tangent-feature "
        f"permutation importance"
    )


    for artifact in fold_objects:

        model = artifact[
            "model"
        ]


        X_test = artifact[
            "X_test"
        ]


        epoch_ids = artifact[
            "test_window_epoch_ids"
        ]


        epoch_labels = artifact[
            "epoch_labels"
        ]


        feature_names = artifact[
            "feature_names"
        ]


        feature_pairs = artifact[
            "feature_pairs"
        ]


        baseline_pred = model.predict(
            X_test
        )


        y_true, y_pred, _ = epoch_vote(
            baseline_pred,
            epoch_ids,
            epoch_labels
        )


        baseline_acc = accuracy_score(
            y_true,
            y_pred
        )


        for feature_index in range(
            X_test.shape[1]
        ):

            for repeat in range(
                PERMUTATION_REPEATS
            ):

                X_perm = X_test.copy()


                permutation = rng.permutation(
                    len(X_perm)
                )


                X_perm[
                    :,
                    feature_index
                ] = X_perm[
                    permutation,
                    feature_index
                ]


                pred = model.predict(
                    X_perm
                )


                yt, yp, _ = epoch_vote(
                    pred,
                    epoch_ids,
                    epoch_labels
                )


                perm_acc = accuracy_score(
                    yt,
                    yp
                )


                importance = (
                    baseline_acc
                    -
                    perm_acc
                ) * 100


                pair = feature_pairs[
                    feature_index
                ]


                all_rows.append(
                    {

                        "language":
                            language,

                        "fold":
                            artifact[
                                "fold"
                            ],

                        "feature_index":
                            feature_index,

                        "feature_name":
                            feature_names[
                                feature_index
                            ],

                        "channel_1":
                            pair[0],

                        "channel_2":
                            pair[1],

                        "repeat":
                            repeat + 1,

                        "importance_pp":
                            importance
                    }
                )


    raw = pd.DataFrame(
        all_rows
    )


    summary = (
        raw
        .groupby(
            [
                "language",
                "feature_index",
                "feature_name",
                "channel_1",
                "channel_2"
            ],
            as_index=False
        )
        .agg(
            importance_pp_mean=(
                "importance_pp",
                "mean"
            ),

            importance_pp_sd=(
                "importance_pp",
                "std"
            )
        )
    )


    summary[
        "rank"
    ] = summary[
        "importance_pp_mean"
    ].rank(
        ascending=False,
        method="min"
    ).astype(int)


    summary = summary.sort_values(
        "rank"
    )


    return raw, summary


# ============================================================
# SHAP HELPERS
# ============================================================

def normalize_shap_array(
    shap_values,
    n_features
):

    # Older SHAP:
    # list of [n_samples x n_features]
    # one array per class

    if isinstance(
        shap_values,
        list
    ):

        arr = np.stack(
            shap_values,
            axis=-1
        )

        return arr


    arr = np.asarray(
        shap_values
    )


    # n_samples x n_features
    if arr.ndim == 2:

        return arr[
            :,
            :,
            None
        ]


    # Usually:
    # n_samples x n_features x n_outputs

    if (
        arr.ndim == 3
        and
        arr.shape[1]
        ==
        n_features
    ):

        return arr


    # Sometimes:
    # n_samples x n_outputs x n_features

    if (
        arr.ndim == 3
        and
        arr.shape[2]
        ==
        n_features
    ):

        return np.transpose(
            arr,
            (
                0,
                2,
                1
            )
        )


    raise RuntimeError(
        f"Unexpected SHAP shape: "
        f"{arr.shape}"
    )


# ============================================================
# SHAP ANALYSIS
#
# Uses SVM decision function rather than probability.
#
# This avoids probability=True, which is slower and
# unnecessary for feature attribution.
# ============================================================

def shap_feature_importance(
    language,
    fold_objects
):

    rng = np.random.default_rng(
        SEED
    )


    fold_feature_importance = []

    fold_class_importance = []


    reference_names = None

    reference_pairs = None


    print(
        f"\n{language} SHAP analysis"
    )


    for artifact in fold_objects:

        model = artifact[
            "model"
        ]


        X_train = artifact[
            "X_train"
        ]


        X_test = artifact[
            "X_test"
        ]


        feature_names = artifact[
            "feature_names"
        ]


        feature_pairs = artifact[
            "feature_pairs"
        ]


        reference_names = feature_names

        reference_pairs = feature_pairs


        n_bg = min(
            SHAP_BACKGROUND,
            len(X_train)
        )


        n_test = min(
            SHAP_TEST_SAMPLES_PER_FOLD,
            len(X_test)
        )


        bg_idx = rng.choice(
            len(X_train),
            size=n_bg,
            replace=False
        )


        test_idx = rng.choice(
            len(X_test),
            size=n_test,
            replace=False
        )


        background = X_train[
            bg_idx
        ]


        X_explain = X_test[
            test_idx
        ]


        print(
            f"  fold {artifact['fold']}: "
            f"background={n_bg}, "
            f"explained={n_test}"
        )


        explainer = shap.KernelExplainer(
            model.decision_function,
            background
        )


        shap_values = explainer.shap_values(
            X_explain,
            nsamples=SHAP_NSAMPLES
        )


        arr = normalize_shap_array(
            shap_values,
            X_test.shape[1]
        )


        # arr:
        # samples x features x classes


        # Overall feature attribution
        overall = np.mean(
            np.abs(
                arr
            ),
            axis=(
                0,
                2
            )
        )


        # Per-class attribution
        per_class = np.mean(
            np.abs(
                arr
            ),
            axis=0
        )


        fold_feature_importance.append(
            overall
        )


        fold_class_importance.append(
            per_class
        )


    overall_mean = np.mean(
        np.stack(
            fold_feature_importance
        ),
        axis=0
    )


    overall_sd = np.std(
        np.stack(
            fold_feature_importance
        ),
        axis=0,
        ddof=1
    )


    per_class_mean = np.mean(
        np.stack(
            fold_class_importance
        ),
        axis=0
    )


    rows = []


    for i, name in enumerate(
        reference_names
    ):

        pair = reference_pairs[
            i
        ]


        row = {

            "language":
                language,

            "feature_index":
                i,

            "feature_name":
                name,

            "channel_1":
                pair[0],

            "channel_2":
                pair[1],

            "mean_abs_shap":
                overall_mean[
                    i
                ],

            "shap_sd":
                overall_sd[
                    i
                ]
        }


        # Save class-specific SHAP too

        for class_position in range(
            per_class_mean.shape[1]
        ):

            if class_position < len(COMMANDS):

                class_name = COMMANDS[
                    class_position
                ]

            else:

                class_name = (
                    f"class_"
                    f"{class_position}"
                )


            row[
                f"shap_{class_name}"
            ] = per_class_mean[
                i,
                class_position
            ]


        rows.append(
            row
        )


    df = pd.DataFrame(
        rows
    )


    df[
        "rank"
    ] = df[
        "mean_abs_shap"
    ].rank(
        ascending=False,
        method="min"
    ).astype(int)


    return df.sort_values(
        "rank"
    )


# ============================================================
# TOPOMAP
# ============================================================

def topomap(
    values,
    ax,
    title,
    cmap,
    vmin,
    vmax
):

    info = mne.create_info(
        CHANNELS,
        sfreq=FS,
        ch_types="eeg"
    )


    montage = (
        mne.channels
        .make_standard_montage(
            "standard_1020"
        )
    )


    info.set_montage(
        montage,
        match_case=False,
        on_missing="raise"
    )


    try:

        im, _ = mne.viz.plot_topomap(
            values,
            info,
            axes=ax,
            show=False,
            contours=6,
            cmap=cmap,
            vlim=(
                vmin,
                vmax
            )
        )

    except TypeError:

        # compatibility with older MNE

        im, _ = mne.viz.plot_topomap(
            values,
            info,
            axes=ax,
            show=False,
            contours=6,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )


    ax.set_title(
        title
    )


    return im


# ============================================================
# CHANNEL FIGURES
# ============================================================

def make_channel_figures(
    channel_df
):

    pivot = channel_df.pivot(
        index="channel",
        columns="language",
        values="importance_pp"
    ).reindex(
        CHANNELS
    )


    if (
        "English"
        not in pivot.columns
        or
        "Bangla"
        not in pivot.columns
    ):

        return


    eng = pivot[
        "English"
    ].to_numpy()


    bn = pivot[
        "Bangla"
    ].to_numpy()


    diff = (
        bn
        -
        eng
    )


    # ================================================
    # Topomaps
    # ================================================

    positive_max = max(
        np.max(
            np.abs(
                eng
            )
        ),
        np.max(
            np.abs(
                bn
            )
        ),
        1e-6
    )


    diff_max = max(
        np.max(
            np.abs(
                diff
            )
        ),
        1e-6
    )


    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            10.5,
            3.4
        )
    )


    im1 = topomap(
        eng,
        axes[0],
        "(a) English",
        "viridis",
        0,
        positive_max
    )


    im2 = topomap(
        bn,
        axes[1],
        "(b) Bangla",
        "viridis",
        0,
        positive_max
    )


    im3 = topomap(
        diff,
        axes[2],
        "(c) Bangla - English",
        "RdBu_r",
        -diff_max,
        diff_max
    )


    fig.colorbar(
        im1,
        ax=axes[:2],
        shrink=0.72,
        label="Accuracy reduction (percentage points)"
    )


    fig.colorbar(
        im3,
        ax=axes[2],
        shrink=0.72,
        label="Importance difference (percentage points)"
    )


    fig.suptitle(
        "Channel importance from leave-one-channel-out ablation",
        y=1.02
    )


    save_figure(
        fig,
        "Fig_XAI_Channel_Topomaps"
    )


    # ================================================
    # Channel grouped bar plot
    # ================================================

    x = np.arange(
        len(CHANNELS)
    )


    width = 0.38


    fig, ax = plt.subplots(
        figsize=(
            9,
            4.2
        )
    )


    ax.bar(
        x - width / 2,
        eng,
        width,
        label="English"
    )


    ax.bar(
        x + width / 2,
        bn,
        width,
        label="Bangla"
    )


    ax.axhline(
        0,
        linewidth=0.8
    )


    ax.set_xticks(
        x
    )


    ax.set_xticklabels(
        CHANNELS,
        rotation=45,
        ha="right"
    )


    ax.set_ylabel(
        "Accuracy reduction (percentage points)"
    )


    ax.set_xlabel(
        "EEG channel"
    )


    ax.set_title(
        "Channel-level ablation importance"
    )


    ax.legend(
        frameon=False
    )


    fig.tight_layout()


    save_figure(
        fig,
        "Fig_XAI_Channel_Importance_Bar"
    )


    # ================================================
    # English vs Bangla channel correlation
    # ================================================

    rho, p = spearmanr(
        eng,
        bn
    )


    fig, ax = plt.subplots(
        figsize=(
            5.2,
            4.6
        )
    )


    ax.scatter(
        eng,
        bn,
        s=45
    )


    for i, ch in enumerate(
        CHANNELS
    ):

        ax.annotate(
            ch,
            (
                eng[i],
                bn[i]
            ),
            fontsize=8,
            xytext=(
                4,
                3
            ),
            textcoords="offset points"
        )


    ax.set_xlabel(
        "English importance (pp)"
    )


    ax.set_ylabel(
        "Bangla importance (pp)"
    )


    ax.set_title(
        f"English-Bangla channel importance\n"
        f"Spearman rho = {rho:.3f}, p = {p:.4f}"
    )


    fig.tight_layout()


    save_figure(
        fig,
        "Fig_XAI_Channel_Correlation"
    )


    pd.DataFrame(
        [
            {
                "spearman_rho":
                    rho,

                "p_value":
                    p
            }
        ]
    ).to_csv(
        OUT_DIR
        /
        "XAI_Channel_Importance_Correlation.csv",
        index=False
    )


# ============================================================
# REGION FIGURE
# ============================================================

def make_region_figure(
    region_df
):

    region_order = list(
        REGIONS.keys()
    )


    pivot = region_df.pivot(
        index="region",
        columns="language",
        values="importance_pp"
    ).reindex(
        region_order
    )


    x = np.arange(
        len(region_order)
    )


    width = 0.38


    fig, ax = plt.subplots(
        figsize=(
            8.2,
            4.5
        )
    )


    if "English" in pivot.columns:

        ax.bar(
            x - width / 2,
            pivot[
                "English"
            ].to_numpy(),
            width,
            label="English"
        )


    if "Bangla" in pivot.columns:

        ax.bar(
            x + width / 2,
            pivot[
                "Bangla"
            ].to_numpy(),
            width,
            label="Bangla"
        )


    ax.axhline(
        0,
        linewidth=0.8
    )


    ax.set_xticks(
        x
    )


    ax.set_xticklabels(
        region_order,
        rotation=25,
        ha="right"
    )


    ax.set_ylabel(
        "Accuracy reduction (percentage points)"
    )


    ax.set_xlabel(
        "EEG region"
    )


    ax.set_title(
        "Regional importance from leave-one-region-out ablation"
    )


    ax.legend(
        frameon=False
    )


    fig.tight_layout()


    save_figure(
        fig,
        "Fig_XAI_Regional_Importance"
    )


# ============================================================
# FEATURE IMPORTANCE FIGURE
# ============================================================

def make_feature_figure(
    df,
    language,
    value_column,
    title,
    filename,
    xlabel
):

    top = (
        df[
            df[
                "language"
            ]
            ==
            language
        ]
        .sort_values(
            value_column,
            ascending=False
        )
        .head(
            TOP_FEATURES_PLOT
        )
        .sort_values(
            value_column,
            ascending=True
        )
    )


    fig, ax = plt.subplots(
        figsize=(
            7.2,
            5.3
        )
    )


    ax.barh(
        top[
            "feature_name"
        ],
        top[
            value_column
        ]
    )


    ax.set_xlabel(
        xlabel
    )


    ax.set_ylabel(
        "Tangent-space feature"
    )


    ax.set_title(
        title
    )


    fig.tight_layout()


    save_figure(
        fig,
        filename
    )


# ============================================================
# RUN ONE LANGUAGE
# ============================================================

def run_language_xai(
    data
):

    language = data[
        "language"
    ]


    print(
        "\n"
        +
        "=" * 90
    )

    print(
        f"{language.upper()} XAI"
    )

    print(
        "=" * 90
    )


    # ========================================================
    # BASELINE
    # ========================================================

    full_channels = list(
        range(
            N_CHANNELS
        )
    )


    baseline = evaluate_channel_subset(
        data,
        full_channels,
        collect_fold_objects=True
    )


    print(
        f"\n{language} baseline "
        f"epoch accuracy: "
        f"{baseline['accuracy'] * 100:.2f}%"
    )


    print(
        f"{language} fold mean: "
        f"{baseline['fold_mean'] * 100:.2f}% "
        f"± "
        f"{baseline['fold_sd'] * 100:.2f}%"
    )


    print(
        f"Tangent dimension: "
        f"{baseline['n_features']}"
    )


    # ========================================================
    # CHANNEL IMPORTANCE
    # ========================================================

    ch_df = channel_ablation(
        data,
        baseline[
            "accuracy"
        ]
    )


    # ========================================================
    # REGION IMPORTANCE
    # ========================================================

    region_df = region_ablation(
        data,
        baseline[
            "accuracy"
        ]
    )


    # ========================================================
    # FEATURE PERMUTATION IMPORTANCE
    # ========================================================

    perm_raw, perm_summary = (
        permutation_feature_importance(
            language,
            baseline[
                "fold_objects"
            ]
        )
    )


    # ========================================================
    # SHAP
    # ========================================================

    shap_df = None


    if RUN_SHAP:

        shap_df = shap_feature_importance(
            language,
            baseline[
                "fold_objects"
            ]
        )


    # ========================================================
    # BASELINE/FOLD TABLES
    # ========================================================

    baseline_row = pd.DataFrame(
        [
            {

                "language":
                    language,

                "n_epochs":
                    len(
                        data[
                            "epochs"
                        ]
                    ),

                "n_windows":
                    len(
                        data[
                            "windows"
                        ]
                    ),

                "window_seconds":
                    WINDOW_SAMPLES
                    /
                    FS,

                "overlap_percent":
                    100
                    *
                    (
                        1
                        -
                        STEP_SAMPLES
                        /
                        WINDOW_SAMPLES
                    ),

                "n_channels":
                    N_CHANNELS,

                "tangent_dimension":
                    baseline[
                        "n_features"
                    ],

                "pooled_epoch_accuracy":
                    baseline[
                        "accuracy"
                    ],

                "fold_mean_accuracy":
                    baseline[
                        "fold_mean"
                    ],

                "fold_sd_accuracy":
                    baseline[
                        "fold_sd"
                    ],

                "classifier":
                    "RBF-SVM",

                "C":
                    SVM_C
            }
        ]
    )


    fold_df = pd.DataFrame(
        baseline[
            "fold_results"
        ]
    )


    fold_df.insert(
        0,
        "language",
        language
    )


    return {

        "baseline":
            baseline_row,

        "folds":
            fold_df,

        "channels":
            ch_df,

        "regions":
            region_df,

        "permutation_raw":
            perm_raw,

        "permutation":
            perm_summary,

        "shap":
            shap_df
    }


# ============================================================
# MAIN
# ============================================================

def main():

    np.random.seed(
        SEED
    )


    print(
        "=" * 90
    )

    print(
        "RIEMANNIAN-SVM XAI ANALYSIS"
    )

    print(
        "=" * 90
    )


    print(
        "\nChannel order:"
    )

    print(
        CHANNELS
    )


    print(
        "\nWindow:",
        WINDOW_SAMPLES,
        "samples =",
        WINDOW_SAMPLES / FS,
        "s"
    )


    print(
        "Step:",
        STEP_SAMPLES,
        "samples"
    )


    print(
        "Windows/epoch:",
        N_WINDOWS_PER_EPOCH
    )


    print(
        "\nFull 12-channel tangent dimension should be:"
    )

    print(
        N_CHANNELS
        *
        (
            N_CHANNELS
            +
            1
        )
        //
        2
    )


    # ========================================================
    # RESOLVE DATA FOLDERS
    # ========================================================

    eng_dir = resolve_directory(
        ENG_DIR,
        "English"
    )


    bn_dir = resolve_directory(
        BN_DIR,
        "Bangla"
    )


    # ========================================================
    # LOAD
    # ========================================================

    english = load_language(
        "English",
        eng_dir
    )


    bangla = load_language(
        "Bangla",
        bn_dir
    )


    # ========================================================
    # WINDOWING
    # ========================================================

    create_windows(
        english
    )


    create_windows(
        bangla
    )


    # ========================================================
    # XAI
    # ========================================================

    english_result = run_language_xai(
        english
    )


    bangla_result = run_language_xai(
        bangla
    )


    results = [
        english_result,
        bangla_result
    ]


    # ========================================================
    # SAVE NUMERIC OUTPUTS
    # ========================================================

    baseline_df = pd.concat(
        [
            r[
                "baseline"
            ]
            for r in results
        ],
        ignore_index=True
    )


    folds_df = pd.concat(
        [
            r[
                "folds"
            ]
            for r in results
        ],
        ignore_index=True
    )


    channel_df = pd.concat(
        [
            r[
                "channels"
            ]
            for r in results
        ],
        ignore_index=True
    )


    region_df = pd.concat(
        [
            r[
                "regions"
            ]
            for r in results
        ],
        ignore_index=True
    )


    perm_raw_df = pd.concat(
        [
            r[
                "permutation_raw"
            ]
            for r in results
        ],
        ignore_index=True
    )


    perm_df = pd.concat(
        [
            r[
                "permutation"
            ]
            for r in results
        ],
        ignore_index=True
    )


    baseline_df.to_csv(
        OUT_DIR
        /
        "Table_XAI_Baseline.csv",
        index=False
    )


    folds_df.to_csv(
        OUT_DIR
        /
        "Table_XAI_Fold_Results.csv",
        index=False
    )


    channel_df.to_csv(
        OUT_DIR
        /
        "Table_XAI_Channel_Importance.csv",
        index=False
    )


    region_df.to_csv(
        OUT_DIR
        /
        "Table_XAI_Regional_Importance.csv",
        index=False
    )


    perm_raw_df.to_csv(
        OUT_DIR
        /
        "Table_XAI_Permutation_Raw.csv",
        index=False
    )


    perm_df.to_csv(
        OUT_DIR
        /
        "Table_XAI_Feature_Importance.csv",
        index=False
    )


    # ========================================================
    # SHAP TABLE
    # ========================================================

    if RUN_SHAP:

        shap_frames = [
            r[
                "shap"
            ]
            for r in results
            if r[
                "shap"
            ]
            is not None
        ]


        shap_df = pd.concat(
            shap_frames,
            ignore_index=True
        )


        shap_df.to_csv(
            OUT_DIR
            /
            "Table_XAI_SHAP_Importance.csv",
            index=False
        )

    else:

        shap_df = None


    # ========================================================
    # FIGURES
    # ========================================================

    make_channel_figures(
        channel_df
    )


    make_region_figure(
        region_df
    )


    for language in [
        "English",
        "Bangla"
    ]:

        make_feature_figure(
            perm_df,
            language,
            "importance_pp_mean",
            (
                f"{language} tangent-feature "
                f"permutation importance"
            ),
            (
                f"Fig_XAI_Permutation_"
                f"{language}"
            ),
            (
                "Epoch-voted accuracy reduction "
                "(percentage points)"
            )
        )


        if (
            RUN_SHAP
            and
            shap_df
            is not None
        ):

            make_feature_figure(
                shap_df,
                language,
                "mean_abs_shap",
                (
                    f"{language} tangent-feature "
                    f"SHAP importance"
                ),
                (
                    f"Fig_XAI_SHAP_"
                    f"{language}"
                ),
                "Mean absolute SHAP value"
            )


    # ========================================================
    # PAPER-SUMMARY TABLE
    # ========================================================

    summary_rows = []


    for language in [
        "English",
        "Bangla"
    ]:

        baseline_acc = baseline_df[
            baseline_df[
                "language"
            ]
            ==
            language
        ][
            "pooled_epoch_accuracy"
        ].iloc[0]


        top_channel = (
            channel_df[
                channel_df[
                    "language"
                ]
                ==
                language
            ]
            .sort_values(
                "importance_pp",
                ascending=False
            )
            .iloc[0]
        )


        top_region = (
            region_df[
                region_df[
                    "language"
                ]
                ==
                language
            ]
            .sort_values(
                "importance_pp",
                ascending=False
            )
            .iloc[0]
        )


        top_perm = (
            perm_df[
                perm_df[
                    "language"
                ]
                ==
                language
            ]
            .sort_values(
                "importance_pp_mean",
                ascending=False
            )
            .iloc[0]
        )


        row = {

            "language":
                language,

            "baseline_accuracy_percent":
                baseline_acc
                *
                100,

            "most_important_channel":
                top_channel[
                    "channel"
                ],

            "channel_importance_pp":
                top_channel[
                    "importance_pp"
                ],

            "most_important_region":
                top_region[
                    "region"
                ],

            "region_importance_pp":
                top_region[
                    "importance_pp"
                ],

            "top_permutation_feature":
                top_perm[
                    "feature_name"
                ],

            "permutation_importance_pp":
                top_perm[
                    "importance_pp_mean"
                ]
        }


        if (
            RUN_SHAP
            and
            shap_df
            is not None
        ):

            top_shap = (
                shap_df[
                    shap_df[
                        "language"
                    ]
                    ==
                    language
                ]
                .sort_values(
                    "mean_abs_shap",
                    ascending=False
                )
                .iloc[0]
            )


            row[
                "top_shap_feature"
            ] = top_shap[
                "feature_name"
            ]


            row[
                "top_shap_value"
            ] = top_shap[
                "mean_abs_shap"
            ]


        summary_rows.append(
            row
        )


    summary_df = pd.DataFrame(
        summary_rows
    )


    summary_df.to_csv(
        OUT_DIR
        /
        "Table_XAI_Paper_Summary.csv",
        index=False
    )


    # ========================================================
    # PRINT FINAL SUMMARY
    # ========================================================

    print(
        "\n"
        +
        "=" * 90
    )

    print(
        "FINAL XAI SUMMARY"
    )

    print(
        "=" * 90
    )


    print(
        "\nBASELINE:"
    )


    for _, row in baseline_df.iterrows():

        print(
            f"{row['language']:8s}: "
            f"{row['pooled_epoch_accuracy'] * 100:.2f}% "
            f"| tangent dim "
            f"{int(row['tangent_dimension'])}"
        )


    print(
        "\nTOP CHANNELS:"
    )


    for language in [
        "English",
        "Bangla"
    ]:

        temp = channel_df[
            channel_df[
                "language"
            ]
            ==
            language
        ].sort_values(
            "importance_pp",
            ascending=False
        )


        print(
            language,
            ":",
            ", ".join(
                [
                    f"{r.channel}"
                    f" ({r.importance_pp:+.2f} pp)"
                    for r in temp.head(
                        5
                    ).itertuples()
                ]
            )
        )


    print(
        "\nTOP REGIONS:"
    )


    for language in [
        "English",
        "Bangla"
    ]:

        temp = region_df[
            region_df[
                "language"
            ]
            ==
            language
        ].sort_values(
            "importance_pp",
            ascending=False
        )


        print(
            language,
            ":",
            ", ".join(
                [
                    f"{r.region}"
                    f" ({r.importance_pp:+.2f} pp)"
                    for r in temp.itertuples()
                ]
            )
        )


    print(
        "\nSaved all results to:"
    )

    print(
        OUT_DIR
    )


    print(
        "\nIMPORTANT:"
    )

    print(
        "Do not use these XAI figures in the final paper "
        "until the baseline accuracy is checked against "
        "your established English/Bangla baseline."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()