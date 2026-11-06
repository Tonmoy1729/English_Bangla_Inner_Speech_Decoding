# ============================================================
# PAPER FIGURE:
# BEST SINGLE-CHANNEL REMOVAL
# ENGLISH + BANGLA INNER-SPEECH EEG
#
# Output:
#   1. English epoch-level confusion matrix
#   2. Bangla epoch-level confusion matrix
#   3. English vs Bangla class-wise accuracy bar chart
#   4. Summary CSV
#   5. Epoch-level prediction CSV
#
# Pipeline kept identical to the XAI analysis:
#   - 7-s epochs
#   - 2-s windows
#   - 50% overlap
#   - per-window/channel z-score
#   - OAS covariance
#   - train-only Riemannian tangent space
#   - train-only StandardScaler
#   - RBF-SVM, C=10
#   - epoch-position heldout evaluation
#   - epoch-level majority voting
# ============================================================

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support
)

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


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

# 7 seconds × 256 Hz
EPOCH_SAMPLES = 1792

# 2-second window
WINDOW_SAMPLES = 512

# 50% overlap = 1-second step
STEP_SAMPLES = 256

WINDOW_STARTS = list(
    range(
        0,
        EPOCH_SAMPLES - WINDOW_SAMPLES + 1,
        STEP_SAMPLES
    )
)

N_WINDOWS_PER_EPOCH = len(WINDOW_STARTS)


# ============================================================
# 3. MODEL SETTINGS
# ============================================================

SVM_C = 10
SVM_GAMMA = "scale"
SEED = 42

np.random.seed(SEED)


# ============================================================
# 4. CLASS LABELS
# ============================================================

COMMANDS = [
    "back",
    "down",
    "forward",
    "left",
    "right",
    "up"
]

DISPLAY_COMMANDS = [
    "Back",
    "Down",
    "Forward",
    "Left",
    "Right",
    "Up"
]

COMMAND_TO_ID = {
    command: i
    for i, command in enumerate(COMMANDS)
}

TOKEN_TO_COMMAND = {

    # English
    "back": "back",
    "down": "down",
    "forward": "forward",
    "left": "left",
    "right": "right",
    "up": "up",

    # Bangla transliteration used in filenames
    "pichone": "back",
    "niche": "down",
    "samne": "forward",
    "bame": "left",
    "dane": "right",
    "upore": "up"
}


# ============================================================
# 5. PAPER PLOT STYLE
# ============================================================

plt.rcParams.update({

    "font.family": "serif",

    "font.size": 10,

    "axes.titlesize": 11,

    "axes.labelsize": 10,

    "xtick.labelsize": 8,

    "ytick.labelsize": 8,

    "legend.fontsize": 9,

    "figure.dpi": 150
})


# ============================================================
# 6. FIND DATASET DIRECTORY
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
            keyword.lower() in p.name.lower()
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
        f"Expected:\n{preferred}"
    )


# ============================================================
# 7. DETECT CLASS FROM FILENAME
# ============================================================

def detect_command(filepath):

    stem = filepath.stem.lower()

    tokens = re.split(
        r"[_\-\s]+",
        stem
    )

    for token in reversed(tokens):

        if token in TOKEN_TO_COMMAND:

            return TOKEN_TO_COMMAND[
                token
            ]

    raise ValueError(
        f"Cannot determine command from:\n"
        f"{filepath.name}"
    )


# ============================================================
# 8. FIND / READ META FILE
# ============================================================

def find_meta_file(csv_path):

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

    for candidate in candidates:

        if candidate.exists():

            return candidate

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

    # If metadata are absent
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
# 9. LOAD ONE LANGUAGE
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

    print(
        f"\nLoading {language}: "
        f"{len(files)} files"
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
                f"Unexpected epoch length in "
                f"{file.name}: {pnts}. "
                f"Expected {EPOCH_SAMPLES}."
            )

        if n_epochs < 1:

            continue

        usable = (
            n_epochs
            *
            EPOCH_SAMPLES
        )

        raw = raw[
            :usable
        ]

        for epoch_position in range(
            n_epochs
        ):

            segment = raw[
                epoch_position * EPOCH_SAMPLES:
                (epoch_position + 1) * EPOCH_SAMPLES
            ]

            # Original CSV:
            # time × channels
            #
            # Convert to:
            # channels × time

            segment = segment.T

            epoch_data.append(
                segment
            )

            epoch_labels.append(
                label
            )

            # Critical:
            # which repetition / epoch position
            # inside the command recording?
            epoch_positions.append(
                epoch_position
            )

            epoch_files.append(
                file.name
            )

        if (
            file_no % 30 == 0
            or
            file_no == len(files)
        ):

            print(
                f"  {file_no}/"
                f"{len(files)} files loaded"
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
        f"{language} epoch shape:",
        epochs.shape
    )

    print(
        f"{language} class counts:",
        {
            DISPLAY_COMMANDS[c]:
                int(
                    np.sum(
                        labels == c
                    )
                )
            for c in range(
                len(COMMANDS)
            )
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
            )
    }


# ============================================================
# 10. WINDOWING
# ============================================================

def create_windows(data):

    epochs = data[
        "epochs"
    ]

    windows = []

    window_epoch_ids = []

    window_labels = []

    for epoch_id in range(
        len(epochs)
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

            window = epoch[
                :,
                start:stop
            ].copy()

            # ------------------------------------------------
            # Per-window, per-channel z-score
            # ------------------------------------------------

            mean = window.mean(
                axis=1,
                keepdims=True
            )

            std = window.std(
                axis=1,
                keepdims=True
            )

            window = (
                window
                -
                mean
            ) / (
                std
                +
                1e-8
            )

            windows.append(
                window
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
# 11. EPOCH-LEVEL MAJORITY VOTING
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

    for epoch_id in unique_epochs:

        mask = (
            epoch_ids
            ==
            epoch_id
        )

        epoch_predictions = predictions[
            mask
        ]

        counts = np.bincount(
            epoch_predictions.astype(int),
            minlength=len(COMMANDS)
        )

        final_prediction = int(
            np.argmax(
                counts
            )
        )

        y_true.append(
            int(
                epoch_labels[
                    epoch_id
                ]
            )
        )

        y_pred.append(
            final_prediction
        )

    return (
        np.asarray(
            y_true,
            dtype=int
        ),
        np.asarray(
            y_pred,
            dtype=int
        ),
        unique_epochs
    )


# ============================================================
# 12. EVALUATE A CHANNEL SUBSET
#
# Identical logic to the XAI channel-ablation evaluation.
#
# Returns actual epoch predictions so that we can generate
# the new confusion matrices and class-wise results.
# ============================================================

def evaluate_channel_subset(
    data,
    keep_channel_indices
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

    # --------------------------------------------------------
    # OAS covariance
    # --------------------------------------------------------

    covariances = Covariances(
        estimator="oas"
    ).transform(
        windows
    )

    all_true = []

    all_pred = []

    all_epoch_ids = []

    fold_rows = []

    max_epoch_position = int(
        np.max(
            epoch_positions
        )
    )

    # --------------------------------------------------------
    # Epoch-position heldout evaluation
    #
    # Fold 1:
    #   test repetition 1 from every recording
    #
    # Fold 2:
    #   test repetition 2
    #
    # etc.
    # --------------------------------------------------------

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

        train_cov = covariances[
            train_window_mask
        ]

        test_cov = covariances[
            test_window_mask
        ]

        y_train = window_labels[
            train_window_mask
        ]

        test_window_epoch_ids = (
            window_epoch_ids[
                test_window_mask
            ]
        )

        # ----------------------------------------------------
        # TRAIN-ONLY Riemannian tangent-space mapping
        # ----------------------------------------------------

        tangent_space = TangentSpace(
            metric="riemann"
        )

        X_train = tangent_space.fit_transform(
            train_cov
        )

        X_test = tangent_space.transform(
            test_cov
        )

        # ----------------------------------------------------
        # TRAIN-ONLY feature scaling
        # ----------------------------------------------------

        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            X_train
        )

        X_test = scaler.transform(
            X_test
        )

        # ----------------------------------------------------
        # RBF-SVM
        # ----------------------------------------------------

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

        window_predictions = model.predict(
            X_test
        )

        # ----------------------------------------------------
        # Epoch voting
        # ----------------------------------------------------

        (
            y_true_epoch,
            y_pred_epoch,
            evaluated_epoch_ids
        ) = epoch_vote(
            window_predictions,
            test_window_epoch_ids,
            epoch_labels
        )

        fold_accuracy = accuracy_score(
            y_true_epoch,
            y_pred_epoch
        )

        fold_rows.append(
            {
                "fold":
                    fold + 1,

                "accuracy":
                    fold_accuracy,

                "n_test_epochs":
                    len(
                        y_true_epoch
                    )
            }
        )

        all_true.extend(
            y_true_epoch
        )

        all_pred.extend(
            y_pred_epoch
        )

        all_epoch_ids.extend(
            evaluated_epoch_ids
        )

    all_true = np.asarray(
        all_true,
        dtype=int
    )

    all_pred = np.asarray(
        all_pred,
        dtype=int
    )

    all_epoch_ids = np.asarray(
        all_epoch_ids,
        dtype=int
    )

    pooled_accuracy = accuracy_score(
        all_true,
        all_pred
    )

    return {

        "accuracy":
            pooled_accuracy,

        "y_true":
            all_true,

        "y_pred":
            all_pred,

        "epoch_ids":
            all_epoch_ids,

        "fold_results":
            pd.DataFrame(
                fold_rows
            )
    }


# ============================================================
# 13. FIND CHANNEL-IMPORTANCE CSV
# ============================================================

def find_channel_importance_csv():

    candidates = sorted(
        OUT_DIR.glob(
            "Table_XAI_Channel_Importance*.csv"
        )
    )

    if len(candidates) == 0:

        raise FileNotFoundError(
            "\nCould not find "
            "Table_XAI_Channel_Importance*.csv "
            f"in:\n{OUT_DIR}"
        )

    # Use newest matching file
    csv_file = max(
        candidates,
        key=lambda p: p.stat().st_mtime
    )

    print(
        "\nUsing channel importance CSV:"
    )

    print(
        csv_file
    )

    return csv_file


# ============================================================
# 14. SELECT BEST SINGLE-CHANNEL REMOVAL
# ============================================================

def get_best_removed_channel(
    channel_df,
    language
):

    subset = channel_df[
        channel_df[
            "language"
        ].str.lower()
        ==
        language.lower()
    ].copy()

    if len(subset) == 0:

        raise RuntimeError(
            f"No rows found for {language}"
        )

    best = subset.sort_values(
        "ablated_accuracy",
        ascending=False
    ).iloc[0]

    return {

        "channel":
            str(
                best[
                    "channel"
                ]
            ),

        "baseline_accuracy":
            float(
                best[
                    "baseline_accuracy"
                ]
            ),

        "ablated_accuracy":
            float(
                best[
                    "ablated_accuracy"
                ]
            ),

        "importance_pp":
            float(
                best[
                    "importance_pp"
                ]
            )
    }


# ============================================================
# 15. CLASS-WISE METRICS
# ============================================================

def calculate_classwise_metrics(
    y_true,
    y_pred
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(
            len(COMMANDS)
        )
    )

    true_totals = cm.sum(
        axis=1
    )

    predicted_totals = cm.sum(
        axis=0
    )

    diagonal = np.diag(
        cm
    ).astype(float)

    recall = np.divide(
        diagonal,
        true_totals,
        out=np.zeros_like(
            diagonal
        ),
        where=true_totals != 0
    )

    precision = np.divide(
        diagonal,
        predicted_totals,
        out=np.zeros_like(
            diagonal
        ),
        where=predicted_totals != 0
    )

    return (
        cm,
        recall,
        precision
    )


# ============================================================
# 16. PAPER-STYLE CONFUSION MATRIX
#
# Main matrix:
#   row-normalized percentages
#
# Right side:
#   recall / class accuracy and class error
#
# Bottom:
#   precision and prediction error
#
# This closely matches the visual structure used in the
# current paper figure.
# ============================================================

def plot_paper_confusion_matrix(
    fig,
    outer_spec,
    result,
    language,
    removed_channel
):

    y_true = result[
        "y_true"
    ]

    y_pred = result[
        "y_pred"
    ]

    cm, recall, precision = (
        calculate_classwise_metrics(
            y_true,
            y_pred
        )
    )

    row_totals = cm.sum(
        axis=1,
        keepdims=True
    )

    cm_percent = np.divide(
        cm,
        row_totals,
        out=np.zeros_like(
            cm,
            dtype=float
        ),
        where=row_totals != 0
    ) * 100.0

    # --------------------------------------------------------
    # Layout:
    #
    # [ main CM ] [ recall ] [ error ]
    # [ precision / prediction error ]
    # --------------------------------------------------------

    inner = outer_spec.subgridspec(
        2,
        3,
        width_ratios=[
            8.5,
            1.15,
            1.15
        ],
        height_ratios=[
            8.5,
            1.25
        ],
        wspace=0.08,
        hspace=0.08
    )

    ax_cm = fig.add_subplot(
        inner[
            0,
            0
        ]
    )

    ax_recall = fig.add_subplot(
        inner[
            0,
            1
        ]
    )

    ax_error = fig.add_subplot(
        inner[
            0,
            2
        ]
    )

    bottom_inner = inner[
        1,
        0
    ].subgridspec(
        2,
        1,
        hspace=0.05
    )

    ax_precision = fig.add_subplot(
        bottom_inner[
            0,
            0
        ]
    )

    ax_pred_error = fig.add_subplot(
        bottom_inner[
            1,
            0
        ]
    )

    # --------------------------------------------------------
    # Main confusion matrix
    # --------------------------------------------------------

    image = ax_cm.imshow(
        cm_percent,
        cmap="Blues",
        vmin=0,
        vmax=100
    )

    ax_cm.set_xticks(
        np.arange(
            len(COMMANDS)
        )
    )

    ax_cm.set_yticks(
        np.arange(
            len(COMMANDS)
        )
    )

    ax_cm.set_xticklabels(
        DISPLAY_COMMANDS,
        rotation=45,
        ha="right"
    )

    ax_cm.set_yticklabels(
        DISPLAY_COMMANDS
    )

    ax_cm.set_xlabel(
        "Predicted Class"
    )

    ax_cm.set_ylabel(
        "True Class"
    )

    ax_cm.set_title(
        f"{language} Inner Speech — Epoch-Heldout RBF-SVM\n"
        f"Remove {removed_channel} | "
        f"Accuracy = "
        f"{result['accuracy'] * 100:.2f}%",
        fontweight="bold"
    )

    for i in range(
        len(COMMANDS)
    ):

        for j in range(
            len(COMMANDS)
        ):

            value = cm_percent[
                i,
                j
            ]

            text_color = (
                "white"
                if value >= 50
                else "black"
            )

            ax_cm.text(
                j,
                i,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=7.5,
                color=text_color
            )

    # white cell separators
    ax_cm.set_xticks(
        np.arange(
            -0.5,
            len(COMMANDS),
            1
        ),
        minor=True
    )

    ax_cm.set_yticks(
        np.arange(
            -0.5,
            len(COMMANDS),
            1
        ),
        minor=True
    )

    ax_cm.grid(
        which="minor",
        linewidth=1.0
    )

    ax_cm.tick_params(
        which="minor",
        bottom=False,
        left=False
    )

    # --------------------------------------------------------
    # Right marginal:
    # per-class recall / correct %
    # --------------------------------------------------------

    recall_percent = (
        recall
        *
        100
    )

    recall_error = (
        100
        -
        recall_percent
    )

    ax_recall.imshow(
        recall_percent.reshape(
            -1,
            1
        ),
        cmap="Blues",
        vmin=0,
        vmax=100,
        aspect="auto"
    )

    ax_recall.set_xticks([])
    ax_recall.set_yticks([])

    for i, value in enumerate(
        recall_percent
    ):

        ax_recall.text(
            0,
            i,
            f"{value:.1f}%",
            ha="center",
            va="center",
            fontsize=7
        )

    # --------------------------------------------------------
    # Right marginal:
    # error %
    # --------------------------------------------------------

    ax_error.imshow(
        recall_error.reshape(
            -1,
            1
        ),
        cmap="Oranges",
        vmin=0,
        vmax=100,
        aspect="auto"
    )

    ax_error.set_xticks([])
    ax_error.set_yticks([])

    for i, value in enumerate(
        recall_error
    ):

        ax_error.text(
            0,
            i,
            f"{value:.1f}%",
            ha="center",
            va="center",
            fontsize=7
        )

    # --------------------------------------------------------
    # Bottom marginal:
    # per-predicted-class precision
    # --------------------------------------------------------

    precision_percent = (
        precision
        *
        100
    )

    precision_error = (
        100
        -
        precision_percent
    )

    ax_precision.imshow(
        precision_percent.reshape(
            1,
            -1
        ),
        cmap="Blues",
        vmin=0,
        vmax=100,
        aspect="auto"
    )

    ax_precision.set_xticks([])
    ax_precision.set_yticks([])

    for j, value in enumerate(
        precision_percent
    ):

        ax_precision.text(
            j,
            0,
            f"{value:.1f}%",
            ha="center",
            va="center",
            fontsize=7
        )

    # --------------------------------------------------------
    # Bottom error %
    # --------------------------------------------------------

    ax_pred_error.imshow(
        precision_error.reshape(
            1,
            -1
        ),
        cmap="Oranges",
        vmin=0,
        vmax=100,
        aspect="auto"
    )

    ax_pred_error.set_xticks([])
    ax_pred_error.set_yticks([])

    for j, value in enumerate(
        precision_error
    ):

        ax_pred_error.text(
            j,
            0,
            f"{value:.1f}%",
            ha="center",
            va="center",
            fontsize=7
        )

    return (
        cm,
        recall_percent,
        precision_percent,
        image
    )


# ============================================================
# 17. COMBINED PAPER FIGURE
# ============================================================

def create_combined_paper_figure(
    english_result,
    bangla_result,
    english_removed,
    bangla_removed
):

    fig = plt.figure(
        figsize=(
            12.5,
            10.5
        )
    )

    main_grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[
            1.15,
            0.75
        ],
        hspace=0.40,
        wspace=0.26
    )

    # --------------------------------------------------------
    # English CM
    # --------------------------------------------------------

    (
        eng_cm,
        eng_class_accuracy,
        eng_precision,
        image_eng
    ) = plot_paper_confusion_matrix(
        fig,
        main_grid[
            0,
            0
        ],
        english_result,
        "English",
        english_removed
    )

    # --------------------------------------------------------
    # Bangla CM
    # --------------------------------------------------------

    (
        bn_cm,
        bn_class_accuracy,
        bn_precision,
        image_bn
    ) = plot_paper_confusion_matrix(
        fig,
        main_grid[
            0,
            1
        ],
        bangla_result,
        "Bangla",
        bangla_removed
    )

    # --------------------------------------------------------
    # Class-wise bar chart
    # --------------------------------------------------------

    ax_bar = fig.add_subplot(
        main_grid[
            1,
            :
        ]
    )

    x = np.arange(
        len(COMMANDS)
    )

    width = 0.36

    english_bars = ax_bar.bar(
        x - width / 2,
        eng_class_accuracy,
        width,
        label="English"
    )

    bangla_bars = ax_bar.bar(
        x + width / 2,
        bn_class_accuracy,
        width,
        label="Bangla"
    )

    ax_bar.set_xticks(
        x
    )

    ax_bar.set_xticklabels(
        DISPLAY_COMMANDS
    )

    ax_bar.set_ylabel(
        "Accuracy (%)"
    )

    ax_bar.set_xlabel(
        "Class"
    )

    ax_bar.set_ylim(
        0,
        100
    )

    ax_bar.set_title(
        "Class-wise Inner Speech Classification Accuracy",
        fontweight="bold"
    )

    ax_bar.legend(
        frameon=False
    )

    ax_bar.grid(
        axis="y",
        alpha=0.25
    )

    # --------------------------------------------------------
    # Value labels
    # --------------------------------------------------------

    for bars in [
        english_bars,
        bangla_bars
    ]:

        for bar in bars:

            height = bar.get_height()

            ax_bar.text(
                bar.get_x()
                +
                bar.get_width() / 2,
                height + 1.2,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8
            )

    # --------------------------------------------------------
    # Panel labels
    # --------------------------------------------------------

    fig.text(
        0.27,
        0.505,
        "(a) English epoch-level classification",
        ha="center",
        fontweight="bold",
        fontsize=11
    )

    fig.text(
        0.74,
        0.505,
        "(b) Bangla epoch-level classification",
        ha="center",
        fontweight="bold",
        fontsize=11
    )

    fig.text(
        0.50,
        0.055,
        "(c) Per-class classification accuracy",
        ha="center",
        fontweight="bold",
        fontsize=11
    )

    fig.subplots_adjust(
        top=0.96,
        bottom=0.10,
        left=0.06,
        right=0.98
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    png_path = (
        OUT_DIR
        /
        "Fig_Epoch_CM_Classwise_Best_Channel_Removal.png"
    )

    pdf_path = (
        OUT_DIR
        /
        "Fig_Epoch_CM_Classwise_Best_Channel_Removal.pdf"
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        "\nSaved figure:"
    )

    print(
        png_path
    )

    print(
        pdf_path
    )

    return (
        eng_cm,
        bn_cm,
        eng_class_accuracy,
        bn_class_accuracy
    )


# ============================================================
# 18. SAVE NUMERICAL RESULTS
# ============================================================

def save_results(
    english,
    bangla,
    english_result,
    bangla_result,
    english_best,
    bangla_best,
    eng_cm,
    bn_cm,
    eng_class_accuracy,
    bn_class_accuracy
):

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    summary_rows = []

    for (
        language,
        result,
        best
    ) in [

        (
            "English",
            english_result,
            english_best
        ),

        (
            "Bangla",
            bangla_result,
            bangla_best
        )
    ]:

        (
            precision,
            recall,
            f1,
            _
        ) = precision_recall_fscore_support(
            result[
                "y_true"
            ],
            result[
                "y_pred"
            ],
            labels=np.arange(
                len(COMMANDS)
            ),
            average="macro",
            zero_division=0
        )

        summary_rows.append(
            {
                "language":
                    language,

                "removed_channel":
                    best[
                        "channel"
                    ],

                "original_baseline_accuracy_percent":
                    best[
                        "baseline_accuracy"
                    ] * 100,

                "csv_ablated_accuracy_percent":
                    best[
                        "ablated_accuracy"
                    ] * 100,

                "recomputed_accuracy_percent":
                    result[
                        "accuracy"
                    ] * 100,

                "macro_precision_percent":
                    precision * 100,

                "macro_recall_percent":
                    recall * 100,

                "macro_f1_percent":
                    f1 * 100,

                "n_evaluated_epochs":
                    len(
                        result[
                            "y_true"
                        ]
                    )
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        OUT_DIR
        /
        "Table_Best_Channel_Removal_Summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------------
    # Class-wise results
    # --------------------------------------------------------

    classwise_rows = []

    for class_id, class_name in enumerate(
        DISPLAY_COMMANDS
    ):

        classwise_rows.append(
            {
                "class":
                    class_name,

                "english_accuracy_percent":
                    eng_class_accuracy[
                        class_id
                    ],

                "bangla_accuracy_percent":
                    bn_class_accuracy[
                        class_id
                    ]
            }
        )

    classwise_df = pd.DataFrame(
        classwise_rows
    )

    classwise_path = (
        OUT_DIR
        /
        "Table_Best_Channel_Removal_Classwise.csv"
    )

    classwise_df.to_csv(
        classwise_path,
        index=False
    )

    # --------------------------------------------------------
    # Confusion matrices
    # --------------------------------------------------------

    eng_cm_df = pd.DataFrame(
        eng_cm,
        index=DISPLAY_COMMANDS,
        columns=DISPLAY_COMMANDS
    )

    bn_cm_df = pd.DataFrame(
        bn_cm,
        index=DISPLAY_COMMANDS,
        columns=DISPLAY_COMMANDS
    )

    eng_cm_df.to_csv(
        OUT_DIR
        /
        "CM_English_Best_Channel_Removal.csv"
    )

    bn_cm_df.to_csv(
        OUT_DIR
        /
        "CM_Bangla_Best_Channel_Removal.csv"
    )

    # --------------------------------------------------------
    # Epoch-level predictions
    # --------------------------------------------------------

    prediction_frames = []

    for (
        language,
        data,
        result,
        best
    ) in [

        (
            "English",
            english,
            english_result,
            english_best
        ),

        (
            "Bangla",
            bangla,
            bangla_result,
            bangla_best
        )
    ]:

        epoch_ids = result[
            "epoch_ids"
        ]

        prediction_df = pd.DataFrame(
            {
                "language":
                    language,

                "removed_channel":
                    best[
                        "channel"
                    ],

                "epoch_id":
                    epoch_ids,

                "source_file":
                    data[
                        "epoch_files"
                    ][
                        epoch_ids
                    ],

                "true_class_id":
                    result[
                        "y_true"
                    ],

                "predicted_class_id":
                    result[
                        "y_pred"
                    ]
            }
        )

        prediction_df[
            "true_class"
        ] = prediction_df[
            "true_class_id"
        ].map(
            lambda x:
                DISPLAY_COMMANDS[
                    int(x)
                ]
        )

        prediction_df[
            "predicted_class"
        ] = prediction_df[
            "predicted_class_id"
        ].map(
            lambda x:
                DISPLAY_COMMANDS[
                    int(x)
                ]
        )

        prediction_df[
            "correct"
        ] = (
            prediction_df[
                "true_class_id"
            ]
            ==
            prediction_df[
                "predicted_class_id"
            ]
        )

        prediction_frames.append(
            prediction_df
        )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True
    )

    predictions_path = (
        OUT_DIR
        /
        "Epoch_Predictions_Best_Channel_Removal.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    print(
        "\nSaved tables:"
    )

    print(
        summary_path
    )

    print(
        classwise_path
    )

    print(
        predictions_path
    )


# ============================================================
# 19. MAIN
# ============================================================

def main():

    print(
        "=" * 90
    )

    print(
        "BEST SINGLE-CHANNEL REMOVAL:"
        " PAPER CONFUSION MATRICES"
        " + CLASS-WISE ACCURACY"
    )

    print(
        "=" * 90
    )

    print(
        "\nChannels:"
    )

    print(
        CHANNELS
    )

    print(
        "\nWindow configuration:"
    )

    print(
        f"{WINDOW_SAMPLES} samples "
        f"= {WINDOW_SAMPLES / FS:.1f} s"
    )

    print(
        f"Step = {STEP_SAMPLES} samples "
        f"= {STEP_SAMPLES / FS:.1f} s"
    )

    print(
        f"Windows per epoch = "
        f"{N_WINDOWS_PER_EPOCH}"
    )

    # --------------------------------------------------------
    # Dataset folders
    # --------------------------------------------------------

    eng_dir = resolve_directory(
        ENG_DIR,
        "English"
    )

    bn_dir = resolve_directory(
        BN_DIR,
        "Bangla"
    )

    # --------------------------------------------------------
    # Read existing XAI channel-ablation results
    # --------------------------------------------------------

    channel_csv = (
        find_channel_importance_csv()
    )

    channel_df = pd.read_csv(
        channel_csv
    )

    english_best = (
        get_best_removed_channel(
            channel_df,
            "English"
        )
    )

    bangla_best = (
        get_best_removed_channel(
            channel_df,
            "Bangla"
        )
    )

    print(
        "\n"
        +
        "=" * 90
    )

    print(
        "BEST CHANNEL REMOVAL FROM XAI CSV"
    )

    print(
        "=" * 90
    )

    print(
        "\nEnglish:"
    )

    print(
        f"  Remove channel    : "
        f"{english_best['channel']}"
    )

    print(
        f"  Full model        : "
        f"{english_best['baseline_accuracy'] * 100:.2f}%"
    )

    print(
        f"  After removal     : "
        f"{english_best['ablated_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Accuracy change   : "
        f"{-english_best['importance_pp']:+.2f} pp"
    )

    print(
        "\nBangla:"
    )

    print(
        f"  Remove channel    : "
        f"{bangla_best['channel']}"
    )

    print(
        f"  Full model        : "
        f"{bangla_best['baseline_accuracy'] * 100:.2f}%"
    )

    print(
        f"  After removal     : "
        f"{bangla_best['ablated_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Accuracy change   : "
        f"{-bangla_best['importance_pp']:+.2f} pp"
    )

    # --------------------------------------------------------
    # Load EEG
    # --------------------------------------------------------

    english = load_language(
        "English",
        eng_dir
    )

    bangla = load_language(
        "Bangla",
        bn_dir
    )

    # --------------------------------------------------------
    # Windowing
    # --------------------------------------------------------

    create_windows(
        english
    )

    create_windows(
        bangla
    )

    # --------------------------------------------------------
    # Remove selected English channel
    # --------------------------------------------------------

    english_removed_index = (
        CHANNELS.index(
            english_best[
                "channel"
            ]
        )
    )

    english_keep = [
        i
        for i in range(
            N_CHANNELS
        )
        if i
        !=
        english_removed_index
    ]

    # --------------------------------------------------------
    # Remove selected Bangla channel
    # --------------------------------------------------------

    bangla_removed_index = (
        CHANNELS.index(
            bangla_best[
                "channel"
            ]
        )
    )

    bangla_keep = [
        i
        for i in range(
            N_CHANNELS
        )
        if i
        !=
        bangla_removed_index
    ]

    # --------------------------------------------------------
    # Recompute actual epoch predictions
    # --------------------------------------------------------

    print(
        "\nEvaluating English..."
    )

    english_result = (
        evaluate_channel_subset(
            english,
            english_keep
        )
    )

    print(
        "\nEvaluating Bangla..."
    )

    bangla_result = (
        evaluate_channel_subset(
            bangla,
            bangla_keep
        )
    )

    # --------------------------------------------------------
    # Verify against XAI CSV
    # --------------------------------------------------------

    print(
        "\n"
        +
        "=" * 90
    )

    print(
        "RECOMPUTED RESULTS"
    )

    print(
        "=" * 90
    )

    print(
        f"\nEnglish "
        f"(remove {english_best['channel']}): "
        f"{english_result['accuracy'] * 100:.2f}%"
    )

    print(
        f"Expected from CSV: "
        f"{english_best['ablated_accuracy'] * 100:.2f}%"
    )

    print(
        f"\nBangla "
        f"(remove {bangla_best['channel']}): "
        f"{bangla_result['accuracy'] * 100:.2f}%"
    )

    print(
        f"Expected from CSV: "
        f"{bangla_best['ablated_accuracy'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Safety verification
    # --------------------------------------------------------

    eng_difference = abs(
        english_result[
            "accuracy"
        ]
        -
        english_best[
            "ablated_accuracy"
        ]
    )

    bn_difference = abs(
        bangla_result[
            "accuracy"
        ]
        -
        bangla_best[
            "ablated_accuracy"
        ]
    )

    tolerance = 1e-9

    if eng_difference > tolerance:

        print(
            "\nWARNING:"
            " English recomputed accuracy does not "
            "exactly match the channel-ablation CSV."
        )

    if bn_difference > tolerance:

        print(
            "\nWARNING:"
            " Bangla recomputed accuracy does not "
            "exactly match the channel-ablation CSV."
        )

    # --------------------------------------------------------
    # Generate figure
    # --------------------------------------------------------

    (
        eng_cm,
        bn_cm,
        eng_class_accuracy,
        bn_class_accuracy
    ) = create_combined_paper_figure(

        english_result,
        bangla_result,

        english_best[
            "channel"
        ],

        bangla_best[
            "channel"
        ]
    )

    # --------------------------------------------------------
    # Print confusion matrices
    # --------------------------------------------------------

    print(
        "\nEnglish confusion matrix:"
    )

    print(
        eng_cm
    )

    print(
        "\nBangla confusion matrix:"
    )

    print(
        bn_cm
    )

    # --------------------------------------------------------
    # Print class-wise accuracy
    # --------------------------------------------------------

    print(
        "\nCLASS-WISE ACCURACY"
    )

    print(
        "-" * 70
    )

    for i, command in enumerate(
        DISPLAY_COMMANDS
    ):

        print(
            f"{command:8s} | "
            f"English "
            f"{eng_class_accuracy[i]:6.2f}% | "
            f"Bangla "
            f"{bn_class_accuracy[i]:6.2f}%"
        )

    # --------------------------------------------------------
    # Save all numerical outputs
    # --------------------------------------------------------

    save_results(
        english,
        bangla,
        english_result,
        bangla_result,
        english_best,
        bangla_best,
        eng_cm,
        bn_cm,
        eng_class_accuracy,
        bn_class_accuracy
    )

    print(
        "\n"
        +
        "=" * 90
    )

    print(
        "DONE"
    )

    print(
        "=" * 90
    )

    print(
        "\nMain figure:"
    )

    print(
        OUT_DIR
        /
        "Fig_Epoch_CM_Classwise_Best_Channel_Removal.png"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()