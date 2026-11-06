import os
import re
import warnings
from glob import glob
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

from sklearn.covariance import OAS
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
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
    r"\FINAL_ABLATION"
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

CLASS_IDS = [1, 2, 3, 4, 5, 6]

CLASS_NAMES = {
    1: "back",
    2: "down",
    3: "forward",
    4: "left",
    5: "right",
    6: "up"
}

LABEL_TO_ID = {
    "back": 1,
    "down": 2,
    "forward": 3,
    "left": 4,
    "right": 5,
    "up": 6
}


# ============================================================
# FILE LABEL MAPPING
# ============================================================

ENG_MAP = {
    "back": "back",
    "backward": "back",
    "down": "down",
    "forward": "forward",
    "left": "left",
    "right": "right",
    "up": "up"
}

BN_MAP = {
    "pichone": "back",
    "pichonee": "back",
    "niche": "down",
    "samne": "forward",
    "ete": "forward",
    "bame": "left",
    "dane": "right",
    "upore": "up"
}


# ============================================================
# FOUR ABLATION CONDITIONS
# ============================================================

METHODS = [
    "Sample covariance + Euclidean + RBF-SVM",
    "OAS covariance + Euclidean + RBF-SVM",
    "OAS + Riemann tangent space, 52-D + RBF-SVM",
    "OAS + Riemann tangent space, 78-D + RBF-SVM"
]


# ============================================================
# METADATA PARSING
# ============================================================

def parse_meta(meta_path):

    with open(
        meta_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        text = f.read().lower()

    def get_value(*keys):

        for key in keys:

            match = re.search(
                rf"{re.escape(key)}\s*=\s*([0-9]+(?:\.[0-9]+)?)",
                text
            )

            if match:
                return float(match.group(1))

        return None

    fs = get_value(
        "srate",
        "sampling_rate",
        "sample_frequency_rate"
    )

    channels = get_value(
        "channels"
    )

    epochs = get_value(
        "epochs"
    )

    points = get_value(
        "pnts_per_epoch",
        "points_per_epoch"
    )

    if None in (
        fs,
        channels,
        epochs,
        points
    ):

        raise ValueError(
            f"Could not parse metadata:\n"
            f"{meta_path}"
        )

    return (
        float(fs),
        int(channels),
        int(epochs),
        int(points)
    )


# ============================================================
# FIND META FILE
# ============================================================

def find_meta_file(
        csv_path,
        data_dir
):

    base = os.path.splitext(
        os.path.basename(csv_path)
    )[0]

    candidates = [

        os.path.join(
            data_dir,
            base + "_meta.txt"
        ),

        os.path.join(
            data_dir,
            base.replace(
                "_pp",
                ""
            ) + "_meta.txt"
        ),

        os.path.join(
            data_dir,
            base + "_pp_meta.txt"
        )
    ]

    for path in candidates:

        if os.path.exists(path):

            return path

    raise FileNotFoundError(
        f"No metadata file found for:\n"
        f"{csv_path}"
    )


# ============================================================
# LABEL PARSING
# ============================================================

def normalize_token(text):

    text = text.strip().lower()

    return re.sub(
        r"[^a-z]",
        "",
        text
    )


def get_label(
        filename_base,
        language
):

    parts = filename_base.split("_")

    if len(parts) < 2:

        raise ValueError(
            f"Cannot parse label from:\n"
            f"{filename_base}"
        )

    token = normalize_token(
        parts[1]
    )

    if language == "English":

        mapping = ENG_MAP

    else:

        mapping = BN_MAP

    if token not in mapping:

        raise ValueError(
            f"Unknown {language} label "
            f"'{token}' in file:\n"
            f"{filename_base}"
        )

    label_name = mapping[token]

    label_id = LABEL_TO_ID[
        label_name
    ]

    return (
        label_id,
        label_name
    )


# ============================================================
# LOAD ONE FILE
# ============================================================

def load_file(
        csv_path,
        data_dir,
        language
):

    base = os.path.splitext(
        os.path.basename(csv_path)
    )[0]

    meta_path = find_meta_file(
        csv_path,
        data_dir
    )

    (
        fs,
        n_channels,
        meta_epochs,
        points_per_epoch
    ) = parse_meta(
        meta_path
    )

    if n_channels != N_CHANNELS:

        raise ValueError(
            f"{base}: expected "
            f"{N_CHANNELS} channels, "
            f"found {n_channels}"
        )

    if (
        points_per_epoch
        != EXPECTED_POINTS_PER_EPOCH
    ):

        raise ValueError(
            f"{base}: expected "
            f"{EXPECTED_POINTS_PER_EPOCH} "
            f"points per epoch, found "
            f"{points_per_epoch}"
        )

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    data = np.loadtxt(
        csv_path,
        delimiter=",",
        dtype=np.float32
    )

    if data.ndim == 1:

        data = data.reshape(
            -1,
            n_channels
        )

    data = data[
        :,
        :n_channels
    ]

    # --------------------------------------------------------
    # Determine actual epoch count from CSV itself
    # --------------------------------------------------------

    total_rows = data.shape[0]

    if (
        total_rows
        % points_per_epoch
        != 0
    ):

        raise ValueError(
            f"{base}: row count "
            f"{total_rows} cannot be "
            f"divided into epochs of "
            f"{points_per_epoch} samples."
        )

    actual_epochs = (
        total_rows
        // points_per_epoch
    )

    # --------------------------------------------------------
    # Metadata mismatch is only a warning
    # --------------------------------------------------------

    if actual_epochs != meta_epochs:

        warnings.warn(
            f"{base}: metadata says "
            f"{meta_epochs} epochs, "
            f"CSV contains "
            f"{actual_epochs}. "
            f"Using CSV count."
        )

    if actual_epochs != 4:

        print(
            f"NOTICE: {base} has "
            f"{actual_epochs} valid epochs."
        )

    # --------------------------------------------------------
    # Subject + class
    # --------------------------------------------------------

    subject = base.split("_")[0]

    (
        label_id,
        label_name
    ) = get_label(
        base,
        language
    )

    # --------------------------------------------------------
    # Reconstruct EEG epochs
    # Keep same arrangement as previous pipeline
    # --------------------------------------------------------

    epochs = np.zeros(
        (
            actual_epochs,
            points_per_epoch,
            n_channels
        ),
        dtype=np.float32
    )

    for ch in range(
        n_channels
    ):

        epochs[:, :, ch] = (
            data[:, ch]
            .reshape(
                points_per_epoch,
                actual_epochs
            )
            .T
        )

    return {

        "subject":
            subject,

        "file_id":
            base,

        "label_id":
            label_id,

        "label_name":
            label_name,

        "epochs":
            epochs,

        "fs":
            fs,

        "epoch_count":
            actual_epochs
    }


# ============================================================
# LOAD ENTIRE DATASET
# ============================================================

def load_dataset(
        data_dir,
        language
):

    csv_files = sorted(
        glob(
            os.path.join(
                data_dir,
                "*_pp.csv"
            )
        )
    )

    if not csv_files:

        raise FileNotFoundError(
            f"No *_pp.csv files found:\n"
            f"{data_dir}"
        )

    items = []

    print(
        "\nLoading",
        language,
        "dataset..."
    )

    for path in csv_files:

        item = load_file(
            path,
            data_dir,
            language
        )

        items.append(
            item
        )

    return items


# ============================================================
# DATASET REPORT
# ============================================================

def print_dataset_report(
        items,
        language
):

    by_subject = defaultdict(
        list
    )

    for item in items:

        by_subject[
            item["subject"]
        ].append(
            item
        )

    print(
        "\n" + "=" * 90
    )

    print(
        f"{language.upper()} "
        f"DATASET REPORT"
    )

    print(
        "=" * 90
    )

    print(
        "Subjects:",
        len(by_subject)
    )

    print(
        "Command files:",
        len(items)
    )

    total_epochs = sum(
        x["epoch_count"]
        for x in items
    )

    print(
        "Total available epochs:",
        total_epochs
    )

    print(
        "Total potential windows:",
        total_epochs * 6
    )

    print(
        "\nAvailable directions "
        "per subject:"
    )

    report_rows = []

    for subject, files in sorted(
        by_subject.items()
    ):

        directions = sorted(
            {
                item["label_name"]
                for item in files
            }
        )

        missing = sorted(
            set(
                CLASS_NAMES.values()
            )
            - set(directions)
        )

        n_epochs = sum(
            x["epoch_count"]
            for x in files
        )

        print(
            f"{subject:<18} "
            f"files={len(files):<2} "
            f"epochs={n_epochs:<3} "
            f"available={directions} "
            f"missing={missing}"
        )

        report_rows.append({

            "Language":
                language,

            "Subject":
                subject,

            "N_files":
                len(files),

            "N_epochs":
                n_epochs,

            "Available_directions":
                ", ".join(
                    directions
                ),

            "Missing_directions":
                ", ".join(
                    missing
                )
        })

    report_df = pd.DataFrame(
        report_rows
    )

    report_df.to_csv(
        os.path.join(
            OUT_DIR,
            f"{language}_dataset_availability.csv"
        ),
        index=False
    )


# ============================================================
# WINDOWING
# ============================================================

def windowize(
        epoch_TxC,
        fs
):

    win = int(
        WIN_SEC * fs
    )

    step = int(
        win * (
            1.0 - OVERLAP
        )
    )

    if step < 1:

        step = 1

    T, C = epoch_TxC.shape

    windows = []

    for start in range(
        0,
        T - win + 1,
        step
    ):

        segment = (
            epoch_TxC[
                start:
                start + win,
                :
            ]
            .copy()
        )

        if DO_Z:

            mean = segment.mean(
                axis=0,
                keepdims=True
            )

            std = segment.std(
                axis=0,
                keepdims=True
            )

            segment = (
                segment - mean
            ) / (
                std + 1e-6
            )

        # pyRiemann expects
        # channels x samples
        windows.append(
            segment.T
        )

    return windows


# ============================================================
# CREATE TRAIN/TEST FOR ONE SUBJECT + ONE EPOCH INDEX
# ============================================================

def create_subject_fold(
        subject_files,
        test_epoch
):

    X_train = []
    y_train = []

    X_test = []
    y_test = []

    meta_test = []

    # --------------------------------------------------------
    # Use every available file.
    # There is NO six-file requirement.
    # --------------------------------------------------------

    for item in subject_files:

        E = item["epoch_count"]

        # --------------------------------------------
        # Use every existing epoch in this file
        # --------------------------------------------

        for epoch_id in range(E):

            windows = windowize(
                item["epochs"][
                    epoch_id
                ],
                item["fs"]
            )

            if len(windows) == 0:

                continue

            # ----------------------------------------
            # Test only if this epoch index exists
            # ----------------------------------------

            is_test = (
                epoch_id
                == test_epoch
            )

            for window_id, window in enumerate(
                windows
            ):

                if is_test:

                    X_test.append(
                        window
                    )

                    y_test.append(
                        item[
                            "label_id"
                        ]
                    )

                    meta_test.append({

                        "subject":
                            item[
                                "subject"
                            ],

                        "file_id":
                            item[
                                "file_id"
                            ],

                        "epoch_id":
                            epoch_id,

                        "window_id":
                            window_id,

                        "label_id":
                            item[
                                "label_id"
                            ],

                        "label_name":
                            item[
                                "label_name"
                            ]
                    })

                else:

                    X_train.append(
                        window
                    )

                    y_train.append(
                        item[
                            "label_id"
                        ]
                    )

    # --------------------------------------------------------
    # Nothing to train or test
    # --------------------------------------------------------

    if (
        len(X_train) == 0
        or
        len(X_test) == 0
    ):

        return None

    X_train = np.asarray(
        X_train,
        dtype=np.float64
    )

    y_train = np.asarray(
        y_train,
        dtype=int
    )

    X_test = np.asarray(
        X_test,
        dtype=np.float64
    )

    y_test = np.asarray(
        y_test,
        dtype=int
    )

    meta_test = pd.DataFrame(
        meta_test
    )

    # --------------------------------------------------------
    # Need at least two classes to fit SVM
    # --------------------------------------------------------

    training_classes = set(
        np.unique(
            y_train
        )
    )

    if len(
        training_classes
    ) < 2:

        return None

    # --------------------------------------------------------
    # A class can be impossible to predict if it has
    # no training epochs.
    #
    # Example:
    # a direction only has one epoch and that epoch
    # is currently held out.
    #
    # Only those impossible test windows are removed.
    # --------------------------------------------------------

    valid_test_mask = np.array(
        [
            label
            in training_classes
            for label in y_test
        ],
        dtype=bool
    )

    if not np.all(
        valid_test_mask
    ):

        removed_labels = sorted(
            set(
                y_test[
                    ~valid_test_mask
                ]
            )
        )

        print(
            "    Skipping test data "
            "for classes absent from "
            "training:",
            [
                CLASS_NAMES[x]
                for x
                in removed_labels
            ]
        )

    X_test = X_test[
        valid_test_mask
    ]

    y_test = y_test[
        valid_test_mask
    ]

    meta_test = (
        meta_test.loc[
            valid_test_mask
        ]
        .reset_index(
            drop=True
        )
    )

    if len(
        X_test
    ) == 0:

        return None

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        meta_test
    )


# ============================================================
# SYMMETRIC VECTORIZATION
# ============================================================

def symmetric_vectorize(
        matrices
):

    n, C, _ = matrices.shape

    rows, cols = (
        np.triu_indices(
            C
        )
    )

    n_features = (
        C * (
            C + 1
        ) // 2
    )

    features = np.zeros(
        (
            n,
            n_features
        ),
        dtype=np.float64
    )

    for i, matrix in enumerate(
        matrices
    ):

        vector = matrix[
            rows,
            cols
        ].copy()

        off_diag = (
            rows != cols
        )

        vector[
            off_diag
        ] *= np.sqrt(
            2.0
        )

        features[i] = vector

    return features


# ============================================================
# SAMPLE COVARIANCE
# ============================================================

def sample_cov_features(
        X_train,
        X_test
):

    train_cov = np.asarray(
        [
            np.cov(
                x,
                bias=False
            )
            for x in X_train
        ]
    )

    test_cov = np.asarray(
        [
            np.cov(
                x,
                bias=False
            )
            for x in X_test
        ]
    )

    return (
        symmetric_vectorize(
            train_cov
        ),
        symmetric_vectorize(
            test_cov
        )
    )


# ============================================================
# OAS EUCLIDEAN COVARIANCE
# ============================================================

def oas_cov_features(
        X_train,
        X_test
):

    def calculate_oas(x):

        estimator = OAS(
            assume_centered=True
        )

        # sklearn:
        # samples x features
        estimator.fit(
            x.T
        )

        return (
            estimator.covariance_
        )

    train_cov = np.asarray(
        [
            calculate_oas(x)
            for x in X_train
        ]
    )

    test_cov = np.asarray(
        [
            calculate_oas(x)
            for x in X_test
        ]
    )

    return (
        symmetric_vectorize(
            train_cov
        ),
        symmetric_vectorize(
            test_cov
        )
    )


# ============================================================
# RIEMANN TANGENT FEATURES
# ============================================================

def riemann_features(
        X_train,
        X_test
):

    covariance = Covariances(
        estimator="oas"
    )

    tangent = TangentSpace(
        metric="riemann"
    )

    train_cov = (
        covariance.fit_transform(
            X_train
        )
    )

    test_cov = (
        covariance.transform(
            X_test
        )
    )

    train_ts = (
        tangent.fit_transform(
            train_cov
        )
    )

    test_ts = (
        tangent.transform(
            test_cov
        )
    )

    return (
        train_ts,
        test_ts
    )


# ============================================================
# FEATURE PIPELINE
# ============================================================

def build_features(
        method,
        X_train,
        y_train,
        X_test
):

    # --------------------------------------------------------
    # Method 1
    # --------------------------------------------------------

    if method == METHODS[0]:

        (
            F_train,
            F_test
        ) = sample_cov_features(
            X_train,
            X_test
        )

    # --------------------------------------------------------
    # Method 2
    # --------------------------------------------------------

    elif method == METHODS[1]:

        (
            F_train,
            F_test
        ) = oas_cov_features(
            X_train,
            X_test
        )

    # --------------------------------------------------------
    # Methods 3 and 4
    # --------------------------------------------------------

    elif method in (
        METHODS[2],
        METHODS[3]
    ):

        (
            F_train,
            F_test
        ) = riemann_features(
            X_train,
            X_test
        )

        # ----------------------------------------------------
        # Controlled 52-D ablation
        # Feature selection uses TRAINING data only.
        # ----------------------------------------------------

        if method == METHODS[2]:

            n_available = (
                F_train.shape[1]
            )

            k = min(
                52,
                n_available
            )

            selector = SelectKBest(
                score_func=f_classif,
                k=k
            )

            with warnings.catch_warnings():

                warnings.simplefilter(
                    "ignore"
                )

                F_train = (
                    selector.fit_transform(
                        F_train,
                        y_train
                    )
                )

            F_test = (
                selector.transform(
                    F_test
                )
            )

    else:

        raise ValueError(
            f"Unknown method: "
            f"{method}"
        )

    # --------------------------------------------------------
    # Feature scaling fitted TRAIN ONLY
    # --------------------------------------------------------

    scaler = StandardScaler()

    F_train = (
        scaler.fit_transform(
            F_train
        )
    )

    F_test = (
        scaler.transform(
            F_test
        )
    )

    return (
        F_train,
        F_test
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def classify(
        F_train,
        y_train,
        F_test
):

    unique_classes = np.unique(
        y_train
    )

    if len(
        unique_classes
    ) < 2:

        return None

    model = SVC(

        kernel="rbf",

        C=SVM_C,

        gamma=SVM_GAMMA,

        decision_function_shape="ovr"
    )

    model.fit(
        F_train,
        y_train
    )

    return model.predict(
        F_test
    )


# ============================================================
# MAJORITY VOTE
# ============================================================

def majority_vote(
        predictions
):

    counts = Counter(
        int(x)
        for x in predictions
    )

    highest = max(
        counts.values()
    )

    winners = [
        cls
        for cls, count
        in counts.items()
        if count == highest
    ]

    # deterministic tie rule
    return min(
        winners
    )


def make_epoch_predictions(
        meta_test,
        window_predictions
):

    df = meta_test.copy()

    df[
        "pred_label"
    ] = window_predictions

    rows = []

    for (
        subject,
        file_id,
        epoch_id
    ), group in df.groupby(
        [
            "subject",
            "file_id",
            "epoch_id"
        ],
        sort=True
    ):

        true_label = int(
            group[
                "label_id"
            ].iloc[0]
        )

        predicted_label = (
            majority_vote(
                group[
                    "pred_label"
                ].values
            )
        )

        rows.append({

            "subject":
                subject,

            "file_id":
                file_id,

            "epoch_id":
                epoch_id,

            "true_label":
                true_label,

            "pred_label":
                predicted_label
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
        y_true,
        y_pred
):

    # --------------------------------------------------------
    # Use all classes that actually occur in the pooled data
    # --------------------------------------------------------

    labels_present = sorted(
        set(
            list(y_true)
            +
            list(y_pred)
        )
    )

    return {

        "Accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "Precision":
            precision_score(
                y_true,
                y_pred,
                labels=labels_present,
                average="macro",
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_true,
                y_pred,
                labels=labels_present,
                average="macro",
                zero_division=0
            ),

        "F1":
            f1_score(
                y_true,
                y_pred,
                labels=labels_present,
                average="macro",
                zero_division=0
            )
    }


# ============================================================
# EVALUATE DATASET
# ============================================================

def evaluate_dataset(
        items,
        language
):

    by_subject = defaultdict(
        list
    )

    for item in items:

        by_subject[
            item[
                "subject"
            ]
        ].append(
            item
        )

    subjects = sorted(
        by_subject.keys()
    )

    # --------------------------------------------------------
    # Determine maximum epoch count found in dataset.
    # Usually 4.
    # --------------------------------------------------------

    max_epoch_count = max(
        item[
            "epoch_count"
        ]
        for item in items
    )

    summary_rows = []
    fold_rows = []
    prediction_rows = []

    for method in METHODS:

        print(
            "\n" +
            "#" * 100
        )

        print(
            f"{language} | "
            f"{method}"
        )

        print(
            "#" * 100
        )

        pooled_true = []
        pooled_pred = []

        # ====================================================
        # Hold out epoch index 0, 1, 2, 3...
        # ====================================================

        for test_epoch in range(
            max_epoch_count
        ):

            print(
                f"\nFold "
                f"{test_epoch + 1}/"
                f"{max_epoch_count}: "
                f"holding epoch index "
                f"{test_epoch}"
            )

            fold_true = []
            fold_pred = []

            subjects_used = 0

            # =================================================
            # Every subject is allowed.
            # No 6-direction requirement.
            # =================================================

            for subject in subjects:

                subject_files = (
                    by_subject[
                        subject
                    ]
                )

                available_directions = sorted(
                    {
                        x["label_name"]
                        for x in
                        subject_files
                    }
                )

                # ---------------------------------------------
                # Run with whatever files exist
                # ---------------------------------------------

                split = create_subject_fold(
                    subject_files,
                    test_epoch
                )

                if split is None:

                    print(
                        f"  {subject}: "
                        f"no usable test data "
                        f"for this fold. "
                        f"Skipping fold only."
                    )

                    continue

                (
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    meta_test
                ) = split

                # ---------------------------------------------
                # At least 2 classes required
                # ---------------------------------------------

                if (
                    len(
                        np.unique(
                            y_train
                        )
                    )
                    < 2
                ):

                    print(
                        f"  {subject}: "
                        f"less than two "
                        f"training classes. "
                        f"Skipping this fold."
                    )

                    continue

                try:

                    (
                        F_train,
                        F_test
                    ) = build_features(
                        method,
                        X_train,
                        y_train,
                        X_test
                    )

                except Exception as e:

                    print(
                        f"  {subject}: "
                        f"feature extraction "
                        f"failed ({e}). "
                        f"Skipping this fold."
                    )

                    continue

                predictions = classify(
                    F_train,
                    y_train,
                    F_test
                )

                if predictions is None:

                    continue

                epoch_df = (
                    make_epoch_predictions(
                        meta_test,
                        predictions
                    )
                )

                if epoch_df.empty:

                    continue

                subjects_used += 1

                epoch_df[
                    "language"
                ] = language

                epoch_df[
                    "method"
                ] = method

                epoch_df[
                    "fold"
                ] = (
                    test_epoch + 1
                )

                epoch_df[
                    "available_directions"
                ] = (
                    ", ".join(
                        available_directions
                    )
                )

                prediction_rows.extend(
                    epoch_df.to_dict(
                        orient="records"
                    )
                )

                fold_true.extend(
                    epoch_df[
                        "true_label"
                    ].tolist()
                )

                fold_pred.extend(
                    epoch_df[
                        "pred_label"
                    ].tolist()
                )

            # =================================================
            # No valid test predictions
            # =================================================

            if len(
                fold_true
            ) == 0:

                print(
                    "  No usable epochs "
                    "in this fold."
                )

                continue

            # =================================================
            # Fold metrics
            # =================================================

            metrics = calculate_metrics(
                fold_true,
                fold_pred
            )

            fold_rows.append({

                "Language":
                    language,

                "Approach":
                    method,

                "Fold":
                    test_epoch + 1,

                "Subjects_used":
                    subjects_used,

                "N_test_epochs":
                    len(
                        fold_true
                    ),

                "Accuracy":
                    metrics[
                        "Accuracy"
                    ],

                "Precision":
                    metrics[
                        "Precision"
                    ],

                "Recall":
                    metrics[
                        "Recall"
                    ],

                "F1":
                    metrics[
                        "F1"
                    ]
            })

            pooled_true.extend(
                fold_true
            )

            pooled_pred.extend(
                fold_pred
            )

            print(
                f"  Subjects used: "
                f"{subjects_used}"
            )

            print(
                f"  Test epochs  : "
                f"{len(fold_true)}"
            )

            print(
                f"  Accuracy     : "
                f"{metrics['Accuracy'] * 100:.2f}%"
            )

            print(
                f"  Precision    : "
                f"{metrics['Precision'] * 100:.2f}%"
            )

            print(
                f"  Recall       : "
                f"{metrics['Recall'] * 100:.2f}%"
            )

            print(
                f"  F1           : "
                f"{metrics['F1'] * 100:.2f}%"
            )

        # ====================================================
        # OVERALL POOLED RESULTS
        # ====================================================

        if len(
            pooled_true
        ) == 0:

            print(
                f"No valid data for "
                f"{language} | "
                f"{method}"
            )

            continue

        overall = calculate_metrics(
            pooled_true,
            pooled_pred
        )

        current_folds = pd.DataFrame(
            [
                row
                for row in fold_rows
                if (
                    row[
                        "Language"
                    ] == language
                    and
                    row[
                        "Approach"
                    ] == method
                )
            ]
        )

        if (
            "52-D"
            in method
        ):

            feature_dim = 52

        else:

            feature_dim = 78

        summary_rows.append({

            "Language":
                language,

            "Approach":
                method,

            "Feature_dim":
                feature_dim,

            "Subjects_available":
                len(
                    subjects
                ),

            "Accuracy":
                overall[
                    "Accuracy"
                ],

            "Precision":
                overall[
                    "Precision"
                ],

            "Recall":
                overall[
                    "Recall"
                ],

            "F1":
                overall[
                    "F1"
                ],

            "Accuracy_SD":
                current_folds[
                    "Accuracy"
                ].std(
                    ddof=1
                ),

            "F1_SD":
                current_folds[
                    "F1"
                ].std(
                    ddof=1
                ),

            "N_test_epochs":
                len(
                    pooled_true
                )
        })

        print(
            "\nOVERALL RESULT"
        )

        print(
            f"Available subjects: "
            f"{len(subjects)}"
        )

        print(
            f"Evaluated epochs  : "
            f"{len(pooled_true)}"
        )

        print(
            f"Accuracy          : "
            f"{overall['Accuracy'] * 100:.2f}%"
        )

        print(
            f"Precision         : "
            f"{overall['Precision'] * 100:.2f}%"
        )

        print(
            f"Recall            : "
            f"{overall['Recall'] * 100:.2f}%"
        )

        print(
            f"F1-score          : "
            f"{overall['F1'] * 100:.2f}%"
        )

        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        cm = confusion_matrix(
            pooled_true,
            pooled_pred,
            labels=CLASS_IDS
        )

        cm_df = pd.DataFrame(
            cm,
            index=[
                CLASS_NAMES[x]
                for x
                in CLASS_IDS
            ],
            columns=[
                CLASS_NAMES[x]
                for x
                in CLASS_IDS
            ]
        )

        safe_method = (
            method
            .replace(
                " ",
                "_"
            )
            .replace(
                "+",
                "plus"
            )
            .replace(
                ",",
                ""
            )
        )

        cm_df.to_csv(
            os.path.join(
                OUT_DIR,
                f"{language}_"
                f"{safe_method}_"
                f"confusion_matrix.csv"
            )
        )

    return (
        pd.DataFrame(
            summary_rows
        ),
        pd.DataFrame(
            fold_rows
        ),
        pd.DataFrame(
            prediction_rows
        )
    )


# ============================================================
# PUBLICATION TABLE
# ============================================================

def make_publication_table(
        dataframe
):

    table = (
        dataframe.copy()
    )

    percentage_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Accuracy_SD",
        "F1_SD"
    ]

    for column in percentage_columns:

        table[
            column
        ] = (
            table[
                column
            ]
            * 100
        ).round(
            2
        )

    return table[
        [
            "Language",
            "Approach",
            "Feature_dim",
            "Subjects_available",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Accuracy_SD",
            "F1_SD",
            "N_test_epochs"
        ]
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUT_DIR,
        exist_ok=True
    )

    print(
        "=" * 90
    )

    print(
        "FINAL EEG ABLATION STUDY"
    )

    print(
        "=" * 90
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    eng_items = load_dataset(
        ENG_DIR,
        "English"
    )

    bn_items = load_dataset(
        BN_DIR,
        "Bangla"
    )

    # --------------------------------------------------------
    # Availability reports
    # --------------------------------------------------------

    print_dataset_report(
        eng_items,
        "English"
    )

    print_dataset_report(
        bn_items,
        "Bangla"
    )

    # --------------------------------------------------------
    # Run English
    # --------------------------------------------------------

    (
        eng_summary,
        eng_folds,
        eng_predictions
    ) = evaluate_dataset(
        eng_items,
        "English"
    )

    # --------------------------------------------------------
    # Run Bangla
    # --------------------------------------------------------

    (
        bn_summary,
        bn_folds,
        bn_predictions
    ) = evaluate_dataset(
        bn_items,
        "Bangla"
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    summary = pd.concat(
        [
            eng_summary,
            bn_summary
        ],
        ignore_index=True
    )

    fold_metrics = pd.concat(
        [
            eng_folds,
            bn_folds
        ],
        ignore_index=True
    )

    predictions = pd.concat(
        [
            eng_predictions,
            bn_predictions
        ],
        ignore_index=True
    )

    publication_table = (
        make_publication_table(
            summary
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    summary.to_csv(
        os.path.join(
            OUT_DIR,
            "ablation_summary_raw.csv"
        ),
        index=False
    )

    fold_metrics.to_csv(
        os.path.join(
            OUT_DIR,
            "ablation_foldwise_metrics.csv"
        ),
        index=False
    )

    predictions.to_csv(
        os.path.join(
            OUT_DIR,
            "ablation_epoch_predictions.csv"
        ),
        index=False
    )

    publication_table.to_csv(
        os.path.join(
            OUT_DIR,
            "ablation_PUBLICATION_TABLE.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Final console output
    # --------------------------------------------------------

    print(
        "\n" +
        "=" * 120
    )

    print(
        "FINAL PUBLICATION TABLE"
    )

    print(
        "=" * 120
    )

    print(
        publication_table.to_string(
            index=False
        )
    )

    print(
        "\nSaved results:"
    )

    print(
        OUT_DIR
    )

    print(
        "\nMain table:"
    )

    print(
        os.path.join(
            OUT_DIR,
            "ablation_PUBLICATION_TABLE.csv"
        )
    )


if __name__ == "__main__":

    main()