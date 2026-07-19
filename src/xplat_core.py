# src/xplat_core.py

import os
import time
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import warnings

warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)


def calculate_bland_altman(errors: np.ndarray):
    if len(errors) < 2 or np.std(errors) == 0:
        return 0.0, 0.0
    bias = float(np.mean(errors))
    sd = float(np.std(errors))
    loa_width = 1.96 * sd * 2.0
    return bias, loa_width


def get_correlations(arr1: np.ndarray, arr2: np.ndarray):
    if len(arr1) < 2 or np.std(arr1) == 0 or np.std(arr2) == 0:
        return 0.0, 0.0
    p, _ = pearsonr(arr1, arr2)
    s, _ = spearmanr(arr1, arr2)
    return float(p), float(s)


def _make_cv_with_groups(df: pd.DataFrame, n_splits: int, random_seed: int,
                         pair_ids: np.ndarray = None):
    """
    Helper to construct either GroupKFold (when pair_ids provided)
    or standard KFold (when pair_ids is None).
    Assumes pair_ids is one group label per row in df, in the same order
    as the per-gene .txt files.
    """
    if pair_ids is not None:
        if len(pair_ids) != len(df.index):
            # one pair_id per row is required
            return None, None
        groups = pair_ids
        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            return None, None
        n_splits_effective = min(n_splits, len(unique_groups))
        cv = GroupKFold(n_splits=n_splits_effective)
        return cv, groups
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        return cv, None


def process_single_gene(
    gene_file: str,
    input_dir: str,
    n_splits: int,
    random_seed: int,
    pair_ids: np.ndarray = None,
):
    try:
        gene_path = os.path.join(input_dir, gene_file)
        df = pd.read_csv(gene_path, header=None, sep="\t")
        df.columns = ["x", "y"]
        gene_name = os.path.splitext(gene_file)[0]

        df["x_log"] = np.log1p(pd.to_numeric(df["x"], errors="coerce"))
        df["y_log"] = np.log1p(pd.to_numeric(df["y"], errors="coerce"))
        df.dropna(subset=["x_log", "y_log"], inplace=True)

        if df.empty:
            return None

        # Construct CV splitter (GroupKFold if pair_ids provided)
        cv, groups = _make_cv_with_groups(df, n_splits, random_seed, pair_ids)
        if cv is None:
            print(f"Warning: cannot construct CV for gene {gene_name} (groups issue). Skipping.")
            return None

        range_x_log = df["x_log"].max() - df["x_log"].min() if not df["x_log"].empty else 0.0
        range_y_log = df["y_log"].max() - df["y_log"].min() if not df["y_log"].empty else 0.0
        range_x_raw = df["x"].max() - df["x"].min() if not df["x"].empty else 0.0
        range_y_raw = df["y"].max() - df["y"].min() if not df["y"].empty else 0.0

        agg_metrics = {
            "rmse_x_log": [], "rmse_y_log": [],
            "rmse_x_raw": [], "rmse_y_raw": [],
            "mae_x_log": [], "mae_y_log": [],
            "nrmse_x_log": [], "nrmse_y_log": [],
            "p_xy": [], "s_xy": [],
            "p_xp": [], "s_xp": [],
            "p_py": [], "s_py": [],
            "p_pp": [], "s_pp": [],
            "bias_x": [], "loa_width_x": [],
            "bias_y": [], "loa_width_y": [],
        }
        fold_rows = []

        poly = PolynomialFeatures(2)
        splitter = cv.split(df, groups=groups) if groups is not None else cv.split(df)

        for fold_idx, (train_idx, test_idx) in enumerate(splitter):
            train = df.iloc[train_idx]
            test = df.iloc[test_idx]

            if len(train) == 0 or len(test) == 0:
                continue

            # array -> seq (x_log -> y_log)
            if np.std(train["x_log"]) == 0:
                pred_y_log = np.full_like(
                    test["y_log"],
                    train["y_log"].mean() if not train["y_log"].empty else 0.0,
                )
            else:
                m_y = LinearRegression().fit(
                    poly.fit_transform(train[["x_log"]]), train["y_log"]
                )
                pred_y_log = m_y.predict(poly.fit_transform(test[["x_log"]]))

            # seq -> array (y_log -> x_log)
            if np.std(train["y_log"]) == 0:
                pred_x_log = np.full_like(
                    test["x_log"],
                    train["x_log"].mean() if not train["x_log"].empty else 0.0,
                )
            else:
                m_x = LinearRegression().fit(
                    poly.fit_transform(train[["y_log"]]), train["x_log"]
                )
                pred_x_log = m_x.predict(poly.fit_transform(test[["y_log"]]))

            err_y = pred_y_log - test["y_log"].to_numpy()
            err_x = pred_x_log - test["x_log"].to_numpy()

            f_mae_y = mean_absolute_error(test["y_log"], pred_y_log)
            f_rmse_y = np.sqrt(mean_squared_error(test["y_log"], pred_y_log))
            f_mae_x = mean_absolute_error(test["x_log"], pred_x_log)
            f_rmse_x = np.sqrt(mean_squared_error(test["x_log"], pred_x_log))

            rmse_y_raw = np.sqrt(
                np.mean((test["y"].to_numpy() - np.expm1(pred_y_log)) ** 2)
            ) if len(test["y"]) > 0 else 0.0
            rmse_x_raw = np.sqrt(
                np.mean((test["x"].to_numpy() - np.expm1(pred_x_log)) ** 2)
            ) if len(test["x"]) > 0 else 0.0

            nrmse_x_fold = (f_rmse_x / range_x_log) if range_x_log > 0 else 0.0
            nrmse_y_fold = (f_rmse_y / range_y_log) if range_y_log > 0 else 0.0

            p_xy, s_xy = get_correlations(
                test["x_log"].to_numpy(), test["y_log"].to_numpy()
            )
            p_xp, s_xp = get_correlations(
                test["x_log"].to_numpy(), pred_y_log
            )
            p_py, s_py = get_correlations(
                pred_x_log, test["y_log"].to_numpy()
            )
            p_pp, s_pp = get_correlations(pred_x_log, pred_y_log)

            bias_x_fold, loa_width_x_fold = calculate_bland_altman(err_x)
            bias_y_fold, loa_width_y_fold = calculate_bland_altman(err_y)

            agg_metrics["rmse_y_log"].append(f_rmse_y)
            agg_metrics["rmse_x_log"].append(f_rmse_x)
            agg_metrics["mae_y_log"].append(f_mae_y)
            agg_metrics["mae_x_log"].append(f_mae_x)
            agg_metrics["rmse_y_raw"].append(rmse_y_raw)
            agg_metrics["rmse_x_raw"].append(rmse_x_raw)
            agg_metrics["nrmse_x_log"].append(nrmse_x_fold)
            agg_metrics["nrmse_y_log"].append(nrmse_y_fold)
            agg_metrics["p_xy"].append(p_xy)
            agg_metrics["s_xy"].append(s_xy)
            agg_metrics["p_xp"].append(p_xp)
            agg_metrics["s_xp"].append(s_xp)
            agg_metrics["p_py"].append(p_py)
            agg_metrics["s_py"].append(s_py)
            agg_metrics["p_pp"].append(p_pp)
            agg_metrics["s_pp"].append(s_pp)
            agg_metrics["bias_x"].append(bias_x_fold)
            agg_metrics["loa_width_x"].append(loa_width_x_fold)
            agg_metrics["bias_y"].append(bias_y_fold)
            agg_metrics["loa_width_y"].append(loa_width_y_fold)

            sample_data = []
            for i_sample in range(len(test)):
                sample_data.extend([
                    float(test["x_log"].iloc[i_sample]),
                    float(pred_x_log[i_sample]),
                    float(err_x[i_sample]),
                    float(test["y_log"].iloc[i_sample]),
                    float(pred_y_log[i_sample]),
                    float(err_y[i_sample]),
                ])

            fold_metrics_str = (
                f"\t{f_mae_x}\t{f_rmse_x}\t{f_mae_y}\t{f_rmse_y}"
                f"\t{p_xy}\t{s_xy}\t{p_xp}\t{s_xp}\t{p_py}\t{s_py}\t{p_pp}\t{s_pp}"
                f"\t{bias_x_fold}\t{loa_width_x_fold}\t{bias_y_fold}\t{loa_width_y_fold}"
                f"\t{nrmse_x_fold}\t{nrmse_y_fold}"
            )
            line_str = f"{gene_name}\t" + "\t".join(map(str, sample_data)) + fold_metrics_str + "\n"
            fold_rows.append((fold_idx + 1, line_str))

        if len(agg_metrics["rmse_x_log"]) == 0:
            return None

        avg_rmse_x = float(np.mean(agg_metrics["rmse_x_log"]))
        avg_rmse_y = float(np.mean(agg_metrics["rmse_y_log"]))
        avg_mae_x = float(np.mean(agg_metrics["mae_x_log"]))
        avg_mae_y = float(np.mean(agg_metrics["mae_y_log"]))
        avg_rmse_x_raw = float(np.mean(agg_metrics["rmse_x_raw"]))
        avg_rmse_y_raw = float(np.mean(agg_metrics["rmse_y_raw"]))
        avg_nrmse_x = float(np.mean(agg_metrics["nrmse_x_log"]))
        avg_nrmse_y = float(np.mean(agg_metrics["nrmse_y_log"]))

        avg_p_xy = float(np.mean(agg_metrics["p_xy"]))
        avg_s_xy = float(np.mean(agg_metrics["s_xy"]))
        avg_p_xp = float(np.mean(agg_metrics["p_xp"]))
        avg_s_xp = float(np.mean(agg_metrics["s_xp"]))
        avg_p_py = float(np.mean(agg_metrics["p_py"]))
        avg_s_py = float(np.mean(agg_metrics["s_py"]))
        avg_p_pp = float(np.mean(agg_metrics["p_pp"]))
        avg_s_pp = float(np.mean(agg_metrics["s_pp"]))

        avg_bias_x = float(np.mean(agg_metrics["bias_x"]))
        avg_loa_width_x = float(np.mean(agg_metrics["loa_width_x"]))
        avg_bias_y = float(np.mean(agg_metrics["bias_y"]))
        avg_loa_width_y = float(np.mean(agg_metrics["loa_width_y"]))

        summary_line_content = [
            gene_name, avg_rmse_x, avg_rmse_y, avg_mae_x, avg_mae_y,
            avg_rmse_x_raw, avg_rmse_y_raw,
            range_x_raw, range_y_raw, range_x_log, range_y_log,
            avg_nrmse_x, avg_nrmse_y,
            avg_p_xy, avg_s_xy, avg_p_xp, avg_s_xp, avg_p_py, avg_s_py, avg_p_pp, avg_s_pp,
            avg_bias_x, avg_loa_width_x, avg_bias_y, avg_loa_width_y,
        ]
        summary_line = "\t".join(map(str, summary_line_content)) + "\n"

        return summary_line, fold_rows

    except Exception as e:
        print(f"Error processing gene {gene_file}: {e}")
        return None


def process_single_gene_linear(
    gene_file: str,
    input_dir: str,
    n_splits: int,
    random_seed: int,
    pair_ids: np.ndarray = None,
):
    try:
        gene_path = os.path.join(input_dir, gene_file)
        df = pd.read_csv(gene_path, header=None, sep="\t")
        df.columns = ["x", "y"]
        gene_name = os.path.splitext(gene_file)[0]

        df["x_log"] = np.log1p(pd.to_numeric(df["x"], errors="coerce"))
        df["y_log"] = np.log1p(pd.to_numeric(df["y"], errors="coerce"))
        df.dropna(subset=["x_log", "y_log"], inplace=True)

        if df.empty:
            return None

        cv, groups = _make_cv_with_groups(df, n_splits, random_seed, pair_ids)
        if cv is None:
            print(f"Warning: cannot construct CV for gene {gene_name} (groups issue). Skipping.")
            return None

        range_x_log = df["x_log"].max() - df["x_log"].min() if not df["x_log"].empty else 0.0
        range_y_log = df["y_log"].max() - df["y_log"].min() if not df["y_log"].empty else 0.0
        range_x_raw = df["x"].max() - df["x"].min() if not df["x"].empty else 0.0
        range_y_raw = df["y"].max() - df["y"].min() if not df["y"].empty else 0.0

        agg_metrics = {
            "rmse_x_log": [], "rmse_y_log": [],
            "rmse_x_raw": [], "rmse_y_raw": [],
            "mae_x_log": [], "mae_y_log": [],
            "nrmse_x_log": [], "nrmse_y_log": [],
            "p_xy": [], "s_xy": [],
            "p_xp": [], "s_xp": [],
            "p_py": [], "s_py": [],
            "p_pp": [], "s_pp": [],
            "bias_x": [], "loa_width_x": [],
            "bias_y": [], "loa_width_y": [],
        }
        fold_rows = []

        splitter = cv.split(df, groups=groups) if groups is not None else cv.split(df)

        for fold_idx, (train_idx, test_idx) in enumerate(splitter):
            train = df.iloc[train_idx]
            test = df.iloc[test_idx]

            if len(train) == 0 or len(test) == 0:
                continue

            # array -> seq (x_log -> y_log)
            if np.std(train["x_log"]) == 0:
                pred_y_log = np.full_like(
                    test["y_log"],
                    train["y_log"].mean() if not train["y_log"].empty else 0.0,
                )
            else:
                m_y = LinearRegression().fit(train[["x_log"]], train["y_log"])
                pred_y_log = m_y.predict(test[["x_log"]])

            # seq -> array (y_log -> x_log)
            if np.std(train["y_log"]) == 0:
                pred_x_log = np.full_like(
                    test["x_log"],
                    train["x_log"].mean() if not train["x_log"].empty else 0.0,
                )
            else:
                m_x = LinearRegression().fit(train[["y_log"]], train["x_log"])
                pred_x_log = m_x.predict(test[["y_log"]])

            err_y = pred_y_log - test["y_log"].to_numpy()
            err_x = pred_x_log - test["x_log"].to_numpy()

            f_mae_y = mean_absolute_error(test["y_log"], pred_y_log)
            f_rmse_y = np.sqrt(mean_squared_error(test["y_log"], pred_y_log))
            f_mae_x = mean_absolute_error(test["x_log"], pred_x_log)
            f_rmse_x = np.sqrt(mean_squared_error(test["x_log"], pred_x_log))

            rmse_y_raw = np.sqrt(
                np.mean((test["y"].to_numpy() - np.expm1(pred_y_log)) ** 2)
            ) if len(test["y"]) > 0 else 0.0
            rmse_x_raw = np.sqrt(
                np.mean((test["x"].to_numpy() - np.expm1(pred_x_log)) ** 2)
            ) if len(test["x"]) > 0 else 0.0

            nrmse_x_fold = (f_rmse_x / range_x_log) if range_x_log > 0 else 0.0
            nrmse_y_fold = (f_rmse_y / range_y_log) if range_y_log > 0 else 0.0

            p_xy, s_xy = get_correlations(
                test["x_log"].to_numpy(), test["y_log"].to_numpy()
            )
            p_xp, s_xp = get_correlations(
                test["x_log"].to_numpy(), pred_y_log
            )
            p_py, s_py = get_correlations(
                pred_x_log, test["y_log"].to_numpy()
            )
            p_pp, s_pp = get_correlations(pred_x_log, pred_y_log)

            bias_x_fold, loa_width_x_fold = calculate_bland_altman(err_x)
            bias_y_fold, loa_width_y_fold = calculate_bland_altman(err_y)

            agg_metrics["rmse_y_log"].append(f_rmse_y)
            agg_metrics["rmse_x_log"].append(f_rmse_x)
            agg_metrics["mae_y_log"].append(f_mae_y)
            agg_metrics["mae_x_log"].append(f_mae_x)
            agg_metrics["rmse_y_raw"].append(rmse_y_raw)
            agg_metrics["rmse_x_raw"].append(rmse_x_raw)
            agg_metrics["nrmse_x_log"].append(nrmse_x_fold)
            agg_metrics["nrmse_y_log"].append(nrmse_y_fold)
            agg_metrics["p_xy"].append(p_xy)
            agg_metrics["s_xy"].append(s_xy)
            agg_metrics["p_xp"].append(p_xp)
            agg_metrics["s_xp"].append(s_xp)
            agg_metrics["p_py"].append(p_py)
            agg_metrics["s_py"].append(s_py)
            agg_metrics["p_pp"].append(p_pp)
            agg_metrics["s_pp"].append(s_pp)
            agg_metrics["bias_x"].append(bias_x_fold)
            agg_metrics["loa_width_x"].append(loa_width_x_fold)
            agg_metrics["bias_y"].append(bias_y_fold)
            agg_metrics["loa_width_y"].append(loa_width_y_fold)

            sample_data = []
            for i_sample in range(len(test)):
                sample_data.extend([
                    float(test["x_log"].iloc[i_sample]),
                    float(pred_x_log[i_sample]),
                    float(err_x[i_sample]),
                    float(test["y_log"].iloc[i_sample]),
                    float(pred_y_log[i_sample]),
                    float(err_y[i_sample]),
                ])

            fold_metrics_str = (
                f"\t{f_mae_x}\t{f_rmse_x}\t{f_mae_y}\t{f_rmse_y}"
                f"\t{p_xy}\t{s_xy}\t{p_xp}\t{s_xp}\t{p_py}\t{s_py}\t{p_pp}\t{s_pp}"
                f"\t{bias_x_fold}\t{loa_width_x_fold}\t{bias_y_fold}\t{loa_width_y_fold}"
                f"\t{nrmse_x_fold}\t{nrmse_y_fold}"
            )
            line_str = f"{gene_name}\t" + "\t".join(map(str, sample_data)) + fold_metrics_str + "\n"
            fold_rows.append((fold_idx + 1, line_str))

        if len(agg_metrics["rmse_x_log"]) == 0:
            return None

        avg_rmse_x = float(np.mean(agg_metrics["rmse_x_log"]))
        avg_rmse_y = float(np.mean(agg_metrics["rmse_y_log"]))
        avg_mae_x = float(np.mean(agg_metrics["mae_x_log"]))
        avg_mae_y = float(np.mean(agg_metrics["mae_y_log"]))
        avg_rmse_x_raw = float(np.mean(agg_metrics["rmse_x_raw"]))
        avg_rmse_y_raw = float(np.mean(agg_metrics["rmse_y_raw"]))
        avg_nrmse_x = float(np.mean(agg_metrics["nrmse_x_log"]))
        avg_nrmse_y = float(np.mean(agg_metrics["nrmse_y_log"]))

        avg_p_xy = float(np.mean(agg_metrics["p_xy"]))
        avg_s_xy = float(np.mean(agg_metrics["s_xy"]))
        avg_p_xp = float(np.mean(agg_metrics["p_xp"]))
        avg_s_xp = float(np.mean(agg_metrics["s_xp"]))
        avg_p_py = float(np.mean(agg_metrics["p_py"]))
        avg_s_py = float(np.mean(agg_metrics["s_py"]))
        avg_p_pp = float(np.mean(agg_metrics["p_pp"]))
        avg_s_pp = float(np.mean(agg_metrics["s_pp"]))

        avg_bias_x = float(np.mean(agg_metrics["bias_x"]))
        avg_loa_width_x = float(np.mean(agg_metrics["loa_width_x"]))
        avg_bias_y = float(np.mean(agg_metrics["bias_y"]))
        avg_loa_width_y = float(np.mean(agg_metrics["loa_width_y"]))

        summary_line_content = [
            gene_name, avg_rmse_x, avg_rmse_y, avg_mae_x, avg_mae_y,
            avg_rmse_x_raw, avg_rmse_y_raw,
            range_x_raw, range_y_raw, range_x_log, range_y_log,
            avg_nrmse_x, avg_nrmse_y,
            avg_p_xy, avg_s_xy, avg_p_xp, avg_s_xp, avg_p_py, avg_s_py, avg_p_pp, avg_s_pp,
            avg_bias_x, avg_loa_width_x, avg_bias_y, avg_loa_width_y,
        ]
        summary_line = "\t".join(map(str, summary_line_content)) + "\n"
        return summary_line, fold_rows

    except Exception as e:
        print(f"Error processing gene {gene_file}: {e}")
        return None


def process_single_gene_identity(
    gene_file: str,
    input_dir: str,
    n_splits: int,
    random_seed: int,
    pair_ids: np.ndarray = None,
):
    try:
        gene_path = os.path.join(input_dir, gene_file)
        df = pd.read_csv(gene_path, header=None, sep="\t")
        df.columns = ["x", "y"]
        gene_name = os.path.splitext(gene_file)[0]

        df["x_log"] = np.log1p(pd.to_numeric(df["x"], errors="coerce"))
        df["y_log"] = np.log1p(pd.to_numeric(df["y"], errors="coerce"))
        df.dropna(subset=["x_log", "y_log"], inplace=True)

        if df.empty:
            return None

        cv, groups = _make_cv_with_groups(df, n_splits, random_seed, pair_ids)
        if cv is None:
            print(f"Warning: cannot construct CV for gene {gene_name} (groups issue). Skipping.")
            return None

        range_x_log = df["x_log"].max() - df["x_log"].min() if not df["x_log"].empty else 0.0
        range_y_log = df["y_log"].max() - df["y_log"].min() if not df["y_log"].empty else 0.0
        range_x_raw = df["x"].max() - df["x"].min() if not df["x"].empty else 0.0
        range_y_raw = df["y"].max() - df["y"].min() if not df["y"].empty else 0.0

        agg_metrics = {
            "rmse_x_log": [], "rmse_y_log": [],
            "rmse_x_raw": [], "rmse_y_raw": [],
            "mae_x_log": [], "mae_y_log": [],
            "nrmse_x_log": [], "nrmse_y_log": [],
            "p_xy": [], "s_xy": [],
            "p_xp": [], "s_xp": [],
            "p_py": [], "s_py": [],
            "p_pp": [], "s_pp": [],
            "bias_x": [], "loa_width_x": [],
            "bias_y": [], "loa_width_y": [],
        }
        fold_rows = []

        splitter = cv.split(df, groups=groups) if groups is not None else cv.split(df)

        for fold_idx, (train_idx, test_idx) in enumerate(splitter):
            train = df.iloc[train_idx]
            test = df.iloc[test_idx]

            if len(train) == 0 or len(test) == 0:
                continue

            # Identity baseline: array -> seq and seq -> array
            pred_y_log = test["x_log"].to_numpy()
            pred_x_log = test["y_log"].to_numpy()

            err_y = pred_y_log - test["y_log"].to_numpy()
            err_x = pred_x_log - test["x_log"].to_numpy()

            f_mae_y = mean_absolute_error(test["y_log"], pred_y_log)
            f_rmse_y = np.sqrt(mean_squared_error(test["y_log"], pred_y_log))
            f_mae_x = mean_absolute_error(test["x_log"], pred_x_log)
            f_rmse_x = np.sqrt(mean_squared_error(test["x_log"], pred_x_log))

            rmse_y_raw = np.sqrt(
                np.mean((test["y"].to_numpy() - np.expm1(pred_y_log)) ** 2)
            ) if len(test["y"]) > 0 else 0.0
            rmse_x_raw = np.sqrt(
                np.mean((test["x"].to_numpy() - np.expm1(pred_x_log)) ** 2)
            ) if len(test["x"]) > 0 else 0.0

            nrmse_x_fold = (f_rmse_x / range_x_log) if range_x_log > 0 else 0.0
            nrmse_y_fold = (f_rmse_y / range_y_log) if range_y_log > 0 else 0.0

            p_xy, s_xy = get_correlations(
                test["x_log"].to_numpy(), test["y_log"].to_numpy()
            )
            p_xp, s_xp = get_correlations(
                test["x_log"].to_numpy(), pred_y_log
            )
            p_py, s_py = get_correlations(
                pred_x_log, test["y_log"].to_numpy()
            )
            p_pp, s_pp = get_correlations(pred_x_log, pred_y_log)

            bias_x_fold, loa_width_x_fold = calculate_bland_altman(err_x)
            bias_y_fold, loa_width_y_fold = calculate_bland_altman(err_y)

            agg_metrics["rmse_y_log"].append(f_rmse_y)
            agg_metrics["rmse_x_log"].append(f_rmse_x)
            agg_metrics["mae_y_log"].append(f_mae_y)
            agg_metrics["mae_x_log"].append(f_mae_x)
            agg_metrics["rmse_y_raw"].append(rmse_y_raw)
            agg_metrics["rmse_x_raw"].append(rmse_x_raw)
            agg_metrics["nrmse_x_log"].append(nrmse_x_fold)
            agg_metrics["nrmse_y_log"].append(nrmse_y_fold)
            agg_metrics["p_xy"].append(p_xy)
            agg_metrics["s_xy"].append(s_xy)
            agg_metrics["p_xp"].append(p_xp)
            agg_metrics["s_xp"].append(s_xp)
            agg_metrics["p_py"].append(p_py)
            agg_metrics["s_py"].append(s_py)
            agg_metrics["p_pp"].append(p_pp)
            agg_metrics["s_pp"].append(s_pp)
            agg_metrics["bias_x"].append(bias_x_fold)
            agg_metrics["loa_width_x"].append(loa_width_x_fold)
            agg_metrics["bias_y"].append(bias_y_fold)
            agg_metrics["loa_width_y"].append(loa_width_y_fold)

            sample_data = []
            for i_sample in range(len(test)):
                sample_data.extend([
                    float(test["x_log"].iloc[i_sample]),
                    float(pred_x_log[i_sample]),
                    float(err_x[i_sample]),
                    float(test["y_log"].iloc[i_sample]),
                    float(pred_y_log[i_sample]),
                    float(err_y[i_sample]),
                ])

            fold_metrics_str = (
                f"\t{f_mae_x}\t{f_rmse_x}\t{f_mae_y}\t{f_rmse_y}"
                f"\t{p_xy}\t{s_xy}\t{p_xp}\t{s_xp}\t{p_py}\t{s_py}\t{p_pp}\t{s_pp}"
                f"\t{bias_x_fold}\t{loa_width_x_fold}\t{bias_y_fold}\t{loa_width_y_fold}"
                f"\t{nrmse_x_fold}\t{nrmse_y_fold}"
            )
            line_str = f"{gene_name}\t" + "\t".join(map(str, sample_data)) + fold_metrics_str + "\n"
            fold_rows.append((fold_idx + 1, line_str))

        if len(agg_metrics["rmse_x_log"]) == 0:
            return None

        avg_rmse_x = float(np.mean(agg_metrics["rmse_x_log"]))
        avg_rmse_y = float(np.mean(agg_metrics["rmse_y_log"]))
        avg_mae_x = float(np.mean(agg_metrics["mae_x_log"]))
        avg_mae_y = float(np.mean(agg_metrics["mae_y_log"]))
        avg_rmse_x_raw = float(np.mean(agg_metrics["rmse_x_raw"]))
        avg_rmse_y_raw = float(np.mean(agg_metrics["rmse_y_raw"]))
        avg_nrmse_x = float(np.mean(agg_metrics["nrmse_x_log"]))
        avg_nrmse_y = float(np.mean(agg_metrics["nrmse_y_log"]))

        avg_p_xy = float(np.mean(agg_metrics["p_xy"]))
        avg_s_xy = float(np.mean(agg_metrics["s_xy"]))
        avg_p_xp = float(np.mean(agg_metrics["p_xp"]))
        avg_s_xp = float(np.mean(agg_metrics["s_xp"]))
        avg_p_py = float(np.mean(agg_metrics["p_py"]))
        avg_s_py = float(np.mean(agg_metrics["s_py"]))
        avg_p_pp = float(np.mean(agg_metrics["p_pp"]))
        avg_s_pp = float(np.mean(agg_metrics["s_pp"]))

        avg_bias_x = float(np.mean(agg_metrics["bias_x"]))
        avg_loa_width_x = float(np.mean(agg_metrics["loa_width_x"]))
        avg_bias_y = float(np.mean(agg_metrics["bias_y"]))
        avg_loa_width_y = float(np.mean(agg_metrics["loa_width_y"]))

        summary_line_content = [
            gene_name, avg_rmse_x, avg_rmse_y, avg_mae_x, avg_mae_y,
            avg_rmse_x_raw, avg_rmse_y_raw,
            range_x_raw, range_y_raw, range_x_log, range_y_log,
            avg_nrmse_x, avg_nrmse_y,
            avg_p_xy, avg_s_xy, avg_p_xp, avg_s_xp, avg_p_py, avg_s_py, avg_p_pp, avg_s_pp,
            avg_bias_x, avg_loa_width_x, avg_bias_y, avg_loa_width_y,
        ]
        summary_line = "\t".join(map(str, summary_line_content)) + "\n"
        return summary_line, fold_rows

    except Exception as e:
        print(f"Error processing gene {gene_file}: {e}")
        return None


def _load_pair_ids(pairing_file: str):
    if pairing_file is None:
        return None
    if not os.path.exists(pairing_file):
        print(f"Error: pairing file not found at '{pairing_file}'.")
        return None
    pairing_df = pd.read_csv(pairing_file, header=None, sep="\t")
    return pairing_df.iloc[:, 0].to_numpy()


def run_xplat_for_organism(
    name: str,
    input_dir: str,
    output_dir: str,
    output_prefix: str,
    n_splits: int,
    random_seed: int,
    pairing_file: str = None,
    buffer_size: int = 100,
):
    if not os.path.exists(input_dir):
        print(f"Error: input directory for {name} not found at '{input_dir}'.")
        return

    pair_ids = _load_pair_ids(pairing_file)

    os.makedirs(output_dir, exist_ok=True)
    output_file_final = os.path.join(
        output_dir,
        f"{output_prefix}-results-polynomial-log_transformed.{n_splits}fold_avg.txt",
    )
    fold_results_dir = os.path.join(output_dir, f"{output_prefix}-fold_results")
    os.makedirs(fold_results_dir, exist_ok=True)

    gene_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    print(f"Processing {len(gene_files)} genes for {name}...")
    start_time = time.time()

    header_summary = (
        "gene\tavg_rmse_x\tavg_rmse_y\tavg_mae_x\tavg_mae_y"
        "\tavg_rmse_x_raw\tavg_rmse_y_raw"
        "\trange_x_raw\trange_y_raw\trange_x_log\trange_y_log"
        "\tavg_nrmse_x\tavg_nrmse_y"
        "\tavg_p_xy\tavg_s_xy\tavg_p_xp\tavg_s_xp\tavg_p_py\tavg_s_py\tavg_p_pp\tavg_s_pp"
        "\tavg_bias_x\tavg_loa_width_x\tavg_bias_y\tavg_loa_width_y\n"
    )
    with open(output_file_final, "w") as f_out:
        f_out.write(header_summary)

    fold_file_header = (
        "# Gene\tfollowed by flat sample data (rawX, predX, errX, rawY, predY, errY)"
        " and ending with X_MAE, X_RMSE, Y_MAE, Y_RMSE"
        "\tpearson_xy\tspearman_xy\tpearson_x_predy\tspearman_x_predy"
        "\tpearson_predx_y\tspearman_predx_y\tpearson_predx_predy\tspearman_predx_predy"
        "\tbias_x\tloa_width_x\tbias_y\tloa_width_y\tnrmse_x\tnrmse_y\n"
    )
    for i in range(1, n_splits + 1):
        fold_path = os.path.join(fold_results_dir, f"fold{i}.txt")
        with open(fold_path, "w") as f_fold:
            f_fold.write(fold_file_header)

    summary_buffer = []
    folds_buffer = {i: [] for i in range(1, n_splits + 1)}
    completed = 0

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_single_gene,
                gf,
                input_dir,
                n_splits,
                random_seed,
                pair_ids,
            ): gf
            for gf in gene_files
        }
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result:
                s_line, f_rows = result
                summary_buffer.append(s_line)
                for f_num, f_line in f_rows:
                    folds_buffer[f_num].append(f_line)

            if completed % buffer_size == 0 or completed == len(gene_files):
                with open(output_file_final, "a") as f_out:
                    f_out.writelines(summary_buffer)
                summary_buffer = []
                for f_num in range(1, n_splits + 1):
                    fold_path = os.path.join(fold_results_dir, f"fold{f_num}.txt")
                    with open(fold_path, "a") as f_fold:
                        f_fold.writelines(folds_buffer[f_num])
                    folds_buffer[f_num] = []

            if completed % 50 == 0 or completed == len(gene_files):
                elapsed = time.time() - start_time
                eta = (elapsed / completed) * (len(gene_files) - completed)
                print(
                    f"Progress for {name}: {completed}/{len(gene_files)} | "
                    f"Elapsed: {timedelta(seconds=int(elapsed))} | "
                    f"ETA: {timedelta(seconds=int(eta))}"
                )

    print(f"Processing complete for {name}.")


def run_linear_for_organism(
    name: str,
    input_dir: str,
    output_dir: str,
    output_prefix: str,
    n_splits: int,
    random_seed: int,
    pairing_file: str = None,
    buffer_size: int = 100,
):
    if not os.path.exists(input_dir):
        print(f"Error: input directory for {name} not found at '{input_dir}'.")
        return

    pair_ids = _load_pair_ids(pairing_file)

    os.makedirs(output_dir, exist_ok=True)
    output_file_final = os.path.join(
        output_dir,
        f"{output_prefix}-results-linear-log_transformed.{n_splits}fold_avg.txt",
    )
    fold_results_dir = os.path.join(output_dir, f"{output_prefix}-fold_results-linear")
    os.makedirs(fold_results_dir, exist_ok=True)

    gene_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    print(f"Processing {len(gene_files)} genes for {name} (Linear Regression)...")
    start_time = time.time()

    header_summary = (
        "gene\tavg_rmse_x\tavg_rmse_y\tavg_mae_x\tavg_mae_y"
        "\tavg_rmse_x_raw\tavg_rmse_y_raw"
        "\trange_x_raw\trange_y_raw\trange_x_log\trange_y_log"
        "\tavg_nrmse_x\tavg_nrmse_y"
        "\tavg_p_xy\tavg_s_xy\tavg_p_xp\tavg_s_xp\tavg_p_py\tavg_s_py\tavg_p_pp\tavg_s_pp"
        "\tavg_bias_x\tavg_loa_width_x\tavg_bias_y\tavg_loa_width_y\n"
    )
    with open(output_file_final, "w") as f_out:
        f_out.write(header_summary)

    fold_file_header = (
        "# Gene\tfollowed by flat sample data (rawX, predX, errX, rawY, predY, errY)"
        " and ending with X_MAE, X_RMSE, Y_MAE, Y_RMSE"
        "\tpearson_xy\tspearman_xy\tpearson_x_predy\tspearman_x_predy"
        "\tpearson_predx_y\tspearman_predx_y\tpearson_predx_predy\tspearman_predx_predy"
        "\tbias_x\tloa_width_x\tbias_y\tloa_width_y\tnrmse_x\tnrmse_y\n"
    )
    for i in range(1, n_splits + 1):
        fold_path = os.path.join(fold_results_dir, f"fold{i}.txt")
        with open(fold_path, "w") as f_fold:
            f_fold.write(fold_file_header)

    summary_buffer = []
    folds_buffer = {i: [] for i in range(1, n_splits + 1)}
    completed = 0

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_single_gene_linear,
                gf,
                input_dir,
                n_splits,
                random_seed,
                pair_ids,
            ): gf
            for gf in gene_files
        }
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result:
                s_line, f_rows = result
                summary_buffer.append(s_line)
                for f_num, f_line in f_rows:
                    folds_buffer[f_num].append(f_line)

            if completed % buffer_size == 0 or completed == len(gene_files):
                with open(output_file_final, "a") as f_out:
                    f_out.writelines(summary_buffer)
                summary_buffer = []
                for f_num in range(1, n_splits + 1):
                    fold_path = os.path.join(fold_results_dir, f"fold{f_num}.txt")
                    with open(fold_path, "a") as f_fold:
                        f_fold.writelines(folds_buffer[f_num])
                    folds_buffer[f_num] = []

            if completed % 50 == 0 or completed == len(gene_files):
                elapsed = time.time() - start_time
                eta = (elapsed / completed) * (len(gene_files) - completed)
                print(
                    f"Progress for {name} (Linear): {completed}/{len(gene_files)} | "
                    f"Elapsed: {timedelta(seconds=int(elapsed))} | "
                    f"ETA: {timedelta(seconds=int(eta))}"
                )

    print(f"Processing complete for {name} (Linear Regression).")


def run_identity_for_organism(
    name: str,
    input_dir: str,
    output_dir: str,
    output_prefix: str,
    n_splits: int,
    random_seed: int,
    pairing_file: str = None,
    buffer_size: int = 100,
):
    if not os.path.exists(input_dir):
        print(f"Error: input directory for {name} not found at '{input_dir}'.")
        return

    pair_ids = _load_pair_ids(pairing_file)

    os.makedirs(output_dir, exist_ok=True)
    output_file_final = os.path.join(
        output_dir,
        f"{output_prefix}-results-identity-log_transformed.{n_splits}fold_avg.txt",
    )
    fold_results_dir = os.path.join(output_dir, f"{output_prefix}-fold_results-identity")
    os.makedirs(fold_results_dir, exist_ok=True)

    gene_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    print(f"Processing {len(gene_files)} genes for {name} (Identity Baseline)...")
    start_time = time.time()

    header_summary = (
        "gene\tavg_rmse_x\tavg_rmse_y\tavg_mae_x\tavg_mae_y"
        "\tavg_rmse_x_raw\tavg_rmse_y_raw"
        "\trange_x_raw\trange_y_raw\trange_x_log\trange_y_log"
        "\tavg_nrmse_x\tavg_nrmse_y"
        "\tavg_p_xy\tavg_s_xy\tavg_p_xp\tavg_s_xp\tavg_p_py\tavg_s_py\tavg_p_pp\tavg_s_pp"
        "\tavg_bias_x\tavg_loa_width_x\tavg_bias_y\tavg_loa_width_y\n"
    )
    with open(output_file_final, "w") as f_out:
        f_out.write(header_summary)

    fold_file_header = (
        "# Gene\tfollowed by flat sample data (rawX, predX, errX, rawY, predY, errY)"
        " and ending with X_MAE, X_RMSE, Y_MAE, Y_RMSE"
        "\tpearson_xy\tspearman_xy\tpearson_x_predy\tspearman_x_predy"
        "\tpearson_predx_y\tspearman_predx_y\tpearson_predx_predy\tspearman_predx_predy"
        "\tbias_x\tloa_width_x\tbias_y\tloa_width_y\tnrmse_x\tnrmse_y\n"
    )
    for i in range(1, n_splits + 1):
        fold_path = os.path.join(fold_results_dir, f"fold{i}.txt")
        with open(fold_path, "w") as f_fold:
            f_fold.write(fold_file_header)

    summary_buffer = []
    folds_buffer = {i: [] for i in range(1, n_splits + 1)}
    completed = 0

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_single_gene_identity,
                gf,
                input_dir,
                n_splits,
                random_seed,
                pair_ids,
            ): gf
            for gf in gene_files
        }
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result:
                s_line, f_rows = result
                summary_buffer.append(s_line)
                for f_num, f_line in f_rows:
                    folds_buffer[f_num].append(f_line)

            if completed % buffer_size == 0 or completed == len(gene_files):
                with open(output_file_final, "a") as f_out:
                    f_out.writelines(summary_buffer)
                summary_buffer = []
                for f_num in range(1, n_splits + 1):
                    fold_path = os.path.join(fold_results_dir, f"fold{f_num}.txt")
                    with open(fold_path, "a") as f_fold:
                        f_fold.writelines(folds_buffer[f_num])
                    folds_buffer[f_num] = []

            if completed % 50 == 0 or completed == len(gene_files):
                elapsed = time.time() - start_time
                eta = (elapsed / completed) * (len(gene_files) - completed)
                print(
                    f"Progress for {name} (Identity): {completed}/{len(gene_files)} | "
                    f"Elapsed: {timedelta(seconds=int(elapsed))} | "
                    f"ETA: {timedelta(seconds=int(eta))}"
                )

    print(f"Processing complete for {name} (Identity Baseline).")