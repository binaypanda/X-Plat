import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_bland_altman(errors):
    if len(errors) < 2 or np.std(errors) == 0:
        return 0.0, 0.0
    bias = np.mean(errors)
    sd = np.std(errors)
    loa_width = 1.96 * sd * 2
    return bias, loa_width


def get_correlations(arr1, arr2):
    if len(arr1) < 2 or np.std(arr1) == 0 or np.std(arr2) == 0:
        return 0.0, 0.0
    p, _ = pearsonr(arr1, arr2)
    s, _ = spearmanr(arr1, arr2)
    return p, s


def create_harmony_consolidated_files(species_list, base_dir=".", n_splits=10):
    """
    base_dir: repo_root/results

    Expects:
      - tdm_inputs/{species}_tdm_input_arr/seq (test matrices)
      - harmony/{species}_harmony_output (HARMONY fold outputs)
      - xplat/{species}/{species}-results-polynomial-log_transformed.10fold_avg.txt
    Writes:
      - harmony/{species}_harmony_fold_results/fold{i}.txt
      - harmony/{species}-results-harmony-log_transformed.10fold_avg.txt
      - harmony/{species}-results-harmony-fold_mae_summary.txt
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats._stats_py")

    print("Preprocessing and consolidating HARMONY results...")

    tdm_root = os.path.join(base_dir, "tdm_inputs")
    harmony_root = os.path.join(base_dir, "harmony")
    poly_root = os.path.join(base_dir, "xplat")

    for species in species_list:
        print(f"Consolidating HARMONY results for {species}...")

        harmony_input_arr_dir = os.path.join(tdm_root, f"{species}_tdm_input_arr")
        harmony_input_seq_dir = os.path.join(tdm_root, f"{species}_tdm_input_seq")
        harmony_output_dir = os.path.join(harmony_root, f"{species}_harmony_output")

        consolidated_fold_dir = os.path.join(harmony_root, f"{species}_harmony_fold_results")
        os.makedirs(consolidated_fold_dir, exist_ok=True)

        poly_summary_path = os.path.join(
            poly_root, species, f"{species}-results-polynomial-log_transformed.10fold_avg.txt"
        )
        gene_ranges_df = None
        if os.path.exists(poly_summary_path):
            gene_ranges_df = pd.read_csv(poly_summary_path, sep="\t")
            gene_ranges_df["gene"] = (
                gene_ranges_df["gene"].astype(str).str.replace('"', "").str.strip()
            )
            gene_ranges_df.set_index("gene", inplace=True)
        else:
            print(f"Warning: Polynomial summary not found for {species}. NRMSE will be 0.0.")

        gene_summary_data = []
        fold_gene_metrics = {}

        for i in range(1, n_splits + 1):
            fold_output_path = os.path.join(consolidated_fold_dir, f"fold{i}.txt")

            raw_x_path = os.path.join(harmony_input_arr_dir, f"fold{i}_arr.txt")
            raw_y_path = os.path.join(harmony_input_seq_dir, f"fold{i}_seq.txt")
            harmony_pred_path = os.path.join(harmony_output_dir, f"fold{i}.txt")

            if not (
                os.path.exists(raw_x_path)
                and os.path.exists(raw_y_path)
                and os.path.exists(harmony_pred_path)
            ):
                print(f"  Skipping fold {i} for {species} due to missing files.")
                continue

            df_raw_x = pd.read_csv(raw_x_path, sep="\t", header=None, index_col=0)
            df_raw_y = pd.read_csv(raw_y_path, sep="\t", header=None, index_col=0)
            df_harmony_pred = pd.read_csv(harmony_pred_path, sep="\t", header=0, index_col=0)

            df_raw_x.index = df_raw_x.index.astype(str).str.replace('"', "").str.strip()
            df_raw_y.index = df_raw_y.index.astype(str).str.replace('"', "").str.strip()
            df_harmony_pred.index = (
                df_harmony_pred.index.astype(str).str.replace('"', "").str.strip()
            )

            common_genes = df_raw_x.index.intersection(df_raw_y.index)
            common_genes = common_genes.intersection(df_harmony_pred.index)

            if gene_ranges_df is not None:
                common_genes = common_genes.intersection(gene_ranges_df.index)

            if common_genes.empty:
                print(f"  Skipping fold {i} for {species} due to no common genes.")
                continue

            lines_for_fold_file = [
                "# Gene\t"
                "samples(rawX,predX,errX,rawY,predY,errY)"
                "\tX_MAE\tX_RMSE\tY_MAE\tY_RMSE"
                "\tpearson_xy\tspearman_xy"
                "\tpearson_x_predy\tspearman_x_predy"
                "\tpearson_predx_y\tspearman_predx_y"
                "\tpearson_predx_predy\tspearman_predx_predy"
                "\tbias_x\tloa_width_x\tbias_y\tloa_width_y"
                "\tnrmse_x\tnrmse_y\n"
            ]

            for gene in common_genes:
                if gene not in fold_gene_metrics:
                    fold_gene_metrics[gene] = {
                        "rmse_x": [],
                        "rmse_y": [],
                        "mae_x": [],
                        "mae_y": [],
                        "rmse_x_raw": [],
                        "rmse_y_raw": [],
                        "nrmse_x": [],
                        "nrmse_y": [],
                        "p_xy": [],
                        "s_xy": [],
                        "p_xp": [],
                        "s_xp": [],
                        "p_py": [],
                        "s_py": [],
                        "p_pp": [],
                        "s_pp": [],
                        "bias_x": [],
                        "loa_width_x": [],
                        "bias_y": [],
                        "loa_width_y": [],
                    }

            for gene in common_genes:
                raw_x = df_raw_x.loc[gene].values.flatten()
                raw_y = df_raw_y.loc[gene].values.flatten()

                pred_x = df_harmony_pred.loc[
                    gene, df_harmony_pred.columns.str.startswith("x.")
                ].values.flatten()
                pred_y = df_harmony_pred.loc[
                    gene, df_harmony_pred.columns.str.startswith("y.")
                ].values.flatten()

                min_len = min(len(raw_x), len(raw_y), len(pred_x), len(pred_y))
                raw_x_f, pred_x_f = raw_x[:min_len], pred_x[:min_len]
                err_x_f = pred_x_f - raw_x_f

                raw_y_f, pred_y_f = raw_y[:min_len], pred_y[:min_len]
                err_y_f = pred_y_f - raw_y_f

                if len(raw_x_f) == 0 or len(raw_y_f) == 0:
                    continue

                samples_output = []
                for sample_idx in range(len(raw_x_f)):
                    samples_output.extend(
                        [
                            raw_x_f[sample_idx],
                            pred_x_f[sample_idx],
                            err_x_f[sample_idx],
                            raw_y_f[sample_idx],
                            pred_y_f[sample_idx],
                            err_y_f[sample_idx],
                        ]
                    )

                mae_x = mean_absolute_error(raw_x_f, pred_x_f)
                rmse_x = np.sqrt(mean_squared_error(raw_x_f, pred_x_f))
                mae_y = mean_absolute_error(raw_y_f, pred_y_f)
                rmse_y = np.sqrt(mean_squared_error(raw_y_f, pred_y_f))

                p_xy, s_xy = get_correlations(raw_x_f, raw_y_f)
                p_xp, s_xp = get_correlations(raw_x_f, pred_y_f)
                p_py, s_py = get_correlations(pred_x_f, raw_y_f)
                p_pp, s_pp = get_correlations(pred_x_f, pred_y_f)

                bias_x, loa_width_x = calculate_bland_altman(err_x_f)
                bias_y, loa_width_y = calculate_bland_altman(err_y_f)

                nrmse_x, nrmse_y = 0.0, 0.0
                if gene_ranges_df is not None and gene in gene_ranges_df.index:
                    range_x_log = gene_ranges_df.loc[gene, "range_x_log"]
                    range_y_log = gene_ranges_df.loc[gene, "range_y_log"]
                    if range_x_log > 0:
                        nrmse_x = rmse_x / range_x_log
                    if range_y_log > 0:
                        nrmse_y = rmse_y / range_y_log

                fold_gene_metrics[gene]["rmse_x"].append(rmse_x)
                fold_gene_metrics[gene]["rmse_y"].append(rmse_y)
                fold_gene_metrics[gene]["mae_x"].append(mae_x)
                fold_gene_metrics[gene]["mae_y"].append(mae_y)
                fold_gene_metrics[gene]["nrmse_x"].append(nrmse_x)
                fold_gene_metrics[gene]["nrmse_y"].append(nrmse_y)
                fold_gene_metrics[gene]["p_xy"].append(p_xy)
                fold_gene_metrics[gene]["s_xy"].append(s_xy)
                fold_gene_metrics[gene]["p_xp"].append(p_xp)
                fold_gene_metrics[gene]["s_xp"].append(s_xp)
                fold_gene_metrics[gene]["p_py"].append(p_py)
                fold_gene_metrics[gene]["s_py"].append(s_py)
                fold_gene_metrics[gene]["p_pp"].append(p_pp)
                fold_gene_metrics[gene]["s_pp"].append(s_pp)
                fold_gene_metrics[gene]["bias_x"].append(bias_x)
                fold_gene_metrics[gene]["loa_width_x"].append(loa_width_x)
                fold_gene_metrics[gene]["bias_y"].append(bias_y)
                fold_gene_metrics[gene]["loa_width_y"].append(loa_width_y)

                metrics_str = (
                    f"{mae_x}\t{rmse_x}\t{mae_y}\t{rmse_y}\t"
                    f"{p_xy}\t{s_xy}\t{p_xp}\t{s_xp}\t"
                    f"{p_py}\t{s_py}\t{p_pp}\t{s_pp}\t"
                    f"{bias_x}\t{loa_width_x}\t{bias_y}\t{loa_width_y}\t"
                    f"{nrmse_x}\t{nrmse_y}"
                )

                lines_for_fold_file.append(
                    f"{gene}\t" + "\t".join(map(str, samples_output)) + f"\t{metrics_str}\n"
                )

            with open(fold_output_path, "w") as f_out:
                f_out.writelines(lines_for_fold_file)

            print(f"    Fold {i} processed and saved to {fold_output_path}.")

        for gene, metrics_list_dict in fold_gene_metrics.items():
            if not metrics_list_dict["rmse_x"]:
                continue

            avg_rmse_x = np.mean(metrics_list_dict["rmse_x"])
            avg_rmse_y = np.mean(metrics_list_dict["rmse_y"])
            avg_mae_x = np.mean(metrics_list_dict["mae_x"])
            avg_mae_y = np.mean(metrics_list_dict["mae_y"])
            avg_nrmse_x = np.mean(metrics_list_dict["nrmse_x"])
            avg_nrmse_y = np.mean(metrics_list_dict["nrmse_y"])
            avg_p_xy = np.mean(metrics_list_dict["p_xy"])
            avg_s_xy = np.mean(metrics_list_dict["s_xy"])
            avg_p_xp = np.mean(metrics_list_dict["p_xp"])
            avg_s_xp = np.mean(metrics_list_dict["s_xp"])
            avg_p_py = np.mean(metrics_list_dict["p_py"])
            avg_s_py = np.mean(metrics_list_dict["s_py"])
            avg_p_pp = np.mean(metrics_list_dict["p_pp"])
            avg_s_pp = np.mean(metrics_list_dict["s_pp"])
            avg_bias_x = np.mean(metrics_list_dict["bias_x"])
            avg_loa_width_x = np.mean(metrics_list_dict["loa_width_x"])
            avg_bias_y = np.mean(metrics_list_dict["bias_y"])
            avg_loa_width_y = np.mean(metrics_list_dict["loa_width_y"])

            range_x_raw = range_y_raw = range_x_log = range_y_log = 0.0
            if gene_ranges_df is not None and gene in gene_ranges_df.index:
                range_x_raw = gene_ranges_df.loc[gene, "range_x_raw"]
                range_y_raw = gene_ranges_df.loc[gene, "range_y_raw"]
                range_x_log = gene_ranges_df.loc[gene, "range_x_log"]
                range_y_log = gene_ranges_df.loc[gene, "range_y_log"]

            gene_summary_data.append(
                {
                    "gene": gene,
                    "avg_rmse_x": avg_rmse_x,
                    "avg_rmse_y": avg_rmse_y,
                    "avg_mae_x": avg_mae_x,
                    "avg_mae_y": avg_mae_y,
                    "avg_rmse_x_raw": np.nan,
                    "avg_rmse_y_raw": np.nan,
                    "range_x_raw": range_x_raw,
                    "range_y_raw": range_y_raw,
                    "range_x_log": range_x_log,
                    "range_y_log": range_y_log,
                    "avg_nrmse_x": avg_nrmse_x,
                    "avg_nrmse_y": avg_nrmse_y,
                    "avg_p_xy": avg_p_xy,
                    "avg_s_xy": avg_s_xy,
                    "avg_p_xp": avg_p_xp,
                    "avg_s_xp": avg_s_xp,
                    "avg_p_py": avg_p_py,
                    "avg_s_py": avg_s_py,
                    "avg_p_pp": avg_p_pp,
                    "avg_s_pp": avg_s_pp,
                    "avg_bias_x": avg_bias_x,
                    "avg_loa_width_x": avg_loa_width_x,
                    "avg_bias_y": avg_bias_y,
                    "avg_loa_width_y": avg_loa_width_y,
                }
            )

        if gene_summary_data:
            summary_df = pd.DataFrame(gene_summary_data)
            cols = summary_df.columns.tolist()
            idx_rmse_y = cols.index("avg_rmse_y")
            if "avg_mae_x" in cols and "avg_mae_y" in cols:
                cols.insert(idx_rmse_y + 1, cols.pop(cols.index("avg_mae_x")))
                cols.insert(idx_rmse_y + 2, cols.pop(cols.index("avg_mae_y")))
            summary_df = summary_df[cols]

            summary_output_path = os.path.join(
                harmony_root, f"{species}-results-harmony-log_transformed.10fold_avg.txt"
            )
            summary_df.to_csv(summary_output_path, sep="\t", index=False)
            print(f"  Overall HARMONY summary for {species} saved to {summary_output_path}")

            species_fold_mae_rows = []
            for fold_num in range(1, n_splits + 1):
                fold_file_path = os.path.join(consolidated_fold_dir, f"fold{fold_num}.txt")
                if not os.path.exists(fold_file_path):
                    continue

                try:
                    with open(fold_file_path, "r") as f:
                        header_description_line = f.readline().strip()

                    parts = header_description_line.lstrip("# ").split("\t")
                    metric_names_from_file = parts[2:]

                    df_fold = pd.read_csv(fold_file_path, sep="\t", header=None, skiprows=1)
                    if df_fold.empty:
                        continue

                    num_gene_col = 1
                    num_metric_cols = len(metric_names_from_file)
                    num_sample_data_cols = df_fold.shape[1] - num_gene_col - num_metric_cols

                    all_column_names = (
                        ["gene"]
                        + [f"sample_col_{i}" for i in range(num_sample_data_cols)]
                        + metric_names_from_file
                    )
                    df_fold.columns = all_column_names

                    if "X_MAE" in df_fold.columns and "Y_MAE" in df_fold.columns:
                        avg_x_mae = df_fold["X_MAE"].mean()
                        avg_y_mae = df_fold["Y_MAE"].mean()
                        species_fold_mae_rows.append(
                            {
                                "species": species,
                                "fold": fold_num,
                                "avg_x_mae": avg_x_mae,
                                "avg_y_mae": avg_y_mae,
                            }
                        )
                except Exception as e:
                    print(
                        f"  Warning: Could not parse MAE metrics for {species}, fold {fold_num}: {e}"
                    )

            if species_fold_mae_rows:
                mae_df = pd.DataFrame(species_fold_mae_rows)
                mae_output_path = os.path.join(
                    harmony_root, f"{species}-results-harmony-fold_mae_summary.txt"
                )
                mae_df.to_csv(mae_output_path, sep="\t", index=False)

                overall_avg_x_mae = mae_df["avg_x_mae"].mean()
                overall_avg_y_mae = mae_df["avg_y_mae"].mean()

                print(f"  Fold-level MAE summary saved to {mae_output_path}")
                print(
                    f"  Overall Average HARMONY X_MAE for {species} across folds: "
                    f"{overall_avg_x_mae:.4f}"
                )
                print(
                    f"  Overall Average HARMONY Y_MAE for {species} across folds: "
                    f"{overall_avg_y_mae:.4f}"
                )

    print("All HARMONY consolidation and summary creation complete.")


if __name__ == "__main__":
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    base_dir = os.path.join(repo_root, "results")
    species_list = ["rat", "arabidopsis", "human"]
    create_harmony_consolidated_files(species_list, base_dir=base_dir, n_splits=10)
