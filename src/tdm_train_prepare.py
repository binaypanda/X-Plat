import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GroupKFold


def reformat_train_folds_for_tdm(organism_config, n_splits: int = 10):
    """
    From per-gene raw files, create training matrices for each fold:
    GeneName\tSample1\tSample2...

    Uses GroupKFold per gene with the same pairing constraints
    as the main X-Plat implementation (no pair_id split across folds).
    """
    species = organism_config["name"]
    source_dir = organism_config["dir"]
    output_dir_arr = organism_config["out_arr"]
    output_dir_seq = organism_config["out_seq"]
    pairing_file = organism_config.get("pairing_file", None)

    print(f"Reformatting training folds for {species}...")

    os.makedirs(output_dir_arr, exist_ok=True)
    os.makedirs(output_dir_seq, exist_ok=True)

    folds_arr = {i: [] for i in range(1, n_splits + 1)}
    folds_seq = {i: [] for i in range(1, n_splits + 1)}

    # Load pair_ids once per organism if provided
    if pairing_file is not None:
        if not os.path.exists(pairing_file):
            print(f"Error: pairing file for {species} not found at '{pairing_file}'.")
            return
        pairing_df = pd.read_csv(pairing_file, header=None, sep="\t")
        pair_ids_full = pairing_df.iloc[:, 0].to_numpy()
    else:
        pair_ids_full = None

    gene_files = [f for f in os.listdir(source_dir) if f.endswith(".txt")]

    for gf in gene_files:
        gene_name = os.path.splitext(gf)[0]
        gene_path = os.path.join(source_dir, gf)

        df = pd.read_csv(gene_path, header=None, sep="\t")
        if df.shape[1] < 2:
            continue
        df.columns = ["x", "y"]

        df["x_log"] = np.log1p(pd.to_numeric(df["x"], errors="coerce"))
        df["y_log"] = np.log1p(pd.to_numeric(df["y"], errors="coerce"))
        df.dropna(subset=["x_log", "y_log"], inplace=True)

        if df.empty:
            continue

        # Align pair_ids to df after NaN removal (if provided)
        if pair_ids_full is not None:
            if len(pair_ids_full) != len(df.index):
                print(
                    f"Warning: pair_ids length mismatch for gene {gene_name} "
                    f"({len(pair_ids_full)} vs {len(df)}). Skipping this gene."
                )
                continue
            groups = pair_ids_full
            unique_groups = np.unique(groups)
            if len(unique_groups) < 2:
                # Cannot do group-based CV with <2 groups
                continue
            n_splits_effective = min(n_splits, len(unique_groups))
            cv = GroupKFold(n_splits=n_splits_effective)
        else:
            # Fallback: standard KFold if no pairing file is provided
            if len(df) < n_splits:
                continue
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            groups = None

        splitter = cv.split(df, groups=groups) if groups is not None else cv.split(df)

        for fold_idx, (train_idx, _) in enumerate(splitter):
            train_data = df.iloc[train_idx]

            arr_line = f"{gene_name}\t" + "\t".join(train_data["x_log"].astype(str)) + "\n"
            seq_line = f"{gene_name}\t" + "\t".join(train_data["y_log"].astype(str)) + "\n"

            # fold indices go 1..n_splits_effective
            folds_arr[fold_idx + 1].append(arr_line)
            folds_seq[fold_idx + 1].append(seq_line)

    for i in range(1, n_splits + 1):
        arr_out = os.path.join(output_dir_arr, f"fold{i}_arr.txt")
        seq_out = os.path.join(output_dir_seq, f"fold{i}_seq.txt")

        with open(arr_out, "w") as f_arr:
            f_arr.writelines(folds_arr[i])
        with open(seq_out, "w") as f_seq:
            f_seq.writelines(folds_seq[i])

        print(f"  Fold {i} train matrices written to {arr_out} and {seq_out}")

    print(f"Completed training reformatting for {species}.")


if __name__ == "__main__":
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    data_dir = os.path.join(repo_root, "data")
    out_root = os.path.join(repo_root, "results", "harmony2_train_inputs")

    train_configs = [
        {
            "name": "rat",
            "dir": os.path.join(data_dir, "rat", "rat-gene-files"),
            "out_arr": os.path.join(out_root, "rat_harmony2_train_input_arr"),
            "out_seq": os.path.join(out_root, "rat_harmony2_train_input_seq"),
            # new: pairing file path
            "pairing_file": os.path.join(data_dir, "rat", "rat_pairing.txt"),
        },
        {
            "name": "arabidopsis",
            "dir": os.path.join(data_dir, "arabidopsis", "arabidopsis-gene-files"),
            "out_arr": os.path.join(out_root, "arabidopsis_harmony2_train_input_arr"),
            "out_seq": os.path.join(out_root, "arabidopsis_harmony2_train_input_seq"),
            "pairing_file": os.path.join(data_dir, "arabidopsis", "arabidopsis_pairing.txt"),
        },
        {
            "name": "human",
            "dir": os.path.join(data_dir, "human", "human-gene-files"),
            "out_arr": os.path.join(out_root, "human_harmony2_train_input_arr"),
            "out_seq": os.path.join(out_root, "human_harmony2_train_input_seq"),
            "pairing_file": os.path.join(data_dir, "human", "human_pairing.txt"),
        },
    ]

    for config in train_configs:
        reformat_train_folds_for_tdm(config, n_splits=10)