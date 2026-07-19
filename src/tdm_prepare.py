# src/tdm_prepare.py

import os


def reformat_folds_for_tdm(input_dir: str, species_prefix: str, n_splits: int = 10):
    """
    Convert X-Plat fold result files into matrix format for TDM:
    GeneName\tSample1\tSample2...

    input_dir: directory containing fold{i}.txt for this species.
    """
    print(f"Reformatting folds for {species_prefix}...")

    # Base: repo root / results / tdm_inputs
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    tdm_out_root = os.path.join(repo_root, "results", "tdm_inputs")

    output_dir_arr = os.path.join(tdm_out_root, f"{species_prefix}_tdm_input_arr")
    output_dir_seq = os.path.join(tdm_out_root, f"{species_prefix}_tdm_input_seq")

    os.makedirs(output_dir_arr, exist_ok=True)
    os.makedirs(output_dir_seq, exist_ok=True)

    for i in range(1, n_splits + 1):
        fold_file = os.path.join(input_dir, f"fold{i}.txt")
        if not os.path.exists(fold_file):
            continue

        arr_matrix_lines = []
        seq_matrix_lines = []

        with open(fold_file, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                gene_name = parts[0]

                # For each gene line:
                # gene_name, then flat sample data (6 values per sample),
                # then 18 metric columns at the end.
                if len(parts) <= 1:
                    samples_raw = []
                else:
                    samples_raw = parts[1:-18]

                x_vals = []
                y_vals = []
                for j in range(0, len(samples_raw), 6):
                    if j + 3 < len(samples_raw):
                        # x_log at position j, y_log at position j+3
                        x_vals.append(samples_raw[j])
                        y_vals.append(samples_raw[j + 3])
                    else:
                        print(
                            f"Warning: malformed sample data for gene {gene_name} in fold {i}. "
                            "Skipping remaining samples."
                        )
                        break

                arr_matrix_lines.append(f"{gene_name}\t" + "\t".join(x_vals) + "\n")
                seq_matrix_lines.append(f"{gene_name}\t" + "\t".join(y_vals) + "\n")

        arr_out = os.path.join(output_dir_arr, f"fold{i}_arr.txt")
        seq_out = os.path.join(output_dir_seq, f"fold{i}_seq.txt")
        with open(arr_out, "w") as f_arr:
            f_arr.writelines(arr_matrix_lines)
        with open(seq_out, "w") as f_seq:
            f_seq.writelines(seq_matrix_lines)

    print(f"Completed reformatting for {species_prefix}.")