import os
import rpy2.robjects as robjects


def execute_tdm_transformation(species_list, base_dir="results", n_splits=10):
    """
    Run the TDM R package on the reformatted matrix files for seq -> array.

    Expects (relative to repo_root/results):
      - tdm_inputs/{species}_tdm_input_arr/fold{i}_arr.txt
      - tdm_inputs/{species}_tdm_input_seq/fold{i}_seq.txt

    Writes:
      - tdm_outputs/{species}_tdm_output/fold{i}_tdm_transformed.txt
    """

    robjects.r('''
        if (!requireNamespace("BiocManager", quietly = TRUE))
            install.packages("BiocManager", repos="https://cran.rstudio.com")
        if (!requireNamespace("devtools", quietly = TRUE))
            install.packages("devtools", repos="https://cran.rstudio.com")
        if (!requireNamespace("TDM", quietly = TRUE))
            devtools::install_github("greenelab/TDM")
        library(TDM)
        library(data.table)
    ''')

    tdm_inputs_root = os.path.join(base_dir, "tdm_inputs")
    tdm_outputs_root = os.path.join(base_dir, "tdm_outputs")

    for species in species_list:
        print(f"Running TDM for {species}...")
        output_dir = os.path.join(tdm_outputs_root, f"{species}_tdm_output")
        os.makedirs(output_dir, exist_ok=True)

        input_arr_dir = os.path.join(tdm_inputs_root, f"{species}_tdm_input_arr")
        input_seq_dir = os.path.join(tdm_inputs_root, f"{species}_tdm_input_seq")

        for i in range(1, n_splits + 1):
            ref_file = os.path.join(input_arr_dir, f"fold{i}_arr.txt")
            target_file = os.path.join(input_seq_dir, f"fold{i}_seq.txt")
            output_file = os.path.join(output_dir, f"fold{i}_tdm_transformed.txt")

            if not (os.path.exists(ref_file) and os.path.exists(target_file)):
                continue

            robjects.r(f'''
                ref_data <- fread("{ref_file}", header=FALSE)
                target_data <- fread("{target_file}", header=FALSE)

                if (nrow(ref_data) > 0 && nrow(target_data) > 0) {{
                    setnames(ref_data, 1, "gene")
                    setnames(target_data, 1, "gene")

                    common_genes <- intersect(ref_data$gene, target_data$gene)
                    ref_subset <- ref_data[gene %in% common_genes]
                    target_subset <- target_data[gene %in% common_genes]

                    tdm_result <- tdm_transform(
                        target_data   = target_subset,
                        ref_data      = ref_subset,
                        inv_reference = FALSE,
                        log_target    = FALSE
                    )

                    write.table(
                        tdm_result,
                        "{output_file}",
                        sep = "\\t",
                        row.names = FALSE,
                        quote = FALSE
                    )
                }}
            ''')
            if i % 2 == 0:
                print(f"  Fold {i} complete...")

    print("TDM transformation across all species and folds complete.")


def execute_tdm_reverse(species_list, base_dir="results", n_splits=10):
    """
    Run TDM for array -> seq (predicting seq distribution from array inputs).

    Expects (relative to repo_root/results):
      - tdm_inputs/{species}_tdm_input_seq/fold{i}_seq.txt
      - tdm_inputs/{species}_tdm_input_arr/fold{i}_arr.txt

    Writes:
      - tdm_outputs/{species}_tdm_output_reverse/fold{i}_tdm_transformed.txt
    """

    # Assume TDM and data.table already loaded; otherwise:
    robjects.r('library(TDM); library(data.table)')

    tdm_inputs_root = os.path.join(base_dir, "tdm_inputs")
    tdm_outputs_root = os.path.join(base_dir, "tdm_outputs")

    for species in species_list:
        print(f"Running reverse TDM (arr -> seq) for {species}...")
        output_dir = os.path.join(tdm_outputs_root, f"{species}_tdm_output_reverse")
        os.makedirs(output_dir, exist_ok=True)

        input_seq_dir = os.path.join(tdm_inputs_root, f"{species}_tdm_input_seq")
        input_arr_dir = os.path.join(tdm_inputs_root, f"{species}_tdm_input_arr")

        for i in range(1, n_splits + 1):
            ref_file = os.path.join(input_seq_dir, f"fold{i}_seq.txt")
            target_file = os.path.join(input_arr_dir, f"fold{i}_arr.txt")
            output_file = os.path.join(output_dir, f"fold{i}_tdm_transformed.txt")

            if not (os.path.exists(ref_file) and os.path.exists(target_file)):
                continue

            robjects.r(f'''
                ref_data <- fread("{ref_file}", header=FALSE)
                target_data <- fread("{target_file}", header=FALSE)

                setnames(ref_data, 1, "gene")
                setnames(target_data, 1, "gene")

                common_genes <- intersect(ref_data$gene, target_data$gene)
                ref_subset <- ref_data[gene %in% common_genes]
                target_subset <- target_data[gene %in% common_genes]

                tdm_result <- tdm_transform(
                    target_data   = target_subset,
                    ref_data      = ref_subset,
                    inv_reference = FALSE,
                    log_target    = FALSE
                )

                write.table(
                    tdm_result,
                    "{output_file}",
                    sep = "\\t",
                    row.names = FALSE,
                    quote = FALSE
                )
            ''')

    print("Reverse TDM (arr -> seq) complete.")