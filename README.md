# X-Plat: A polynomial regression-based tool for cross-platform transformation of expression and methylation data

X-Plat is a per-gene supervised polynomial regression method for cross-platform transformation of expression and methylation data. This repository contains the code used to implement and evaluate X-Plat, together with the downstream TDM, HARMONY (Shambhala), and HARMONY2 (Shambhala2) benchmarking pipelines.

Note that HARMONY and HARMONY2 operate on multi-gene expression profiles and are not designed to run on single-gene inputs. In our benchmarking pipelines, these methods are therefore applied at the profile level rather than per gene.

The repository includes our code under the MIT License. Any bundled third-party tools remain subject to their original licenses.

## Repository contents

- `src/xplat_core.py` — core implementation of the per-gene cross-validation pipelines, including polynomial, linear, and identity baselines, plus fold-level and gene-level metrics.
- `src/xplat_run.py` — command-line runner that executes the X-Plat and baseline pipelines for all organisms defined in `config.yaml`.
- `src/tdm_prepare.py`, `src/tdm_run.py` — prepare TDM input matrices from X-Plat fold results and write them to `results/tdm_inputs/`.
- `src/tdm_core.py`, `src/tdm_run_transform.py`, `src/tdm_run_reverse.py` — run forward and reverse TDM transformations via R through `rpy2`, writing outputs to `results/tdm_outputs/`.
- `src/tdm_metrics.py`, `src/tdm_run_metrics.py` — compute TDM metrics and consolidated fold and summary files under `results/tdm/`.
- `src/harmony_run.R`, `src/harmony_preprocess.py`, `src/harmony_run_preprocess.py`, `src/harmony_run_metrics.py` — run HARMONY on TDM-prepared matrices and summarize results under `results/harmony/`.
- `src/harmony2_run.R`, `src/harmony2_metrics.py` — run the HARMONY2/Shambhala2 benchmark using TDM inputs and dedicated train inputs, and compute fold-level and per-gene metrics under `results/harmony2/`.
- `config.yaml` — configuration file specifying organisms, input and output directories, and cross-validation settings.
- `data/` — input per-gene files, one `.txt` per gene, with two columns: `x` and `y`.
- `results/` — all pipeline outputs, organized into subfolders for X-Plat, TDM, HARMONY, and HARMONY2.

## Environment setup

Create a virtual environment and install the Python dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` should include:

```text
rpy2
pandas
numpy
scikit-learn
scipy
PyYAML
```

`rpy2` requires a working R installation, because it is a bridge to an external R runtime rather than a replacement for R itself.

## Input data layout

For each organism, place the per-gene `.txt` files in:

- `data/rat/rat-gene-files/`
- `data/arabidopsis/arabidopsis-gene-files/`
- `data/human/human-gene-files/`

Each file should be a tab-separated file with two columns and no header:

- column 1: source platform values (`x`)
- column 2: target platform values (`y`)

The `config.yaml` file points to these directories and to the X-Plat output locations:

```yaml
random_seed: 42
n_splits: 10
shuffle_folds: true

organisms:
  rat:
    input_dir: "data/rat/rat-gene-files"
    output_dir: "results/xplat/rat"
    output_prefix: "rat"
  arabidopsis:
    input_dir: "data/arabidopsis/arabidopsis-gene-files"
    output_dir: "results/xplat/arabidopsis"
    output_prefix: "arabidopsis"
  human:
    input_dir: "data/human/human-gene-files"
    output_dir: "results/xplat/human"
    output_prefix: "human"
```

This keeps all X-Plat outputs under `results/xplat/<organism>/`, while TDM, HARMONY, and HARMONY2 use their own subfolders under `results/`.

## X-Plat pipeline

Run the X-Plat pipeline from the repository root with:

```bash
python src/xplat_run.py --config config.yaml --methods xplat linear identity
```

For each organism, this command:

- reads the per-gene files from `data/<organism>-gene-files/`
- performs `n_splits`-fold cross-validation using the fixed `random_seed` from `config.yaml`
- uses group-aware fold assignment so paired samples remain in the same fold
- fits per-gene second-degree polynomial models (`xplat`), linear models (`linear`), and identity baselines (`identity`)
- computes RMSE, MAE, normalized RMSE, correlations, and Bland–Altman summaries
- writes per-gene fold-averaged metrics to:

  - `results/xplat/<organism>/<prefix>-results-polynomial-log_transformed.10fold_avg.txt`
  - `results/xplat/<organism>/<prefix>-results-linear-log_transformed.10fold_avg.txt`
  - `results/xplat/<organism>/<prefix>-results-identity-log_transformed.10fold_avg.txt`

- writes per-fold sample-level outputs to:

  - `results/xplat/<organism>/<prefix>-fold_results/fold*.txt` for polynomial
  - `results/xplat/<organism>/<prefix>-fold_results-linear/fold*.txt` for linear
  - `results/xplat/<organism>/<prefix>-fold_results-identity/fold*.txt` for identity

These X-Plat fold results are the starting point for the TDM and HARMONY pipelines.

## TDM pipeline

TDM is an R package accessed from Python through `rpy2`.

### Setup

To run the TDM benchmark:

1. Install R on your system.
2. Install `rpy2` in your Python environment.
3. Ensure the R packages `devtools`, `data.table`, and `TDM` are available.

If `TDM` is missing, the code will attempt to install it automatically with:

```r
devtools::install_github("greenelab/TDM")
```
### Running TDM

From the repository root, run:

```bash
python src/tdm_run.py --config config.yaml
```

This reads the X-Plat fold results under `results/xplat/<organism>/<prefix>-fold_results/` and writes:

- `results/tdm_inputs/<species>_tdm_input_arr/fold*_arr.txt`
- `results/tdm_inputs/<species>_tdm_input_seq/fold*_seq.txt`

Then run forward TDM (`seq -> array`) with:

```bash
python src/tdm_run_transform.py --config config.yaml
```

This writes:

- `results/tdm_outputs/<species>_tdm_output/fold*_tdm_transformed.txt`

Run reverse TDM (`array -> seq`) with:

```bash
python src/tdm_run_reverse.py --config config.yaml
```

This writes:

- `results/tdm_outputs/<species>_tdm_output_reverse/fold*_tdm_transformed.txt`

Finally, compute TDM metrics and consolidated summaries with:

```bash
python src/tdm_run_metrics.py --config config.yaml
```

This produces:

- fold-wise TDM metrics in `results/tdm/<species>_tdm_fold_results/fold*.txt`
- per-gene summary metrics in `results/tdm/<species>-results-tdm-log_transformed.10fold_avg.txt`

## HARMONY pipeline

HARMONY is an R package for integrating multi-platform or multi-batch data. In this repository, HARMONY operates directly on the TDM-prepared array and sequence matrices.

### Installation

Install the required R packages, including `harmony`, `cluster`, and their dependencies.

### Running HARMONY

From the repository root, run:

```bash
Rscript src/harmony_run.R results/tdm_inputs
```

This creates raw HARMONY outputs:

- `results/harmony/<species>_harmony_output/fold*.txt`

Then fix tab encoding in those outputs with:

```bash
python src/harmony_run_preprocess.py --config config.yaml
```

This converts literal `\t` sequences to real tab characters in the HARMONY prediction files.

Finally, compute HARMONY metrics and consolidated summaries with:

```bash
python src/harmony_run_metrics.py --config config.yaml --base-dir results
```

This produces:

- fold-wise metrics in `results/harmony/<species>_harmony_fold_results/fold*.txt`
- per-gene HARMONY summaries in `results/harmony/<species>-results-harmony-log_transformed.10fold_avg.txt`

The HARMONY sequence is:

1. Run TDM prepare to generate HARMONY inputs.
2. Run `harmony_run.R` to generate raw HARMONY outputs.
3. Run `harmony_run_preprocess.py` to fix tab encoding.
4. Run `harmony_run_metrics.py` to compute metrics and summaries.

## HARMONY2 / Shambhala2 pipeline

HARMONY2, also known as Shambhala2, is used here as a cross-platform harmonization benchmark. In this repository, HARMONY2 operates on TDM-prepared inputs and writes per-species outputs under `results/harmony2/shambhala2_outputs/`.

### Generate HARMONY2 inputs
```bash
python src/tdm_train_prepare.py
```
### Running Shambhala2

You can run the HARMONY2/Shambhala2 benchmark either from RStudio or from the command line. In both cases, the working directory must be `src/Shambhala2`, because `harmony2_run.R` uses paths relative to that folder.

#### Option 1: From RStudio (interactive)

1. Open RStudio.
2. Set the working directory to the Shambhala2 folder inside `src`:
   ```r
   setwd("src/Shambhala2")
   ```
3. Run the HARMONY2 pipeline by sourcing the script:
   ```r
   source("harmony2_run.R")
   ```

This will:

- read HARMONY2 training inputs from `results/harmony2_train_inputs/`
- use TDM-prepared matrices from `results/tdm_inputs/`
- write harmonized Shambhala2 outputs under:
  - `results/harmony2/shambhala2_outputs/<species>/seq_to_arr/fold{i}_arr.txt`
  - `results/harmony2/shambhala2_outputs/<species>/arr_to_seq/fold{i}_seq.txt`

#### Option 2: From the command line with Rscript

If you prefer to run non-interactively, you can use `Rscript` from a system terminal (PowerShell / CMD on Windows, or a shell on macOS/Linux):

```bash
cd src/Shambhala2

Rscript ../harmony2_run.R
```

Both options produce Shambhala2 outputs under `results/harmony2/shambhala2_outputs/` with the same folder structure; choose the one that fits your workflow.

### HARMONY2 metrics and summaries

Once Shambhala2 has produced its outputs, run:

```bash
python src/harmony2_metrics.py
```

This script reads:

- raw TDM inputs from `results/tdm_inputs/<species>_tdm_input_arr/fold{i}_arr.txt`
- raw TDM inputs from `results/tdm_inputs/<species>_tdm_input_seq/fold{i}_seq.txt`
- HARMONY2 predictions from:

  - `results/harmony2/shambhala2_outputs/<species>/seq_to_arr/fold{i}_arr.txt`
  - `results/harmony2/shambhala2_outputs/<species>/arr_to_seq/fold{i}_seq.txt`

- optional per-gene dynamic ranges from:

  - `results/xplat/<species>-results-polynomial-log_transformed.10fold_avg.txt`

It then computes normalized RMSE using the log-scale ranges.

For each species, `harmony2_metrics.py` produces:

- fold-wise metrics in `results/harmony2/<species>_harmony2_fold_results/fold{i}.txt`
- per-gene summary metrics in `results/harmony2/<species>-results-harmony2-log_transformed.10fold_avg.txt`

The HARMONY2 sequence is:

1. Run TDM prepare to generate HARMONY2 inputs.
2. Run `harmony2_run.R` to generate Shambhala2 harmonized outputs.
3. Run `harmony2_metrics.py` to generate fold-wise metrics and per-gene summaries.

## Reproducibility

- Cross-validation uses `n_splits = 10`, `shuffle_folds = true`, and a fixed `random_seed` from `config.yaml`.
- Fold assignment is group-aware, so paired samples remain in the same fold.
- All paths are relative to the repository root.
- The pipeline for rat, arabidopsis, and human uses the same code, differing only in `input_dir`, `output_dir`, and `output_prefix` entries in `config.yaml`.

## Sample data and example results

This repository includes a small amount of bundled data so you can run the pipeline end-to-end without downloading external datasets.

- `data/rat-gene-files/` contains the full set of per-gene input files for **rat**, and can be used as a complete sample dataset.
- `data/arabidopsis-gene-files/` and `data/human-gene-files/` each contain per-gene input files for a **single gene**, provided as minimal examples for **Arabidopsis** and **human**.
- The `results/` directory also includes example outputs for a single gene for all three organisms, so you can inspect the expected output structure and metrics when verifying your own runs.

These sample inputs and reference results are intended for testing the workflow and understanding the output formats; for full-scale analyses, replace them with your own data.

## Reproducibility

- Cross-validation uses `n_splits = 10`, `shuffle_folds = true`, and a fixed `random_seed` from `config.yaml`.
- Fold assignment is group-aware, so paired samples remain in the same fold.
- All paths are relative to the repository root.
- The pipeline for rat, arabidopsis, and human uses the same code, differing only in `input_dir`, `output_dir`, and `output_prefix` entries in `config.yaml`.

A Colab notebook is included as an optional helper for exploring the workflow and reproducing small examples. For further methodological details, refer to the Methods section of the associated manuscript.
