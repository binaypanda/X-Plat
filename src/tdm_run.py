# src/tdm_run.py

import argparse
import yaml
import os

from tdm_prepare import reformat_folds_for_tdm


def main():
    """
    Prepare TDM input matrices from X-Plat fold results.

    Reads per-organism fold results from:
      <output_dir>/<output_prefix>-fold_results/fold{i}.txt

    Writes TDM matrices via reformat_folds_for_tdm to:
      results/tdm_inputs/{species}_tdm_input_arr/fold{i}_arr.txt
      results/tdm_inputs/{species}_tdm_input_seq/fold{i}_seq.txt
    """
    parser = argparse.ArgumentParser(
        description="Prepare TDM input matrices from X-Plat fold results."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    n_splits = cfg.get("n_splits", 10)
    organisms = cfg.get("organisms", {})

    for name, conf in organisms.items():
        output_dir = conf["output_dir"]
        output_prefix = conf["output_prefix"]
        fold_results_dir = os.path.join(output_dir, f"{output_prefix}-fold_results")

        if not os.path.exists(fold_results_dir):
            print(
                f"Fold results directory for {name} not found at '{fold_results_dir}'. Skipping."
            )
            continue

        reformat_folds_for_tdm(
            input_dir=fold_results_dir,
            species_prefix=name,
            n_splits=n_splits,
        )


if __name__ == "__main__":
    main()