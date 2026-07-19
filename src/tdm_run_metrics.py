import argparse
import os
import yaml

from tdm_metrics import create_tdm_consolidated_files


def main():
    parser = argparse.ArgumentParser(
        description="Compute TDM metrics and consolidated fold/summary files for all organisms."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file describing organisms and n_splits.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="results",
        help="Base directory where TDM input/output folders and polynomial summaries live.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    n_splits = cfg.get("n_splits", 10)
    organisms = cfg.get("organisms", {})
    species_list = list(organisms.keys())

    create_tdm_consolidated_files(
        species_list=species_list,
        base_dir=os.path.abspath(args.base_dir),
        n_splits=n_splits,
    )


if __name__ == "__main__":
    main()