import argparse
import os
import yaml

from tdm_core import execute_tdm_reverse


def main():
    parser = argparse.ArgumentParser(
        description="Run reverse TDM (arr -> seq) for all organisms."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="results",
        help="Base directory for TDM input/output folders.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    n_splits = cfg.get("n_splits", 10)
    organisms = cfg.get("organisms", {})
    species_list = list(organisms.keys())

    execute_tdm_reverse(
        species_list=species_list,
        base_dir=os.path.abspath(args.base_dir),
        n_splits=n_splits,
    )


if __name__ == "__main__":
    main()