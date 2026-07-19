import argparse
import os
import yaml

from harmony_preprocess import preprocess_harmony_outputs


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess HARMONY prediction files (fix literal \\t to real tabs)."
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
        help="Base directory for HARMONY outputs.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    organisms = cfg.get("organisms", {})
    species_list = list(organisms.keys())
    n_splits = cfg.get("n_splits", 10)

    preprocess_harmony_outputs(
        species_list=species_list,
        base_dir=os.path.abspath(args.base_dir),
        n_splits=n_splits,
    )


if __name__ == "__main__":
    main()