# src/xplat_run.py

import argparse
import yaml

from xplat_core import (
    run_xplat_for_organism,
    run_linear_for_organism,
    run_identity_for_organism,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run X-Plat and baselines for all organisms."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["xplat", "linear"],
        help="Methods to run: xplat, linear, identity",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    random_seed = cfg.get("random_seed", 42)
    n_splits = cfg.get("n_splits", 10)
    organisms = cfg.get("organisms", {})

    for name, conf in organisms.items():
        input_dir = conf["input_dir"]
        output_dir = conf["output_dir"]
        output_prefix = conf["output_prefix"]
        # new: optional pairing file per organism
        pairing_file = conf.get("pairing_file", None)

        if "xplat" in args.methods:
            run_xplat_for_organism(
                name=name,
                input_dir=input_dir,
                output_dir=output_dir,
                output_prefix=output_prefix,
                n_splits=n_splits,
                random_seed=random_seed,
                pairing_file=pairing_file,
            )

        if "linear" in args.methods:
            run_linear_for_organism(
                name=name,
                input_dir=input_dir,
                output_dir=output_dir,
                output_prefix=output_prefix,
                n_splits=n_splits,
                random_seed=random_seed,
                pairing_file=pairing_file,
            )

        if "identity" in args.methods:
            run_identity_for_organism(
                name=name,
                input_dir=input_dir,
                output_dir=output_dir,
                output_prefix=output_prefix,
                n_splits=n_splits,
                random_seed=random_seed,
                pairing_file=pairing_file,
            )

    print("All organism processing complete.")


if __name__ == "__main__":
    main()