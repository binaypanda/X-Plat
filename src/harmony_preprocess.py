import argparse
import os
import subprocess


def preprocess_harmony_outputs(species_list, base_dir="results", n_splits=10):
    print("Preprocessing HARMONY prediction files...")

    harmony_root = os.path.join(base_dir, "harmony")

    for species in species_list:
        harmony_output_dir = os.path.join(harmony_root, f"{species}_harmony_output")
        if not os.path.exists(harmony_output_dir):
            print(f"  Warning: HARMONY output directory not found for {species}: {harmony_output_dir}")
            continue

        print(f"  Processing files in {harmony_output_dir} for {species}...")
        for i in range(1, n_splits + 1):
            harmony_pred_path = os.path.join(harmony_output_dir, f"fold{i}.txt")
            if os.path.exists(harmony_pred_path):
                subprocess.run(
                    ["sed", "-i", r"s/\\t/\t/g", harmony_pred_path],
                    check=True,
                )
                if i == 1:
                    print(f"    Preprocessed {os.path.basename(harmony_pred_path)}")
            else:
                print(f"    Warning: File not found: {harmony_pred_path}")

    print("HARMONY prediction file preprocessing complete.")