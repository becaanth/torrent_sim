import argparse
import glob
import json
import os
import pandas as pd

def convert_manifest_to_summary(morris_dir, out_csv):
    rows = []
    # Search for manifest.jsonl in single or multi-strategy subdirectories
    manifest_paths = glob.glob(os.path.join(morris_dir, "**", "manifest.jsonl"), recursive=True)
    if not manifest_paths:
        manifest_paths = glob.glob(os.path.join(morris_dir, "manifest.jsonl"))

    if not manifest_paths:
        raise FileNotFoundError(f"No manifest.jsonl files found in '{morris_dir}'")

    for path in manifest_paths:
        # Extract strategy name if nested under subdirectories (e.g., results/morris_study/rarest_random/manifest.jsonl)
        rel_path = os.path.relpath(path, morris_dir)
        dir_parts = os.path.dirname(rel_path).split(os.sep)
        strategy = dir_parts[0] if dir_parts[0] and dir_parts[0] != "." else "default"

        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                row = {
                    "trial_index": data.get("trial_index"),
                    "strategy": strategy,
                    "role": "TARGET",  # Role filter used during morris reduction
                }
                # Flatten parameter inputs and output metrics
                row.update(data.get("params", {}))
                row.update(data.get("outputs", {}))
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Successfully converted {len(df)} trials to '{out_csv}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Morris manifest.jsonl to plot_sweep summary.csv")
    parser.add_argument("morris_dir", help="Directory containing Morris output (e.g., results/morris_base_scenario)")
    parser.add_argument("--out", default="summary.csv", help="Output CSV path (default: summary.csv)")
    args = parser.parse_args()

    convert_manifest_to_summary(args.morris_dir, args.out)