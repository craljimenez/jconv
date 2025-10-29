import argparse
import pandas as pd
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(
        description="Unify multiple metric CSV files from the results folder into a single CSV file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default="results",
        help="Directory containing the CSV files to unify."
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default="results/unified_metrics.csv",
        help="Path to the output unified CSV file."
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: Input directory not found at '{args.input_dir}'")
        return

    csv_files = sorted(list(args.input_dir.glob("metrics_*.csv")))

    if not csv_files:
        print(f"No metric CSV files found in '{args.input_dir}'.")
        return

    all_dfs = []
    print(f"Found {len(csv_files)} CSV files to process.")

    # Regex to parse: metrics_${model_type}${model_suffix}_${cam_type}_layer-${layer_slug}.csv
    # It captures model_type, an optional model_suffix, and cam_type.
    # Example: metrics_unet_hybrid_orth_gradcam_layer-default.csv
    # -> model_type='unet_hybrid', model_suffix='_orth', cam_type='gradcam'
    # Example: metrics_unet_gradcam_layer-default.csv
    # -> model_type='unet', model_suffix='', cam_type='gradcam'
    pattern = re.compile(r"metrics_(?P<model_type>.*?)(?P<model_suffix>_orth)?_(?P<cam_type>gradcam\+\+|gradcam|scorecam)_layer-.*")

    for csv_file in csv_files:
        match = pattern.match(csv_file.name)
        if match:
            info = match.groupdict()
            model_type = info.get("model_type", "unknown")
            model_suffix = (info.get("model_suffix") or "").strip('_') or "none"
            cam_type = info.get("cam_type", "unknown")
        else:
            model_type = "unknown"
            model_suffix = "unknown"
            cam_type = "unknown"
            print(f"Warning: Could not extract info from filename: {csv_file.name}")

        df = pd.read_csv(csv_file)
        df["model_type"] = model_type
        df["model_suffix"] = model_suffix
        df["cam_type"] = cam_type
        all_dfs.append(df)

    if not all_dfs:
        print("No dataframes to concatenate.")
        return

    unified_df = pd.concat(all_dfs, ignore_index=True)

    # Reorder columns to have model_type, model_suffix, and cam_type appear early
    cols = unified_df.columns.tolist()
    new_order = ["model_type", "model_suffix", "cam_type"]
    for col_name in reversed(new_order):
        if col_name in cols:
            cols.insert(0, cols.pop(cols.index(col_name)))
    unified_df = unified_df[cols]

    # Create parent directory for output file if it doesn't exist
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Save the unified dataframe
    unified_df.to_csv(args.output_csv, index=False)
    print(f"Successfully created unified CSV file at: {args.output_csv}")


if __name__ == "__main__":
    main()
