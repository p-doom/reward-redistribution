import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tbparse import SummaryReader
import logging

# --- Configuration ---
SLURM_LOG_ROOT = "log_slurm_njobs_IRR_tuning" # The top-level directory containing Slurm job subdirs
# Base path for resolving TB paths found in .out files.
# Assumes the script is run from the project root where 'results/...' exists.
# If not, change this to the absolute path of your project root.
PROJECT_ROOT = "."
AGGREGATED_OUTPUT_DIR = "results/tensorboard_aggregated_IRR_tuning" # Optional: Where to save aggregated plots/data
PLOT_LOCALLY = False
# Optional: Set to True to write aggregated mean/std back to new TB event files
WRITE_AGGREGATED_TB_LOGS = True
if WRITE_AGGREGATED_TB_LOGS:
    try:
        import tensorflow as tf
    except ImportError:
        logging.warning("tensorflow not found. Cannot write aggregated TB logs. Install it (`pip install tensorflow`)")
        WRITE_AGGREGATED_TB_LOGS = False
# --- End Configuration ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_tb_paths_in_file(filepath):
    """Searches a file for TensorBoard log paths."""
    paths = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Regex to find the specific logging pattern
                match = re.search(r'Logging to tensorboard in path:\s*(.*)', line)
                if match:
                    path = match.group(1).strip()
                    # Make path relative to project root if necessary
                    # Assuming paths in .out are already relative to PROJECT_ROOT
                    full_path = os.path.join(PROJECT_ROOT, path)
                    if os.path.isdir(full_path): # Check if the directory actually exists
                         paths.append(full_path)
                    else:
                         logging.warning(f"Found path {path} in {filepath}, but directory {full_path} does not exist. Skipping.")

    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
    except Exception as e:
        logging.error(f"Error reading {filepath}: {e}")
    return paths

def aggregate_runs(tb_paths, slurm_job_name):
    """Loads data from multiple TB paths, aggregates, and optionally plots/saves."""
    if not tb_paths:
        logging.warning(f"No valid TensorBoard paths found for {slurm_job_name}. Skipping aggregation.")
        return

    logging.info(f"Aggregating {len(tb_paths)} runs for job '{slurm_job_name}': {tb_paths}")

    all_scalars_df = pd.DataFrame()
    run_map = {} # Store mapping from simple run index to actual tb_path

    # 1. Load data from all runs in the group
    for i, path in enumerate(tb_paths):
        run_id = f"run_{i}"
        run_map[run_id] = path
        try:
            reader = SummaryReader(path)

            # Use scalars attribute which returns a long-format DataFrame
            df_run = reader.scalars
            if df_run.empty:
                 logging.warning(f"No scalar data found in {path}")
                 continue

            # Add a check for expected columns in long format
            if not {'step', 'tag', 'value'}.issubset(df_run.columns):
                 logging.error(f"Expected columns 'step', 'tag', 'value' not found in data from {path}. Columns found: {df_run.columns}. Skipping this run.")
                 continue

            df_run['run_id'] = run_id # Add identifier for this specific run
            all_scalars_df = pd.concat([all_scalars_df, df_run], ignore_index=True)
            logging.info(f"Successfully loaded {len(df_run)} scalar points from {path}")
        except Exception as e:
            logging.error(f"Could not read or process data from {path}: {e}")

    if all_scalars_df.empty:
        logging.warning(f"No scalar data loaded for job '{slurm_job_name}'. Cannot aggregate.")
        return

    # Add a check here just to be safe, before accessing 'tag'
    if 'tag' not in all_scalars_df.columns:
        logging.error(f"Critical error: 'tag' column is missing after loading and concatenating data for job '{slurm_job_name}'. Columns present: {all_scalars_df.columns}")
        return


    # 2. Aggregate per tag
    # This part of the code should now work correctly as it expects the 'tag' column
    aggregated_results = {}
    unique_tags = all_scalars_df['tag'].unique()
    logging.info(f"Found tags for job '{slurm_job_name}': {unique_tags}")


    for tag in unique_tags:
        tag_df = all_scalars_df[all_scalars_df['tag'] == tag].copy()

        # Ensure 'step' is numeric for interpolation/aggregation
        tag_df['step'] = pd.to_numeric(tag_df['step'], errors='coerce')
        tag_df.dropna(subset=['step'], inplace=True) # Remove rows where step couldn't be converted
        tag_df['step'] = tag_df['step'].astype(int)


        # Pivot table to have steps as index and runs as columns
        try:
            # Keep only necessary columns before pivoting
            pivot_df = tag_df[['step', 'run_id', 'value']].pivot_table(
                index='step', columns='run_id', values='value'
            )
        except Exception as e:
            logging.error(f"Could not pivot data for tag '{tag}' in job '{slurm_job_name}': {e}. Skipping tag.")
            continue

        # Optional: Interpolate missing values if steps don't align perfectly
        # pivot_df = pivot_df.interpolate(method='index', limit_direction='both') # Linear interpolation based on step index

        # Calculate mean and std dev across runs (columns) for each step (row)
        # skipna=True is default and handles cases where runs end early/steps mismatch without interpolation
        agg_df = pd.DataFrame({
            'mean': pivot_df.mean(axis=1, skipna=True),
            'std': pivot_df.std(axis=1, skipna=True),
            'count': pivot_df.count(axis=1), # Number of runs with data at this step
            'min': pivot_df.min(axis=1, skipna=True),
            'max': pivot_df.max(axis=1, skipna=True),
        })
        # Remove steps where aggregation is meaningless (e.g., only one run had data)
        # agg_df = agg_df[agg_df['count'] > 1]

        # Handle cases where std is NaN (e.g., only one run contributed at a step)
        agg_df['std'] = agg_df['std'].fillna(0)

        aggregated_results[tag] = agg_df
        logging.info(f"Aggregated tag '{tag}' for job '{slurm_job_name}' over {agg_df.index.min()}-{agg_df.index.max()} steps.")

    # 3. Output / Plot / Save Aggregated Results
    os.makedirs(AGGREGATED_OUTPUT_DIR, exist_ok=True)
    job_output_dir = os.path.join(AGGREGATED_OUTPUT_DIR, slurm_job_name)
    os.makedirs(job_output_dir, exist_ok=True)

    # --- Plotting (Optional) ---
    if PLOT_LOCALLY:
        for tag, agg_df in aggregated_results.items():
            if agg_df.empty: continue

            plt.figure(figsize=(12, 6))
            plt.plot(agg_df.index, agg_df['mean'], label='Mean')
            plt.fill_between(agg_df.index,
                            agg_df['mean'] - agg_df['std'],
                            agg_df['mean'] + agg_df['std'],
                            alpha=0.3, label='Mean ± 1 Std Dev')
            plt.xlabel("Step")
            plt.ylabel(tag.replace('_', ' ').title()) # Prettier label
            plt.title(f"Aggregated '{tag}' for Job '{slurm_job_name}' ({len(tb_paths)} runs)")
            plt.legend()
            plt.grid(True)
            # Sanitize tag name for filename
            safe_tag_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', tag)
            plot_filename = os.path.join(job_output_dir, f"aggregated_{safe_tag_name}.png")
            try:
                plt.savefig(plot_filename)
                logging.info(f"Saved aggregated plot to {plot_filename}")
            except Exception as e:
                logging.error(f"Failed to save plot {plot_filename}: {e}")
            plt.close() # Close the plot to free memory

            # --- Save aggregated data to CSV (Optional) ---
            csv_filename = os.path.join(job_output_dir, f"aggregated_{safe_tag_name}.csv")
            try:
                agg_df.to_csv(csv_filename)
                logging.info(f"Saved aggregated data to {csv_filename}")
            except Exception as e:
                logging.error(f"Failed to save CSV {csv_filename}: {e}")

    # --- Write back to TensorBoard (Optional) ---
    if WRITE_AGGREGATED_TB_LOGS and aggregated_results:
        tb_writer_path = os.path.join(job_output_dir, 'tb_summary')
        logging.info(f"Writing aggregated TensorBoard logs to {tb_writer_path}")
        writer = tf.summary.create_file_writer(tb_writer_path)
        with writer.as_default():
            for tag, agg_df in aggregated_results.items():
                if agg_df.empty: continue
                for step, row in agg_df.iterrows():
                     # Log mean and bounds - choose names carefully to avoid clashes
                     tf.summary.scalar(f"{tag}/mean", row['mean'], step=step)
                     tf.summary.scalar(f"{tag}/std", row['std'], step=step)
                     tf.summary.scalar(f"{tag}/mean_plus_std", row['mean'] + row['std'], step=step)
                     tf.summary.scalar(f"{tag}/mean_minus_std", row['mean'] - row['std'], step=step)
                     tf.summary.scalar(f"{tag}/count", row['count'], step=step)
        writer.close()

# --- Main Script Logic ---
if __name__ == "__main__":
    if not os.path.isdir(SLURM_LOG_ROOT):
        logging.error(f"Slurm log root directory not found: {SLURM_LOG_ROOT}")
        exit(1)

    slurm_job_dirs = [d for d in glob.glob(os.path.join(SLURM_LOG_ROOT, '*')) if os.path.isdir(d)]
    logging.info(f"Found {len(slurm_job_dirs)} potential job directories in {SLURM_LOG_ROOT}")

    for job_dir in slurm_job_dirs:
        job_name = os.path.basename(job_dir)
        logging.info(f"Processing directory: {job_dir}")

        # 2. Check for overrides.json
        overrides_file = os.path.join(job_dir, "overrides.json")
        if not os.path.exists(overrides_file):
            logging.info(f"  Skipping '{job_name}': 'overrides.json' not found.")
            continue
        logging.info(f"  Found 'overrides.json' in '{job_name}'.")

        # 3. Find the .out file
        out_files = glob.glob(os.path.join(job_dir, '*.out'))
        if not out_files:
            logging.warning(f"  Skipping '{job_name}': No '.out' file found in {job_dir}")
            continue
        # If multiple .out files exist, process the first one found. Adjust if needed.
        out_file_path = out_files[0]
        if len(out_files) > 1:
             logging.warning(f"  Multiple '.out' files found in {job_dir}. Using '{out_file_path}'.")

        # 4. Find TensorBoard paths in the .out file
        logging.info(f"  Searching for TensorBoard paths in {out_file_path}...")
        tensorboard_paths = find_tb_paths_in_file(out_file_path)

        # 5. Aggregate logs from these paths
        if tensorboard_paths:
             aggregate_runs(tensorboard_paths, job_name)
        else:
             logging.warning(f"  No valid TensorBoard paths found in '{out_file_path}' for job '{job_name}'.")

    logging.info("Aggregation process finished.")