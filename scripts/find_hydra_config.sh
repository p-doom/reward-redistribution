#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <TensorBoard_Run_ID>"
    echo "Error: TensorBoard run ID not provided."
    exit 1
fi

TB_RUN="$1"

HYDRA_MULTIRUN_BASE_DIR="multirun/"

if [[ ! -d "$HYDRA_MULTIRUN_BASE_DIR" ]]; then
    echo "Error: Hydra multirun directory '$HYDRA_MULTIRUN_BASE_DIR' not found."
    exit 1
fi

# We pipe to head -n 1 to take the first match if multiple exist. Adjust if needed.
MATCHING_LOG_FILE=$(grep -rl --include='*.out' "$TB_RUN" "$HYDRA_MULTIRUN_BASE_DIR" | head -n 1)

if [[ -n "$MATCHING_LOG_FILE" ]]; then
  echo "Found matching Submitit log file: $MATCHING_LOG_FILE"

  # --- Derive Hydra run directory from Submitit log path ---
  # Example MATCHING_LOG_FILE: multirun/DATE/TIME/.submitit/JOB_ARRAY/JOB_ARRAY_TASK_log.out
  # Example Target Hydra Dir:  multirun/DATE/TIME/TASK/

  SUBMITIT_LOG_DIR=$(dirname "$MATCHING_LOG_FILE")
  MULTIRUN_TIMESTAMP_DIR=$(dirname "$(dirname "$SUBMITIT_LOG_DIR")")

  # Extract the task ID from the log filename (e.g., from 1348727_0_0_log.out -> 0)
  LOG_FILENAME=$(basename "$MATCHING_LOG_FILE")
  # Remove the suffix _log.out
  LOG_BASENAME=${LOG_FILENAME%_log.out}
  # Get the part after the last underscore (this should be the task ID)
  TASK_ID=${LOG_BASENAME##*_}

  # Construct the path to the actual Hydra run directory
  HYDRA_RUN_DIR="$MULTIRUN_TIMESTAMP_DIR/$TASK_ID"

  if [[ ! -d "$HYDRA_RUN_DIR" ]]; then
      echo "Error: Derived Hydra run directory '$HYDRA_RUN_DIR' does not exist."
      echo "       Based on Submitit log: $MATCHING_LOG_FILE"
      echo "       Extracted Task ID: $TASK_ID"
      exit 1
  fi

  # Construct the path to the .hydra directory
  HYDRA_CONFIG_DIR="$HYDRA_RUN_DIR/.hydra"

  if [[ -d "$HYDRA_CONFIG_DIR" ]]; then
    echo "Associated Hydra run directory:    $HYDRA_RUN_DIR"
    echo "Associated Hydra config directory: $HYDRA_CONFIG_DIR"
    echo "--------------------------------------------------"
    echo "Config path:    $HYDRA_CONFIG_DIR/config.yaml"
    echo "Overrides path: $HYDRA_CONFIG_DIR/overrides.yaml"
    echo "--------------------------------------------------"
  else
    echo "Error: Found Submitit log file '$MATCHING_LOG_FILE' and derived Hydra run directory '$HYDRA_RUN_DIR',"
    echo "       but could not find expected '.hydra' directory within it."
  fi
else
  echo "--------------------------------------------------"
  echo "Error: No Submitit log file (.out) found containing the run '$TB_RUN' within '$HYDRA_MULTIRUN_BASE_DIR'."
  echo "Please check:"
  echo "1. The TB_RUN variable in the script is set correctly."
  echo "2. The HYDRA_MULTIRUN_BASE_DIR variable points to the correct location."
  echo "3. Your script actually printed the TensorBoard path to the standard output/error during the run (check .out files)."
  echo "--------------------------------------------------"
fi
