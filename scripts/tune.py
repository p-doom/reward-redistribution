import optuna
import submitit
import time
import os  # For getting process ID for logging
import stoix.systems.ppo.anakin.ff_ppo as ppo
import hydra
from omegaconf import DictConfig, OmegaConf
# ==============================================================================
# Configuration
# ==============================================================================

# --- Optuna Storage ---
# REQUIRED for n_jobs > 1 (multiprocessing).
# Use a file path accessible by all potential processes.
# If running n_jobs on a single machine, a local path is fine.
# If workers might run on different cluster nodes, use a path on a
# shared network filesystem (e.g., NFS home directory).
STORAGE_URL = "sqlite:///my_hpo_study_njobs.db"
STUDY_NAME = "hpo_study_njobs_v1" # Choose a descriptive name

SEEDS=[1,2,3,4,5]
ENVS=[
    # TODO add which we want to tune on 
    "navix/four_rooms",
    "navix/four_rooms_7x7",
    "navix/four_rooms_9x9",
    "navix/four_rooms_11x11",
    "navix/four_rooms_13x13",
]

# --- Evaluation Function ---
# This is the function that will actually be run on the SLURM nodes.
# It should take parameters directly and return the objective value.
# It should NOT contain Optuna or Submitit logic.
def run_job_on_node(num_envs):
    # TODO check if we can pmap instead of looping!
    total_return = 0
    num_runs = 0

    for seed in SEEDS:
        for env in ENVS:
            with hydra.initialize(
                config_path="../stoix/configs/default/anakin",
                version_base="1.2"):
                # 3. Compose the configuration
                #    'config_name' is the name of your main config file (without .yaml)
                #    'overrides' is a list of strings, just like on the command line
                cfg = hydra.compose(
                    config_name="default_ff_ppo",
                    overrides=[
                        f"env={env}",
                        f"arch.seed={seed}",
                        f"arch.total_num_envs={num_envs}"
                    ]
                )

                # Optional: Print the composed config to verify
                # print("Composed Configuration in Caller:")
                # print(OmegaConf.to_yaml(cfg))

                # 4. Call your function directly with the composed config
                print("\n--- Calling train_model function ---")
                final_absolute_return = ppo.hydra_entry_point(cfg) # TODO figure out a way to pass num_envs in the config.  
                print("--- train_model function finished ---")

                # 5. Use the returned value
                print(f"\nReceived final_absolute_return in caller script: {final_absolute_return}")
                total_return += final_absolute_return
                num_runs += 1

    average_return = total_return / num_runs
    print(f"\nAverage final_absolute_return over all runs: {average_return}")
    return average_return


    

# --- Submitit / SLURM Configuration ---
SLURM_PARTITION = "NORMAL" # <-- partition on cremers cluster 
SLURM_LOG_FOLDER_BASE = "log_slurm_njobs" # Base directory for SLURM logs
SLURM_TIMEOUT_MIN = 120     # TODO Max time for one evaluation job (set to 2h)
SLURM_CPUS_PER_TASK = 4     # TODO set to 4
SLURM_MEM_GB = 5            # TODO set to 5
SLURM_GPUS_PER_NODE = 1     # TODO Set to >0 if your function needs GPUs

# --- Optimization Parameters ---
N_PARALLEL_JOBS = 5  # How many Optuna trials/processes/SLURM jobs to run concurrently TODO ideally 10 because thats queue size. 
TOTAL_TRIALS = 20    # Total number of trials to run for the study TODO 500?

# ==============================================================================
# Optuna Objective Function (Synchronous)
# ==============================================================================

# This function is called by each Optuna worker process.
# It is SYNCHRONOUS (no async/await).
def objective(trial):
    """
    Optuna objective function. Runs synchronously in an Optuna worker process.
    Suggests parameters, submits the evaluation job via submitit, waits
    blockingly for the result, and returns it.
    """
    # 1. Suggest hyperparameters
    # x = trial.suggest_float('x', -10, 10)
    # y = trial.suggest_float('y', -10, 10)
    num_envs = 2 ** trial.suggest_int('num_envs_exp', 5, 15)

    # Get identifiers for logging
    trial_number = trial.number
    process_id = os.getpid()

    # print(f"Optuna Process PID {process_id} (Trial {trial_number}): Starting objective. Params: x={x:.4f}, y={y:.4f}", flush=True)
    print(f"Optuna Process PID {process_id} (Trial {trial_number}): Starting objective. Params: num_envs={num_envs}", flush=True)


    # 2. Configure submitit for this specific trial
    # Create a unique log folder for this trial's SLURM job
    log_folder = os.path.join(SLURM_LOG_FOLDER_BASE, f"trial_{trial_number}_%j") # %j = SLURM Job ID
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        timeout_min=SLURM_TIMEOUT_MIN,
        slurm_partition=SLURM_PARTITION, # Make sure this is set correctly
        cpus_per_task=SLURM_CPUS_PER_TASK,
        gpus_per_node=SLURM_GPUS_PER_NODE,
        mem_gb=SLURM_MEM_GB,
        name=f"optuna_{STUDY_NAME}_t{trial_number}" # Unique SLURM job name
    )

    # 3. Submit the evaluation function to SLURM
    print(f"Optuna Process PID {process_id} (Trial {trial_number}): Submitting SLURM job.", flush=True)
    # Pass the parameters directly to the function defined above
    # job = executor.submit(my_function_on_node, x, y) # TODO this works
    job = executor.submit(run_job_on_node, num_envs) # TODO this doesn't work
    print(f"Optuna Process PID {process_id} (Trial {trial_number}): Submitted SLURM job {job.job_id}. Waiting blockingly...", flush=True)

    # 4. Wait BLOCKINGLY for the SLURM job to finish
    try:
        # job.result() blocks THIS Optuna process until the SLURM job is done.
        result = job.result() # This will contain the return value of my_function_on_node
        print(f"Optuna Process PID {process_id} (Trial {trial_number}): Job {job.job_id} completed successfully. Result: {result:.4f}", flush=True)
        # 5. Return the numerical result to Optuna
        return result

    except Exception as e:
        # Handle job failures (timeout, error in the node function, etc.)
        job_state = "UNKNOWN"
        stderr = "N/A"
        try:
            job_state = job.state
            stderr = job.stderr() # Try to get standard error output
        except Exception:
            pass # Ignore errors getting state/stderr if job object is problematic

        print(f"Optuna Process PID {process_id} (Trial {trial_number}): Job {job.job_id} FAILED. State: {job_state}.", flush=True)
        print(f"    Error: {e}", flush=True)
        if stderr:
             print(f"    Stderr (last 500 chars): ... {stderr[-500:]}", flush=True)

        # Tell Optuna that this trial failed and should be pruned
        # This prevents Optuna from considering it a valid result
        raise optuna.TrialPruned(f"Submitit job {job.job_id} failed or timed out.")


# ==============================================================================
# Main Script Execution
# ==============================================================================

if __name__ == "__main__":
    print(f"Starting Optuna HPO script.")
    print(f"  Study Name: {STUDY_NAME}")
    print(f"  Storage: {STORAGE_URL}")
    print(f"  Parallel Jobs (Optuna Processes): {N_PARALLEL_JOBS}")
    print(f"  Total Trials: {TOTAL_TRIALS}")
    print(f"  SLURM Partition: {SLURM_PARTITION}")

    # Ensure the base log directory exists
    os.makedirs(SLURM_LOG_FOLDER_BASE, exist_ok=True)
    print(f"  SLURM logs will be stored under ./{SLURM_LOG_FOLDER_BASE}/")

    # --- Instantiate the QMCSampler ---
    qmc_sampler = optuna.samplers.QMCSampler(seed=42) 
    # --- Create or load the study using the persistent storage ---
    # --- Pass the instantiated sampler using the 'sampler' argument ---
    study = optuna.create_study(
        storage=STORAGE_URL,
        study_name=STUDY_NAME,
        direction="maximize",   # or "minimize" depending on your objective
        load_if_exists=True,    # Allows resuming if the script is interrupted
        sampler=qmc_sampler     # <--- PASS THE SAMPLER HERE
    )

    # --- Run the optimization loop ---
    # Optuna handles creating/managing 'N_PARALLEL_JOBS' processes.
    # Each process connects to the storage and runs the 'objective' function.
    print("\nStarting study.optimize()...")
    try:
        study.optimize(
            objective,              # Pass the SYNCHRONOUS objective function
            n_trials=TOTAL_TRIALS,  # Total number of trials to run across all processes
            n_jobs=N_PARALLEL_JOBS  # Use multiprocessing managed by Optuna
            # timeout=3600          # Optional: Total time limit in seconds for the study
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user (KeyboardInterrupt).")
    except Exception as e:
        print(f"\nAn unexpected error occurred during study.optimize: {e}")
        # Consider adding more specific error handling if needed
    finally:
         print("\nOptimization loop finished or was interrupted.")

    # --- Print Final Study Summary ---
    print("\n" + "="*40)
    print("Final Study Summary")
    print("="*40)

    # Reload the study from storage to ensure we have the latest state
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_URL)

        print(f"Study Name: {study.study_name}")
        print(f"Direction: {study.direction}")
        print(f"Number of trials recorded: {len(study.trials)}")

        # Get trial state counts
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED] # Includes failures handled above
        fail_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]     # Should be 0 if handled by prune
        running_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]

        print(f"  Trials Completed Successfully: {len(complete_trials)}")
        print(f"  Trials Pruned (Failures/Timeouts): {len(pruned_trials)}")
        print(f"  Trials Explicitly Failed (Unexpected): {len(fail_trials)}")
        print(f"  Trials Still Marked Running (If any): {len(running_trials)}")

        # Print best trial information if any completed successfully
        if complete_trials:
            print("-" * 20)
            print(f"Best trial found:")
            print(f"  Number: {study.best_trial.number}")
            print(f"  Value (Objective Score): {study.best_value:.6f}")
            print(f"  Parameters: ")
            for key, value in study.best_params.items():
                print(f"    {key}: {value}")
        elif pruned_trials:
            print("\nNo trials completed successfully (all failed or were pruned).")
        else:
            print("\nNo trials seem to have run or completed.")

    except Exception as e:
        print(f"\nAn error occurred while loading or summarizing the study results: {e}")

    print("="*40)
    print("Script finished.")