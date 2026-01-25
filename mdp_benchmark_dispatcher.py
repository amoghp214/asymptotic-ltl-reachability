#!/usr/bin/env python3
"""
MDP Benchmark Dispatcher

This script dispatches SLURM benchmark testing jobs for MDP models.
It maintains a list of benchmark configurations and submits SLURM jobs
for each configuration.
"""

import subprocess
import argparse
from datetime import datetime


# List of benchmark parameters: (job_name, mdp_name, version, confidence_error, minimum_transition_probability, min_num_iterations, max_num_iterations, convergence_threshold, num_policy_accuracy_sims)
BENCHMARK_CONFIGS = [
    ("consensus_2", "consensus.2", "1", 0.01, 0.5, 5, 100, 0.0005, 1000),
    ("csma_2-2", "csma.2-2", "1", 0.01, 0.25, 5, 50, 0.001, 200),
    ("firewire_abst", "firewire_abst", "1", 0.01, 0.5, 5, 100, 0.0005, 1000),
    ("ij_10", "ij.10", "1", 0.01, 0.5, 5, 50, 0.0005, 1000),
    ("ij_3", "ij.3", "1", 0.01, 0.5, 5, 100, 0.0001, 1000),
    ("pacman", "pacman", "2", 0.01, 0.08, 5, 100, 0.001, 200),
    ("philosophers_mdp_3", "philosophers-mdp.3", "1", 0.01, 0.5, 5, 100, 0.0005, 1000),
    ("rabin_3", "rabin.3", "1", 0.01, 0.03125, 5, 100, 0.0005, 1000),
    ("zeroconf", "zeroconf", "1", 0.01, 0.0001025262467191601, 5, 100, 0.0005, 100),
]


def create_slurm_script(
    job_name, 
    mdp_name, 
    version=1, 
    confidence_error=0.01, 
    min_transition_prob=0.001,
    min_num_iterations=5,
    max_num_iterations=100,
    convergence_threshold=0.0005,
    num_policy_accuracy_sims=1000,
    trial=0,
    account="N/A"
):
    """
    Create a SLURM batch script for a single benchmark job.
    
    Args:
        job_name: Name of the job
        mdp_name: Name of the MDP model
        version: Version of the MDP model
        confidence_error: Confidence error parameter
        min_transition_prob: Minimum transition probability parameter
        account: SLURM account name
    
    Returns:
        Path to the created SLURM script
    """
    script_name = f"slurm_{job_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sh"
    
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm/logs/slurm_{job_name}_%j.log
#SBATCH --error=slurm/logs/slurm_{job_name}_%j.err
#SBATCH --time=36:00:00
#SBATCH --account={account}
#SBATCH --partition=phoenix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Environment setup
module load anaconda3
conda activate ltl-reachability

# Run the MDP learning benchmark
python main.py -m {mdp_name} -v {version} -c {confidence_error} -p {min_transition_prob} -n {min_num_iterations} -x {max_num_iterations} -t {convergence_threshold} -a {num_policy_accuracy_sims} -i {trial}
"""
    
    script_path = "slurm/jobs/" + script_name
    with open(script_path, 'w') as f:
        f.write(slurm_script)
    
    return script_path


def submit_job(script_path):
    """
    Submit a SLURM job script.
    
    Args:
        script_path: Path to the SLURM script
    
    Returns:
        Job ID if successful, None otherwise
    """
    try:
        result = subprocess.run(
            ['sbatch', script_path],
            capture_output=True,
            text=True,
            check=True
        )
        # Extract job ID from output (format: "Submitted batch job XXXXX")
        job_id = result.stdout.strip().split()[-1]
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Error: sbatch command not found. Are you on a SLURM cluster?")
        return None


def dispatch_benchmarks(configs=None, dry_run=False, num_trials=1, account=None):
    """
    Dispatch all benchmark jobs.
    
    Args:
        configs: List of benchmark configurations. If None, uses BENCHMARK_CONFIGS.
        dry_run: If True, only print what would be done without submitting.
        num_trials: Number of trials to run for each configuration.
        account: SLURM account name.
    """
    if configs is None:
        configs = BENCHMARK_CONFIGS
    
    submitted_jobs = []
    
    for job_name, mdp_name, version, confidence_error, min_transition_prob, min_num_iterations, max_num_iterations, convergence_threshold, num_policy_accuracy_sims in configs:
        print(f"\nProcessing job: {job_name}")
        print(f"  MDP: {mdp_name}, Version: {version}")
        print(f"  Confidence Error: {confidence_error}, Min Transition Prob: {min_transition_prob}")

        for trial in range(1, num_trials + 1):
            trial_job_name = f"{job_name}_trial{trial}"
            print(f"    Trial {trial}:")
        
            if dry_run:
                print(f"  [DRY RUN] Would run: python main.py -m {mdp_name} -v {version} -c {confidence_error} -p {min_transition_prob} -n {min_num_iterations} -x {max_num_iterations} -t {convergence_threshold} -a {num_policy_accuracy_sims} -i {trial}")
            else:
                script_path = create_slurm_script(
                    trial_job_name, 
                    mdp_name, 
                    version, 
                    confidence_error, 
                    min_transition_prob, 
                    min_num_iterations, 
                    max_num_iterations, 
                    convergence_threshold, 
                    num_policy_accuracy_sims, 
                    trial,
                    account
                )
                print(f"  Created SLURM script: {script_path}")
                
                job_id = submit_job(script_path)
                if job_id:
                    print(f"  Successfully submitted job with ID: {job_id}")
                    submitted_jobs.append((job_name, job_id))
                else:
                    print(f"  Failed to submit job")
    
    return submitted_jobs


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Dispatch MDP benchmark testing jobs to SLURM"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without submitting them"
    )
    parser.add_argument(
        "--job",
        type=str,
        help="Submit only a specific job by name"
    )
    parser.add_argument(
        "-t",
        "--num_trials",
        type=int,
        default=1,
        help="Number of trials to run for each benchmark job (default: 1)"
    )
    parser.add_argument(
        "-a",
        "--account",
        type=str,
        required=True,
        help="SLURM account name (required)"
    )
    
    args = parser.parse_args()
    
    configs = BENCHMARK_CONFIGS
    
    if args.job:
        configs = [c for c in BENCHMARK_CONFIGS if c[0] == args.job]
        if not configs:
            print(f"Error: Job '{args.job}' not found in BENCHMARK_CONFIGS")
            return
    
    print("=" * 70)
    print("MDP Benchmark Dispatcher")
    print("=" * 70)
    print(f"Total jobs to submit: {len(configs)}")
    print(f"Account: {args.account}")
    if args.dry_run:
        print("[DRY RUN MODE] - No jobs will be submitted")
    print("=" * 70)
    
    submitted_jobs = dispatch_benchmarks(configs, dry_run=args.dry_run, num_trials=args.num_trials, account=args.account)
    
    print("\n" + "=" * 70)
    if not args.dry_run and submitted_jobs:
        print(f"Successfully submitted {len(submitted_jobs)} jobs:")
        for job_name, job_id in submitted_jobs:
            print(f"  {job_name}: {job_id}")
    print("=" * 70)


if __name__ == "__main__":
    main()
