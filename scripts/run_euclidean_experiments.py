#!/usr/bin/env python3
"""
Script to run MDS embeddings on all datasets with different dimensions.
"""
import os
import subprocess
import time
from datetime import datetime

# Configuration
DATASETS = [
    "BindingDB_IC50",
    "BindingDB_Kd",
    "BindingDB_Ki",
    "DAVIS",
    "KIBA"
]
DIMENSIONS = [2, 4, 8, 16, 32, 64, 128]
BASE_DIR = "."
MDS_SCRIPT = "../src/experiments/euclidean_smacof.py"  # Path to the simplified MDS script
OUT_DIR = "results/euclidean"
LOG_DIR = "results/logs/"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def main():
    # Create a timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR + f"mds_run_{timestamp}.log"
    
    print(f"Starting MDS embedding generation for all datasets")
    print(f"Log will be saved to: {log_file}")
    
    with open(log_file, "w") as log:
        log.write(f"MDS Embedding Run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write("="*80 + "\n\n")
        
        total_runs = len(DATASETS) * len(DIMENSIONS)
        current_run = 0
        
        for dataset in DATASETS:
            dataset_path = os.path.join(BASE_DIR, "datasets", dataset)
            input_file = os.path.join(dataset_path, "sample.csv")
            
            if not os.path.exists(input_file):
                msg = f"ERROR: Input file not found: {input_file}"
                print(msg)
                log.write(msg + "\n")
                continue
                
            for dim in DIMENSIONS:
                current_run += 1
                output_dir = os.path.join(BASE_DIR, OUT_DIR, f"{dataset}_dim{dim}")
                
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Build command
                cmd = [
                    "python", MDS_SCRIPT,
                    "-i", input_file,
                    "-o", output_dir,
                    "--dim", str(dim),
                    "-m", str(300)
                ]
                
                # Log and execute
                cmd_str = " ".join(cmd)
                progress = f"[{current_run}/{total_runs}]"
                
                header = f"\n{progress} Running: {dataset} with dimension {dim}"
                print(header)
                log.write(header + "\n")
                log.write(f"Command: {cmd_str}\n")
                log.flush()
                
                # Time the execution
                start_time = time.time()
                
                try:
                    # Run the command and capture output
                    process = subprocess.Popen(
                        cmd, 
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True
                    )
                    
                    # Stream and log output in real-time
                    for line in process.stdout:
                        print(f"  {line.strip()}")
                        log.write(line)
                        log.flush()
                    
                    process.wait()
                    
                    # Check return code
                    if process.returncode != 0:
                        log.write(f"ERROR: Process returned non-zero exit code: {process.returncode}\n")
                except Exception as e:
                    error_msg = f"ERROR: Failed to run command: {str(e)}"
                    print(error_msg)
                    log.write(error_msg + "\n")
                
                # Calculate and log execution time
                elapsed_time = time.time() - start_time
                time_msg = f"Completed in {elapsed_time:.2f} seconds"
                print(time_msg)
                log.write(time_msg + "\n")
                log.write("-"*80 + "\n")
                log.flush()
    
    print(f"All runs completed. Check {log_file} for details.")

if __name__ == "__main__":
    main()