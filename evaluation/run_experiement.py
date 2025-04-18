#!/usr/bin/env python
"""
Script to run multiple prompt optimization experiments with different weight configurations.
"""

import subprocess
import argparse
import logging
from datetime import datetime
import os

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run multiple prompt optimization experiments")
    parser.add_argument("--task", type=str, default="GSM8K_DSPy", 
                        help="The task to evaluate the model on")
    parser.add_argument("--model", type=str, default="claude-3-7-sonnet-20250219", 
                        help="Claude model to use")
    parser.add_argument("--custom_prompt", type=str, default=None, 
                        help="Custom starting prompt (overrides task's default prompt)")
    parser.add_argument("--prompt_file", type=str, default=None, 
                        help="File containing custom starting prompt")
    parser.add_argument("--batch_size", type=int, default=3, 
                        help="The batch size to use for training")
    parser.add_argument("--max_epochs", type=int, default=10, 
                        help="The maximum number of epochs to train for")
    parser.add_argument("--eval_samples", type=int, default=5, 
                        help="Number of samples to use for evaluation")
    parser.add_argument("--num_threads", type=int, default=4, 
                        help="Number of threads for evaluation")
    parser.add_argument("--thinking_budget", type=int, default=16000, 
                        help="Budget for thinking tokens")
    parser.add_argument("--thinking_enabled", action="store_true", default=True, 
                        help="Enable Claude's thinking feature")
    parser.add_argument("--run_validation", action="store_true", default=True,
                        help="Run validation after each step and revert if performance decreases")
    parser.add_argument("--configs", type=str, nargs="+", 
                        default=["0.3,0.7", "0.5,0.5", "0.7,0.3"],
                        help="List of accuracy,efficiency weight pairs to test (comma-separated)")
    return parser.parse_args()

def run_experiment(args, accuracy_weight, efficiency_weight):
    """Run a single optimization experiment with the specified weights."""
    cmd = [
        "python", "balanced_prompt_optimization.py",
        "--task", args.task,
        "--model", args.model,
        "--batch_size", str(args.batch_size),
        "--max_epochs", str(args.max_epochs),
        "--accuracy_weight", str(accuracy_weight),
        "--efficiency_weight", str(efficiency_weight),
        "--eval_samples", str(args.eval_samples),
        "--num_threads", str(args.num_threads),
        "--thinking_budget", str(args.thinking_budget),
    ]
    
    # Add optional arguments if provided
    if args.custom_prompt:
        cmd.extend(["--custom_prompt", args.custom_prompt])
    if args.prompt_file:
        cmd.extend(["--prompt_file", args.prompt_file])
    if args.thinking_enabled:
        cmd.append("--thinking_enabled")
    if args.run_validation:
        cmd.append("--run_validation")
    
    logger.info(f"Running experiment with weights: accuracy={accuracy_weight}, efficiency={efficiency_weight}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    logger.info(f"Running experiment with weights: accuracy={accuracy_weight}, efficiency={efficiency_weight}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Stream output to log
        for line in process.stdout:
            logger.info(line.strip())
        
        process.wait()
        if process.returncode == 0:
            logger.info(f"Experiment completed successfully")
        else:
            logger.error(f"Experiment failed with return code {process.returncode}")
        
        return process.returncode
    except Exception as e:
        logger.error(f"Error running experiment: {e}")
        return -1

def main():
    args = parse_args()
    
    logger.info("Starting experiments with the following configurations:")
    for config in args.configs:
        accuracy_weight, efficiency_weight = map(float, config.split(","))
        logger.info(f"  - Accuracy: {accuracy_weight}, Efficiency: {efficiency_weight}")
    
    results = []
    
    for config in args.configs:
        accuracy_weight, efficiency_weight = map(float, config.split(","))
        
        logger.info("=" * 80)
        logger.info(f"STARTING EXPERIMENT: Accuracy={accuracy_weight}, Efficiency={efficiency_weight}")
        logger.info("=" * 80)
        
        return_code = run_experiment(args, accuracy_weight, efficiency_weight)
        
        results.append({
            "accuracy_weight": accuracy_weight,
            "efficiency_weight": efficiency_weight,
            "success": return_code == 0
        })
    
    # Log summary of all experiments
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 80)
    
    for result in results:
        status = "SUCCESS" if result["success"] else "FAILED"
        logger.info(f"Accuracy={result['accuracy_weight']}, Efficiency={result['efficiency_weight']}: {status}")
    
    logger.info("\nAll experiments completed. Check the results directory for detailed results.")

if __name__ == "__main__":
    main()