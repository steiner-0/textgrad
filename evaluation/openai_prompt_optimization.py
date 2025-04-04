import argparse
import concurrent.futures
from dotenv import load_dotenv
load_dotenv(override=True)

from tqdm import tqdm
import textgrad as tg
from textgrad.tasks import load_task
from textgrad.engine.openai_thinking_engine import ThinkingO1Engine
from textgrad.openai_token_efficiency_loss import O1ReasoningEfficiencyLoss

import numpy as np
import json
import os
import random

def config():
    parser = argparse.ArgumentParser(description="Optimize a prompt for OpenAI o1's reasoning token efficiency.")
    parser.add_argument("--task", type=str, default="GSM8K_DSPy", help="The task to evaluate the model on.")
    parser.add_argument("--model", type=str, default="o1-2024-12-17", help="OpenAI o1 model to use.")
    parser.add_argument("--reasoning_effort", type=str, default="medium", choices=["low", "medium", "high"], 
                        help="Reasoning effort level for the o1 model.")
    parser.add_argument("--batch_size", type=int, default=20, help="The batch size to use for training.")
    parser.add_argument("--max_epochs", type=int, default=5, help="The maximum number of epochs to train for.")
    parser.add_argument("--accuracy_weight", type=float, default=0.3, help="Weight for accuracy.")
    parser.add_argument("--token_weight", type=float, default=0.7, help="Weight for token efficiency.")
    parser.add_argument("--num_threads", type=int, default=20, help="Number of threads for evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key (otherwise uses OPENAI_API_KEY env var).")
    parser.add_argument("--eval_samples", type=int, default=5, help="Number of samples to use for evaluation.")
    return parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def eval_sample(sample, model, eval_fn, task_eval_fn=None):
    """Evaluate a single sample for accuracy and token usage."""
    x, y = sample
    
    # Create variables
    x_var = tg.Variable(x, requires_grad=False, role_description="query to the model")
    y_var = tg.Variable(y, requires_grad=False, role_description="correct answer")
    
    # Get model's response
    response = model(x_var)
    
    # Check accuracy
    if task_eval_fn:
        try:
            eval_output = task_eval_fn(inputs=dict(prediction=response, ground_truth_answer=y_var))
            accuracy = int(eval_output.value)
        except:
            eval_output = task_eval_fn([x_var, y_var, response])
            accuracy = int(task_eval_fn.parse_output(eval_output))
    else:
        # Default string match
        accuracy = 1 if y in response.value else 0
    
    # Get token usage metrics
    reasoning_tokens = model.engine.get_last_reasoning_tokens()
    completion_tokens = model.engine.last_completion_tokens
    total_tokens = model.engine.last_total_tokens
    
    return {
        "question": x,
        "correct_answer": y,
        "response": response.value,
        "accuracy": accuracy,
        "reasoning_tokens": reasoning_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }

def eval_dataset(dataset, model, eval_fn, task_eval_fn=None, max_samples=None):
    """Evaluate a dataset for accuracy and token efficiency."""
    if max_samples is None or max_samples > len(dataset):
        max_samples = len(dataset)
    
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for i in range(max_samples):
            sample = dataset[i]
            future = executor.submit(eval_sample, sample, model, eval_fn, task_eval_fn)
            futures.append(future)
        
        with tqdm(total=len(futures), desc="Evaluating") as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                
                # Update progress display
                avg_accuracy = np.mean([r["accuracy"] for r in results])
                avg_tokens = np.mean([r["reasoning_tokens"] for r in results])
                pbar.set_description(f"Acc: {avg_accuracy:.3f}, Tokens: {avg_tokens:.1f}")
                pbar.update(1)
    
    # Calculate metrics
    metrics = {
        "accuracy": np.mean([r["accuracy"] for r in results]),
        "reasoning_tokens": np.mean([r["reasoning_tokens"] for r in results]),
        "completion_tokens": np.mean([r["completion_tokens"] for r in results]),
        "total_tokens": np.mean([r["total_tokens"] for r in results]),
        "accuracy_stdev": np.std([r["accuracy"] for r in results]),
        "reasoning_tokens_stdev": np.std([r["reasoning_tokens"] for r in results]),
    }
    
    return results, metrics

def try_different_reasoning_levels(model, sample, task_eval_fn):
    """Test the same sample with different reasoning effort levels."""
    x, y = sample
    x_var = tg.Variable(x, requires_grad=False, role_description="query to the model")
    y_var = tg.Variable(y, requires_grad=False, role_description="correct answer")
    
    results = {}
    
    for effort in ["low", "medium", "high"]:
        model.engine.set_reasoning_effort(effort)
        response = model(x_var)
        
        # Check accuracy
        if task_eval_fn:
            try:
                eval_output = task_eval_fn(inputs=dict(prediction=response, ground_truth_answer=y_var))
                accuracy = int(eval_output.value)
            except:
                eval_output = task_eval_fn([x_var, y_var, response])
                accuracy = int(task_eval_fn.parse_output(eval_output))
        else:
            accuracy = 1 if y in response.value else 0
        
        results[effort] = {
            "response": response.value,
            "accuracy": accuracy,
            "reasoning_tokens": model.engine.get_last_reasoning_tokens(),
            "completion_tokens": model.engine.last_completion_tokens,
            "total_tokens": model.engine.last_total_tokens
        }
    
    # Reset to original reasoning effort
    model.engine.set_reasoning_effort(args.reasoning_effort)
    return results

# Main execution
if __name__ == "__main__":
    args = config()
    set_seed(args.seed)
    
    # Create OpenAI o1 engine with reasoning support
    o1_engine = ThinkingO1Engine(
        model_string=args.model,
        reasoning_effort=args.reasoning_effort,
        api_key=args.api_key
    )
    
    # Set as backward engine for TextGrad
    tg.set_backward_engine(o1_engine, override=True)
    
    # Load dataset and evaluation function
    train_set, val_set, test_set, task_eval_fn = load_task(
        args.task, 
        evaluation_api=o1_engine
    )
    
    print(f"Dataset loaded: {args.task}")
    print(f"Train/Val/Test sizes: {len(train_set)}/{len(val_set)}/{len(test_set)}")
    
    # Get initial system prompt
    STARTING_SYSTEM_PROMPT = train_set.get_task_description()
    print(f"Initial system prompt: {STARTING_SYSTEM_PROMPT}")
    
    # Create the system prompt variable
    system_prompt = tg.Variable(
        STARTING_SYSTEM_PROMPT,
        requires_grad=True,
        role_description="system prompt designed to encourage efficient reasoning"
    )
    
    # Create model and loss function
    model = tg.BlackboxLLM(o1_engine, system_prompt=system_prompt)
    token_loss = O1ReasoningEfficiencyLoss(
        evaluation_api=o1_engine,
        accuracy_weight=args.accuracy_weight,
        token_weight=args.token_weight
    )
    
    # Create optimizer
    optimizer = tg.TextualGradientDescent(
        engine=o1_engine,
        parameters=[system_prompt],
        constraints=[
            "The prompt must encourage efficient reasoning while maintaining accuracy.",
            "The prompt should guide the model to use as few tokens as possible for reasoning.",
            "The prompt must be clear and specific, directing the model to solve problems directly.",
            f"The prompt should work well with reasoning_effort={args.reasoning_effort}."
        ]
    )
    
    # Store results
    results = {
        "initial_prompt": STARTING_SYSTEM_PROMPT,
        "epochs": [],
        "final_prompt": "",
        "task": args.task,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "accuracy_weight": args.accuracy_weight,
        "token_weight": args.token_weight
    }
    
    # Evaluate initial performance
    print("\nEvaluating initial performance...")
    initial_results, initial_metrics = eval_dataset(
        test_set, 
        model, 
        token_loss, 
        task_eval_fn, 
        max_samples=args.eval_samples
    )
    
    print(f"Initial metrics: {initial_metrics}")
    results["initial_metrics"] = initial_metrics
    
    # Test different reasoning levels on a sample problem
    print("\nTesting different reasoning effort levels on a sample problem...")
    sample_problem = test_set[0]
    reasoning_level_results = try_different_reasoning_levels(model, sample_problem, task_eval_fn)
    results["reasoning_level_comparison"] = reasoning_level_results
    
    # Print reasoning level comparison
    for level, level_results in reasoning_level_results.items():
        print(f"Level {level}: Accuracy={level_results['accuracy']}, Tokens={level_results['reasoning_tokens']}")
    
    # Training loop
    train_loader = tg.tasks.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    
    for epoch in range(args.max_epochs):
        epoch_data = {
            "epoch": epoch,
            "prompt": system_prompt.value,
            "steps": []
        }
        
        for step, (batch_x, batch_y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            optimizer.zero_grad()
            step_losses = []
            
            for (x, y) in zip(batch_x, batch_y):
                # Create variables
                x_var = tg.Variable(x, requires_grad=False, role_description="query to the model")
                y_var = tg.Variable(y, requires_grad=False, role_description="correct answer")
                
                # Get model response
                response = model(x_var)
                
                # Compute token efficiency loss
                loss = token_loss(system_prompt, x_var, response, y_var)
                step_losses.append(loss)
                
                # Calculate metrics for this example
                accuracy = 1 if y in response.value else 0
                reasoning_tokens = o1_engine.get_last_reasoning_tokens()
                completion_tokens = o1_engine.last_completion_tokens
                
                step_data = {
                    "question": x,
                    "answer": y,
                    "response": response.value,
                    "accuracy": accuracy,
                    "reasoning_tokens": reasoning_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": reasoning_tokens + completion_tokens
                }
                epoch_data["steps"].append(step_data)
            
            # Backward pass through all losses
            for loss in step_losses:
                loss.backward()
            
            # Update the prompt
            optimizer.step()
            
            print(f"\nPrompt after step {step}:")
            print(system_prompt.value)
            
            # Break after a few steps to keep the process manageable
            if step >= 2:
                break
        
        # Evaluate on validation set
        val_results, val_metrics = eval_dataset(
            val_set,
            model,
            token_loss,
            task_eval_fn,
            max_samples=args.eval_samples
        )
        
        epoch_data["validation_metrics"] = val_metrics
        results["epochs"].append(epoch_data)
        
        print(f"\nEpoch {epoch} validation metrics: {val_metrics}")
    
    # Test final prompt with different reasoning levels
    print("\nTesting optimized prompt with different reasoning levels...")
    final_reasoning_results = {}
    for effort in ["low", "medium", "high"]:
        o1_engine.set_reasoning_effort(effort)
        effort_results, effort_metrics = eval_dataset(
            test_set,
            model,
            token_loss,
            task_eval_fn,
            max_samples=args.eval_samples
        )
        final_reasoning_results[effort] = effort_metrics
        print(f"Optimized prompt with {effort} reasoning: {effort_metrics}")
    
    # Final evaluation on test set with original reasoning effort
    o1_engine.set_reasoning_effort(args.reasoning_effort)
    final_results, final_metrics = eval_dataset(
        test_set,
        model,
        token_loss,
        task_eval_fn,
        max_samples=args.eval_samples * 2
    )
    
    results["final_prompt"] = system_prompt.value
    results["final_metrics"] = final_metrics
    results["final_reasoning_level_comparison"] = final_reasoning_results
    
    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, f"o1_reasoning_opt_{args.task}_{args.model.replace('-', '_')}_{args.reasoning_effort}.json")
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n=== Optimization Complete ===")
    print(f"Initial metrics: {initial_metrics}")
    print(f"Final metrics: {final_metrics}")
    print(f"\nInitial prompt: {STARTING_SYSTEM_PROMPT}")
    print(f"\nFinal prompt: {system_prompt.value}")
    print(f"\nResults saved to: {result_file}")