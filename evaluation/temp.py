import argparse
import concurrent.futures
from dotenv import load_dotenv
load_dotenv(override=True)

from tqdm import tqdm
import textgrad as tg
from textgrad.tasks import load_task
from textgrad.o1_reasoning_efficiency_loss import O1ReasoningEfficiencyLoss

import numpy as np
import json
import os
import random

def config():
    parser = argparse.ArgumentParser(description="Optimize a prompt for OpenAI's reasoning efficiency.")
    parser.add_argument("--task", type=str, default="BBH_object_counting", help="The task to evaluate the model on.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use.")
    parser.add_argument("--evaluation_engine", type=str, default="gpt-4o", help="OpenAI model to use for evaluation.")
    parser.add_argument("--test_engine", type=str, default="gpt-3.5-turbo", help="Model to test optimized prompts on.")
    parser.add_argument("--batch_size", type=int, default=3, help="The batch size to use for training.")
    parser.add_argument("--max_epochs", type=int, default=3, help="The maximum number of epochs to train for.")
    parser.add_argument("--reasoning_effort", type=str, default="medium", 
                         choices=["low", "medium", "high"], 
                         help="Target level of reasoning effort.")
    parser.add_argument("--accuracy_weight", type=float, default=0.3, help="Weight for accuracy.")
    parser.add_argument("--efficiency_weight", type=float, default=0.7, help="Weight for reasoning efficiency.")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads for evaluation.")
    parser.add_argument("--run_validation", action="store_true", help="Whether to run validation or not.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def eval_sample(sample, model, task_eval_fn=None):
    """Evaluate a single sample for accuracy and efficiency."""
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
    
    # Estimate response qualities
    response_length = len(response.value.split())
    
    return {
        "question": x,
        "correct_answer": y,
        "response": response.value,
        "accuracy": accuracy,
        "response_length": response_length
    }

def eval_dataset(dataset, model, task_eval_fn=None, max_samples=None):
    """Evaluate a dataset for accuracy and efficiency."""
    if max_samples is None or max_samples > len(dataset):
        max_samples = len(dataset)
    
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for i in range(max_samples):
            sample = dataset[i]
            future = executor.submit(eval_sample, sample, model, task_eval_fn)
            futures.append(future)
        
        with tqdm(total=len(futures), desc="Evaluating") as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                
                # Update progress display
                avg_accuracy = np.mean([r["accuracy"] for r in results])
                avg_length = np.mean([r["response_length"] for r in results])
                pbar.set_description(f"Acc: {avg_accuracy:.3f}, Length: {avg_length:.1f}")
                pbar.update(1)
    
    # Calculate metrics
    metrics = {
        "accuracy": np.mean([r["accuracy"] for r in results]),
        "response_length": np.mean([r["response_length"] for r in results]),
        "accuracy_stdev": np.std([r["accuracy"] for r in results]),
        "response_length_stdev": np.std([r["response_length"] for r in results]),
    }
    
    return results, metrics

def run_validation_revert(system_prompt, results, model, eval_fn, val_set):
    """Run validation and revert if performance degrades."""
    val_performance = np.mean(eval_dataset(val_set, model, eval_fn, max_samples=5)[0])
    previous_performance = np.mean(results["validation_acc"][-1])
    print(f"Validation accuracy: {val_performance}")
    print(f"Previous accuracy: {previous_performance}")
    previous_prompt = results["prompt"][-1]
    
    if val_performance < previous_performance:
        print(f"Rejected prompt: {system_prompt.value}")
        system_prompt.set_value(previous_prompt)
        val_performance = previous_performance

    results["validation_acc"].append(val_performance)
    return val_performance

# Main execution
if __name__ == "__main__":
    args = config()
    set_seed(args.seed)
    
    # Create engines
    eval_engine = tg.get_engine(engine_name=args.evaluation_engine)
    test_engine = tg.get_engine(engine_name=args.test_engine)
    
    # Set as backward engine for TextGrad
    tg.set_backward_engine(eval_engine, override=True)
    
    # Load dataset and evaluation function
    train_set, val_set, test_set, task_eval_fn = load_task(
        args.task, 
        evaluation_api=eval_engine
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
        role_description=f"system prompt designed for {args.reasoning_effort} reasoning effort"
    )
    
    # Create models for optimization and testing
    optimization_model = tg.BlackboxLLM(eval_engine, system_prompt=system_prompt)
    test_model = tg.BlackboxLLM(test_engine, system_prompt=system_prompt)
    
    # Create efficiency loss
    efficiency_loss = O1ReasoningEfficiencyLoss(
        evaluation_api=eval_engine,
        accuracy_weight=args.accuracy_weight,
        efficiency_weight=args.efficiency_weight,
        reasoning_effort=args.reasoning_effort
    )
    
    # Create optimizer
    optimizer = tg.TextualGradientDescent(
        engine=eval_engine,
        parameters=[system_prompt],
        constraints=[f"The prompt must encourage {args.reasoning_effort} reasoning effort while maintaining accuracy."]
    )
    
    # Store results
    results = {
        "initial_prompt": STARTING_SYSTEM_PROMPT,
        "epochs": [],
        "prompt": [STARTING_SYSTEM_PROMPT],
        "validation_acc": [0.0],
        "test_acc": [0.0],
        "final_prompt": "",
        "task": args.task,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "accuracy_weight": args.accuracy_weight,
        "efficiency_weight": args.efficiency_weight
    }
    
    # Evaluate initial performance
    print("\nEvaluating initial performance...")
    initial_results, initial_metrics = eval_dataset(
        test_set, 
        test_model, 
        task_eval_fn, 
        max_samples=5
    )
    
    print(f"Initial metrics: {initial_metrics}")
    results["test_acc"][0] = initial_metrics["accuracy"]
    
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
                response = optimization_model(x_var)
                
                # Compute efficiency loss
                loss = efficiency_loss(system_prompt, x_var, response, y_var)
                step_losses.append(loss)
                
                # Calculate metrics for this example
                accuracy = 1 if y in response.value else 0
                response_length = len(response.value.split())
                
                step_data = {
                    "question": x,
                    "answer": y,
                    "response": response.value,
                    "accuracy": accuracy,
                    "response_length": response_length
                }
                epoch_data["steps"].append(step_data)
            
            # Backward pass through all losses
            for loss in step_losses:
                loss.backward()
            
            # Update the prompt
            optimizer.step()
            
            print(f"\nPrompt after step {step}:")
            print(system_prompt.value)
            
            # Run validation if requested
            if args.run_validation:
                val_acc = run_validation_revert(system_prompt, results, test_model, task_eval_fn, val_set)
                print(f"Validation accuracy: {val_acc:.3f}")
            
            # Test on the test set
            test_results, test_metrics = eval_dataset(
                test_set,
                test_model,
                task_eval_fn,
                max_samples=5
            )
            results["test_acc"].append(test_metrics["accuracy"])
            results["prompt"].append(system_prompt.value)
            
            print(f"Test accuracy: {test_metrics['accuracy']:.3f}")
            
            # Break after a few steps to keep the process manageable
            if step >= 2:
                break
        
        # Save epoch data
        results["epochs"].append(epoch_data)
    
    # Final evaluation
    final_results, final_metrics = eval_dataset(
        test_set,
        test_model,
        task_eval_fn,
        max_samples=10
    )
    
    results["final_prompt"] = system_prompt.value
    results["final_metrics"] = final_metrics
    
    # Save results
    results_dir = "./results/prompt_optimization"
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, f"openai_reasoning_opt_{args.task}_{args.reasoning_effort}.json")
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n=== Optimization Complete ===")
    print(f"Initial metrics: {initial_metrics}")
    print(f"Final metrics: {final_metrics}")
    print(f"\nInitial prompt: {STARTING_SYSTEM_PROMPT}")
    print(f"\nFinal prompt: {system_prompt.value}")
    print(f"\nResults saved to: {result_file}")