import argparse
import concurrent.futures
from dotenv import load_dotenv
load_dotenv(override=True)

from tqdm import tqdm
import textgrad as tg
from textgrad.tasks import load_task
from textgrad.engine.claude_thinking_engine import ThinkingChatAnthropic
from textgrad.claude_token_efficiency_loss import ClaudeThinkingEfficiencyLoss

import numpy as np
import json
import os

def config():
    parser = argparse.ArgumentParser(description="Optimize a prompt for Claude's thinking token efficiency.")
    parser.add_argument("--task", type=str, default="GSM8K_DSPy", help="The task to evaluate the model on.")
    parser.add_argument("--model", type=str, default="claude-3-7-sonnet-20250219", help="Claude model to use.")
    parser.add_argument("--thinking_budget", type=int, default=16000, help="Budget for thinking tokens.")
    parser.add_argument("--batch_size", type=int, default=5, help="The batch size to use for training.")
    parser.add_argument("--max_epochs", type=int, default=40, help="The maximum number of epochs to train for.")
    parser.add_argument("--accuracy_weight", type=float, default=0.3, help="Weight for accuracy.")
    parser.add_argument("--token_weight", type=float, default=0.7, help="Weight for token efficiency.")
    parser.add_argument("--num_threads", type=int, default=20, help="Number of threads for evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    import random
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
    
    # Get thinking tokens used
    thinking_tokens = model.engine.get_last_thinking_tokens()
    
    return {
        "question": x,
        "correct_answer": y,
        "response": response.value,
        "accuracy": accuracy,
        "thinking_tokens": thinking_tokens
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
                avg_tokens = np.mean([r["thinking_tokens"] for r in results])
                pbar.set_description(f"Acc: {avg_accuracy:.3f}, Tokens: {avg_tokens:.1f}")
                pbar.update(1)
    
    # Calculate metrics
    metrics = {
        "accuracy": np.mean([r["accuracy"] for r in results]),
        "thinking_tokens": np.mean([r["thinking_tokens"] for r in results]),
        "accuracy_stdev": np.std([r["accuracy"] for r in results]),
        "thinking_tokens_stdev": np.std([r["thinking_tokens"] for r in results]),
    }
    
    return results, metrics

# Main execution
if __name__ == "__main__":
    args = config()
    set_seed(args.seed)
    
    # Create Claude engine with thinking support
    claude_engine = ThinkingChatAnthropic(
        model_string=args.model,
        thinking_enabled=True,
        thinking_budget=args.thinking_budget
    )
    
    # Set as backward engine for TextGrad
    tg.set_backward_engine(claude_engine, override=True)
    
    # Load dataset and evaluation function
    train_set, val_set, test_set, task_eval_fn = load_task(
        args.task, 
        evaluation_api=claude_engine
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
    model = tg.BlackboxLLM(claude_engine, system_prompt=system_prompt)
    token_loss = ClaudeThinkingEfficiencyLoss(
        evaluation_api=claude_engine,
        accuracy_weight=args.accuracy_weight,
        token_weight=args.token_weight
    )
    
    # Create optimizer
    optimizer = tg.TextualGradientDescent(
        engine=claude_engine,
        parameters=[system_prompt],
        constraints=["The prompt must encourage efficient thinking while maintaining accuracy."]
    )
    
    # Store results
    results = {
        "initial_prompt": STARTING_SYSTEM_PROMPT,
        "epochs": [],
        "final_prompt": "",
        "task": args.task,
        "model": args.model,
        "thinking_budget": args.thinking_budget,
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
                thinking_tokens = claude_engine.get_last_thinking_tokens()
                
                step_data = {
                    "question": x,
                    "answer": y,
                    "response": response.value,
                    "accuracy": accuracy,
                    "thinking_tokens": thinking_tokens
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
        
        print(f"\nEpoch {epoch} metrics: {val_metrics}")
    
    # Final evaluation
    final_results, final_metrics = eval_dataset(
        test_set,
        model,
        token_loss,
        task_eval_fn,
        max_samples=args.eval_samples * 2
    )
    
    results["final_prompt"] = system_prompt.value
    results["final_metrics"] = final_metrics
    
    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, f"claude_thinking_opt_{args.task}_{args.model.split('-')[-1]}.json")
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n=== Optimization Complete ===")
    print(f"Initial metrics: {initial_metrics}")
    print(f"Final metrics: {final_metrics}")
    print(f"\nInitial prompt: {STARTING_SYSTEM_PROMPT}")
    print(f"\nFinal prompt: {system_prompt.value}")
    print(f"\nResults saved to: {result_file}")