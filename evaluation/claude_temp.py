import argparse
import concurrent.futures
from dotenv import load_dotenv
load_dotenv(override=True)

from tqdm import tqdm
import textgrad as tg
from textgrad.tasks.gsm8k import GSM8K
import numpy as np
import random
import json
import os

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def config():
    parser = argparse.ArgumentParser(description="Optimize a prompt for GSM8K with Claude")
    parser.add_argument("--model", type=str, default="claude-3-opus-20240229", 
                       help="Claude model to use")
    parser.add_argument("--batch_size", type=int, default=3, 
                       help="The batch size to use for training")
    parser.add_argument("--max_epochs", type=int, default=5, 
                       help="The maximum number of epochs to train for")
    parser.add_argument("--num_threads", type=int, default=4, 
                       help="Number of threads for evaluation")
    parser.add_argument("--train_size", type=int, default=-1, 
                       help="Number of training examples to use. -1 for all.")
    parser.add_argument("--val_size", type=int, default=100, 
                       help="Number of validation examples to use")
    parser.add_argument("--test_size", type=int, default=200, 
                       help="Number of test examples to use")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    parser.add_argument("--results_dir", type=str, default="./results", 
                       help="Directory to save results")
    return parser.parse_args()

def eval_sample(sample, model, eval_fn):
    """Evaluate a single sample for accuracy."""
    x, y = sample
    
    # Create variables
    x_var = tg.Variable(x, requires_grad=False, role_description="query to the model")
    y_var = tg.Variable(y, requires_grad=False, role_description="correct answer")
    
    # Get model's response
    response = model(x_var)
    
    # Check accuracy using the string-based equality function
    from textgrad.tasks.big_bench_hard import string_based_equality_fn
    accuracy = string_based_equality_fn(response, y_var)
    
    return {
        "question": x,
        "correct_answer": y,
        "response": response.value,
        "accuracy": accuracy
    }

def eval_dataset(dataset, model, eval_fn, max_samples=None):
    """Evaluate a dataset for accuracy."""
    if max_samples is None or max_samples > len(dataset):
        max_samples = len(dataset)
    
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for i in range(max_samples):
            sample = dataset[i]
            future = executor.submit(eval_sample, sample, model, eval_fn)
            futures.append(future)
        
        with tqdm(total=len(futures), desc="Evaluating") as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                
                # Update progress display
                avg_accuracy = np.mean([r["accuracy"] for r in results])
                pbar.set_description(f"Accuracy: {avg_accuracy:.3f}")
                pbar.update(1)
    
    # Calculate metrics
    metrics = {
        "accuracy": np.mean([r["accuracy"] for r in results]),
        "accuracy_stdev": np.std([r["accuracy"] for r in results]),
    }
    
    return results, metrics

# Main execution
if __name__ == "__main__":
    args = config()
    set_seed(args.seed)
    
    # Create Claude engine
    claude_engine = tg.get_engine(engine_name=args.model)
    
    # Set as backward engine for TextGrad
    tg.set_backward_engine(claude_engine, override=True)
    
    # Load entire GSM8K dataset (or customize sizes)
    train_set = GSM8K(subset="main", split="train")
    if args.train_size > 0:
        # Use a subset if specified
        train_indices = list(range(len(train_set)))
        random.shuffle(train_indices)
        train_indices = train_indices[:args.train_size]
        train_subset = [train_set[i] for i in train_indices]
    else:
        # Use the entire training set
        train_subset = [train_set[i] for i in range(len(train_set))]
    
    # Load validation and test sets
    val_set = GSM8K(subset="main", split="val")
    test_set = GSM8K(subset="main", split="test")
    
    # Limit val and test size if specified
    val_subset = [val_set[i] for i in range(min(args.val_size, len(val_set)))]
    test_subset = [test_set[i] for i in range(min(args.test_size, len(test_set)))]
    
    print(f"Dataset loaded: GSM8K")
    print(f"Train/Val/Test sizes: {len(train_subset)}/{len(val_subset)}/{len(test_subset)}")
    
    # Get initial system prompt
    STARTING_SYSTEM_PROMPT = train_set.get_task_description()
    print(f"Initial system prompt: {STARTING_SYSTEM_PROMPT}")
    
    # Create the system prompt variable
    system_prompt = tg.Variable(
        STARTING_SYSTEM_PROMPT,
        requires_grad=True,
        role_description="system prompt for optimal mathematical reasoning"
    )
    
    # Create model and loss function
    model = tg.BlackboxLLM(claude_engine, system_prompt=system_prompt)
    
    # Use TextLoss for optimization
    loss_system_prompt = "You will evaluate the accuracy and efficiency of responses to mathematical problems."
    loss_fn = tg.TextLoss(loss_system_prompt, engine=claude_engine)
    
    # Create optimizer
    optimizer = tg.TextualGradientDescent(
        engine=claude_engine,
        parameters=[system_prompt],
        constraints=["The prompt must guide the model to solve math problems step-by-step with clear reasoning."]
    )
    
    # Store results
    results = {
        "initial_prompt": STARTING_SYSTEM_PROMPT,
        "epochs": [],
        "final_prompt": "",
        "model": args.model,
    }
    
    # Evaluate initial performance
    print("\nEvaluating initial performance...")
    initial_results, initial_metrics = eval_dataset(
        test_subset, 
        model, 
        loss_fn, 
        max_samples=min(50, len(test_subset))  # Test on a subset first
    )
    
    print(f"Initial metrics: {initial_metrics}")
    
    # Create a dataloader for batched training
    train_loader = tg.tasks.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(args.max_epochs):
        epoch_data = {
            "epoch": epoch,
            "prompt": system_prompt.value,
            "steps": []
        }
        
        # Process mini-batches
        for step, (batch_x, batch_y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            optimizer.zero_grad()
            step_losses = []
            
            for (x, y) in zip(batch_x, batch_y):
                # Create variables
                x_var = tg.Variable(x, requires_grad=False, role_description="query to the model")
                y_var = tg.Variable(y, requires_grad=False, role_description="correct answer")
                
                # Get model response
                response = model(x_var)
                
                # Compute loss
                loss = loss_fn(response)
                step_losses.append(loss)
                
                # Calculate metrics for this example
                from textgrad.tasks.big_bench_hard import string_based_equality_fn
                accuracy = string_based_equality_fn(response, y_var)
                
                step_data = {
                    "question": x,
                    "answer": y,
                    "response": response.value,
                    "accuracy": accuracy
                }
                epoch_data["steps"].append(step_data)
            
            # Backward pass through all losses
            for loss in step_losses:
                loss.backward()
            
            # Update the prompt
            optimizer.step()
            
            print(f"\nPrompt after step {step}:")
            print(system_prompt.value)
            
            # Evaluate on validation set periodically
            if step % 5 == 0:
                val_results, val_metrics = eval_dataset(
                    val_subset,
                    model,
                    loss_fn,
                    max_samples=min(20, len(val_subset))
                )
                print(f"Validation accuracy: {val_metrics['accuracy']:.3f}")
        
        # Evaluate on full validation set at end of epoch
        val_results, val_metrics = eval_dataset(
            val_subset,
            model,
            loss_fn,
            max_samples=len(val_subset)
        )
        
        epoch_data["validation_metrics"] = val_metrics
        results["epochs"].append(epoch_data)
        
        print(f"\nEpoch {epoch} validation metrics: {val_metrics}")
    
    # Final evaluation on test set
    final_results, final_metrics = eval_dataset(
        test_subset,
        model,
        loss_fn,
        max_samples=len(test_subset)
    )
    
    results["final_prompt"] = system_prompt.value
    results["final_metrics"] = final_metrics
    
    # Save results
    os.makedirs(args.results_dir, exist_ok=True)
    result_file = os.path.join(args.results_dir, f"gsm8k_full_claude_{args.model.split('-')[-1]}.json")
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n=== Optimization Complete ===")
    print(f"Initial metrics: {initial_metrics}")
    print(f"Final metrics: {final_metrics}")
    print(f"\nInitial prompt: {STARTING_SYSTEM_PROMPT}")
    print(f"\nFinal prompt: {system_prompt.value}")
    print(f"\nResults saved to: {result_file}")