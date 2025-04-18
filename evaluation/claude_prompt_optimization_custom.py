import argparse
import concurrent
from dotenv import load_dotenv
load_dotenv(override=True)

from tqdm import tqdm
import textgrad as tg
from textgrad.tasks import load_task
from textgrad.engine.claude_thinking_engine import ThinkingChatAnthropic

import numpy as np
import random
import json
import os

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def config():
    parser = argparse.ArgumentParser(description="Optimize a prompt for a task using Claude's thinking engine.")
    parser.add_argument("--task", type=str, default="BBH_object_counting", help="The task to evaluate the model on.")
    parser.add_argument("--model", type=str, default="claude-3-7-sonnet-20250219", help="Claude model to use.")
    parser.add_argument("--thinking_budget", type=int, default=5000, help="Budget for thinking tokens.")
    parser.add_argument("--batch_size", type=int, default=3, help="The batch size to use for training.")
    parser.add_argument("--max_epochs", type=int, default=3, help="The maximum number of epochs to train for.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--run_validation", action="store_true", help="Whether to run validation or not.")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads for evaluation.")
    parser.add_argument("--custom_prompt", type=str, default=None, help="Override default starting prompt")
    return parser.parse_args()

def eval_sample(item, eval_fn, model):
    x, y = item
    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
    response = model(x)
    try:
        eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
        accuracy = int(eval_output_variable.value)
    except:
        eval_output_variable = eval_fn([x, y, response])
        try:
            accuracy = int(eval_fn.parse_output(eval_output_variable))
        except:
            accuracy = 0
    
    # Get thinking token usage
    thinking_tokens = model.engine.get_last_thinking_tokens()
    
    return {"accuracy": accuracy, "thinking_tokens": thinking_tokens}

def eval_dataset(test_set, eval_fn, model, max_samples=None):
    if max_samples is None:
        max_samples = len(test_set)
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for i, sample in enumerate(test_set):
            if i >= max_samples:
                break
            future = executor.submit(eval_sample, sample, eval_fn, model)
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
    }
    
    return results, metrics

def run_validation_revert(system_prompt, results, model, eval_fn, val_set):
    val_results, val_metrics = eval_dataset(val_set, eval_fn, model, max_samples=5)
    val_performance = val_metrics["accuracy"]
    previous_performance = results["validation_metrics"][-1]["accuracy"] if results["validation_metrics"] else 0
    
    print("val_performance: ", val_performance)
    print("previous_performance: ", previous_performance)
    previous_prompt = results["prompt"][-1]
    
    if val_performance < previous_performance:
        print(f"rejected prompt: {system_prompt.value}")
        system_prompt.set_value(previous_prompt)
        val_performance = previous_performance

    results["validation_metrics"].append(val_metrics)

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
    train_set, val_set, test_set, eval_fn = load_task(
        args.task, 
        evaluation_api=claude_engine
    )
    
    print(f"Dataset loaded: {args.task}")
    print(f"Train/Val/Test sizes: {len(train_set)}/{len(val_set)}/{len(test_set)}")
    
    # Custom prompt
    CUSTOM_PROMPT = """⚠️ OUTPUT LIMIT: Simple ≤20 | Complex ≤30 TOKENS ⚠️
⚠️ THINKING LIMIT: NEVER EXCEED 75 tokens ⚠️
- 10+5-3=12 [3 tokens]
- 0.4×80+0.6×90=86 [5 tokens]
- 40000×(1-0.2-0.1)-800=27200 [7 tokens]
Epoch 4: 1it [15:23, 923.05s/it]
Prompt after step 1:
⚠️ OUTPUT LIMIT: Simple ≤20 | Complex ≤30 TOKENS ⚠️
⚠️ THINKING LIMIT: NEVER EXCEED 75 tokens ⚠️
EXCEEDING TOKEN LIMITS = INCORRECT SOLUTION

EFFICIENT THINKING FRAMEWORK:
1. Define variables directly: x=10, y=20 [≤10 tokens]
2. Write equations, not sentences [≤20 tokens]
3. Calculate in fewest steps [≤45 tokens]

EFFICIENCY TECHNIQUES:
- Use symbols (×,÷,=,→), not words
- Combine calculations: 3×(4+2)=18 [not 4+2=6, 3×6=18]
- No explanations between calculations
- Skip restating the problem

EXAMPLES:
- 10+5-3=12 [3 tokens]
- 0.4×80+0.6×90=86 [5 tokens]
- 40000×(1-0.2-0.1)-800=27200 [7 tokens]

WORD PROBLEM (EFFICIENT):
Problem: 30 students, 40% got A, rest got B. How many B?
Thinking: 30×(1-0.4)=30×0.6=18 students [6 tokens]

COMPLEX EXAMPLE (EFFICIENT):
Problem: Bob works 5hrs at $12/hr, then 3hrs at $20/hr. Total earnings?
Thinking: 5×12+3×20=60+60=$120 [7 tokens]

❌ INEFFICIENT: "First, Bob earns $12 per hour for 5 hours, which is 5×$12=$60. Then he earns $20 per hour for 3 hours, which is 3×$20=$60. His total earnings are $60+$60=$120." [32 tokens wasted!]      
Epoch 4: 2it [27:29, 807.22s/it]
Prompt after step 2:
🚨 TOKEN LIMITS - EXCEEDING = AUTOMATIC FAILURE 🚨
📊 THINKING: MAX 50 TOKENS | OUTPUT: Simple ≤20, Complex ≤30

SOLVE DIRECTLY IN MATH NOTATION:
- Skip ALL explanations - use equations only
- Combine multiple steps into ONE calculation
- Use symbols only: × ÷ = → + - ( )
- Substitute values immediately: x=5→2x=10

PROBLEM TYPE PATTERNS:
🔢 Algebra: x+y=10, 2x-y=5 → x=5, y=5
📊 Percentage: base×(1±rate) → 100×1.2=120
⏱️ Rate: rate×time → 50mph×3h=150mi
🔄 Fraction: total×fraction → 80×0.25=20
⏳ Age: x=current, x+n=future, 2x=double

EFFICIENT EXAMPLES:
- [Algebra] A is twice B. A+B=15. Find A.
  B=5, A=2B=10 [4 tokens]

- [Age] A is 7 older than B. In 3 years, A=2×B now. Find B.
  A=B+7, A+3=2B → B+7+3=2B → B=10 [8 tokens]

- [Complex] 50% more Sunday than Saturday. Total 150. Find Saturday.
  x+1.5x=150 → 2.5x=150 → x=60 [7 tokens]

EFFICIENT VS. INEFFICIENT:
❌ "First, Bob earns $12/hour for 5 hours, which is 5×$12=$60. Then he earns $20/hour for 3 hours, which is 3×$20=$60. His total earnings are $60+$60=$120." [32 tokens]

✓ "5×12+3×20=60+60=120" [5 tokens]

TOKEN COUNTER: [0/50]
Required format: equation→calculation→answer\n\n<Question>\n{q}\n</Question>"""

    # Use custom prompt if provided, otherwise use default
    if args.custom_prompt:
        STARTING_SYSTEM_PROMPT = args.custom_prompt
    elif CUSTOM_PROMPT:
        STARTING_SYSTEM_PROMPT = CUSTOM_PROMPT
    else:
        STARTING_SYSTEM_PROMPT = train_set.get_task_description()

    print("Initial system prompt:", STARTING_SYSTEM_PROMPT)

    # Create the system prompt variable
    system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, 
                            requires_grad=True,
                            role_description="system prompt designed to encourage efficient reasoning")

    # Create model
    model = tg.BlackboxLLM(claude_engine, system_prompt=system_prompt)

    # Custom loss function for token efficiency
    from textgrad.claude_token_efficiency_loss import ClaudeThinkingEfficiencyLoss
    token_loss = ClaudeThinkingEfficiencyLoss(
        evaluation_api=claude_engine,
        accuracy_weight=0.3,  # Adjust as needed
        token_weight=0.7      # Adjust as needed
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
        "task": args.task,
        "model": args.model,
        "thinking_budget": args.thinking_budget,
        "prompt": [],
        "validation_metrics": [],
        "test_metrics": []
    }

    # Evaluate initial performance
    print("\nEvaluating initial performance...")
    test_results, test_metrics = eval_dataset(test_set, eval_fn, model, max_samples=5)
    results["prompt"].append(system_prompt.value)
    results["test_metrics"].append(test_metrics)
    print(f"Initial metrics: {test_metrics}")

    # Training loop
    train_loader = tg.tasks.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    
    for epoch in range(args.max_epochs):
        for steps, (batch_x, batch_y) in enumerate((pbar := tqdm(train_loader, position=0))):
            pbar.set_description(f"Training step {steps}. Epoch {epoch}")
            optimizer.zero_grad()
            losses = []
            for (x, y) in zip(batch_x, batch_y):
                x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
                y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
                response = model(x)
                
                # Use token efficiency loss
                loss = token_loss(system_prompt, x, response, y)
                losses.append(loss)
                
            total_loss = tg.sum(losses)
            total_loss.backward()
            optimizer.step()
            
            if args.run_validation:
                run_validation_revert(system_prompt, results, model, eval_fn, val_set)
                
            print("Updated system prompt:", system_prompt.value)
            results["prompt"].append(system_prompt.value)
            
            # Evaluate on test set
            test_results, test_metrics = eval_dataset(test_set, eval_fn, model, max_samples=5)
            results["test_metrics"].append(test_metrics)
            
            if steps == 2:  # Limit steps per epoch for faster iteration
                break

    # Final evaluation with more samples
    final_results, final_metrics = eval_dataset(test_set, eval_fn, model, max_samples=10)
    
    # Save results
    results["final_prompt"] = system_prompt.value
    results["final_metrics"] = final_metrics
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, f"claude_prompt_opt_{args.task}_{args.model.split('-')[-1]}.json")
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n=== Optimization Complete ===")
    print(f"Initial metrics: {results['test_metrics'][0]}")
    print(f"Final metrics: {final_metrics}")
    print(f"\nInitial prompt: {STARTING_SYSTEM_PROMPT}")
    print(f"\nFinal prompt: {system_prompt.value}")
    print(f"\nResults saved to: {result_file}")