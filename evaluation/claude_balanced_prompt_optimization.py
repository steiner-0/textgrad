"""
Balanced Prompt Optimization Tool

This script optimizes prompts for both accuracy and reasoning efficiency
using Claude 3.7 Sonnet with streaming support. It alternates between optimization 
goals according to specified weights.
"""

import argparse
import concurrent.futures
from dotenv import load_dotenv
import os
import random
import json
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union

import textgrad as tg
from textgrad.tasks import load_task
from textgrad.engine.anthropic import ChatAnthropic
from textgrad.variable import Variable
from textgrad.autograd import FormattedLLMCall

# Import Anthropic libraries for streaming
from anthropic import Anthropic

# Define a thinking-enabled Claude engine with streaming support
class StreamingThinkingChatAnthropic(ChatAnthropic):
    """Extended ChatAnthropic engine with thinking parameter and streaming support."""
    
    def __init__(
        self,
        model_string="claude-3-7-sonnet-20250219",
        system_prompt="You are a helpful, creative, and smart assistant.",
        thinking_enabled=True,
        thinking_budget=40000,
        is_multimodal=False,
    ):
        super().__init__(
            model_string=model_string,
            system_prompt=system_prompt,
            is_multimodal=is_multimodal
        )
        self.thinking_enabled = thinking_enabled
        self.thinking_budget = thinking_budget
        self.last_thinking = None
        self.last_thinking_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        
    def generate(self, content, system_prompt=None, **kwargs):
        """Override generate to include thinking parameter and track token usage, with streaming support."""
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        
        # Configure thinking parameter
        thinking_config = None
        if self.thinking_enabled:
            thinking_config = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget
            }
        
        # Set up storage for response and thinking content
        complete_response = ""
        thinking_content = ""
        
        # Define a function to run the streaming request
        def run_streaming_request():
            nonlocal complete_response, thinking_content
            
            # Create the client for streaming
            with self.client.messages.stream(
                model=self.model_string,
                max_tokens=60000,
                system=sys_prompt_arg,
                thinking=thinking_config,
                messages=[
                    {"role": "user", "content": content}
                ]
            ) as stream:
                print("\nStreaming response...")
                
                # Process the streaming response events
                for event in stream:
                    if event.type == "content_block_start":
                        print(f"\nStarting {event.content_block.type} block...")
                    
                    elif event.type == "content_block_delta":
                        if event.delta.type == "thinking_delta":
                            thinking_content += event.delta.thinking
                            print("*", end="", flush=True)  # Indicate thinking with *
                        elif event.delta.type == "text_delta":
                            complete_response += event.delta.text
                            print(".", end="", flush=True)  # Indicate response with .
                    
                    elif event.type == "content_block_stop":
                        print("\nBlock complete.")
            
            print("\nStreaming complete.")
        
        # Run the streaming request
        run_streaming_request()
        
        # Estimate thinking tokens (could use a proper tokenizer)
        import tiktoken
        try:
            encoder = tiktoken.get_encoding("cl100k_base")
            self.last_thinking_tokens = len(encoder.encode(thinking_content))
        except:
            # Fallback to character-based estimation if tokenizer fails
            self.last_thinking_tokens = len(thinking_content) // 4
        
        self.last_completion_tokens = len(encoder.encode(complete_response))
        self.last_total_tokens = self.last_thinking_tokens + self.last_completion_tokens
        
        # Store thinking as a simple object with text attribute
        class ThinkingContent:
            def __init__(self, text):
                self.text = text
                
        self.last_thinking = ThinkingContent(text=thinking_content)
        
        return complete_response
    
    def get_last_thinking_text(self):
        """Get the last thinking text."""
        return self.last_thinking.text 
    
    def get_last_thinking_tokens(self):
        """Get token count from the last thinking process."""
        return self.last_thinking_tokens


# Define the accuracy optimization loss
class AccuracyLoss(tg.autograd.Module):
    """Loss function focused on improving answer accuracy."""
    
    def __init__(self, evaluation_api):
        """Initialize the accuracy loss."""
        super().__init__()
        self.evaluation_api = evaluation_api
        
        # Create system prompt for accuracy evaluation
        self.system_prompt = Variable(
            "You are an evaluator that provides feedback on how to improve accuracy for problem-solving.",
            requires_grad=False,
            role_description="system prompt for accuracy evaluation"
        )
        
        # Format string for evaluation
        self.format_string = (
            "Below is a system prompt, a question, and the model's response to it.\n\n"
            "System Prompt: {system_prompt}\n\n"
            "Question: {question}\n\n"
            "Model Response: {response}\n\n"
            "Correct Answer: {correct_answer}\n\n"
            "The model's answer is {'correct' if is_correct else 'incorrect'}.\n\n"
            "Provide specific feedback on how the system prompt could be improved to make the model's "
            "response more accurate while solving this type of problem. Focus on encouraging more "
            "precise reasoning steps, better problem understanding, and reliable calculation methods."
        )
        
        self.fields = {
            "system_prompt": None, 
            "question": None, 
            "response": None, 
            "correct_answer": None,
            "is_correct": None
        }
        
        self.formatted_llm_call = FormattedLLMCall(
            engine=self.evaluation_api,
            format_string=self.format_string,
            fields=self.fields,
            system_prompt=self.system_prompt
        )
    
    def forward(self, system_prompt, question, response, correct_answer):
        """
        Calculate the loss based on accuracy.
        
        Args:
            system_prompt: The system prompt used
            question: The question asked
            response: The model's response
            correct_answer: The correct answer
            
        Returns:
            A Variable containing feedback on how to improve accuracy
        """
        # Check if the answer is correct (simple string matching for demo)
        # In practice, you'd want to use a more sophisticated evaluation
        is_correct = correct_answer.value in response.value
        
        # Prepare inputs for the formatter
        inputs = {
            "system_prompt": system_prompt,
            "question": question,
            "response": response,
            "correct_answer": correct_answer,
            "is_correct": Variable(str(is_correct), requires_grad=False, role_description="correctness flag")
        }
        
        # Get feedback on accuracy
        accuracy_feedback = self.formatted_llm_call(
            inputs=inputs,
            response_role_description="feedback on accuracy"
        )
        
        return accuracy_feedback


# Define the efficiency optimization loss
class EfficiencyLoss(tg.autograd.Module):
    """Loss function focused on improving reasoning efficiency."""
    
    def __init__(self, evaluation_api):
        """Initialize the efficiency loss."""
        super().__init__()
        self.evaluation_api = evaluation_api
        
        # System prompt for efficiency evaluation
        self.system_prompt = Variable(
            "You are an evaluator that provides feedback on how to make prompts encourage more efficient reasoning.",
            requires_grad=False,
            role_description="system prompt for efficiency evaluation"
        )
        
        # Format string for evaluation
        self.format_string = (
            "Below is a system prompt, a question, and the model's thinking process when solving it.\n\n"
            "System Prompt: {system_prompt}\n\n"
            "Question: {question}\n\n"
            "Thinking Process:\n{thinking_text}\n\n"
            "The model used {token_count} tokens for this thinking process.\n\n"
            "Evaluate this system prompt on reasoning efficiency. Provide specific feedback on how the "
            "system prompt could be improved to make the model reason more efficiently while still reaching "
            "the correct answer. Focus on encouraging the model to eliminate unnecessary steps, reduce redundancy, "
            "and adopt a more concise thinking process."
        )
        
        self.fields = {
            "system_prompt": None, 
            "question": None, 
            "thinking_text": None, 
            "token_count": None
        }
        
        self.formatted_llm_call = FormattedLLMCall(
            engine=self.evaluation_api,
            format_string=self.format_string,
            fields=self.fields,
            system_prompt=self.system_prompt
        )
    
    def forward(self, system_prompt, question, response, correct_answer=None):
        """
        Calculate the loss based on thinking efficiency.
        
        Args:
            system_prompt: The system prompt used
            question: The question asked
            response: The model's response
            correct_answer: The correct answer (optional)
            
        Returns:
            A Variable containing feedback on how to improve efficiency
        """
        # Get the thinking text and token count
        thinking_text = self.evaluation_api.get_last_thinking_text()
        token_count = self.evaluation_api.get_last_thinking_tokens()
        
        # Prepare inputs for the formatter
        inputs = {
            "system_prompt": system_prompt,
            "question": question,
            "thinking_text": Variable(thinking_text, requires_grad=False, role_description="model's thinking process"),
            "token_count": Variable(str(token_count), requires_grad=False, role_description="token count")
        }
        
        # Get feedback on efficiency
        efficiency_feedback = self.formatted_llm_call(
            inputs=inputs,
            response_role_description="feedback on thinking efficiency"
        )
        
        return efficiency_feedback


def eval_sample(sample, model, task_eval_fn=None):
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
            try:
                eval_output = task_eval_fn([x_var, y_var, response])
                accuracy = int(task_eval_fn.parse_output(eval_output))
            except:
                # Fallback to string matching
                accuracy = 1 if y in response.value else 0
    else:
        # Default string match
        accuracy = 1 if y in response.value else 0
    
    # Get token usage
    token_count = model.engine.get_last_thinking_tokens()
    
    return {
        "question": x,
        "correct_answer": y,
        "response": response.value,
        "accuracy": accuracy,
        "token_count": token_count
    }


def eval_dataset(dataset, model, task_eval_fn=None, max_samples=None, num_threads=16):
    """Evaluate a dataset for accuracy and token efficiency."""
    if max_samples is None or max_samples > len(dataset):
        max_samples = len(dataset)
    
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
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
                avg_tokens = np.mean([r["token_count"] for r in results])
                pbar.set_description(f"Acc: {avg_accuracy:.3f}, Tokens: {avg_tokens:.1f}")
                pbar.update(1)
    
    # Calculate metrics
    metrics = {
        "accuracy": np.mean([r["accuracy"] for r in results]),
        "token_count": np.mean([r["token_count"] for r in results]),
        "accuracy_stdev": np.std([r["accuracy"] for r in results]),
        "token_count_stdev": np.std([r["token_count"] for r in results]),
    }
    
    return results, metrics

def run_validation_revert(system_prompt, previous_prompt, model, eval_fn, val_set, max_samples=None, num_threads=None):
    """
    Evaluate performance on validation set and revert to previous prompt if performance decreases.
    
    Args:
        system_prompt: Current system prompt Variable
        previous_prompt: Previous system prompt string value
        model: The model to evaluate
        eval_fn: Evaluation function
        val_set: Validation dataset
        max_samples: Maximum number of validation samples to use
        num_threads: Number of threads for parallel evaluation
        
    Returns:
        float: Validation accuracy after potential reversion
    """
    # Evaluate on validation set
    val_results, val_metrics = eval_dataset(
        val_set, 
        model, 
        eval_fn, 
        max_samples=max_samples,
        num_threads=num_threads
    )
    
    current_val_performance = val_metrics["accuracy"]
    
    # If a previous prompt exists, compare performance
    if previous_prompt:
        # Store current prompt value
        current_prompt = system_prompt.value
        
        # Temporarily revert to previous prompt for comparison
        system_prompt.set_value(previous_prompt)
        
        # Evaluate with previous prompt
        prev_val_results, prev_val_metrics = eval_dataset(
            val_set,
            model,
            eval_fn,
            max_samples=max_samples,
            num_threads=num_threads
        )
        
        previous_val_performance = prev_val_metrics["accuracy"]
        
        print(f"Validation performance - Current: {current_val_performance:.3f}, Previous: {previous_val_performance:.3f}")
        
        # If current prompt performs worse, keep reverting to previous prompt
        # Otherwise, restore the current prompt
        if current_val_performance < previous_val_performance:
            print(f"Reverting to previous prompt (validation accuracy: {previous_val_performance:.3f} > {current_val_performance:.3f})")
            return previous_val_performance
        else:
            # Restore current prompt as it performs better
            system_prompt.set_value(current_prompt)
            print(f"Keeping current prompt (validation accuracy: {current_val_performance:.3f} >= {previous_val_performance:.3f})")
            return current_val_performance
    
    # If no previous prompt to compare, just return current performance
    return current_val_performance

def config():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Balanced prompt optimization for accuracy and efficiency.")
    parser.add_argument("--task", type=str, default="GSM8K_DSPy", help="The task to evaluate the model on.")
    parser.add_argument("--model", type=str, default="claude-3-7-sonnet-20250219", help="Claude model to use.")
    parser.add_argument("--custom_prompt", type=str, default=None, help="Custom starting prompt (overrides task's default prompt).")
    parser.add_argument("--prompt_file", type=str, default=None, help="File containing custom starting prompt.")
    parser.add_argument("--batch_size", type=int, default=3, help="The batch size to use for training.")
    parser.add_argument("--max_epochs", type=int, default=10, help="The maximum number of epochs to train for.")
    parser.add_argument("--accuracy_weight", type=float, default=0.3, help="Weight for accuracy optimization (0-1).")
    parser.add_argument("--efficiency_weight", type=float, default=0.7, help="Weight for efficiency optimization (0-1).")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads for evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--eval_samples", type=int, default=5, help="Number of samples to use for evaluation.")
    parser.add_argument("--thinking_budget", type=int, default=40000, help="Budget for thinking tokens.")
    parser.add_argument("--thinking_enabled", action="store_true", default=True, help="Enable Claude's thinking feature.")
    parser.add_argument("--run_validation", action="store_true", default=True, help="Run validation after each step and revert if performance decreases.")
    parser.add_argument("--disable_streaming", action="store_true", help="Disable streaming responses.")
    return parser.parse_args()


def main():
    """Main execution function."""
    # Load environment variables
    load_dotenv(override=True)
    
    # Parse arguments
    args = config()
    
    # Set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create Claude engine with thinking support and streaming
    claude_engine = StreamingThinkingChatAnthropic(
        model_string=args.model,
        thinking_enabled=args.thinking_enabled,
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
    
    # Determine starting prompt
    if args.custom_prompt:
        # Use custom prompt provided as command line argument
        STARTING_SYSTEM_PROMPT = args.custom_prompt
        print(f"Using custom prompt from command line argument")
    elif args.prompt_file:
        # Load custom prompt from file
        try:
            with open(args.prompt_file, 'r') as f:
                STARTING_SYSTEM_PROMPT = f.read().strip()
            print(f"Loaded custom prompt from file: {args.prompt_file}")
        except Exception as e:
            print(f"Error loading prompt file: {e}")
            print(f"Falling back to task's default prompt")
            STARTING_SYSTEM_PROMPT = train_set.get_task_description()
    else:
        # Use task's default prompt
        STARTING_SYSTEM_PROMPT = train_set.get_task_description()
        print(f"Using task's default prompt")
    
    print(f"Initial system prompt: {STARTING_SYSTEM_PROMPT}")
    
    # Create the system prompt variable
    system_prompt = tg.Variable(
        STARTING_SYSTEM_PROMPT,
        requires_grad=True,
        role_description="system prompt optimized for both accuracy and efficiency"
    )
    
    # Create model and loss functions
    model = tg.BlackboxLLM(claude_engine, system_prompt=system_prompt)
    accuracy_loss = AccuracyLoss(evaluation_api=claude_engine)
    efficiency_loss = EfficiencyLoss(evaluation_api=claude_engine)
    
    # Create optimizer
    optimizer = tg.TextualGradientDescent(
        engine=claude_engine,
        parameters=[system_prompt],
        constraints=[
            "The prompt must balance accuracy and efficiency in problem-solving.",
            "The prompt should be clear and concise, avoiding redundancy."
        ]
    )
    
    # Store results
    results = {
        "initial_prompt": STARTING_SYSTEM_PROMPT,
        "epochs": [],
        "final_prompt": "",
        "task": args.task,
        "model": args.model,
        "accuracy_weight": args.accuracy_weight,
        "efficiency_weight": args.efficiency_weight
    }
    
    # Normalize weights
    total_weight = args.accuracy_weight + args.efficiency_weight
    accuracy_weight = args.accuracy_weight / total_weight
    efficiency_weight = args.efficiency_weight / total_weight
    
    print(f"Normalized weights - Accuracy: {accuracy_weight:.2f}, Efficiency: {efficiency_weight:.2f}")
    
    # Evaluate initial performance
    print("\nEvaluating initial performance...")
    initial_results, initial_metrics = eval_dataset(
        test_set, 
        model, 
        task_eval_fn, 
        max_samples=args.eval_samples,
        num_threads=args.num_threads
    )
    
    print(f"Initial metrics: {initial_metrics}")
    results["initial_metrics"] = initial_metrics
    
    # Keep track of prompts and validation scores
    previous_prompt = None
    val_accuracy_history = []
    
    # Training loop
    train_loader = tg.tasks.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    
    for epoch in range(args.max_epochs):
        epoch_data = {
            "epoch": epoch,
            "prompt": system_prompt.value,
            "steps": []
        }
        
        # Determine focus for this epoch based on weights
        # If accuracy_weight is 0.3 and efficiency_weight is 0.7, 
        # we'll focus on accuracy 30% of the time and efficiency 70% of the time
        focus_on_accuracy = random.random() < accuracy_weight
        
        focus_name = "accuracy" if focus_on_accuracy else "efficiency"
        print(f"\nEpoch {epoch}: Focusing on {focus_name}")
        
        loss_fn = accuracy_loss if focus_on_accuracy else efficiency_loss
        
        for step, (batch_x, batch_y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            # Save the current prompt before updating
            previous_prompt = system_prompt.value
            
            optimizer.zero_grad()
            step_losses = []
            
            for (x, y) in zip(batch_x, batch_y):
                # Create variables
                x_var = tg.Variable(x, requires_grad=False, role_description="query to the model")
                y_var = tg.Variable(y, requires_grad=False, role_description="correct answer")
                
                # Get model response
                print(f"\nProcessing example: {x[:100]}...")  # Show start of the query
                response = model(x_var)
                
                # Compute loss based on current focus
                loss = loss_fn(system_prompt, x_var, response, y_var)
                step_losses.append(loss)
                
                # Calculate metrics for this example
                accuracy = 1 if y in response.value else 0
                token_count = claude_engine.get_last_thinking_tokens()
                
                step_data = {
                    "focus": focus_name,
                    "question": x,
                    "answer": y,
                    "response": response.value,
                    "accuracy": accuracy,
                    "token_count": token_count,
                }
                epoch_data["steps"].append(step_data)
            
            # Backward pass through all losses
            for loss in step_losses:
                loss.backward()
            
            # Update the prompt
            optimizer.step()
            
            print(f"\nPrompt after step {step}:")
            print(system_prompt.value)
            
            # Validate and potentially revert
            if args.run_validation:
                val_accuracy = run_validation_revert(
                    system_prompt=system_prompt,
                    previous_prompt=previous_prompt,
                    model=model,
                    eval_fn=task_eval_fn,
                    val_set=val_set,
                    max_samples=args.eval_samples,
                    num_threads=args.num_threads
                )
                val_accuracy_history.append(val_accuracy)
            
            # Break after a few steps to keep the process manageable
            if step >= 2:
                break
        
        # Evaluate on validation set if not done already in run_validation_revert
        if not args.run_validation:
            val_results, val_metrics = eval_dataset(
                val_set,
                model,
                task_eval_fn,
                max_samples=args.eval_samples,
                num_threads=args.num_threads
            )
            epoch_data["validation_metrics"] = val_metrics
            print(f"\nEpoch {epoch} validation metrics: {val_metrics}")
        else:
            # If we've been running validation during steps, use the latest result
            epoch_data["validation_metrics"] = {"accuracy": val_accuracy_history[-1] if val_accuracy_history else 0}
        
        epoch_data["focus"] = focus_name
        results["epochs"].append(epoch_data)
    
    # Final evaluation on test set
    final_results, final_metrics = eval_dataset(
        test_set,
        model,
        task_eval_fn,
        max_samples=args.eval_samples * 2,
        num_threads=args.num_threads
    )
    
    results["final_prompt"] = system_prompt.value
    results["final_metrics"] = final_metrics
    
    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(
        results_dir, 
        f"balanced_opt_{args.task}_acc{args.accuracy_weight}_eff{args.efficiency_weight}.json"
    )
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n=== Optimization Complete ===")
    print(f"Initial metrics: {initial_metrics}")
    print(f"Final metrics: {final_metrics}")
    print(f"\nInitial prompt: {STARTING_SYSTEM_PROMPT}")
    print(f"\nFinal prompt: {system_prompt.value}")
    print(f"\nResults saved to: {result_file}")


if __name__ == "__main__":
    main()