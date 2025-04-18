# Balanced Prompt Optimization System

This system optimizes prompts for both answer accuracy and reasoning efficiency using Claude 3.7 Sonnet. It alternates between optimization goals according to user-specified weights.

## Overview

The system contains:

1. `balanced_prompt_optimization.py` - The main optimization script that balances accuracy and efficiency
2. `run_experiments.py` - A utility to run multiple experiments with different weight configurations
3. `analyze_results.py` - A tool to analyze and visualize experiment results

## How It Works

### Weight-Based Optimization

This system optimizes prompts by balancing two objectives:

1. **Accuracy**: Improving the correctness of answers
2. **Efficiency**: Reducing token usage while maintaining accuracy

The balance between these objectives is controlled by two weights:
- `accuracy_weight`: Controls focus on improving answer correctness
- `efficiency_weight`: Controls focus on reducing token usage

For example, with weights of 0.3 for accuracy and 0.7 for efficiency, the system will:
- Focus on accuracy improvement approximately 30% of the time
- Focus on efficiency improvement approximately 70% of the time

The actual alternation between objectives is probabilistic and follows the specified weights.

### Implementation Details

The system uses TextGrad to optimize prompts:

1. For each epoch, the system randomly selects a focus (accuracy or efficiency) based on the weights
2. The appropriate loss function is used to generate feedback about the prompt
3. Gradients are computed and the prompt is updated
4. The process repeats for the specified number of epochs

Throughout training, the system tracks both accuracy and token usage metrics to evaluate performance.

## Usage

### Basic Usage

Run the main optimization script with specified weights:

```bash
python balanced_prompt_optimization.py --task GSM8K_DSPy --accuracy_weight 0.3 --efficiency_weight 0.7
```

### Using a Custom Prompt

You can specify a custom starting prompt in two ways:

```bash
# Directly in the command line
python balanced_prompt_optimization.py --task GSM8K_DSPy --custom_prompt "Solve math problems step by step"

# From a file
python balanced_prompt_optimization.py --task GSM8K_DSPy --prompt_file my_prompts/math_prompt.txt
```

### Using Validation with Reversion

To enable validation after each step with potential reversion to the previous prompt if performance decreases:

```bash
python balanced_prompt_optimization.py --task GSM8K_DSPy --run_validation
```

### Running Multiple Experiments

To test multiple weight configurations:

```bash
python run_experiments.py --task GSM8K_DSPy --configs "0.3,0.7" "0.5,0.5" "0.7,0.3"
```

With a custom starting prompt and validation:

```bash
python run_experiments.py --task GSM8K_DSPy --prompt_file my_prompts/math_prompt.txt --configs "0.3,0.7" "0.5,0.5" --run_validation
```

### Analyzing Results

After running experiments, analyze the results:

```bash
python analyze_results.py --task GSM8K_DSPy --output analysis_report.html
```

This generates an HTML report and comparison plots.

## Parameters

### Main Script Parameters

- `--task`: The task to evaluate (e.g., "GSM8K_DSPy")
- `--model`: Claude model to use (default: "claude-3-7-sonnet-20250219")
- `--custom_prompt`: Custom starting prompt (overrides task's default prompt)
- `--prompt_file`: File containing custom starting prompt
- `--batch_size`: Batch size for training (default: 3)
- `--max_epochs`: Maximum number of epochs (default: 10)
- `--accuracy_weight`: Weight for accuracy optimization (0-1)
- `--efficiency_weight`: Weight for efficiency optimization (0-1)
- `--num_threads`: Number of threads for parallel evaluation (default: 4)
- `--eval_samples`: Number of samples to use for evaluation (default: 5)
- `--thinking_budget`: Budget for thinking tokens (default: 16000)
- `--thinking_enabled`: Enable Claude's thinking feature (default: True)
- `--run_validation`: Run validation and revert to previous prompt if performance decreases

## Output

The system generates:

1. JSON result files in the `results/` directory
2. An HTML analysis report with interactive charts
3. Comparison plots in the `plots/` directory

## Example Results

For a task like GSM8K_DSPy, you might see results like:

| Weights (Acc/Eff) | Initial Acc | Final Acc | Improvement | Initial Tokens | Final Tokens | Reduction | % Reduction |
|-------------------|-------------|-----------|-------------|----------------|--------------|-----------|-------------|
| 0.3/0.7           | 0.600       | 0.650     | 0.050       | 1200.5         | 950.3        | 250.2     | 20.8%       |
| 0.5/0.5           | 0.600       | 0.680     | 0.080       | 1200.5         | 1050.2       | 150.3     | 12.5%       |
| 0.7/0.3           | 0.600       | 0.720     | 0.120       | 1200.5         | 1100.1       | 100.4     | 8.4%        |

These results would show that:
- Higher accuracy weights lead to better accuracy improvements
- Higher efficiency weights lead to better token reductions
- There's a trade-off between accuracy and efficiency

## Requirements

- Python 3.8+
- textgrad
- pandas
- matplotlib
- numpy
- tabulate
- tqdm

## Environment Setup

1. Install dependencies:
```bash
pip install textgrad pandas matplotlib numpy tabulate tqdm
```

2. Set up API keys:
```bash
export ANTHROPIC_API_KEY=your_api_key_here
```