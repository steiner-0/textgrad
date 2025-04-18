import re
from .mmlu import MMLU, MMLUInstanceDataset
from .base import Dataset, DataLoader
from .leetcode import LeetCodeHardEval

from typing import Tuple, Callable
from textgrad import Variable
from textgrad.engine import EngineLM

AVAILABLE_DATASETS = [
    "BBH_object_counting",
    "BBH_word_sorting",
    "GSM8K_DSPy",
    "MATH500",
]

AVAILABLE_INSTANCE_DATASETS = [
    "MMLU_machine_learning",
    "MMLU_college_physics",
    "GPQA_diamond"
    "LeetCodeHardEval"
]

def load_task(task_name: str, evaluation_api: EngineLM, *args, **kwargs) -> Tuple[Dataset, Dataset, Callable]:
    """
    Args:
        task_name: the name of the task to evaluate
        evaluation_api: the engine to use for evaluation, if needed
    """
    if "object_counting" in task_name:
        from textgrad.loss import MultiFieldTokenParsedEvaluation
        from .big_bench_hard import BigBenchHard, string_based_equality_fn
        from textgrad.autograd.string_based_ops import StringBasedFunction
        task_name = task_name[4:]
        train_set = BigBenchHard(task_name, split="train", *args, **kwargs)
        val_set = BigBenchHard(task_name, split="val", *args, **kwargs)
        test_set = BigBenchHard(task_name, split="test", *args, **kwargs)
        role_descriptions = [
            "Question for the task",
            "Ground truth answer",
            "Reasoning and prediction from the language model"
        ]
        fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
        eval_fn = StringBasedFunction(string_based_equality_fn, function_purpose=fn_purpose)
        return train_set, val_set, test_set, eval_fn
    
    elif "BBH" in task_name:
        from textgrad.loss import MultiFieldTokenParsedEvaluation
        from .big_bench_hard import BigBenchHard
        task_name = task_name[4:]
        train_set = BigBenchHard(task_name, split="train", *args, **kwargs)
        val_set = BigBenchHard(task_name, split="val", *args, **kwargs)
        test_set = BigBenchHard(task_name, split="test", *args, **kwargs)
        role_descriptions = [
            "Question for the task",
            "Ground truth answer",
            "Reasoning and prediction from the language model"
        ]
        
        evaluation_instruction = "Below is a question from a question-answering task, the ground truth answer, and reasoning with the final prediction. Is the final prediction correct, i.e. the same as the ground truth answer? Say only 1 (yes) or 0 (no). Return your response within <ACCURACY> </ACCURACY> tags. e.g.<ACCURACY> 0 </ACCURACY> or <ACCURACY> 1 </ACCURACY>"
        eval_instruction = Variable(evaluation_instruction, requires_grad=False, role_description="evaluation instruction for the task")
        eval_fn = MultiFieldTokenParsedEvaluation(
            eval_instruction,
            engine=evaluation_api,
            role_descriptions=role_descriptions,
            parse_tags=["<ACCURACY>", "</ACCURACY>"]
        )
        
        return train_set, val_set, test_set, eval_fn
    
    elif task_name == "GSM8K_DSPy":
        from textgrad.tasks.gsm8k import GSM8K_DSPy
        from .big_bench_hard import string_based_equality_fn
        from textgrad.autograd.string_based_ops import StringBasedFunction
        evaluation_instruction = "Below is a prediction we got for a question answering task, and the correct final answer. Is the final answer correct? Say only 1 (yes) or 0 (no). Return 1 if and only if the final answer is correct. Return your response within <ACCURACY> </ACCURACY> tags. e.g.<ACCURACY> 0 </ACCURACY> or <ACCURACY> 1 </ACCURACY>"
        system_prompt = Variable("You are a language model that evaluates the accuracy of a prediction for a mathematical question answering task. Only call a prediction accurate if it is the same as the ground truth answer.", requires_grad=False, role_description="system prompt for the evaluation")
        # Should we do train/test like this?
        train_set = GSM8K_DSPy(split="train", *args, **kwargs)
        val_set = GSM8K_DSPy(split="val", *args, **kwargs)
        test_set = GSM8K_DSPy(split="test", *args, **kwargs)
        role_descriptions = [
            "Question for the task",
            "Ground truth answer",
            "Prediction from the language model"
        ]
        fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
        eval_fn = StringBasedFunction(string_based_equality_fn, function_purpose=fn_purpose)
        return train_set, val_set, test_set, eval_fn
    elif task_name == "MATH500":
        from textgrad.loss import MultiFieldTokenParsedEvaluation
        from .math500 import MATH500
        from .big_bench_hard import string_based_equality_fn
        from textgrad.autograd.string_based_ops import StringBasedFunction
        
        train_set = MATH500(split="train", *args, **kwargs)
        val_set = MATH500(split="val", *args, **kwargs)
        test_set = MATH500(split="test", *args, **kwargs)
        
        # We'll use similar evaluation to GSM8K
        fn_purpose = "The function that checks if the prediction is correct for MATH500."
        
        # Custom string equality function for MATH500
        def math500_equality_fn(prediction, ground_truth_answer):
            # Extract answer from prediction
            response_text = prediction.value
            correct_answer = ground_truth_answer.value
            
            # Extract answer from response
            answer_patterns = [
                r"(?:final answer|answer)(?:\s*[:=]\s*)(.*?)(?:\s*$|\n)",
                r"(?:\s|^)(?:final answer|answer)(?:\s+is\s+)(.*?)(?:\s*$|\n)",
                r"(?:\s|^)(?:the answer is\s+)(.*?)(?:\s*$|\n)",
                r"(?:\s|^)(?:therefore,\s+.*?=\s*)(.*?)(?:\s*$|\n)"
            ]
            
            extracted_answer = None
            for pattern in answer_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    extracted_answer = match.group(1).strip()
                    break
            
            if not extracted_answer:
                # If no clear answer pattern, take the last line or equation result
                lines = response_text.strip().split('\n')
                for line in reversed(lines):
                    if '=' in line:
                        extracted_answer = line.split('=')[-1].strip()
                        break
                if not extracted_answer and lines:
                    extracted_answer = lines[-1].strip()
            
            # Clean up both answers for comparison
            def clean_answer(ans):
                # Remove whitespace, convert to lowercase
                ans = re.sub(r'\s+', '', ans.lower())
                # Remove common math notation that doesn't affect the value
                ans = re.sub(r'\\', '', ans)  # LaTeX backslashes
                ans = re.sub(r'[(){}[\]]', '', ans)  # Brackets
                return ans
            
            if extracted_answer:
                clean_extracted = clean_answer(extracted_answer)
                clean_correct = clean_answer(correct_answer)
                
                # Check for exact match
                is_correct = clean_extracted == clean_correct
                
                return int(is_correct)
            return 0  # No answer extracted
        
        eval_fn = StringBasedFunction(math500_equality_fn, function_purpose=fn_purpose)
        return train_set, val_set, test_set, eval_fn
    else:
        raise ValueError(f"Task {task_name} not found.")


def load_instance_task(task_name: str, evaluation_api: EngineLM, *args, **kwargs):
    if "MMLU_" in task_name:
        subset = task_name[5:]
        test_set = MMLUInstanceDataset(evaluation_api=evaluation_api, subset=subset, split="test", *args, **kwargs)
        return test_set
    elif "GPQA" in task_name:
        from .gpqa import GPQAInstanceDataset
        test_set = GPQAInstanceDataset(evaluation_api=evaluation_api, subset=task_name.lower(), *args, **kwargs)
        return test_set
    elif task_name in ["LeetCodeHardEval"]:
        dataset = LeetCodeHardEval()
        return dataset
    else:
        raise ValueError(f"Instance task {task_name} not found.")