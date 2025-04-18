# textgrad/tasks/math500.py
import platformdirs
import re
from .base import Dataset
from textgrad.variable import Variable
from textgrad.loss import MultiFieldTokenParsedEvaluation
from textgrad.loss import MultiChoiceTestTime

class MATH500(Dataset):
    def __init__(self, evaluation_api, root: str=None, split: str="test", *args, **kwargs):
        """
        MATH500 dataset from HuggingFaceH4/MATH-500
        """
        from datasets import load_dataset
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
            
        self.root = root
        self.split = split
        assert split in ["test"]  # MATH500 only has a test split
        self.data = load_dataset('HuggingFaceH4/MATH-500', split=split)
        self.evaluation_api = evaluation_api
        self._task_description = 'You will solve a mathematical problem. Think step by step, showing your work, and provide a clear final answer.'
        
    def __getitem__(self, index):
        row = self.data[index]
        question = row["problem"]
        answer = row["answer"]
        return question, answer

    def __len__(self):
        return len(self.data)

    def get_task_description(self):
        return self._task_description

    def _get_instance_test_time_objective(self, question: str):
        """Define the loss function for test time optimization."""
        evaluation_instruction = "Below is a mathematical problem and a solution. Your job is to investigate the solution. Think step by step about whether the solution is correct or if there are any errors. Be critical and creative in your feedback."
        eval_fn = MultiChoiceTestTime(evaluation_instruction, engine=self.evaluation_api)
        
        def test_time_objective(instance: Variable):
            return eval_fn(question, instance)
        
        return test_time_objective
        
    def _get_instance_eval_fn(self, question_prompt: str, answer: str):
        """Define the evaluation function for scoring the response."""
        def eval_string_based(response_text, correct_answer):
            # This function extracts the final answer from the response and compares it with the correct answer
            # We'll need to clean up both answers to handle formatting variations
            
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
                
                return 1 if is_correct else 0
            return 0  # No answer extracted
        
        eval_fn = lambda response: eval_string_based(response.value, answer)
        return eval_fn