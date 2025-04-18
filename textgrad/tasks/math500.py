# textgrad/tasks/math500.py
import platformdirs
import re
from .base import Dataset

class MATH500(Dataset):
    def __init__(self, root: str=None, split: str="test", *args, **kwargs):
        """
        MATH500 dataset from HuggingFaceH4/MATH-500
        """
        from datasets import load_dataset
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
            
        self.root = root
        self.split = split
        assert split in ["train", "val", "test"]  # We'll simulate train/val splits
        
        # Load the dataset
        full_dataset = load_dataset('HuggingFaceH4/MATH-500', split="test")
        
        # For compatibility with the rest of the codebase, we'll split the test set
        # into train/val/test splits (since MATH500 only has a test split)
        import numpy as np
        np.random.seed(42)  # For reproducibility
        indices = np.random.permutation(len(full_dataset))
        
        if split == "train":
            # Use 60% for training
            split_indices = indices[:int(0.6 * len(indices))]
        elif split == "val":
            # Use 20% for validation
            split_indices = indices[int(0.6 * len(indices)):int(0.8 * len(indices))]
        else:  # test
            # Use 20% for testing
            split_indices = indices[int(0.8 * len(indices)):]
            
        self.data = full_dataset.select(split_indices)
        self._task_description = 'You will solve a mathematical problem. Think step by step, showing your work, and provide a clear final answer in the following format: "Answer: [your answer]".'
        
    def __getitem__(self, index):
        row = self.data[index]
        question = row["problem"]
        answer = row["answer"]
        return question, answer

    def __len__(self):
        return len(self.data)

    def get_task_description(self):
        return self._task_description