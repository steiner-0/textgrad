# token_counting_loss.py - place in textgrad/loss/

import re
from typing import Dict, Union, List
import tiktoken
from textgrad.variable import Variable
from textgrad.engine import EngineLM
from textgrad.autograd import Module
from textgrad.autograd import FormattedLLMCall

class TokenEfficiencyLoss(Module):
    """A loss function that rewards token efficiency while maintaining accuracy."""
    
    def __init__(self, 
                 evaluation_api: EngineLM,
                 accuracy_weight: float = 0.3,
                 token_weight: float = 0.7,
                 encoder_name: str = "cl100k_base"):  # Default for gpt-4 models
        """
        Initialize the token efficiency loss.
        
        Args:
            evaluation_api: The API to use for accuracy evaluation
            accuracy_weight: Weight for the accuracy component (0-1)
            token_weight: Weight for the token efficiency component (0-1)
            encoder_name: The name of the tiktoken encoder to use
        """
        super().__init__()
        self.evaluation_api = evaluation_api
        self.accuracy_weight = accuracy_weight
        self.token_weight = token_weight
        self.encoder = tiktoken.get_encoding(encoder_name)
        
        # System prompt for efficiency evaluation
        self.system_prompt = Variable(
            "You are an evaluator that provides feedback on how to make text more concise and efficient.",
            requires_grad=False,
            role_description="system prompt for token efficiency evaluation"
        )
        
        # Format string for evaluation
        self.format_string = (
            "Below is a question and a model's response to it.\n\n"
            "Question: {question}\n\n"
            "Response: {response}\n\n"
            "Evaluate this response on token efficiency. The model uses too many tokens in its reasoning. "
            "Provide specific feedback on how the prompt could be improved to make the model's "
            "reasoning process more concise while still maintaining accuracy. "
            "Focus on eliminating unnecessary steps, redundant explanations, and verbose phrasing."
        )
        
        self.fields = {"question": None, "response": None}
        self.formatted_llm_call = FormattedLLMCall(
            engine=self.evaluation_api,
            format_string=self.format_string,
            fields=self.fields,
            system_prompt=self.system_prompt
        )
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text."""
        return len(self.encoder.encode(text))
    
    def extract_reasoning(self, response: str) -> str:
        """Extract the reasoning part from the response."""
        # Different models format their responses differently
        # This is a simple implementation - may need refinement for specific models
        
        # Try to find the 'Answer:' part which often marks the end of reasoning
        parts = response.split("Answer:")
        if len(parts) > 1:
            return parts[0].strip()
        
        # If no 'Answer:' part, return the whole response
        return response
    
    def forward(self, question: Variable, response: Variable, correct_answer: Variable) -> Variable:
        """
        Calculate the loss based on token efficiency and accuracy.
        
        Args:
            question: The question asked
            response: The model's response
            correct_answer: The correct answer
            
        Returns:
            A Variable containing feedback on how to improve token efficiency
        """
        # Extract the reasoning part
        reasoning = self.extract_reasoning(response.value)
        token_count = self.count_tokens(reasoning)
        
        # Prepare inputs for the formatter
        inputs = {
            "question": question,
            "response": response
        }
        
        # Get feedback on token efficiency
        efficiency_feedback = self.formatted_llm_call(
            inputs=inputs,
            response_role_description="feedback on token efficiency"
        )
        
        # Add token count information to the feedback
        token_info = f"The response used {token_count} tokens for reasoning. "
        efficiency_feedback.value = token_info + efficiency_feedback.value
        
        return efficiency_feedback