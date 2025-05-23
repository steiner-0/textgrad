from textgrad.variable import Variable
from textgrad.autograd import Module
from textgrad.autograd import FormattedLLMCall
from textgrad.engine.deepseek_thinking_engine import ThinkingDeepseekEngine

class DeepseekThinkingEfficiencyLoss(Module):
    """A loss function that rewards token efficiency using DeepSeek's thinking capabilities."""
    
    def __init__(self, 
                 evaluation_api,
                 accuracy_weight=0.3,
                 token_weight=0.7):
        """
        Initialize the token efficiency loss specifically for DeepSeek models.
        
        Args:
            evaluation_api: The API to use for accuracy evaluation
            accuracy_weight: Weight for the accuracy component (0-1)
            token_weight: Weight for the token efficiency component (0-1)
        """
        super().__init__()
        self.evaluation_api = evaluation_api
        self.accuracy_weight = accuracy_weight
        self.token_weight = token_weight
        
        # Ensure we're using a DeepSeek API with thinking support
        assert isinstance(self.evaluation_api, ThinkingDeepseekEngine), "This loss function requires a ThinkingDeepseekEngine engine"
        
        # System prompt for efficiency evaluation
        self.system_prompt = Variable(
            "You are an evaluator that provides feedback on how to make prompts encourage more efficient reasoning.",
            requires_grad=False,
            role_description="system prompt for token efficiency evaluation"
        )
        
        # Format string for evaluation
        self.format_string = (
            "Below is a system prompt and the token usage information when the model was solving a problem.\n\n"
            "System Prompt: {system_prompt}\n\n"
            "Question: {question}\n\n"
            "Response: {response}\n\n"
            "The model used {reasoning_tokens} tokens for its reasoning process.\n\n"
            "Evaluate this system prompt on token efficiency. Provide specific feedback on how the "
            "system prompt could be improved to make the model think more efficiently while still reaching "
            "the correct answer. Focus on encouraging the model to eliminate unnecessary steps, reduce redundancy, "
            "and adopt a more concise thinking process."
        )
        
        self.fields = {
            "system_prompt": None, 
            "question": None, 
            "response": None, 
            "reasoning_tokens": None
        }
        
        self.formatted_llm_call = FormattedLLMCall(
            engine=self.evaluation_api,
            format_string=self.format_string,
            fields=self.fields,
            system_prompt=self.system_prompt
        )
    
    def forward(self, system_prompt, question, response, correct_answer=None):
        """
        Calculate the loss based on token efficiency and accuracy.
        
        Args:
            system_prompt: The system prompt used
            question: The question asked
            response: The model's response
            correct_answer: The correct answer (optional)
            
        Returns:
            A Variable containing feedback on how to improve token efficiency
        """
        # Get the token counts
        reasoning_tokens = self.evaluation_api.last_thinking_tokens
        # completion_tokens = self.evaluation_api.last_completion_tokens
        
        # Check if the answer is correct (if correct_answer is provided)
        # is_correct = False
        # if correct_answer is not None:
        #     is_correct = correct_answer.value in response.value
        
        # Prepare inputs for the formatter
        inputs = {
            "system_prompt": system_prompt,
            "question": question,
            "response": response,
            "reasoning_tokens": Variable(str(reasoning_tokens), requires_grad=False, role_description="reasoning token count")
        }
        
        # Get feedback on token efficiency
        efficiency_feedback = self.formatted_llm_call(
            inputs=inputs,
            response_role_description="feedback on token efficiency"
        )
        
        # Add information about correctness to the feedback if available
        # if correct_answer is not None:
        #     correctness_info = f"\n\nThe answer was {'correct' if is_correct else 'incorrect'}. "
        #     correctness_info += "The system prompt should be optimized to maintain accuracy while reducing token usage."
        #     efficiency_feedback.value += correctness_info
        
        return efficiency_feedback