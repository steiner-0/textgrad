from textgrad.variable import Variable
from textgrad.autograd import Module
from textgrad.autograd import FormattedLLMCall
from textgrad.engine.claude_thinking_engine import ThinkingChatAnthropic

class ClaudeThinkingEfficiencyLoss(Module):
    """A loss function that rewards token efficiency using Claude's thinking parameter."""
    
    def __init__(self, 
                 evaluation_api,
                 accuracy_weight=0.3,
                 token_weight=0.7):
        """
        Initialize the token efficiency loss specifically for Claude.
        
        Args:
            evaluation_api: The API to use for accuracy evaluation
            accuracy_weight: Weight for the accuracy component (0-1)
            token_weight: Weight for the token efficiency component (0-1)
        """
        super().__init__()
        self.evaluation_api = evaluation_api
        self.accuracy_weight = accuracy_weight
        self.token_weight = token_weight
        
        # Ensure we're using a Claude API with thinking support
        assert isinstance(self.evaluation_api, ThinkingChatAnthropic), "This loss function requires a ThinkingChatAnthropic engine"
        
        # System prompt for efficiency evaluation
        self.system_prompt = Variable(
            "You are an evaluator that provides feedback on how to make prompts encourage more efficient reasoning.",
            requires_grad=False,
            role_description="system prompt for token efficiency evaluation"
        )
        
        # Format string for evaluation
        self.format_string = (
            "Below is a system prompt and the thinking trace it produced when the model was solving a problem.\n\n"
            "System Prompt: {system_prompt}\n\n"
            "Question: {question}\n\n"
            "Thinking Trace:\n{thinking_trace}\n\n"
            "The model used {token_count} tokens for its thinking process.\n\n"
            "Evaluate this system prompt on token efficiency. Provide specific feedback on how the "
            "system prompt could be improved to make the model think more efficiently while still reaching "
            "the correct answer. Focus on encouraging the model to eliminate unnecessary steps, reduce redundancy, "
            "and adopt a more concise thinking process."
        )
        
        self.fields = {"system_prompt": None, "question": None, "thinking_trace": None, "token_count": None}
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
        # Get the thinking trace and token count
        thinking_trace = self.evaluation_api.last_thinking.text if hasattr(self.evaluation_api.last_thinking, 'text') else ""
        token_count = self.evaluation_api.get_last_thinking_tokens()
        
        # Prepare inputs for the formatter
        inputs = {
            "system_prompt": system_prompt,
            "question": question,
            "thinking_trace": Variable(thinking_trace, requires_grad=False, role_description="model's thinking process"),
            "token_count": Variable(str(token_count), requires_grad=False, role_description="token count")
        }
        
        # Get feedback on token efficiency
        efficiency_feedback = self.formatted_llm_call(
            inputs=inputs,
            response_role_description="feedback on token efficiency"
        )
        
        return efficiency_feedback