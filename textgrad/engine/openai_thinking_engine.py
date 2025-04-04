from textgrad.engine.openai import ChatOpenAI
import os
import tiktoken
from openai import OpenAI

class ThinkingO1Engine(ChatOpenAI):
    """Extended ChatOpenAI engine with OpenAI o1 reasoning capabilities."""
    
    def __init__(self, model_string="o1-2024-12-17", 
                 system_prompt="You are a helpful, creative, and smart assistant.",
                 reasoning_effort="medium",
                 api_key=None):
        """
        Initialize an OpenAI o1 engine with reasoning capabilities.
        
        Args:
            model_string: The OpenAI o1 model to use, defaults to "o1-2024-12-17"
            system_prompt: The system prompt to use
            reasoning_effort: Reasoning effort level ("low", "medium", "high")
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY environment variable)
        """
        # Handle API key if provided
        if api_key is not None:
            original_openai_key = os.getenv("OPENAI_API_KEY")
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Initialize parent class
        super().__init__(model_string=model_string, system_prompt=system_prompt)
        
        # Restore original OpenAI key if needed
        if api_key is not None and original_openai_key:
            os.environ["OPENAI_API_KEY"] = original_openai_key
        
        # Set reasoning parameters
        self.reasoning_effort = reasoning_effort
        self.last_reasoning_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.last_reasoning = None
    
    def generate(self, prompt, system_prompt=None, temperature=0, max_tokens=2000, top_p=0.99):
        """Override generate to include reasoning_effort parameter and track token usage."""
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        
        # Check cache first
        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none
        
        # Make API call with reasoning parameter
        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": prompt}
            ],
            reasoning_effort=self.reasoning_effort
        )
        
        # Get response text
        response_text = response.choices[0].message.content
        
        # Cache the result
        self._save_cache(sys_prompt_arg + prompt, response_text)
        
        # Store usage data
        usage = response.usage.model_dump()
        if 'completion_tokens_details' in usage and 'reasoning_tokens' in usage['completion_tokens_details']:
            self.last_reasoning_tokens = usage['completion_tokens_details']['reasoning_tokens']
        else:
            self.last_reasoning_tokens = 0
        
        self.last_completion_tokens = usage.get('completion_tokens', 0)
        self.last_total_tokens = usage.get('total_tokens', 0)
        
        # Try to extract reasoning trace if available in the response structure
        self.last_reasoning = "Reasoning process not directly accessible"
        if hasattr(response, 'reasoning') and response.reasoning:
            self.last_reasoning = response.reasoning
        
        return response_text
    
    def get_last_reasoning_tokens(self):
        """Get token count from the last reasoning process."""
        return self.last_reasoning_tokens
    
    def set_reasoning_effort(self, effort_level):
        """Set the reasoning effort level."""
        if effort_level not in ["low", "medium", "high"]:
            raise ValueError("reasoning_effort must be one of: 'low', 'medium', 'high'")
        self.reasoning_effort = effort_level