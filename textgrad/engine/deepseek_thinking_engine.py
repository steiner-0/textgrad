from textgrad.engine.openai import ChatOpenAI
import os
import tiktoken
from openai import OpenAI

class ThinkingDeepseekEngine(ChatOpenAI):
    """Extended ChatOpenAI engine with DeepSeek thinking parameter support."""
    
    def __init__(self, model_string="deepseek-reasoner", 
                 system_prompt="You are a helpful, creative, and smart assistant.",
                 api_key=None,
                 base_url="https://api.deepseek.com/v1"):
        """
        Initialize a DeepSeek engine with thinking capabilities.
        
        Args:
            model_string: The DeepSeek model to use, defaults to "deepseek-reasoner"
            system_prompt: The system prompt to use
            thinking_enabled: Whether to enable thinking mode
            thinking_budget: The token budget for thinking
            api_key: DeepSeek API key (if None, will use DEEPSEEK_API_KEY environment variable)
            base_url: DeepSeek API base URL
        """
        # Handle API key
        if api_key is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if api_key is None:
                raise ValueError("Please set the DEEPSEEK_API_KEY environment variable or provide api_key parameter")
        
        # Initialize parent class
        super().__init__(model_string=model_string, system_prompt=system_prompt)
        
        # Create custom client
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Set thinking parameters
        self.last_thinking = None
        self.last_thinking_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
    
    def generate(self, prompt, system_prompt=None, temperature=0, max_tokens=2000, top_p=0.99):
        """Override generate to include thinking parameter and track token usage."""
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        
        # Check cache first
        # cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        # if cache_or_none is not None:
        #     return cache_or_none
        
        # Make API call, ensure correct parameters for DeepSeek
        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Get response text
        response_text = response.choices[0].message.content
        
        # Cache the result
        # self._save_cache(sys_prompt_arg + prompt, response_text)
        
        # Store usage data - handling DeepSeek-specific structure
        usage = response.usage.model_dump()
        self.last_thinking_tokens = usage['completion_tokens_details']['reasoning_tokens']
        
        self.last_completion_tokens = usage.get('completion_tokens')
        self.last_total_tokens = usage.get('total_tokens')
        
        # Get thinking trace if available
        self.last_thinking = "Reasoning process not available in model output"
        
        return response_text
    
    def get_last_thinking_tokens(self):
        """Get token count from the last thinking process."""
        return self.last_thinking_tokens