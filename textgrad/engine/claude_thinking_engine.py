from textgrad.engine.anthropic import ChatAnthropic
import os
from anthropic import Anthropic
import tiktoken

class ThinkingChatAnthropic(ChatAnthropic):
    """Extended ChatAnthropic engine with thinking parameter support."""
    
    def __init__(self, model_string="claude-3-7-sonnet-20250219", 
                 system_prompt="You are a helpful, creative, and smart assistant.",
                 thinking_enabled=True,
                 thinking_budget=16000):
        super().__init__(model_string=model_string, system_prompt=system_prompt)
        self.thinking_enabled = thinking_enabled
        self.thinking_budget = thinking_budget
        self.last_thinking = None
    
    def generate(self, prompt, system_prompt=None, temperature=0, max_tokens=20000, top_p=0.99):
        """Override generate to include thinking parameter."""
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        
        # Check cache first
        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none
        
        # Configure thinking parameter
        thinking_config = None
        if self.thinking_enabled:
            thinking_config = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget
            }
            
        # Make API call
        response = self.client.messages.create(
            model=self.model_string,
            messages=[
                {"role": "user", "content": prompt}
            ],
            system=sys_prompt_arg,
            max_tokens=max_tokens,
            thinking=thinking_config
        )
        
        # Get response text
        response_text = response.content[1].text
        
        # Cache the result
        self._save_cache(sys_prompt_arg + prompt, response_text)
        
        # Store thinking in a property that can be accessed
        self.last_thinking = response.content[0].thinking 
        
        return response_text
    
    def get_last_thinking_tokens(self):
        """Get token count from the last thinking process."""
        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(self.last_thinking))