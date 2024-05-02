from openai import OpenAI
import tiktoken

from .base import BaseLLM


class OpenAILLM(BaseLLM):

    def __init__(self, model_id_or_path: str, model_loader: str="openai", context_len: int=None):
        super().__init__(model_id_or_path, model_loader, context_len)
        self.model_name = model_id_or_path
        self.client = OpenAI()
        self.tokenizer = tiktoken.encoding_for_model(model_id_or_path)
    
    def generate(self, messages, max_tokens=512, stop=None, **kw):
        if stop:
            if isinstance(stop, str):
                stop = [stop]
        else:
            stop = []
        res = self.client.chat.completions.create(messages=messages, model=self.model_name, max_tokens=max_tokens, stop=stop, **kw)
        return res.choices[0].message.content
    
    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)
