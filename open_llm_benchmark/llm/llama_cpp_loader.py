import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from llama_cpp import Llama

from open_llm_benchmark.llm.base import BaseLLM


LlamacppLoaderParams = {
    "n_gpu_layers": -1,
    "verbose": False
}


class LlamaCppLLM(BaseLLM):

    def __init__(self, model_id_or_path, context_len=4096, **kw):
        super().__init__(model_id_or_path, "llamacpp", context_len)

        loader_params = {
            "n_gpu_layers": -1,
            "verbose": False,
            "n_ctx": context_len,
            }
        loader_params.update(kw)
        self.model = Llama(model_id_or_path, **loader_params)
        self.is_llama3 = (128009 == self.model.token_eos()) or (128001 == self.model.token_eos())
    
    def generate(self, messages: List[Dict[str, str]], max_tokens: int=512, stop: list=None, **kw) -> str:
        if stop:
            if isinstance(stop, str):
                stop = [stop]
        else:
            stop = []
        if self.is_llama3:
            stop.append("<|eot_id|>")
        response = self.model.create_chat_completion(messages, max_tokens=max_tokens, stop=stop, **kw)
        return response["choices"][0]["message"]["content"]

    def encode(self, text, add_bos=False, special=False):
        return self.model.tokenizer_.encode(text, add_bos=add_bos, special=special)

    def decode(self, ids):
        return self.model.tokenizer_.decode(ids)
