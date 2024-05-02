import torch
import gc
import logging
from typing import List, Dict, Any, Optional
import torch
import transformers
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
transformers.logging.set_verbosity(transformers.logging.ERROR)

from open_llm_benchmark.llm.base import BaseLLM


class WordsStoppingCriteria(StoppingCriteria):
    def __init__(self, stops = []):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for seq in input_ids:
            for stop in self.stops:
                if len(seq) >= len(stop) and torch.all((stop == seq[-len(stop):])).item():
                    return True
        return False


class HuggingFaceLLM(BaseLLM):

    def __init__(self, model_id_or_path, context_len=4096, **kw):
        super().__init__(model_id_or_path, "huggingface", context_len)
        loader_params = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        loader_params.update(kw)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_id_or_path, **loader_params)
        
        # Add <|eot_id|> as EOS token to Llama3
        if "<|eot_id|>" in self.tokenizer.vocab:
            terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
            self.eos_token_id = terminators
        else:
            self.eos_token_id = [self.tokenizer.eos_token_id]

    
    def generate(self, messages: List[Dict[str, str]], max_tokens: int=512, stop: list=None, **kw) -> str:
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        if stop:
            if isinstance(stop, str):
                stop = [stop]
        else:
            stop = []
        stopping_criteria = StoppingCriteriaList([WordsStoppingCriteria([self.tokenizer(s, add_special_tokens=False, return_tensors="pt").input_ids[0].to(self.model.device) for s in stop])])
        
        outputs = self.model.generate(input_ids, 
                                      max_new_tokens=max_tokens,
                                      eos_token_id=self.eos_token_id,
                                      stopping_criteria=stopping_criteria, **kw)
        response = outputs[0][input_ids.shape[-1]:]
        torch.cuda.empty_cache()
        gc.collect()
        return self.tokenizer.decode(response, skip_special_tokens=True)

    def encode(self, text: str) -> List[str]:
        return self.tokenizer(text=text, add_special_tokens=False).input_ids
    
    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=False)
