from vllm import LLM, SamplingParams
import torch
from typing import List, Dict, Any, Optional
from .base import BaseLLM


class VLLM(BaseLLM):

    def __init__(self, model_id_or_path, context_len=4096, **kw):
        super().__init__(model_id_or_path, "vllm", context_len)
        loader_params = {
            "trust_remote_code": True, 
            "gpu_memory_utilization": 0.9, 
            # "max_seq_len": context_len, 
            # "max_context_len_to_capture": context_len,
            "tensor_parallel_size": torch.cuda.device_count()}
        loader_params.update(kw)
        self.model = LLM(model_id_or_path, **loader_params)
        self.tokenizer = self.model.get_tokenizer()
        self.is_llama3 = "<|eot_id|>" in self.tokenizer.vocab
    
    def generate(self, messages: List[Dict[str, str]], max_tokens: int=512, stop: list=None, **kw) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if stop:
            if isinstance(stop, str):
                stop = [stop]
        else:
            stop = []
        stop_token_ids = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")] if self.is_llama3 else [self.tokenizer.eos_token_id]
        sampling_params = SamplingParams(max_tokens=max_tokens, stop=stop, stop_token_ids=stop_token_ids, **kw)
        outputs = self.model.generate([prompt], sampling_params=sampling_params, use_tqdm=False)[0].outputs[0].text
        return outputs

    def encode(self, text: str) -> List[str]:
        return self.tokenizer(text=text, add_special_tokens=False).input_ids
    
    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=False)
