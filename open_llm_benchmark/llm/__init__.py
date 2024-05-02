try: 
    from .hf_loader import HuggingFaceLLM
except:
    print("Install transformers to access HuggingFaceLLM")

try: 
    from .llama_cpp_loader import LlamaCppLLM
except:
    print("Install llama-cpp-pyhon to access LlamaCppLLM")

try:
    from .vllm_loader import VLLM
except:
    print("Install vllm to access VLLM")

try:
    from .openai_loader import OpenAILLM
except:
    print("Install openai and tiktoken to access OpenAILLM")

from .base import BaseLLM


def auto_llm_loader(model_loader, model_id_or_path, context_len=4096, **kw):
    if model_loader in ["llama_cpp", "llamacpp"]:
        return LlamaCppLLM(model_id_or_path, context_len, **kw)
    elif model_loader in ["hf", "huggingface", "transformers"]:
        return HuggingFaceLLM(model_id_or_path, context_len, **kw)
    elif model_loader in ["vllm", "vLLM"]:
        return VLLM(model_id_or_path, context_len, **kw)
    elif model_loader in ["openai", "OpenAI"]:
        return OpenAILLM(model_id_or_path, context_len, **kw)
    else:
        raise NotImplementedError
