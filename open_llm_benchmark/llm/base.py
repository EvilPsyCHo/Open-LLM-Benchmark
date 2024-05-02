from abc import ABC, abstractclassmethod
from typing import Any, Union
from pathlib import Path


class BaseLLM(ABC):

    def __init__(self, model_id_or_path: str, model_loader: str, context_len: int):
        self.model_name = Path(model_id_or_path).stem
        self.model_loader = model_loader
        self.context_len = context_len
    @abstractclassmethod
    def generate(self, *args: Any, **kwds: Any) -> Any: ...

    @abstractclassmethod
    def encode(self, *args: Any, **kwds: Any) -> Any: ...

    @abstractclassmethod
    def decode(self, *args: Any, **kwds: Any) -> Any: ...
