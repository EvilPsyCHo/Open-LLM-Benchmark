import os
import sys
import json
import asyncio
from tqdm import tqdm
from itertools import combinations, product
import numpy as np
from pathlib import Path
from typing import List, Union, Optional, Dict
from llama_index.core import SimpleDirectoryReader, Document
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import json
import os
import glob

from open_llm_benchmark.evaluator import ReferenceGrader
from open_llm_benchmark.llm import BaseLLM, auto_llm_loader
from open_llm_benchmark.agent import ReActAgent
from open_llm_benchmark.evaluator import ReferenceGrader



class AgentTask:

    def __init__(self, 
                 agent,
                 evaluate_llm: BaseLLM, 
                 ) -> None:
        self.react_agent = agent
        self.evaluator = ReferenceGrader(evaluate_llm)
        print(self)
    
    def __repr__(self):
        info = "\n\n<<Task: Agent>>\n"
        return info
    
    def __str__(self) -> str:
        return self.__repr__()


    def _run_one(self, query, reference_answer, verbose=False):
        response = self.react_agent.run(query, verbose=verbose)
        num_rounds = len(response["trajectory"])
        if response["response"]:
            try:
                eval_result = self.evaluator.evaluate(query, reference_answer, response["response"])
            except:
                eval_result = {"reasoning": "Eval Error", "score": -1}
        else:
            eval_result = {"reasoning": "Agent Error", "score": 0}
        eval_result["generated_answer"] = response["response"]
        eval_result.update(
            {"query": query, "reference_answer": reference_answer, "generated_answer": response["response"], "num_rounds": num_rounds}
        )
        return eval_result
    

    def run(self, query_and_answer, save_dir=None, verbose=False):
        results = []
        for q, a in tqdm(query_and_answer, total=len(query_and_answer)):
            results.append(self._run_one(q, a, verbose))
        res_df = pd.DataFrame(results)
        res_df = res_df[res_df.score >=0].reset_index(drop=True)

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
            res_df.to_csv(save_dir / "agent_task.csv", index=False)
        
        mean_score = res_df.score.mean()
        mean_rounds = res_df.num_rounds.mean()
        print(f"Average Score {mean_score:.1f} / 5, Average Rounds {mean_rounds:.1f}")
        return results


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir")
    parser.add_argument("--test_model", type=str, help="Test model_id_or_path.")
    parser.add_argument("--test_model_loader", type=str, help="Optional hf, llamacpp.")
    parser.add_argument("--test_model_loader_params", default=None, type=str, help="model loader parameters in Json format")
    parser.add_argument("--eval_model", default=None, type=str, help="Eval model id or path. if not specified, the test model is used.")
    parser.add_argument("--eval_model_loader", default=None, type=str, help="Optional hf, llamacpp. if not specified, the test model is used.")
    parser.add_argument("--eval_model_loader_params", default=None, type=str, help="model loader parameters in Json format")
    args = parser.parse_args()
    
    args.test_model_loader_params = json.loads(args.test_model_loader_params) if args.test_model_loader_params else {}
    args.eval_model_loader_params = json.loads(args.eval_model_loader_params) if args.test_model_loader_params else {}

    ######## Fuctions ###########

    def multiply(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """Multiple two numbers and returns the result"""
        return a * b


    def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """Add two numbers and returns the result"""
        return a + b


    def subtract(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """Subtract two numbers and returns the result"""
        return a - b


    def divide(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """Divides two numbers and returns the result"""
        return a / b

    def query_weather(city: str, date: str) -> str:
        # Query the weather for a specified date (YYYY-MM-DD format) and city.
        if city == "New York":
            return f"The weather in New York on {date} is cloudy with a temperature of 23 degrees."
        elif city == "Bei Jing":
            return f"The weather in Bei Jing on {date} is rainy with a temperature of 21 degrees."
        else:
            return f"The weather in {city} on {date} is sunny with a temperature of 20 degrees."

    tools = [multiply, add, subtract, divide, query_weather]
    query_and_answer = [
        ("What is 188+133", "321"),
        ("What is 188 * 13", "2444"),
        ("What is (121 + 2) * 5 * (100-2)?", "60270"),
        ("What is (44 + 2) * 5 * (13-2)?", "2530"),
        ("What is (23 + 2) * (10-3)?", "175"),
        ("What is ((23 + 2) * (10-3)) * 3?", "525"),
        ("What is the weather like in New York on April 2nd, 2023?", "The weather in New York on 2023-04-02 is cloudy with a temperature of 23 degrees."),
        ("What is the weather like in Bei Jing on April 2nd, 2023?", "The weather in Bei Jing on 2023-04-02 is cloudy with a temperature of 21 degrees."),
        ("Today is May 2nd, 2024, how many degrees lower is the temperature in Paris compared to New York?", "The temperature in Paris is 3 degrees lower than in New York."),
        ("Today is April 2nd, 2023, how many degrees higher is the temperature in New York compared to Bei Jing?", "The temperature in New York is 2 degrees higher than in Bei Jing.")
    ]

    test_model = auto_llm_loader(args.test_model_loader, args.test_model, context_len=4000, **args.test_model_loader_params)
    eval_model = test_model if not args.eval_model else auto_llm_loader(args.eval_model_loader, args.eval_model, context_len=4000, **args.eval_model_loader_params)
    agent = ReActAgent(test_model, tools)
    
    agent_eval_task = AgentTask(
        agent=agent,
        evaluate_llm=eval_model,
    )
    agent_eval_task.run(query_and_answer, args.save_dir, verbose=True)


if __name__ == "__main__":
    main()
