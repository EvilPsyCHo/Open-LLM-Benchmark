import os
import sys
import json
import asyncio
import tqdm
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


PROMPT = """### Instruct
Answer the question based the context, keep your response concise and direct

### Context
{context}

### Question
{question}
"""
DEFAULT_NEEDLE = "Jerry's favorite snack is Hot Cheetos."
DEFAULT_QUESTION = "What is Jerry's favorite snack?"
DEFAULT_ANSWER = "Hot Cheetos"
DEFAULT_HAYSTACK_FILE = Path(__file__).parent / "uber_2021.pdf"

class RetrievalTask:

    def __init__(self, 
                 test_llm: BaseLLM, 
                 evaluate_llm: BaseLLM, 
                 needle: str=None, 
                 ref_answer: str=None,
                 question: str=None,
                 haystack: str=None,
                 ctx_len_min=1000,
                 ctx_len_max=8000,
                 ctx_bins=8,
                 depth_min=10,
                 depth_max=100,
                 depth_bins=10,
                 ) -> None:
        self.test_llm = test_llm
        self.evaluator = ReferenceGrader(evaluate_llm)

        self.haystack = haystack if haystack else self.load_file(DEFAULT_HAYSTACK_FILE)
        self.haystack_tokens = self.test_llm.encode(self.haystack)
        print(f"Haystack context length: {len(self.haystack_tokens)}")
        
        self.needle = needle if needle else DEFAULT_NEEDLE
        self.needle_tokens = self.test_llm.encode(self.needle)
        self.ref_answer = ref_answer if ref_answer else DEFAULT_ANSWER
        self.question = question if question else DEFAULT_QUESTION
        # check whether the context length and depth are legal
        self.ctx_len_max = min(ctx_len_max, len(self.haystack_tokens))
        self.ctx_len_min = min(ctx_len_min, ctx_len_max)
        self.ctx_bins = ctx_bins
        self.depth_max = min(100, depth_max)
        self.depth_min = max(10, min(depth_min, depth_max))
        self.depth_bins = depth_bins
        print(self)
    
    def __repr__(self):
        info = "\n\n<<Task: Find needle in haystack>>\n"
        info += f"Model: {self.test_llm.model_name}\n"
        info += f"Context length from {self.ctx_len_min} to {self.ctx_len_max} with {self.ctx_bins} bins\n"
        info += f"Depth from {self.depth_min}% to {self.depth_max}% with {self.depth_bins} bins\n"
        return info
    
    def __str__(self) -> str:
        return self.__repr__()

        
    def load_file(self, path):
        data = SimpleDirectoryReader(
            input_files=[path]
        ).load_data()
        text = Document(text="\n\n".join([d.get_content() for d in data])).get_content()
        return text
    
    def insert_needle(self, context_length, depth_percent):
        context_tokens = self.haystack_tokens[:max(100, context_length-len(self.needle_tokens))]
        newline_tokens = self.test_llm.encode("\n")
        insertion_point = int(len(context_tokens) * (depth_percent / 100))
        new_context_tokens = context_tokens[:insertion_point] + newline_tokens + newline_tokens + self.needle_tokens + newline_tokens + newline_tokens + context_tokens[insertion_point:]
        new_context = self.test_llm.decode(new_context_tokens)
        return new_context

    def _run_one(self, context_length, depth_percent):

        context = self.insert_needle(context_length, depth_percent)
        prompt = PROMPT.format(context=context, question=self.question)
        generated_answer = self.test_llm.generate(messages=[{"role": "user", "content": prompt}])
        try:
            eval_result = self.evaluator.evaluate(self.question, self.ref_answer, generated_answer)
        except:
            eval_result = {"reasoning": "error", "score": -1}
        eval_result.update({
            "generated_answer": generated_answer,
            "reference_answer": self.ref_answer,
            "context_length": context_length,
            "depth_percent": depth_percent,
        })
        
        return eval_result

    async def _arun_one(self, context_length, depth_percent):

        context = self.insert_needle(context_length, depth_percent)
        prompt = PROMPT.format(context=context, question=self.question)
        generated_answer = self.test_llm.generate(messages=[{"role": "user", "content": prompt}])
        try:
            eval_result = self.evaluator.evaluate(self.question, self.ref_answer, generated_answer)
        except:
            eval_result = {"reasoning": "error", "score": -1}
        eval_result.update({
            "generated_answer": generated_answer,
            "reference_answer": self.ref_answer,
            "context_length": context_length,
            "depth_percent": depth_percent,
        })
        
        return eval_result
    
    async def _arun(self):
        context_length_bins = np.linspace(start=self.ctx_len_min, stop=self.ctx_len_max, num=self.ctx_bins, endpoint=True)
        context_length_bins = np.round(context_length_bins).astype(int)
        depth_length_bins = np.linspace(start=self.depth_min, stop=self.depth_max, num=self.depth_bins, endpoint=True)
        tasks = product(context_length_bins, depth_length_bins, repeat=1)
        tasks = [self._arun_one(*task) for task in tasks]
        res = []
        pbar = tqdm.tqdm(total=len(tasks))
        for f in asyncio.as_completed(tasks):
            res.append(await f)
            pbar.update()
        return res

    def arun(self):
        result = asyncio.run(self._arun())
        return result
    

    def run(self, save_dir=None):
        context_length_bins = np.linspace(start=self.ctx_len_min, stop=self.ctx_len_max, num=self.ctx_bins, endpoint=True)
        context_length_bins = np.round(context_length_bins).astype(int)
        depth_length_bins = np.linspace(start=self.depth_min, stop=self.depth_max, num=self.depth_bins, endpoint=True)
        tasks = list(product(context_length_bins, depth_length_bins, repeat=1))
        res = []
        for task in tqdm.tqdm(tasks, total=len(tasks)):
            res.append(self._run_one(*task))
        
        res_df = pd.DataFrame(res)
        mean_score = res_df.score.mean()
        if (res_df.score == -1).sum() > 0:
            res_df["score"] = res_df["score"].astype(float)
            res_df.loc[res_df.score == -1, "score"] = mean_score
        print(f"OverAll Score: {res_df.score.mean():.1f} / 5")

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
            
            pivot_table = pd.pivot_table(res_df, values='score', index=['depth_percent', 'context_length'], aggfunc='mean').reset_index() # This will aggregate
            pivot_table = pivot_table.pivot(index="depth_percent", columns="context_length", values="score") # This will turn into a proper pivot
            pivot_table.columns = [f"{i/1000:.1f}k" for i in pivot_table.columns.values]
            # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
            cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

            # Create the heatmap with better aesthetics
            f = plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
            sns.heatmap(
                pivot_table,
                # annot=True,
                fmt="g",
                cmap=cmap,
                cbar_kws={'label': 'score'},
                vmin=1,
                vmax=5,
            )

            # More aesthetics
            plt.title(f'Open-LLM-Benchmark@github\nModel: {self.test_llm.model_name}; Loader: {self.test_llm.model_loader}; Task: Retrieval needle in a haystack')  # Adds a title
            plt.xlabel('Token Limit')  # X-axis label
            plt.ylabel('Depth Percent')  # Y-axis label
            plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
            plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
            plt.tight_layout()  # Fits everything neatly into the figure area
            f.savefig(save_dir / "retrieval_task.png")
            res_df.to_csv(save_dir / "retrieval_task.csv", index=False)
        return res


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
    parser.add_argument("--needle", default=None, type=str, help="Needle test text, if not specified, the defualt needle is used")
    parser.add_argument("--question", default=None, type=str, help="Needle-relevant question, if not specified, the defualt question is used")
    parser.add_argument("--ref_answer", default=None, type=str, help="Reference answer for the question, if not specified, the defualt ref answer is used")
    parser.add_argument("--haystack", default=None, type=str, help="Context text, should be a pdf or text filepath, if not specified, the defualt haystack(about 300K tokens) is used")
    parser.add_argument("--ctx_len_min", default=1000, type=int, help="The minimum context length, should be shorter than the length of haystack.")
    parser.add_argument("--ctx_len_max", default=8000, type=int, help="The maximum context length, should be shorter than the length of haystack.")
    parser.add_argument("--ctx_bins", default=10, type=int, help="The number of segments for context testing length.")
    parser.add_argument("--depth_min", default=10, type=int, help="The minimum depth of probing, expressed as a percentage, should be a int between 0~100 and lower than depth_max")
    parser.add_argument("--depth_max", default=100, type=int, help="The minimum depth of probing, expressed as a percentage, should be a int between 0~100 and bigger than depth_min")
    parser.add_argument("--depth_bins", default=10, type=int, help="The minimum of segments for depth.")
    args = parser.parse_args()
    
    args.test_model_loader_params = json.loads(args.test_model_loader_params) if args.test_model_loader_params else {}
    args.eval_model_loader_params = json.loads(args.eval_model_loader_params) if args.test_model_loader_params else {}

    test_model = auto_llm_loader(args.test_model_loader, args.test_model, context_len=args.ctx_len_max+1000, **args.test_model_loader_params)
    eval_model = test_model if not args.eval_model else auto_llm_loader(args.eval_model_loader, args.eval_model, context_len=args.ctx_len_max+1000, **args.eval_model_loader_params)
    needle_task = RetrievalTask(
        test_llm=test_model,
        evaluate_llm=eval_model,
        needle=args.needle,
        ref_answer=args.ref_answer,
        question=args.question,
        haystack=args.haystack,
        ctx_len_min=args.ctx_len_min,
        ctx_len_max=args.ctx_len_max,
        ctx_bins=args.ctx_bins,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        depth_bins=args.depth_bins,
    )
    needle_task.run(args.save_dir)


if __name__ == "__main__":
    main()