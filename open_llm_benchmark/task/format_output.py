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

from open_llm_benchmark.evaluator import CriteriaGrader
from open_llm_benchmark.llm import BaseLLM, auto_llm_loader


JSON_CRETERIA = """The answer is in legal json format."""
CODE_CRETERIA = """The answer is a complete python code, The code is wrapped with <python>...</python>."""
NUMBER_CRETERIA = """The answer is a number without any other characters."""

python_suffix = "\nJust answer the python code without any explanation. Make sure the python code is wrapped with <python>\n...</python>."

JSON_EXAMPLES = [
"""Extract the named entity from the paragraph bellow, the named entity include person name, address and time.

In the quaint town of Willowbrook, nestled amidst rolling hills and ancient oaks, lived three peculiar individuals: Evelyn Hawthorne, the reclusive librarian; Henry Sinclair, the eccentric clockmaker; and Amelia Grey, the enigmatic artist.
One chilly evening, as the clock tower chimed midnight, a mysterious letter arrived at Evelyn’s doorstep. The parchment bore no return address, only the words: “Meet me at the Moonlit Café, by the old oak tree, tomorrow at dusk.”

Response with JSON format, filling the entities into the corresponding list, if not keep the list empty. Just response the json result without any other explanation:
{
    "person name": [],
    "address": [],
    "time": [],
}""",


"""Perform the following arithmetic operations and provide the results in JSON format: Add 45 and 15, subtract 10 from 50, multiply 9 by 8, and divide 100 by 4.""",

"""From the weather description, "Today, New York enjoys clear skies with a high of 75°F and a low of 59°F. There is a 10% chance of rain and wind speeds are around 5 mph from the northeast", extract the temperature, precipitation chance, and wind speed, and format these details into JSON.""",

"""Given the movie descriptions below, classify each movie into genres and output the results in JSON format.

The space crew battles an alien creature that hunts them one by one.
A couple's relationship is tested when uninvited guests arrive at their home.""",

'''Create a summary for a fictional basketball game in which the Los Angeles Lakers beat the Miami Heat 102-97. LeBron James scored 30 points and had 10 assists. Output the game's final score, top player, and his statistics in JSON format.''',

'''From the recipe description, "To make a classic apple pie, you need apples, sugar, butter, and cinnamon. The pie should be baked at 350°F for about 45 minutes", extract the ingredients and the baking temperature and time, then output these in JSON format.'''
]

NUMBER_EXAMPLE = [

'''Jim's brother is 17 years older and jim is 3 years older than his brother. How old is jim now? Just answer the age number without any explanation.''',
'''A baker made 15 loaves of bread. If he sold 9 loaves, how many loaves does he have left? Just answer the age number without any explanation.''',
'''A train travels at a speed of 60 kilometers per hour. How far does it travel in 4 hours? Just answer the age number without any explanation.''',
'''If you buy 3 apples and each apple costs $2, how much do you spend in total? Just answer the age number without any explanation.''',
'''There are 24 students in a classroom. If 1/3 of them are absent, how many students are present?''',
'''A rectangle has a length of 8 meters and a width of 3 meters. What is the area of the rectangle?''',
]

CODE_EXAMPLE = [
'''How to calculate the edit distance between two strings?  Write a python code to calcualte the edit distance between 'bananan' and 'ana'.''' + python_suffix,

'''Write a Python code to plot the first 10 numbers in the Fibonacci sequence using matplotlib.''' + python_suffix,

'''Write a Python code to check if the numbers from 1 to 20 are prime or not.''' + python_suffix,

'''Write a Python code to sort a list of tuples by the second item in each tuple.''' + python_suffix,

'''Write a Python code to generate a random password of length 8 including uppercase, lowercase, numbers, and symbols.''' + python_suffix,

'''Write a Python code to convert a list of temperatures from Celsius to Fahrenheit.''' + python_suffix,
]


class FormatOutputTask:

    def __init__(self, 
                 test_llm: BaseLLM, 
                 evaluate_llm: BaseLLM, 
                 ) -> None:
        self.test_llm = test_llm
        self.evaluator = CriteriaGrader(evaluate_llm)
        print(self)
    
    def __repr__(self):
        info = "\n\n<<Task: Format Output>>\n"
        return info
    
    def __str__(self) -> str:
        return self.__repr__()


    def _run_one(self, query, criteria):
        generated_answer = self.test_llm.generate(messages=[{"role": "user", "content": query}], max_tokens=1024)
        try:
            eval_result = self.evaluator.evaluate(query, criteria, generated_answer)
        except:
            eval_result = {"reasoning": "Eval Error", "score": -1}
        eval_result["generated_answer"] = generated_answer
        return eval_result
    

    def run(self, save_dir=None):
        results = []
        for json_example in tqdm(JSON_EXAMPLES, total=len(JSON_EXAMPLES)):
            res = self._run_one(json_example, JSON_CRETERIA)
            res["type"] = "json"
            results.append(res)
        for num_example in tqdm(NUMBER_EXAMPLE, total=len(JSON_EXAMPLES)):
            res = self._run_one(num_example, NUMBER_CRETERIA)
            res["type"] = "number"
            results.append(res)
        for code_example in tqdm(CODE_EXAMPLE, total=len(JSON_EXAMPLES)):
            res = self._run_one(code_example, CODE_CRETERIA)
            res["type"] = "code"
            results.append(res)
        
        res_df = pd.DataFrame(results)
        res_df = res_df[res_df.score != -1].reset_index(drop=False)
        for t in res_df["type"].unique():
            ms = res_df[res_df["type"] == t].score.mean()
            print(f"Type {t:<10} score : {ms:.1f} / 5")

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
            res_df.to_csv(save_dir / "format_output_task.csv", index=False)
            f = plt.figure(figsize=(17.5, 8))
            title=f'Open-LLM-Benchmark@github\nTask: Format Output score from 1 to 5, higher is better\nModel: {self.test_llm.model_name}; Loader: {self.test_llm.model_loader}'
            sns.barplot(data=res_df, x='type', y='score', hue='type', dodge=False, capsize=0.1, errorbar=None, palette="viridis", legend=False)
            plt.title(title)
            plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
            plt.tight_layout()  # Fits everything neatly into the figure area
            plt.savefig(save_dir / "format_output_task.png")
            res_df.to_csv("format_output_task.csv", index=False)
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
    args = parser.parse_args()
    
    args.test_model_loader_params = json.loads(args.test_model_loader_params) if args.test_model_loader_params else {}
    args.eval_model_loader_params = json.loads(args.eval_model_loader_params) if args.test_model_loader_params else {}

    test_model = auto_llm_loader(args.test_model_loader, args.test_model, context_len=4096, **args.test_model_loader_params)
    eval_model = test_model if not args.eval_model else auto_llm_loader(args.eval_model_loader, args.eval_model, context_len=4000, **args.eval_model_loader_params)
    format_output_task = FormatOutputTask(
        test_llm=test_model,
        evaluate_llm=eval_model,
    )
    format_output_task.run(args.save_dir)


if __name__ == "__main__":
    main()