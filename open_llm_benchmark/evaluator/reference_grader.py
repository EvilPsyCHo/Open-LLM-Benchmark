from open_llm_benchmark.llm import BaseLLM
import re

# modified from llama_index ReAct prompt.
# Output a single score that represents a holistic evaluation.
# You must return your response in a line with only the score.
# Do not return answers in any other format.
# On a separate line provide your reasoning for the score as well.
DEFAULT_SYSTEM_TEMPLATE = """
You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- a user query, and
- a generated answer

You may also be given a reference answer to use for reference in your evaluation.
Your job is to judge the relevance and correctness of the generated answer.

Follow these guidelines for scoring:
- Thought first and then give the score.
- Your score has to be between 1 and 5, where 1 is the worst and 5 is the best.
- If the generated answer is not relevant to the user query, \
you should give a score of 1.
- If the generated answer is relevant but contains mistakes, \
you should give a score between 2 and 3.
- If the generated answer is relevant and fully correct, \
you should give a score between 4 and 5.

Example Response:
Thought: The generated answer has the exact same metrics as the reference answer, but it is not as concise.
Score: 4
"""

DEFAULT_USER_TEMPLATE = """
## User Query
{query}

## Reference Answer
{reference_answer}

## Generated Answer
{generated_answer}
"""

class ReferenceGrader:

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def parse_output(self, response):
        pattern = r"\s*Thought:(.*?)\s*Score:(.*?)(?:$)"

        match = re.search(pattern, response, re.DOTALL)
        if not match:
            raise ValueError(
                f"Could not extract final answer from input text: {response}"
            )

        thought = match.group(1).strip()
        score = int(match.group(2).strip())
        return {"reasoning": thought, "score": score}

    def evaluate(self, query, reference_answer, generated_answer):
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_TEMPLATE},
            {"role": "user", "content": DEFAULT_USER_TEMPLATE.format(query=query, reference_answer=reference_answer, generated_answer=generated_answer)}
        ]
        response = self.llm.generate(messages)
        return self.parse_output(response)
