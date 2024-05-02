from open_llm_benchmark.llm import BaseLLM
import re


DEFAULT_SYSTEM_TEMPLATE = """
You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- a criteria
- a generated answer

Your job is to judge correctness of the generated answer based on the criteria.

Follow these guidelines for scoring:
- Thought first and then give the score.
- Your score has to be between 1 and 5, where 1 is the worst and 5 is the best.
- If the generated answer totally does not meet the criteria, you should give a score of 1.
- If the generated answer partially meets the criteria, you should give a score between 2 and 3.
- If the generated answer meets criteria perfectly, you should give a score between 4 and 5.

Example Response:
Thought: The generated answer has the exact same metrics as the reference answer, but it is not as concise.
Score: 4
"""

# DEFAULT_USER_TEMPLATE = """
# ## User Query
# {query}

# ## Criteria
# {criteria}

# ## Generated Answer
# {generated_answer}
# """

DEFAULT_USER_TEMPLATE = """## Criteria
{criteria}

## Generated Answer
{generated_answer}
"""

class CriteriaGrader:

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

    def evaluate(self, query, criteria, generated_answer):
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_TEMPLATE},
            {"role": "user", "content": DEFAULT_USER_TEMPLATE.format(query=query, criteria=criteria, generated_answer=generated_answer)}
        ]
        response = self.llm.generate(messages)
        return self.parse_output(response)
