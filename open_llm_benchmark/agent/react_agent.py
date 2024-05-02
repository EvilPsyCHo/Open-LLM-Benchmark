from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.agent import ReActAgent
from typing import Dict, List, Any, Sequence, Callable, Literal
import re
import json
from rich import print


REACT_PROMPT = '''You are an AI agent capable of using a variety of tools to answer question.

You have access to the following tools:
{{tools_desc}}

### Reponse Format

Response using the follow format:

Thought: think step by steps, how to solve the question, put your thought process here.
Action:
```json
{
    "tool": $TOOL_NAME,
    "args": $TOOL_ARGS
}
```
Observation: tool output
...(this Thought/Action/Observation can repeat N times until you get enough information to answer the question)
Thought: I now know the final answer 
Final Answer: make sure output the final answer here

$TOOL_NAME is the name of the tool. $TOOL_ARGS is a dictionary input matching the requirement of the tool.
'''

# copy from llama-index
def get_react_tool_descriptions(tools: Sequence[BaseTool]) -> List[str]:
    tool_descs = []
    for tool in tools:
        tool_desc = (
            f"> Tool Name: {tool.metadata.name}\n"
            f"Tool Description: {tool.metadata.description}\n"
            f"Tool Args: {tool.metadata.fn_schema_str}\n"
        )
        tool_descs.append(tool_desc)
    return tool_descs


class ReActState:
    def __init__(self, step: Literal["Question", "ReAct", "Answer", "Observation", "Error"], content: str, parsed_content: Any=None, successful: bool=True):
        self.step = step
        self.successful = successful
        self.content = content
        self.parsed_content = parsed_content
    
    @property
    def rich_message(self):
        return f"[green3]<<{self.step}>>[/green3]\n{self.content}"
    
    def __repr__(self) -> str:
        return f"<<{self.step}>>\n{self.content}"
    
    @property
    def question(self):
        if self.step == "Question":
            return self.content
        else:
            raise ValueError(f"Step {self.step} doesn't contain question.")
    
    @property
    def tool_name(self):
        if self.step == "ReAct":
            return self.parsed_content["action"]["tool"]
        else:
            raise ValueError(f"Step {self.step} doesn't contain tool_name.")
    
    @property
    def tool_args(self):
        if self.step == "ReAct":
            return self.parsed_content["action"]["args"]
        else:
            raise ValueError(f"Step {self.step} doesn't contain tool_args.")
    
    @property
    def thought(self):
        if self.step == "ReAct" or self.step == "Answer":
            return self.parsed_content["thought"]
        else:
            raise ValueError(f"Step {self.step} doesn't contain thought.")
    
    @property
    def answer(self):
        if self.step == "Answer":
            return self.parsed_content["answer"]
        else:
            raise ValueError(f"Step {self.step} doesn't contain answer.")


class ReActOutputParser:
    """ReAct Output parser."""

    def parse(self, output: str):
        if "Final Answer:" in output:
            parsed_content = self.extract_final_answer(output)
            return ReActState(step="Answer", content=output, parsed_content=parsed_content)
        
        elif "Action:" in output:
            parsed_content = self.extract_action(output)
            return ReActState(step="ReAct", content=output, parsed_content=parsed_content)
        
        else:
            raise ValueError("Model reponse containt neither 'Action:' nor 'Final Answer'")
    
    def extract_action(self, input_text: str):
        pattern = r"Thought:([\s\S]*?)Action:([\s\S]*)"
        match = re.search(pattern, input_text)
        if not match:
            raise ValueError(f"Could not extract Thought/Action from input text: {input_text}")

        thought = match.group(1).strip()
        action = match.group(2).strip()

        json_block_pattern = "```json([\s\S]*?)(```|$)"
        match = re.search(json_block_pattern, action)
        if not match:
            raise ValueError(f"Could not extract Action JSON block from input text: {action}")
        action_json = match.group(1).strip()
        import dirtyjson
        action_json = dirtyjson.loads(action_json)
        
        # import json
        # raw_message = json.dumps(action_json, indent=4)
        return {"thought": thought, "action": action_json}


    def extract_final_answer(self, response):
        pattern = r"\s*Thought:([\s\S]*?)Final Answer:([\s\S]*?)(?:$)"
        match = re.search(pattern, response, re.DOTALL)

        pattern2 = r"Final Answer:([\s\S]*?)(?:$)"
        match2 = re.search(pattern2, response, re.DOTALL)

        if match:
            thought = match.group(1).strip()
            answer = match.group(2).strip()
        elif match2:
            thought = "None"
            answer = match2.group(1)
        else:
            raise ValueError(
                f"Could not extract final answer from input text: {response}"
            )

        
        # raw_message = f"Thought: {thought}\nFinal Answer: {answer}"
        return {"thought": thought, "answer": answer}


class ReActAgent:

    def __init__(self, llm, tools: List[Callable], system_prompt: str=None, max_rounds: int=10, output_parser=None):
        self.llm = llm
        tools = [FunctionTool.from_defaults(fn=t) for t in tools]
        self.tools = {f.metadata.get_name(): f for f in tools}
        self.system_prompt = system_prompt or REACT_PROMPT
        self.max_rounds = max_rounds
        self.output_parser = output_parser or ReActOutputParser()

    def format_system_prompt(self):
        tools_name = ", ".join([tool.metadata.get_name() for tool in self.tools.values()])
        tools_desc = "\n".join(get_react_tool_descriptions(self.tools.values()))
        return self.system_prompt.replace("{{tools_desc}}", tools_desc).replace("{{tools_name}}", tools_name)
        
    def run(self, query, num_rounds=None, verbose=False):
        # Initialization
        num_rounds = min(num_rounds, self.max_rounds) if num_rounds else self.max_rounds
        messages = [
            {"role": "system", "content": self.format_system_prompt()},
            {"role": "user", "content": f"{query}"}
            ]
        step_state = ReActState(step="Question", content=query)
        trajectory = [step_state]
        if verbose:
            print(step_state.rich_message)
        for round in range(1, 1+num_rounds):
            try:
                model_output = self.llm.generate(messages, stop=["Observation:"], max_tokens=512)
                messages.append({"role": "assistant", "content": model_output})
                step_state = self.output_parser.parse(model_output)
                trajectory.append(step_state)
                if verbose:
                    print(step_state.rich_message)
                
                if step_state.step == "ReAct":
                    func = self.tools[step_state.tool_name]
                    args = step_state.tool_args
                    observation = func(**args).content
                    step_state = ReActState(step="Observation", content=observation)
                    trajectory.append(step_state)
                    if verbose:
                        print(step_state.rich_message)
                    messages.append({"role": "user", "content": f"Observation: {observation}"})
                elif step_state.step == "Answer":
                    return {"response": step_state.answer, "trajectory": trajectory, "error": None}
                else:
                    if verbose: print("ERROR: Contains neither 'Action' nor 'Final Answer' in response.")
                    return {"response": None, "trajectory": trajectory, "error": "Contains neither 'Action' nor 'Final Answer' in response."}
            except Exception as e:
                if verbose: print(f"ERROR: {str(e)}")
                return {"response": None, "trajectory": trajectory, "error": str(e)}
        if verbose: print("ERROR: Exceeded limitation of max rounds.")
        return {"response": None, "trajectory": trajectory, "error": "Exceeded limitation of max rounds."}
