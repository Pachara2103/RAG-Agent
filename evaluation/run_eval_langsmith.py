import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from langsmith import Client, evaluate
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT, RAG_GROUNDEDNESS_PROMPT
from src.graph.workflow import get_workflow
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

client = Client()
dataset_name = "test-set-1"
experiment_name = f"experiment-{dataset_name}"


def run_full_graph(inputs: dict):

    agent = get_workflow()

    user_message = HumanMessage(content=inputs["question"])

    res = agent.invoke({"messages": [user_message]})
    answer = res["messages"][-1].content

    if isinstance(answer, list):
        answer = "".join(
            [item.get("text", "") for item in answer if item.get("type") == "text"]
        )

    return {"answer": answer, "context": res["context"]}


def correctness(inputs: dict, outputs: dict, reference_outputs: dict):
    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="google_genai:gemini-2.5-flash",
        feedback_key="correctness",
    )
    eval_result = evaluator(
        inputs=inputs,
        outputs=outputs,
        reference_outputs=reference_outputs,  # (จาก Dataset ส่วน outputs
    )
    return eval_result


def groundness(inputs: dict, outputs: dict):
    evaluator = create_llm_as_judge(
        prompt=RAG_GROUNDEDNESS_PROMPT,
        model="google_genai:gemini-2.5-flash",
        feedback_key="faithfulness",
    )

    eval_result = evaluator(
        context={"context": outputs["context"]},
        outputs={"output": outputs["answer"]},
    )
    return eval_result


experiment_results = client.evaluate(
    run_full_graph,
    data=dataset_name,
    evaluators=[correctness, groundness],
    experiment_prefix=experiment_name,
    # max_concurrency=2,
)
