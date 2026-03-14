from langchain_core.prompts import ChatPromptTemplate
from src.graph.state import GradeDocument, GradeAnswer
from src.model import qwen

def get_document_grader_chain():

    system_propmt = """
    You are a grader assessing relevance of a retrieved document to a user question.    
    Instructions:
    1. If the document contains keywords or semantic meaning related to the question -> Score 'yes'.
    2. Even if the document says "NO" or "FORBIDDEN" regarding the question -> Score 'yes' (because it contains the relevant information).
    3. Score 'no' only if the document is talking about a completely different topic.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_propmt),
            ("human", "Document: {document}\n\nQuestion: {question}"),
        ]
    )

    chain = prompt | qwen.with_structured_output(GradeDocument)

    return chain


def get_answer_grade_chain():

    system_prompt = """
    You are a grader assessing whether an answer addresses the question.
    
    Instruction:
    1. If the answer resolves the question -> score 'yes'
    2. If the answer says "I don't know" or "Not found" -> score 'yes' (to accept the output and stop retrying)
    3. If the answer talks about something completely unrelated -> score 'no'
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Question: {question}\n\nAnswer: {answer}\n\n",  ),
        ]
    )

    chain = prompt | qwen.with_structured_output(GradeAnswer)

    return chain
