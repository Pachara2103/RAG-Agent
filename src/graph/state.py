from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from pydantic import BaseModel, Field


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    documents: list[str]
    filtered_documents: list[Document]
    context: str
    grade: str
    
class GradeAnswer(BaseModel):
    """Binary score for checking if the answer is grounded and addresses the question."""
    binary_score: str = Field(
        description="Answer is grounded in the facts and addresses the question, 'yes' or 'no'"
    )
    
class GradeDocument(BaseModel):
    """Binary score for checking if the document is related to the question."""
    binary_score: str = Field(
        description="Answer 'yes' if the document contains some keyword to the question, 'no' otherwise"
)