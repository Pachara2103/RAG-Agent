from src.graph.state import AgentState
from langchain_core.messages import HumanMessage

def should_continue(state: AgentState):
    """ฟังก์ชันตรวจสอบว่า Agent ต้องการเรียก Tool หรือไม่"""
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"
    return "end"

def should_regenerate(state: AgentState):
    """ฟังก์ชันตรวจสอบว่าควร regenerate answer หรือไม่"""
    if state["grade"]=='no':
        state['messages'].append(HumanMessage(content="คำตอบของคุณไม่ตรงกับเอกสารอ้างอิง กรุณาตอบใหม่จากข้อมูลในเอกสาร หรือหากข้อมูลไม่เพียงพอให้ใช้ 'search' tool "))
        return 'regenerate'
    return 'answer'