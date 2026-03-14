from langchain_core.prompts import ChatPromptTemplate
from src.model import qwen

def get_rewriter_chain():

    system_prompt = """ 
    คุณคือ 'ระบบเเก้ไขคำผิด'
    Instructions:
    1. เเก้ใขคำในประโยคที่มีคำที่เขียนผิด เเล้ว output ประโยคเดิมพร้อมคำที่ถูกต้อง
    2. คงภาษาต้นฉบับไว้ดังเดิม "ห้ามแปลภาษา"
    """
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Input: {question}\nOutput:"),
        ]
    )

    chain = prompt | qwen

    return chain
