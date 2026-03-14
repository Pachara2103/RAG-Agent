from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.model import gemini
from src.tools import search

def get_agent_chain():

    system_propmt = """
      คุณคือผู้ช่วยตอบคำถาม ให้ใช้เอกสารที่ได้ต่อไปนี้ในการตอบคำถามของผู้ใช้ \nContext:\n{context}\n
      Instructions:
      - หากเอกสารมีข้อมูลเพียงพอ ให้ตอบคำถามโดยตรงจากเอกสาร
      - หากเอกสารไม่เพียงพอ หรือจำเป็นต้องใช้ข้อมูลที่อัปเดตมากกว่า ให้ใช้เครื่องมือ 'search'
      - หากไม่ทราบคำตอบ ให้ตอบว่าไม่ทราบ และห้ามเดาข้อมูลขึ้นมาเอง
      - ต้องตอบให้ "กระชับ"  และ "ตรงประเด็น" ที่สุด
      - หากมี context ที่ยาว **ให้สรุปใจความสำคัญไม่เกิน 3-5 บรรทัด** เท่านั้น
      - ห้ามเขียนเกริ่นนำยืดยาว (No preamble) หรือเขียนสรุปตอนท้ายซ้ำซ้อน      
    """
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_propmt),
            MessagesPlaceholder(variable_name="messages"),
            # Prompt เพื่อรับประวัติการคุย เมื่อใช้ ChatPromptTemplate
        ]
    )
    
    chain = prompt | gemini.bind_tools([search])

    return chain
