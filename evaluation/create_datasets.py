from langsmith import Client
from dotenv import load_dotenv
load_dotenv()

client = Client()
dataset_name="test-set-1"

dataset = client.create_dataset( dataset_name=dataset_name, description="A sample dataset.")
examples = [
   {
        "inputs": {"question": "CEO ของบริษัทคือใคร"},
        "outputs": {"answer": "นายณัฐพล วิริยะกุล"},
    },
    {
        "inputs": {"question": "บริษัทตั้งอยู่ที่ไหน"},
        "outputs": {"answer": "อาคาร Tech Tower ชั้น 25 ถนนสาทร เขตบางรัก กรุงเทพมหานคร"},
    },
    {
        "inputs": {"question": "ทำงานมา 2 ปี ได้วันลาพักร้อนกี่วัน"},
        "outputs": {"answer": "10 วันต่อปี (สำหรับพนักงานอายุงาน 1-3 ปี)"},
    },
    {
        "inputs": {"question": "วันลาพักร้อนสะสมไปปีหน้าได้ไหม"},
        "outputs": {"answer": "ได้ แต่สะสมได้ไม่เกิน 5 วัน"},
    },

    {
        "inputs": {"question": "เอา Flash Drive ส่วนตัวมาใช้ได้ไหม"},
        "outputs": {"answer": "ไม่ได้ (ห้ามเด็ดขาด) หากฝ่าฝืนมีบทลงโทษทางวินัยสูงสุดคือการตักเตือนเป็นลายลักษณ์อักษร"},
    },

    {
        "inputs": {"question": "ประเทศไทยอนุญาตให้นำสัตว์เลี้ยงขึ้นรถไฟฟ้า BTS หรือไม่"},
        "outputs": {"answer": "ไม่ได้"},
    },
]


client.create_examples(dataset_id=dataset.id, examples=examples)
