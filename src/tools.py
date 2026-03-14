from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool


@tool(response_format="content_and_artifact")  # <--- ต้องประกาศตรงนี้ด้วย
def search(input: str):
    """Use this tool to search for more information based on the user question"""
    print("retrieving information on website...\n\n")
    tool = TavilySearchResults( max_results=2 )  # 2 web ที่เกียวข้อง ประหยัด

    results = tool.invoke(input)
    content_parts = []
    if isinstance(results, list):
        for doc in results:
            content_parts.append(f"Source: {doc.get('url', '')}\nContent: {doc.get('content', '')}")
        content_str = "\n\n".join(content_parts)
    else:
        content_str = str(results)

    artifact = results
    return content_str, artifact

# ตัวแรก จะถูกจับใส่ content
# ตัวที่สอง จะถูกจับใส่ artifact (อัตโนมัติ)

# ToolMessage(
#     content="Found 2 relevant results.",
#     artifact=[
#         {
#             "title": "...",
#             "url": "...",
#             "content": "...",
#             "score": 0.99
#         },
#         ...
#     ]
# )

