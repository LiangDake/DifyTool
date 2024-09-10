from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate

from base import *

PROMPT_TEMPLATE = """
仅根据以下context回答问题，要求详细使用并列举每一个context：

{context}

---

根据以上context回答问题：{question}
"""


# 线索发现 图谱 分析 写作
def build_knowledge_base(folder_path, chroma_path):
    for filename in os.listdir(folder_path):
        # 构建完整的文件路径
        if filename.startswith('.'):
            continue
        file_path = os.path.join(folder_path, filename)
        doc = get_doc(file_path)
        chunks = split_documents(doc)
        # 嵌入并存入Chroma数据库
        add_to_chroma(chunks, chroma_path)


def rag_knowledge_base(query_text: str, chroma_path: str, file_num: int): # file_num < 20
    # 嵌入模型
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    # 搜索向量数据库
    results = db.similarity_search_with_score(query_text, k=file_num)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # 输出查找到的内容
    print(context_text)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = Ollama(model="qwen2")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"{response_text}\n\n文件来源如下: {sources}"
    print(formatted_response)
    return formatted_response

# def rag_single_file()



