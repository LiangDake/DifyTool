import os
import shutil

from langchain_community.document_loaders import *
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from paddleocr import PaddleOCR
from langchain.schema.document import Document
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function


def import_file(file_path: str):
    # 获取文件类型
    _, file_type = os.path.splitext(file_path)
    file_type = file_type.lower()

    if file_type in ('.pdf', '.png', '.jpg', 'jpeg'):
        # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
        # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
        ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory

        result = ocr.ocr(file_path, cls=True)
        # 初始化变量以存储所有识别的文本
        page_content = ""

        # 遍历识别结果
        for line in result[0]:
            # 每个line的结构是[[[[x, y], [m, n]], (text, score)]]
            # line_info = line[0]  # 访问包含位置和文本信息的第一部分
            text = line[1][0]  # 提取文本部分
            page_content += text + "\n\n"  # 将文本添加到page_content，每段文本后加两个换行符

        # 移除末尾多余的换行符
        page_content = page_content.strip()
        # 使用定义的 Document 类创建文档对象
        document = Document(metadata={"source": file_path}, page_content=page_content)
        # 构造包含 Document 对象的列表
        doc = [document]

    elif file_type in (".doc", ".docx"):
        doc = Docx2txtLoader(file_path).load()

    elif file_type in (".ppt", ".pptx"):
        doc = UnstructuredPowerPointLoader(file_path).load()

    elif file_type in (".xls", ".xlsx"):
        doc = UnstructuredExcelLoader(file_path).load()

    elif file_type in (".htm", ".html"):
        doc = BSHTMLLoader(file_path, open_encoding="unicode_escape").load()

    elif file_type in ".eml":
        doc = UnstructuredEmailLoader(file_path=file_path, process_attachments=True).load()

    elif file_type in ".csv":
        doc = CSVLoader(file_path).load()

    elif file_type in ".txt":
        doc = TextLoader(file_path, autodetect_encoding=True).load()

    else:
        doc = TextLoader(file_path, autodetect_encoding=True).load()

    return doc


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document], chroma_path):
    # Load the existing database.
    db = Chroma(
        persist_directory=chroma_path, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database(chroma_path):
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

