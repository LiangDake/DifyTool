from email.message import EmailMessage

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from py2neo import Graph
import file_processing
import utils
import os
import shutil
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


# 知识库类
class KnowledgeBase:
    def __init__(self, chroma_path="default"):
        # 默认编码模型
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        # 默认知识库
        self.chroma_path = chroma_path
        if self.chroma_path == "default":
            self.db = Chroma(
                embedding_function=self.embeddings
            )
        else:
            self.db = Chroma(
                persist_directory=self.chroma_path, embedding_function=self.embeddings
            )

    # 文字分离
    def split_documents(self, documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)

    # 计算并记录文件id
    def calculate_chunk_ids(self, chunks):

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

    # 添加文件至chroma
    def add_to_chroma(self, chunks: list[Document]):
        # Load the existing database.

        # Calculate Page IDs.
        chunks_with_ids = self.calculate_chunk_ids(chunks)

        # Add or Update the documents.
        existing_items = self.db.get(include=[])  # IDs are always included by default
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
            self.db.add_documents(new_chunks, ids=new_chunk_ids)
            self.db.persist()
        else:
            print("✅ No new documents to add")

    # 实现多个文件的导入并存储在知识库中
    def add_folder(self, folder_path: str) -> str:
        for filename in os.listdir(folder_path):
            # 构建完整的文件路径
            if filename.startswith('.'):
                continue
            file_path = os.path.join(folder_path, filename)
            doc = file_processing.import_file(file_path)
            if doc is not False:
                chunks = self.split_documents(doc)
                # 嵌入并存入Chroma数据库
                self.add_to_chroma(chunks)
            else:
                print(f"{file_path}文件类型错误或导入失败")
        return "所有文件已全部导入知识库，请检查文件内容"

    def delete_from_chroma(self, file_path):
        pass

    # 清除知识库内容
    def clear_database(self):
        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)

    # Email预处理


# 检索知识库类，其中包括：（相关人物的）查询、线索发现、分析、总结、写作
# 记得添加语音分析功能‼️
class QueryKnowledgeBase():
    def __init__(self, chroma_path="image", file_num=5):
        # 默认编码模型
        self.embeddings = OllamaEmbeddings(model="yxl/m3e")
        # 默认知识库
        self.chroma_path = chroma_path
        if self.chroma_path == "default":
            self.db = Chroma(
                embedding_function=self.embeddings
            )
        else:
            self.db = Chroma(
                persist_directory=self.chroma_path, embedding_function=self.embeddings
            )
        # 默认模型为qwen2:7b
        self.llm = Ollama(model="qwen2")
        # 默认检索片段数
        self.file_num = file_num
        self.PROMPT_TEMPLATE = """
        根据以下context回答问题：{question}，要求详细使用并且详细指出知识来源。
        {context}
        """

    def query_context(self, query_text: str) -> str:
        # 根据查询获取相关内容
        # 搜索向量数据库
        results = self.db.similarity_search_with_score(query_text, k=self.file_num)
        print(results)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        # # 输出查找到的内容
        # print(context_text)
        # print("-----------------------")
        return context_text

    def query_content_with_sources(self, query_text: str) -> str:
        # 根据查询获取相关内容
        # 搜索向量数据库
        results = self.db.similarity_search_with_score(query_text, k=self.file_num)
        # print(results)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        # # 输出查找到的内容
        # print(context_text)
        # print("-----------------------")
        prompt_template = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        # model = Ollama(model="qwen2")
        response_text = self.llm.invoke(prompt)

        sources = [doc.metadata.get("id", None) for doc, _score in results]
        formatted_response = f"{response_text}\n\n文件来源如下: {sources}"
        # print(formatted_response)
        return formatted_response

    def query_content(self, query_text: str) -> str:
        # 根据查询获取相关内容
        # 搜索向量数据库
        results = self.db.similarity_search_with_score(query_text, k=self.file_num)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        prompt_template = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        response_text = self.llm.invoke(prompt)
        return response_text


# 文件类，其中包括：（多文件）总结与分析、（保留格式的高精度）翻译、（高质量、统一格式）写作
class QueryFileBase(QueryKnowledgeBase):
    def __init__(self):
        super().__init__()

    def translate_file(self, file_path: str):
        # 实现翻译逻辑
        # 获取文件中的文字
        doc = file_processing.import_file(file_path)
        # 确保格式正确
        if doc is not False:
            source_text = doc[0].page_content
            translation = utils.translate(
                source_text=source_text
            )
            # 获取文件类型
            file_type = file_processing.get_file_type(file_path)
            if file_type in '.pdf':
                # 如果是 PDF 文件，调用 save_translated_pdf 保存
                translated_file_path = file_processing.save_translated_pdf(file_path, translation)
            else:
                # 否则默认保存为 DOCX
                translated_file_path = file_processing.save_translated_docx(file_path, translation)

            return translated_file_path

    def translate_email(self, file_path: str):
        # 导入邮件获得结果
        result = file_processing.import_email(file_path, "email")
        # 邮件正文
        doc = result["doc"]
        # 邮件头
        email_headers = result["email_headers"]
        # 邮件附件文件夹
        attachment_folder = result["output_folder"]
        # 翻译后的附件文件夹
        translated_attachments = []

        # 对邮件正文进行翻译
        if doc is not False:
            source_text = doc[0].page_content
            translation = utils.translate(
                source_text=source_text
            )
        else:
            # 无正文
            translation = None

        # 检查附件文件夹是否存在文件
        if os.path.exists(attachment_folder) and len(os.listdir(attachment_folder)) > 0:
            # 获取附件文件列表
            attachments = os.listdir(attachment_folder)
            # 遍历附件并翻译
            for file_name in attachments:
                attachment_file_path = os.path.join(attachment_folder, file_name)
                translated_file_path = self.translate_file(attachment_file_path)
                translated_attachments.append(translated_file_path)

        # 重新创建新的EML
        translated_file_path = file_processing.save_translated_email(
            file_path,
            email_headers,
            translation,
            translated_attachments
        )
        # 清空附件文件夹
        file_processing.delete_folder(attachment_folder)
        print(f"{file_path}已成功翻译！\n")

        return translated_file_path


    def conclusion(self, folder_path: str):
        conclusion_prompt = ChatPromptTemplate.from_messages(
            [("system", "简要概括以下内容，要求内容文字连贯，并找到其中的关联性:\\n\\n{context}。以中文输出你概括的内容。")]
        )
        context = []
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            docs = file_processing.import_file(file_path)
            context.append(docs[0])

        chain = create_stuff_documents_chain(self.llm, conclusion_prompt)
        # Invoke chain
        result = chain.invoke({"context": context})
        return result


# 图谱类
class GraphBase(QueryFileBase):
    def __init__(self):
        super().__init__()
        os.environ["NEO4J_URI"] = "bolt://localhost:7687"
        os.environ["NEO4J_USERNAME"] = "neo4j"
        os.environ["NEO4J_PASSWORD"] = "lkzxxzcsc2020"
        os.environ["NEO4J_DATABASE"] = "neo4j"

        self.graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="lkzxxzcsc2020")

        # 连接到Neo4j数据库
        self.graph_db = Graph(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")))

    # 简易翻译
    def simple_translate(self, content):
        translate_template = "将以下文本翻译为中文: {text}"
        translate_prompt = PromptTemplate.from_template(translate_template)
        llm = self.llm
        chain = (
                translate_prompt | llm
        )
        result = chain.invoke({"text": content})
        return result

    def build_knowledge_graph(self, file_path: str):
        doc = file_processing.import_file(file_path)
        llm_transformer = LLMGraphTransformer(
            llm=self.llm
        )
        graph_documents = llm_transformer.convert_to_graph_documents(documents=doc)
        print(f"Nodes:{graph_documents[0].nodes}")
        print(f"Relationships:{graph_documents[0].relationships}")

        try:
            self.graph.add_graph_documents(graph_documents)
            return True
        except Exception:
            return False

    def build_graph(self, folder_path: str):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            docs = file_processing.import_file(file_path)
            # 文件导入成功
            if docs:
                print(f"{file} 已成功读取\n")
                # # 分割文档，邮件其实不需要分割也行
                # documents = text_splitter.split_documents(documents=docs)
                # 获得翻译后的文字
                translated_content = self.simple_translate(docs[0].page_content)
                # 获得翻译后的文档
                translated_docs = [Document(page_content=translated_content)]
                llm_transformer = LLMGraphTransformer(
                    llm=self.llm
                )
                graph_documents = llm_transformer.convert_to_graph_documents(documents=translated_docs)
                # print(f"Nodes:{graph_documents[0].nodes}")
                # print(f"Relationships:{graph_documents[0].relationships}")
                try:
                    self.graph.add_graph_documents(graph_documents)
                    print(f"{file} 已成功导入知识图谱\n\n")
                except Exception:
                    print(f"{file} 未能导入知识图谱\n\n")
                    continue
            else:
                print(f"{file} 导入失败，请检查文件格式\n")
                continue

        print(f"{folder_path} 文件夹所有文件已成功导入知识图谱。\n")


    def query_knowleage_graph(self, query):
        chain = GraphCypherQAChain.from_llm(
            self.llm,
            graph=self.graph,
            verbose=True
        )
        result = chain.invoke({"query": query})
        final_answer = result['result']
        return final_answer

    def delete_knowleage_graph(self):
        try:
            # 清空Neo4j数据库
            self.graph_db.run("MATCH (n) DETACH DELETE n")
            return True
        except Exception:
            return False


# # 初始化图知识库（图形数据库需要更改URL地址与账号密码）
# graph_base = GraphBase()
# # # 构建知识图谱（先分块再翻译再构建）
# # result = graph_base.build_graph('/Users/liangdake/Downloads/工作/人工智能实习/email')
#
# result2 = graph_base.query_knowleage_graph("Find the information about 维多利亚大学")
# print(result2)