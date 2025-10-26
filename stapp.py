import streamlit as st
import os
import PyPDF2
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
from openai import OpenAI
import tiktoken
from typing import List, Dict

class ReadPDFFiles:
    """
    读取 PDF 文件的类，用于从指定路径读取 PDF 文件并进行内容分割。
    """

    def __init__(self, path: str) -> None:
        """
        初始化函数，设定要读取的文件路径，并获取该路径下所有 PDF 文件。
        :param path: 文件夹路径
        """
        self._path = path
        self.file_list = self.get_files()

    def get_files(self):
        """
        遍历指定文件夹，获取 PDF 文件路径列表。
        :return: 文件路径列表
        """
        file_list = []
        for filepath, dirnames, filenames in os.walk(self._path):
            for filename in filenames:
                if filename.endswith(".pdf"):
                    file_list.append(os.path.join(filepath, filename))
        return file_list

    def get_content(self, max_token_len: int = 600, cover_content: int = 150):
        """
        读取 PDF 文件内容并进行分割，将长文本切分为多个块。
        :param max_token_len: 每个文档片段的最大 Token 长度
        :param cover_content: 在每个片段之间重叠的 Token 长度
        :return: 切分后的文档片段列表
        """
        docs = []
        for file in self.file_list:
            try:
                content = self.read_file_content(file)
                chunk_content = self.get_chunk(content, max_token_len=max_token_len, cover_content=cover_content)
                docs.extend(chunk_content)
            except Exception as e:
                print(f"Error processing file {file}: {e}")
        return docs

    @classmethod
    def get_chunk(cls, text, max_token_len=600, cover_content=150):
        encoder = tiktoken.get_encoding("cl100k_base")
        token_len = max_token_len - cover_content
        lines = text.splitlines()
        chunk_text = []
        curr_chunk = ''
        curr_len = 0

        def add_chunk():
            if curr_chunk:
                chunk_text.append(curr_chunk)

        for line in lines:
            line = line.replace(' ', '')
            line_len = len(encoder.encode(line))
            if line_len > max_token_len:
                num_chunks = (line_len + token_len - 1) // token_len
                for i in range(num_chunks):
                    start = i * token_len
                    end = start + token_len
                    while not line[start:end].rstrip().isspace():
                        start += 1
                        end += 1
                        if start >= line_len:
                            break
                    new_chunk = curr_chunk[-cover_content:] + line[start:end]
                    chunk_text.append(new_chunk)
                curr_chunk = curr_chunk[-cover_content:] + line[start:end]
            else:
                if curr_len + line_len <= token_len:
                    curr_chunk += line + '\n'
                    curr_len += line_len + 1
                else:
                    add_chunk()
                    curr_chunk = curr_chunk[-cover_content:] + line
                    curr_len = line_len + cover_content

        add_chunk()
        return chunk_text
    @classmethod
    def read_file_content(cls, file_path: str):
        """
        读取 PDF 文件内容。
        :param file_path: PDF 文件路径
        :return: PDF 文件中的文本内容
        """
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text()
                return text
        except Exception as e:
            print(f"Error reading PDF file {file_path}: {e}")
            return ""




class Embeddings:
    """
    合并后的向量化类，用于将文本转换为向量表示并计算余弦相似度。
    """
    def __init__(self, api_key: str) -> None:
        """
        初始化类，设置 OpenAI API 客户端。
        
        参数：
        api_key (str) - 用于 API 调用的密钥。
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key, base_url="https://ai.devtool.tech/proxy/v1")

    def get_embedding(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
        """
        使用 OpenAI 的 Embedding API 获取文本的向量表示。
        
        参数：
        text (str) - 需要转化为向量的文本。
        model (str) - 使用的 Embedding 模型名称，默认为 'text-embedding-3-large'。
        
        返回：
        list[float] - 文本的向量表示。
        """
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding

    @staticmethod
    def cosine_similarity(vector1: List[float], vector2: List[float]) -> float:
        """
        计算两个向量之间的余弦相似度。
        
        参数：
        vector1 (list[float]) - 第一个向量。
        vector2 (list[float]) - 第二个向量。
        
        返回：
        float - 余弦相似度值。
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        return dot_product / magnitude if magnitude else 0


class VectorStore:
    """
    向量存储类，用于存储文档及其对应的向量表示，并提供持久化和查询功能。
    """
    def __init__(self, documents: list[str] = None):
        """
        初始化向量存储类。

        参数：
        documents (list[str], optional): 文档列表，默认为空列表。
        """
        if documents is None:
            documents = []
        self.documents = documents
        self.vectors = []

    def get_vector(self, embedding_model):
        """
        使用传入的嵌入模型将文档向量化。

        参数：
        embedding_model: 用于生成向量的模型（需继承自 Embeddings 类）。

        返回：
        list[list[float]]: 文档对应的向量列表。
        """
        self.vectors = [embedding_model.get_embedding(doc) for doc in self.documents]
        return self.vectors

    def persist(self, storage_path='storage'):
        """
        将文档和对应的向量表示持久化到本地目录。

        参数：
        storage_path (str): 存储路径，默认为 'storage'。
        """
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        np.save(os.path.join(storage_path, 'vectors.npy'), self.vectors)
        with open(os.path.join(storage_path, 'documents.txt'), 'w') as file:
            for doc in self.documents:
                file.write(f"{doc}\n")

    def load_vector(self, storage_path='storage'):
        """
        从本地加载之前保存的文档和向量数据。

        参数：
        storage_path (str): 存储路径，默认为 'storage'。
        """
        self.vectors = np.load(os.path.join(storage_path, 'vectors.npy')).tolist()
        with open(os.path.join(storage_path, 'documents.txt'), 'r') as file:
            self.documents = [line.strip() for line in file.readlines()]

    def get_similarity(self, vector1, vector2):
        """
        计算两个向量的余弦相似度。

        参数：
        vector1 (list[float]): 第一个向量。
        vector2 (list[float]): 第二个向量。

        返回：
        float: 两个向量的余弦相似度，范围在 -1 到 1 之间。
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        return dot_product / magnitude if magnitude else 0

    def query(self, query_text, embedding_model, num_results=1):
        """
        根据用户的查询文本检索最相关的文档片段。

        参数：
        query_text (str): 用户的查询文本。
        embedding_model: 用于将查询向量化的嵌入模型。
        num_results (int): 返回最相似的文档数量，默认为 1。

        返回：
        list[str]: 返回最相似的文档列表。
        """
        query_vector = embedding_model.get_embedding(query_text)
        similarities = [self.get_similarity(query_vector, vector) for vector in self.vectors]
        top_indices = np.argsort(similarities)[-num_results:][::-1]
        return [self.documents[idx] for idx in top_indices]


class LLModel:
    """
    合并后的模型类，既可以处理本地模型加载，也可以通过 API 调用生成回答。
    """
    def __init__(self, api_key: str = None, model_path: str = '') -> None:
        """
        初始化模型类。

        参数：
        api_key (str, optional): OpenAI API 的密钥，如果使用 API 模型则需要提供。
        model_path (str): 用于存储模型文件的路径，如果是本地模型则需要提供。
        """
        self.api_key = api_key
        self.model_path = model_path
        if api_key:
            self.client = OpenAI(api_key=api_key, base_url="https://ai.devtool.tech/proxy/v1")

    def chat(self, prompt: str, history: List[Dict] = [], content: str = '') -> str:
        """
        根据不同情况生成回答。

        参数：
        prompt (str): 用户的提问内容。
        history (List[Dict], optional): 之前的对话历史（字典列表）。
        content (str, optional): 提供的上下文内容。

        返回：
        str: 模型生成的答案。
        """
        
        PROMPT_TEMPLATE = dict(
                PROMPT_TEMPLATE_1="""
                以下是一个可能与问题相关的参考段落。
                1. 如果参考段落与问题相关，请先总结参考段落的内容；
                2. 如果参考段落与问题无关，则运用你的知识储备来回答用户的问题。
                3. 始终使用中文进行回答。

                问题: {question}
                可参考的上下文：
                ···
                {context}
                ···
                有用的回答:"""
            )
        
        if self.api_key:
            # 构建包含问题和上下文的完整提示
            prompts = PROMPT_TEMPLATE['PROMPT_TEMPLATE_1'].format(question=prompt, context=content)
            # 调用 GPT-4o 模型进行推理
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompts}
                ]
            ).choices[0].message.content
            # 返回模型生成的第一个回答
            return response
        else:
            # 如果是本地模型，这里可以添加加载本地模型并生成回答的逻辑
            pass
def main():
    st.title("PDF 文档内容检索")

    # 1. 选择目录/文件
    dir_path = st.text_input("请输入包含 PDF 文件的目录路径","")
    if dir_path:
        # 检查路径是否存在
        if os.path.isdir(dir_path):
            # 使用 ReadPDFFiles 类读取目录下的 PDF 文件
            file_reader = ReadPDFFiles(path=dir_path)
            st.write("读取到的文件列表:", file_reader.get_files())
            # 读取文件内容并进行分割
            document_chunks = file_reader.get_content(max_token_len=500, cover_content=200)

            # 2. 展示前3个chunk
            for i in range(min(3, len(document_chunks))):
                st.write(f"Chunk {i+1}:", document_chunks[i])
        else:
            st.error("输入的路径不是一个有效的目录，请重新输入。")
            
        

        # 3. 展示相似的文档和大模型最终生成的结果
        # 初始化向量存储和嵌入模型
        
        vector_store = VectorStore(documents=document_chunks)
        api_key = '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'

        embedding_model = Embeddings(api_key=api_key)  # 替换为你的 OpenAI API 密钥

        # 获取文档向量并存储
        vector_store.get_vector(embedding_model)

        # 4. 输入问题
        user_query = st.text_input("请输入你的问题")
        if user_query:
            # 查询最相似的文档片段
            similar_docs = vector_store.query(user_query, embedding_model, num_results=1)
            st.write("最相似的文档片段:", similar_docs[0])

            chat = LLModel(api_key=api_key)
            answer = chat.chat(user_query, [], similar_docs)
            
            st.write("输出结果:", answer)
            
            
if __name__ == "__main__":
    main()



