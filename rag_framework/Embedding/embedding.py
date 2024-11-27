# 初始化Embedding类
from volcenginesdkarkruntime import Ark
from typing import List, Any
from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel
import os

# 导入内存存储库，该库允许我们在RAM中临时存储数据
from langchain.storage import InMemoryStore

# 导入与嵌入相关的库。OpenAIEmbeddings是用于生成嵌入的工具，而CacheBackedEmbeddings允许我们缓存这些嵌入
from langchain.embeddings import CacheBackedEmbeddings

from rag_framework.Embedding.loader import *



class DoubaoEmbeddings(BaseModel, Embeddings):
    client: Ark = None
    api_key: str = ""
    model: str

    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.api_key == "":
            self.api_key = os.environ["OPENAI_API_KEY"]
        self.client = Ark(
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=self.api_key
        )

    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.
        Args:
            texts (str): 要生成 embedding 的文本.
        Return:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表.
        """
        embeddings = self.client.embeddings.create(model=self.model, input=text)
        return embeddings.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    class Config:
        arbitrary_types_allowed = True

def cacheCreator(embeddings, store):
    return CacheBackedEmbeddings.from_bytes_store(
        embeddings,  # 实际生成嵌入的工具
        store,  # 嵌入的缓存位置
        namespace=embeddings.model,  # 嵌入缓存的命名空间
    )





class MemoryCreator():
    """
        参数：
    
    
    """
    def __init__(self, memory_type = "cache",store=InMemoryStore()):
        self.store = store
        if memory_type == "cache":
            self.createFunc = cacheCreator
        elif memory_type == "vector":
            self.createFunc = 
    def create(self, embeddings):
        return  self.createFunc(embeddings=embeddings, store=self.store)


