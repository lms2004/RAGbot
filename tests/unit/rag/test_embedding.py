import sys
import os

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

from rag_framework.Embedding.embedding import *

def test_DoubaoEmbedding():
    embeddings_model = DoubaoEmbeddings(
        model=os.environ["EMBEDDING_MODELEND"],
    )

    # Embed文本
    embeddings = embeddings_model.embed_documents(
        [
            "您好，有什么需要帮忙的吗？",
            "哦，你好！昨天我订的花几天送达",
            "请您提供一些订单号？",
            "12345678",
        ]
    )
    print(len(embeddings), len(embeddings[0]))

    # Embed查询
    embedded_query = embeddings_model.embed_query("刚才对话中的订单号是多少?")
    print(embedded_query[:3])

def test_cacheMemory():
    # 创建一个OpenAIEmbeddings的实例，这将用于实际计算文档的嵌入
    underlying_embeddings = DoubaoEmbeddings(
        model=os.environ["EMBEDDING_MODELEND"],
    )

    embedder = MemoryCreator().create(underlying_embeddings)

    # 使用embedder为两段文本生成嵌入。
    # 结果，即嵌入向量，将被存储在上面定义的内存存储中。
    embeddings = embedder.embed_documents(["你好", "智能鲜花客服"])
    print(embeddings)

def test_vectorMemory():
    """ 
    本文件是【检索增强生成：通过 RAG 助力鲜花运营】章节的配套代码，课程链接：https://juejin.cn/book/7387702347436130304/section/7388069959185727524
    您可以点击最上方的“运行“按钮，直接运行该文件；更多操作指引请参考Readme.md文件。
    """
    # 设置OpenAI的API密钥
    import os



    embeddings = DoubaoEmbeddings(
        model=os.environ["EMBEDDING_MODELEND"],
    )

    # 导入文档加载器模块，并使用TextLoader来加载文本文件
    from langchain_community.document_loaders import TextLoader
    from langchain_openai import ChatOpenAI  # ChatOpenAI模型

    loader = TextLoader("./OneFlower/花语大全.txt", encoding="utf8")

    # 使用VectorstoreIndexCreator来从加载器创建索引
    from langchain.indexes import VectorstoreIndexCreator

    index = VectorstoreIndexCreator(embedding=embeddings).from_loaders([loader])

    llm = ChatOpenAI(model=os.environ["LLM_MODELEND"], temperature=0)

    # 定义查询字符串, 使用创建的索引执行查询
    query = "玫瑰花的花语是什么？"
    result = index.query(llm=llm, question=query)
    print(result)  # 打印查询结果

    # 替换成你所需要的工具
    from langchain.text_splitter import CharacterTextSplitter

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    from langchain_community.vectorstores import Qdrant

    index_creator = VectorstoreIndexCreator(
        vectorstore_cls=Qdrant,
        embedding=embeddings,
        text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0),
    )
   


if __name__ == "__main__":
    # test_DoubaoEmbedding()
    test_cacheMemory()
    test_vectorMemory()
