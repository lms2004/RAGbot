import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def loader(path) -> list[str]:
    """
    加载指定路径下的所有文档内容并返回一个文档列表。

    参数：
    - path (str): 文档目录的相对路径或绝对路径

    返回：
    - list[str]: 包含文档内容的字符串列表。每个文档内容以一个字符串表示。

    注意事项：
    - 支持文件类型: PDF (.pdf), Word (.docx), 纯文本 (.txt)
    - 目录下的每个文件会被自动识别类型并使用相应的加载器加载。
    - 返回的文档列表按文件夹中文件的顺序排列。
    """
    path = os.path.abspath(path)

    documents = []
    for filepath in os.listdir(path):
        filepath = os.path.join(path,filepath)

        if filepath.endswith(".pdf"):
            loader = PyPDFLoader(filepath)  
            documents.extend(loader.load())
        if filepath.endswith(".docx"):
            loader = Docx2txtLoader(filepath)
            documents.extend(loader.load())    
        if filepath.endswith(".txt"):
            loader = TextLoader(filepath)
            documents.extend(loader.load())
    return documents


print(loader("./RAG/loader/docs/"))