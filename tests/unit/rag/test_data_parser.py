import sys
sys.path.append('/mnt/e/RAGbot')

import unittest
from unittest.mock import patch
from langchain.output_parsers import ResponseSchema

from rag_framework.output_parser.dataframe import createDataFrame  # 导入要测试的函数
from rag_framework.model.openai_client import chat  # 模拟 chat


input_template = """您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
{format_instructions}"""



# 定义我们想要接收的响应模式
response_schemas = [
    ResponseSchema(name="description", description="鲜花的描述文案"),
    ResponseSchema(name="reason", description="问什么要这样写这个文案"),
]

# 数据准备
flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["50", "30", "20"]

data = [flowers, prices]

# 调用函数创建 DataFrame
df = createDataFrame(["flower_name", "price"], response_schemas, input_template, chat, data)

# 打印结果
print(df.to_dict(orient="records"))
