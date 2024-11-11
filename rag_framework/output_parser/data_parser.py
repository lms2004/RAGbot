# 标准包
from typing import List
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

def create_output_parser(response_schemas: List[ResponseSchema]) -> StructuredOutputParser:
    """
        创建一个结构化输出解析器。
        
        参数：
            response_schemas (List[ResponseSchema]): 定义输出数据结构的响应模式列表。
        
        返回：
            StructuredOutputParser: 配置好的结构化输出解析器，可用于解析模型输出。
        
        示例：
            response_schemas = [
                ResponseSchema(name="description", description="鲜花的描述文案"),
                ResponseSchema(name="reason", description="问什么要这样写这个文案"),
            ]
            parser = create_output_parser(response_schemas)
    """
    return StructuredOutputParser.from_response_schemas(response_schemas)
