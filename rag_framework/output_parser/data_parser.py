# 标准库导入
import json
from typing import Type

from langchain.output_parsers import PydanticOutputParser, OutputFixingParser,RetryWithErrorOutputParser
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI

import os

class OutputParser:
    """
    OutputParser 是一个通用的输出解析器类，用于解析语言模型（LLM）的响应为特定的结构化格式。
    它特别支持 JSON 格式的解析，并结合 Pydantic 模型验证和 LangChain 的输出修复机制。

    属性:
        outputType (str): 指定解析器的输出类型，例如 "json" 或 "jsonFix"。
        parser (Union[PydanticOutputParser, OutputFixingParser]): 内部使用的解析器实例。

    方法:
        __init__(outputType: str, ObjectType: Type = None):
            初始化解析器，根据指定的输出类型和数据模型配置解析器实例。

        get_format_instructions() -> str:
            获取格式化指令，指导语言模型输出符合预期格式的响应。

        parse(output: str) -> dict:
            将语言模型的响应字符串解析为结构化数据，支持自动修复机制。
    """

    def __init__(self, outputType: str, ObjectType: Type = None):
        """
        初始化 OutputParser 实例。

        参数:
            outputType (str): 输出类型，例如 "json" 或 "jsonFix"。
            ObjectType (Type, 可选): Pydantic 模型类，用于解析 JSON 数据。

        异常:
            ValueError: 当 outputType 为 "json" 且未指定 ObjectType 时，抛出异常。
        """
        self.outputType = outputType
        if outputType == "json" and ObjectType is not None:
            # 配置标准 JSON 解析器
            self.parser = JsonOutputParser(pydantic_object=ObjectType)
        elif outputType == "jsonPlus" and ObjectType is not None:
            # 配置标准 Pydantic 解析器
            self.parser = PydanticOutputParser(pydantic_object=ObjectType)
        elif outputType == "jsonFix" and ObjectType is not None:
            parser = PydanticOutputParser(pydantic_object=ObjectType)
            # 配置带自动修复功能的解析器
            self.parser = OutputFixingParser.from_llm(
                parser=parser,
                llm=ChatOpenAI(
                    model=os.environ.get("LLM_MODELEND"),
                ),
            )
        elif outputType == "jsonRetry":
            parser = PydanticOutputParser(pydantic_object=ObjectType)
            self.parser = RetryWithErrorOutputParser.from_llm(
                parser=parser,                 
                llm=ChatOpenAI(
                    model=os.environ.get("LLM_MODELEND"),
                    temperature=0.0,
                ))
        else:
            raise ValueError("Invalid configuration: For 'json' outputType, ObjectType must be specified.")

    def get_format_instructions(self) -> str:
        """
        获取格式化指令，用于指导语言模型生成符合解析器要求的输出。

        返回:
            str: 格式化指令字符串。

        示例:
            parser = OutputParser(outputType="json", ObjectType=YourPydanticModel)
            format_instructions = parser.get_format_instructions()
            print("输出格式指令：", format_instructions)
        """
        instructions = (self.parser.get_format_instructions())

        # 提取 JSON schema 部分
        start = instructions.find("```") + 3
        end = instructions.rfind("```")
        schema_str = instructions[start:end].strip()

        # 转换为字典
        schema = (json.loads(schema_str)).get("properties",{})

        return schema

    def get_parser(self):
        return self.parser
    def parse(self, output: str, **args) -> dict:
        """
        解析语言模型输出为结构化数据。

        参数:
            output (str): 语言模型生成的响应字符串。
            args: 可选参数，用于支持 jsonRetry 类型传递 prompt。

        返回:
            dict: 解析后的结构化数据。
        """
        if self.outputType == "jsonRetry":
            if "prompt" not in args:
                raise ValueError("jsonRetry requires a 'prompt' argument.")
            return self.parser.parse_with_prompt(output, args["prompt"])
        return self.parser.parse(output)
