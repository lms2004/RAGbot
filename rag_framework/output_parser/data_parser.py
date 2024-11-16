# 标准包
from typing import List
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.output_parsers import PydanticOutputParser


class OutputParser:
    """
    OutputParser 是一个通用的输出解析器类，用于处理语言模型的输出，并将其解析为特定的数据格式。
    它支持多种输出类型，特别是 JSON 格式的结构化数据解析。

    Attributes:
        outputType (str): 输出类型，例如 "json" 或其他格式。
        ObjectType (Type, optional): 当输出类型为 "json" 时，指定的 Pydantic 数据模型类型，用于解析 JSON 数据。

    Methods:
        __init__(outputType: str, ObjectType=None):
            初始化 OutputParser 实例，根据指定的输出类型配置解析器。

        get_format_instructions() -> str:
            获取格式化指令，指导语言模型生成符合预期的数据格式。

        parse(output: str) -> dict:
            解析语言模型输出，将其转化为指定的数据格式。
    """

    def __init__(self, outputType: str, ObjectType=None):
        """
        初始化 OutputParser 实例。

        Args:
            outputType (str): 输出类型，例如 "json" 或其他格式。
            ObjectType (Type, optional): 当输出类型为 "json" 时，指定的 Pydantic 数据模型类型，用于解析 JSON 数据。

        Raises:
            ValueError: 如果 outputType 为 "json" 且 ObjectType 未指定时，抛出异常。
        """
        self.outputType = outputType
        if outputType == "json" and ObjectType is not None:
            self.parser = PydanticOutputParser(pydantic_object=ObjectType)
        else:
            raise ValueError("Invalid configuration: For 'json' outputType, ObjectType must be specified.")

    def get_format_instructions(self) -> str:
        """
        获取格式化指令，用于指导语言模型输出符合特定格式的响应。

        Returns:
            str: 格式化指令字符串。

        Example:
            parser = OutputParser(outputType="json", ObjectType=YourPydanticModel)
            format_instructions = parser.get_format_instructions()
            print("输出格式：", format_instructions)
        """
        return self.parser.get_format_instructions()

    def parse(self, output: str) -> dict:
        """
        解析语言模型的输出，将其转化为特定的数据格式。

        Args:
            output (str): 语言模型的输出字符串。

        Returns:
            dict: 解析后的输出数据，以字典形式返回。

        Example:
            parser = OutputParser(outputType="json", ObjectType=YourPydanticModel)
            parsed_output = parser.parse(output)
            print("解析后的输出：", parsed_output)
        """
        return self.parser.parse(output)
