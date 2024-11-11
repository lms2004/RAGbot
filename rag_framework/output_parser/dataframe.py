from pandas import DataFrame
from typing import List
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from rag_framework.prompt.template import createPrompt

def createDataFrame(input_schema_names, response_schemas, input_template, chat, data: List):
    """
    通过提示模板和模型输出处理输入数据，基于定义的响应模式解析结果并生成 DataFrame。

    参数:
        input_schema_names (List[str]): 输入数据的字段名列表。
        response_schemas (List[ResponseSchema]): 用于解析模型输出的响应模式列表。
        input_template (str): 用于生成模型输入的提示模板。
        chat (ChatOpenAI): 用于生成响应的聊天模型。
        data (List[List[Any]]): 输入数据，与输入模式一一对应。

    返回:
        DataFrame: 一个填充了输入数据和模型解析响应的 DataFrame。
    """
    # 列名：从 response_schemas 中提取字段名
    output_schema_names = [schema.name for schema in response_schemas]
    schema_names = input_schema_names + output_schema_names

    # 创建一个空的 DataFrame，列名来自 schema_names
    df = DataFrame(columns=schema_names)

    # 创建输出解析器
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    # 创建用于生成模型输入的 prompt 模板
    prompt = createPrompt(input_template, output_parser)
    print(prompt)
    
    # 遍历每一列数据
    for j in range(len(data[0])):
        input_data = {}

        # 填充 input_data 字典
        for i in range(len(input_schema_names)):
            input_data[input_schema_names[i]] = data[i][j]

        # 使用字典解包填充模板
        input_prompt = prompt.format(**input_data)  # 使用字典解包填充模板
        
        # 获取模型输出
        output = chat.predict(input_prompt)

        # 解析模型的输出（这是一个字典结构）
        parsed_output = output_parser.parse(output)

        # 在解析后的输出中添加输入数据字段
        for i in range(len(input_schema_names)):
            parsed_output[input_schema_names[i]] = data[i][j]

        # 将解析后的输出添加到 DataFrame
        df.loc[len(df)] = parsed_output

    return df
