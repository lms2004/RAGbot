from typing import List
from langchain.prompts import PromptTemplate


def createPrompt(input_template: str, output_parser):
    """
    创建一个带有格式指示的提示模板


    参数：
        input_template (str): 用于创建提示的字符串模板，包含占位符等待填充的动态信息。
        output_parser: 一个输出解析器对象，提供格式化输出的指示。该对象需要实现`get_format_instructions()`方法，
                       该方法返回一个字符串，表示模型应当遵循的输出格式说明。
    
    返回：
        PromptTemplate: 根据输入模板和格式指示创建的提示模板。
    
    示例：
        output_parser = SomeOutputParser()
        prompt = createPrompt("生成一个关于{topic}的总结。", output_parser)
        # 该函数将返回一个包含格式指示的PromptTemplate实例，用于生成与"topic"相关的总结。
    """
    # 获取格式指示
    format_instructions = output_parser.get_format_instructions()
    
    # 根据模板创建提示，同时在模板中加入格式指示作为附加变量
    prompt = PromptTemplate.from_template(
        input_template, partial_variables={"format_instructions": format_instructions}
    )
    
    return prompt

def createChatPrompt(message_templates: List):


