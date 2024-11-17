import sys
import os

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

from rag_framework.prompt.template import *
from rag_framework.output_parser.data_parser import *
from rag_framework.model.Chat import *


prompt_template = """您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower} ，您能提供一个吸引人的简短中文描述吗？
{format_instructions}"""

from pydantic import BaseModel, Field
class FlowerDescription(BaseModel):
    flower_type: str = Field(description="鲜花的种类")
    price: int = Field(description="鲜花的价格")
    description: str = Field(description="鲜花的描述文案")
    reason: str = Field(description="为什么要这样写这个文案")


def test_ChatModel_response():
    # 创建模型实例
    model = ChatModel("openai")

    output_parser = OutputParser("json", FlowerDescription)

    # 实例化一个 PromptCreator 对象，指定模板类型为 "cust"
    prompt = PromptCreator(prompt_type="custInstr", prompt_template=prompt_template, 
                           output_parser=output_parser)
    
    input_schema_names = ["price","flower"]

    input = prompt.get_prompt(input_schema_names, ["50", "玫瑰"])

    # 打印提示
    print("提示：", input)
    output = model.response(input)
    # 打印输出内容
    print(f"response: {output}")

def test_ChatModel_predict():
    # 创建模型实例
    model = ChatModel("openai")
    
    output_parser = OutputParser("json", FlowerDescription)

    # 实例化一个 PromptCreator 对象，指定模板类型为 "cust"
    prompt = PromptCreator(prompt_type="custInstr", prompt_template=prompt_template, 
                           output_parser=output_parser)
    
    input_schema_names = ["price","flower"]

    input = prompt.get_prompt(input_schema_names, ["50", "玫瑰"])

    # 打印提示
    print("提示：", input)
    output = model.predict(input)
    # 打印输出内容
    print(f"predict: {output}")

