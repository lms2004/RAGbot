import sys
import os

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)


from rag_framework.prompt.template import *
from rag_framework.model.Chat import * 
from rag_framework.output_parser.data_parser import *

# CoT 的关键部分，AI 解释推理过程，并加入一些先前的对话示例（Few-Shot Learning）
system_prompt_cot = """
作为一个为花店电商公司工作的AI助手，我的目标是帮助客户根据他们的喜好做出明智的决定。 

我会按部就班的思考，先理解客户的需求，然后考虑各种鲜花的涵义，最后根据这个需求，给出我的推荐。
同时，我也会向客户解释我这样推荐的原因。

示例 1:
  人类：我想找一种象征爱情的花。
  AI：首先，我理解你正在寻找一种可以象征爱情的花。在许多文化中，红玫瑰被视为爱情的象征，这是因为它们的红色通常与热情和浓烈的感情联系在一起。因此，考虑到这一点，我会推荐红玫瑰。红玫瑰不仅能够象征爱情，同时也可以传达出强烈的感情，这是你在寻找的。

示例 2:
  人类：我想要一些独特和奇特的花。
  AI：从你的需求中，我理解你想要的是独一无二和引人注目的花朵。兰花是一种非常独特并且颜色鲜艳的花，它们在世界上的许多地方都被视为奢侈品和美的象征。因此，我建议你考虑兰花。选择兰花可以满足你对独特和奇特的要求，而且，兰花的美丽和它们所代表的力量和奢侈也可能会吸引你。
"""



# 用户的询问
human_template = "{human_input}"
ai_template = "{output}"
message_templates = [
    "You are a helpful assistant in a flower shop.",
    "What flowers are good for a wedding?",
    "{response}"
]

# 定义我们想要接收的数据格式
from pydantic import BaseModel, Field
class FlowerDescription(BaseModel):
    flower_type: str = Field(description="鲜花的种类")
    price: int = Field(description="鲜花的价格")
    description: str = Field(description="鲜花的描述文案")
    reason: str = Field(description="为什么要这样写这个文案")

def test_Cust_template():
    PromptTemplate = """您是一位专业的鲜花店文案撰写员。
    对于售价为 {price} 元的 {flower} ，您能提供一个吸引人的简短中文描述吗？
    """
    # 实例化一个 PromptCreator 对象，指定模板类型为 "cust"
    creator = PromptCreator(prompt_type="cust", prompt_template=PromptTemplate)
    prompt = creator.get_prompt_template()
    print(f"测试 模板：  {prompt}")

    input_schema_names = ["price","flower"]
    data = [100, "玫瑰"]

    print(f"测试 提示词：{creator.get_prompt(input_schema_names, data)}")

def test_CustInstr_template():
    PromptTemplate = """您是一位专业的鲜花店文案撰写员。
    对于售价为 {price} 元的 {flower} ，您能提供一个吸引人的简短中文描述吗？
    {format_instructions}"""

    output_parser = OutputParser("json", FlowerDescription)

    # 实例化一个 PromptCreator 对象，指定模板类型为 "cust"
    creator = PromptCreator(prompt_type="custInstr", prompt_template=PromptTemplate, output_parser=output_parser)
    prompt = creator.get_prompt_template()
    print(f"测试 模板：  {prompt}")
    
    input_schema_names = ["price","flower"]
    data = [100, "玫瑰"]

    print(f"测试 提示词：{creator.get_prompt(input_schema_names, data)}")

def test_chat_template():
    chat_creator = PromptCreator(prompt_type="chat", message_templates=[system_prompt_cot, human_template,"  "])
    prompt = chat_creator.get_prompt_template()
    print(f"测试 提示词模板:  {prompt}\n")

    prompt = prompt.format_prompt(
        human_input="我想为我的女朋友购买一些花。她喜欢粉色和紫色。你有什么建议吗?"
    ).to_messages() 
    llm = ChatModel("openai")
    # 接收用户的询问，返回回答结果
    response = llm.response(prompt)

    print(f"测试  模型回复：  {response}\n")

def test_fewshot_template():
    fewshot_creator = PromptCreator(prompt_type="fewshot", isSelector=False)
    fewshot_template = fewshot_creator.get_prompt_template()
    print(f"测试 模板：  {fewshot_template}")

    prompt = fewshot_creator.get_prompt_template().format_prompt(
        flower_type='黄玫瑰',
        occasion='忠贞'
    )
    print(f"测试 提示词： {prompt}")
    
    llm = ChatModel("openai")
    # 接收用户的询问，返回回答结果
    response = llm.response(prompt)

    print(f"测试  模型回复：  {response}\n")

def test_fewshot_selector_template():
    fewshot_creator = PromptCreator(prompt_type="fewshot", isSelector=True)
    fewshot_template = fewshot_creator.get_prompt_template()
    print(f"测试 模板：  {fewshot_template}")

    prompt = fewshot_creator.get_prompt_template().format_prompt(
        flower_type='黄玫瑰',
        occasion='忠贞'
    )
    
    print(f"测试 提示词： {prompt}")
    
    llm = ChatModel("openai")
    # 接收用户的询问，返回回答结果
    response = llm.response(prompt)
    print(f"测试  模型回复：  {response}\n")


if __name__ == "__main__":
    test_Cust_template()
    test_CustInstr_template()
    # test_chat_template()
    # test_fewshot_template()
    # test_fewshot_selector_template()