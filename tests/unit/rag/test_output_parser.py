import sys
import os

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

from rag_framework.prompt.template import *
from rag_framework.output_parser.data_parser import *
from rag_framework.model.Chat import *
import pandas as pd


# 定义我们想要接收的数据格式
from pydantic import BaseModel, Field
class FlowerDescription(BaseModel):
    flower_type: str = Field(description="鲜花的种类")
    price: int = Field(description="鲜花的价格")
    description: str = Field(description="鲜花的描述文案")
    reason: str = Field(description="为什么要这样写这个文案")

prompt_template = """您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower} ，您能提供一个吸引人的简短中文描述吗？
{format_instructions}"""




def test_json_instructions():
    output_parser = output_parser("json", FlowerDescription)

    # 测试 get_format_instructions 方法
    print(output_parser.get_format_instructions())
    

def test_json_parse():

    df = pd.DataFrame(columns=["flower_type", "price", "description", "reason"])
    
    chat = ChatModel()

    output_parser = OutputParser("json", FlowerDescription)
    
    # 数据准备
    flowers = ["玫瑰", "百合", "康乃馨"]
    prices = ["50", "30", "20"]

    prompt_template = """您是一位专业的鲜花店文案撰写员。
    对于售价为 {price} 元的 {flower} ，您能提供一个吸引人的简短中文描述吗？
    {format_instructions}"""

    # 实例化一个 PromptCreator 对象，指定模板类型为 "cust"
    prompt = PromptCreator(prompt_type="custInstr", prompt_template=prompt_template, 
                           output_parser=output_parser)

    input_schema_names = ["price","flower"]
    # ------Part 5
    for flower, price in zip(flowers, prices):
        # 根据提示准备模型的输入
        input = prompt.get_prompt(input_schema_names, [price, flower])
        # 打印提示
        print("提示：", input)

        # 获取模型的输出
        output = chat.predict(input)

        # 打印模型的输出
        print("模型的输出：", output)

        # 解析模型的输出
        parsed_output = output_parser.parse(output)
        print(f"解析后的输出：{parsed_output}")
        parsed_output_dict = parsed_output.dict()  # 将Pydantic格式转换为字典

        # 将解析后的输出添加到DataFrame中
        df.loc[len(df)] = parsed_output.dict()

    # 打印字典
    print("输出的数据：", df.to_dict(orient="records"))




if __name__ == "__main__":
    # test_json_parse()
    # test_json_instructions()
    pass
