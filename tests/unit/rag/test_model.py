import sys
import os

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

from rag_framework.prompt.template import *
from rag_framework.model.Chat import *
from langchain_core.prompts import PromptTemplate


# 创建原始模板
template = """You are a flower shop assistant.\n
For {price} of {flower_name} , can you write something for me?
"""

# 根据原始模板创建LangChain提示模板
prompt = PromptTemplate.from_template(template)




def test_ChatModelVaild():
    # 创建模型实例
    model = ChatModel("openai")
    # 输入提示
    input = prompt.format(flower_name="玫瑰", price="50")
    output = model.response(input)
    # 打印输出内容
    print(output)

test_ChatModelVaild()
