import sys
sys.path.append('/mnt/e/RAGbot')

from rag_framework.prompt.template import *
from rag_framework.model.Chat import *
from langchain_core.prompts import PromptTemplate


# 创建原始模板
template = """You are a flower shop assistant.\n
For {price} of {flower_name} , can you write something for me?
"""

# 根据原始模板创建LangChain提示模板
prompt = PromptTemplate.from_template(template)

def testModelVaild():
    # 创建模型实例
    model = createHuggingFaceChat()
    # 输入提示
    input = prompt.format(flower_name="玫瑰", price="50")
    model_kwargs = {
        'max_new_tokens': 250,  # Set to a value within the model's limits
        'temperature': 0.7,     # Adjust temperature if needed
    }
    output = model.invoke(input, **model_kwargs)


    # 打印输出内容
    print(output)

testModelVaild()
