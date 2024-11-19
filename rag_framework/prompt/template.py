import ast
import json
from typing import List
from langchain.prompts import PipelinePromptTemplate, FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
# 构建路由链
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import (
    MULTI_PROMPT_ROUTER_TEMPLATE as RounterTemplate,
)

from pydantic import BaseModel  # 使用 pydantic v2 的 BaseModel

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate)


from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma

# 初始化Embedding类
from volcenginesdkarkruntime import Ark
from typing import List, Any
from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel
import os

class DoubaoEmbeddings(BaseModel, Embeddings):
    client: Ark = None
    api_key: str = ""
    model: str

    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.api_key == "":
            self.api_key = os.environ["OPENAI_API_KEY"]
        self.client = Ark(
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=self.api_key
        )

    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.
        Args:
            texts (str): 要生成 embedding 的文本.
        Return:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表.
        """
        embeddings = self.client.embeddings.create(model=self.model, input=text)
        return embeddings.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    class Config:
        arbitrary_types_allowed = True

# 示例选择器
def createSelector(samples):
    """
    创建并初始化一个示例选择器，用于根据给定的示例数据选择最相关的示例。
    
    该函数使用语义相似性示例选择器（SemanticSimilarityExampleSelector），
    基于嵌入模型计算示例之间的相似性，选择与输入最相似的示例。
    
        参数：
            samples (list): 示例数据列表，每个示例包含输入与输出的配对。
        
        返回：
            example_selector (SemanticSimilarityExampleSelector): 一个示例选择器对象，
            用于从示例数据中根据输入查询选择最相关的示例。
    """
    
    # 初始化示例选择器
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        samples,  # 传入示例数据，包含输入输出配对
        DoubaoEmbeddings(  # 使用自定义嵌入模型生成文本的嵌入表示
            model=os.environ.get("EMBEDDING_MODELEND"),  # 获取嵌入模型的配置
        ),
        Chroma,  # 使用Chroma作为向量数据库，存储并查询嵌入向量
        k=1,  # 返回最相关的1个示例
    )
    
    return example_selector

def create_CustInstr_PromptTemplate(prompt_template, output_parser, **kwargs):
    """
    创建并初始化一个输出解析器（output_parser），用于解析模型生成的输出。
        参数:
            prompt_template (str): 包含输出解析器的提示模板。
            output_parser (output_parser): 用于解析模型输出的输出解析器对象。
        返回:
            PromptTemplate: 初始化后的 PromptTemplate 对象，用于格式化提示。
        示例：
            output_parser = output_parser("json", FlowerDescription)
            create_output_parser_PromptTemplate(prompt_template, output_parser)
    """
    # 获取 format_instructions 的内容
    format_instructions = output_parser.get_format_instructions()


    # 将单引号字符串解析为 Python 字典
    python_dict = ast.literal_eval(str(format_instructions))

    # 提取 description 字段
    description_only = {key: value['description'] for key, value in python_dict.items() if 'description' in value}

    # 转换为 JSON 格式字符串
    description_json = json.dumps(description_only, ensure_ascii=False, indent=2)
    description_json = description_json.replace("{", "{{").replace("}", "}}")
    
    # 获取输出解析器的格式说明
    filled_template = prompt_template.replace("{format_instructions}",str(description_json))

    # 创建 PromptTemplate 对象
    prompt = PromptTemplate(template=filled_template,partial_variables= {**kwargs})
    return prompt


# ChatPromptTemplate 
def createChatPromptTemplate(message_templates):
    """
    根据字符串消息模板封装为 LangChain 可以使用的形式
        参数：
            message_templates: 包含 3 个模板的列表 [sys, human, ai]，若某个模板为空，则可以传递空字符串。
            示例：
                message_templates = [
                    "You are a helpful assistant in a flower shop.",
                    "What flowers are good for a wedding?",
                    "{response}"
                ]
        返回：
            ChatPromptTemplate：封装好的聊天提示模板。
    """
    
    # 确保 message_templates 至少包含 3 个元素（可以为空字符串）
    if len(message_templates) != 3:
        raise ValueError("message_templates must contain exactly 3 templates: system, human, and AI")
    
    # 如果模板为空字符串，则使用默认值
    sysTemplate = SystemMessagePromptTemplate.from_template(message_templates[0] if message_templates[0] else "You are a helpful assistant.")
    usrTemplate = HumanMessagePromptTemplate.from_template(message_templates[1] if message_templates[1] else "Hello, how can I help you?")
    outputTemplate = AIMessagePromptTemplate.from_template(message_templates[2] if message_templates[2] else "{response}")
    
    # 使用 from_messages 构建 ChatPromptTemplate
    return ChatPromptTemplate.from_messages(
        [sysTemplate, usrTemplate, outputTemplate]
    )

# FewShotPromptTemplate 
def createFewShotPromptTemplate():
    samples = [
        {
            "flower_type": "玫瑰",
            "occasion": "爱情",
            "ad_copy": "玫瑰，浪漫的象征，是你向心爱的人表达爱意的最佳选择。",
        },
        {
            "flower_type": "康乃馨",
            "occasion": "母亲节",
            "ad_copy": "康乃馨代表着母爱的纯洁与伟大，是母亲节赠送给母亲的完美礼物。",
        },
        {
            "flower_type": "百合",
            "occasion": "庆祝",
            "ad_copy": "百合象征着纯洁与高雅，是你庆祝特殊时刻的理想选择。",
        },
        {
            "flower_type": "向日葵",
            "occasion": "鼓励",
            "ad_copy": "向日葵象征着坚韧和乐观，是你鼓励亲朋好友的最好方式。",
        },
    ]

    prompt_sample = PromptTemplate(
        input_variables=["flower_type", "occasion", "ad_copy"],
        template="鲜花类型: {flower_type}\n场合: {occasion}\n文案: {ad_copy}",
    )
    
    return FewShotPromptTemplate(
        examples=samples,
        example_prompt=prompt_sample,
        suffix="鲜花类型: {flower_type}\n场合: {occasion}",
        input_variables=["flower_type", "occasion"]
    )

def createSelcetorFewShotPromptTemplate():
    samples = [
        {
            "flower_type": "玫瑰",
            "occasion": "爱情",
            "ad_copy": "玫瑰，浪漫的象征，是你向心爱的人表达爱意的最佳选择。",
        },
        {
            "flower_type": "康乃馨",
            "occasion": "母亲节",
            "ad_copy": "康乃馨代表着母爱的纯洁与伟大，是母亲节赠送给母亲的完美礼物。",
        },
        {
            "flower_type": "百合",
            "occasion": "庆祝",
            "ad_copy": "百合象征着纯洁与高雅，是你庆祝特殊时刻的理想选择。",
        },
        {
            "flower_type": "向日葵",
            "occasion": "鼓励",
            "ad_copy": "向日葵象征着坚韧和乐观，是你鼓励亲朋好友的最好方式。",
        },
    ]

    prompt_sample = PromptTemplate(
        input_variables=["flower_type", "occasion", "ad_copy"],
        template="鲜花类型: {flower_type}\n场合: {occasion}\n文案: {ad_copy}",
    )

    # 创建一个使用示例选择器的FewShotPromptTemplate对象
    prompt = FewShotPromptTemplate(
        example_selector=createSelector(samples),
        example_prompt=prompt_sample,
        suffix="鲜花类型: {flower_type}\n场合: {occasion}",
        input_variables=["flower_type", "occasion"],
    )
    return prompt




class PromptFactory:
    """
    PromptFactory 类用于注册和创建不同类型的提示模板。
    """
    _creators = {}

    @classmethod
    def register_prompt(cls, prompt_type: str, creator):
        """
        注册提示模板类型及其创建方法。
        
        参数:
            prompt_type (str): 提示模板的类型。
            creator (callable): 负责创建该类型模板的函数。
        """
        cls._creators[prompt_type] = creator

    @classmethod
    def create_prompt(cls, prompt_type: str, **kwargs):
        """
        创建指定类型的提示模板对象。

        参数:
            prompt_type (str): 提示模板的类型。
            **kwargs: 传递给创建器的动态参数。

        返回:
            object: 创建的提示模板对象。

        异常:
            ValueError: 如果未注册 prompt_type 的创建器。
        """
        if prompt_type not in cls._creators:
            raise ValueError(f"Invalid promptType '{prompt_type}'. Supported types are: {list(cls._creators.keys())}")
        return cls._creators[prompt_type](**kwargs)


def Cust_creator(prompt_template, **kwargs):
    """
    创建自定义提示模板。
    返回:
        PromptTemplate: 创建的自定义提示模板。
    """
    return PromptTemplate(template=prompt_template, **kwargs)

def CustInstr_creator(prompt_template, output_parser, **kwargs):
    """
    创建自定义指令型提示模板。
    返回:
        PromptTemplate: 创建的自定义指令型提示模板。
    """
    return create_CustInstr_PromptTemplate(prompt_template, output_parser)

# 注册 Few-Shot Prompt 的创建逻辑
def fewshot_creator(isSelector: bool = False):
    """
    创建 Few-Shot 提示模板。

    参数:
        isSelector (bool): 是否使用选择器模式。

    返回:
        FewShotPromptTemplate: 创建的 Few-Shot 提示模板。
    """
    if isSelector:
        return createSelcetorFewShotPromptTemplate()
    return createFewShotPromptTemplate()


# 注册 Chat Prompt 的创建逻辑
def chat_creator(message_templates):
    """
    创建 Chat 提示模板。

    参数:
        message_templates (list): 包含 3 个字符串的消息模板列表 [sys, human, ai]。

    返回:
        ChatPromptTemplate: 创建的 Chat 提示模板。
    """
    return createChatPromptTemplate(message_templates)


def Router_Infos_creator(keys, descriptions, templates, **kwargs):
    """
    创建路由提示模板。
    返回:
        RouterPromptTemplate: 创建的路由提示模板。
    """
    templates_ = []
    for i in range(len(templates)):
        template = {
            "key": keys[i],
            "description": descriptions[i],
            "template": templates[i],
            **kwargs
        }
        templates_.append(template)


    return templates_

def Router_creator(prompt_infos, **kwargs):
    destinations = [f"{p['key']}: {p['description']}" for p in prompt_infos]
    router_template = RounterTemplate.format(destinations="\n".join(destinations))
    
    # 构建路由提示模板,选择不同路由下模板（选择作用的模板）

    return router_template

# 注册提示模板类型
PromptFactory.register_prompt("cust", Cust_creator)
PromptFactory.register_prompt("custInstr", CustInstr_creator)
PromptFactory.register_prompt("fewshot", fewshot_creator)
PromptFactory.register_prompt("chat", chat_creator)
PromptFactory.register_prompt("router_infos", Router_Infos_creator)
PromptFactory.register_prompt("router", Router_creator)


class PromptCreator:
    """
    PromptCreator 类用于动态生成提示模板（Prompt Template）。
    通过调用 PromptFactory 提供的工厂方法，根据给定的类型和参数创建所需的提示模板。

    Attributes:
        prompt_type (str): 提示模板的类型（如 "fewshot" 或 "chat"）。
        prompt_template (object): 创建的提示模板对象。
    """

    def __init__(self, prompt_type: str, **kwargs):
        """
        初始化 PromptCreator 实例。

        参数:
            prompt_type (str): 提示模板的类型。
                               - "fewshot": 创建 Few-Shot 提示模板。
                               - "chat": 创建 Chat 提示模板。
            **kwargs: 用于创建指定类型提示模板的附加参数。
                      - 对于 "fewshot"，可能需要 `isSelector` 参数。
                      - 对于 "chat"，需要提供 `message_templates`。

        属性:
            self.prompt_type (str): 存储指定的提示模板类型。
            self.prompt_template (object): 调用 PromptFactory 创建的提示模板对象。

        异常:
            ValueError: 如果提供的 prompt_type 未注册到 PromptFactory，将抛出异常。
        """
        self.prompt_type = prompt_type
        self.prompt_template = PromptFactory.create_prompt(prompt_type, **kwargs)

    def get_prompt_template(self):
        """
        获取生成的提示模板对象。

        返回:
            object: 当前实例中存储的提示模板对象，通常是 FewShotPromptTemplate 或 ChatPromptTemplate。

        用途:
            可用于直接查看或进一步操作生成的提示模板。
        """
        return self.prompt_template

    def get_prompt(self, input_schema_names=None, data=None):
        """
        根据输入数据生成提示。
        参数:
            input_schema_names (list): 输入数据的字段名列表。
            data (list): 输入数据，与输入模式一一对应。
        返回:
            str: 生成的提示字符串。
        """
        if self.prompt_type == "router":
            router_prompt = PromptTemplate(
                template=self.prompt_template,
                input_variables=["input"],
                output_parser=RouterOutputParser(),
            )
            return router_prompt
        
        if input_schema_names is None and data is None:
            raise ValueError("Both input_schema_names and data cannot be None.")
        
        # 检查 input_schema_names 和 data 是否都为空或都不为空
        if input_schema_names is not None and data is None:
            # 构造 input_data 字典
            input_data = {name: f"{{{name}}}" for name in input_schema_names}

            return self.prompt_template.format(**input_data)
        
        input_data = dict(zip(input_schema_names, data))
        # 使用字典解包填充模板
        return self.prompt_template.format(**input_data)

