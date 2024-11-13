from typing import List
from langchain.prompts import PipelinePromptTemplate, FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate

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

# 创建具体prompt模板

# PromptTemplate
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

    return FewShotPromptTemplate(
        examples=createSelector(samples),
        example_prompt=prompt_sample,
        suffix="鲜花类型: {flower_type}\n场合: {occasion}",
        input_variables=["flower_type", "occasion"]
    )



# others
def customizedPromptTemplate(full_templates, input_prompts:list):
    """
        参数：完整提示词模板
                input_prompts 模板占位符对应prompt列表
        返回：个性化自定义prompt
    
    """
    return PipelinePromptTemplate(final_prompt = full_templates, pipeline_prompts=input_prompts)




