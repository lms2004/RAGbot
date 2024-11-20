import os
from typing import Tuple
from langchain_core.prompts import PromptTemplate


from langchain.chains.router.llm_router import LLMRouterChain

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
)

from langchain.chains.llm import LLMChain
# 构建多提示链
from langchain.chains.router import MultiPromptChain

# from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
import tiktoken

from rag_framework.output_parser.data_parser import *
from rag_framework.prompt.template import PromptCreator



class ChatOpenAIIn05(ChatOpenAI):
    def _get_encoding_model(self) -> Tuple[str, tiktoken.Encoding]:
        """
        Override the method to return a hardcoded valid model and its encoding.
        """
        # Set the model to a valid one to avoid errors
        model = "gpt-3.5-turbo"
        return model, tiktoken.encoding_for_model(model)



def create_openai_chat(temperature=0.8, max_tokens=100):
    """
    创建一个 OpenAI 聊天模型实例。

    参数:
        temperature (float): 控制生成文本的创造性程度，值越高，生成的文本越随机和创造性。默认值为 0.8。
        max_tokens (int): 模型生成的最大 token 数限制，默认值为 100。

    返回:
        ChatOpenAIIn05: 配置完成的 OpenAI 聊天模型实例。
    """
    return ChatOpenAIIn05(model=os.environ.get("LLM_MODELEND"), temperature=temperature, max_tokens=max_tokens)


def create_huggingface_chat(repo_id="google/flan-t5-large"):
    """
    创建一个 HuggingFace 聊天模型实例。

    参数:
        repo_id (str): HuggingFace 模型仓库的标识符，默认值为 "google/flan-t5-large"。

    返回:
        None: 当前未实现 HuggingFace 聊天模型接口，需补充实现。
    """
    # TODO: 替换 None 为 HuggingFace 模型接口的实现。
    return None

def create_llm_chain(prompt, llm, parser=None):
    """
    创建一个基于 LLMChain 的聊天模型链。
        参数:
            prompt (PromptTemplate): 用于生成聊天模型输入的提示模板。
        返回:
            LLMChain: 配置完成的 LLMChain 实例。
        方法：
            invoke(input, **kwargs):
                调用聊天接口并返回生成的响应。
                参数:
                    input (str): 占位符字典，用于填充提示模板。
                    **kwargs: 附加参数，用于定制聊天模型的行为。
                返回:
                    object: 聊天模型生成的完整响应对象。
    """
    if parser is None:
        return prompt | llm
    return prompt | llm | parser


class ChatModel:
    """
    ChatModel 类用于封装聊天模型接口的创建和管理，支持 OpenAI 和 HuggingFace 模型。

    属性:
        chat (object): 聊天模型接口实例，由指定的模型类型创建。

    方法:
        __init__(model_type="openai", max_tokens=100, temperature=0.8):
            初始化 ChatModel 实例，根据指定的 model_type 创建相应的聊天接口。
        response(input, **kwargs):
            调用聊天接口并返回生成的响应。
        predict(input, **kwargs):
            调用聊天接口并生成响应文本，返回模型生成的主要内容。
    """

    def __init__(self, model_type="openai", max_tokens=100, temperature=0.8):
        """
        初始化 ChatModel 类实例。

        参数:
            model_type (str): 指定模型类型，支持 "openai" 和 "huggingface"，默认值为 "openai"。
            max_tokens (int): 限制模型生成文本的最大 token 数，默认值为 100。
            temperature (float): 控制生成文本的创造性程度，默认值为 0.8。

        异常:
            ValueError: 当传入的 model_type 不是 "openai" 或 "huggingface" 时抛出异常。
        """
        self.model_type = model_type
        self.max_tokens = max_tokens
        self.temperature = temperature

        if model_type == "openai":
            # 创建 OpenAI 聊天模型接口
            self.chat = create_openai_chat(temperature=self.temperature, max_tokens=self.max_tokens)
        elif model_type == "huggingface":
            # 创建 HuggingFace 聊天模型接口
            self.chat = create_huggingface_chat()
        else:
            # 如果传入无效的模型类型，抛出异常
            raise ValueError("Invalid model type. Please choose 'openai' or 'huggingface'.")

    def response(self, input, **kwargs):
        """
        调用聊天接口并返回响应对象。

        参数:
            input (str): 用户输入的文本。
            **kwargs: 附加参数，用于定制聊天模型的行为。

        返回:
            object: 聊天模型生成的完整响应对象。

        异常:
            ValueError: 如果聊天模型实例未初始化，则抛出异常。
        """
        if not self.chat:
            raise ValueError("Chat model instance is not initialized.")
        return self.chat.invoke(input, **kwargs)

    def predict(self, input, **kwargs):
        """
        调用聊天接口并返回生成的响应内容。

        参数:
            input (str): 用户输入的文本。
            **kwargs: 附加参数，用于定制聊天模型的行为。

        返回:
            str: 聊天模型生成的响应内容。如果模型未返回 'content' 属性，则返回空字符串。

        异常:
            ValueError: 如果聊天模型实例未初始化，则抛出异常。
        """
        if not self.chat:
            raise ValueError("Chat model instance is not initialized.")
        response = self.chat.invoke(input, **kwargs)
        return getattr(response, "content", "")


class ChatModelChain:
    """
    Chat_Model_Chain 类用于封装聊天模型接口的创建和管理，支持 OpenAI 和 HuggingFace 模型。
        属性:
            llm (object): 聊天模型接口实例，由指定的模型类型创建。
        方法:
            __init__(model_type="openai", max_tokens=100, temperature=0.8):
                初始化 Chat_Model_Chain 实例，根据指定的 model_type 创建相应的聊天接口。
            getSingleChain(prompt):
                创建一个基于 LLMChain 的聊天模型链。
                参数:
                    prompt (PromptTemplate): 用于生成聊天模型输入的提示模板。
                返回:
                    LLMChain: 配置完成的 LLMChain 实例。
            getSequentialChain(prompts,chains):
                创建一个基于 SequentialChain 的聊天模型链。
                参数:
                    prompts (list): 用于生成聊天模型输入的提示模板列表。
                    chains (list): 用于生成聊天模型输入的提示模板列表。
                返回:
                    SequentialChain: 配置完成的 SequentialChain 实例。
    """
    def __init__(self, model_type="openai", max_tokens=100, temperature=0.8):
        """
        初始化 Chat_Model_Chain 类实例。
        参数:
            model_type (str): 指定模型类型，支持 "openai" 和 "huggingface"，默认值为 "openai"。
            max_tokens (int): 限制模型生成文本的最大 token 数，默认值为 100。
            temperature (float): 控制生成文本的创造性
        """
        self.model_type = model_type
        self.max_tokens = max_tokens
        self.temperature = temperature
        if model_type == "openai":
            # 创建 OpenAI 聊天模型链接口
            self.llm = create_openai_chat(temperature=self.temperature, max_tokens=self.max_tokens)
        elif model_type == "huggingface":
            # 创建 HuggingFace 聊天模型链接口
            self.llm = create_huggingface_chat()
        else:
            # 如果传入无效的模型类型，抛出异常
            raise ValueError("Invalid model type. Please choose 'openai' or 'huggingface'.")
   
    
    def getSingleChain(self, prompt, parser=None):
        """
        创建一个基于 LLMChain 的聊天模型链。
            参数:
                prompt (PromptTemplate): 用于生成聊天模型输入的提示模板。
            返回:
                LLMChain: 配置完成的 LLMChain 实例。
        """
        return create_llm_chain(prompt, self.llm, parser)


    def getSequentialChain(self, prompts:list, output_models:list):
        """
        创建一个基于 SequentialChain 的聊天模型链。
            参数:
                prompts (list): 用于生成聊天模型输入的提示模板列表。
                output_models (list): 用于生成聊天模型输入的提示模板列表。
            返回:
                SequentialChain: 配置完成的 SequentialChain 实例。
        """
        # 确保 prompts 和 output_models 的数量一致
        if len(prompts) != len(output_models):
            raise ValueError("The length of prompts and output_models must match!")

        # 初始化第一个链
        current_chain = prompts[0] | self.llm | OutputParser('json', output_models[0]).get_parser()

        # 动态连接其余的链
        for i in range(1, len(prompts)):
            next_chain = prompts[i] | self.llm | OutputParser('json', output_models[i]).get_parser()
            current_chain = current_chain | next_chain


        return current_chain


    def getChainsMap(self, prompt_infos):
        """
        创建一个基于 ChainMap 的聊天模型链。
            参数:
                chains (list): 用于生成聊天模型输入的提示模板列表。
            返回:
                ChainMap: 配置完成的 ChainMap 实例。
        """
        chain_map = {}

        for info in prompt_infos:
            prompt = PromptTemplate(template=info["template"], input_variables=["input"])

            chain = LLMChain(llm=self.llm, prompt=prompt, verbose=True)
            chain_map[info["key"]] = chain
        return chain_map


    def getRouterChain(self, router_prompt):

        """
        创建一个基于 LLMRouterChain 的聊天模型链。
            参数:
                router_prompt (PromptTemplate): 路由提示词
            返回:
                LLMRouterChain: 配置完成的 LLMRouterChain 实例。
        """

        router_chain = LLMRouterChain.from_llm(self.llm, router_prompt, verbose=True)
        return router_chain
        

    def getDefaultChain(self):
        default_chain = ConversationChain(llm=self.llm, output_key="text", verbose=True)
        return default_chain
    

    def getMutipleRouterChain(self, keys, descriptions, templates):
        """
        创建一个基于 MultiPromptChain 的聊天模型链。
            参数:
                keys (list): 用于生成聊天模型输入的提示模板列表。
                descriptions (list): 用于生成聊天模型输入的提示模板列表。
                templates (list): 用于生成聊天模型输入的提示模板列表。
            返回:
                MultiPromptChain: 配置完成的 MultiPromptChain 实例。
        """

        Router_infos = PromptCreator(
            prompt_type="router_infos",
            keys=keys,
            descriptions=descriptions,
            templates=templates
        ).get_prompt_template()
        
        chain_map = ChatModelChain().getChainsMap(Router_infos)

        Router_creator = PromptCreator(
            prompt_type="router",
            prompt_infos=Router_infos
        )

        router_prompt = Router_creator.get_prompt()
        router_chain = ChatModelChain().getRouterChain(router_prompt)

        default_chain = ChatModelChain().getDefaultChain()

        chain = MultiPromptChain(
            router_chain=router_chain,
            destination_chains=chain_map,
            default_chain=default_chain,
            verbose=True,
        )
        return chain

    def getConversationChain(self):
        return ConversationChain(llm=self.llm, output_key="text", verbose=True)

    def getConservationChain(self, memory_type=None, **kwargs):
        if memory_type is None:
            return self.getDefaultChain()

        if memory_type == "buffer":
            return ConversationChain(llm=self.llm, output_key="text", verbose=True, memory=ConversationBufferMemory())

        elif memory_type == "window":
            return ConversationChain(llm=self.llm, output_key="text", verbose=True, memory=ConversationBufferWindowMemory(**kwargs))

        elif memory_type == "summary":
            return ConversationChain(llm=self.llm, output_key="text", verbose=True, memory=ConversationSummaryMemory(llm=self.llm))

        elif memory_type == "summary_buffer":
            return ConversationChain(llm=self.llm, output_key="text", verbose=True, memory=ConversationSummaryBufferMemory(llm=self.llm , **kwargs))
        else:
            raise ValueError("Invalid memory type. Please choose 'buffer', 'buffer_window', 'summary', 'summary_buffer'.")















