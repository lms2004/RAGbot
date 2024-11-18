import os
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.chains import SequentialChain
from langchain_core.runnables import RunnableLambda, RunnableSequence


from rag_framework.output_parser.data_parser import OutputParser
# from langchain_huggingface import HuggingFaceEndpoint








import os
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel




def create_openai_chat(temperature=0.8, max_tokens=100):
    """
    创建一个 OpenAI 聊天模型实例。

    参数:
        temperature (float): 控制生成文本的创造性程度，值越高，生成的文本越随机和创造性。默认值为 0.8。
        max_tokens (int): 模型生成的最大 token 数限制，默认值为 100。

    返回:
        ChatOpenAI: 配置完成的 OpenAI 聊天模型实例。
    """
    return ChatOpenAI(model=os.environ.get("LLM_MODELEND"), temperature=temperature, max_tokens=max_tokens)


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

def create_llm_chain(llm, prompt, parser=None):
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
        return create_llm_chain(self.llm, prompt, parser)
    
    def getSequentialChain(self, prompts:list, output_models:list):
        
        # introduction_chain = prompts[0] | self.llm | OutputParser("json", output_models[0]).get_parser()

        # review_chain = prompts[1] | self.llm | OutputParser("json", output_models[1]).get_parser()

        # social_post_chain = prompts[2] | self.llm | OutputParser("json", output_models[2]).get_parser()

        # # 按顺序运行三个链
        # overall_chain = introduction_chain | review_chain | social_post_chain

        # 调整指令，明确要求输出 JSON 格式
        introduction_prompt_template = """
        你是一个植物学家。给定花的名称和颜色，你需要为这种花写一个200字左右的介绍。
        花名: {name}
        颜色: {color}

        请确保以下是严格的 JSON 格式输出：
        {{
        "content": "花名和颜色的描述",
        "introduction": "关于该花的介绍"
        }}
        """

        introduction_prompt = PromptTemplate.from_template(introduction_prompt_template)
        introduction_chain = prompts[0] | self.llm | OutputParser('json', output_models[0]).get_parser()
        response = introduction_chain.invoke({"name": "玫瑰", "color": "红色"})
        print(response)

        review_prompt_template = """
        你是一位鲜花评论家。根据鲜花的介绍，你需要为这种花写一篇200字左右的评论。
        鲜花介绍:
        {introduction}

        请确保以下是严格的 JSON 格式输出：
        {{
        "introduction": "鲜花介绍",
        "review": "针对鲜花的评论"
        }}
        """
        review_prompt = PromptTemplate.from_template(review_prompt_template)
        review_chain = prompts[1] | self.llm | OutputParser('json', output_models[1]).get_parser()
        
        
        response = review_chain.invoke(response.dict())
        print(response)

        social_post_prompt_template = """
        你是一家花店的社交媒体经理。给定一种花的介绍和评论，你需要为这种花写一篇社交媒体的帖子，300字左右。
        鲜花介绍:
        {introduction}
        花评人对上述花的评论:
        {review}

        请确保以下是严格的 JSON 格式输出：
        {{
        "introduction": "鲜花介绍",
        "review": "针对鲜花的评论",
        "social_post_text": "社交媒体帖子内容"
        }}
        """
        social_post_prompt = PromptTemplate.from_template(social_post_prompt_template)

        social_post_chain = prompts[2] | self.llm | OutputParser('json', output_models[2]).get_parser()
        response = social_post_chain.invoke(response.dict())
        print(response)
        # 按顺序运行三个链
        overall_chain = introduction_chain | review_chain | social_post_chain

        return overall_chain