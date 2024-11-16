import os
from langchain_openai import ChatOpenAI

# from langchain_huggingface import HuggingFaceEndpoint

def create_openai_chat(temperature=0.8, max_tokens=100):
    """
    创建一个 OpenAI 聊天模型实例。

    参数:
        temperature (float): 模型生成文本的创造性控制参数，默认为 0.8。
        max_tokens (int): 模型生成的最大 token 数，默认为 100。

    返回:
        ChatOpenAI: 配置好的 OpenAI 聊天模型实例。
    """
    return ChatOpenAI(model=os.environ.get("LLM_MODELEND"), temperature=temperature, max_tokens=max_tokens)


def create_huggingface_chat(repo_id="google/flan-t5-large"):
    """
    创建一个 HuggingFace 聊天模型实例。

    参数:
        repo_id (str): HuggingFace 模型仓库 ID，默认为 "google/flan-t5-large"。

    返回:
        None: 当前未实现 HuggingFace 接口。
    """
    # TODO: 替换 None 为实际实现代码
    return None

class ChatModel:
    """
    ChatModel 类用于创建和管理聊天模型实例，根据指定模型类型初始化相应接口。

    属性:
        chat (object): 由指定模型类型创建的聊天接口对象。

    方法:
        __init__(model_type="openai", max_tokens=100, temperature=0.8):
            初始化 ChatModel 类实例，根据指定的 model_type 创建对应聊天接口。
        response(input, **kwargs):
            调用聊天接口并返回生成的响应。
    """

    def __init__(self, model_type="openai", max_tokens=100, temperature=0.8):
        """
        初始化 ChatModel 类实例。

        参数:
            model_type (str): 指定模型类型，默认为 "openai"。
                              可选值为 "openai" 或 "huggingface"。
            max_tokens (int): 模型生成的最大 token 数，默认为 100。
            temperature (float): 控制生成文本的创造性程度，默认为 0.8。

        异常:
            ValueError: 当传入的 model_type 不是 "openai" 或 "huggingface" 时抛出此异常。
        """
        self.model_type = model_type
        self.max_tokens = max_tokens
        self.temperature = temperature

        if model_type == "openai":
            # 初始化 OpenAI 聊天模型接口
            self.chat = create_openai_chat(temperature=self.temperature, max_tokens=self.max_tokens)
        elif model_type == "huggingface":
            # 初始化 HuggingFace 聊天模型接口
            self.chat = create_huggingface_chat()
        else:
            # 如果传入无效的模型类型，则抛出异常
            raise ValueError("Invalid model type. Choose 'openai' or 'huggingface'.")

    def response(self, input, **kwargs):
        """
        调用聊天接口并返回响应。

        参数:
            input (str): 用户输入的文本。
            **kwargs: 可选附加参数，用于自定义聊天模型行为。

        返回:
            str: 聊天模型生成的响应。
        """
        if not self.chat:
            raise ValueError("Chat model instance is not initialized.")
        return self.chat.invoke(input, **kwargs)
