import os
from langchain_openai import ChatOpenAI

# from langchain_huggingface import HuggingFaceEndpoint

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
