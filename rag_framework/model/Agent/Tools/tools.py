from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.chains import LLMMathChain

# 设置OpenAI和SERPAPI的API密钥
import os

# 加载所需的库
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI  # ChatOpenAI模型

from langchain_community.agent_toolkits.playwright.toolkit import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser



class ToolsFactory:
    _tools = {}
    @classmethod
    def registerTool(cls, tool_type, tool_creator):
        cls._tools[tool_type] = tool_creator

    @classmethod
    def createTool(cls, tool_type, **kwargs):
        return cls._tools[tool_type](**kwargs)

def defaultCreator(llm, tools_names):
    return load_tools(tools_names, llm=llm)

def customCreator(llm, tools_names):
    tools = []
    for tool in tools_names:
        if tool == "Search" or tool == "Intermediate Answer":
            search = SerpAPIWrapper()
            tools.append(Tool(
                    name=tool,
                    func=search.run,
                    description="useful for when you need to answer questions about current events",
                ))
        elif tool == "Calculator":
            llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
            tools.append(Tool(
                    name="Calculator",
                    func=llm_math_chain.run,
                    description="useful for when you need to answer questions about math",
                ))
        else:
            raise Exception("Unsupported tool type")
    return tools


ToolsFactory.registerTool("default", defaultCreator)
ToolsFactory.registerTool("custom", customCreator)


class ToolsCreator:
    """
        参数：
            tool_type (str): 工具类型。
            model_type (str): 模型类型。
            max_tokens (int): 最大令牌数。
            temperature (int): 模型不确定性的度量。
    """
    def __init__(self, tool_type="default", model_type="openai", max_tokens=400, temperature=0):
        self.model_type = model_type
        self.llm = ChatOpenAI(model=os.environ["LLM_MODELEND"], temperature=temperature, max_tokens=max_tokens)
        self.tooltype = tool_type
        self.tools = []

    def create(self, tools_names):
        """
        根据工具名称列表创建工具。
        
        参数：
        tools_names (list): 工具名称列表。
        
        返回：
        list: 创建的工具列表。
        """
        return ToolsFactory.createTool(self.tooltype, llm=self.llm, tools_names=tools_names)