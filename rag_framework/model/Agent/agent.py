# 设置OpenAI和SERPAPI的API密钥
import os

# 加载所需的库
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI  # ChatOpenAI模型

from langchain_community.agent_toolkits.playwright.toolkit import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser

from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)

class AgentFactory:
    _agents = {}

    @classmethod
    def resgiterAgent(cls, agentType ,agentCreator):
        cls._agents[agentType] = agentCreator
    
    @classmethod
    def createAgent(cls, agent_type, **kwargs):
        return cls._agents[agent_type](**kwargs)


def defaultCreator(llm, tools):
    return initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

def planCreator(llm, tools):
    model = llm

    planner = load_chat_planner(model)
    executor = load_agent_executor(model, tools, verbose=True)

    return PlanAndExecute(planner=planner, executor=executor, verbose=True)

def structedCreator(llm):
    async_browser = create_async_playwright_browser()
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    tools = toolkit.get_tools()

    return  initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

def selfAskCreator(llm, tools):
    return initialize_agent(tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)


    
AgentFactory.resgiterAgent("default",defaultCreator)
AgentFactory.resgiterAgent("plan",planCreator)
AgentFactory.resgiterAgent("structed",structedCreator)
AgentFactory.resgiterAgent("self_ask",selfAskCreator)




class ChatAgent:
    """"
    参数:
    
        agent_type: 代理类型，目前支持default和plan两种类型。

        model_type: 模型类型，目前支持openai和llm-math两种类型。

        max_tokens: 模型生成的最大token数，默认为400。

        temperature: 模型生成的温度，默认为0。

    返回：
        agent: 代理对象。
    
    """
    def __init__(self, agent_type = "default", model_type="openai", max_tokens=400, temperature=0):
        self.model_type = model_type
        self.llm = ChatOpenAI(model=os.environ["LLM_MODELEND"], temperature=temperature, max_tokens=max_tokens)
        self.agent_type = agent_type

    def getAgent(self, tools=None):
        if tools is None:
            return AgentFactory.createAgent(agent_type=self.agent_type, llm=self.llm)
            
        return AgentFactory.createAgent(agent_type=self.agent_type, llm=self.llm, tools=tools)