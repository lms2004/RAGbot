# 设置OpenAI和SERPAPI的API密钥
import os

# 加载所需的库
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI  # ChatOpenAI模型


class ChatAgent:
    def __init__(self, model_type="openai", max_tokens=400, temperature=0):
        self.model_type = model_type
        self.llm = ChatOpenAI(model=os.environ["LLM_MODELEND"], temperature=temperature, max_tokens=max_tokens)

    def getAgent(self, tools):
        tools = load_tools(tools, llm=self.llm)
        return initialize_agent(
            tools, self.llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
        )