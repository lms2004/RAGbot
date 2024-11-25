import sys
import os

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

from rag_framework.prompt.template import *
from rag_framework.output_parser.data_parser import *
from rag_framework.model.Chat import *
from rag_framework.model.Agent.agent import *
from rag_framework.model.Agent.Tools.tools import *

from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.chains import LLMMathChain

def test_ChatAgent_getAgent():
    # 设置工具
    tool_names = ["serpapi", "llm-math"]
    tools =  ToolsCreator().create(tool_names)
    agent = ChatAgent().getAgent(tools)
    # 跑起来
    agent.invoke(
        "目前市场上玫瑰花的平均价格是多少？如果我在此基础上加价15%卖出，应该如何定价？"
    )


def test_ChatAgent_planAgent():

    tools = ToolsCreator("custom").create(["Search", "Calculator"])

    agent = ChatAgent("plan").getAgent(tools)
    agent.invoke("在纽约，100美元能买几束玫瑰?")


def test_ChatAgent_structuredAgent():
    agent = ChatAgent("structed").getAgent()

    async def main():
        response = await agent.arun("What are the headers on https://python.langchain.com?")
        print(response)


    import asyncio

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

def test_ChatAgent_selfAskWithSearch():
    tools = ToolsCreator("custom").create(["Intermediate Answer"])

    self_ask_with_search = ChatAgent("self_ask").getAgent(tools)
    self_ask_with_search.run("使用玫瑰作为国花的国家的首都是哪里?")



if __name__ == "__main__":
    # test_ChatAgent_getAgent()
    # test_ChatAgent_planAgent()
    # test_ChatAgent_structuredAgent()
    test_ChatAgent_selfAskWithSearch()
