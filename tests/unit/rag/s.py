import os


from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)
from langchain_openai import ChatOpenAI  # ChatOpenAI模型

# 初始化大模型
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.chains import LLMMathChain

search = SerpAPIWrapper()
llm = ChatOpenAI(model=os.environ["LLM_MODELEND"], temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    ),
]

model = ChatOpenAI(model=os.environ["LLM_MODELEND"], temperature=0)

planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

agent.run("在纽约，100美元能买几束玫瑰?")