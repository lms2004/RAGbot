""" 
本文件是【工具和工具箱：LangChain 中的 Tool 和 Toolkits 一览】章节的配套代码，课程链接：https://juejin.cn/book/7387702347436130304/section/7388071002242023462
您可以点击最上方的“运行“按钮，直接运行该文件；更多操作指引请参考Readme.md文件。
"""
# 设置OpenAI API的密钥
import os

# 导入与Gmail交互所需的工具包
from langchain_community.agent_toolkits.gmail.toolkit import GmailToolkit


# 初始化Gmail工具包
toolkit = GmailToolkit()

# 从gmail工具中导入一些有用的功能
from langchain_community.tools.gmail.utils import build_resource_service, get_gmail_credentials

# 获取 Gmail API 的凭证，并指定相关的权限范围
# 访问 Gmail 服务的代码可能无法正常运行
credentials = get_gmail_credentials(
    token_file="./token.json",  # Token文件路径
    scopes=["https://mail.google.com/"],  # 具有完全的邮件访问权限
    client_secrets_file="./credentials.json",  # 客户端的秘密文件路径
)
# 使用凭证构建API资源服务
api_resource = build_resource_service(credentials=credentials)
toolkit = GmailToolkit(api_resource=api_resource)

# 获取工具
tools = toolkit.get_tools()
print(tools)

# 导入与聊天模型相关的包
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

# 初始化聊天模型
llm = ChatOpenAI(model=os.environ.get("LLM_MODELEND"), temperature=0)


# 通过指定的工具和聊天模型初始化agent
agent = initialize_agent(
    tools=toolkit.get_tools(),
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
)

# 使用agent运行一些查询或指令
result = agent.invoke("今天易速鲜花客服给我发邮件了么？最新的邮件是谁发给我的？")

# 打印结果
print(result)
