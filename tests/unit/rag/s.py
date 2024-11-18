# # # from langchain.chains import SequentialChain, LLMChain
# # # from langchain.prompts import PromptTemplate
# # # from langchain_openai import ChatOpenAI
# # # import os
# # # # 定义第一个链：生成鲜花的介绍
# # # llm = ChatOpenAI(model=os.environ.get("LLM_MODELEND"), temperature=0.7)
# # # template_intro = PromptTemplate(
# # #     input_variables=["name", "color"],
# # #     template="你是植物学家。描述一种叫做{name}，颜色为{color}的花。"
# # # )
# # # introduction_chain = template_intro | llm

# # # # 定义第二个链：生成鲜花评论
# # # template_review = PromptTemplate(
# # #     input_variables=["introduction"],
# # #     template="你是一位花评人。基于以下介绍写一篇评论：{introduction}"
# # # )
# # # review_chain = template_review | llm

# # # # 定义第三个链：生成社交媒体文案
# # # template_social = PromptTemplate(
# # #     input_variables=["introduction", "review"],
# # #     template="根据介绍和评论生成一篇社交媒体帖子：\n介绍: {introduction}\n评论: {review}"
# # # )
# # # social_post_chain = template_social | llm

# # # # 创建SequentialChain
# # # overall_chain = introduction_chain | review_chain | social_post_chain

# # # # 运行链
# # # result = overall_chain.invoke({"name": "玫瑰", "color": "红色"})
# # # print(result)


# from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
# from openai import BaseModel
# import os
# class M(BaseModel):
#     content: str
#     summary: str
# parser = JsonOutputParser(pydantic_object=M)

# llm = ChatOpenAI(model=os.environ.get("LLM_MODELEND"), temperature=0.7)

# summarizing_prompt_template = """
# 总结以下文本为一个 20 字以内的句子，输出格式为原始 json，json 里面有字段 content、summary:
# ---
# {content}
# """
# prompt = PromptTemplate.from_template(summarizing_prompt_template)
# summarizing_chain = prompt | llm | parser

# translating_prompt_template = """
# 将{summary}翻译成英文，输出格式为原始 json，json 里面有字段 content、summary、translated，其中 content 的值为 {content}，summary 的值为 {summary}，translated 的值为翻译后的英文句子。:
# """
# prompt = PromptTemplate.from_template(translating_prompt_template)
# translating_chain = prompt | llm | StrOutputParser()

# overall_chain = summarizing_chain | translating_chain

# print(overall_chain)
# print(overall_chain.invoke({"content": "这是一个测试。"}))


"""
本文件是【链（上）：写一篇完美鲜花推文？用SequentialChain链接不同的组件】章节的配套代码。
"""
import os
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel





# 定义数据模型，用于 JSON 格式校验
class IntroductionModel(BaseModel):
    content: str
    introduction: str


class ReviewModel(BaseModel):
    introduction: str
    review: str


class SocialPostModel(BaseModel):
    introduction: str
    review: str
    social_post_text: str


# 配置 OpenAI 模型
llm = ChatOpenAI(
    temperature=0.7,
    model=os.environ.get("LLM_MODELEND", "gpt-3.5-turbo"),
)

# 调整指令，明确要求输出 JSON 格式
introduction_prompt_template = """


请确保以下是严格的 JSON 格式输出：
{{
  "content": "花名和颜色的描述",
  "introduction": "关于该花的介绍"
}}
"""




introduction_prompt = PromptTemplate.from_template(introduction_prompt_template)

print(introduction_prompt)


introduction_parser = JsonOutputParser(pydantic_object=IntroductionModel)
introduction_chain = introduction_prompt | llm | introduction_parser

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
review_parser = JsonOutputParser(pydantic_object=ReviewModel)
review_chain = review_prompt | llm | review_parser

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

print(social_post_prompt)
social_post_parser = JsonOutputParser(pydantic_object=SocialPostModel)
social_post_chain = social_post_prompt | llm | social_post_parser

# 按顺序运行三个链
overall_chain = introduction_chain | review_chain | social_post_chain

# 测试数据
input_data = {"name": "玫瑰", "color": "黑色"}

# 执行链
try:
    result = overall_chain.invoke(input_data)
    print("Final Result:", result)
except Exception as e:
    print(f"Error occurred: {e}")
