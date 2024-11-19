import sys
import os

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

from rag_framework.prompt.template import *
from rag_framework.output_parser.data_parser import *
from rag_framework.model.Chat import *


prompt_template = """您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower} ，您能提供一个吸引人的简短中文描述吗？
{format_instructions}"""

from pydantic import BaseModel, Field
class FlowerDescription(BaseModel):
    flower_type: str = Field(description="鲜花的种类")
    price: int = Field(description="鲜花的价格")
    description: str = Field(description="鲜花的描述文案")
    reason: str = Field(description="为什么要这样写这个文案")


def test_ChatModel_response():
    # 创建模型实例
    model = ChatModel("openai")

    output_parser = OutputParser("json", FlowerDescription)

    # 实例化一个 PromptCreator 对象，指定模板类型为 "cust"
    prompt = PromptCreator(prompt_type="custInstr", prompt_template=prompt_template, 
                           output_parser=output_parser)
    
    input_schema_names = ["price","flower"]

    input = prompt.get_prompt(input_schema_names, ["50", "玫瑰"])

    # 打印提示
    print("提示：", input)
    output = model.response(input)
    # 打印输出内容
    print(f"response: {output}")

def test_ChatModel_predict():
    # 创建模型实例
    model = ChatModel("openai")
    
    output_parser = OutputParser("json", FlowerDescription)

    # 实例化一个 PromptCreator 对象，指定模板类型为 "cust"
    prompt = PromptCreator(prompt_type="custInstr", prompt_template=prompt_template, 
                           output_parser=output_parser)
    
    input_schema_names = ["price","flower"]

    input = prompt.get_prompt(input_schema_names, ["50", "玫瑰"])

    # 打印提示
    print("提示：", input)
    output = model.predict(input)
    # 打印输出内容
    print(f"predict: {output}")

def test_ChatModelChain_getSingleChain():
    chat_model_chain = ChatModelChain()

    prompt_template = """
    你是一位专业的鲜花店文案撰写员。
    根据以下信息，请生成关于鲜花的吸引人且简短的中文描述，并附带花语信息：
    - 花名: {flower}
    - 季节: {season}

    请输出结果，并确保格式符合以下说明：
    {format_instructions}
    """
    

    class Flower(BaseModel):
        flower_type: str = Field(description="鲜花的种类")
        FlowerDescription: str = Field(description="（请根据花的种类和对应的季节）写出花语")

    output_parser = OutputParser("json", Flower)
    
    # 创建模型实例
    prompt = PromptCreator("custInstr",
                           prompt_template=prompt_template,
                           output_parser=output_parser).get_prompt_template()
    
    chat_chain = chat_model_chain.getSingleChain(prompt, output_parser.get_parser())

    response = chat_chain.invoke({"flower": "玫瑰", "season": "夏季"})
    print(response)



template1 = """
你是一个植物学家。给定花的名称和类型，你需要为这种花写一个200字左右的介绍。
花名: {name}
颜色: {color}

请确保以下是严格的 JSON 格式输出：
{format_instructions}
"""

# 第二个LLMChain：根据鲜花的介绍写出鲜花的评论
template2 = """
你是一位鲜花评论家。给定一种花的介绍，你需要为这种花写一篇200字左右的评论。
鲜花介绍:
{introduction}


请确保以下是严格的 JSON 格式输出：
{format_instructions}
"""

# 第三个LLMChain：根据鲜花的介绍和评论写出一篇自媒体的文案
template3 = """
你是一家花店的社交媒体经理。给定一种花的介绍和评论，你需要为这种花写一篇社交媒体的帖子，300字左右。
鲜花介绍:
{introduction}
花评人对上述花的评论:
{review}


请确保以下是严格的 JSON 格式输出：
{format_instructions}
"""

def test_ChatModelChain_getSequentialChain():
    chat_model_chain = ChatModelChain(max_tokens=400)


    # 输出模型：定义第一个链的输出
    class IntroductionOutput(BaseModel):
        introduction: str = Field(description="关于该花的介绍")

    # 输出模型：定义第二个链的输出
    class ReviewOutput(BaseModel):
        introduction: str = Field(description="从第一个链传递的鲜花介绍")
        review: str = Field(description="根据鲜花介绍撰写的评论内容")

    # 输出模型：定义第三个链的输出
    class SocialPostOutput(BaseModel):
        introduction: str = Field(description="从第一个链传递的鲜花介绍")
        review: str = Field(description="从第二个链传递的评论内容")
        social_post_text: str = Field(description="根据介绍和评论生成的社交媒体文案")


    output_models = [IntroductionOutput, ReviewOutput, SocialPostOutput]

    input_schema_names = ["name","color"]

    introduction   =    PromptCreator("custInstr", 
                                        prompt_template=template1
                                      , output_parser=OutputParser("json", IntroductionOutput)).get_prompt_template()

    input_schema_names = ["introduction"]
    review         =    PromptCreator("custInstr",
                                     prompt_template=template2
                                    , output_parser=OutputParser("json", ReviewOutput)).get_prompt_template()
   
    input_schema_names = ["introduction", "review"]
    social_media   =    PromptCreator("custInstr",
                                     prompt_template=template3
                                  , output_parser=OutputParser("json", SocialPostOutput)).get_prompt_template()


    prompts = [introduction, review, social_media]

    chains = chat_model_chain.getSequentialChain(prompts=prompts, 
                                                 output_models=output_models)


    # 将模型转为字典供链调用
    result = chains.invoke({"name": "玫瑰", "color": "黑色"})

    # 输出结果
    print(result)


def test_ChatModelChain_getChainMap():
    # 构建两个场景的模板
    flower_care_template = """
    你是一个经验丰富的园丁，擅长解答关于养花育花的问题。
    下面是需要你来回答的问题:
    {input}
    """

    flower_deco_template = """
    你是一位网红插花大师，擅长解答关于鲜花装饰的问题。
    下面是需要你来回答的问题:
    {input}
    """

    templates = [flower_care_template, flower_deco_template]
    keys = ["flower_care", "flower_decoration"]
    descriptions = ["适合回答关于鲜花护理的问题", "适合回答关于鲜花装饰的问题"]

    prompt_infos = PromptCreator(
        prompt_type="router_infos", 
        keys=keys, 
        descriptions=descriptions,
        templates=templates
    ).get_prompt_template()

    chain_map = ChatModelChain().getChainsMap(prompt_infos)


def test_ChatModelChain_getRouterChain():
    # 构建两个场景的模板
    flower_care_template = """
    你是一个经验丰富的园丁，擅长解答关于养花育花的问题。
    下面是需要你来回答的问题:
    {input}
    """

    flower_deco_template = """
    你是一位网红插花大师，擅长解答关于鲜花装饰的问题。
    下面是需要你来回答的问题:
    {input}
    """

    templates = [flower_care_template, flower_deco_template]
    keys = ["flower_care", "flower_decoration"]
    descriptions = ["适合回答关于鲜花护理的问题", "适合回答关于鲜花装饰的问题"]

    Router_infos = PromptCreator(
        prompt_type="router_infos",
        keys=keys,
        descriptions=descriptions,
        templates=templates
    ).get_prompt_template()
    Router_creator_prompt = PromptCreator(
        prompt_type="router",
        prompt_infos=Router_infos
    ).get_prompt()


    Router_Chain = ChatModelChain().getRouterChain(Router_creator_prompt)


def test_ChatModelChain_getDefaultChain():
    default_chain = ChatModelChain().getDefaultChain()


def test_ChatModelChain_getMutipleRouterChain():
    import warnings
    import os
    warnings.filterwarnings("ignore")

    # 设置OpenAI API密钥

    # 构建两个场景的模板
    flower_care_template = """
    你是一个经验丰富的园丁，擅长解答关于养花育花的问题。
    下面是需要你来回答的问题:
    {input}
    """

    flower_deco_template = """
    你是一位网红插花大师，擅长解答关于鲜花装饰的问题。
    下面是需要你来回答的问题:
    {input}
    """

    templates = [flower_care_template, flower_deco_template]
    keys = ["flower_care", "flower_decoration"]
    descriptions = ["适合回答关于鲜花护理的问题", "适合回答关于鲜花装饰的问题"]

    chain = ChatModelChain().getMutipleRouterChain(
        keys=keys,
        descriptions=descriptions,
        templates=templates
    )

    # 测试1
    print(chain.invoke("如何为玫瑰浇水？"))
    # 测试2
    print(chain.invoke("如何为婚礼场地装饰花朵？"))
    # 测试3
    print(chain.invoke("如何区分阿豆和罗豆？"))


if __name__ == "__main__":
    # test_ChatModelChain_getSingleChain()
    # test_ChatModelChain_getSequentialChain()
    # test_ChatModelChain_getChainMap()
    # test_ChatModelChain_getRouterChain()
    # test_ChatModelChain_getDefaultChain()
    test_ChatModelChain_getMutipleRouterChain()

