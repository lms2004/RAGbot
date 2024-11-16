import sys
import os

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

from rag_framework.output_parser.data_parser import *

import pandas as pd


# 定义我们想要接收的数据格式
from pydantic import BaseModel, Field
class FlowerDescription(BaseModel):
    flower_type: str = Field(description="鲜花的种类")
    price: int = Field(description="鲜花的价格")
    description: str = Field(description="鲜花的描述文案")
    reason: str = Field(description="为什么要这样写这个文案")


def test_OutputParserGet_format_instructions():
    outputParser = OutputParser("json", FlowerDescription)

    # 测试 get_format_instructions 方法
    print(outputParser.get_format_instructions())

def test_OutputParserParse():
    outputParser = OutputParser("json", FlowerDescription)
    # 测试 parse 方法
    output = '{"flower_type": "玫瑰", "price": 50, "description": "玫瑰是一种美丽的花朵，它的颜色鲜艳，花朵形状优美，花朵上的花瓣也很美丽。", "reason": "玫瑰是一种象征爱情的花朵，它的颜色鲜艳，花朵形状优美，花朵上的花瓣也很美丽。"}'




test_OutputParserGet_format_instructions()