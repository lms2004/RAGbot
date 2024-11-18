import json

# 定义 input_variables
input_variables = ["'flower_type'", 'flower', 'price']

# 定义动态的输出格式说明
output_format = {
    "flower_type": {
        "description": "鲜花的种类",
        "title": "Flower Type",
        "type": "string"
    },
    "price": {
        "description": "鲜花的价格",
        "title": "Price",
        "type": "integer"
    },
    "description": {
        "description": "鲜花的描述文案",
        "title": "Description",
        "type": "string"
    },
    "reason": {
        "description": "为什么要这样写这个文案",
        "title": "Reason",
        "type": "string"
    }
}

# 将输出格式说明转换为 JSON 字符串
format_instructions = json.dumps(output_format, ensure_ascii=False, indent=2)

# 动态构造模板
template = f"""
您是一位专业的鲜花店文案撰写员。
对于售价为 {{price}} 元的 {{flower}} ，您能提供一个吸引人的简短中文描述吗？

请确保以下是严格的 JSON 格式输出：
{{
{format_instructions}
}}
"""

# 打印结果
print("Input Variables:", input_variables)
print("Template:\n", template)
