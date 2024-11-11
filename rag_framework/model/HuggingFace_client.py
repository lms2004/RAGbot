from langchain_huggingface import HuggingFaceEndpoint
import os

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

# 创建模型实例
model = HuggingFaceEndpoint(repo_id="google/flan-t5-large")