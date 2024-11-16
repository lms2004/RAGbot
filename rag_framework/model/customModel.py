from transformers import LlamaForCausalLM, LlamaTokenizer, GPT2LMHeadModel, GPT2Tokenizer, BitsAndBytesConfig
import torch

# 保存模型和分词器的目录
save_directory_llama = "E:/RAGbot/rag_framework/model/Models/Llama-2-7b-chat-hf"
save_directory_gpt2 = "E:/RAGbot/rag_framework/model/Models/gpt2"

# 配置量化和设备映射
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

def load_model(model_type="llama"):
    """加载指定类型的模型"""
    try:
        if model_type == "llama":
            model = LlamaForCausalLM.from_pretrained(
                save_directory_llama,
                quantization_config=bnb_config,
                device_map="auto"  # 自动分配设备
            )
            tokenizer = LlamaTokenizer.from_pretrained(save_directory_llama)
            print("Llama 模型加载成功")
        elif model_type == "gpt2":
            model = GPT2LMHeadModel.from_pretrained(save_directory_gpt2)
            tokenizer = GPT2Tokenizer.from_pretrained(save_directory_gpt2)
            print("GPT-2 模型加载成功")
        else:
            raise ValueError("不支持的模型类型！请选择 'llama' 或 'gpt2'。")
        
        return model, tokenizer
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None, None

def chat_model(prompt, model_type="llama"):
    model, tokenizer = load_model(model_type)
    if not model or not tokenizer:
        return "模型加载失败，请检查日志。"

    # 将输入的 prompt 编码为模型可以理解的格式
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 使用模型生成回复
    try:
        output = model.generate(
            inputs['input_ids'],
            max_length=150,
            num_return_sequences=1,
            top_p=0.95,
            temperature=0.4,
            do_sample=True  # 开启采样，而不是贪心搜索
        )
    except Exception as e:
        print(f"生成文本失败: {e}")
        return "生成文本失败，请重试。"

    # 解码生成的输出并返回
    output_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return output_text


# 示例调用
prompt = "Can you write me a creative story?"
response_llama = chat_model(prompt, model_type="llama")
print("Llama 模型回复:", response_llama)

# 如果你想测试 GPT-2 模型，可以取消以下代码注释
# response_gpt2 = chat_model(prompt, model_type="gpt2")
# print("GPT-2 模型回复:", response_gpt2)
