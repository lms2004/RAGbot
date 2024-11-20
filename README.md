# RAG Framework: 基于 RAG 的问答系统

## 使用方法

1. **安装依赖**：
    在项目根目录下运行以下命令来安装所需依赖：
    ```bash
    pip install -r requirements.txt
    ```

2. **设置环境变量**：
    您需要设置以下环境变量：

    ```bash
    export EMBEDDING_MODELEND="Doubao-embedding"
    export LLM_MODELEND="Doubao-pro-32k"
    export OPENAI_BASE_URL="https://a0ai-api.zijieapi.com/api/llm/v1"
    export SERPAPI_API_KEY=<YOUR_API_KEY>
    export OPENAI_API_KEY=<YOUR_API_KEY>
    ```

    请将 `<YOUR_API_KEY>` 替换为您的实际 API 密钥。
