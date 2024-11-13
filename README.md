需要几个环境变量：
   export EMBEDDING_MODELEND="Doubao-embedding"
   export LLM_MODELEND="Doubao-pro-32k"  
   export OPENAI_BASE_URL="https://a0ai-api.zijieapi.com/api/llm/v1"

   export OPENAI_API_KEY=<YOUR_API_KEY>


```
RAGbot
├─ rag_framework
│  ├─ config.py
│  ├─ langchain_pipeline.py
│  ├─ llama_index_pipeline.py
│  ├─ loader.py
│  ├─ model
│  │  ├─ HuggingFace_client.py
│  │  ├─ __pycache__
│  │  │  └─ openai.cpython-310.pyc
│  │  └─ openai_client.py
│  ├─ output_parser
│  │  └─ data_parser.py
│  └─ prompt
│     └─ template.py
└─ tests
   └─ unit
      └─ rag
         └─ data_parser.py

```