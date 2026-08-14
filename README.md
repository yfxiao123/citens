# 文献综述智能体 (Literature Review Agent)

课程大作业项目：构建一个文献综述智能体，输入研究主题，自动完成论文检索、筛选、信息抽取、主题组织和综述撰写全流程。

## 功能

- **论文检索**：自动生成多组关键词，并发检索 arXiv + Semantic Scholar + OpenAlex
- **论文筛选**：LLM 逐篇相关性评分，保留高相关论文
- **信息抽取**：从摘要中提取研究问题、方法、主要发现、局限性等信息
- **主题组织**：自动识别 3-6 个主题维度，归类论文并梳理逻辑关系
- **综述撰写**：生成带引用的完整综述 Markdown 文件
- **双交互方式**：CLI 命令行 + Gradio Web 界面

## 项目结构

```
├── app.py                  # CLI 入口
├── webui.py                # Gradio Web 界面
├── src/
│   ├── config.py           # 配置管理
│   ├── models.py           # Pydantic 数据模型
│   ├── llm.py              # LLM 统一调用（OpenAI / Ollama）
│   ├── pipeline.py         # 主流程编排
│   ├── search/
│   │   ├── arxiv.py        # arXiv API 检索
│   │   ├── semantic_scholar.py  # Semantic Scholar API 检索
│   │   ├── openalex.py     # OpenAlex API 检索
│   │   └── base.py         # 三源并发检索 + 去重
│   └── agents/
│       ├── keyword_agent.py    # 关键词生成
│       ├── filter_agent.py     # 论文筛选
│       ├── extract_agent.py    # 信息抽取
│       ├── organize_agent.py   # 主题组织
│       └── write_agent.py      # 综述撰写
├── output/                 # 运行输出目录（自动生成）
├── requirements.txt
└── .env.example
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填写配置：

```bash
# 使用 OpenAI（默认）
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini

# 或使用 Ollama 本地模型
# LLM_PROVIDER=ollama
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=qwen2.5:7b
```

### 3. 运行

**CLI 模式**：
```bash
python app.py 大语言模型在金融领域的应用
```

**Web 界面**：
```bash
python webui.py
```

运行结果保存在 `output/` 目录下。

## Pipeline 流程

```
用户输入主题 → 关键词生成(6-8条) → 三源检索(arXiv+S2+OpenAlex)
  → LLM筛选(保留≥15篇) → 信息抽取(5个字段)
  → 主题组织(3-6个维度) → 综述撰写(引言+各主题+总结)
```

## 技术栈

- **LLM 后端**：OpenAI / Ollama 双后端可切换
- **学术 API**：arXiv + Semantic Scholar + OpenAlex
- **数据模型**：Pydantic v2
- **Web 界面**：Gradio
- **运行环境**：Python 3.10+
