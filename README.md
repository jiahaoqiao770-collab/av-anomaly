# 自动驾驶数据生产链路异常诊断与大模型分析系统

面向自动驾驶数据生产链路（采集 → 标注 → 质检），对多源生产数据进行清洗、指标计算与异常检测，
并结合大模型（RAG + Agent）对异常原因做归因分析，输出可读的诊断结论。

## 项目背景

自动驾驶数据生产链路包含数据采集、标注、质检等多个环节，任一环节出现异常（采集缺失、
标注返工率飙升、质检通过率下降）都会直接影响模型训练数据的质量，且问题往往在数周后才被发现。

本项目尝试用「规则 + 统计 + 机器学习」三层手段自动识别链路异常，再借助大模型把异常现象
转化为可执行的诊断建议。

## 技术栈

| 环节 | 选型 | 说明 |
|---|---|---|
| 数据处理 | Python 3.11 / Pandas / DuckDB | CSV 清洗、指标聚合 |
| 异常检测 | scikit-learn | 孤立森林 / 聚类 / 阈值规则 |
| 模型微调 | LLaMA-Factory（LoRA / SFT） | 领域问答与诊断报告生成 |
| 推理部署 | vLLM（生产）/ Ollama（本地调试） | 高并发 vs 快速验证 |
| RAG & Agent | LangChain / FAISS / LangGraph | 知识库检索 + 多步诊断编排 |
| 服务与演示 | FastAPI / Streamlit | 接口 + 可视化页面 |

## 目录结构

```
av-anomaly/
├── data/            # 原始与中间数据（不入库）
├── notebooks/       # 探索性分析
├── src/             # 核心代码
│   ├── ingest/      # 数据读取与清洗
│   ├── metrics/     # 指标计算
│   ├── detect/      # 异常检测
│   ├── rag/         # 知识库与检索
│   └── agent/       # Agent 编排
├── configs/         # 配置文件
├── tests/           # 测试
├── .gitignore
└── README.md
```

## 开发环境

- Windows 11 + WSL2 (Ubuntu 24.04) + VS Code Remote-WSL
- Miniforge + Python 3.11
- GPU：RTX 4060 Laptop 8GB

```bash
conda activate av
```

## 快速开始

```bash
git clone git@github.com:jiahaoqiao770-collab/av-anomaly.git
cd av-anomaly
conda activate av
```

## 进度

- [x] 开发环境搭建（WSL2 / Python 3.11 / Git / SSH）
- [ ] 数据清洗与指标计算
- [ ] 异常检测模型
- [ ] RAG 知识库
- [ ] Agent 编排
- [ ] LoRA 微调与评测
- [ ] FastAPI + Streamlit 演示

## 说明

本项目为个人学习与课程实践项目。数据与模型权重不纳入版本管理，可通过脚本重新生成或从指定地址下载。