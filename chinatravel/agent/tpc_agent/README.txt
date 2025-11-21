======================================
TPC Agent - 使用说明
队伍：AIC-2025-99973300 - z1ha0
======================================

本目录是自包含的 TPC Agent 实现，包含所有运行所需的代码和模型权重。

## 目录结构

tpc_agent/
├── base.py              - BaseAgent 基类
├── llms.py              - LLM 初始化（支持 tpc_agent/local_llm 路径）
├── load_model.py        - 模型加载函数
├── tpc_agent.py         - UrbanTrip Agent 实现
├── tpc_agent_llm.py     - UrbanTripLLM Agent 实现
├── tpc_llm.py           - TPCLLM 实现
├── utils.py             - 工具函数
├── local_llm/           - 本地模型权重（Qwen3-8B 等）
└── README.txt           - 本文件

## 快速开始

### 方法 1: 直接运行（推荐）

如果外部 run_exp.py 已更新（包含 use_tpc_llms 参数）:

```bash
python run_exp.py --splits tpc_phase_2_online_test --agent UrbanTripLLM --llm Qwen3-8B
```

### 方法 2: 手动修改（如果外部代码未更新）

如果外部 run_exp.py 未更新，需要修改两个文件：

#### 1. 修改 chinatravel/agent/load_model.py

在 init_llm 函数中添加 use_tpc_llms 参数支持：

```python
def init_llm(llm_name, max_model_len=None, use_tpc_llms=False):
    if use_tpc_llms:
        try:
            from .tpc_agent.load_model import init_llm as tpc_init_llm
            return tpc_init_llm(llm_name, max_model_len)
        except Exception as e:
            print(f"Warning: Failed to use tpc_agent llms: {e}")

    # ... 原有代码
```

#### 2. 修改 run_exp.py

在调用 init_llm 时传入 use_tpc_llms=True：

```python
# 添加判断逻辑
use_tpc_llms = args.agent in ["TPCAgent", "UrbanTrip", "UrbanTripLLM"]

kwargs = {
    "method": args.agent,
    "env": WorldEnv(),
    "backbone_llm": init_llm(args.llm, max_model_len=max_model_len, use_tpc_llms=use_tpc_llms),
    # ... 其他参数
}
```

然后运行：
```bash
python run_exp.py --splits tpc_phase_2_online_test --agent UrbanTripLLM --llm Qwen3-8B
```

## 支持的 Agent

- **UrbanTripLLM**: 带 LLM 增强的版本（推荐）
  - 使用命令: `--agent UrbanTripLLM --llm Qwen3-8B`

- **UrbanTrip**: 基础版本
  - 使用命令: `--agent UrbanTrip --llm Qwen3-8B`

- **TPCAgent**: 官方模板（已被上述两个实现替代）
  - 使用命令: `--agent TPCAgent --llm TPCLLM`

## 支持的模型

本目录的 local_llm/ 包含以下模型权重：
- Qwen3-8B（完整权重约 16GB）
- Qwen3-4B（如果有）

支持的 LLM 参数：
- `--llm Qwen3-8B`: 使用 Qwen3-8B 模型（16K context）
- `--llm Qwen3-4B`: 使用 Qwen3-4B 模型
- `--llm Qwen-4B`: 映射到 Qwen3-4B
- `--llm TPCLLM`: 使用 TPCLLM（空实现，需自行补充）

## 模型路径说明

llms.py 中的 Qwen 类会自动按以下优先级查找模型：
1. tpc_agent/local_llm/{model_name}
2. chinatravel/local_llm/{model_name}

确保模型权重在 tpc_agent/local_llm/ 目录下即可。

## 验证运行

### 单条查询测试
```bash
python run_exp.py --splits tpc_phase_2_online_test --agent UrbanTripLLM --llm Qwen3-8B --index e20241028160339524446
```

### 完整测试集
```bash
python run_exp.py --splits tpc_phase_2_online_test --agent UrbanTripLLM --llm Qwen3-8B --output_name test_run
```

### 评测
```bash
python pack_submission.py --method test_run --splits tpc_phase_2_online_test --output test_run.zip
python eval_from_zip.py --zip test_run.zip --splits tpc_phase_2_online_test
```

## 注意事项

1. **模型权重**: 确保 local_llm/ 目录存在且包含完整的模型权重
2. **GPU 内存**: Qwen3-8B 需要至少 24GB GPU 内存
3. **上下文长度**: 默认使用 16384 tokens，可在 run_exp.py 中调整
4. **依赖包**: 需要 vllm, transformers, torch 等依赖

## 常见问题

**Q: 提示找不到模型文件？**
A: 检查 local_llm/ 目录是否包含 config.json, pytorch_model.bin 等文件

**Q: 运行时提示导入错误？**
A: 确保外部代码的 load_model.py 和 run_exp.py 已按上述说明修改

**Q: 如何切换到其他模型？**
A: 将模型权重放到 local_llm/ 目录，然后使用 --llm {model_name} 参数

## 联系方式

队伍参赛编号：AIC-2025-99973300
团队名称：z1ha0
联系邮箱：zihao3351@gmail.com
QQ：2957383725
手机：17614628870
